from Transformer.Metrics import metric
import Transformer.Model
from Transformer.DataFrame import DataFrameDataset
from torch.utils.data import DataLoader
from Transformer.Tools import EarlyStopping, adjustLearningRate

import pandas as pd
import numpy as np
import torch

import joblib
import torch.nn as nn
from torch import optim

import logging
import os
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),  # save to file
        logging.StreamHandler()               # print to console
    ]
)

class Model_Interface():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        self.model = self.buildModel().to(self.device)

    def buildModel(self):
        model = Transformer.Model.Model(self.args).float()
        return model

    def getData(self, flag, data):
        featureScaler = None
        targetScaler = None

        if flag != 'train' and hasattr(self.args, 'checkpoints'):
            try:
                featureScaler = joblib.load(os.path.join(self.args.checkpoints, 'featureScaler.pkl'))
                targetScaler = joblib.load(os.path.join(self.args.checkpoints, 'targetScaler.pkl'))
            except FileNotFoundError:
                print("Warning: Scaler files not found, fitting new scalers")

        dataSet = DataFrameDataset(  # Use new stock-aware dataset
            df=data,
            flag=flag,
            size=(self.args.seqLen, self.args.labelLen, self.args.predLen),
            target=self.args.target,
            auxilFeatures=self.args.auxilFeatures,
            featureScaler=featureScaler,
            targetScaler=targetScaler,
            stockColumn='ticker'  # Add this parameter
        )

        # NEW: Window count verification
        print(f"{flag.upper()} dataset:")
        print(f"- Total rows in DataFrame: {len(data)}")
        print(f"- Generated windows: {len(dataSet)}")
        if hasattr(dataSet, 'valid_indices'):
            print(f"- First window starts at index: {dataSet.valid_indices[0]}")
            print(f"- Last window starts at index: {dataSet.valid_indices[-1]}")

        dataLoader = DataLoader(
            dataSet,
            batch_size=self.args.batchSize,
            shuffle=flag == 'train',
            num_workers=self.args.numWorkers,
            drop_last=(flag != 'pred')
        )

        print(f"- Total batches: {len(dataLoader)}")
        print(f"- Batch size: {self.args.batchSize}")
        print(f"- Estimated samples per epoch: {len(dataLoader)*self.args.batchSize}\n")
        
        return dataSet, dataLoader
    
    def splitData(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('date')

        # Calculate split points
        totalLen = len(df)
        trainEnd = int(totalLen * 0.7)
        valEnd = trainEnd + int(totalLen * 0.15)

        # Split the data
        trainDf = df.iloc[:trainEnd]
        valDf = df.iloc[trainEnd:valEnd]
        testDf = df.iloc[valEnd:]

        return trainDf.reset_index(), valDf.reset_index(), testDf.reset_index()


    def vali(self, valiData, valiLoader, criterion):
        losses, rmses, mapes = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batchX, batchY, batchXMark, batchYMark) in enumerate(valiLoader):
                batchX = batchX.float().to(self.device)
                batchY = batchY.float()

                batchXMark = batchXMark.float().to(self.device)
                batchYMark = batchYMark.float().to(self.device)

                decInp = torch.zeros_like(batchY[:, -self.args.predLen:, :]).float()
                decInp = torch.cat([batchY[:, :self.args.labelLen, :], decInp], dim=1).float().to(self.device)

                outputs = self.model(batchX, batchXMark, decInp, batchYMark)[0]
                fDim = -1
                outputs = outputs[:, -self.args.predLen:, fDim:]
                targets = batchY[:, -self.args.predLen:, fDim:].to(self.device)

                mse, rmse, mape = compute_metrics(outputs, targets)
                losses.append(mse)
                rmses.append(rmse)
                mapes.append(mape)

        return np.mean(losses), np.mean(rmses), np.mean(mapes)

    def train(self, data):
        trainDf, valDf, testDf = self.splitData(data)

        print(f"\nData Split Verification:")
        print(f"Training samples: {len(trainDf)} rows ({len(trainDf)/len(data):.1%})")
        print(f"Validation samples: {len(valDf)} rows ({len(valDf)/len(data):.1%})")
        print(f"Testing samples: {len(testDf)} rows ({len(testDf)/len(data):.1%})")
        print(f"Total samples: {len(trainDf)+len(valDf)+len(testDf)} (original: {len(data)})\n")
        trainData, trainLoader = self.getData(flag='train', data=trainDf)

        path = self.args.checkpoints

        os.makedirs(path, exist_ok=True)
        if hasattr(trainData, 'featureScaler'):
            joblib.dump(trainData.featureScaler, os.path.join(path, 'featureScaler.pkl'))
            print(f"Saved feature scaler to {os.path.join(path, 'featureScaler.pkl')}")
        if hasattr(trainData, 'targetScaler'):
            joblib.dump(trainData.targetScaler, os.path.join(path, 'targetScaler.pkl'))

        valData, valLoader = self.getData(flag='val', data=valDf)
        testData, testLoader = self.getData(flag='test', data=testDf)

        timeNow = time.time()
        trainSteps = len(trainLoader)
        earlyStopping = EarlyStopping(patience=self.args.patience)

        modelOptim = optim.Adam(self.model.parameters(), lr=self.args.learningRate)
        criterion = nn.MSELoss()

        logging.info("Starting training loop...")
        logging.info(f"Train steps per epoch: {trainSteps}")


        for epoch in range(self.args.trainEpochs):
            iterCount = 0
            trainLoss = []
            epochTime = time.time()
            self.model.train()

            for i, (batchX, batchY, batchXMark, batchYMark) in enumerate(trainLoader):
                iterCount += 1
                modelOptim.zero_grad()

                batchX = batchX.float().to(self.device)
                batchY = batchY.float().to(self.device)
                batchXMark = batchXMark.float().to(self.device)
                batchYMark = batchYMark.float().to(self.device)

                decInp = torch.zeros_like(batchY[:, -self.args.predLen:, :]).float()
                decInp = torch.cat([batchY[:, :self.args.labelLen, :], decInp], dim=1).to(self.device)

                # === Forward pass ===

                outputs = self.model(batchX, batchXMark, decInp, batchYMark)[0]

                fDim = -1
                outputs = outputs[:, -self.args.predLen:, fDim:]
                batchY = batchY[:, -self.args.predLen:, fDim:]
                loss = criterion(outputs, batchY)
                trainLoss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - timeNow) / iterCount
                    leftTime = speed * ((self.args.trainEpochs - epoch) * trainSteps - i)
                    logging.info(
                        f"Iter {i + 1}/{trainSteps}, Epoch {epoch + 1}/{self.args.trainEpochs} | "
                        f"Loss: {loss.item():.7f} | Speed: {speed:.4f}s/iter | ETA: {leftTime/60:.2f} min"
                    )
                    iterCount = 0
                    timeNow = time.time()

                loss.backward()
                modelOptim.step()


            trainMse = np.mean(trainLoss)
            trainRmse = np.sqrt(trainMse)
            trainMape = np.nan  # Not calculated for train set (optional)

            valMse, valRmse, valMape = self.vali(valData, valLoader, criterion)
            testMse, testRmse, testMape = self.vali(testData, testLoader, criterion)

            logging.info(
                f"[Epoch {epoch + 1:03d}] "
                f"Train MSE: {trainMse:.7f} | RMSE: {trainRmse:.7f} | "
                f"Val MSE: {valMse:.7f} | RMSE: {valRmse:.7f} | MAPE: {valMape:.3f}% | "
                f"Test MSE: {testMse:.7f} | RMSE: {testRmse:.7f} | MAPE: {testMape:.3f}% | "
            )


            earlyStopping(valMse, self.model, path)
            if earlyStopping.earlyStop:
                print("Early stopping")
                break

            adjustLearningRate(modelOptim, epoch + 1, self.args)

        # === Load best model weights ===
        bestModelPath = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(bestModelPath))
        print(f"Trained model loaded from {bestModelPath}")    
       
    def predict(self, seqX, seqXMark, seqYMark, load=False, setting=None, targetScaler=None):
        self.model.train()

        if load and setting:
            path = os.path.join(self.args.checkpoints, setting)
            bestModelPath = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(bestModelPath))

        with torch.no_grad():
            seqX = seqX.float().to(self.device)
            seqXMark = seqXMark.float().to(self.device)
            seqYMark = seqYMark.float().to(self.device)

            decInp = torch.zeros([seqX.shape[0], self.args.predLen, seqX.shape[2]]).float()
            decInp = torch.cat([seqX[:, -self.args.labelLen:, :], decInp], dim=1).float().to(self.device)

            outputs = self.model(seqX, seqXMark, decInp, seqYMark)[0]

            preds = outputs.detach().cpu().numpy().squeeze(0)
            closePreds = targetScaler.inverse_transform(preds.reshape(-1, 1)).flatten()

            return closePreds
        
def compute_metrics(outputs, targets):
    """Return MSE, RMSE, and MAPE."""
    mse = torch.mean((outputs - targets) ** 2).item()
    rmse = np.sqrt(mse)
    mape = torch.mean(torch.abs((targets - outputs) / (targets + 1e-8))).item() * 100
    return mse, rmse, mape
