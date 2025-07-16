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

import os
import time
import warnings
warnings.filterwarnings('ignore')


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

        dataSet = DataFrameDataset(
            df=data,
            flag=flag,
            size=(self.args.seqLen, self.args.labelLen, self.args.predLen),
            target=self.args.target,
            auxilFeatures=self.args.auxilFeatures,
            featureScaler=featureScaler,
            targetScaler=targetScaler
        )

        dataLoader = DataLoader(
            dataSet,
            batch_size=self.args.batchSize,
            shuffle=flag == 'train',
            num_workers=self.args.numWorkers,
            drop_last=(flag != 'pred')
        )
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
        totalLoss = []
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
                batchY = batchY[:, -self.args.predLen:, fDim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batchY.detach().cpu()

                loss = criterion(pred, true)

                totalLoss.append(loss)
        totalLoss = np.average(totalLoss)
        self.model.train()
        return totalLoss

    def train(self, data):
        trainDf, valDf, testDf = self.splitData(data)
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
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - timeNow) / iterCount
                    leftTime = speed * ((self.args.trainEpochs - epoch) * trainSteps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {leftTime:.4f}s")
                    iterCount = 0
                    timeNow = time.time()

                loss.backward()
                modelOptim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epochTime:.2f}s")
            trainLoss = np.average(trainLoss)
            valLoss = self.vali(valData, valLoader, criterion)
            testLoss = self.vali(testData, testLoader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {trainSteps} | "
                  f"Train Loss: {trainLoss:.7f} Vali Loss: {valLoss:.7f} Test Loss: {testLoss:.7f}")

            earlyStopping(valLoss, self.model, path)
            if earlyStopping.earlyStop:
                print("Early stopping")
                break

            adjustLearningRate(modelOptim, epoch + 1, self.args)

        # === Load best model weights ===
        bestModelPath = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(bestModelPath))
        print(f"Trained model loaded from {bestModelPath}")    
       
    def test(self, data, test=0):
        testData, testLoader = self.getData(flag='test', data=data)
        if test:
            print('loading model')
            self.model.loadStateDict(torch.load(os.path.join('./checkpoints/', 'checkpoint.pth')))

        preds = []
        trues = []
        folderPath = './test_results/'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        self.model.eval()
        with torch.no_grad():
            for i, (batchX, batchY, batchXMark, batchYMark) in enumerate(testLoader):
                batchX = batchX.float().to(self.device)
                batchY = batchY.float().to(self.device)

                batchXMark = batchXMark.float().to(self.device)
                batchYMark = batchYMark.float().to(self.device)

                decInp = torch.zeros_like(batchY[:, -self.args.predLen:, :]).float()
                decInp = torch.cat([batchY[:, :self.args.labelLen, :], decInp], dim=1).float().to(self.device)

                outputs = self.model(batchX, batchXMark, decInp, batchYMark)[0]

                fDim = -1
                outputs = outputs[:, -self.args.predLen:, fDim:]
                batchY = batchY[:, -self.args.predLen:, fDim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batchY = batchY.detach().cpu().numpy()

                pred = outputs
                true = batchY

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batchX.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folderPath = './results/'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folderPath + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folderPath + 'pred.npy', preds)
        np.save(folderPath + 'true.npy', trues)

        invTransformPreds = testData.inverseTransform(preds[:, 0, -1])
        invTransformTrues = testData.inverseTransform(trues[:, 0, -1])
        np.save(folderPath + 'scaled_back_pred.npy', invTransformPreds)
        np.save(folderPath + 'scaled_back_true.npy', invTransformTrues)

        return

    def predict(self, seqX, seqXMark, seqYMark, load=False, setting=None):
        self.model.eval()

        featureScaler = joblib.load(os.path.join(self.args.checkpoints, 'feature_scaler.pkl'))
        targetScaler = joblib.load(os.path.join(self.args.checkpoints, 'target_scaler.pkl'))

        if load and setting:
            path = os.path.join(self.args.checkpoints, setting)
            bestModelPath = path + '/' + 'checkpoint.pth'
            self.model.loadStateDict(torch.load(bestModelPath))

        with torch.no_grad():
            seqXNP = seqX.numpy().squeeze(0)

            auxilFeatures = seqXNP[:, :-1]
            scaledAuxil = featureScaler.transform(auxilFeatures)
            targetValues = seqXNP[:, -1:]
            scaledSeqX = np.concatenate([scaledAuxil, targetValues], axis=1)

            seqX = torch.from_numpy(scaledSeqX).unsqueeze(0).float().to(self.device)
            seqXMark = seqXMark.float().to(self.device)
            seqYMark = seqYMark.float().to(self.device)

            decInp = torch.zeros([seqX.shape[0], self.args.predLen, seqX.shape[2]]).float()
            decInp = torch.cat([seqX[:, -self.args.labelLen:, :], decInp], dim=1).float().to(self.device)

            outputs = self.model(seqX, seqXMark, decInp, seqYMark)[0]

            preds = outputs.detach().cpu().numpy().squeeze(0)
            closePreds = targetScaler.inverse_transform(preds.reshape(-1, 1)).flatten()

            return closePreds