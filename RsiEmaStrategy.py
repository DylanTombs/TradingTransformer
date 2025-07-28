import backtrader as bt
import pandas as pd
import torch
import os
import joblib
import random
from Transformer.Interface import Model_Interface
from Transformer.DataFrame import DataFrameDataset
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace as dotdict


class RsiEmaStrategy(bt.Strategy):
    params = (
        ('emaPeriod', 10),
        ('rsiPeriod', 14),
        ('seqLen', 30),
        ('labelLen', 10),
        ('predLen', 5),
        ('buy_threshold', 1.005),
        ('sell_threshold', 0.995),
        ('rsi_buy', 40),
        ('rsi_sell', 60),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.emaPeriod)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsiPeriod)
        self.buffer = []

        self.columns = ['date', 'close',
            'volume_zscore', 'rsi', 'macd', 'overnight_gap',
            'return_lag_1', 'return_lag_3', 'return_lag_5', 'volatility']
        args = self.setupArgs()

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
        self.model = Model_Interface(args)

        modelPath = os.path.join(args.checkpoints, 'checkpoint.pth')
        self.model.model.load_state_dict(torch.load(modelPath))
        self.model.model.eval()

        self.prediction = 0
        self.uncertainty = 0

    def setupArgs(self):
        args = dotdict()
        
        args.target = 'close'
        args.auxilFeatures = [
            'volume_zscore', 'rsi', 'macd', 'overnight_gap',
            'return_lag_1', 'return_lag_3', 'return_lag_5', 'volatility']
        args.checkpoints = './checkpoints/'

        args.seqLen = 30
        args.labelLen = 10
        args.predLen = 5

        args.encIn = len([args.target]) + len(args.auxilFeatures)
        args.decIn = args.encIn
        args.cOut = 1
        args.dModel = 256
        args.nHeads = 4
        args.eLayers = 2
        args.dLayers = 1
        args.dFf = 256
        args.factor = 1
        args.dropout = 0.05

        args.num_workers = 0
        args.itr = 1
        args.trainEpochs = 100
        args.batchSize = 128
        args.patience = 10
        args.learningRate = 0.0005

        args.devices = '0,1,2,3'
        args.seed = 1234

        args.pHiddenDims = [128, 128]
        args.pHiddenLayers = 2

        return args

    def next(self):
        minLookback = max(26, 20, 15, 6)
        totalRequired = self.p.seqLen + minLookback

        if len(self.data) < totalRequired:
            return

        rsi = self.calculateRsi()
        macd = self.calculateMacd()
        volZ = self.calculateVolumeZscore()
        vol = self.calculateVolatility()
        overnightGap = self.calculateOvernightGap()
        ret1 = self.calculateReturn(1)
        ret3 = self.calculateReturn(3)
        ret5 = self.calculateReturn(5)

        row = {
            'date' : self.data.datetime.date(0),
            'close': self.data.close[0],
            'volume': self.data.volume[0],
            'open': self.data.open[0],
            'volume_zscore': volZ,
            'rsi': rsi,
            'macd': macd,
            'overnight_gap': overnightGap,
            'return_lag_1': ret1,
            'return_lag_3': ret3,
            'return_lag_5': ret5,
            'volatility': vol,
        }


        if any([v == 0.0 or pd.isna(v) for k, v in row.items() if k != 'open']):
            return

        self.buffer.append(row)
        if len(self.buffer) > self.p.seqLen:
            self.buffer.pop(0)

        if len(self.buffer) == self.p.seqLen:
            try:

                
                dfWindow = pd.DataFrame(self.buffer)[self.columns]
                featureScaler = joblib.load(os.path.join(self.model.args.checkpoints, 'featureScaler.pkl'))
                targetScaler = joblib.load(os.path.join(self.model.args.checkpoints, 'targetScaler.pkl'))
                
                
                pred_data = DataFrameDataset(
                    df=dfWindow,  # Your new data (ensure same columns/order as training!)
                    flag='pred',  # Or 'test', but avoid 'train' to prevent scaler fitting
                    size=(self.model.args.seqLen, self.model.args.labelLen, self.model.args.predLen),
                    target=self.model.args.target,
                    auxilFeatures=self.model.args.auxilFeatures,
                    featureScaler=featureScaler,  # Provide pre-fit scalers
                    targetScaler=targetScaler
                )
                pred_loader = DataLoader(
                    pred_data,
                    batch_size=1,  # Predict one sequence at a time
                    shuffle=False,  # Critical!
                    num_workers=0
                )
                preds = []
                with torch.no_grad():
                    for _ in range(5): 
                        for (batchX, batchY, batchXMark, batchYMark) in pred_loader:
                            pred = self.model.predict(
                                batchX.to(self.model.device),
                                batchXMark.to(self.model.device),
                                batchYMark.to(self.model.device),
                                targetScaler=targetScaler  # Pass scaler to avoid reloading
                            )
                            preds.append(pred[-1])

                preds = np.array(preds)
                self.prediction = preds.mean()
                self.uncertainty = preds.std()

                currentPrice = self.data.close[0]

                if self.uncertainty < 0.01:
                    if rsi < self.p.rsi_buy and self.prediction > currentPrice * self.p.buy_threshold and self.getposition().size == 0:
                        self.buy(size=10)
                    elif rsi > self.p.rsi_sell and self.prediction < currentPrice * self.p.sell_threshold and self.getposition().size > 0:
                        self.close()

            except Exception as e:
                pass
                print(f"Prediction error: {str(e)}")

    def calculateMacd(self):
        if len(self.data) < 26:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(26)][::-1])
        ema12 = closes[-12:].mean()
        ema26 = closes.mean()
        return ema12 - ema26

    def calculateVolumeZscore(self):
        if len(self.data) < 20:
            return 0.0
        volumes = np.array([self.data.volume[-i] for i in range(20)][::-1])
        currentVolume = volumes[-1]
        meanVolume = volumes.mean()
        stdVolume = volumes.std() + 1e-6
        return (currentVolume - meanVolume) / stdVolume

    def calculateVolatility(self):
        if len(self.data) < 20:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(20)][::-1])
        returns = np.diff(closes) / closes[:-1]
        return returns.std()

    def calculateOvernightGap(self):
        if len(self.data) < 2:
            return 0.0
        prevClose = self.data.close[-1]
        currentOpen = self.data.open[0]
        return np.log(currentOpen / prevClose)

    def calculateReturn(self, lag):
        if len(self.data) < lag + 1:
            return 0.0
        currentClose = self.data.close[0]
        pastClose = self.data.close[-lag]
        return (currentClose / pastClose) - 1

    def calculateRsi(self):
        if len(self.data) < 15:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(15)][::-1])
        deltas = np.diff(closes)
        gains = deltas.clip(min=0)
        losses = -deltas.clip(max=0)
        avgGain = gains.mean()
        avgLoss = losses.mean() + 1e-10
        rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))
