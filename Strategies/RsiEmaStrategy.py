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
        ('buy_uncertainty', 0.08)
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.emaPeriod)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsiPeriod)
        self.buffer = []

        self.columns = ['date', 'close', 'high','low','volume','adj close','P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3','obv',
        'volume_zscore', 'rsi', 'macd','macds','macdh','sma','lma','sema','lema','overnight_gap',
        'return_lag_1', 'return_lag_3', 'return_lag_5', 'volatility', 'SR_K', 'SR_D', 
                    'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG','ticker']
        args = self.setupArgs()

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
        self.model = Model_Interface(args)

        modelPath = os.path.join(args.checkpoints, 'Model3.pth')
        self.model.model.load_state_dict(torch.load(modelPath))
        self.model.model.eval()

        self.prediction = 0
        self.uncertainty = 0

    def setupArgs(self):
        args = dotdict()
        
        args.target = 'close'
        args.auxilFeatures = ['high','low','volume','adj close','P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3','obv',
        'volume_zscore', 'rsi', 'macd','macds','macdh','sma','lma','sema','lema','overnight_gap',
        'return_lag_1', 'return_lag_3', 'return_lag_5', 'volatility', 'SR_K', 'SR_D', 
                    'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']
        args.checkpoints = './checkpoints/'

        args.seqLen = 30
        args.labelLen = 10
        args.predLen = 5

        args.encIn = len([args.target]) + len(args.auxilFeatures)
        args.decIn = args.encIn
        args.cOut = 1
        args.dModel = 256
        args.nHeads = 8
        args.eLayers = 3
        args.dLayers = 2
        args.dFf = 512
        args.factor = 1
        args.dropout = 0.1

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

        # Current bar values
        current_date = self.data.datetime.date(0)
        close = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]
        open_price = self.data.open[0]
        volume = self.data.volume[0]

        # Calculate Pivot Points
        P = (high + low + close) / 3
        R1 = 2 * P - low
        R2 = P + (high - low)
        R3 = high + 2 * (P - low)
        S1 = 2 * P - high
        S2 = P - (high - low)
        S3 = low - 2 * (high - P)

        # Calculate OBV
        obv = self.calculateOBV()

        # Calculate indicators
        rsi = self.calculateRsi()
        macd = self.calculateMacd()
        volZ = self.calculateVolumeZscore()
        vol = self.calculateVolatility()
        overnightGap = self.calculateOvernightGap()
        ret1 = self.calculateReturn(1)
        ret3 = self.calculateReturn(3)
        ret5 = self.calculateReturn(5)

        # Calculate Stochastic RSI
        SR_RSI_K, SR_RSI_D = self.calculateStochRSI(rsi)

        # Calculate regular Stochastic
        SR_K, SR_D = self.calculateStochastic()

        # Calculate moving averages
        sma = self.calculateSMA(20)
        lma = self.calculateSMA(50)
        sema = self.calculateEMA(12)
        lema = self.calculateEMA(26)

        # Calculate MACD components
        macds = self.calculateMACDSignal()
        macdh = macd - macds if macds is not None else 0

        # Calculate other features
        atr = self.calculateATR()
        HL_PCT = (high - low) / close * 100
        PCT_CHG = self.calculatePctChange()

        # Create feature row
        row = {
            'date': current_date,
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'adj close': close,
            'P': P, 
            'R1': R1, 
            'R2': R2, 
            'R3': R3, 
            'S1': S1, 
            'S2': S2, 
            'S3': S3,
            'obv': obv,
            'open': open_price,
            'volume_zscore': volZ,
            'rsi': rsi,
            'macd': macd, 
            'macds': macds,
            'macdh': macdh,
            'sma': sma,
            'lma': lma,
            'sema': sema,
            'lema': lema,
            'overnight_gap': overnightGap,
            'return_lag_1': ret1,
            'return_lag_3': ret3,
            'return_lag_5': ret5,
            'volatility': vol, 
            'SR_K': SR_K, 
            'SR_D': SR_D, 
            'SR_RSI_K': SR_RSI_K,
            'SR_RSI_D': SR_RSI_D, 
            'ATR': atr, 
            'HL_PCT': HL_PCT, 
            'PCT_CHG': PCT_CHG,
            'ticker':'PEP'
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
                    df=dfWindow,  
                    flag='pred',  
                    size=(self.model.args.seqLen, self.model.args.labelLen, self.model.args.predLen),
                    target=self.model.args.target,
                    auxilFeatures=self.model.args.auxilFeatures,
                    featureScaler=featureScaler,  
                    targetScaler=targetScaler,
                    stockColumn= 'ticker'
                )
                pred_loader = DataLoader(
                    pred_data,
                    batch_size=1,  
                    shuffle=False,  
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
                                targetScaler=targetScaler  
                            )
                            preds.append(pred[-1])

                preds = np.array(preds)
                self.prediction = preds.mean()

                currentPrice = self.data.close[0]
                current_position = self.getposition().size
                entry_price = self.getposition().price if current_position > 0 else None

                # SIMPLE DRAWDOWN PROTECTION
                if not hasattr(self, 'peak_value'):
                    self.peak_value = self.broker.get_value()
                
                current_value = self.broker.get_value()
                if current_value > self.peak_value:
                    self.peak_value = current_value
                
                current_drawdown = (self.peak_value - current_value) / self.peak_value
                
                # Stop trading if drawdown too high
                if current_drawdown > 0.25:  # 25% max drawdown
                    if current_position > 0:
                        self.close()
                    return

                # Core trading metrics (no ML uncertainty)
                rsi_oversold = rsi < self.p.rsi_buy  # e.g., 30
                rsi_overbought = rsi > self.p.rsi_sell  # e.g., 70

                # Price action signals
                price_above_prediction = self.prediction > currentPrice * self.p.buy_threshold
                price_below_prediction = self.prediction < currentPrice * self.p.sell_threshold

                # Momentum confirmation
                recent_low = currentPrice == min(self.data.close.get(size=5))  # 5-period low
                recent_high = currentPrice == max(self.data.close.get(size=5))  # 5-period high

                # SMALLER POSITION SIZES (main change)
                max_position = 0.40  # Reduced from 90% to 40%

                # Entry: Buy dips in uptrend
                if current_position == 0:
                    if (rsi_oversold and price_above_prediction) or recent_low:
                        self.order_target_percent(target=0.20)  # Start smaller

                # Scale in: Add on continued weakness with strong signal
                elif current_position < max_position:
                    if rsi < (self.p.rsi_buy-15) and price_above_prediction:  # Deeper dip
                        self.order_target_percent(target=max_position)

                # Exit: Sell peaks or trend reversal
                if current_position > 0:
                    # Full exit on overbought + prediction reversal
                    if rsi_overbought and price_below_prediction:
                        self.close()

                    # Partial profit taking on recent highs
                    elif recent_high and rsi > (self.p.rsi_sell - 5):
                        self.order_target_percent(target=current_position * 0.5)

                    # TIGHTER STOP LOSS
                    elif currentPrice < entry_price * 0.97:  # 3% stop loss (was 5%)
                        self.close()

                    # VOLATILITY STOP - exit if market gets too volatile
                    elif vol > 0.05:  # Adjust this threshold based on your data
                        self.order_target_percent(target=current_position * 0.5)  # Reduce position

                    elif current_position == 0.5 and rsi < 45:  # Re-enter on dip after profit-taking
                        self.order_target_percent(target=0.20)  # Smaller re-entry


            except Exception as e:
                print(f"Error: {e}")

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

    def calculateOBV(self):
        if len(self.data) < 2:
            return self.data.volume[0]

        prevClose = self.data.close[-1]
        currentVolume = self.data.volume[0]

        if not hasattr(self, 'obv'):
            self.obv = currentVolume
            return self.obv

        if self.data.close[0] > prevClose:
            self.obv += currentVolume
        elif self.data.close[0] < prevClose:
            self.obv -= currentVolume

        return self.obv

    def calculateStochastic(self):
        if len(self.data) < 14:
            return 50, 50

        high_14 = max(self.data.high.get(size=14))
        low_14 = min(self.data.low.get(size=14))
        close = self.data.close[0]

        SR_K = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)

        if not hasattr(self, 'SR_K_buffer'):
            self.SR_K_buffer = []

        self.SR_K_buffer.append(SR_K)
        if len(self.SR_K_buffer) > 3:
            self.SR_K_buffer.pop(0)

        SR_D = sum(self.SR_K_buffer) / len(self.SR_K_buffer) if self.SR_K_buffer else SR_K

        return SR_K, SR_D

    def calculateStochRSI(self, rsi):
        if not hasattr(self, 'RSIBuffer'):
            self.RSIBuffer = []

        self.RSIBuffer.append(rsi)
        if len(self.RSIBuffer) > 14:
            self.RSIBuffer.pop(0)

        if len(self.RSIBuffer) < 14:
            return 50, 50

        rsi_high = max(self.RSIBuffer)
        rsi_low = min(self.RSIBuffer)

        SR_RSI_K = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)

        if not hasattr(self, 'SR_RSI_K_buffer'):
            self.SR_RSI_K_buffer = []

        self.SR_RSI_K_buffer.append(SR_RSI_K)
        if len(self.SR_RSI_K_buffer) > 3:
            self.SR_RSI_K_buffer.pop(0)

        SR_RSI_D = sum(self.SR_RSI_K_buffer) / len(self.SR_RSI_K_buffer) if self.SR_RSI_K_buffer else SR_RSI_K

        return SR_RSI_K, SR_RSI_D

    def calculateSMA(self, period):
        if len(self.data) < period:
            return self.data.close[0]
        return sum(self.data.close.get(size=period)) / period
    
    def calculateEMA(self, span):
        if not hasattr(self, f'ema_{span}'):
            setattr(self, f'ema_{span}', self.data.close[0])
            return self.data.close[0]

        prevEMA = getattr(self, f'ema_{span}')
        alpha = 2 / (span + 1)
        currentEMA = alpha * self.data.close[0] + (1 - alpha) * prevEMA
        setattr(self, f'ema_{span}', currentEMA)
        return currentEMA

    def calculateMACDSignal(self):
        if len(self.data) < 26 + 9:
            return None

        MACDValues = []
        for i in range(9):
            closes = np.array([self.data.close[-i-j] for j in range(26)][::-1])
            ema12 = closes[-12:].mean()
            ema26 = closes.mean()
            MACDValues.append(ema12 - ema26)

        return sum(MACDValues) / len(MACDValues)

    def calculateATR(self):
        if len(self.data) < 14:
            return 0.0

        trueRanges = []
        for i in range(14):
            currentHigh = self.data.high[-i]
            currentLow = self.data.low[-i]
            if i < len(self.data)-1:
                prevClose = self.data.close[-i-1]
            else:
                prevClose = currentLow

            tr1 = currentHigh - currentLow
            tr2 = abs(currentHigh - prevClose)
            tr3 = abs(currentLow - prevClose)
            trueRanges.append(max(tr1, tr2, tr3))

        return sum(trueRanges) / len(trueRanges)

    def calculatePctChange(self):
        if len(self.data) < 2:
            return 0.0
        return (self.data.close[0] / self.data.close[-1] - 1) * 100
