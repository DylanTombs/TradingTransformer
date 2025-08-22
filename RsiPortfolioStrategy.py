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


class RsiPortfolioStrategy(bt.Strategy):
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
        ('buy_uncertainty', 0.08),
        ('max_positions', 5),
        ('position_size', 0.20)
    )

    def __init__(self):
        # Create separate indicators, buffers, and predictions for each stock
        self.emas = {}
        self.rsis = {}
        self.buffers = {}
        self.predictions = {}
        self.stock_names = {}
        
        # Initialize for each data feed
        for i, data in enumerate(self.datas):
            stock_name = getattr(data, '_name', f'STOCK_{i}')
            self.stock_names[data] = stock_name
            
            self.emas[data] = bt.indicators.EMA(data.close, period=self.p.emaPeriod)
            self.rsis[data] = bt.indicators.RSI(data.close, period=self.p.rsiPeriod)
            self.buffers[data] = []
            self.predictions[data] = 0

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

        # Initialize per-stock OBV tracking
        self.obvs = {}

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

        for data in self.datas:
            if len(data) < totalRequired:
                continue
            
            self.process_single_stock(data)

    def process_single_stock(self, data):
        # Current bar values
        current_date = data.datetime.date(0)
        close = data.close[0]
        high = data.high[0]
        low = data.low[0]
        open_price = data.open[0]
        volume = data.volume[0]

        # Calculate Pivot Points
        P = (high + low + close) / 3
        R1 = 2 * P - low
        R2 = P + (high - low)
        R3 = high + 2 * (P - low)
        S1 = 2 * P - high
        S2 = P - (high - low)
        S3 = low - 2 * (high - P)

        # Calculate OBV
        obv = self.calculateOBV(data)

        # Calculate indicators
        rsi = self.calculateRsi(data)
        macd = self.calculateMacd(data)
        volZ = self.calculateVolumeZscore(data)
        vol = self.calculateVolatility(data)
        overnightGap = self.calculateOvernightGap(data)
        ret1 = self.calculateReturn(1, data)
        ret3 = self.calculateReturn(3, data)
        ret5 = self.calculateReturn(5, data)

        # Calculate Stochastic RSI
        SR_RSI_K, SR_RSI_D = self.calculateStochRSI(rsi, data)

        # Calculate regular Stochastic
        SR_K, SR_D = self.calculateStochastic(data)

        # Calculate moving averages
        sma = self.calculateSMA(20, data)
        lma = self.calculateSMA(50, data)
        sema = self.calculateEMA(12, data)
        lema = self.calculateEMA(26, data)

        # Calculate MACD components
        macds = self.calculateMACDSignal(data)
        macdh = macd - macds if macds is not None else 0

        # Calculate other features
        atr = self.calculateATR(data)
        HL_PCT = (high - low) / close * 100
        PCT_CHG = self.calculatePctChange(data)

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
            'ticker': self.stock_names[data]  # Use actual stock name
        }

        if any([v == 0.0 or pd.isna(v) for k, v in row.items() if k != 'open']):
            return

        # Use per-stock buffer
        self.buffers[data].append(row)
        if len(self.buffers[data]) > self.p.seqLen:
            self.buffers[data].pop(0)

        if len(self.buffers[data]) == self.p.seqLen:
            try:
                dfWindow = pd.DataFrame(self.buffers[data])[self.columns]
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
                    stockColumn='ticker'
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
                self.predictions[data] = preds.mean()  # Store per-stock prediction

                currentPrice = data.close[0]
                current_position = self.getposition(data).size  # Fixed: add data parameter
                entry_price = self.getposition(data).price if current_position > 0 else None  # Fixed: add data parameter

                # Use per-stock prediction
                prediction = self.predictions[data]

                # Core trading metrics (no ML uncertainty)
                rsi_oversold = rsi < self.p.rsi_buy
                rsi_overbought = rsi > self.p.rsi_sell

                # Price action signals
                price_above_prediction = prediction > currentPrice * self.p.buy_threshold
                price_below_prediction = prediction < currentPrice * self.p.sell_threshold

                # Momentum confirmation
                recent_low = currentPrice == min(data.close.get(size=5))
                recent_high = currentPrice == max(data.close.get(size=5))

                # Portfolio constraint: count active positions
                active_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)

                # Entry: Buy dips in uptrend
                if current_position == 0:
                    if ((rsi_oversold and price_above_prediction) or recent_low) and active_positions < self.p.max_positions:
                        self.order_target_percent(data=data, target=self.p.position_size)

                # Scale in: Add on continued weakness with strong signal
                elif current_position < self.p.position_size * 0.9:
                    if rsi < (self.p.rsi_buy-15) and price_above_prediction:
                        self.order_target_percent(data=data, target=self.p.position_size)

                # Exit: Sell peaks or trend reversal
                if current_position > 0:
                    # Full exit on overbought + prediction reversal
                    if rsi_overbought and price_below_prediction:
                        self.close(data=data)

                    # Partial profit taking on recent highs
                    elif recent_high and rsi > (self.p.rsi_sell - 5):
                        self.order_target_percent(data=data, target=current_position * 0.5)

                    # Stop loss: prediction significantly wrong
                    elif entry_price and currentPrice < entry_price * 0.95:
                        self.close(data=data)

                    elif current_position < self.p.position_size * 0.6 and rsi < 45:
                        self.order_target_percent(data=data, target=self.p.position_size)

            except Exception as e:
                print(f"Error processing {self.stock_names[data]}: {e}")

    def calculateMacd(self, data):
        if len(data) < 26:
            return 0.0
        closes = np.array([data.close[-i] for i in range(26)][::-1])
        ema12 = closes[-12:].mean()
        ema26 = closes.mean()
        return ema12 - ema26

    def calculateVolumeZscore(self, data):
        if len(data) < 20:
            return 0.0
        volumes = np.array([data.volume[-i] for i in range(20)][::-1])
        currentVolume = volumes[-1]
        meanVolume = volumes.mean()
        stdVolume = volumes.std() + 1e-6
        return (currentVolume - meanVolume) / stdVolume

    def calculateVolatility(self, data):
        if len(data) < 20:
            return 0.0
        closes = np.array([data.close[-i] for i in range(20)][::-1])
        returns = np.diff(closes) / closes[:-1]
        return returns.std()

    def calculateOvernightGap(self, data):
        if len(data) < 2:
            return 0.0
        prevClose = data.close[-1]
        currentOpen = data.open[0]
        return np.log(currentOpen / prevClose)

    def calculateReturn(self, lag, data):
        if len(data) < lag + 1:
            return 0.0
        currentClose = data.close[0]
        pastClose = data.close[-lag]
        return (currentClose / pastClose) - 1

    def calculateRsi(self, data):
        if len(data) < 15:
            return 0.0
        closes = np.array([data.close[-i] for i in range(15)][::-1])
        deltas = np.diff(closes)
        gains = deltas.clip(min=0)
        losses = -deltas.clip(max=0)
        avgGain = gains.mean()
        avgLoss = losses.mean() + 1e-10
        rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))

    def calculateOBV(self, data):
        if len(data) < 2:
            if data not in self.obvs:
                self.obvs[data] = data.volume[0]
            return self.obvs[data]

        prevClose = data.close[-1]
        currentVolume = data.volume[0]

        if data not in self.obvs:
            self.obvs[data] = currentVolume
            return self.obvs[data]

        if data.close[0] > prevClose:
            self.obvs[data] += currentVolume
        elif data.close[0] < prevClose:
            self.obvs[data] -= currentVolume

        return self.obvs[data]

    def calculateStochastic(self, data):
        if len(data) < 14:
            return 50, 50

        high_14 = max(data.high.get(size=14))
        low_14 = min(data.low.get(size=14))
        close = data.close[0]

        SR_K = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)

        stoch_buffer_key = f'SR_K_buffer_{id(data)}'
        if not hasattr(self, stoch_buffer_key):
            setattr(self, stoch_buffer_key, [])

        buffer = getattr(self, stoch_buffer_key)
        buffer.append(SR_K)
        if len(buffer) > 3:
            buffer.pop(0)

        SR_D = sum(buffer) / len(buffer) if buffer else SR_K

        return SR_K, SR_D

    def calculateStochRSI(self, rsi, data):
        rsi_buffer_key = f'RSIBuffer_{id(data)}'
        if not hasattr(self, rsi_buffer_key):
            setattr(self, rsi_buffer_key, [])

        buffer = getattr(self, rsi_buffer_key)
        buffer.append(rsi)
        if len(buffer) > 14:
            buffer.pop(0)

        if len(buffer) < 14:
            return 50, 50

        rsi_high = max(buffer)
        rsi_low = min(buffer)

        SR_RSI_K = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)

        stoch_rsi_buffer_key = f'SR_RSI_K_buffer_{id(data)}'
        if not hasattr(self, stoch_rsi_buffer_key):
            setattr(self, stoch_rsi_buffer_key, [])

        stoch_buffer = getattr(self, stoch_rsi_buffer_key)
        stoch_buffer.append(SR_RSI_K)
        if len(stoch_buffer) > 3:
            stoch_buffer.pop(0)

        SR_RSI_D = sum(stoch_buffer) / len(stoch_buffer) if stoch_buffer else SR_RSI_K

        return SR_RSI_K, SR_RSI_D

    def calculateSMA(self, period, data):
        if len(data) < period:
            return data.close[0]
        return sum(data.close.get(size=period)) / period
    
    def calculateEMA(self, span, data):
        ema_key = f'ema_{span}_{id(data)}'
        if not hasattr(self, ema_key):
            setattr(self, ema_key, data.close[0])
            return data.close[0]

        prevEMA = getattr(self, ema_key)
        alpha = 2 / (span + 1)
        currentEMA = alpha * data.close[0] + (1 - alpha) * prevEMA
        setattr(self, ema_key, currentEMA)
        return currentEMA

    def calculateMACDSignal(self, data):
        if len(data) < 26 + 9:
            return None

        MACDValues = []
        for i in range(9):
            closes = np.array([data.close[-i-j] for j in range(26)][::-1])
            ema12 = closes[-12:].mean()
            ema26 = closes.mean()
            MACDValues.append(ema12 - ema26)

        return sum(MACDValues) / len(MACDValues)

    def calculateATR(self, data):
        if len(data) < 14:
            return 0.0

        trueRanges = []
        for i in range(14):
            currentHigh = data.high[-i]
            currentLow = data.low[-i]
            if i < len(data)-1:
                prevClose = data.close[-i-1]
            else:
                prevClose = currentLow

            tr1 = currentHigh - currentLow
            tr2 = abs(currentHigh - prevClose)
            tr3 = abs(currentLow - prevClose)
            trueRanges.append(max(tr1, tr2, tr3))

        return sum(trueRanges) / len(trueRanges)

    def calculatePctChange(self, data):
        if len(data) < 2:
            return 0.0
        return (data.close[0] / data.close[-1] - 1) * 100



