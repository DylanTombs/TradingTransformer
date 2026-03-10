import pandas as pd
import numpy as np

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