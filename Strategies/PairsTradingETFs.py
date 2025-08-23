import backtrader as bt
import pandas as pd
import numpy as np

class PairsTradingETFs(bt.Strategy):
    def __init__(self):
       self.stock1 
       self.stock2

    def next(self):
        self.zScore = self.calculateZScore()
        priceSpread = self.calculateSpread()
        if priceSpread > self.zScore:
            self.buy(size=10)
        else:                           # trailing stop implementation
            self.close()

    def calculateZScore(self):
        if len(self.data) < 20:
            return 0.0
        volumes = np.array([self.data.volume[-i] for i in range(20)][::-1])
        currentVolume = volumes[-1]
        meanVolume = volumes.mean()
        stdVolume = volumes.std() + 1e-6
        return (currentVolume - meanVolume) / stdVolume
    
    def calculateSpread(self):
        pass

    