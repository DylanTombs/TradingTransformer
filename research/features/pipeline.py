from technicalIndicators import *

class FeaturePipeline:

    def __init__(self):
        pass

    def transform(self, df):


        df["rsi"] = calculateRsi(df["close"])
        df["macd"] = calculateMacd(df["close"])
        df["volatility"] = calculateVolatility(df["close"])

        return df