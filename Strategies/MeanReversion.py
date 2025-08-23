import backtrader as bt

class MeanReversionSimple(bt.Strategy):
    params = dict(
        smi_period=20,
        rsi_period=14,
        buffer=5.0,
        target_alloc=0.20,   # 20% portfolio allocation on entry
    )

    def __init__(self):
        self.smi = StochasticMomentumIndex(self.data, period=self.p.smi_period)
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        price = self.data.close[0]
        smi_val = float(self.smi[0])
        rsi_val = float(self.rsi[0])

        # entry condition: price is below smi + rsi + buffer
        if not self.position and price < smi_val + rsi_val + self.p.buffer:
            # go long to 20% portfolio allocation
            self.order_target_percent(target=self.p.target_alloc)

        # partial take-profit condition: RSI > 60 (for example)
        elif self.position and rsi_val > 60:
            # reduce position by half
            current_target = self.p.target_alloc * 0.5
            self.order_target_percent(target=current_target)

        # full exit condition: RSI > 70 (strong overbought)
        elif self.position and rsi_val > 70:
            self.close()


class StochasticMomentumIndex(bt.Indicator):
    lines = ('smi',)
    params = dict(period=20, smooth1=3, smooth2=3)

    def __init__(self):
        h = bt.ind.Highest(self.data.high, period=self.p.period)
        l = bt.ind.Lowest(self.data.low, period=self.p.period)
        m = h - l
        d1 = bt.ind.EMA(m, period=self.p.smooth1)
        d2 = bt.ind.EMA(d1, period=self.p.smooth2)
        c = self.data.close - (h + l) / 2.0
        c1 = bt.ind.EMA(c, period=self.p.smooth1)
        c2 = bt.ind.EMA(c1, period=self.p.smooth2)
        denom = (d2 / 2.0)
        self.l.smi = bt.If(denom == 0, float(0.0), 100.0 * c2 / denom)




