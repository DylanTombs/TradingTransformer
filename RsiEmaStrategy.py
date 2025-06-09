import backtrader as bt
import pandas as pd


class RsiEmaStrategy(bt.Strategy):
    params = (
        ('ema_period', 10),
        ('rsi_period', 14),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        if len(self) < max(self.p.ema_period, self.p.rsi_period):
            return  # Indicators not ready

        rsi_value = self.rsi[0]
        if pd.isna(rsi_value):
            return  # Ignore NaNs

        if rsi_value < 30 and self.getposition().size == 0:
            cash = self.broker.getcash()
            price = self.data.close[0]
            if cash >= price * 10:  # Can afford 10 shares
                self.buy(size=10)
            else:
                print(f"Not enough cash to buy 100 shares. Need {price*100:.2f}, have {cash:.2f}")

        elif rsi_value > 70 and self.getposition().size > 0:
            self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
        # These are normal intermediate states - don't log them
            return
        
        if order.status in [order.Completed]:
            action = 'BUY' if order.isbuy() else 'SELL'
            print(f'{action} EXECUTED on {self.datetime.date(0)} | '
              f'Price: {order.executed.price:.2f} | '
              f'Cost: {order.executed.value:.2f} | '
              f'Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            price = order.price if order.price is not None else 'N/A'
            print(f"Order {order.getstatusname()} - "
              f"Size: {order.size} | "
              f"Price: {price} | "
              f"Reason: {order.getstatusname()}")