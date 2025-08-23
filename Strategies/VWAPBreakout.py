import backtrader as bt
import pandas as pd
import numpy as np
from datetime import time

class VWAPIndicator(bt.Indicator):
    lines = ('vwap',)
    params = (
        ('vwap_freq', 'D'),
        ('price_col', 'close'),
        ('vol_col', 'volume'),
    )
    
    def __init__(self):
        self.current_period = None
        self.cum_vol = 0
        self.cum_pv = 0
        self.addminperiod(1)

    def next(self):
        current_dt = self.data.datetime.datetime()
        
        # Determine current period
        if self.p.vwap_freq == 'D':
            current_period = current_dt.date()
        else:
            current_period = pd.Timestamp(current_dt).to_period(self.p.vwap_freq).start_time.date()
        
        # Reset if period changed
        if current_period != self.current_period:
            self.current_period = current_period
            self.cum_vol = 0
            self.cum_pv = 0
        
        # Calculate price
        if self.p.price_col.lower() == 'typical':
            price = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        else:
            price = self.data.close[0]
        
        volume = self.data.volume[0]
        self.cum_vol += volume
        self.cum_pv += price * volume
        
        # Calculate VWAP
        if self.cum_vol > 0:
            self.lines.vwap[0] = self.cum_pv / self.cum_vol
        else:
            self.lines.vwap[0] = price

class VWAPBreakout(bt.Strategy):
    """
    • Long when Close crosses above VWAP
    • Short when Close crosses below VWAP
    Optional filter: take longs only if Close > day's Open (bull bias)
    Always flat by 15:45-16:00 ET bar
    """
    
    params = dict(
        intraday_close_time=time(15, 45),
        atr_stop=1.5,
        atr_period=14,
        vwap_freq='D',
        price_col='close',
        vol_col='volume'
    )

    def __init__(self):
        # VWAP indicator
        self.vwap = VWAPIndicator(
            self.data,
            vwap_freq=self.p.vwap_freq,
            price_col=self.p.price_col,
            vol_col=self.p.vol_col
        )
        
        # ATR for trailing stop
        if self.p.atr_stop:
            self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # Track daily open
        self.daily_open = None
        self.current_date = None

    def next(self):
        current_dt = self.data.datetime.datetime()
        current_date = current_dt.date()
        
        # Update daily open
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_open = self.data.open[0]

        close = self.data.close[0]
        vwap_val = self.vwap[0]

        # Entry logic
        if not self.position:
            if (close > vwap_val) and (close > self.daily_open):
                self.buy()
            elif (close < vwap_val) and (close < self.daily_open):
                self.sell()

        # Trailing stop
        if self.position and self.p.atr_stop:
            price = self.data.close[0]
            atr_val = self.atr[0]
            trail = self.p.atr_stop * atr_val
            
            if self.position.size > 0:  # Long
                new_sl = price - trail
                if not hasattr(self, 'stop_loss') or new_sl > getattr(self, 'stop_loss', 0):
                    self.stop_loss = new_sl
                if price <= self.stop_loss:
                    self.close()
            elif self.position.size < 0:  # Short
                new_sl = price + trail
                if not hasattr(self, 'stop_loss') or new_sl < getattr(self, 'stop_loss', float('inf')):
                    self.stop_loss = new_sl
                if price >= self.stop_loss:
                    self.close()

        # Intraday exit
        if self.position:
            bar_time = current_dt.time()
            if bar_time >= self.p.intraday_close_time:
                self.close()