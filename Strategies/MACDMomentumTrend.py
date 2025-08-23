import backtrader as bt
import numpy as np
import pandas as pd

import backtrader as bt
import numpy as np

class MacdMomentumTrend(bt.Strategy):
    params = (
        ('ema_period', 50),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_period', 14),
        ('atr_period', 20),
        ('max_position', 1.0),
        ('min_position', 0.0),  # Allow going to cash for bad performers
        ('stop_loss_atr_mult', 4.0),  # Even wider stops
        ('trailing_stop_atr_mult', 3.0),
        ('momentum_lookback', 20),  # Longer momentum
        ('trend_filter_period', 200),
        ('performance_lookback', 252),  # 1 year performance tracking
        ('underperform_threshold', -0.1),  # -10% vs market proxy
        ('reentry_cooldown', 10),  # Days to wait before re-entering
        ('position_boost_threshold', 1.15),  # 15% momentum boost threshold
    )
    
    def __init__(self):
        # Core indicators
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_period)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.params.trend_filter_period)
        
        # MACD system
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # Risk and momentum
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.momentum = self.data.close / self.data.close(-self.params.momentum_lookback)
        
        # Performance tracking
        self.returns_1y = (self.data.close / self.data.close(-self.params.performance_lookback)) - 1
        
        # State tracking
        self.entry_price = None
        self.highest_price = None
        self.last_exit_date = None
        self.consecutive_losses = 0
        self.trade_returns = []
        
        # Adaptive parameters
        self.performance_score = 0
        self.volatility_regime = 'normal'  # 'low', 'normal', 'high'
        
    def next(self):
        self.update_market_regime()
        self.update_performance_score()
        
        if not self.position:
            if self.should_enter():
                self.execute_entry()
        else:
            if self.should_exit():
                self.execute_exit()
            else:
                self.manage_position_size()
    
    def update_market_regime(self):
        """Detect current market volatility regime"""
        if len(self.data) < 60:
            return
            
        # Calculate 20-day volatility
        recent_returns = []
        for i in range(1, 21):
            if len(self.data) > i:
                ret = (self.data.close[-i+1] / self.data.close[-i]) - 1
                recent_returns.append(ret)
        
        if len(recent_returns) > 10:
            vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            if vol < 0.15:
                self.volatility_regime = 'low'
            elif vol > 0.30:
                self.volatility_regime = 'high'
            else:
                self.volatility_regime = 'normal'
    
    def update_performance_score(self):
        """Track how well this stock is performing"""
        if len(self.data) < self.params.performance_lookback:
            self.performance_score = 0
            return
            
        # 1-year return vs absolute threshold
        yearly_return = self.returns_1y[0] if self.returns_1y[0] is not None else 0
        
        # Score based on performance and trend consistency
        if yearly_return > 0.20:  # +20% annual
            self.performance_score = 2  # High performer
        elif yearly_return > 0.05:  # +5% annual
            self.performance_score = 1  # Average performer  
        elif yearly_return > -0.10:  # Better than -10%
            self.performance_score = 0  # Neutral
        else:
            self.performance_score = -1  # Underperformer
    
    def should_enter(self):
        """Enhanced entry logic based on stock performance and regime"""
        if len(self.data) < self.params.trend_filter_period:
            return False
            
        # Cooldown period after exit
        if (self.last_exit_date is not None and 
            len(self.data) - self.last_exit_date < self.params.reentry_cooldown):
            return False
        
        price = self.data.close[0]
        
        # Basic trend requirements (more flexible for good performers)
        if self.performance_score >= 1:
            # High performers: just need basic uptrend
            trend_ok = price > self.ema[0]
        else:
            # Poor performers: need stronger confirmation
            trend_ok = (price > self.ema[0] > self.ema_long[0] and 
                       self.macd.macd[0] > self.macd.signal[0])
        
        # Momentum check
        momentum_ok = True
        if len(self.data) > self.params.momentum_lookback:
            if self.performance_score >= 1:
                # High performers: minimal momentum required
                momentum_ok = self.momentum[0] > 1.0
            else:
                # Poor performers: need strong momentum
                momentum_ok = self.momentum[0] > 1.05
        
        # MACD confirmation
        macd_ok = True
        if self.performance_score <= 0:
            # Underperformers need MACD confirmation
            macd_ok = self.macd.macd[0] > 0
        
        # Volatility regime adjustment
        if self.volatility_regime == 'high' and self.performance_score <= 0:
            return False  # Skip poor performers in high vol
            
        # RSI check (only for underperformers)
        rsi_ok = True
        if self.performance_score < 0:
            rsi_ok = self.rsi[0] < 75  # Avoid overbought entries for poor stocks
        
        return trend_ok and momentum_ok and macd_ok and rsi_ok
    
    def execute_entry(self):
        """Adaptive position sizing based on performance and regime"""
        base_size = self.calculate_position_size()
        self.order_target_percent(target=base_size)
        self.entry_price = self.data.close[0]
        self.highest_price = self.data.close[0]
    
    def calculate_position_size(self):
        """Dynamic position sizing based on multiple factors"""
        # Base size depends on performance score
        if self.performance_score >= 2:
            base_size = 1.0  # Full position for high performers
        elif self.performance_score == 1:
            base_size = 0.8  # Large position for good performers
        elif self.performance_score == 0:
            base_size = 0.5  # Medium position for neutral
        else:
            base_size = 0.3  # Small position for underperformers
        
        # Volatility regime adjustment
        if self.volatility_regime == 'low':
            base_size *= 1.1  # Slightly more in low vol
        elif self.volatility_regime == 'high':
            base_size *= 0.8  # Less in high vol
        
        # Momentum boost for strong momentum
        if (len(self.data) > self.params.momentum_lookback and 
            self.momentum[0] > self.params.position_boost_threshold):
            base_size *= 1.1
        
        # Consecutive loss reduction
        if self.consecutive_losses > 2:
            base_size *= 0.7
        elif self.consecutive_losses > 4:
            base_size *= 0.5
        
        return max(self.params.min_position, min(self.params.max_position, base_size))
    
    def should_exit(self):
        """Adaptive exit logic based on performance and regime"""
        price = self.data.close[0]
        
        # Update highest price for trailing stops
        if price > self.highest_price:
            self.highest_price = price
        
        # Performance-based exit criteria
        if self.performance_score >= 1:
            # High performers: very loose exits
            return self.exit_high_performer(price)
        else:
            # Poor performers: tighter exits
            return self.exit_poor_performer(price)
    
    def exit_high_performer(self, price):
        """Loose exit criteria for good performing stocks"""
        # Only exit on major trend breaks
        major_trend_break = (price < self.ema_long[0] and 
                           self.macd.macd[0] < -0.002 and
                           self.rsi[0] < 40)
        
        # Wide trailing stop
        trailing_stop = False
        if self.atr[0] > 0 and self.highest_price > 0:
            stop_level = self.highest_price - (self.params.trailing_stop_atr_mult * self.atr[0])
            trailing_stop = price < stop_level
        
        return major_trend_break or trailing_stop
    
    def exit_poor_performer(self, price):
        """Tighter exit criteria for poor performing stocks"""
        # Trend reversal
        trend_exit = price < self.ema[0]
        
        # MACD bearish
        macd_exit = (self.macd.macd[0] < self.macd.signal[0] and 
                    self.macd.macd[0] < 0)
        
        # Tighter trailing stop
        trailing_stop = False
        if self.atr[0] > 0 and self.highest_price > 0:
            stop_level = self.highest_price - (self.params.trailing_stop_atr_mult * 0.8 * self.atr[0])
            trailing_stop = price < stop_level
        
        # RSI overbought
        rsi_exit = self.rsi[0] > 80
        
        return trend_exit or macd_exit or trailing_stop or rsi_exit
    
    def execute_exit(self):
        """Execute exit and update state"""
        self.close()
        
        # Track performance
        if self.entry_price:
            trade_return = (self.data.close[0] / self.entry_price) - 1
            if trade_return < -0.02:  # Loss > 2%
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        self.entry_price = None
        self.highest_price = None
        self.last_exit_date = len(self.data)
    
    def manage_position_size(self):
        """Dynamically adjust position size while in trade"""
        if len(self.data) % 20 == 0:  # Check every 20 bars
            current_size = abs(self.position.size) / self.broker.get_value()
            optimal_size = self.calculate_position_size()
            
            # Only adjust if significant difference
            if abs(current_size - optimal_size) > 0.1:
                self.order_target_percent(target=optimal_size)
    
    def notify_trade(self, trade):
        """Track trade statistics"""
        if trade.isclosed:
            pnl_percent = trade.pnlcomm / abs(trade.value) if trade.value != 0 else 0
            self.trade_returns.append(pnl_percent)


# Performance-based backtesting framework
class PerformanceBacktester:
    """Backtesting framework optimized for different stock performance profiles"""
    
    @staticmethod
    def create_cerebro(optimization_mode='adaptive'):
        cerebro = bt.Cerebro(optdatas=True, optreturn=False)
        
        if optimization_mode == 'adaptive':
            # Optimize for adaptive parameters
            cerebro.optstrategy(
                MacdMomentumTrend,
                ema_period=range(30, 70, 20),
                momentum_lookback=range(15, 30, 5),
                stop_loss_atr_mult=[3.0, 4.0, 5.0],
                position_boost_threshold=[1.10, 1.15, 1.20],
            )
        else:
            cerebro.addstrategy(MacdMomentumTrend)
        
        # Comprehensive analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # System Quality Number
        cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar')
        
        return cerebro
    
    @staticmethod
    def evaluate_results(results):
        """Comprehensive result evaluation"""
        best_results = []
        
        for result in results:
            strat = result[0]
            
            # Extract key metrics
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 100)
            trades = strat.analyzers.trades.get_analysis()
            
            total_trades = trades.get('total', {}).get('closed', 0)
            win_rate = trades.get('won', {}).get('total', 0) / max(total_trades, 1) * 100
            profit_factor = trades.get('won', {}).get('pnl', {}).get('total', 0) / max(abs(trades.get('lost', {}).get('pnl', {}).get('total', 1)), 1)
            
            # Composite score emphasizing consistency
            score = (sharpe * 0.4 + 
                    (100 - max_dd) / 100 * 0.3 + 
                    min(win_rate / 100, 0.6) * 0.2 + 
                    min(profit_factor / 3, 1) * 0.1)
            
            best_results.append({
                'params': strat.params,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'score': score
            })
        
        return sorted(best_results, key=lambda x: x['score'], reverse=True)


# Simple comparison strategy
class BuyAndHold(bt.Strategy):
    """Buy and hold for performance comparison"""
    def next(self):
        if not self.position:
            self.order_target_percent(target=1.0)