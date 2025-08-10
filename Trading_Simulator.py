import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your existing strategies
from RsiEmaStrategy import RsiEmaStrategy  


class StrategyEvaluator(bt.Analyzer):
    def __init__(self):
        self.start_val = None
        self.end_val = None
        self.trades = []
        self.values = []
        self.trade_returns = []
        self.equity_curve = []

        
    def start(self):
        self.start_val = self.strategy.broker.getvalue()

    def next(self):
        self.values.append(self.strategy.broker.getvalue())

        
    def stop(self):
        self.end_val = self.strategy.broker.getvalue()
        
    def notify_trade(self, trade):
        if trade.isclosed:

            start_val = self.values[trade.baropen] if trade.baropen < len(self.values) else self.values[-1]
            end_val = self.values[trade.barclose - 1] if trade.barclose - 1 < len(self.values) else self.values[-1]
            pct_return = (end_val - start_val) / start_val if start_val != 0 else 0.0

            self.trade_returns.append(pct_return)

            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'duration': trade.barclose - trade.baropen,
                'price': trade.price,
                'size': trade.history[0].size if trade.history else 1,  # Fallback
                'return': pct_return
            })


    def get_analysis(self):
        returns = np.array(self.trade_returns)
        
        return {
            'sharpe_ratio': self._calc_sharpe(returns),
            'max_drawdown': self._calc_max_drawdown(),
            'win_rate': self._calc_win_rate(),
            'profit_factor': self._calc_profit_factor(),
            'avg_risk_reward': self._calc_avg_risk_reward(),
            'bad_trades': self._flag_bad_trades(returns)  
        }
    
    def _calc_sharpe(self, returns, risk_free_rate=0.0):
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252 
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(252)

        
    def _calc_max_drawdown(self):
        peak = self.values[0]
        max_dd = 0.0
        for val in self.values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100  # percentage
        
    def _calc_win_rate(self):
        wins = [t['return'] for t in self.trades if t['return'] > 0]
        return (len(wins) / len(self.trades)) * 100 if self.trades else 0.0
        
    def _calc_profit_factor(self):
        gross_profit = sum(t['pnlcomm'] for t in self.trades if t['pnlcomm'] > 0)
        gross_loss = abs(sum(t['pnlcomm'] for t in self.trades if t['pnlcomm'] < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
    def _calc_avg_risk_reward(self):
        reward = []
        risk = []
        for trade in self.trades:
            if trade['pnlcomm'] > 0:
                reward.append(trade['pnlcomm'])
            else:
                risk.append(abs(trade['pnlcomm']))
        avg_reward = np.mean(reward) if reward else 0.0
        avg_risk = np.mean(risk) if risk else 0.0
        return avg_reward / avg_risk if avg_risk != 0 else 0.0
    
    def _flag_bad_trades(self, returns, threshold_std=1.5):
       
        if len(returns) < 2:
            return []

        mean = np.mean(returns)
        std = np.std(returns)

        flagged = [
            i for i, r in enumerate(returns)
            if r < (mean - threshold_std * std)
        ]

        return flagged

def RunSimulation(strategy):
    cerebro = bt.Cerebro()
    cerebro.adddata(datafeed)
    cerebro.addstrategy(strategy)
    cerebro.broker.set_cash(1000)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(StrategyEvaluator, _name='custom_metrics')
    
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    
    # Print advanced metrics
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    custom = strat.analyzers.custom_metrics.get_analysis()
    
    print("\n=== Performance Metrics ===")
    print(f"Sharpe Ratio: {sharpe['sharperatio']:.2f}")
    print(f"Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
    print(f"Win Rate: {custom['win_rate']:.2f}%")
    print(f"Profit Factor: {custom['profit_factor']:.2f}")
    print(f"Avg Risk/Reward: {custom['avg_risk_reward']:.2f}:1")
    print(f"Total Return: {(cerebro.broker.getvalue() / 1000 - 1) * 100:.2f}%")
    
    if custom.get('bad_trades'):
        print(f"\n⚠️ Trades hurting Sharpe: {custom['bad_trades']}")
        for i in custom['bad_trades']:
            t = strat.analyzers.custom_metrics.trades[i]
            print(f"Trade {i}: Return={t['return']:.4f}, PnL={t['pnlcomm']:.2f}, Duration={t['duration']} bars")

    cerebro.plot(style='candlestick')


    equity = strat.analyzers.custom_metrics.values  # or cerebro.run()[0].equity_curve

    plt.figure(figsize=(10, 5))
    plt.plot(equity, label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Bar Number')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# 3. Load Data (unchanged)
# ---------------------------
data = pd.read_csv('NKE.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

datafeed = bt.feeds.PandasData(
    dataname=data,
    datetime=None,
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
    openinterest=-1
)

# Run with your preferred strategy
RunSimulation(RsiEmaStrategy)