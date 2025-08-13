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

def RunSimulation(strategy, symbol_files, cash=1000):
    results_summary = []
    for symbol_file in symbol_files:

        data = pd.read_csv(symbol_file)
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
        cerebro = bt.Cerebro()
        cerebro.adddata(datafeed)
        cerebro.addstrategy(strategy)
        cerebro.broker.set_cash(cash)
    
    # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(StrategyEvaluator, _name='custom_metrics')

        startVal = cerebro.broker.getvalue()
        results = cerebro.run()
        endVal = cerebro.broker.getvalue()

        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        custom = strat.analyzers.custom_metrics.get_analysis()

        results_summary.append({
            'Symbol': symbol_file.replace('.csv', ''),
            'Start Value': startVal,
            'End Value': endVal,
            'Total Return %': (endVal / startVal - 1) * 100,
            'Sharpe Ratio': sharpe.get('sharperatio', 0.0),
            'Max Drawdown %': drawdown['max']['drawdown'],
            'Win Rate %': custom['win_rate'],
            'Profit Factor': custom['profit_factor'],
            'Avg R/R': custom['avg_risk_reward'],
        })
    
        print(f"{symbol_file.replace('.csv', '')} sharpe ratio: {sharpe.get('sharperatio', 0.0)}")
        if custom.get('bad_trades'):
            print(f"\n⚠️ Trades hurting Sharpe: {custom['bad_trades']}")
            for i in custom['bad_trades']:
                t = strat.analyzers.custom_metrics.trades[i]
                print(f"Trade {i}: Return={t['return']:.4f}, PnL={t['pnlcomm']:.2f}, Duration={t['duration']} bars")

    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values(by='Total Return %', ascending=False)

    # Optional: Bar plot of returns
    plt.figure(figsize=(8, 5))
    plt.bar(df_results['Symbol'], df_results['Total Return %'])
    plt.title(f'Strategy Benchmark')
    plt.ylabel('Return %')
    plt.grid(True, axis='y')
    plt.show()

    return df_results


    #equity = strat.analyzers.custom_metrics.values  # or cerebro.run()[0].equity_curve

    #plt.figure(figsize=(10, 5))
    #plt.plot(equity, label='Equity Curve')
    #plt.title('Equity Curve')
    #plt.xlabel('Bar Number')
    #plt.ylabel('Portfolio Value')
    #plt.grid(True)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

symbol_files = [
    'PEP.csv',
    'BX.csv',
    'ASML.csv',
    'UNH.csv',
    'AMZN.csv',
    'KDP.csv',
    'NKE.csv',
    'PFE.csv',
    'PG.csv',
    'SHOP.csv',
    'UNH.csv'
]
# Run with your preferred strategy
RunSimulation(RsiEmaStrategy, symbol_files)