import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from Strategies.RsiEmaStrategy import RsiEmaStrategy
from Strategies.MACDMomentumTrend import MacdMomentumTrend
from Strategies.MeanReversion import MeanReversionSimple
from Strategies.VWAPBreakout import VWAPBreakout
from RsiPortfolioStrategy import RsiPortfolioStrategy


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

def RunSimulation(strategy, symbol_files, cash=1000, saveResults = True):
    results_dir = 'Results/VWAPStrat/Results1'
    results_summary = []
    all_equity = {}
    all_trades = []
    for symbol_file in symbol_files:

        symbol = symbol = os.path.basename(symbol_file).replace('.csv', '')
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
        cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
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
            'Symbol': symbol,
            'Start Value': startVal,
            'End Value': endVal,
            'Total Return %': (endVal / startVal - 1) * 100,
            'Sharpe Ratio': sharpe.get('sharperatio', 0.0),
            'Max Drawdown %': drawdown['max']['drawdown'],
            'Win Rate %': custom['win_rate'],
            'Profit Factor': custom['profit_factor'],
            'Avg R/R': custom['avg_risk_reward'],
        })

        if hasattr(strat.analyzers.custom_metrics, 'trades'):
            for trade in strat.analyzers.custom_metrics.trades:
                trade['symbol'] = symbol
                all_trades.append(trade)

        
        if saveResults:
            fig = cerebro.plot(style='candlestick', iplot=False)[0][0]
            plt.title(f'{symbol} - Strategy')
            fig = cerebro.plot(style='candlestick', iplot=False)[0][0]
            plt.title(f'{symbol} - Strategy')
            plot_path = os.path.join(results_dir, f'{symbol}_Strategy.png')
            fig.savefig(plot_path)
            plt.close(fig)
        
            equity = strat.analyzers.custom_metrics.values  # or cerebro.run()[0].equity_curve

            plt.figure(figsize=(10, 5))
            plt.plot(equity, label='Equity Curve')
            plt.title('Equity Curve')
            plt.xlabel('Bar Number')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
        
            plot_path = os.path.join(results_dir, f'{symbol}_Equity_Curve.png')
            plt.savefig(plot_path)
            plt.close(fig)
        
        # Store equity data for aggregate plots
        equity_data = pd.DataFrame({
            'Date': data.index,
            'Equity': cerebro.broker.getvalue(),
            'Symbol': symbol
        })
        all_equity[symbol] = equity_data
        
        print(f"\nCompleted {symbol}")

        equity_data = pd.DataFrame({
            'Date': data.index,
            'Equity': cerebro.broker.getvalue(),
            'Symbol': symbol
        })
        all_equity[symbol] = equity_data
    
        print(f"{symbol} sharpe ratio: {sharpe.get('sharperatio', 0.0)}")
        if custom.get('bad_trades'):
            print(f"\n Trades hurting Sharpe: {custom['bad_trades']}")
            for i in custom['bad_trades']:
                t = strat.analyzers.custom_metrics.trades[i]
                print(f"Trade {i}: Return={t['return']:.4f}, PnL={t['pnlcomm']:.2f}, Duration={t['duration']} bars")

    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values(by='Total Return %', ascending=False)

    
    results_file = os.path.join(results_dir, f'strategy_results1.csv')
    df_results.to_csv(results_file, index=False)
    print(f"\nSaved results to {results_file}")

    create_aggregate_plots(df_results, all_equity, all_trades, saveResults, results_dir)
    
    return df_results, all_equity, all_trades

def RunPortfolioSimulation(strategy, symbol_files, cash=100000, saveResults=True):

    results_dir = 'Results/Results6'
    os.makedirs(results_dir, exist_ok=True)

    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(cash)

    # Optional but recommended: model some friction
    #cerebro.broker.setcommission(commission=0.0005)  # 5 bps per side
    # If you want slippage modeling, add a slippage scheme here.

    # Load all symbols into one broker
    for symbol_file in symbol_files:
        symbol = symbol_file.replace('.csv', '')
        data = pd.read_csv(symbol_file)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        datafeed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='open', high='high', low='low', close='close',
            volume='volume', openinterest=-1
        )
        cerebro.adddata(datafeed, name=symbol)

    # One strategy trading all data feeds simultaneously
    cerebro.addstrategy(strategy)

    # Portfolio-level analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')   # annualized Sharpe
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')  # equity curve returns
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')      # portfolio drawdown
    cerebro.addanalyzer(StrategyEvaluator, _name='custom_metrics')    # your custom trade stats

    startVal = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    endVal = cerebro.broker.getvalue()

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    custom = strat.analyzers.custom_metrics.get_analysis()

    summary = {
        'Start Value': startVal,
        'End Value': endVal,
        'Total Return %': (endVal / startVal - 1) * 100,
        'Sharpe Ratio': sharpe.get('sharperatio', 0.0),
        'Max Drawdown %': drawdown['max']['drawdown'],
        'Win Rate %': custom['win_rate'],
        'Profit Factor': custom['profit_factor'],
        'Avg R/R': custom['avg_risk_reward'],
    }
    df_results = pd.DataFrame([summary])

    # Portfolio equity curve from your analyzer
    plt.figure(figsize=(12, 6))
    plt.plot(strat.analyzers.custom_metrics.values, label="Portfolio Equity")
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Bars")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()

    if saveResults:
        plt.savefig(os.path.join(results_dir, "portfolio_equity.png"))
        plt.close()
        df_results.to_csv(os.path.join(results_dir, "portfolio_results.csv"), index=False)
    else:
        plt.show()

    print("\nPortfolio Results:")
    print(df_results)

    return df_results, strat

def create_aggregate_plots(df_results, all_equity, all_trades, save_results, results_dir):
    """Create various aggregate plots for strategy analysis"""
    
    # 1. Performance Bar Plot
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='Symbol', y='Total Return %', data=df_results)
    plt.title('Strategy Performance by Symbol')
    plt.ylabel('Total Return %')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
    if save_results:
        plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
        plt.close()
    else:
        plt.show()
    
    # 2. Risk-Reward Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Max Drawdown %', y='Total Return %', 
                    size='Sharpe Ratio', hue='Symbol',
                    data=df_results, sizes=(50, 200))
    plt.title('Risk-Reward Profile')
    plt.grid(True)
    
    if save_results:
        plt.savefig(os.path.join(results_dir, 'risk_reward.png'))
        plt.close()
    else:
        plt.show()
    
    # 3. Combined Equity Curve
    plt.figure(figsize=(14, 7))
    for symbol, equity_data in all_equity.items():
        plt.plot(equity_data['Date'], equity_data['Equity'], label=symbol)
    
    plt.title('Combined Equity Curves')
    plt.ylabel('Portfolio Value')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    if save_results:
        plt.savefig(os.path.join(results_dir, 'combined_equity.png'))
        plt.close()
    else:
        plt.show()
    
    # 4. Trade Analysis (if we have trades)
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        
        # Trade Duration Distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df_trades['duration'], bins=20, kde=True)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Duration (bars)')
        
        # Trade Returns Distribution
        plt.subplot(1, 2, 2)
        sns.histplot(df_trades['return']*100, bins=20, kde=True)
        plt.title('Trade Returns Distribution')
        plt.xlabel('Return %')
        plt.tight_layout()
        
        if save_results:
            plt.savefig(os.path.join(results_dir, 'trade_distributions.png'))
            plt.close()
        else:
            plt.show()
        
        # Win/Loss by Symbol
        if len(df_trades['symbol'].unique()) > 1:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='symbol', y='return', data=df_trades)
            plt.title('Return Distribution by Symbol')
            plt.ylabel('Return')
            plt.xlabel('Symbol')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            
            if save_results:
                plt.savefig(os.path.join(results_dir, 'returns_by_symbol.png'))
                plt.close()
            else:
                plt.show()

symbol_files = [
        "BackTesting/ASML.csv",
        "BackTesting/UNH.csv",
        "BackTesting/DFS.csv",
        "BackTesting/PEP.csv",
        "BackTesting/LMT.csv",
    ]

symbol_files = [
    os.path.join("Backtesting/Volatile", file) 
        for file in os.listdir("Backtesting/Volatile") 
        if file.endswith('.csv')
]

#RunSimulation(RsiEmaStrategy, symbol_files)
RunSimulation(VWAPBreakout, symbol_files, saveResults=True)