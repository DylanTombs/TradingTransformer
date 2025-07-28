import optuna
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from RsiEmaStrategy import RsiEmaStrategy  # Make sure this points to your updated strategy file

# === Load your datasets here ===
TICKERS = ["PFE", "PEP","NKE","BX","ASML"]  # Add more tickers as needed

def load_data(ticker):
    df = pd.read_csv(f"{ticker}.csv", index_col='date', parse_dates=True)
    df = df.sort_index()
    data = bt.feeds.PandasData(dataname=df)
    return data

def run_backtest(params, tickers):
    results = []
    for ticker in tickers:
        cerebro = bt.Cerebro()
        data = load_data(ticker)
        cerebro.adddata(data)

        cerebro.addstrategy(
            RsiEmaStrategy,
            emaPeriod=params['emaPeriod'],
            rsiPeriod=params['rsiPeriod'],
            seqLen=params['seqLen'],
            labelLen=params['labelLen'],
            predLen=params['predLen'],
            buy_threshold=params['buy_threshold'],
            sell_threshold=params['sell_threshold'],
            rsi_buy=params['rsi_buy'],
            rsi_sell=params['rsi_sell']
        )

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        result = cerebro.run()[0]

        sharpe = result.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        dd = result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
        total_return = result.analyzers.returns.get_analysis().get('rnorm100', 0.0)


        results.append({
            "ticker": ticker,
            "sharpe": sharpe,
            "drawdown": dd,
            "return": total_return
        })

    # Aggregate Sharpe ratios across assets (mean or risk-weighted)
    mean_sharpe = np.mean([r['sharpe'] for r in results])
    return mean_sharpe

def objective(trial):
    params = {
        "emaPeriod": trial.suggest_int("emaPeriod", 5, 30),
        "rsiPeriod": trial.suggest_int("rsiPeriod", 7, 21),
        "seqLen": 30,
        "labelLen": 10,
        "predLen": 5,
        "buy_threshold": trial.suggest_float("buy_threshold", 1.002, 1.015),
        "sell_threshold": trial.suggest_float("sell_threshold", 0.985, 0.998),
        "rsi_buy": trial.suggest_int("rsi_buy", 30, 50),
        "rsi_sell": trial.suggest_int("rsi_sell", 50, 70),
    }

    try:
        score = run_backtest(params, TICKERS)
        return score
    except Exception as e:
        print(f"Trial failed: {e}")
        return -999  # Penalize invalid trials

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="RsiEma_OptStudy")
    study.optimize(objective, n_trials=50, timeout=360000)  # Adjust trials/time as needed

    print("Best parameters:")
    print(study.best_trial.params)
    print(f"Best Sharpe: {study.best_value}")
