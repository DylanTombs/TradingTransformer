import optuna
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import random
import hashlib

from RsiEmaStrategy import RsiEmaStrategy  # Ensure path is correct

# === CONFIG ===
TICKERS = ['PEP', 'BX', 'ASML', 'UNH']
RESULTS_CSV = Path("experiment_results.csv")
SEED = 42

# === SEEDING FOR REPRODUCIBILITY ===
np.random.seed(SEED)
random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(ticker):
    df = pd.read_csv(f"{ticker}.csv", index_col='date', parse_dates=True)
    df = df.sort_index()
    return bt.feeds.PandasData(dataname=df)

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

        sharpe = result.analyzers.sharpe.get_analysis().get('sharperatio', np.nan)
        dd = result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', np.nan)
        total_return = result.analyzers.returns.get_analysis().get('rnorm100', np.nan)

        results.append({
            "ticker": ticker,
            "sharpe": sharpe,
            "drawdown": dd,
            "return": total_return
        })

    df_results = pd.DataFrame(results)
    mean_sharpe = df_results["sharpe"].mean()
    return mean_sharpe, df_results

def save_trial_results(trial, mean_sharpe, per_ticker_df):
    trial_data = {
        "trial_number": trial.number,
        "timestamp": datetime.utcnow().isoformat(),
        "config_hash": hashlib.md5(str(trial.params).encode()).hexdigest(),
        "mean_sharpe": mean_sharpe,
        **trial.params
    }
    # Merge per-ticker stats into trial_data
    for _, row in per_ticker_df.iterrows():
        trial_data[f"{row['ticker']}_sharpe"] = row["sharpe"]
        trial_data[f"{row['ticker']}_drawdown"] = row["drawdown"]
        trial_data[f"{row['ticker']}_return"] = row["return"]

    # Append to CSV
    df_out = pd.DataFrame([trial_data])
    if RESULTS_CSV.exists():
        df_out.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_out.to_csv(RESULTS_CSV, index=False)

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
        mean_sharpe, per_ticker_df = run_backtest(params, TICKERS)
        save_trial_results(trial, mean_sharpe, per_ticker_df)
        return mean_sharpe
    except Exception as e:
        print(f"[Trial {trial.number}] FAILED: {e}")
        save_trial_results(trial, -999, pd.DataFrame({"ticker": TICKERS, "sharpe": [np.nan]*len(TICKERS), "drawdown": np.nan, "return": np.nan}))
        return -999  # Penalize

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="RsiEma_OptStudy")
    study.optimize(objective, n_trials=50, timeout=360000)

    print("\n=== BEST PARAMETERS ===")
    print(study.best_trial.params)
    print(f"Best Sharpe: {study.best_value}")
    print(f"Results saved to {RESULTS_CSV}")

