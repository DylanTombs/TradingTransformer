import pandas as pd
import numpy as np



def calculate_technical_indicators(df):
    """Compute technical indicators and features."""
    df = df.copy()
    returns = df['close'].pct_change()

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].clip(20, 80)

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    # Volume z-score
    rolling_vol = df['volume'].rolling(20)
    df['volume_zscore'] = (df['volume'] - rolling_vol.mean()) / (rolling_vol.std() + 1e-6)

    # Volatility
    df['volatility'] = returns.rolling(20).std()

    # Gaps and return lags
    df['overnight_gap'] = np.log(df['open'] / df['close'].shift(1))
    for lag in [1, 3, 5]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    #Target: 3-day forward return (binary)
    #future_returns = df['close'].pct_change(3).shift(-3)
    #df['target'] = (future_returns > 0).astype(np.float32)

    # Drop unnecessary and clean
    df.drop(columns=['open', 'high', 'low', 'log_close'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

def process_ticker_to_df(ticker_path):
    """Processes a single ticker CSV and returns a cleaned DataFrame with stock ID."""
    df = pd.read_csv(ticker_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = calculate_technical_indicators(df)
    return df

def create_combined_csv(ticker_paths, output_path='all_stocks_processed.csv'):
    """Combines processed DataFrames from multiple tickers into a single CSV."""

    all_data = []
    for path in ticker_paths:
        print(f"Processing {path}...")
        df = process_ticker_to_df(path)
        all_data.append(df)

    combined_df = pd.concat(all_data)
    combined_df.reset_index(inplace=True)  # 'timestamp' becomes a column again
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to: {output_path}")

# Example usage
if __name__ == '__main__':
    ticker_paths = [
        'Data/AAPL.csv',
        'Data/ADBE.csv',
        'Data/AMD.csv',
        'Data/AXP.csv',
        'Data/BAC.csv',
        'Data/BLK.csv',
        'Data/C.csv',
        'Data/CRM.csv',
        'Data/CVX.csv',
        'Data/GOOGL.csv',
        'Data/GS.csv',
        'Data/INTC.csv',
        'Data/JNJ.csv',
        'Data/JPM.csv',
        'Data/MA.csv',
        'Data/META.csv',
        'Data/MS.csv',
        'Data/MSFT.csv',
        'Data/TSLA.csv',
        'Data/V.csv',
        'Data/WMT.csv',
    ]
    create_combined_csv(ticker_paths)
