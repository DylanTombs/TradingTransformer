import numpy as np
import pandas as pd

def calculate_technical_indicators(df):
    """Compute technical indicators and features."""
    df = df.copy()
    returns = df['close'].pct_change()

    # Keep original OHLCV data (needed for your auxilFeatures)
    # Don't drop 'high', 'low' as they're in your feature list
    
    # Calculate 'adj close' if not present (assuming it's the same as close for now)
    if 'adj close' not in df.columns:
        df['adj close'] = df['close']

    # Pivot Points and Support/Resistance levels
    df['P'] = (df['high'] + df['low'] + df['close']) / 3  # Pivot Point
    df['R1'] = 2 * df['P'] - df['low']  # Resistance 1
    df['R2'] = df['P'] + (df['high'] - df['low'])  # Resistance 2
    df['R3'] = df['high'] + 2 * (df['P'] - df['low'])  # Resistance 3
    df['S1'] = 2 * df['P'] - df['high']  # Support 1
    df['S2'] = df['P'] - (df['high'] - df['low'])  # Support 2
    df['S3'] = df['low'] - 2 * (df['high'] - df['P'])  # Support 3

    # On-Balance Volume (OBV)
    df['obv'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()

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
    df['macds'] = df['macd'].ewm(span=9, adjust=False).mean()  # MACD Signal
    df['macdh'] = df['macd'] - df['macds']  # MACD Histogram

    # Moving Averages
    df['sma'] = df['close'].rolling(20).mean()  # Simple Moving Average
    df['lma'] = df['close'].rolling(50).mean()  # Long Moving Average
    df['sema'] = df['close'].ewm(span=12, adjust=False).mean()  # Short EMA
    df['lema'] = df['close'].ewm(span=26, adjust=False).mean()  # Long EMA

    # Volume z-score
    rolling_vol = df['volume'].rolling(20)
    df['volume_zscore'] = (df['volume'] - rolling_vol.mean()) / (rolling_vol.std() + 1e-6)

    # Volatility
    df['volatility'] = returns.rolling(20).std()

    # Gaps and return lags
    df['overnight_gap'] = np.log(df['open'] / df['close'].shift(1))
    for lag in [1, 3, 5]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    # Stochastic Oscillator (%K and %D)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['SR_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['SR_D'] = df['SR_K'].rolling(3).mean()

    # Stochastic RSI (%K and %D)
    rsi_low_14 = df['rsi'].rolling(14).min()
    rsi_high_14 = df['rsi'].rolling(14).max()
    df['SR_RSI_K'] = 100 * (df['rsi'] - rsi_low_14) / (rsi_high_14 - rsi_low_14 + 1e-10)
    df['SR_RSI_D'] = df['SR_RSI_K'].rolling(3).mean()

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    df['ATR'] = true_range.rolling(14).mean()

    # High-Low Percentage
    df['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100

    # Percentage Change
    df['PCT_CHG'] = df['close'].pct_change() * 100

    # Drop unnecessary columns and clean
    df.drop(columns=['open', 'log_close'], inplace=True, errors='ignore')
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
    from pathlib import Path
    
    all_data = []
    for path in ticker_paths:
        print(f"Processing {path}...")
        
        # Extract ticker from filename (e.g., "AAPL_data.csv" -> "AAPL")
        ticker = Path(path).stem.split('_')[0].upper()
        
        df = process_ticker_to_df(path)
        df['ticker'] = ticker  # Add ticker column
        all_data.append(df)

    combined_df = pd.concat(all_data)
    combined_df.reset_index(inplace=True)  # 'timestamp' becomes a column again
    combined_df.rename(columns={'timestamp': 'date'}, inplace=True)  # Match splitData expectation
    
    # Sort by ticker then date to group stocks together
    combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
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
