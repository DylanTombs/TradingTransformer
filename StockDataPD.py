import pandas as pd
import os
import requests
import time
from datetime import datetime

# Configuration
API_KEY = ""  # Replace with your key

def get_alphavantage_data(symbol, start_date, end_date):
    """Fetches daily OHLCV data with built-in caching"""
    cache_file = f"{symbol}.csv"
    
    # Try cache first
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col='date', parse_dates=True)
        print(f"Loaded cached data for {symbol}")
        return data
    
    # API request
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full&datatype=csv"
    
    try:
        print(f"Downloading {symbol} from Alpha Vantage...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse CSV
        data = pd.read_csv(response.url, index_col='timestamp', parse_dates=True)
        data = data.sort_index()
        data = data.loc[start_date:end_date]
        
        # Save to cache
        data.to_csv(cache_file)
        print(f"Saved new data for {symbol}")
        return data
        
    except Exception as e:
        print(f"API failed: {e}")
        return get_backup_data(symbol)  # Fallback to local data

def get_backup_data(symbol):
    """Emergency fallback with pre-loaded sample data"""
    print("⚠️ Using built-in backup data")
    sample_data = {
        'AAPL': pd.DataFrame({
            'open': [134.50, 135.00],
            'high': [135.00, 136.50],
            'low': [133.50, 134.50],
            'close': [134.80, 136.20],
            'volume': [1000000, 1200000]
        }, index=pd.to_datetime(['2020-01-02', '2020-01-03']))
    }
    return sample_data.get(symbol, pd.DataFrame())

# Usage Example
aapl_data = get_alphavantage_data("ASML", "2006-01-01", "2023-01-01")
print(aapl_data.head())