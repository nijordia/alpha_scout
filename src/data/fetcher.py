import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_stock_data(symbol, start_date=None, end_date=None):
    """
    Fetch stock data from Yahoo Finance using yfinance library.
    
    :param symbol: Stock ticker symbol (e.g., 'AAPL')
    :param start_date: Start date in 'YYYY-MM-DD' format (default: 1 year ago)
    :param end_date: End date in 'YYYY-MM-DD' format (default: today)
    :return: DataFrame with OHLCV data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        # Use yfinance to download the data
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date)

        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Convert timezone-aware dates to timezone-naive
        if 'Date' in data.columns and hasattr(data['Date'].iloc[0], 'tz') and data['Date'].iloc[0].tz is not None:
            data['Date'] = data['Date'].dt.tz_localize(None)
        
        # Rename columns to lowercase for consistency
        new_columns = []
        for col in data.columns:
            if isinstance(col, tuple):
                # Join multi-level column names and convert to lowercase
                new_col = str(col[0]).lower()
                new_columns.append(new_col)
            else:
                # Just convert single level columns to lowercase
                new_columns.append(col.lower())
        
        data.columns = new_columns
        
        if len(data) > 0:
            print(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
        else:
            print(f"No data found for {symbol}")
            return None
            
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def test_fetch_apple_stock():
    """
    Test function to fetch Apple stock data and print basic information.
    """
    # Fetch one year of Apple stock data
    apple_data = fetch_stock_data('AAPL')
    
    if apple_data is not None:
        print("\nApple Stock Data Sample:")
        print(apple_data.head())
        
        print("\nData Shape:", apple_data.shape)
        print("\nData Types:", apple_data.dtypes)
        
        print("\nDate Range:")
        print(f"Start: {apple_data['date'].min()}")
        print(f"End: {apple_data['date'].max()}")
        
        print("\nBasic Statistics:")
        print(apple_data[['open', 'high', 'low', 'close', 'volume']].describe())
    else:
        print("Failed to fetch Apple stock data.")
    
    return apple_data

# If this file is executed directly, run the test function
#if __name__ == "__main__":
#    test_fetch_apple_stock()