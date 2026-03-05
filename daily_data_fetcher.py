import yfinance as yf
import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    :param data: pandas Series of closing prices.
    :param window: The period for RSI calculation (default is 14).
    :return: pandas Series with RSI values.
    """
    diff = data.diff(1).dropna()
    gain = ((diff > 0) * diff).fillna(0)
    loss = ((diff < 0) * abs(diff)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_daily_data(ticker_symbol):
    """
    Fetches daily OHLCV data and calculates RSI for a given ticker symbol using yfinance.
    :param ticker_symbol: The stock ticker symbol (e.g., "RELIANCE.NS").
    :return: A pandas DataFrame with OHLCV and RSI data, or an empty DataFrame if data is not found or an error occurs.
    """
    try:
        # Download daily OHLCV data for the last 5 years
        df = yf.download(ticker_symbol, period="5y", interval="1d", progress=False)
        if df.empty:
            print(f"  No daily data found for {ticker_symbol} for the last 5 years.")
            return pd.DataFrame()

        # Flatten MultiIndex columns produced by yfinance >= 0.2
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty:
            # Ensure 'Close' column is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)

            # Calculate RSI
            df['RSI'] = calculate_rsi(df['Close'])

        return df
    except Exception as e:
        print(f"Error fetching daily data for {ticker_symbol}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    sample_ticker = "RELIANCE.NS"
    daily_df = get_daily_data(sample_ticker)
    if not daily_df.empty:
        print(f"\nDaily Data for {sample_ticker}:\n{daily_df.head()}")
