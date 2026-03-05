import pandas as pd
import numpy as np
import os
from datetime import timedelta
from prophet import Prophet
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Import necessary functions from time_series_analyzer
from time_series_analyzer import load_and_prepare_data, calculate_mape, classify_prediction

# Configuration
PREDICTION_PERIOD = 30  # Predict 30 days ahead
TEST_SET_SIZE = 30  # Use last 30 days for testing

def calculate_technical_indicators(df):
    # Calculate Bollinger Bands
    # window=20, window_dev=2 are common values
    df['bb_bbm'] = BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_mavg()
    df['bb_bbh'] = BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_hband()
    df['bb_bbl'] = BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_lband()

    # Calculate MACD
    # window_slow=26, window_fast=12, window_sign=9 are common values
    df['macd'] = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
    df['macd_signal'] = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_signal()

    # Calculate RSI (already have 44-day MA, assuming basic RSI needed here if not already in load_and_prepare_data)
    # window=14 is common value
    df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Calculate additional moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    return df

def calculate_lagged_features(df):
    # Daily returns for volatility
    df['daily_return'] = df['Close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=PREDICTION_PERIOD).std()

    # Lagged monthly returns (approx. 21 trading days per month)
    for lag in [3, 6, 9, 12]:
        df[f'lagged_return_{lag}m'] = df['Close'].pct_change(periods=21 * lag)

    return df

def run_prophet_enhanced_model(daily_df, ticker_symbol):
    prophet_df = daily_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    if pd.api.types.is_datetime64_any_dtype(prophet_df['ds']) and prophet_df['ds'].dt.tz is not None:
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    # Merge technical indicators and lagged features
    # Ensure indices align correctly
    prophet_df = pd.merge(prophet_df, daily_df.reset_index().rename(columns={'Date': 'ds'}), on='ds', how='left')

    # Drop any remaining NaNs introduced by feature calculation (e.g., initial rolling periods)
    prophet_df.dropna(inplace=True)

    if len(prophet_df) <= (TEST_SET_SIZE + PREDICTION_PERIOD):
        print(f"    Not enough data for Prophet train-test split for {ticker_symbol}. Skipping.")
        return np.nan

    train_df = prophet_df.iloc[:-(TEST_SET_SIZE + PREDICTION_PERIOD)]
    actual_future_for_mape = prophet_df.iloc[-PREDICTION_PERIOD:]

    m = Prophet()

    # Add regressors
    regressors = ['bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'macd_signal', 'rsi', 'MA_7', 'MA_21', 'MA_50', 'MA_200', 'volatility']
    for lag in [3, 6, 9, 12]:
        regressors.append(f'lagged_return_{lag}m')

    for regressor in regressors:
        if regressor in train_df.columns:
            m.add_regressor(regressor)
        else:
            print(f"Warning: Regressor {regressor} not found in training data for {ticker_symbol}. Skipping.")
            
    m.fit(train_df)

    future_pred_df = m.make_future_dataframe(periods=PREDICTION_PERIOD, include_history=False)

    # Merge future_pred_df with actual future values for regressors
    # This is critical: Prophet needs future values of regressors for prediction
    # For real-world prediction, these would need to be forecasted or external data.
    # For evaluation, we use actual future values.
    future_pred_df = pd.merge(future_pred_df, prophet_df[regressors + ['ds']], on='ds', how='left')
    future_pred_df.dropna(inplace=True)

    if future_pred_df.empty or len(future_pred_df) < PREDICTION_PERIOD:
        print(f"    Not enough future regressor data for {ticker_symbol}. Skipping prediction.")
        return np.nan

    forecast = m.predict(future_pred_df)

    # MAPE calculation
    y_true_mape = actual_future_for_mape['y'].values
    y_pred_mape = forecast['yhat'].values

    # Ensure y_true_mape and y_pred_mape have the same length
    min_len = min(len(y_true_mape), len(y_pred_mape))
    y_true_mape = y_true_mape[:min_len]
    y_pred_mape = y_pred_mape[:min_len]

    if min_len == 0:
        print(f"    No data for MAPE calculation for {ticker_symbol}. Skipping.")
        return np.nan

    prophet_mape = calculate_mape(y_true_mape, y_pred_mape)

    print(f"    Prophet Enhanced MAPE for {ticker_symbol}: {prophet_mape:.2f}")
    return prophet_mape

def main(ticker_list_file='nse_tickers_screener.csv', daily_data_dir='daily_data'):
    print("\n--- Starting Prophet Enhanced Prediction ---")

    try:
        df_tickers = pd.read_csv(ticker_list_file)
        tickers_to_process = df_tickers['Ticker'].tolist()
        # For faster comparison, process a subset of tickers.
        tickers_to_process = tickers_to_process[:25]
        print(f"Loaded {len(tickers_to_process)} tickers from {ticker_list_file}")
    except FileNotFoundError:
        print(f"Error: Ticker list file '{ticker_list_file}' not found.")
        return
    except KeyError:
        print(f"Error: '{ticker_list_file}' must contain a 'Ticker' column.")
        return

    all_mapes = []
    for i, ticker in enumerate(tickers_to_process):
        print(f"\nProcessing {i+1}/{len(tickers_to_process)}: {ticker}")
        daily_df, _ = load_and_prepare_data(ticker, daily_data_dir) # Fundamental data not used for direct prediction here

        if daily_df.empty or 'Close' not in daily_df.columns:
            print(f"  Skipping {ticker} due to missing or empty daily data.")
            continue

        # `load_and_prepare_data` already sets 'Date' as index, ensure it's a proper datetime index
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.to_datetime(daily_df.index)
        daily_df.sort_index(inplace=True)

        # Calculate new technical indicators and lagged features
        daily_df = calculate_technical_indicators(daily_df)
        daily_df = calculate_lagged_features(daily_df)

        mape = run_prophet_enhanced_model(daily_df, ticker)
        if not np.isnan(mape):  # Only append if MAPE is a valid number
            all_mapes.append(mape)

    avg_mape = np.mean(all_mapes) if all_mapes else np.nan
    print(f"\n--- Prophet Enhanced Prediction Summary ---")
    print(f"Average MAPE for Prophet with Enhanced Features: {avg_mape:.2f}")

if __name__ == "__main__":
    # Install required libraries if not already installed:
    # pip install prophet ta scikit-learn
    main()
