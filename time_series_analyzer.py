import pandas as pd
import numpy as np
import os
from datetime import timedelta

# For Prophet model
from prophet import Prophet

# For ARIMA model
# import pmdarima as pm # Commented out due to installation issues

# For Theta model (using Exponential Smoothing as a proxy)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def calculate_moving_average(data, window=44):
    """
    Calculates the Simple Moving Average (SMA) for a given data series.
    :param data: pandas Series of prices.
    :param window: The period for SMA calculation (default is 44).
    :return: pandas Series with SMA values.
    """
    return data.rolling(window=window).mean()

def _get_price_by_date(df, target_date):
    """
    Helper function to get price by date, handling potential missing dates and timezones.
    """
    # Ensure target_date is timezone-naive if df index is naive
    if df.index.tz is not None and target_date.tz is None:
        target_date = target_date.tz_localize(df.index.tz)
    elif df.index.tz is None and target_date.tz is not None:
        target_date = target_date.tz_localize(None)
    
    # Try direct lookup first, then nearest if not found
    if target_date in df.index:
        return df.loc[target_date]['Close']
    else:
        # Fallback to absolute nearest if direct lookup fails
        # Using get_indexer for more robust nearest date lookup
        idx = df.index.get_indexer([target_date], method='nearest')[0]
        return df.iloc[idx]['Close']

def load_and_prepare_data(ticker_symbol, daily_data_dir='daily_data', fundamental_file='all_fundamental_data.csv'):
    """
    Loads daily and fundamental data for a given ticker and prepares it for time series analysis.
    Calculates 44-day Moving Average and combines relevant features.
    :param ticker_symbol: The stock ticker symbol (e.g., "RELIANCE.NS").
    :param daily_data_dir: Directory where daily data CSVs are stored.
    :param fundamental_file: Path to the aggregated fundamental data CSV.
    :return: A tuple containing (prepared_daily_df, fundamental_data_series).
    """
    print(f"Loading and preparing data for {ticker_symbol}...")

    # Load Daily Data
    daily_file_name = f'{ticker_symbol.replace(".NS", "")}_daily.csv'
    daily_file_path = os.path.join(daily_data_dir, daily_file_name)
    daily_df = pd.DataFrame()
    try:
        daily_df = pd.read_csv(daily_file_path, index_col=0, parse_dates=True) # Use index_col=0 for the first column
        daily_df.index.name = 'Date' # Explicitly name the index 'Date'
        daily_df.sort_index(inplace=True)
        print(f"  Loaded daily data from {daily_file_path}.")
    except FileNotFoundError:
        print(f"  Daily data file not found for {ticker_symbol}: {daily_file_path}")
        return pd.DataFrame(), pd.Series()
    except Exception as e:
        print(f"  Error loading daily data for {ticker_symbol}: {e}")
        return pd.DataFrame(), pd.Series()

    # Feature Engineering for Daily Data
    if not daily_df.empty and 'Close' in daily_df.columns:
        # Ensure the index is a timezone-naive datetime index before any calculations or model prep
        if pd.api.types.is_datetime64_any_dtype(daily_df.index):
            if daily_df.index.tz is not None:
                daily_df.index = daily_df.index.tz_localize(None) # Remove timezone for entire DataFrame index
        
        daily_df['MA_44'] = calculate_moving_average(daily_df['Close'], window=44)
        # Ensure RSI is present or calculate if needed (it should be from daily_data_fetcher)
        if 'RSI' not in daily_df.columns:
            print(f"  Warning: RSI not found in daily data for {ticker_symbol}. Recalculating.")
            from daily_data_fetcher import calculate_rsi
            daily_df['RSI'] = calculate_rsi(daily_df['Close'])
        print(f"  Calculated MA_44 and ensured RSI for {ticker_symbol}.")

    # Prophet expects columns 'ds' (datestamp) and 'y' (value)
    # Prophet does not support timezone-aware datetimes, so ensure 'ds' is naive.
    prophet_df = daily_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    # Double-check and remove timezone here again for robustness if it somehow re-appeared
    if pd.api.types.is_datetime64_any_dtype(prophet_df['ds']) and prophet_df['ds'].dt.tz is not None:
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    # Theta/ETS expects a simple series with a regular frequency
    ts_data = daily_df['Close']
    # Ensure the index has a frequency, fill missing dates, and ensure timezone-naive
    if not ts_data.index.empty:
        ts_data = ts_data.asfreq('D')
        if ts_data.isnull().any():
            print(f"  Warning: `ts_data` for {ticker_symbol} contains NaNs after setting daily frequency. Filling with ffill.")
            ts_data = ts_data.ffill().bfill() # ffill and then bfill to handle leading NaNs
        # Ensure timezone-naive for statsmodels as well
        if ts_data.index.tz is not None:
            ts_data.index = ts_data.index.tz_localize(None)
    else:
        print(f"  Warning: `ts_data` for {ticker_symbol} is empty after preparation.")
        ts_data = pd.Series()

    # Load Fundamental Data
    fundamental_data = pd.Series()
    try:
        df_fundamental = pd.read_csv(fundamental_file)
        fundamental_data = df_fundamental[df_fundamental['Ticker'] == ticker_symbol].iloc[0]
        print(f"  Loaded fundamental data for {ticker_symbol}.")
    except FileNotFoundError:
        print(f"  Fundamental data file not found: {fundamental_file}")
    except IndexError:
        print(f"  Fundamental data not found for {ticker_symbol} in {fundamental_file}.")
    except Exception as e:
        print(f"  Error loading fundamental data for {ticker_symbol}: {e}")

    return daily_df, fundamental_data

def run_time_series_prediction(ticker_symbol, daily_df, fundamental_data):
    """
    Runs time series prediction models (Prophet, ARIMA, Theta) for a given ticker.
    Classifies predictions and assigns confidence scores.
    :return: A dictionary with predictions, classifications, confidence, and MAPE for each model.
    """
    results = {
        'ticker': ticker_symbol,
        'prophet_prediction': None, 'prophet_mape': None, 'prophet_class': None, 'prophet_confidence': None,
        'arima_prediction': None, 'arima_mape': None, 'arima_class': None, 'arima_confidence': None,
        'theta_prediction': None, 'theta_mape': None, 'theta_class': None, 'theta_confidence': None,
    }

    if daily_df.empty or 'Close' not in daily_df.columns:
        print(f"  Skipping prediction for {ticker_symbol}: Insufficient daily data.")
        return results

    # For monthly prediction, we need to resample daily data to monthly or predict daily and then aggregate
    # For simplicity, we'll aim to predict the last month's closing price or an aggregate.
    # Let's predict the closing price one month ahead from the last available date.
    last_date = daily_df.index[-1]
    prediction_target_date = last_date + timedelta(days=30) # Roughly one month ahead

    # Prepare data for models
    # Prophet expects columns 'ds' (datestamp) and 'y' (value)
    prophet_df = daily_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # ARIMA/Theta expect a simple series
    ts_data = daily_df['Close']

    # --- Prophet Model (with Train-Test Split for MAPE) ---
    print(f"  Running Prophet for {ticker_symbol}...")
    try:
        if len(prophet_df) > 60: # Ensure enough data for both train and test (e.g., > 2 months)
            # Train-test split: use the last 30 days for testing, rest for training
            train_size = len(prophet_df) - 30
            train_df = prophet_df.iloc[:train_size]
            test_df = prophet_df.iloc[train_size:]

            m = Prophet()
            m.fit(train_df)

            # Make predictions for the test period
            future_test = m.make_future_dataframe(periods=len(test_df), include_history=False)
            forecast_test = m.predict(future_test)

            # Calculate MAPE on the test set
            y_true = test_df['y'].values
            y_pred = forecast_test['yhat'].values
            prophet_mape = calculate_mape(y_true, y_pred)
            results['prophet_mape'] = prophet_mape
            print(f"    Prophet MAPE on test set: {prophet_mape:.2f}%")

            # Now, predict one month (30 days) into the actual future using the full dataset
            m_full = Prophet() # Retrain on full data for actual future prediction
            m_full.fit(prophet_df)
            future_pred = m_full.make_future_dataframe(periods=30, include_history=False) # Predict 30 days into the future from last date
            forecast_pred = m_full.predict(future_pred)
            
            prophet_pred_value = forecast_pred['yhat'].iloc[-1]
            results['prophet_prediction'] = prophet_pred_value
            results['prophet_lower_interval'] = forecast_pred['yhat_lower'].iloc[-1]
            results['prophet_upper_interval'] = forecast_pred['yhat_upper'].iloc[-1]
            print(f"    Prophet predicted (1-month ahead): {prophet_pred_value:.2f}")
            print(f"    Prophet 95% Confidence Interval: [{results['prophet_lower_interval']:.2f}, {results['prophet_upper_interval']:.2f}]")

            # Calculate predicted monthly return
            last_close_price_full_data = _get_price_by_date(prophet_df, prediction_target_date)
            if last_close_price_full_data != 0:
                prophet_monthly_return = ((prophet_pred_value - last_close_price_full_data) / last_close_price_full_data) * 100
                results['prophet_monthly_return'] = prophet_monthly_return
                results['prophet_class'] = classify_prediction(prophet_monthly_return)
                results['prophet_confidence'] = calculate_confidence_and_boost(
                    prophet_mape, 
                    fundamental_data, 
                    last_close_price_full_data, 
                    results['prophet_lower_interval'], 
                    results['prophet_upper_interval']
                ) # Use actual MAPE for confidence
                print(f"    Prophet monthly return: {prophet_monthly_return:.2f}%, Class: {results['prophet_class']}, Confidence: {results['prophet_confidence']:.2f}")

        else:
            print(f"    Not enough data for Prophet train-test split for {ticker_symbol}. Skipping MAPE.")
            # Fallback for future prediction without MAPE if not enough data for split
            if len(prophet_df) > 2:
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=30) # Predict 30 days into the future
                forecast = m.predict(future)
                prophet_pred_value = forecast['yhat'].iloc[-1]
                results['prophet_prediction'] = prophet_pred_value
                results['prophet_lower_interval'] = forecast['yhat_lower'].iloc[-1]
                results['prophet_upper_interval'] = forecast['yhat_upper'].iloc[-1]
                print(f"    Prophet predicted (1-month ahead, no MAPE): {prophet_pred_value:.2f}")
                print(f"    Prophet 95% Confidence Interval: [{results['prophet_lower_interval']:.2f}, {results['prophet_upper_interval']:.2f}]")

                last_close_price_full_data = _get_price_by_date(prophet_df, prediction_target_date)
                if last_close_price_full_data != 0:
                    prophet_monthly_return = ((prophet_pred_value - last_close_price_full_data) / last_close_price_full_data) * 100
                    results['prophet_monthly_return'] = prophet_monthly_return
                    results['prophet_class'] = classify_prediction(prophet_monthly_return)
                    results['prophet_confidence'] = calculate_confidence_and_boost(
                        np.nan, 
                        fundamental_data, 
                        last_close_price_full_data,
                        results['prophet_lower_interval'],
                        results['prophet_upper_interval']
                    ) # No MAPE, use neutral for confidence
                    print(f"    Prophet monthly return: {prophet_monthly_return:.2f}%, Class: {results['prophet_class']}, Confidence: {results['prophet_confidence']:.2f}")

    except Exception as e:
        print(f"    Error running Prophet for {ticker_symbol}: {e}")

    # --- ARIMA Model ---
    # Temporarily commenting out ARIMA due to installation issues with pmdarima on Python 3.13.
    # Consider using a more stable Python version (e.g., 3.9-3.12) for pmdarima, or a different ARIMA implementation.
    # print(f"  Running ARIMA for {ticker_symbol}...")
    # try:
    #     # pmdarima automatically finds best ARIMA orders
    #     if len(ts_data) > 50: # Requires sufficient data
    #         arima_model = pm.auto_arima(ts_data, seasonal=False, suppress_warnings=True, stepwise=True)
    #         # Predict next 30 days, then take the last one or average for monthly
    #         forecast_arima = arima_model.predict(n_periods=30)
    #         arima_pred_value = forecast_arima[-1]
    #         results['arima_prediction'] = arima_pred_value
    #         print(f"    ARIMA predicted: {arima_pred_value:.2f}")
    #         results['arima_mape'] = np.nan
    #         results['arima_class'] = "N/A"
    #         results['arima_confidence'] = np.nan
    #     else:
    #         print(f"    Not enough data for ARIMA for {ticker_symbol}.")

    # except Exception as e:
    #     print(f"    Error running ARIMA for {ticker_symbol}: {e}")

    # --- Theta Model (using Exponential Smoothing as proxy) ---
    # Temporarily commenting out Theta/ETS due to persistent errors and warnings with statsmodels.tsa.api.ExponentialSmoothing.
    # For a robust Theta implementation, consider specialized time series libraries or custom development, possibly
    # inspired by R's forecast package, or a more stable Python environment.
    # print(f"  Running Theta (Exponential Smoothing) for {ticker_symbol}...")
    # try:
    #     if not ts_data.empty and len(ts_data) > 2:
    #         # Simple Exponential Smoothing as a proxy for Theta
    #         # Theta method is often implemented with 2 simple exponential smoothing models or similar.
    #         # Attempting to make it more robust with different initialization and simpler model if complex one fails.
    #         try:
    #             fit_ets = ExponentialSmoothing(ts_data, seasonal_periods=30, trend='add', seasonal='add', initialization_method="estimated").fit(optimized=True)
    #         except Exception as e:
    #             print(f"    Complex ExponentialSmoothing failed for {ticker_symbol} ({e}). Trying SimpleExpSmoothing.")
    #             # Fallback to a simpler model if the complex one fails
    #             fit_ets = SimpleExpSmoothing(ts_data, initialization_method="estimated").fit(optimized=True)

    #         forecast_ets = fit_ets.forecast(30) # Predict next 30 days
    #         theta_pred_value = forecast_ets[-1]
    #         results['theta_prediction'] = theta_pred_value
    #         print(f"    Theta (ETS) predicted: {theta_pred_value:.2f}")

    #         # Calculate predicted monthly return
    #         last_close_price = ts_data.iloc[-1]
    #         if last_close_price != 0:
    #             theta_monthly_return = ((theta_pred_value - last_close_price) / last_close_price) * 100
    #             results['theta_monthly_return'] = theta_monthly_return
    #             results['theta_class'] = classify_prediction(theta_monthly_return)
    #             results['theta_confidence'] = calculate_confidence_and_boost(results['theta_mape'], fundamental_data) # MAPE is nan for now
    #             print(f"    Theta (ETS) monthly return: {theta_monthly_return:.2f}%, Class: {results['theta_class']}, Confidence: {results['theta_confidence']:.2f}")

    #         results['theta_mape'] = np.nan
    #         # results['theta_class'] = "N/A"
    #         # results['theta_confidence'] = "N/A"
    #     else:
    #         print(f"    Not enough data for Theta (ETS) for {ticker_symbol} (len={len(ts_data)}).")
    # except Exception as e:
    #     print(f"    Critical Error running Theta (ETS) for {ticker_symbol}: {e}")
    
    return results

def classify_prediction(monthly_return):
    """
    Classifies a predicted monthly return into categories.
    :param monthly_return: Predicted monthly return percentage.
    :return: Classification string.
    """
    if monthly_return > 7:
        return "stellar"
    elif 2 <= monthly_return <= 7:
        return "profit"
    elif 0 <= monthly_return < 2:
        return "breakeven"
    elif -2 <= monthly_return < 0:
        return "loss"
    else:
        return "disaster"


def calculate_confidence_and_boost(model_mape, fundamental_data, last_close_price, lower_interval, upper_interval):
    """
    Calculates a basic confidence score and boosts/reduces it based on MAPE,
    prediction intervals, and fundamental data.
    :param model_mape: MAPE of the model (lower is better).
    :param fundamental_data: pandas Series of fundamental data for the ticker.
    :param last_close_price: The last known closing price of the stock.
    :param lower_interval: The lower bound of the prediction interval.
    :param upper_interval: The upper bound of the prediction interval.
    :return: Confidence score (0-100).
    """
    # Base confidence: inversely proportional to MAPE. Less MAPE -> higher confidence.
    if pd.isna(model_mape) or model_mape is None:
        base_confidence = 50  # Neutral confidence if MAPE is not available
    else:
        # Example: 100 - (MAPE * 2). Adjust multiplier as needed for sensitivity.
        # Ensure confidence stays within 0-100.
        base_confidence = min(100, max(0, 100 - (model_mape * 2)))

    confidence_adjustments = 0

    # Apply penalty/boost based on prediction intervals
    if last_close_price is not None and lower_interval is not None and upper_interval is not None:
        # Penalty if any part of the interval is less than the current price
        # This implies a risk of price falling below current levels.
        if lower_interval < last_close_price or upper_interval < last_close_price:
            print(f"    Confidence adjustment: Applying penalty because interval includes prices below current price ({last_close_price:.2f}).")
            confidence_adjustments -= 15 # Example penalty

        # Heavy boost if the lower end of the confidence interval predicts > 7% return
        if last_close_price != 0:
            lower_interval_return = ((lower_interval - last_close_price) / last_close_price) * 100
            if lower_interval_return > 7:
                print(f"    Confidence adjustment: Applying heavy boost as lower interval return ({lower_interval_return:.2f}%) is > 7%.")
                confidence_adjustments += 20 # Example heavy boost

    # Boosting/Reducing confidence with fundamental data (existing logic)
    if not fundamental_data.empty:
        pe_ratio = fundamental_data.get('PE Ratio')
        debt_to_equity = fundamental_data.get('Debt To Equity')

        if pe_ratio is not None and pe_ratio > 50:  # High PE might reduce confidence for high growth predictions
            confidence_adjustments -= 5
        elif pe_ratio is not None and pe_ratio < 10:  # Low PE might increase confidence for value predictions
            confidence_adjustments += 5

        if debt_to_equity is not None and debt_to_equity > 1:  # High D/E might reduce confidence
            confidence_adjustments -= 10
        elif debt_to_equity is not None and debt_to_equity < 0.5:  # Low D/E might increase confidence
            confidence_adjustments += 5

    final_confidence = min(100, max(0, base_confidence + confidence_adjustments))
    return final_confidence

def calculate_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    :param y_true: Actual values.
    :param y_pred: Predicted values.
    :return: MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by replacing 0s in y_true with a small epsilon
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, np.finfo(float).eps))) * 100


if __name__ == "__main__":
    ticker_list_file = 'nse_tickers_screener.csv'
    daily_data_directory = 'daily_data'
    fundamental_data_file = 'all_fundamental_data.csv'
    output_results_file = 'time_series_predictions.xlsx' # Changed to Excel file

    # Load tickers to process (e.g., a subset for demonstration)
    try:
        df_tickers = pd.read_csv(ticker_list_file)
        # Process all tickers
        tickers_to_process = df_tickers['Ticker'].tolist()
        print(f"Processing all {len(tickers_to_process)} tickers for time series analysis.")
    except FileNotFoundError:
        print(f"Error: Ticker list file '{ticker_list_file}' not found.")
        exit()
    except KeyError:
        print(f"Error: '{ticker_list_file}' must contain a 'Ticker' column.")
        exit()

    all_prediction_results = []

    for ticker in tickers_to_process:
        daily_df, fundamental_data = load_and_prepare_data(ticker, daily_data_directory, fundamental_data_file)
        if not daily_df.empty:
            prediction_results = run_time_series_prediction(ticker, daily_df, fundamental_data)
            all_prediction_results.append(prediction_results)
        else:
            print(f"Skipping {ticker} due to missing or empty daily data.")

    # Save results to CSV
    if all_prediction_results:
        final_results_df = pd.DataFrame(all_prediction_results)
        final_results_df.to_excel(output_results_file, index=False) # Save to Excel
        print(f"All prediction results saved to {output_results_file}")
    else:
        print("No prediction results to save.")
