import pandas as pd
import numpy as np
import os
from datetime import timedelta

# For Prophet
from prophet import Prophet

# For XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# For LSTM (using TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor

# For metrics and comparison
from sklearn.metrics import mean_absolute_percentage_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary functions from time_series_analyzer
from time_series_analyzer import load_and_prepare_data, classify_prediction, calculate_mape

# Configuration
SEQUENCE_LENGTH = 30 # For LSTM, number of past days to consider for prediction
PREDICTION_PERIOD = 30 # Predict 30 days ahead
TEST_SET_SIZE = 30 # Use last 30 days for testing

def create_sequences(data, sequence_length):
    """
    Creates sequences for LSTM model.
    """
    xs, ys = [], []
    for i in range(len(data) - sequence_length - PREDICTION_PERIOD):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length + PREDICTION_PERIOD -1, 0] # Predict the Close price after PREDICTION_PERIOD days (index 0 for Close)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(units=50, dropout_rate=0.2, n_features=1):
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, n_features))) # Address UserWarning, dynamic features
    model.add(LSTM(units=units, return_sequences=True))  # First LSTM should return sequences
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))  # Last LSTM does not return sequences
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_actual_monthly_return(daily_df, test_start_date, prediction_period):
    """
    Calculates the actual monthly return for the test period.
    :param daily_df: DataFrame with daily data, indexed by date.
    :param test_start_date: The start date of the test period.
    :param prediction_period: The number of days for the prediction (e.g., 30 for one month).
    :return: Actual monthly return percentage.
    """
    if daily_df.empty or 'Close' not in daily_df.columns:
        return np.nan

    # Helper to get price by date, handling potential missing dates and timezones
    def get_price_by_date(df, target_date):
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

    start_price = get_price_by_date(daily_df, test_start_date)

    # Get the closing price at the end of the predicted period (actual future)
    actual_end_date = test_start_date + timedelta(days=prediction_period)
    end_price = get_price_by_date(daily_df, actual_end_date)

    if start_price is None or end_price is None or start_price == 0:
        return np.nan

    return ((end_price - start_price) / start_price) * 100

def run_prophet_model(daily_df):
    """
    Runs the Prophet model for prediction and returns prediction and MAPE.
    """
    prophet_df = daily_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    if pd.api.types.is_datetime64_any_dtype(prophet_df['ds']) and prophet_df['ds'].dt.tz is not None:
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    if len(prophet_df) <= (TEST_SET_SIZE + PREDICTION_PERIOD):
        print("    Not enough data for Prophet train-test split. Skipping Prophet.")
        return None, np.nan, np.nan, np.nan, np.nan

    train_df = prophet_df.iloc[:-(TEST_SET_SIZE + PREDICTION_PERIOD)]
    test_df_for_mape = prophet_df.iloc[-(TEST_SET_SIZE + PREDICTION_PERIOD):-PREDICTION_PERIOD]
    actual_future_for_mape = prophet_df.iloc[-PREDICTION_PERIOD:]

    m = Prophet()
    m.fit(train_df)

    future_test = m.make_future_dataframe(periods=len(test_df_for_mape) + PREDICTION_PERIOD, include_history=False)
    forecast_test = m.predict(future_test)

    # MAPE calculation
    y_true_mape = actual_future_for_mape['y'].values
    y_pred_mape = forecast_test['yhat'].iloc[-PREDICTION_PERIOD:].values
    prophet_mape = calculate_mape(y_true_mape, y_pred_mape)

    # Actual 1-month ahead prediction
    m_full = Prophet()
    m_full.fit(prophet_df.iloc[:-PREDICTION_PERIOD]) # Train on all data except the last month's actual future
    future_pred = m_full.make_future_dataframe(periods=PREDICTION_PERIOD, include_history=False)
    forecast_pred = m_full.predict(future_pred)
    prophet_prediction = forecast_pred['yhat'].iloc[-1]
    prophet_lower_interval = forecast_pred['yhat_lower'].iloc[-1]
    prophet_upper_interval = forecast_pred['yhat_upper'].iloc[-1]

    # Get last close price for calculating monthly return (from before the actual future)
    last_close_for_return = prophet_df['y'].iloc[-(PREDICTION_PERIOD + 1)]
    if last_close_for_return != 0:
        prophet_monthly_return = ((prophet_prediction - last_close_for_return) / last_close_for_return) * 100
    else:
        prophet_monthly_return = np.nan

    return prophet_prediction, prophet_mape, prophet_monthly_return, prophet_lower_interval, prophet_upper_interval

def run_xgboost_model(daily_df):
    """
    Runs the XGBoost model for prediction and returns prediction and MAPE.
    """
    if len(daily_df) <= (SEQUENCE_LENGTH + PREDICTION_PERIOD + TEST_SET_SIZE): # Needs more data for lagged features and split
        print("    Not enough data for XGBoost. Skipping XGBoost.")
        return None, np.nan, np.nan, np.nan, np.nan

    # Feature engineering: lagged features, moving averages, RSI
    df_xgb = daily_df[['Close', 'MA_44', 'RSI']].copy()
    for i in range(1, SEQUENCE_LENGTH + 1):
        df_xgb[f'Close_lag_{i}'] = df_xgb['Close'].shift(i)
        df_xgb[f'MA_44_lag_{i}'] = df_xgb['MA_44'].shift(i)
        df_xgb[f'RSI_lag_{i}'] = df_xgb['RSI'].shift(i)

    # Target: close price after PREDICTION_PERIOD days
    df_xgb['Target'] = df_xgb['Close'].shift(-PREDICTION_PERIOD)
    df_xgb.dropna(inplace=True)

    if df_xgb.empty:
        print("    XGBoost: DataFrame is empty after feature engineering and dropping NaNs. Skipping.")
        return None, np.nan, np.nan, np.nan, np.nan

    X = df_xgb.drop('Target', axis=1)
    y = df_xgb['Target']

    if len(X) <= TEST_SET_SIZE:
        print("    Not enough data for XGBoost train-test split after feature engineering. Skipping.")
        return None, np.nan, np.nan, np.nan, np.nan

    # Split data
    X_train, X_test = X.iloc[:-TEST_SET_SIZE], X.iloc[-TEST_SET_SIZE:]
    y_train, y_test = y.iloc[:-TEST_SET_SIZE], y.iloc[-TEST_SET_SIZE:]

    # Fixed parameters for XGBoost (removed GridSearchCV for faster execution)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # MAPE calculation
    y_pred_mape = model.predict(X_test)
    xgboost_mape = calculate_mape(y_test, y_pred_mape)

    # Confidence interval estimation
    residuals = y_test - y_pred_mape
    std_dev_residuals = np.std(residuals)

    # Actual 1-month ahead prediction
    # We need the last `SEQUENCE_LENGTH` data points to predict the next `PREDICTION_PERIOD` days
    # Create a single row for prediction based on the very last available data
    last_data_point = daily_df[['Close', 'MA_44', 'RSI']].iloc[-1-PREDICTION_PERIOD:] # Get data up to last actual observed point before future to predict
    
    predict_row_data = {}
    for i in range(1, SEQUENCE_LENGTH + 1):
        if len(last_data_point) >= i:
            predict_row_data[f'Close_lag_{i}'] = last_data_point['Close'].iloc[-i]
            predict_row_data[f'MA_44_lag_{i}'] = last_data_point['MA_44'].iloc[-i]
            predict_row_data[f'RSI_lag_{i}'] = last_data_point['RSI'].iloc[-i]
        else:
            # Handle cases where not enough historical data for full sequence
            predict_row_data[f'Close_lag_{i}'] = 0
            predict_row_data[f'MA_44_lag_{i}'] = 0
            predict_row_data[f'RSI_lag_{i}'] = 0

    # Ensure all columns expected by the model are present
    predict_df = pd.DataFrame([predict_row_data], columns=X_train.columns)
    xgboost_prediction = model.predict(predict_df)[0]

    # Calculate confidence intervals for the final prediction
    xgboost_lower_interval = xgboost_prediction - 1.96 * std_dev_residuals
    xgboost_upper_interval = xgboost_prediction + 1.96 * std_dev_residuals

    # Get last close price for calculating monthly return
    last_close_for_return = daily_df['Close'].iloc[-(PREDICTION_PERIOD + 1)] # Last close price before the predicted month
    if last_close_for_return != 0:
        xgboost_monthly_return = ((xgboost_prediction - last_close_for_return) / last_close_for_return) * 100
    else:
        xgboost_monthly_return = np.nan

    return xgboost_prediction, xgboost_mape, xgboost_monthly_return, xgboost_lower_interval, xgboost_upper_interval

def run_lstm_model(daily_df):
    """
    Runs the LSTM model for prediction and returns prediction and MAPE.
    """
    if len(daily_df) <= (SEQUENCE_LENGTH + PREDICTION_PERIOD + TEST_SET_SIZE):
        print("    Not enough data for LSTM. Skipping LSTM.")
        return None, np.nan, np.nan, np.nan, np.nan

    tf.keras.backend.clear_session() # Clear TF backend session for a fresh model

    # Use 'Close', 'MA_44', 'RSI' prices for LSTM
    data_features = daily_df[['Close', 'MA_44', 'RSI']].values
    n_features = data_features.shape[1]

    # Scale features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_features = feature_scaler.fit_transform(data_features)

    # Scale target (Close price) separately
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(daily_df[['Close']].values)

    # Create sequences using scaled features and scaled target
    # The target `y` in create_sequences now comes from scaled_target
    xs, ys = [], []
    for i in range(len(scaled_data_features) - SEQUENCE_LENGTH - PREDICTION_PERIOD):
        x = scaled_data_features[i:(i + SEQUENCE_LENGTH)]
        y = scaled_target[i + SEQUENCE_LENGTH + PREDICTION_PERIOD -1, 0] # Predict the scaled Close price
        xs.append(x)
        ys.append(y)

    X_seq, y_seq = np.array(xs), np.array(ys)

    if len(X_seq) == 0:
        print("    LSTM: No sequences created. Skipping.")
        return None, np.nan, np.nan, np.nan, np.nan

    # Split data
    X_train, X_test = X_seq[:-TEST_SET_SIZE], X_seq[-TEST_SET_SIZE:]
    y_train, y_test = y_seq[:-TEST_SET_SIZE], y_seq[-TEST_SET_SIZE:]

    # Reshape for LSTM input [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

    # Fixed parameters for LSTM (removed GridSearchCV for faster execution)
    model = build_lstm_model(units=50, dropout_rate=0.2, n_features=n_features)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0) # verbose=0 to suppress output

    # MAPE calculation
    y_pred_scaled = model.predict(X_test, verbose=0)
    # Inverse transform predictions and actuals using target_scaler
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_mape = calculate_mape(y_test_original, y_pred)
    print(f"    LSTM: y_test_original (first 5): {y_test_original[:5]}, y_pred (first 5): {y_pred[:5]}")
    print(f"    LSTM: MAPE calculation inputs: y_test_original contains nan: {np.any(np.isnan(y_test_original))}, y_pred contains nan: {np.any(np.isnan(y_pred))}")

    # Confidence interval estimation
    residuals = y_test_original - y_pred
    std_dev_residuals = np.std(residuals)

    # Actual 1-month ahead prediction
    # Need the last `SEQUENCE_LENGTH` actual values to predict the next one
    last_sequence_scaled = scaled_data_features[-(SEQUENCE_LENGTH + PREDICTION_PERIOD):-PREDICTION_PERIOD].reshape(1, SEQUENCE_LENGTH, n_features)
    lstm_prediction_scaled = model.predict(last_sequence_scaled, verbose=0)[0, 0]
    lstm_prediction = target_scaler.inverse_transform([[lstm_prediction_scaled]])[0, 0]

    # Calculate confidence intervals for the final prediction
    lstm_lower_interval = lstm_prediction - 1.96 * std_dev_residuals
    lstm_upper_interval = lstm_prediction + 1.96 * std_dev_residuals

    # Get last close price for calculating monthly return
    last_close_for_return = daily_df['Close'].iloc[-(PREDICTION_PERIOD + 1)] # Last close price before the predicted month
    if last_close_for_return != 0:
        lstm_monthly_return = ((lstm_prediction - last_close_for_return) / last_close_for_return) * 100
    else:
        lstm_monthly_return = np.nan

    return lstm_prediction, lstm_mape, lstm_monthly_return, lstm_lower_interval, lstm_upper_interval

def compare_models(ticker_list_file='nse_tickers_screener.csv',
                   daily_data_dir='daily_data',
                   fundamental_data_file='all_fundamental_data.csv',
                   output_comparison_file='model_comparison_results.csv'):
    """
    Compares Prophet, XGBoost, and LSTM models for a list of tickers.
    Determines the best model based on stellar classification precision.
    """
    print("\n--- Starting Model Comparison ---")

    try:
        df_tickers = pd.read_csv(ticker_list_file)
        tickers_to_process = df_tickers['Ticker'].tolist()
        # For faster comparison, process a subset of tickers.
        # Remove this slicing ([:50]) to process all tickers if resources allow.
        tickers_to_process = tickers_to_process[:25] 
        print(f"Loaded {len(tickers_to_process)} tickers from {ticker_list_file}")
    except FileNotFoundError:
        print(f"Error: Ticker list file '{ticker_list_file}' not found.")
        return
    except KeyError:
        print(f"Error: '{ticker_list_file}' must contain a 'Ticker' column.")
        return

    comparison_results = []

    # Accumulators for overall metrics
    overall_prophet_mape = []
    overall_xgboost_mape = []
    overall_lstm_mape = []
    overall_prophet_ci_width = []
    overall_xgboost_ci_width = []
    overall_lstm_ci_width = []

    # Accumulators for confusion matrix for 'stellar' class
    # prophet_true_stellar = 0
    # prophet_predicted_stellar = 0
    # prophet_correct_stellar = 0

    # xgboost_true_stellar = 0
    # xgboost_predicted_stellar = 0
    # xgboost_correct_stellar = 0

    # lstm_true_stellar = 0
    # lstm_predicted_stellar = 0
    # lstm_correct_stellar = 0

    for i, ticker in enumerate(tickers_to_process):
        print(f"\nProcessing {i+1}/{len(tickers_to_process)}: {ticker}")
        daily_df, _ = load_and_prepare_data(ticker, daily_data_dir, fundamental_data_file) # Fundamental data not used for direct model comparison here

        if daily_df.empty or 'Close' not in daily_df.columns:
            print(f"  Skipping {ticker} due to missing or empty daily data.")
            continue

        # Ensure enough data for test set and prediction period
        if len(daily_df) < (TEST_SET_SIZE + PREDICTION_PERIOD + SEQUENCE_LENGTH): # Use largest requirement
            print(f"  Skipping {ticker} due to insufficient data for comparison (needed at least {TEST_SET_SIZE + PREDICTION_PERIOD + SEQUENCE_LENGTH} days).")
            continue

        # Determine actual classification for the future period
        # Calculate actual return from the last day of the training set to the end of the prediction period
        # Last day of training set is before TEST_SET_SIZE + PREDICTION_PERIOD days from end of daily_df
        test_period_start_date_for_actual_return = daily_df.index[-(TEST_SET_SIZE + PREDICTION_PERIOD + 1)] # The day before the actual future starts
        actual_monthly_return = calculate_actual_monthly_return(daily_df, test_period_start_date_for_actual_return, PREDICTION_PERIOD)
        actual_class = classify_prediction(actual_monthly_return)
        print(f"  Actual Monthly Return: {actual_monthly_return:.2f}%, Actual Class: {actual_class}")

        current_ticker_results = {'ticker': ticker, 'actual_monthly_return': actual_monthly_return, 'actual_class': actual_class}

        # --- Prophet Model ---
        prophet_pred, prophet_mape, prophet_monthly_return, prophet_lower_interval, prophet_upper_interval = run_prophet_model(daily_df)
        if prophet_pred is not None:
            prophet_class = classify_prediction(prophet_monthly_return)
            prophet_ci_width = prophet_upper_interval - prophet_lower_interval
            current_ticker_results.update({
                'prophet_prediction': prophet_pred,
                'prophet_mape': prophet_mape,
                'prophet_monthly_return': prophet_monthly_return,
                'prophet_class': prophet_class,
                'prophet_lower_interval': prophet_lower_interval,
                'prophet_upper_interval': prophet_upper_interval,
                'prophet_ci_width': prophet_ci_width
            })
            if not np.isnan(prophet_mape): overall_prophet_mape.append(prophet_mape)
            if not np.isnan(prophet_ci_width): overall_prophet_ci_width.append(prophet_ci_width)

            # # Update confusion matrix accumulators for Prophet
            # if actual_class == 'stellar':
            #     prophet_true_stellar += 1
            # if prophet_class == 'stellar':
            #     prophet_predicted_stellar += 1
            # if actual_class == 'stellar' and prophet_class == 'stellar':
            #     prophet_correct_stellar += 1

        # --- XGBoost Model ---
        xgboost_pred, xgboost_mape, xgboost_monthly_return, xgboost_lower_interval, xgboost_upper_interval = run_xgboost_model(daily_df)
        if xgboost_pred is not None:
            xgboost_class = classify_prediction(xgboost_monthly_return)
            xgboost_ci_width = xgboost_upper_interval - xgboost_lower_interval
            current_ticker_results.update({
                'xgboost_prediction': xgboost_pred,
                'xgboost_mape': xgboost_mape,
                'xgboost_monthly_return': xgboost_monthly_return,
                'xgboost_class': xgboost_class,
                'xgboost_lower_interval': xgboost_lower_interval,
                'xgboost_upper_interval': xgboost_upper_interval,
                'xgboost_ci_width': xgboost_ci_width
            })
            if not np.isnan(xgboost_mape): overall_xgboost_mape.append(xgboost_mape)
            if not np.isnan(xgboost_ci_width): overall_xgboost_ci_width.append(xgboost_ci_width)

            # # Update confusion matrix accumulators for XGBoost
            # if actual_class == 'stellar':
            #     xgboost_true_stellar += 1
            # if xgboost_class == 'stellar':
            #     xgboost_predicted_stellar += 1
            # if actual_class == 'stellar' and xgboost_class == 'stellar':
            #     xgboost_correct_stellar += 1

        # --- LSTM Model ---
        lstm_pred, lstm_mape, lstm_monthly_return, lstm_lower_interval, lstm_upper_interval = run_lstm_model(daily_df)
        if lstm_pred is not None:
            lstm_class = classify_prediction(lstm_monthly_return)
            lstm_ci_width = lstm_upper_interval - lstm_lower_interval
            current_ticker_results.update({
                'lstm_prediction': lstm_pred,
                'lstm_mape': lstm_mape,
                'lstm_monthly_return': lstm_monthly_return,
                'lstm_class': lstm_class,
                'lstm_lower_interval': lstm_lower_interval,
                'lstm_upper_interval': lstm_upper_interval,
                'lstm_ci_width': lstm_ci_width
            })
            if not np.isnan(lstm_mape): overall_lstm_mape.append(lstm_mape)
            if not np.isnan(lstm_ci_width): overall_lstm_ci_width.append(lstm_ci_width)

            # # Update confusion matrix accumulators for LSTM
            # if actual_class == 'stellar':
            #     lstm_true_stellar += 1
            # if lstm_class == 'stellar':
            #     lstm_predicted_stellar += 1
            # if actual_class == 'stellar' and lstm_class == 'stellar':
            #     lstm_correct_stellar += 1

        comparison_results.append(current_ticker_results)
    
    # # Calculate precision for 'stellar' class for each model
    # prophet_precision_stellar = prophet_correct_stellar / prophet_predicted_stellar if prophet_predicted_stellar > 0 else 0
    # xgboost_precision_stellar = xgboost_correct_stellar / xgboost_predicted_stellar if xgboost_predicted_stellar > 0 else 0
    # lstm_precision_stellar = lstm_correct_stellar / lstm_predicted_stellar if lstm_predicted_stellar > 0 else 0

    # print("\n--- Model Comparison Summary (Precision for 'Stellar' Class) ---")
    # print(f"Prophet Stellar Precision: {prophet_precision_stellar:.2f}")
    # print(f"XGBoost Stellar Precision: {xgboost_precision_stellar:.2f}")
    # print(f"LSTM Stellar Precision: {lstm_precision_stellar:.2f}")

    # Calculate average MAPE and CI width
    avg_prophet_mape = np.mean(overall_prophet_mape) if overall_prophet_mape else np.nan
    avg_xgboost_mape = np.mean(overall_xgboost_mape) if overall_xgboost_mape else np.nan
    avg_lstm_mape = np.mean(overall_lstm_mape) if overall_lstm_mape else np.nan

    avg_prophet_ci_width = np.mean(overall_prophet_ci_width) if overall_prophet_ci_width else np.nan
    avg_xgboost_ci_width = np.mean(overall_xgboost_ci_width) if overall_xgboost_ci_width else np.nan
    avg_lstm_ci_width = np.mean(overall_lstm_ci_width) if overall_lstm_ci_width else np.nan

    print("\n--- Model Comparison Summary ---")
    print(f"Average Prophet MAPE: {avg_prophet_mape:.2f}, Average CI Width: {avg_prophet_ci_width:.2f}")
    print(f"Average XGBoost MAPE: {avg_xgboost_mape:.2f}, Average CI Width: {avg_xgboost_ci_width:.2f}")
    print(f"Average LSTM MAPE: {avg_lstm_mape:.2f}, Average CI Width: {avg_lstm_ci_width:.2f}")

    # Determine the best model based on least MAPE and smallest CI width
    best_model_mape = "None"
    min_mape = np.inf

    if not np.isnan(avg_prophet_mape) and avg_prophet_mape < min_mape:
        min_mape = avg_prophet_mape
        best_model_mape = "Prophet"
    if not np.isnan(avg_xgboost_mape) and avg_xgboost_mape < min_mape:
        min_mape = avg_xgboost_mape
        best_model_mape = "XGBoost"
    if not np.isnan(avg_lstm_mape) and avg_lstm_mape < min_mape:
        min_mape = avg_lstm_mape
        best_model_mape = "LSTM"

    best_model_ci = "None"
    min_ci_width = np.inf

    if not np.isnan(avg_prophet_ci_width) and avg_prophet_ci_width < min_ci_width:
        min_ci_width = avg_prophet_ci_width
        best_model_ci = "Prophet"
    if not np.isnan(avg_xgboost_ci_width) and avg_xgboost_ci_width < min_ci_width:
        min_ci_width = avg_xgboost_ci_width
        best_model_ci = "XGBoost"
    if not np.isnan(avg_lstm_ci_width) and avg_lstm_ci_width < min_ci_width:
        min_ci_width = avg_lstm_ci_width
        best_model_ci = "LSTM"
    
    print(f"\nBest Model based on Least MAPE: {best_model_mape} (Average MAPE: {min_mape:.2f})")
    print(f"Best Model based on Smallest Confidence Interval Width: {best_model_ci} (Average CI Width: {min_ci_width:.2f})")

    # Overall Best Model (Heuristic: prioritize lower MAPE, then smaller CI)
    overall_best_model = "None"
    if best_model_mape == best_model_ci and best_model_mape != "None":
        overall_best_model = best_model_mape
    elif best_model_mape != "None" and best_model_ci != "None":
        # If MAPE winner has a reasonable CI, pick it. Otherwise, consider CI winner.
        # This is a simple heuristic; can be made more sophisticated.
        if avg_prophet_mape == min_mape and not np.isnan(avg_prophet_ci_width):
            overall_best_model = "Prophet"
        elif avg_xgboost_mape == min_mape and not np.isnan(avg_xgboost_ci_width):
            overall_best_model = "XGBoost"
        elif avg_lstm_mape == min_mape and not np.isnan(avg_lstm_ci_width):
            overall_best_model = "LSTM"
        else:
             overall_best_model = "No clear overall best based on current heuristics"
    else:
        overall_best_model = "No sufficient data for overall best model selection"

    print(f"\nOverall Best Model: {overall_best_model}")

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(output_comparison_file, index=False)
        print(f"All model comparison results saved to {output_comparison_file}")

if __name__ == "__main__":
    # Install required libraries if not already installed:
    # pip install prophet xgboost tensorflow scikit-learn

    compare_models()
