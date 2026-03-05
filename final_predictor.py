import pandas as pd
import os
from time_series_analyzer import load_and_prepare_data, run_time_series_prediction

def run_final_predictions(ticker_list_file='nse_tickers_screener.csv',
                          daily_data_dir='daily_data',
                          fundamental_data_file='all_fundamental_data.csv',
                          output_csv_file='final_predictions.csv'):
    """
    Loads tickers, runs the best-performing time series model (Prophet, for now),
    and saves the predictions, classifications, and confidence scores to a CSV file.
    """
    print("\n--- Running Final Predictions (Prophet Model) ---")

    try:
        df_tickers = pd.read_csv(ticker_list_file)
        tickers_to_process = df_tickers['Ticker'].tolist()
        print(f"Loaded {len(tickers_to_process)} tickers from {ticker_list_file}")
    except FileNotFoundError:
        print(f"Error: Ticker list file '{ticker_list_file}' not found.")
        return
    except KeyError:
        print(f"Error: '{ticker_list_file}' must contain a 'Ticker' column.")
        return

    all_final_results = []

    for i, ticker in enumerate(tickers_to_process):
        print(f"Processing {i+1}/{len(tickers_to_process)}: {ticker}")
        daily_df, fundamental_data = load_and_prepare_data(ticker, daily_data_dir, fundamental_data_file)
        
        if not daily_df.empty:
            # Run the Prophet prediction from time_series_analyzer
            prediction_results = run_time_series_prediction(ticker, daily_df, fundamental_data)
            all_final_results.append(prediction_results)
        else:
            print(f"Skipping {ticker} due to missing or empty daily data.")
    
    if all_final_results:
        final_results_df = pd.DataFrame(all_final_results)
        final_results_df.to_csv(output_csv_file, index=False)
        print(f"\nAll final predictions saved to {output_csv_file}")
    else:
        print("No final prediction results to save.")

if __name__ == "__main__":
    run_final_predictions()
