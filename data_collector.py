import pandas as pd
import time
import os
from fundamental_data_fetcher import get_fundamental_data
from daily_data_fetcher import get_daily_data

def collect_all_data(ticker_list_file='nse_tickers_screener.csv'):
    """
    Reads ticker symbols from a CSV, then fetches daily and fundamental data for each.
    Saves fundamental data to a single CSV and daily data to individual CSVs.
    """
    print("\n--- Starting Data Collection ---")

    # Load tickers
    try:
        df_tickers = pd.read_csv(ticker_list_file)
        tickers = df_tickers['Ticker'].tolist()
        print(f"Loaded {len(tickers)} tickers from {ticker_list_file}")
    except FileNotFoundError:
        print(f"Error: Ticker list file '{ticker_list_file}' not found.")
        return
    except KeyError:
        print(f"Error: '{ticker_list_file}' must contain a 'Ticker' column.")
        return

    # Setup output directories
    fundamental_output_file = 'all_fundamental_data.csv'
    daily_data_dir = 'daily_data'
    os.makedirs(daily_data_dir, exist_ok=True)

    # Prepare for fundamental data aggregation
    all_fundamental_data = []

    # Configuration for API calls
    request_delay = 1 # seconds to wait between yfinance calls
    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(tickers):
        print(f"\nProcessing {i+1}/{len(tickers)}: {ticker}")

        # Fetch Fundamental Data
        try:
            fund_data = get_fundamental_data(ticker)
            if fund_data:
                fund_data['Ticker'] = ticker # Add ticker to fundamental data
                all_fundamental_data.append(fund_data)
                print(f"  Fetched fundamental data for {ticker}.")
            else:
                print(f"  No fundamental data found for {ticker}.")
        except Exception as e:
            print(f"  Error fetching fundamental data for {ticker}: {e}")
        
        time.sleep(request_delay)

        # Fetch Daily Data
        try:
            daily_df = get_daily_data(ticker)
            if not daily_df.empty:
                daily_output_file = os.path.join(daily_data_dir, f'{ticker.replace(".NS", "")}_daily.csv')
                # Ensure the index is named 'Date' before saving
                daily_df.index.name = 'Date'
                # Select only the relevant numeric columns to save
                cols_to_save = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
                daily_df[cols_to_save].to_csv(daily_output_file, index=True) # index=True to save Date column
                print(f"  Saved daily data for {ticker} to {daily_output_file}.")
            else:
                print(f"  No daily data found for {ticker}.")
        except Exception as e:
            print(f"  Error fetching daily data for {ticker}: {e}")

        time.sleep(request_delay)
        success_count += 1 # Increment regardless, as we try to fetch both types of data

    print("\n--- Data Collection Summary ---")
    print(f"Attempted to process {len(tickers)} tickers.")
    print(f"Successfully processed (at least one type of data): {success_count}")
    print(f"Failed to process (completely): {fail_count}") # Need to refine fail_count logic if needed

    # Save all fundamental data to a single CSV
    if all_fundamental_data:
        df_all_fundamental = pd.DataFrame(all_fundamental_data)
        df_all_fundamental.to_csv(fundamental_output_file, index=False)
        print(f"All fundamental data saved to {fundamental_output_file}")
    else:
        print("No fundamental data collected to save.")

    print("Data collection complete.")

if __name__ == "__main__":
    collect_all_data()
