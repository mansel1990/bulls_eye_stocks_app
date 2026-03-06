# Stock Data Collection and Analysis

This project aims to collect both fundamental and daily stock data, and provide an orchestration flow to keep the data up-to-date.

## Project Structure

- `fundamental_data_fetcher.py`: Python program to fetch fundamental financial data for stocks.
- `daily_data_fetcher.py`: Python program to fetch daily Open, High, Low, Close, Volume (OHLCV) data and calculate the Relative Strength Index (RSI).
- `stock_screener.py`: Python program for screening stocks based on various financial criteria. It now includes a function to scrape NSE ticker symbols from Screener.in.
- `data_collector.py`: Python program to fetch and store daily and fundamental data for a list of tickers.
- `time_series_analyzer.py`: Python program for time series prediction, classification, and confidence scoring, currently using the Prophet model.
- `model_comparison.py`: Python program to implement and compare XGBoost and LSTM models against Prophet, focusing on 'stellar' classification precision.
- `final_predictor.py`: Python program to run predictions using the chosen "best" time series model (Prophet, for now) and save results to a CSV.
- `orchestration_flow.md`: This file will describe the proposed orchestration flow to keep the collected data updated.

## Data Points Collected

### Fundamental Data (fundamental_data_fetcher.py)

- Price-to-Earnings (PE) Ratio
- Price-to-Book Ratio
- Median PE for the last 50 days
- Quarterly Sales Growth (last 4 quarters)
- Quarterly Profit Growth (last 4 quarters)
- Debt-to-Equity Ratio
- PEG Ratio
- Free Cash Flow (FCF)

### Daily Data (daily_data_fetcher.py)

- Open Price
- High Price
- Low Price
- Close Price
- Volume
- Relative Strength Index (RSI)
