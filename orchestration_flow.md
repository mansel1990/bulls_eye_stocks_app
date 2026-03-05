# Data Orchestration Flow

To keep the fundamental and daily stock data up-to-date, an orchestration mechanism is required. This document outlines a recommended approach, starting with a simple solution and suggesting more robust alternatives for production environments.

## 1. Scheduling Data Collection

Both `fundamental_data_fetcher.py` and `daily_data_fetcher.py` need to be executed periodically.

### Daily Data (`daily_data_fetcher.py`)

-   **Frequency:** Daily, after market close (e.g., 5 PM EST).
-   **Reasoning:** Daily OHLCV and RSI are only complete after the trading day ends.

### Fundamental Data (`fundamental_data_fetcher.py`)

-   **Frequency:** Weekly or Monthly (e.g., every Sunday night or first day of the month).
-   **Reasoning:** Fundamental data does not change as frequently as daily prices. A weekly or monthly update should suffice to capture quarterly earnings releases, financial statement updates, and other material changes.

## 2. Data Storage

For persistent storage and ease of analysis, a database is recommended. Options include:

-   **SQLite:** Simple, file-based database, good for small to medium-sized projects.
-   **PostgreSQL/MySQL:** More robust relational databases, suitable for larger datasets and production environments.

Alternatively, for simpler setups, data can be stored in structured files:

-   **CSV files:** Easy to read and write, but less efficient for querying and managing large volumes of historical data.
-   **Parquet/HDF5:** More efficient binary formats for numerical data, especially good with Pandas DataFrames.

Each data point (fundamental, daily) should be stored with a timestamp and ticker symbol to maintain historical records.

## 3. Error Handling and Logging

Robust error handling and logging are crucial for data pipelines.

-   **In-script Error Handling:** Implement `try-except` blocks within the Python scripts to catch API errors, network issues, or data parsing problems.
-   **Logging:** Use Python's `logging` module to record success, failure, and any warnings during data fetching. This helps in debugging and monitoring.
-   **Retries:** Implement retry mechanisms for transient errors (e.g., API rate limits or temporary network outages).

## 4. Scalability Considerations

When dealing with a large number of stock tickers, consider the following:

-   **Batch Processing:** Fetch data for multiple tickers in batches rather than one by one, especially for fundamental data.
-   **Asynchronous Requests:** For APIs that support it, use asynchronous HTTP requests to speed up data fetching.
-   **API Rate Limits:** Be mindful of API rate limits of `yfinance` or any other financial data provider. Introduce delays or use an API key if available for higher limits.

## 5. Orchestration Tools

### Simple Approach: Cron Jobs (Linux/macOS) or Task Scheduler (Windows)

For a basic setup, cron jobs (on Unix-like systems) or Task Scheduler (on Windows) can be used to schedule the Python scripts.

**Example Cron Job Entry (for daily data at 5 PM EST - adjust timezone as needed):**

```cron
0 22 * * * /usr/bin/python3 /path/to/your/project/daily_data_fetcher.py >> /path/to/your/logs/daily_data.log 2>&1
```

**Example Cron Job Entry (for weekly fundamental data on Sunday at 1 AM EST):**

```cron
0 6 * * SUN /usr/bin/python3 /path/to/your/project/fundamental_data_fetcher.py >> /path/to/your/logs/fundamental_data.log 2>&1
```

*Note: Replace `/path/to/your/project/` and `/path/to/your/logs/` with your actual directory paths.*

### More Robust Solutions (for Production)

For more complex workflows, dependencies, monitoring, and error handling, consider dedicated orchestration tools:

-   **Apache Airflow:** A platform to programmatically author, schedule, and monitor workflows. Ideal for complex ETL pipelines.
-   **Prefect/Dagster:** Modern data orchestration platforms that offer more Python-native ways to build and manage data pipelines.
-   **Cloud-based Solutions (AWS Lambda, Google Cloud Functions, Azure Functions with Event Triggers):** For serverless execution, these services can be triggered on a schedule (e.g., AWS EventBridge, Google Cloud Scheduler) and execute the Python scripts without managing servers.

By implementing a chosen orchestration strategy, the collected stock data can be kept consistently updated for further analysis and model building.
