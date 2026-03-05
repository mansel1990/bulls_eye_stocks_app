import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_fundamental_data(ticker_symbol):
    """
    Fetches fundamental data for a given ticker symbol.
    :param ticker_symbol: The stock ticker symbol (e.g., "AAPL").
    :return: A dictionary containing fundamental data.
    """
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    fundamental_data = {}

    # Price to Earnings (PE) Ratio
    fundamental_data['PE Ratio'] = info.get('trailingPE')

    # Price to Book Ratio
    fundamental_data['Price to Book'] = info.get('priceToBook')

    # Median PE last 50 days (Approximation: yfinance does not directly provide median PE for a period)
    # We will fetch historical data and calculate PE if EPS is available for each day.
    # This is a complex calculation and might require more sophisticated financial data APIs.
    # For now, we'll leave it as a placeholder or use a simpler approximation if possible.
    # We'll fetch 50 days of historical data to potentially calculate this later.
    hist = stock.history(period="60d") # Fetching a bit more to be safe for 50 days
    # To calculate Median PE, we'd need daily EPS, which yfinance does not provide in history.
    # This would typically involve (Daily Close Price / TTM EPS). TTM EPS is not daily.
    # For a true "Median PE last 50 days", a more advanced financial data source would be needed.
    fundamental_data['Median PE last 50 days'] = "N/A - Requires daily EPS or specialized API"


    # Sales Growth Quarterly for last 4 quarters and Profit Growth last 4 quarters
    # yfinance provides 'quarterlyFinancials' and 'quarterlyEarnings'
    quarterly_financials = stock.quarterly_financials
    quarterly_earnings = stock.quarterly_earnings

    # Sales Growth (Revenue Growth)
    # Check for 'Total Revenue' in quarterly_financials.index before accessing
    revenue_data = quarterly_financials.loc['Total Revenue'] if (
        not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index
    ) else pd.Series()
    sales_growth = []
    # Ensure we have enough data points to calculate growth for 4 quarters
    if len(revenue_data) >= 5: # Need 5 quarters for 4 growth rates
        for i in range(1, 5): # Get last 4 quarters' growth
            current_q = revenue_data.iloc[i-1]
            previous_q = revenue_data.iloc[i]
            if previous_q != 0:
                growth = ((current_q - previous_q) / previous_q) * 100
                sales_growth.append(growth)
            else:
                sales_growth.append(0) # Handle division by zero
    fundamental_data['Sales Growth Last 4 Quarters'] = sales_growth # Store up to 4 quarters

    # Profit Growth (Net Income Growth)
    # yfinance quarterly earnings often has 'Revenue' and 'Earnings' (Net Income)
    # Addressing DeprecationWarning: 'Ticker.earnings' is deprecated. Use Ticker.income_stmt
    # Check for 'Net Income' in quarterly_income_stmt.index before accessing
    quarterly_income_stmt = stock.quarterly_income_stmt
    earnings_data = quarterly_income_stmt.loc['Net Income'] if (
        not quarterly_income_stmt.empty and 'Net Income' in quarterly_income_stmt.index
    ) else pd.Series()
    profit_growth = []
    # Ensure we have enough data points to calculate growth for 4 quarters
    if len(earnings_data) >= 5: # Need 5 quarters for 4 growth rates
        for i in range(1, 5): # Get last 4 quarters' growth
            current_q = earnings_data.iloc[i-1]
            previous_q = earnings_data.iloc[i]
            if previous_q != 0:
                growth = ((current_q - previous_q) / previous_q) * 100
                profit_growth.append(growth)
            else:
                profit_growth.append(0) # Handle division by zero
    fundamental_data['Profit Growth Last 4 Quarters'] = profit_growth # Store up to 4 quarters

    # Debt To Equity
    fundamental_data['Debt To Equity'] = info.get('debtToEquity')

    # PEG Ratio
    fundamental_data['PEG Ratio'] = info.get('pegRatio')

    # FCF (Free Cash Flow) - yfinance often has 'Free Cash Flow' in cashflow statement
    cash_flow = stock.cashflow
    if 'Free Cash Flow' in cash_flow.index:
        # Taking the latest available FCF
        fundamental_data['FCF'] = cash_flow.loc['Free Cash Flow'].iloc[0]
    else:
        fundamental_data['FCF'] = None

    return fundamental_data

if __name__ == "__main__":
    ticker = "MSFT"  # Example ticker
    data = get_fundamental_data(ticker)
    print(f"Fundamental data for {ticker}:")
    for key, value in data.items():
        print(f"  {key}: {value}")
