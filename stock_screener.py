import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time

def extract_ticker_from_url(company_url):
    """
    Extracts a potential ticker symbol from a Screener.in company URL.
    Prioritizes alphanumeric symbols over purely numeric internal IDs.
    """
    match = re.search(r'/company/([A-Z0-9]+)/', company_url)
    if match:
        potential_symbol = match.group(1)
        # Check if the potential symbol is purely numeric, if so, it's likely an internal ID.
        if not potential_symbol.isdigit():
            return potential_symbol
    return None

def scrape_screener_tickers():
    """
    Scrapes a list of NSE ticker symbols from screener.in, handling pagination.
    Note: Web scraping can be fragile and break if the website's structure changes.
    :return: A list of NSE ticker symbols, formatted for yfinance (e.g., "RELIANCE.NS").
    """
    base_url = "https://www.screener.in/screens/1250608/nse-1000-companies/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    all_nse_tickers = []
    max_pages = 46  # Based on the information from screener.in (1136 results / 25 per page = ~46 pages)
    request_delay = 2 # seconds to wait between requests

    print("\n--- Scraping NSE Ticker Symbols from Screener.in ---")

    for page_num in range(1, max_pages + 1):
        current_page_url = f"{base_url}?page={page_num}"
        print(f"Fetching page: {page_num}/{max_pages} - {current_page_url}")
        try:
            response = requests.get(current_page_url, headers=headers)
            response.raise_for_status() # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.text, 'html.parser')

            table = soup.find('table')
            if not table:
                print(f"Could not find the main table on page {page_num}. Stopping.")
                break

            rows = table.find_all('tr')
            if len(rows) > 1:
                for row in rows[1:]: # Skip header row
                    cols = row.find_all('td')
                    if cols:
                        name_cell = cols[1] # Assuming 'Name' is the second column (index 1)
                        name_tag = name_cell.find('a')
                        if name_tag and 'href' in name_tag.attrs:
                            company_url_path = name_tag['href']
                            ticker_symbol = extract_ticker_from_url(company_url_path)
                            if ticker_symbol:
                                all_nse_tickers.append(f"{ticker_symbol}.NS")
            else:
                print(f"No data rows found in the table on page {page_num}.")
                # If a page has no data rows, it might be the end, or an error.
                # We will continue for now, but this could be a point to break if consistently empty.

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {current_page_url}: {e}")
            break # Stop on error
        except Exception as e:
            print(f"An error occurred during parsing page {current_page_url}: {e}")
            break # Stop on error
        
        # Introduce a delay between requests to avoid hitting rate limits
        time.sleep(request_delay)

    # Remove duplicates if any (due to consolidation or other reasons)
    all_nse_tickers = list(set(all_nse_tickers))
    print(f"Successfully scraped {len(all_nse_tickers)} unique NSE ticker symbols from Screener.in.")
    return all_nse_tickers

def screen_stocks():
    """
    Screens stocks based on specified criteria.

    Market Capitalization > 500 (Assuming in millions, so > 500,000,000)
    Debt to equity < 1
    Sales growth 3years > 10%
    Profit growth 3Years > 10%
    Return on equity > 10%

    :return: A list of ticker symbols that meet the criteria.
    """
    print("\n--- Stock Screener ---")
    
    # Get NSE tickers by scraping Screener.in
    nse_tickers = scrape_screener_tickers()

    if not nse_tickers:
        print("No NSE tickers found to screen. Please check the scraping function.")
        return []

    # The following logic for screening individual stocks needs a robust data source.
    # yfinance is not ideal for comprehensive screening based on multi-year growth rates
    # and broad market queries for Indian stocks.
    # For demonstration, we'll simulate some filtering with hypothetical data.
    
    # Placeholder for fetching data from a screening API or iterating through tickers
    # For each ticker, you would fetch the required fundamental data and apply the criteria.
    filtered_tickers = []
    # Example of how you *might* integrate with fundamental data fetching (conceptual):
    # from fundamental_data_fetcher import get_fundamental_data
    # for ticker in nse_tickers:
    #     try:
    #         fundamental_data = get_fundamental_data(ticker) # This will work for some symbols with .NS
    #         # Apply your screening logic here using fundamental_data
    #         market_cap = fundamental_data.get('Market Cap', 0) # Placeholder
    #         debt_to_equity = fundamental_data.get('Debt To Equity', 100)
    #         sales_growth_3yr = fundamental_data.get('Sales Growth 3Y', 0) # Placeholder
    #         profit_growth_3yr = fundamental_data.get('Profit Growth 3Y', 0) # Placeholder
    #         roe = fundamental_data.get('Return On Equity', 0) # Placeholder
    #
    #         if (
    #             market_cap > 500_000_000 and
    #             debt_to_equity < 1 and
    #             sales_growth_3yr > 0.10 and
    #             profit_growth_3yr > 0.10 and
    #             roe > 0.10
    #         ):
    #             filtered_tickers.append(ticker)
    #     except Exception as e:
    #         print(f"Could not fetch data for {ticker}: {e}")

    # For this exercise, we will return a placeholder list or a subset of scraped tickers.
    # In a real application, you would replace this with actual data fetching and filtering.
    # Returning the full scraped list here, which can then be saved to CSV.
    return nse_tickers

if __name__ == "__main__":
    # Example usage
    all_nse_tickers = scrape_screener_tickers() # Call the scraping function directly
    if all_nse_tickers:
        output_filename = 'nse_tickers_screener.csv'
        df_tickers = pd.DataFrame({'Ticker': all_nse_tickers})
        df_tickers.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved {len(all_nse_tickers)} NSE ticker symbols to {output_filename}")
    else:
        print("\nNo NSE tickers found to save to CSV.")

    # The screening part is still conceptual and needs robust data fetching
    # filtered_tickers = screen_stocks() # This would use the full list if enabled for actual screening
