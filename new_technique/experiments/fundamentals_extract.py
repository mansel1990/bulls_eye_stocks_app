import os
import time
import datetime
import pandas as pd
import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────────
DAILY_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'daily_data')
OUTPUT_DIR     = os.path.dirname(__file__)
TODAY          = datetime.date.today().strftime('%Y-%m-%d')
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, f'fundamentals_{TODAY}.xlsx')
DELAY          = 0.5  # seconds between requests

# Columns that should be formatted as large integers (crores/billions)
LARGE_INT_COLS = {'Free Cash Flow', 'Market Cap', 'Enterprise Value',
                  'Total Revenue', 'EBITDA', 'Total Cash', 'Total Debt',
                  'Gross Profit', 'Net Income'}

# Column display widths
COL_WIDTHS = {
    # Identity
    'Ticker':               14,
    'Company Name':         28,
    'Sector':               18,
    'Industry':             24,
    # Price & Market
    'Price':                10,
    '52W High':             10,
    '52W Low':              10,
    '52W Change (%)':       14,
    'Beta':                 8,
    'Avg Vol (3M)':         13,
    'Market Cap':           16,
    'Enterprise Value':     16,
    # Valuation
    'P/E Ratio':            10,
    'Forward P/E':          11,
    'PEG Ratio':            10,
    'P/B Ratio':            10,
    'P/S Ratio':            10,
    'EV/EBITDA':            11,
    'EV/Revenue':           11,
    # Profitability
    'Gross Margin (%)':     15,
    'Op. Margin (%)':       14,
    'Net Margin (%)':       13,
    'EBITDA Margin (%)':    16,
    'ROE (%)':              10,
    'ROA (%)':              10,
    'ROIC (%)':             10,
    # Growth
    'Revenue Growth (%)':   17,
    'Earnings Growth (%)':  18,
    'EPS (TTM)':            11,
    'EPS Forward':          11,
    # Balance Sheet
    'Total Revenue':        14,
    'EBITDA':               14,
    'Gross Profit':         13,
    'Net Income':           13,
    'Total Cash':           12,
    'Total Debt':           12,
    'Free Cash Flow':       15,
    'Debt/Equity':          12,
    'Current Ratio':        13,
    'Quick Ratio':          11,
    # Dividends
    'Div. Yield (%)':       14,
    'Div. Rate':            10,
    'Payout Ratio (%)':     15,
    # Analyst
    'Analyst Rating':       14,
    'Target Mean Price':    16,
    'Target High':          11,
    'Target Low':           11,
    'No. of Analysts':      14,
    # Ownership
    'Insiders (%)':         12,
    'Institutions (%)':     16,
}


def pct(val):
    """Convert a decimal ratio to rounded percentage, or None."""
    return round(val * 100, 2) if val is not None else None


def extract_fundamentals(symbol):
    info = yf.Ticker(symbol).info

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        'Ticker':              symbol.replace('.NS', ''),
        'Company Name':        info.get('longName') or info.get('shortName'),
        'Sector':              info.get('sector'),
        'Industry':            info.get('industry'),

        # ── Price & Market ─────────────────────────────────────────────────────
        'Price':               info.get('currentPrice'),
        '52W High':            info.get('fiftyTwoWeekHigh'),
        '52W Low':             info.get('fiftyTwoWeekLow'),
        '52W Change (%)':      pct(info.get('52WeekChange')),
        'Beta':                info.get('beta'),
        'Avg Vol (3M)':        info.get('averageVolume'),
        'Market Cap':          info.get('marketCap'),
        'Enterprise Value':    info.get('enterpriseValue'),

        # ── Valuation ──────────────────────────────────────────────────────────
        'P/E Ratio':           info.get('trailingPE'),
        'Forward P/E':         info.get('forwardPE'),
        'PEG Ratio':           info.get('trailingPegRatio'),   # pegRatio is always None for NSE
        'P/B Ratio':           info.get('priceToBook'),
        'P/S Ratio':           info.get('priceToSalesTrailing12Months'),
        'EV/EBITDA':           info.get('enterpriseToEbitda'),
        'EV/Revenue':          info.get('enterpriseToRevenue'),

        # ── Profitability ──────────────────────────────────────────────────────
        'Gross Margin (%)':    pct(info.get('grossMargins')),
        'Op. Margin (%)':      pct(info.get('operatingMargins')),
        'Net Margin (%)':      pct(info.get('profitMargins')),
        'EBITDA Margin (%)':   pct(info.get('ebitdaMargins')),
        'ROE (%)':             pct(info.get('returnOnEquity')),
        'ROA (%)':             pct(info.get('returnOnAssets')),
        'ROIC (%)':            pct(info.get('returnOnCapital')),

        # ── Growth ─────────────────────────────────────────────────────────────
        'Revenue Growth (%)':  pct(info.get('revenueGrowth')),
        'Earnings Growth (%)': pct(info.get('earningsGrowth')),
        'EPS (TTM)':           info.get('trailingEps'),
        'EPS Forward':         info.get('forwardEps'),

        # ── Balance Sheet / Cash Flow ──────────────────────────────────────────
        'Total Revenue':       info.get('totalRevenue'),
        'EBITDA':              info.get('ebitda'),
        'Gross Profit':        info.get('grossProfits'),
        'Net Income':          info.get('netIncomeToCommon'),
        'Total Cash':          info.get('totalCash'),
        'Total Debt':          info.get('totalDebt'),
        'Free Cash Flow':      info.get('freeCashflow'),
        'Debt/Equity':         info.get('debtToEquity'),
        'Current Ratio':       info.get('currentRatio'),
        'Quick Ratio':         info.get('quickRatio'),

        # ── Dividends ──────────────────────────────────────────────────────────
        'Div. Yield (%)':      pct(info.get('dividendYield')),
        'Div. Rate':           info.get('dividendRate'),
        'Payout Ratio (%)':    pct(info.get('payoutRatio')),

        # ── Analyst ────────────────────────────────────────────────────────────
        'Analyst Rating':      info.get('averageAnalystRating'),
        'Target Mean Price':   info.get('targetMeanPrice'),
        'Target High':         info.get('targetHighPrice'),
        'Target Low':          info.get('targetLowPrice'),
        'No. of Analysts':     info.get('numberOfAnalystOpinions'),

        # ── Ownership ──────────────────────────────────────────────────────────
        'Insiders (%)':        pct(info.get('heldPercentInsiders')),
        'Institutions (%)':    pct(info.get('heldPercentInstitutions')),
    }


def main():
    files   = sorted(f for f in os.listdir(DAILY_DATA_DIR) if f.endswith('_daily.csv'))
    tickers = [f.replace('_daily.csv', '') + '.NS' for f in files]
    total   = len(tickers)
    print(f"Fetching fundamentals for {total} tickers: {OUTPUT_FILE}", flush=True)

    records = []
    success = fail = 0

    for i, symbol in enumerate(tickers):
        print(f"[{i+1}/{total}] {symbol} ... ", end="", flush=True)
        try:
            row = extract_fundamentals(symbol)
            records.append(row)
            print(f"OK  Price={row['Price']}  PE={row['P/E Ratio']}", flush=True)
            success += 1
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            fail += 1
        time.sleep(DELAY)

    df = pd.DataFrame(records)

    # ── Excel output ──────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Fundamentals')

        wb = writer.book
        ws = writer.sheets['Fundamentals']

        fmt_header = wb.add_format({'bold': True, 'bg_color': '#D9E1F2',
                                    'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        fmt_num    = wb.add_format({'num_format': '#,##0.00'})
        fmt_int    = wb.add_format({'num_format': '#,##0'})
        fmt_text   = wb.add_format({})

        text_cols = {'Ticker', 'Company Name', 'Sector', 'Industry', 'Analyst Rating'}

        # Set column widths and number formats
        for col_num, col_name in enumerate(df.columns):
            width = COL_WIDTHS.get(col_name, 12)
            if col_name in text_cols:
                ws.set_column(col_num, col_num, width, fmt_text)
            elif col_name in LARGE_INT_COLS:
                ws.set_column(col_num, col_num, width, fmt_int)
            else:
                ws.set_column(col_num, col_num, width, fmt_num)

        # Re-write header row with header format (overrides set_column default)
        ws.set_row(0, 30, fmt_header)
        for col_num, col_name in enumerate(df.columns):
            ws.write(0, col_num, col_name, fmt_header)

        # Freeze top row & first column (Ticker)
        ws.freeze_panes(1, 1)

        # Auto-filter
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    print(f"\nDone. Success: {success}  Failed: {fail}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Columns: {len(df.columns)}  |  Rows: {len(df)}")


if __name__ == '__main__':
    main()
