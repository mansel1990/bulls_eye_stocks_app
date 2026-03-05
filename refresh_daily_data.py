"""
Incremental daily data refresh.
- Source tickers: NSE Symbols (1).csv  (Scrip column)
- Existing CSVs: only fetch delta from (last_date + 1) to today
- New tickers:   full 5-year fetch
- RSI is recomputed on the full merged history for accuracy
"""
import os, sys, time
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

DAILY_DIR = "D:/stocks prediction/daily_data"
SYMBOLS_CSV = "D:/stocks prediction/NSE Symbols (1).csv"

# ── RSI helper ────────────────────────────────────────────────────────────────
def calc_rsi(close, window=14):
    diff  = close.diff(1)
    gain  = diff.clip(lower=0).fillna(0)
    loss  = (-diff).clip(lower=0).fillna(0)
    avg_g = gain.rolling(window, min_periods=1).mean()
    avg_l = loss.rolling(window, min_periods=1).mean()
    rs    = avg_g / avg_l.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))

# ── Flatten yfinance MultiIndex ───────────────────────────────────────────────
def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ── Fetch helper ──────────────────────────────────────────────────────────────
def fetch(ticker, start=None):
    """Return OHLCV DataFrame or empty. start=date for delta fetch."""
    try:
        if start:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             interval="1d", progress=False)
        else:
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
        df = flatten(df)
        if df.empty:
            return pd.DataFrame()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        return df[['Open','High','Low','Close','Volume']]
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return pd.DataFrame()

# ── Load tickers ──────────────────────────────────────────────────────────────
sym_df  = pd.read_csv(SYMBOLS_CSV)
tickers = sorted(sym_df['Scrip'].dropna().str.strip().tolist())
print(f"Tickers in NSE Symbols: {len(tickers)}", flush=True)

today    = date.today()
success  = fail = skipped = 0

for i, scrip in enumerate(tickers):
    ticker  = scrip + ".NS"
    outfile = os.path.join(DAILY_DIR, scrip + "_daily.csv")
    tag     = f"[{i+1}/{len(tickers)}] {ticker}"

    # ── Case 1: existing file → delta only ───────────────────────────────────
    if os.path.exists(outfile):
        try:
            existing = pd.read_csv(outfile, index_col='Date', parse_dates=True)
            last_dt  = existing.index.max().date()
        except Exception:
            existing = pd.DataFrame()
            last_dt  = None

        if last_dt and last_dt >= today:
            print(f"{tag}  up-to-date ({last_dt})", flush=True)
            skipped += 1
            continue

        # fetch from the day after last known bar
        start = (last_dt + timedelta(days=1)) if last_dt else None
        print(f"{tag}  delta from {start}...", end=" ", flush=True)
        new_df = fetch(ticker, start=start)

        if new_df.empty:
            print("no new data.", flush=True)
            skipped += 1
            continue

        # merge and deduplicate
        combined = pd.concat([existing[['Open','High','Low','Close','Volume']], new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    # ── Case 2: new ticker → full 5Y fetch ───────────────────────────────────
    else:
        print(f"{tag}  full 5Y fetch...", end=" ", flush=True)
        combined = fetch(ticker)
        if combined.empty:
            print("no data.", flush=True)
            fail += 1
            continue

    # ── Recompute RSI on full history ─────────────────────────────────────────
    combined['RSI'] = calc_rsi(combined['Close'])
    combined.index.name = 'Date'
    combined[['Open','High','Low','Close','Volume','RSI']].to_csv(outfile)
    print(f"OK  (last: {combined.index[-1].date()}  rows: {len(combined)})", flush=True)
    success += 1
    time.sleep(0.3)

print(f"\nDone.  Updated: {success}  Skipped/up-to-date: {skipped}  Failed: {fail}", flush=True)
