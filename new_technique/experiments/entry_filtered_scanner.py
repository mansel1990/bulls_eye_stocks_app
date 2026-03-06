"""
Entry-Filtered Candlestick Scanner
====================================
Each pattern has a specific entry trigger rule (derived from execution plan logic).
After the signal bar fires, we look forward up to MAX_ENTRY_BARS to see if price
actually reaches the entry trigger level.

If triggered  -> signal is CONFIRMED; forward return measured from entry price
If not triggered -> signal is SKIPPED (never counted)

Entry rules per pattern:
  Morning Star / Morning Doji Star -> Buy Limit at midpoint of signal bar range
                                      (Low + 50% of High-Low range)
                                      Triggered if any of the next N bars' Low <= entry_price
                                      (price pulled back to our limit)

  Bullish Kicker                   -> Market Buy at Next Open
                                      Always triggered (first bar after signal)
                                      Entry = next bar Open

  Bullish Engulfing / Tweezer Bottom / Piercing Line / Hammer / On Neck
                                   -> Buy Stop at signal High + 0.02
                                      Triggered if any of next N bars' High >= entry_price

  Bullish Harami                   -> Buy Stop at Mother candle (i-1) High + 0.02
                                      Triggered if any of next N bars' High >= entry_price

  Inverted Hammer                  -> Buy Stop at signal High + 0.01
                                      Triggered if any of next N bars' High >= entry_price

  Three White Soldiers / Rising Three Methods / Dragonfly Doji / Bullish Harami Cross
                                   -> Buy Stop at signal High + 0.02
                                      Triggered if any of next N bars' High >= entry_price

Forward return horizons:
  1D:
    market_next_open: entry bar IS next bar → Open-to-Close of entry_bar
    stop / limit:     entry fills intraday on entry_bar → realistic hold is
                      NEXT bar: Open[entry_bar+1] to Close[entry_bar+1]
  3D: Close 3 bars after entry vs actual entry price
  5D: Close 5 bars after entry vs actual entry price

Stop-loss is recorded but NOT used to cut returns (we track raw forward return).
It is included as a column for reference.

Output: sheets/entry_filtered_signals_<YEAR>_<date>.xlsx
        Two sheets: one per year (2024 and 2025)
        Each sheet also has a summary tab with profit rates by pattern.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import datetime
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(__file__)
DAILY_DIR     = os.path.join(BASE, '..', '..', 'daily_data')
SHEETS_DIR    = os.path.join(BASE, 'sheets')
os.makedirs(SHEETS_DIR, exist_ok=True)

TODAY     = datetime.date.today()
TODAY_STR = TODAY.strftime('%Y-%m-%d')
OUT       = os.path.join(SHEETS_DIR, f'entry_filtered_signals_v2_{TODAY_STR}.xlsx')

# Scan the full 2024 + 2025 window
SCAN_START = pd.Timestamp('2024-01-01')
SCAN_END   = pd.Timestamp('2025-12-31')

# How many bars forward to wait for entry to trigger
MAX_ENTRY_BARS = 3   # look up to 3 bars ahead for the entry trigger

# ── Candle helpers ────────────────────────────────────────────────────────────
def body(o, c):          return abs(c - o)
def upper_wick(o, c, h): return h - max(o, c)
def lower_wick(o, c, l): return min(o, c) - l
def full_range(h, l):    return h - l
def is_bull(o, c):       return c > o
def is_bear(o, c):       return c < o
def avg_body(df, idx, n=5):
    start = max(0, idx - n)
    return df['body'].iloc[start:idx].mean()

# ── Pattern detectors ────────────────────────────────────────────────────────
# Return (pattern_name, strength, entry_type, entry_price, sl_price) or None
# entry_type: 'limit' | 'stop' | 'market_next_open'
# entry_price: the trigger level
# sl_price:    stop-loss reference price

def hammer(df, i):
    o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    b   = body(o, c)
    lw  = lower_wick(o, c, l)
    uw  = upper_wick(o, c, h)
    rng = full_range(h, l)
    if rng == 0: return None
    if lw >= 2 * b and uw <= 0.1 * rng:
        entry = h + 0.02      # breakout above signal high
        sl    = l
        return ('Hammer', 'Strongly Bullish', 'stop', entry, sl)
    return None

def inverted_hammer(df, i):
    if i < 1: return None
    o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    b  = body(o, c)
    uw = upper_wick(o, c, h)
    lw = lower_wick(o, c, l)
    rng = full_range(h, l)
    if rng == 0: return None
    prev_bear = is_bear(df['Open'].iloc[i-1], df['Close'].iloc[i-1])
    if prev_bear and uw >= 2 * b and lw <= 0.1 * rng:
        entry = h + 0.01
        sl    = l
        return ('Inverted Hammer', 'Bullish', 'stop', entry, sl)
    return None

def bullish_engulfing(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    h2     = df['High'].iloc[i]
    l2     = df['Low'].iloc[i]
    if is_bear(o1, c1) and is_bull(o2, c2):
        if o2 <= c1 and c2 >= o1:
            entry = h2 + 0.02
            sl    = l2
            return ('Bullish Engulfing', 'Strongly Bullish', 'stop', entry, sl)
    return None

def piercing_line(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    h2, l2 = df['High'].iloc[i], df['Low'].iloc[i]
    mid1   = (o1 + c1) / 2
    if is_bear(o1, c1) and is_bull(o2, c2):
        if o2 < c1 and c2 > mid1 and c2 < o1:
            entry = h2 + 0.02
            sl    = l2
            return ('Piercing Line', 'Bullish', 'stop', entry, sl)
    return None

def morning_star(df, i):
    if i < 2: return None
    o1, c1 = df['Open'].iloc[i-2], df['Close'].iloc[i-2]
    o2, c2 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o3, c3 = df['Open'].iloc[i],   df['Close'].iloc[i]
    h3, l3 = df['High'].iloc[i], df['Low'].iloc[i]
    b2  = body(o2, c2)
    ab  = avg_body(df, i-1)
    if ab == 0: return None
    if is_bear(o1, c1) and b2 < 0.3 * ab and is_bull(o3, c3):
        if c3 > (o1 + c1) / 2:
            # Pullback limit: 50% of signal bar range
            entry = l3 + (h3 - l3) * 0.50
            sl    = l3
            return ('Morning Star', 'Strongly Bullish', 'limit', entry, sl)
    return None

def morning_doji_star(df, i):
    if i < 2: return None
    o1, c1 = df['Open'].iloc[i-2], df['Close'].iloc[i-2]
    o2, c2 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o3, c3 = df['Open'].iloc[i],   df['Close'].iloc[i]
    h3, l3 = df['High'].iloc[i],   df['Low'].iloc[i]
    rng2   = full_range(df['High'].iloc[i-1], df['Low'].iloc[i-1])
    b2     = body(o2, c2)
    is_doji = rng2 > 0 and b2 / rng2 < 0.1
    if is_bear(o1, c1) and is_doji and is_bull(o3, c3):
        entry = l3 + (h3 - l3) * 0.50
        sl    = l3
        return ('Morning Doji Star', 'Strongly Bullish', 'limit', entry, sl)
    return None

def three_white_soldiers(df, i):
    if i < 2: return None
    rows = [(df['Open'].iloc[i-2+k], df['Close'].iloc[i-2+k],
             df['High'].iloc[i-2+k], df['Low'].iloc[i-2+k]) for k in range(3)]
    for o, c, h, l in rows:
        if not is_bull(o, c): return None
        uw = upper_wick(o, c, h)
        b  = body(o, c)
        if b == 0 or uw > 0.2 * b: return None
    for k in range(1, 3):
        o_prev, c_prev = rows[k-1][0], rows[k-1][1]
        o_cur          = rows[k][0]
        if not (o_prev <= o_cur <= c_prev): return None
        if rows[k][1] <= c_prev: return None
    h3, l3 = rows[2][2], rows[2][3]
    entry = h3 + 0.02
    sl    = l3
    return ('Three White Soldiers', 'Strongly Bullish', 'stop', entry, sl)

def bullish_harami(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    h1, l2 = df['High'].iloc[i-1], df['Low'].iloc[i]
    if is_bear(o1, c1) and is_bull(o2, c2):
        if o2 > c1 and c2 < o1:
            # Buy stop above mother candle high
            entry = h1 + 0.02
            sl    = l2
            return ('Bullish Harami', 'Bullish', 'stop', entry, sl)
    return None

def dragonfly_doji(df, i):
    o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    rng = full_range(h, l)
    if rng == 0: return None
    b  = body(o, c)
    lw = lower_wick(o, c, l)
    uw = upper_wick(o, c, h)
    if b / rng < 0.1 and lw > 0.6 * rng and uw < 0.05 * rng:
        entry = h + 0.02
        sl    = l
        return ('Dragonfly Doji', 'Bullish', 'stop', entry, sl)
    return None

def tweezer_bottom(df, i):
    if i < 1: return None
    l1 = df['Low'].iloc[i-1]
    l2 = df['Low'].iloc[i]
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    h2     = df['High'].iloc[i]
    tol = 0.002 * ((l1 + l2) / 2)
    if abs(l1 - l2) <= tol and is_bear(o1, c1) and is_bull(o2, c2):
        entry = h2 + 0.02
        sl    = min(l1, l2)
        return ('Tweezer Bottom', 'Bullish', 'stop', entry, sl)
    return None

def on_neck(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    l1     = df['Low'].iloc[i-1]
    l2, h2 = df['Low'].iloc[i], df['High'].iloc[i]
    if is_bear(o1, c1) and is_bull(o2, c2):
        tol = 0.005 * c1
        if o2 < l1 and abs(c2 - c1) <= tol:
            entry = h2 + 0.02
            sl    = l2
            return ('On Neck (Bullish Setup)', 'Bullish', 'stop', entry, sl)
    return None

def rising_three_methods(df, i):
    if i < 4: return None
    o0, c0 = df['Open'].iloc[i-4], df['Close'].iloc[i-4]
    o4, c4 = df['Open'].iloc[i],   df['Close'].iloc[i]
    if not is_bull(o0, c0) or not is_bull(o4, c4): return None
    if c4 <= c0: return None
    for k in range(1, 4):
        ok, ck = df['Open'].iloc[i-4+k], df['Close'].iloc[i-4+k]
        if not is_bear(ok, ck): return None
        if ok > c0 or ck < df['Low'].iloc[i-4]: return None
    h4, l4 = df['High'].iloc[i], df['Low'].iloc[i]
    entry = h4 + 0.02
    sl    = l4
    return ('Rising Three Methods', 'Strongly Bullish', 'stop', entry, sl)

def bullish_kicker(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    l2     = df['Low'].iloc[i]
    l1     = df['Low'].iloc[i-1]
    if is_bear(o1, c1) and is_bull(o2, c2):
        if o2 > o1:   # gap up at open
            # Market order at next bar's open — always fills
            entry = None    # filled at next open (resolved in scan loop)
            sl    = l1      # yesterday's low as SL
            return ('Bullish Kicker', 'Strongly Bullish', 'market_next_open', entry, sl)
    return None

DETECTORS = [
    hammer, inverted_hammer, bullish_engulfing, piercing_line,
    morning_star, morning_doji_star, three_white_soldiers,
    bullish_harami, dragonfly_doji, tweezer_bottom,
    on_neck, rising_three_methods, bullish_kicker,
]

# ── Entry trigger logic ───────────────────────────────────────────────────────
def find_entry(df, signal_bar_idx, entry_type, entry_price, max_bars):
    """
    Look forward from signal_bar_idx+1 to find if entry is triggered.

    Returns:
      (entry_bar_idx, actual_entry_price) if triggered
      (None, None) if never triggered within max_bars
    """
    n = len(df)

    if entry_type == 'market_next_open':
        # Always triggers at very next bar's open
        j = signal_bar_idx + 1
        if j >= n:
            return None, None
        return j, df['Open'].iloc[j]

    elif entry_type == 'stop':
        # Buy Stop: triggered when a future bar's High >= entry_price
        for offset in range(1, max_bars + 1):
            j = signal_bar_idx + offset
            if j >= n:
                break
            if df['High'].iloc[j] >= entry_price:
                # Assume filled at entry_price (stop order)
                return j, entry_price
        return None, None

    elif entry_type == 'limit':
        # Buy Limit: triggered when a future bar's Low <= entry_price
        for offset in range(1, max_bars + 1):
            j = signal_bar_idx + offset
            if j >= n:
                break
            if df['Low'].iloc[j] <= entry_price:
                # Assume filled at entry_price (limit order)
                return j, entry_price
        return None, None

    return None, None

# ── Forward return from entry bar ─────────────────────────────────────────────
def fwd_return(df, entry_bar_idx, actual_entry, days, intraday_fill=False):
    """
    Return % change measured as follows:
      1D (days=1):
        market_next_open (intraday_fill=False):
          entry bar IS the next bar → Open-to-Close of entry_bar
        stop / limit (intraday_fill=True):
          stop fills intraday on entry_bar; realistic hold starts next morning
          → Open[entry_bar+1] to Close[entry_bar+1]
      3D (days=3): Close[entry_bar + 3] vs actual_entry price
      5D (days=5): Close[entry_bar + 5] vs actual_entry price
    """
    n = len(df)
    if days == 1:
        if intraday_fill:
            # Stop/limit filled intraday — can't use that bar's open
            # Realistic 1D: next bar open-to-close
            j = entry_bar_idx + 1
            if j >= n:
                return None, None
            bar_open  = df['Open'].iloc[j]
            bar_close = df['Close'].iloc[j]
        else:
            # market_next_open: already entering at open of entry_bar
            bar_open  = df['Open'].iloc[entry_bar_idx]
            bar_close = df['Close'].iloc[entry_bar_idx]
        if bar_open == 0 or pd.isna(bar_open) or pd.isna(bar_close):
            return None, None
        pct   = (bar_close - bar_open) / bar_open * 100
        label = 'Profit' if pct > 0 else 'Loss'
        return round(pct, 2), label
    else:
        # 3D / 5D: close N bars after entry vs actual entry price
        j = entry_bar_idx + days
        if j >= n or actual_entry == 0:
            return None, None
        exit_price = df['Close'].iloc[j]
        pct        = (exit_price - actual_entry) / actual_entry * 100
        label      = 'Profit' if pct > 0 else 'Loss'
        return round(pct, 2), label

# ── Per-ticker scan ───────────────────────────────────────────────────────────
def scan_ticker(ticker, df_full):
    df = df_full.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Only scan within 2024-2025
    df = df[(df.index >= SCAN_START) & (df.index <= SCAN_END)]
    if len(df) < 10:
        return []

    df = df.reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    if 'Date' not in df.columns and df.columns[0] != 'Date':
        df.insert(0, 'Date', df.index)

    df['body'] = (df['Close'] - df['Open']).abs()

    signals = []
    n = len(df)

    for i in range(4, n - MAX_ENTRY_BARS - 5):
        # Need forward room: MAX_ENTRY_BARS for entry + 5 for return measurement
        date = df['Date'].iloc[i]
        sig_date = pd.Timestamp(date)
        if sig_date < SCAN_START or sig_date > SCAN_END:
            continue

        for detector in DETECTORS:
            result = detector(df, i)
            if result is None:
                continue

            pattern, strength, entry_type, entry_price, sl_price = result

            # Resolve entry
            entry_bar, actual_entry = find_entry(
                df, i, entry_type, entry_price, MAX_ENTRY_BARS
            )

            if entry_bar is None:
                continue   # entry never triggered — skip this signal entirely

            # Forward returns measured FROM entry_bar / actual_entry
            # intraday_fill=True for stop/limit (fills during the bar, not at its open)
            intraday = (entry_type in ('stop', 'limit'))
            r1_pct, r1_lbl = fwd_return(df, entry_bar, actual_entry, 1, intraday_fill=intraday)
            r3_pct, r3_lbl = fwd_return(df, entry_bar, actual_entry, 3)
            r5_pct, r5_lbl = fwd_return(df, entry_bar, actual_entry, 5)

            # Risk: distance from entry to SL (as %)
            risk_pct = round((actual_entry - sl_price) / actual_entry * 100, 2) if sl_price and actual_entry else None

            signals.append({
                'Ticker':          ticker,
                'Signal Date':     date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10],
                'Entry Date':      df['Date'].iloc[entry_bar].strftime('%Y-%m-%d')
                                   if hasattr(df['Date'].iloc[entry_bar], 'strftime')
                                   else str(df['Date'].iloc[entry_bar])[:10],
                'Pattern':         pattern,
                'Strength':        strength,
                'Entry Type':      entry_type,
                # Signal bar OHLC
                'Sig Open':        round(df['Open'].iloc[i], 2),
                'Sig High':        round(df['High'].iloc[i], 2),
                'Sig Low':         round(df['Low'].iloc[i], 2),
                'Sig Close':       round(df['Close'].iloc[i], 2),
                # Entry details
                'Entry Price':     round(actual_entry, 2),
                'SL Price':        round(sl_price, 2) if sl_price else None,
                'Risk (%)':        risk_pct,
                # Forward returns from entry
                '1D Return (%)':   r1_pct,
                '1D Result':       r1_lbl,
                '3D Return (%)':   r3_pct,
                '3D Result':       r3_lbl,
                '5D Return (%)':   r5_pct,
                '5D Result':       r5_lbl,
            })

    return signals

# ── Run across all tickers ────────────────────────────────────────────────────
files       = sorted(f for f in os.listdir(DAILY_DIR) if f.endswith('_daily.csv'))
all_signals = []
skipped     = 0

print(f"Scanning {len(files)} tickers (2024-2025, entry-filter mode)...", flush=True)
for idx, fname in enumerate(files):
    ticker = fname.replace('_daily.csv', '')
    path   = os.path.join(DAILY_DIR, fname)
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df.get('Volume', 0), errors='coerce').fillna(0).astype(int)
        sigs = scan_ticker(ticker, df)
        all_signals.extend(sigs)
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(files)} done | signals so far: {len(all_signals):,}", flush=True)
    except Exception as e:
        skipped += 1
        print(f"  SKIP {ticker}: {e}", flush=True)

print(f"\nDone. Confirmed signals: {len(all_signals):,}  |  Skipped tickers: {skipped}")

if not all_signals:
    print("No confirmed signals found.")
    exit()

result = pd.DataFrame(all_signals)
result['Year'] = result['Signal Date'].str[:4]
result = result.sort_values(['Signal Date', 'Ticker']).reset_index(drop=True)

# ── Append scores ─────────────────────────────────────────────────────────────
try:
    fs   = pd.read_excel(os.path.join(SHEETS_DIR, 'fundamental_score_2026-02-19.xlsx'),
                         usecols=['Ticker', 'Fundamental Score'])
    result = result.merge(fs, on='Ticker', how='left')
except Exception:
    result['Fundamental Score'] = None

try:
    perf = pd.read_excel(os.path.join(SHEETS_DIR, 'performers_2024_2025.xlsx'),
                         usecols=['Ticker', 'Score']).rename(columns={'Score': 'Performer Score'})
    result = result.merge(perf, on='Ticker', how='left')
except Exception:
    result['Performer Score'] = None

try:
    bs   = pd.read_excel(os.path.join(SHEETS_DIR, 'base_score_2026-02-19.xlsx'),
                         usecols=['Ticker', 'Base Score', 'Status']).rename(
                             columns={'Status': 'Base Status'})
    result = result.merge(bs, on='Ticker', how='left')
except Exception:
    result['Base Score'] = None
    result['Base Status'] = None

# ── Per-pattern summary ───────────────────────────────────────────────────────
def pattern_summary(df):
    rows = []
    for pat, grp in df.groupby('Pattern'):
        n      = len(grp)
        p1     = (grp['1D Result'] == 'Profit').mean() * 100
        p3     = (grp['3D Result'] == 'Profit').mean() * 100
        p5     = (grp['5D Result'] == 'Profit').mean() * 100
        avg_r1 = grp['1D Return (%)'].mean()
        avg_r3 = grp['3D Return (%)'].mean()
        avg_r5 = grp['5D Return (%)'].mean()
        rows.append({
            'Pattern':        pat,
            'Signals':        n,
            '1D Profit %':    round(p1, 1),
            '3D Profit %':    round(p3, 1),
            '5D Profit %':    round(p5, 1),
            'Avg 1D Ret':     round(avg_r1, 2) if not np.isnan(avg_r1) else None,
            'Avg 3D Ret':     round(avg_r3, 2) if not np.isnan(avg_r3) else None,
            'Avg 5D Ret':     round(avg_r5, 2) if not np.isnan(avg_r5) else None,
        })
    return pd.DataFrame(rows).sort_values('3D Profit %', ascending=False)

# ── Print quick summary ───────────────────────────────────────────────────────
print(f"\nSignals by year:")
print(result['Year'].value_counts().sort_index().to_string())

print(f"\nOverall profit rates (entry-filtered):")
print(f"  1D: {(result['1D Result']=='Profit').mean()*100:.1f}%")
print(f"  3D: {(result['3D Result']=='Profit').mean()*100:.1f}%")
print(f"  5D: {(result['5D Result']=='Profit').mean()*100:.1f}%")

print(f"\nEntry type breakdown:")
print(result['Entry Type'].value_counts().to_string())

print(f"\nPattern profit rates (3D, entry-filtered):")
summ = pattern_summary(result)
print(summ[['Pattern','Signals','1D Profit %','3D Profit %','5D Profit %']].to_string(index=False))

# ── Excel output ─────────────────────────────────────────────────────────────
print(f"\nWriting Excel: {OUT}")

df_2024 = result[result['Year'] == '2024'].drop(columns='Year').reset_index(drop=True)
df_2025 = result[result['Year'] == '2025'].drop(columns='Year').reset_index(drop=True)
summ_all  = pattern_summary(result)
summ_2024 = pattern_summary(df_2024)
summ_2025 = pattern_summary(df_2025)

# Replacement for NaN in return cols
for col in ['1D Return (%)','3D Return (%)','5D Return (%)']:
    result[col]   = result[col].where(result[col].notna(), other=None)
    df_2024[col]  = df_2024[col].where(df_2024[col].notna(), other=None)
    df_2025[col]  = df_2025[col].where(df_2025[col].notna(), other=None)

PROFIT_COLOR = '#C6EFCE'
LOSS_COLOR   = '#FFC7CE'
STRONG_COLOR = '#C6EFCE'
BULL_COLOR   = '#FFEB9C'

def write_signal_sheet(writer, df_sheet, sheet_name, wb):
    df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]

    fmt_hdr    = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white',
                                 'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fmt_strong = wb.add_format({'bg_color': STRONG_COLOR, 'font_color': '#276221', 'bold': True})
    fmt_bull   = wb.add_format({'bg_color': BULL_COLOR,   'font_color': '#9C6500'})
    fmt_profit = wb.add_format({'bg_color': PROFIT_COLOR, 'font_color': '#276221',
                                 'num_format': '+0.00;-0.00'})
    fmt_loss   = wb.add_format({'bg_color': LOSS_COLOR,   'font_color': '#9C0006',
                                 'num_format': '+0.00;-0.00'})
    fmt_lbl_p  = wb.add_format({'bg_color': PROFIT_COLOR, 'font_color': '#276221'})
    fmt_lbl_l  = wb.add_format({'bg_color': LOSS_COLOR,   'font_color': '#9C0006'})
    fmt_num    = wb.add_format({'num_format': '#,##0.00'})

    col_widths = {
        'Ticker': 12, 'Signal Date': 13, 'Entry Date': 13, 'Pattern': 24,
        'Strength': 18, 'Entry Type': 18,
        'Sig Open': 10, 'Sig High': 10, 'Sig Low': 10, 'Sig Close': 10,
        'Entry Price': 12, 'SL Price': 10, 'Risk (%)': 10,
        '1D Return (%)': 13, '1D Result': 10,
        '3D Return (%)': 13, '3D Result': 10,
        '5D Return (%)': 13, '5D Result': 10,
        'Fundamental Score': 18, 'Performer Score': 16,
        'Base Score': 12, 'Base Status': 14,
    }
    for ci, cn in enumerate(df_sheet.columns):
        ws.set_column(ci, ci, col_widths.get(cn, 12), fmt_num)
        ws.write(0, ci, cn, fmt_hdr)
    ws.set_row(0, 30)

    strength_ci = df_sheet.columns.get_loc('Strength')
    for ci_name, lbl_name in [
        ('1D Return (%)', '1D Result'),
        ('3D Return (%)', '3D Result'),
        ('5D Return (%)', '5D Result'),
    ]:
        pct_ci = df_sheet.columns.get_loc(ci_name)
        lbl_ci = df_sheet.columns.get_loc(lbl_name)
        for ri, row in df_sheet.iterrows():
            er = ri + 1
            # Strength colour (once per row)
            if ci_name == '1D Return (%)':
                f_str = fmt_strong if row.get('Strength') == 'Strongly Bullish' else fmt_bull
                ws.write(er, strength_ci, row.get('Strength', ''), f_str)

            lbl = row[lbl_name]
            pct = row[ci_name]
            f_lbl = fmt_lbl_p if lbl == 'Profit' else fmt_lbl_l
            f_pct = fmt_profit if lbl == 'Profit' else fmt_loss
            ws.write(er, lbl_ci, lbl if lbl else '', f_lbl if lbl else fmt_num)
            if pct is not None and pct == pct:   # not NaN
                ws.write(er, pct_ci, pct, f_pct)

    # Score colour scales
    for sc_col in ['Fundamental Score', 'Performer Score', 'Base Score']:
        if sc_col in df_sheet.columns:
            ci = df_sheet.columns.get_loc(sc_col)
            ws.conditional_format(1, ci, len(df_sheet), ci, {
                'type': '3_color_scale',
                'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B',
            })

    BASE_COLORS = {
        'Strong Buy': {'bg_color': '#C6EFCE', 'font_color': '#276221'},
        'Good':       {'bg_color': '#FFEB9C', 'font_color': '#9C6500'},
        'Hold':       {'bg_color': '#BDD7EE', 'font_color': '#1F4E79'},
        'AVOID':      {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
    }
    if 'Base Status' in df_sheet.columns:
        bs_ci = df_sheet.columns.get_loc('Base Status')
        for ri, status in enumerate(df_sheet['Base Status']):
            cfg = BASE_COLORS.get(status)
            if cfg:
                ws.write(ri + 1, bs_ci, status, wb.add_format({**cfg, 'bold': True}))

    ws.freeze_panes(1, 2)
    ws.autofilter(0, 0, len(df_sheet), len(df_sheet.columns) - 1)


def write_summary_sheet(writer, summ_df, sheet_name, wb):
    summ_df.to_excel(writer, index=False, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]
    fmt_hdr = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white',
                              'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    for ci, cn in enumerate(summ_df.columns):
        ws.set_column(ci, ci, max(len(cn) + 2, 14))
        ws.write(0, ci, cn, fmt_hdr)
    ws.set_row(0, 28)
    for pct_col in ['1D Profit %', '3D Profit %', '5D Profit %']:
        if pct_col in summ_df.columns:
            ci = summ_df.columns.get_loc(pct_col)
            ws.conditional_format(1, ci, len(summ_df), ci, {
                'type': '3_color_scale',
                'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B',
            })
    ws.autofilter(0, 0, len(summ_df), len(summ_df.columns) - 1)


with pd.ExcelWriter(OUT, engine='xlsxwriter',
                    engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
    wb = writer.book

    if len(df_2024):
        write_signal_sheet(writer, df_2024, '2024 Signals', wb)
    if len(df_2025):
        write_signal_sheet(writer, df_2025, '2025 Signals', wb)
    write_summary_sheet(writer, summ_all,  'Summary (All)',  wb)
    write_summary_sheet(writer, summ_2024, 'Summary 2024',   wb)
    write_summary_sheet(writer, summ_2025, 'Summary 2025',   wb)

print(f"\nSaved: {OUT}")
print(f"  2024 signals: {len(df_2024):,}")
print(f"  2025 signals: {len(df_2025):,}")
