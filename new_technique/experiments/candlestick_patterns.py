"""
Candlestick Pattern Detector
=============================
- Scans all daily_data CSVs for the last 3 months
- Detects bullish & strongly bullish candlestick patterns
- For each signal: records date, ticker, pattern, strength,
  and forward returns at 1-day, 3-day, 5-day (Profit/Loss)
- Output: candlestick_signals_<date>.xlsx
"""

import os
import datetime
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(__file__)
DAILY_DIR     = os.path.join(BASE, '..', '..', 'daily_data')
SHEETS_DIR    = os.path.join(BASE, 'sheets')
os.makedirs(SHEETS_DIR, exist_ok=True)
TODAY         = datetime.date.today()
TODAY_STR     = TODAY.strftime('%Y-%m-%d')
OUT           = os.path.join(SHEETS_DIR, f'candlestick_signals_{TODAY_STR}.xlsx')

# Last 3 months window
LOOKBACK_DATE = pd.Timestamp(TODAY - datetime.timedelta(days=92))

# ── Helpers ────────────────────────────────────────────────────────────────────
def body(o, c):       return abs(c - o)
def upper_wick(o, c, h): return h - max(o, c)
def lower_wick(o, c, l): return min(o, c) - l
def full_range(h, l): return h - l
def is_bullish_candle(o, c): return c > o
def is_bearish_candle(o, c): return c < o

# Average body over prior N candles (volatility reference)
def avg_body(df, idx, n=5):
    start = max(0, idx - n)
    return df['body'].iloc[start:idx].mean()

# ── Pattern definitions ────────────────────────────────────────────────────────
# Each returns (pattern_name, strength) or None
# strength: 'Bullish' | 'Strongly Bullish'

def hammer(df, i):
    o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    b   = body(o, c)
    lw  = lower_wick(o, c, l)
    uw  = upper_wick(o, c, h)
    rng = full_range(h, l)
    if rng == 0: return None
    if lw >= 2 * b and uw <= 0.1 * rng and rng > 0:
        return ('Hammer', 'Strongly Bullish')
    return None

def inverted_hammer(df, i):
    if i < 1: return None
    o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    b  = body(o, c)
    uw = upper_wick(o, c, h)
    lw = lower_wick(o, c, l)
    rng = full_range(h, l)
    if rng == 0: return None
    # After downtrend candle, small body, long upper wick
    prev_bearish = is_bearish_candle(df['Open'].iloc[i-1], df['Close'].iloc[i-1])
    if prev_bearish and uw >= 2 * b and lw <= 0.1 * rng:
        return ('Inverted Hammer', 'Bullish')
    return None

def bullish_engulfing(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    if is_bearish_candle(o1, c1) and is_bullish_candle(o2, c2):
        if o2 <= c1 and c2 >= o1:
            return ('Bullish Engulfing', 'Strongly Bullish')
    return None

def piercing_line(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    mid1 = (o1 + c1) / 2
    if is_bearish_candle(o1, c1) and is_bullish_candle(o2, c2):
        if o2 < c1 and c2 > mid1 and c2 < o1:
            return ('Piercing Line', 'Bullish')
    return None

def morning_star(df, i):
    if i < 2: return None
    o1, c1 = df['Open'].iloc[i-2], df['Close'].iloc[i-2]
    o2, c2 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o3, c3 = df['Open'].iloc[i],   df['Close'].iloc[i]
    b2  = body(o2, c2)
    ab  = avg_body(df, i-1)
    if ab == 0: return None
    small_mid = b2 < 0.3 * ab
    if is_bearish_candle(o1, c1) and small_mid and is_bullish_candle(o3, c3):
        if c3 > (o1 + c1) / 2:
            return ('Morning Star', 'Strongly Bullish')
    return None

def morning_doji_star(df, i):
    if i < 2: return None
    o1, c1 = df['Open'].iloc[i-2], df['Close'].iloc[i-2]
    o2, c2 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o3, c3 = df['Open'].iloc[i],   df['Close'].iloc[i]
    rng2 = full_range(df['High'].iloc[i-1], df['Low'].iloc[i-1])
    b2   = body(o2, c2)
    is_doji = rng2 > 0 and b2 / rng2 < 0.1
    if is_bearish_candle(o1, c1) and is_doji and is_bullish_candle(o3, c3):
        return ('Morning Doji Star', 'Strongly Bullish')
    return None

def three_white_soldiers(df, i):
    if i < 2: return None
    rows = [(df['Open'].iloc[i-2+k], df['Close'].iloc[i-2+k],
             df['High'].iloc[i-2+k], df['Low'].iloc[i-2+k]) for k in range(3)]
    for o, c, h, l in rows:
        if not is_bullish_candle(o, c): return None
        uw = upper_wick(o, c, h)
        b  = body(o, c)
        if b == 0: return None
        if uw > 0.2 * b: return None   # small upper wick required
    # Each open within prior body
    for k in range(1, 3):
        o_prev, c_prev = rows[k-1][0], rows[k-1][1]
        o_cur          = rows[k][0]
        if not (o_prev <= o_cur <= c_prev): return None
        if rows[k][1] <= c_prev: return None  # each close higher
    return ('Three White Soldiers', 'Strongly Bullish')

def bullish_harami(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    if is_bearish_candle(o1, c1) and is_bullish_candle(o2, c2):
        if o2 > c1 and c2 < o1:
            return ('Bullish Harami', 'Bullish')
    return None

def dragonfly_doji(df, i):
    o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    rng = full_range(h, l)
    if rng == 0: return None
    b  = body(o, c)
    lw = lower_wick(o, c, l)
    uw = upper_wick(o, c, h)
    if b / rng < 0.1 and lw > 0.6 * rng and uw < 0.05 * rng:
        return ('Dragonfly Doji', 'Bullish')
    return None

def tweezer_bottom(df, i):
    if i < 1: return None
    l1 = df['Low'].iloc[i-1]
    l2 = df['Low'].iloc[i]
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    tol = 0.002 * ((l1 + l2) / 2)  # 0.2% tolerance
    if abs(l1 - l2) <= tol and is_bearish_candle(o1, c1) and is_bullish_candle(o2, c2):
        return ('Tweezer Bottom', 'Bullish')
    return None

def on_neck(df, i):
    """Bullish On-Neck: bearish day, next opens below low, closes near prior close."""
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    if is_bearish_candle(o1, c1) and is_bullish_candle(o2, c2):
        tol = 0.005 * c1
        if o2 < df['Low'].iloc[i-1] and abs(c2 - c1) <= tol:
            return ('On Neck (Bullish Setup)', 'Bullish')
    return None

def rising_three_methods(df, i):
    if i < 4: return None
    o0, c0 = df['Open'].iloc[i-4], df['Close'].iloc[i-4]
    o4, c4 = df['Open'].iloc[i],   df['Close'].iloc[i]
    if not is_bullish_candle(o0, c0) or not is_bullish_candle(o4, c4): return None
    if c4 <= c0: return None
    for k in range(1, 4):
        ok, ck = df['Open'].iloc[i-4+k], df['Close'].iloc[i-4+k]
        if not is_bearish_candle(ok, ck): return None
        if ok > c0 or ck < df['Low'].iloc[i-4]: return None
    return ('Rising Three Methods', 'Strongly Bullish')

def bullish_kicker(df, i):
    if i < 1: return None
    o1, c1 = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    o2, c2 = df['Open'].iloc[i],   df['Close'].iloc[i]
    if is_bearish_candle(o1, c1) and is_bullish_candle(o2, c2):
        if o2 > o1:   # gap up at open
            return ('Bullish Kicker', 'Strongly Bullish')
    return None

# All detectors list
DETECTORS = [
    hammer, inverted_hammer, bullish_engulfing, piercing_line,
    morning_star, morning_doji_star, three_white_soldiers,
    bullish_harami, dragonfly_doji, tweezer_bottom,
    on_neck, rising_three_methods, bullish_kicker,
]

# ── Main scan ──────────────────────────────────────────────────────────────────
def scan_ticker(ticker, df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Filter to last 3 months
    df = df[df.index >= LOOKBACK_DATE]
    if len(df) < 6:
        return []

    df = df.reset_index()
    df['body'] = abs(df['Close'] - df['Open'])

    signals = []
    n = len(df)

    for i in range(2, n):   # need at least 2 prior candles for multi-candle patterns
        date = df['Date'].iloc[i]
        for detector in DETECTORS:
            result = detector(df, i)
            if result is None:
                continue
            pattern, strength = result

            # Forward returns
            def fwd(days):
                j = i + days
                if j >= n:
                    return None, None
                entry = df['Close'].iloc[i]
                exit_ = df['Close'].iloc[j]
                pct   = (exit_ - entry) / entry * 100
                label = 'Profit' if pct > 0 else 'Loss'
                return round(pct, 2), label

            r1_pct, r1_lbl = fwd(1)
            r3_pct, r3_lbl = fwd(3)
            r5_pct, r5_lbl = fwd(5)

            signals.append({
                'Ticker':        ticker,
                'Date':          date.strftime('%Y-%m-%d'),
                'Pattern':       pattern,
                'Strength':      strength,
                'Open':          round(df['Open'].iloc[i], 2),
                'High':          round(df['High'].iloc[i], 2),
                'Low':           round(df['Low'].iloc[i], 2),
                'Close':         round(df['Close'].iloc[i], 2),
                'Volume':        int(df['Volume'].iloc[i]),
                'RSI':           round(df['RSI'].iloc[i], 2) if pd.notna(df['RSI'].iloc[i]) else None,
                '1D Return (%)': r1_pct,
                '1D Result':     r1_lbl,
                '3D Return (%)': r3_pct,
                '3D Result':     r3_lbl,
                '5D Return (%)': r5_pct,
                '5D Result':     r5_lbl,
            })
    return signals

# ── Run across all tickers ─────────────────────────────────────────────────────
files   = sorted(f for f in os.listdir(DAILY_DIR) if f.endswith('_daily.csv'))
all_signals = []
skipped = 0

print(f"Scanning {len(files)} tickers for bullish patterns (last 3 months)...")
for i, fname in enumerate(files):
    ticker = fname.replace('_daily.csv', '')
    path   = os.path.join(DAILY_DIR, fname)
    try:
        df = pd.read_csv(path, index_col='Date')
        df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].apply(
            pd.to_numeric, errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
        df['RSI']    = pd.to_numeric(df['RSI'], errors='coerce')
        sigs = scan_ticker(ticker, df)
        all_signals.extend(sigs)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} | signals so far: {len(all_signals)}")
    except Exception as e:
        skipped += 1

print(f"\nDone. Total signals: {len(all_signals)}  |  Skipped: {skipped}")

if not all_signals:
    print("No signals found.")
    exit()

result = pd.DataFrame(all_signals)
result = result.sort_values(['Date','Ticker']).reset_index(drop=True)

# ── Append Fundamental Score and Performer Score ───────────────────────────────
fs   = pd.read_excel(os.path.join(SHEETS_DIR, 'fundamental_score_2026-02-19.xlsx'),
                     usecols=['Ticker', 'Fundamental Score'])
perf = pd.read_excel(os.path.join(SHEETS_DIR, 'performers_2024_2025.xlsx'),
                     usecols=['Ticker', 'Score']).rename(columns={'Score': 'Performer Score'})
bs   = pd.read_excel(os.path.join(SHEETS_DIR, 'base_score_2026-02-19.xlsx'),
                     usecols=['Ticker', 'Base Score', 'Status']).rename(
                         columns={'Status': 'Base Status'})

result = result.merge(fs,   on='Ticker', how='left')
result = result.merge(perf, on='Ticker', how='left')
result = result.merge(bs,   on='Ticker', how='left')

# ── Excel output ───────────────────────────────────────────────────────────────
STRONG_COLOR  = '#C6EFCE'   # green
BULLISH_COLOR = '#FFEB9C'   # amber
PROFIT_COLOR  = '#C6EFCE'
LOSS_COLOR    = '#FFC7CE'

# Replace NaN in return columns with None so xlsxwriter handles them cleanly
for col in ['1D Return (%)','3D Return (%)','5D Return (%)']:
    result[col] = result[col].where(result[col].notna(), other=None)

with pd.ExcelWriter(OUT, engine='xlsxwriter',
                    engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
    result.to_excel(writer, index=False, sheet_name='Signals')

    wb = writer.book
    ws = writer.sheets['Signals']

    fmt_hdr    = wb.add_format({'bold': True, 'bg_color': '#D9E1F2',
                                 'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fmt_strong = wb.add_format({'bg_color': STRONG_COLOR,  'font_color': '#276221', 'bold': True})
    fmt_bull   = wb.add_format({'bg_color': BULLISH_COLOR, 'font_color': '#9C6500'})
    fmt_profit = wb.add_format({'bg_color': PROFIT_COLOR,  'font_color': '#276221', 'num_format': '+0.00;-0.00'})
    fmt_loss   = wb.add_format({'bg_color': LOSS_COLOR,    'font_color': '#9C0006', 'num_format': '+0.00;-0.00'})
    fmt_num    = wb.add_format({'num_format': '#,##0.00'})
    fmt_int    = wb.add_format({'num_format': '#,##0'})

    col_widths = {
        'Ticker': 14, 'Date': 12, 'Pattern': 24, 'Strength': 18,
        'Open': 10, 'High': 10, 'Low': 10, 'Close': 10,
        'Volume': 13, 'RSI': 8,
        '1D Return (%)': 13, '1D Result': 10,
        '3D Return (%)': 13, '3D Result': 10,
        '5D Return (%)': 13, '5D Result': 10,
        'Fundamental Score': 17, 'Performer Score': 15,
        'Base Score': 12, 'Base Status': 14,
    }
    for col_num, col_name in enumerate(result.columns):
        ws.set_column(col_num, col_num, col_widths.get(col_name, 12), fmt_num)

    # Integer format for Volume
    vol_idx = result.columns.get_loc('Volume')
    ws.set_column(vol_idx, vol_idx, 13, fmt_int)

    # Header
    ws.set_row(0, 30)
    for col_num, col_name in enumerate(result.columns):
        ws.write(0, col_num, col_name, fmt_hdr)

    # Data rows — colour Strength and Result columns
    strength_idx = result.columns.get_loc('Strength')
    r1lbl_idx = result.columns.get_loc('1D Result')
    r3lbl_idx = result.columns.get_loc('3D Result')
    r5lbl_idx = result.columns.get_loc('5D Result')
    r1pct_idx = result.columns.get_loc('1D Return (%)')
    r3pct_idx = result.columns.get_loc('3D Return (%)')
    r5pct_idx = result.columns.get_loc('5D Return (%)')

    for row_num, row in result.iterrows():
        excel_row = row_num + 1

        # Strength colour
        s_fmt = fmt_strong if row['Strength'] == 'Strongly Bullish' else fmt_bull
        ws.write(excel_row, strength_idx, row['Strength'], s_fmt)

        # Result colours
        for lbl_idx, pct_idx, lbl_col, pct_col in [
            (r1lbl_idx, r1pct_idx, '1D Result', '1D Return (%)'),
            (r3lbl_idx, r3pct_idx, '3D Result', '3D Return (%)'),
            (r5lbl_idx, r5pct_idx, '5D Result', '5D Return (%)'),
        ]:
            lbl = row[lbl_col]
            pct = row[pct_col]
            lf = fmt_profit if lbl == 'Profit' else fmt_loss
            ws.write(excel_row, lbl_idx, lbl if lbl else '', lf if lbl else fmt_num)
            if pct is not None:
                ws.write(excel_row, pct_idx, pct, fmt_profit if lbl == 'Profit' else fmt_loss)

    # Colour scales for Fundamental Score, Performer Score, Base Score
    for score_col_name in ['Fundamental Score', 'Performer Score', 'Base Score']:
        if score_col_name in result.columns:
            ci = result.columns.get_loc(score_col_name)
            ws.conditional_format(1, ci, len(result), ci, {
                'type':      '3_color_scale',
                'min_color': '#F8696B',
                'mid_color': '#FFEB84',
                'max_color': '#63BE7B',
            })

    # Colour Base Status column
    BASE_STATUS_COLORS = {
        'Strong Buy': {'bg_color': '#C6EFCE', 'font_color': '#276221'},
        'Good':       {'bg_color': '#FFEB9C', 'font_color': '#9C6500'},
        'Hold':       {'bg_color': '#BDD7EE', 'font_color': '#1F4E79'},
        'AVOID':      {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
    }
    if 'Base Status' in result.columns:
        bs_status_idx = result.columns.get_loc('Base Status')
        for row_num, status in enumerate(result['Base Status']):
            fmt_cfg = BASE_STATUS_COLORS.get(status)
            if fmt_cfg:
                fmt_bs = wb.add_format({**fmt_cfg, 'bold': True})
                ws.write(row_num + 1, bs_status_idx, status, fmt_bs)

    ws.freeze_panes(1, 2)
    ws.autofilter(0, 0, len(result), len(result.columns) - 1)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nSaved to: {OUT}")
print(f"Total signals: {len(result)}")
print(f"\nSignals by Strength:")
print(result['Strength'].value_counts().to_string())
print(f"\nTop 10 Patterns:")
print(result['Pattern'].value_counts().head(10).to_string())
print(f"\n1D Profit rate: {(result['1D Result']=='Profit').mean()*100:.1f}%")
print(f"3D Profit rate: {(result['3D Result']=='Profit').mean()*100:.1f}%")
print(f"5D Profit rate: {(result['5D Result']=='Profit').mean()*100:.1f}%")
