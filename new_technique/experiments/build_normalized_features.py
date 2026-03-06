"""
Build a normalized feature sheet from fundamentals_2026-02-19.xlsx.

Rules
-----
• Remove raw absolute-scale columns that get replaced by ratios.
• Keep all columns that are already ratios / percentages / prices unchanged.
• Add new normalized columns as specified.
• Split 'Analyst Rating' into 'Analyst Score' (float) and 'Analyst Label' (text).
• Output: normalized_features_<date>.xlsx
"""

import os
import datetime
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
F_FILE  = os.path.join(BASE, 'fundamentals_2026-02-19.xlsx')
TODAY   = datetime.date.today().strftime('%Y-%m-%d')
OUT     = os.path.join(BASE, f'normalized_features_{TODAY}.xlsx')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel(F_FILE)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ── 1. Split Analyst Rating → Analyst Score + Analyst Label ──────────────────
def parse_analyst(val):
    """'1.9 - Buy' → (1.9, 'Buy')   |   NaN → (NaN, NaN)"""
    if pd.isna(val):
        return np.nan, np.nan
    parts = str(val).split(' - ', 1)
    try:
        score = float(parts[0].strip())
    except ValueError:
        score = np.nan
    label = parts[1].strip() if len(parts) == 2 else np.nan
    return score, label

analyst_parsed = df['Analyst Rating'].apply(parse_analyst)
df['Analyst Score'] = analyst_parsed.apply(lambda x: x[0])
df['Analyst Label'] = analyst_parsed.apply(lambda x: x[1])

# ── 2. Compute new normalized columns ────────────────────────────────────────
def safe_div(num, den):
    """Element-wise division; returns NaN where denominator is 0 or NaN."""
    n = pd.to_numeric(num, errors='coerce')
    d = pd.to_numeric(den, errors='coerce')
    return np.where((d.isna()) | (d == 0), np.nan, n / d)

# 52W High Norm  = Price / 52W High * 100   (how close to high, %)
df['52W High Norm']           = safe_div(df['Price'], df['52W High']) * 100

# 52W Low Norm   = 52W Low / Price * 100    (how far above low, %)
df['52W Low Norm']            = safe_div(df['52W Low'], df['Price']) * 100

# P/S Ratio already exists — but recompute cleanly as Market Cap / Total Revenue
df['P/S Ratio']               = safe_div(df['Market Cap'], df['Total Revenue'])

# EV/EBITDA already exists — recompute cleanly
df['EV/EBITDA']               = safe_div(df['Enterprise Value'], df['EBITDA'])

# Revenue / Market Cap  (replaces "Asset Turnover" since Total Assets N/A)
df['Revenue/MarketCap']       = safe_div(df['Total Revenue'], df['Market Cap'])

# EBITDA Margin already exists — recompute cleanly
df['EBITDA Margin (%)']       = safe_div(df['EBITDA'], df['Total Revenue']) * 100

# Gross Margin already exists — recompute cleanly
df['Gross Margin (%)']        = safe_div(df['Gross Profit'], df['Total Revenue']) * 100

# Net / Profit Margin already exists — recompute cleanly
df['Net Margin (%)']          = safe_div(df['Net Income'], df['Total Revenue']) * 100

# Cash / Market Cap  (liquidity proxy; Total Assets unavailable)
df['Cash/MarketCap (%)']      = safe_div(df['Total Cash'], df['Market Cap']) * 100

# Debt / Market Cap  (leverage; Market Cap as base instead of equity book)
df['Debt/MarketCap (%)']      = safe_div(df['Total Debt'], df['Market Cap']) * 100

# Cash Conversion Ratio = Free Cash Flow / Net Income
df['Cash Conversion Ratio']   = safe_div(df['Free Cash Flow'], df['Net Income'])

# Volume / Market Cap * 100  (% of market cap traded daily on avg)
df['Vol/MarketCap (%)']       = safe_div(df['Avg Vol (3M)'] * df['Price'],
                                          df['Market Cap']) * 100

# ── 3. Define final column order ─────────────────────────────────────────────
#   Remove raw absolute-scale cols that have been replaced, and the originals
#   that were split or superseded.
DROP = {
    # Replaced by normalized versions
    'Market Cap', 'Enterprise Value',
    'Total Revenue', 'EBITDA', 'Gross Profit', 'Net Income',
    'Total Cash', 'Total Debt', 'Free Cash Flow',
    'Avg Vol (3M)',
    # Price & 52W raw — kept only as intermediates; normalized versions kept
    '52W High', '52W Low',
    'Price',
    # Analyst Rating original — split into two columns
    'Analyst Rating',
    # Target High/Low raw — keep only Target Mean Price
    'Target High', 'Target Low',
    # Div Rate raw — Div Yield % is sufficient
    'Div. Rate',
}

FINAL_COLS = [
    # ── Identity ────────────────────────────────────────────────────────
    'Ticker', 'Company Name', 'Sector', 'Industry',

    # ── Price Position ───────────────────────────────────────────────────
    '52W High Norm',        # Price / 52W High × 100  (near high = high %)
    '52W Low Norm',         # 52W Low / Price × 100   (near low = high %)
    '52W Change (%)',

    # ── Market / Risk ────────────────────────────────────────────────────
    'Beta',
    'Vol/MarketCap (%)',

    # ── Valuation Multiples ──────────────────────────────────────────────
    'P/E Ratio',
    'Forward P/E',
    'PEG Ratio',
    'P/B Ratio',
    'P/S Ratio',
    'EV/EBITDA',
    'EV/Revenue',

    # ── Profitability ────────────────────────────────────────────────────
    'Gross Margin (%)',
    'Op. Margin (%)',
    'EBITDA Margin (%)',
    'Net Margin (%)',
    'ROE (%)',
    'ROA (%)',
    'ROIC (%)',

    # ── Growth ───────────────────────────────────────────────────────────
    'Revenue Growth (%)',
    'Earnings Growth (%)',
    'Revenue/MarketCap',

    # ── Balance Sheet Ratios ─────────────────────────────────────────────
    'Cash/MarketCap (%)',
    'Debt/MarketCap (%)',
    'Debt/Equity',
    'Current Ratio',
    'Quick Ratio',
    'Cash Conversion Ratio',

    # ── EPS ──────────────────────────────────────────────────────────────
    'EPS (TTM)',
    'EPS Forward',

    # ── Dividends ────────────────────────────────────────────────────────
    'Div. Yield (%)',
    'Payout Ratio (%)',

    # ── Analyst ──────────────────────────────────────────────────────────
    'Analyst Score',
    'Analyst Label',
    'Target Mean Price',
    'No. of Analysts',

    # ── Ownership ────────────────────────────────────────────────────────
    'Insiders (%)',
    'Institutions (%)',
]

# Keep only columns that actually exist in df
FINAL_COLS = [c for c in FINAL_COLS if c in df.columns]
out_df = df[FINAL_COLS].copy()

print(f"Output: {len(out_df)} rows, {len(out_df.columns)} columns")

# ── 4. Write Excel ────────────────────────────────────────────────────────────
TEXT_COLS  = {'Ticker', 'Company Name', 'Sector', 'Industry', 'Analyst Label'}
INT_COLS   = {'No. of Analysts'}

COL_WIDTHS = {
    'Ticker': 14, 'Company Name': 28, 'Sector': 18, 'Industry': 24,
    '52W High Norm': 14, '52W Low Norm': 13, '52W Change (%)': 14,
    'Beta': 8, 'Vol/MarketCap (%)': 16,
    'P/E Ratio': 10, 'Forward P/E': 11, 'PEG Ratio': 10,
    'P/B Ratio': 10, 'P/S Ratio': 10, 'EV/EBITDA': 11, 'EV/Revenue': 11,
    'Gross Margin (%)': 15, 'Op. Margin (%)': 14,
    'EBITDA Margin (%)': 16, 'Net Margin (%)': 13,
    'ROE (%)': 10, 'ROA (%)': 10, 'ROIC (%)': 10,
    'Revenue Growth (%)': 17, 'Earnings Growth (%)': 18,
    'Revenue/MarketCap': 16,
    'Cash/MarketCap (%)': 17, 'Debt/MarketCap (%)': 17,
    'Debt/Equity': 12, 'Current Ratio': 13, 'Quick Ratio': 11,
    'Cash Conversion Ratio': 20,
    'EPS (TTM)': 11, 'EPS Forward': 11,
    'Div. Yield (%)': 14, 'Payout Ratio (%)': 15,
    'Analyst Score': 13, 'Analyst Label': 14,
    'Target Mean Price': 16, 'No. of Analysts': 14,
    'Insiders (%)': 12, 'Institutions (%)': 16,
}

with pd.ExcelWriter(OUT, engine='xlsxwriter') as writer:
    out_df.to_excel(writer, index=False, sheet_name='Normalized Features')

    wb = writer.book
    ws = writer.sheets['Normalized Features']

    fmt_hdr  = wb.add_format({'bold': True, 'bg_color': '#D9E1F2',
                               'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fmt_num  = wb.add_format({'num_format': '#,##0.00'})
    fmt_int  = wb.add_format({'num_format': '#,##0'})
    fmt_pct  = wb.add_format({'num_format': '0.00'})
    fmt_text = wb.add_format({})

    # Section header colours (cosmetic band colouring)
    SECTION_COLORS = {
        'Price Position':    '#FFF2CC',
        'Market / Risk':     '#FCE4D6',
        'Valuation':         '#E2EFDA',
        'Profitability':     '#DDEEFF',
        'Growth':            '#EDE9F6',
        'Balance Sheet':     '#FDE9D9',
        'EPS':               '#E9F5E9',
        'Dividends':         '#FFF0F0',
        'Analyst':           '#F0F0FF',
        'Ownership':         '#F5F5F5',
    }

    col_section = {}
    for c in ['52W High Norm','52W Low Norm','52W Change (%)']: col_section[c] = 'Price Position'
    for c in ['Beta','Vol/MarketCap (%)']: col_section[c] = 'Market / Risk'
    for c in ['P/E Ratio','Forward P/E','PEG Ratio','P/B Ratio','P/S Ratio','EV/EBITDA','EV/Revenue']: col_section[c] = 'Valuation'
    for c in ['Gross Margin (%)','Op. Margin (%)','EBITDA Margin (%)','Net Margin (%)','ROE (%)','ROA (%)','ROIC (%)']: col_section[c] = 'Profitability'
    for c in ['Revenue Growth (%)','Earnings Growth (%)','Revenue/MarketCap']: col_section[c] = 'Growth'
    for c in ['Cash/MarketCap (%)','Debt/MarketCap (%)','Debt/Equity','Current Ratio','Quick Ratio','Cash Conversion Ratio']: col_section[c] = 'Balance Sheet'
    for c in ['EPS (TTM)','EPS Forward']: col_section[c] = 'EPS'
    for c in ['Div. Yield (%)','Payout Ratio (%)']: col_section[c] = 'Dividends'
    for c in ['Analyst Score','Analyst Label','Target Mean Price','No. of Analysts']: col_section[c] = 'Analyst'
    for c in ['Insiders (%)','Institutions (%)']: col_section[c] = 'Ownership'

    # Set column widths & formats
    for col_num, col_name in enumerate(out_df.columns):
        width = COL_WIDTHS.get(col_name, 12)
        if col_name in TEXT_COLS:
            fmt = fmt_text
        elif col_name in INT_COLS:
            fmt = fmt_int
        else:
            fmt = fmt_num
        ws.set_column(col_num, col_num, width, fmt)

    # Header row with section-based background colours
    ws.set_row(0, 36)
    for col_num, col_name in enumerate(out_df.columns):
        sec   = col_section.get(col_name, None)
        color = SECTION_COLORS.get(sec, '#D9E1F2') if sec else '#D9E1F2'
        hfmt  = wb.add_format({'bold': True, 'bg_color': color,
                                'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        ws.write(0, col_num, col_name, hfmt)

    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, len(out_df), len(out_df.columns) - 1)

print(f"Saved to: {OUT}")

# ── 5. Quick sanity check ─────────────────────────────────────────────────────
print("\n=== Sample (3 rows, key new columns) ===")
show = ['Ticker','52W High Norm','52W Low Norm','Revenue/MarketCap',
        'Cash/MarketCap (%)','Debt/MarketCap (%)','Cash Conversion Ratio',
        'Vol/MarketCap (%)','Analyst Score','Analyst Label']
show = [c for c in show if c in out_df.columns]
print(out_df[show].head(3).to_string(index=False))
