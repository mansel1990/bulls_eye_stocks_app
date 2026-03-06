"""
Base Score Calculator
=====================
Implements the get_ultimate_score() logic for all NSE tickers.
Adapted for yfinance NSE data availability:
  - pegRatio       -> trailingPegRatio  (pegRatio always None for NSE)
  - ebit           -> operatingMargins * totalRevenue  (proxy)
  - interestExpense-> debtToEquity * totalRevenue * 0.05 (proxy: 5% interest on debt)
  - totalAssets    -> marketCap / priceToBook  (book value proxy)
  - totalLiab      -> debtToEquity * (marketCap / priceToBook) / 100  (proxy)
  - Z-Score uses available proxies

Output: base_score_<date>.xlsx  (saved to sheets/)
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
DAILY_DIR  = os.path.join(BASE, '..', '..', 'daily_data')
SHEETS_DIR = os.path.join(BASE, 'sheets')
os.makedirs(SHEETS_DIR, exist_ok=True)
TODAY      = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'base_score_{TODAY}.xlsx')
DELAY      = 0.5


def get_ultimate_score(ticker_symbol):
    try:
        info = yf.Ticker(ticker_symbol).info

        # ── 1. DATA EXTRACTION ────────────────────────────────────────────────
        pe     = info.get('trailingPE')     or 100
        peg    = info.get('trailingPegRatio') or 2.0   # NSE uses trailingPegRatio
        pb     = info.get('priceToBook')    or 10.0
        roe    = info.get('returnOnEquity') or 0.0
        margin = info.get('operatingMargins') or 0.0
        de_pct = info.get('debtToEquity')   or 500     # reported as % (e.g. 35 = 35%)
        de     = de_pct / 100                           # convert to decimal ratio
        cr     = info.get('currentRatio')   or 0.0
        rev    = info.get('totalRevenue')   or 1
        mcap   = info.get('marketCap')      or 1

        # EBIT proxy: operating margin × revenue
        ebit_proxy = margin * rev

        # Interest expense proxy: assume 5% interest rate on total debt
        # total debt ≈ D/E × book equity ≈ D/E × (marketCap / P/B)
        book_equity  = mcap / pb if pb and pb != 0 else mcap
        total_debt   = de * book_equity
        int_exp_proxy = total_debt * 0.05
        int_coverage  = (ebit_proxy / int_exp_proxy
                         if int_exp_proxy and int_exp_proxy != 0 else 10.0)
        int_coverage  = min(int_coverage, 50)  # cap at 50 to avoid inf

        # Total assets proxy: book equity + total debt
        assets_proxy = book_equity + total_debt
        # Total liabilities proxy: total debt (simplified)
        liab_proxy   = total_debt if total_debt > 0 else 1

        # Z-Score proxy: 3.3×(EBIT/Assets) + 1.0×(Rev/Assets) + 0.6×(MCap/Liab)
        z_a = ebit_proxy / assets_proxy if assets_proxy else 0
        z_b = rev        / assets_proxy if assets_proxy else 0
        z_c = mcap       / liab_proxy   if liab_proxy   else 0
        z_score = 3.3 * z_a + 1.0 * z_b + 0.6 * z_c

        # ── 2. SCORING ────────────────────────────────────────────────────────
        score = 0

        # Valuation (30 pts)
        val_score = 0
        if pe  < 20:  val_score += 10
        if peg < 1.2: val_score += 10
        if pb  < 3.0: val_score += 10
        score += val_score

        # Profitability (30 pts)
        prof_score = 0
        if roe    > 0.15: prof_score += 15
        if margin > 0.15: prof_score += 15
        score += prof_score

        # Financial Health (40 pts)
        health_score = 0
        if de           < 0.8: health_score += 15
        if int_coverage > 3.0: health_score += 15
        if cr           > 1.2: health_score += 10
        score += health_score

        # ── 3. PENALTIES ─────────────────────────────────────────────────────
        warnings_list = []
        z_penalty = 0
        if z_score < 1.81:
            warnings_list.append(f"ALARM: Z-Score={z_score:.2f} - Distress")
            z_penalty = -40
        elif z_score < 2.99:
            warnings_list.append(f"CAUTION: Z-Score={z_score:.2f} - Grey Zone")
            z_penalty = -10
        score += z_penalty

        lev_penalty = 0
        if de > 2.0:
            warnings_list.append(f"High Leverage D/E={de:.2f}")
            lev_penalty = -20
        score += lev_penalty

        cov_penalty = 0
        if int_coverage < 1.5:
            warnings_list.append(f"Weak Interest Coverage={int_coverage:.2f}")
            cov_penalty = -20
        score += cov_penalty

        final_score = max(0, min(score, 100))

        if final_score > 75:   status = 'Strong Buy'
        elif final_score > 60: status = 'Good'
        elif final_score > 40: status = 'Hold'
        else:                  status = 'AVOID'

        return {
            'Ticker':            ticker_symbol.replace('.NS', ''),
            'Base Score':        final_score,
            'Status':            status,
            'P/E':               round(pe,  2),
            'PEG':               round(peg, 2),
            'P/B':               round(pb,  2),
            'ROE (%)':           round(roe * 100, 2),
            'Op. Margin (%)':    round(margin * 100, 2),
            'D/E':               round(de,  2),
            'Current Ratio':     round(cr,  2),
            'Int. Coverage':     round(int_coverage, 2),
            'Z-Score (proxy)':   round(z_score, 2),
            'Valuation Pts':     val_score,
            'Profitability Pts': prof_score,
            'Health Pts':        health_score,
            'Z Penalty':         z_penalty,
            'Lev Penalty':       lev_penalty,
            'Cov Penalty':       cov_penalty,
            'Warnings':          ' | '.join(warnings_list) if warnings_list else '',
        }

    except Exception as e:
        return {
            'Ticker':  ticker_symbol.replace('.NS', ''),
            'Base Score': None,
            'Status':  f'ERROR: {e}',
            'Warnings': str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────────
files   = sorted(f for f in os.listdir(DAILY_DIR) if f.endswith('_daily.csv'))
tickers = [f.replace('_daily.csv', '') + '.NS' for f in files]
total   = len(tickers)
print(f"Calculating Base Score for {total} tickers...")

records = []
for i, symbol in enumerate(tickers):
    print(f"[{i+1}/{total}] {symbol} ... ", end="", flush=True)
    row = get_ultimate_score(symbol)
    records.append(row)
    score = row.get('Base Score')
    status = row.get('Status', '')
    print(f"{score}  {status}", flush=True)
    time.sleep(DELAY)

df = pd.DataFrame(records)
df = df.sort_values('Base Score', ascending=False, na_position='last').reset_index(drop=True)

# ── Excel output ───────────────────────────────────────────────────────────────
STATUS_COLORS = {
    'Strong Buy': {'bg_color': '#C6EFCE', 'font_color': '#276221'},
    'Good':       {'bg_color': '#FFEB9C', 'font_color': '#9C6500'},
    'Hold':       {'bg_color': '#BDD7EE', 'font_color': '#1F4E79'},
    'AVOID':      {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
}

COL_WIDTHS = {
    'Ticker': 14, 'Base Score': 12, 'Status': 12,
    'P/E': 9, 'PEG': 9, 'P/B': 9,
    'ROE (%)': 10, 'Op. Margin (%)': 15, 'D/E': 9,
    'Current Ratio': 14, 'Int. Coverage': 14, 'Z-Score (proxy)': 15,
    'Valuation Pts': 14, 'Profitability Pts': 16, 'Health Pts': 11,
    'Z Penalty': 11, 'Lev Penalty': 12, 'Cov Penalty': 12,
    'Warnings': 40,
}

with pd.ExcelWriter(OUT, engine='xlsxwriter',
                    engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
    df.to_excel(writer, index=False, sheet_name='Base Score')

    wb = writer.book
    ws = writer.sheets['Base Score']

    fmt_hdr  = wb.add_format({'bold': True, 'bg_color': '#D9E1F2',
                               'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fmt_num  = wb.add_format({'num_format': '#,##0.00'})
    fmt_int  = wb.add_format({'num_format': '#,##0'})
    fmt_text = wb.add_format({})

    int_cols = {'Valuation Pts','Profitability Pts','Health Pts',
                'Z Penalty','Lev Penalty','Cov Penalty','Base Score'}

    for col_num, col_name in enumerate(df.columns):
        w = COL_WIDTHS.get(col_name, 12)
        if col_name in {'Ticker', 'Status', 'Warnings'}:
            ws.set_column(col_num, col_num, w, fmt_text)
        elif col_name in int_cols:
            ws.set_column(col_num, col_num, w, fmt_int)
        else:
            ws.set_column(col_num, col_num, w, fmt_num)

    # Header
    ws.set_row(0, 30)
    for col_num, col_name in enumerate(df.columns):
        ws.write(0, col_num, col_name, fmt_hdr)

    # Colour Base Score column (red→amber→green)
    bs_idx = df.columns.get_loc('Base Score')
    ws.conditional_format(1, bs_idx, len(df), bs_idx, {
        'type': '3_color_scale',
        'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B',
    })

    # Colour Status column per value
    status_idx = df.columns.get_loc('Status')
    for row_num, status in enumerate(df['Status']):
        fmt_cfg = STATUS_COLORS.get(status)
        if fmt_cfg:
            fmt_s = wb.add_format({**fmt_cfg, 'bold': True})
            ws.write(row_num + 1, status_idx, status, fmt_s)

    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)

print(f"\nSaved to: {OUT}")
print(f"Rows: {len(df)}")
print("\n=== Score Distribution ===")
print(df['Status'].value_counts().to_string())
print("\n=== Top 10 ===")
print(df[['Ticker','Base Score','Status','Z-Score (proxy)','Warnings']].head(10).to_string(index=False))
