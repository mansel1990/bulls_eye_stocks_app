"""
Fundamental Score (0-100)
=========================
Uses the top 12 GAM features ranked by R².
Weights are proportional to R²; direction from Pearson correlation sign.

For each feature:
  - Winsorise at 2% to remove extremes
  - Rank-normalise to [0, 1]  (robust to outliers & non-linearity)
  - Flip direction if GAM correlation is negative (higher raw = worse)
  - Multiply by feature weight
  - Sum → rescale final score to [0, 100]

Output: fundamental_score_<date>.xlsx  (joined with normalized_features sheet)
"""

import os
import datetime
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
N_FILE = os.path.join(BASE, 'normalized_features_2026-02-19.xlsx')
TODAY  = datetime.date.today().strftime('%Y-%m-%d')
OUT    = os.path.join(BASE, f'fundamental_score_{TODAY}.xlsx')

# ── Top-12 features: (column_name, R², direction)  ────────────────────────────
# direction = +1 → higher raw value → better score
#            -1 → higher raw value → worse  score
TOP12 = [
    ('52W Change (%)',      0.2255, +1),   # #1  momentum
    ('52W Low Norm',        0.1969, -1),   # #2  low/price×100: high = near low = BAD
    ('P/B Ratio',           0.1328, +1),   # #3  higher book multiple = growth
    ('52W High Norm',       0.1305, +1),   # #4  price/high×100: high = near high = GOOD
    ('P/S Ratio',           0.0788, +1),   # #5  higher sales multiple = growth
    ('Revenue/MarketCap',   0.0731, -1),   # #6  cheap revenue = value trap
    ('Revenue Growth (%)',  0.0706, +1),   # #7  growing revenue = strength
    ('Forward P/E',         0.0643, +1),   # #8  growth expectations
    ('Div. Yield (%)',      0.0610, -1),   # #9  high yield = slow-growth defensive
    ('EV/Revenue',          0.0608, +1),   # #10 growth premium
    ('EPS (TTM)',           0.0593, +1),   # #11 earnings power
    ('Cash/MarketCap (%)',  0.0570, -1),   # #12 high cash ratio = underperformer
]

FEATURE_NAMES = [f for f, _, _ in TOP12]
R2_VALS       = np.array([r for _, r, _ in TOP12])
DIRECTIONS    = np.array([d for _, _, d in TOP12])

# Weights proportional to R²
WEIGHTS = R2_VALS / R2_VALS.sum()

print("Feature weights:")
for (feat, r2, direc), w in zip(TOP12, WEIGHTS):
    sign = '+' if direc == 1 else '-'
    print(f"  [{sign}] {feat:<25}  R²={r2:.4f}  weight={w:.4f}")
print(f"  Total weight = {WEIGHTS.sum():.4f}")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_excel(N_FILE)
print(f"\nLoaded {len(df)} tickers")

# ── Winsorise ─────────────────────────────────────────────────────────────────
WINSOR_PCT = 2

def winsorise(series, pct=WINSOR_PCT):
    clean = series.dropna()
    if len(clean) < 10:
        return series
    lo, hi = np.percentile(clean, [pct, 100 - pct])
    return series.clip(lo, hi)

work = df[['Ticker'] + FEATURE_NAMES].copy()
for col in FEATURE_NAMES:
    if col in work.columns:
        work[col] = winsorise(work[col])

# ── Rank-normalise each feature to [0, 1] ────────────────────────────────────
# Uses percentile rank: ties handled by 'average'; NaN rows stay NaN.
normed = pd.DataFrame({'Ticker': work['Ticker']})
for col in FEATURE_NAMES:
    if col not in work.columns:
        normed[col + '_n'] = np.nan
        continue
    ranked = work[col].rank(method='average', na_option='keep', pct=True)  # 0..1
    normed[col + '_n'] = ranked

# ── Apply direction: flip negatives so 1 always = best ───────────────────────
for i, (col, _, direc) in enumerate(TOP12):
    ncol = col + '_n'
    if direc == -1:
        normed[ncol] = 1.0 - normed[ncol]

# ── Weighted sum → rescale to [0, 100] ───────────────────────────────────────
# Handle NaN: use only available features per row, re-weight accordingly.
score_raw   = pd.Series(np.zeros(len(normed)), index=normed.index)
weight_used = pd.Series(np.zeros(len(normed)), index=normed.index)

for i, (col, _, _) in enumerate(TOP12):
    ncol = col + '_n'
    vals = normed[ncol]
    mask = vals.notna()
    score_raw[mask]   += vals[mask] * WEIGHTS[i]
    weight_used[mask] += WEIGHTS[i]

# Normalise by actual weight used (handles missing features per ticker)
fs = np.where(weight_used > 0, score_raw / weight_used, np.nan)
# Rescale to [0, 100]
fs_series = pd.Series(fs, index=normed.index)
lo, hi = fs_series.quantile(0.01), fs_series.quantile(0.99)
fs_scaled = ((fs_series - lo) / (hi - lo) * 100).clip(0, 100).round(2)

df['Fundamental Score'] = fs_scaled

# ── Count how many of the 12 features were available per ticker ───────────────
avail_count = pd.Series(0, index=normed.index)
for col in FEATURE_NAMES:
    ncol = col + '_n'
    avail_count += normed[ncol].notna().astype(int)
df['Features Used'] = avail_count

# ── Build output: Score + normalized sub-scores alongside original columns ────
# Add per-feature normed values for transparency
for col in FEATURE_NAMES:
    ncol = col + '_n'
    display_col = col + ' [0-1]'
    df[display_col] = normed[ncol].round(4)

# Reorder: identity + score + features-used + sub-scores + rest of sheet
id_cols    = ['Ticker', 'Company Name', 'Sector', 'Industry']
score_cols = ['Fundamental Score', 'Features Used']
sub_cols   = [col + ' [0-1]' for col in FEATURE_NAMES]
rest_cols  = [c for c in df.columns
              if c not in id_cols + score_cols + sub_cols + FEATURE_NAMES]

final_df = df[id_cols + score_cols + FEATURE_NAMES + sub_cols + rest_cols]
final_df = final_df.sort_values('Fundamental Score', ascending=False).reset_index(drop=True)

# ── Excel output ───────────────────────────────────────────────────────────────
with pd.ExcelWriter(OUT, engine='xlsxwriter') as writer:
    final_df.to_excel(writer, index=False, sheet_name='Fundamental Score')

    wb = writer.book
    ws = writer.sheets['Fundamental Score']

    # Formats
    fmt_hdr    = wb.add_format({'bold': True, 'bg_color': '#D9E1F2',
                                 'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fmt_score  = wb.add_format({'bold': True, 'num_format': '0.00',
                                 'bg_color': '#E2EFDA', 'border': 1})
    fmt_num    = wb.add_format({'num_format': '#,##0.00'})
    fmt_int    = wb.add_format({'num_format': '#,##0'})
    fmt_pct01  = wb.add_format({'num_format': '0.0000'})
    fmt_text   = wb.add_format({})

    # Colour scale for Fundamental Score column (col index 4)
    score_col_idx = final_df.columns.get_loc('Fundamental Score')
    ws.conditional_format(1, score_col_idx, len(final_df), score_col_idx, {
        'type':      '3_color_scale',
        'min_color': '#F8696B',   # red  = low
        'mid_color': '#FFEB84',   # amber = mid
        'max_color': '#63BE7B',   # green = high
    })

    # Colour scale for each sub-score [0-1] column
    for col in sub_cols:
        ci = final_df.columns.get_loc(col)
        ws.conditional_format(1, ci, len(final_df), ci, {
            'type': '2_color_scale',
            'min_color': '#FFC7CE',
            'max_color': '#C6EFCE',
        })

    # Column widths
    TEXT_COLS = {'Ticker', 'Company Name', 'Sector', 'Industry', 'Analyst Label'}
    INT_COLS  = {'Features Used', 'No. of Analysts'}
    for col_num, col_name in enumerate(final_df.columns):
        if col_name in TEXT_COLS:
            w, fmt = (28 if col_name == 'Company Name' else 14), fmt_text
        elif col_name in INT_COLS:
            w, fmt = 13, fmt_int
        elif col_name == 'Fundamental Score':
            w, fmt = 17, fmt_score
        elif col_name.endswith('[0-1]'):
            w, fmt = 13, fmt_pct01
        else:
            w, fmt = 13, fmt_num
        ws.set_column(col_num, col_num, w, fmt)

    # Header row
    ws.set_row(0, 36)
    for col_num, col_name in enumerate(final_df.columns):
        ws.write(0, col_num, col_name, fmt_hdr)

    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, len(final_df), len(final_df.columns) - 1)

print(f"\nSaved to: {OUT}")
print(f"Rows: {len(final_df)}  |  Columns: {len(final_df.columns)}")
print(f"\n=== Top 15 by Fundamental Score ===")
print(final_df[['Ticker','Company Name','Fundamental Score','Features Used']
               ].head(15).to_string(index=False))
print(f"\n=== Bottom 5 ===")
print(final_df[['Ticker','Company Name','Fundamental Score','Features Used']
               ].tail(5).to_string(index=False))
