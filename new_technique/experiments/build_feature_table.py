"""
Feature Table Builder
=====================
Joins entry-filtered candlestick signals with ALL available features:

From base_score sheet:
  Base Score, Status, P/E, PEG, P/B, ROE (%), Op. Margin (%),
  D/E, Current Ratio, Int. Coverage, Z-Score (proxy),
  Valuation Pts, Profitability Pts, Health Pts

From normalized_features sheet:
  P/B Ratio, P/S Ratio, EV/EBITDA, EV/Revenue, Div. Yield (%),
  Revenue/MarketCap, Revenue Growth (%), Forward P/E, EPS (TTM),
  and all other fundamental columns

From daily price data (computed at signal date):
  MA44, MA44 Ascending (bool), Close > MA44,
  MA200, Close > MA200,
  52W High Norm, 52W Change (%), 52W Low Norm,
  RSI14,
  Volume TODAY > Volume Yesterday (bool),
  Market Cap (%)  [proxy: marketCap from normalized_features]

From signal / entry data (already in entry_filtered sheet):
  Ticker, Signal Date, Entry Date, Pattern, Strength, Entry Type,
  Sig OHLC, Entry Price, SL Price, Risk (%)

Target variables:
  1D Result, 1D Return (%), 3D Result, 3D Return (%), 5D Result, 5D Return (%)

Output: sheets/feature_table_<date>.xlsx
  Sheet 1: Feature Table (2024 Signals)
  Sheet 2: Feature Table (2025 Signals)
  Sheet 3: Column Guide (explains every column)
"""

import os, datetime, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
DAILY_DIR  = os.path.join(BASE, '..', '..', 'daily_data')
SHEETS_DIR = os.path.join(BASE, 'sheets')
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'feature_table_v2_{TODAY_STR}.xlsx')

SRC_SIGNALS  = os.path.join(SHEETS_DIR, 'entry_filtered_signals_v2_2026-03-05.xlsx')
SRC_BASE     = os.path.join(SHEETS_DIR, 'base_score_2026-02-19.xlsx')
SRC_NORM     = os.path.join(SHEETS_DIR, 'normalized_features_2026-02-19.xlsx')
SRC_FUND     = os.path.join(SHEETS_DIR, 'fundamental_score_2026-02-19.xlsx')
SRC_PERF     = os.path.join(SHEETS_DIR, 'performers_2024_2025.xlsx')

MA_SHORT  = 44
MA_LONG   = 200

# ── Load reference sheets ─────────────────────────────────────────────────────
print("Loading reference sheets...")

# Base Score (has P/E, PEG, P/B, ROE, Margin, D/E, Current Ratio, Int. Coverage, Z-Score, pts)
bs_df = pd.read_excel(SRC_BASE)
bs_df['Ticker'] = bs_df['Ticker'].astype(str).str.strip()
# rename to avoid collision with normalized_features
bs_df = bs_df.rename(columns={
    'P/E':            'BS P/E',
    'PEG':            'BS PEG',
    'P/B':            'BS P/B',
    'ROE (%)':        'BS ROE (%)',
    'Op. Margin (%)': 'BS Op Margin (%)',
    'D/E':            'BS D/E',
    'Current Ratio':  'BS Current Ratio',
    'Int. Coverage':  'BS Int Coverage',
    'Z-Score (proxy)':'BS Z-Score',
    'Valuation Pts':  'Valuation Pts',
    'Profitability Pts': 'Profitability Pts',
    'Health Pts':     'Health Pts',
    'Status':         'Base Status',
})
BS_KEEP = ['Ticker', 'Base Score', 'Base Status',
           'BS P/E', 'BS PEG', 'BS P/B', 'BS ROE (%)', 'BS Op Margin (%)',
           'BS D/E', 'BS Current Ratio', 'BS Int Coverage', 'BS Z-Score',
           'Valuation Pts', 'Profitability Pts', 'Health Pts']
bs_df = bs_df[[c for c in BS_KEEP if c in bs_df.columns]]

# Normalized features (richer fundamental data)
nf_df = pd.read_excel(SRC_NORM)
nf_df['Ticker'] = nf_df['Ticker'].astype(str).str.strip()
NF_KEEP = [
    'Ticker', 'Company Name', 'Sector', 'Industry',
    'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'P/B Ratio', 'P/S Ratio',
    'EV/EBITDA', 'EV/Revenue',
    'ROE (%)', 'Op. Margin (%)', 'Net Margin (%)',
    'Revenue Growth (%)', 'Earnings Growth (%)',
    'Revenue/MarketCap', 'Div. Yield (%)',
    'Debt/Equity', 'Current Ratio',
    'EPS (TTM)', 'EPS Forward',
    '52W High Norm', '52W Low Norm', '52W Change (%)',
]
nf_df = nf_df[[c for c in NF_KEEP if c in nf_df.columns]]

# Fundamental Score
fs_df = pd.read_excel(SRC_FUND, usecols=['Ticker', 'Fundamental Score'])
fs_df['Ticker'] = fs_df['Ticker'].astype(str).str.strip()

# Performer Score
pf_df = pd.read_excel(SRC_PERF, usecols=['Ticker', 'Score']).rename(
    columns={'Score': 'Performer Score'})
pf_df['Ticker'] = pf_df['Ticker'].astype(str).str.strip()

print(f"  Base Score rows: {len(bs_df)}")
print(f"  Norm Features rows: {len(nf_df)}")

# ── Load signals ──────────────────────────────────────────────────────────────
print("Loading signals...")
sig_2024 = pd.read_excel(SRC_SIGNALS, sheet_name='2024 Signals')
sig_2025 = pd.read_excel(SRC_SIGNALS, sheet_name='2025 Signals')
sig_2024['Year'] = '2024'
sig_2025['Year'] = '2025'
signals = pd.concat([sig_2024, sig_2025], ignore_index=True)
signals['Ticker'] = signals['Ticker'].astype(str).str.strip()
signals['Signal Date'] = pd.to_datetime(signals['Signal Date'])
print(f"  Total signals: {len(signals):,}")

# ── Load all daily price data into memory ─────────────────────────────────────
print("Loading daily price data for MA / volume calculations...")
price_cache = {}
files = sorted(f for f in os.listdir(DAILY_DIR) if f.endswith('_daily.csv'))
for fname in files:
    ticker = fname.replace('_daily.csv', '')
    try:
        df = pd.read_csv(os.path.join(DAILY_DIR, fname),
                         index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)
        for col in ['Open','High','Low','Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df.get('Volume', np.nan), errors='coerce')
        price_cache[ticker] = df
    except Exception:
        pass
print(f"  Loaded {len(price_cache)} price series")

# ── Compute technical features at signal date ─────────────────────────────────
def get_technical_features(ticker, signal_date):
    """
    Given ticker + signal date, compute MA/volume/52W features
    using all data UP TO AND INCLUDING signal_date.
    """
    result = {}
    df = price_cache.get(ticker)
    if df is None or len(df) == 0:
        return result

    # Slice up to signal date
    hist = df[df.index <= signal_date].copy()
    if len(hist) < 5:
        return result

    close  = hist['Close']
    volume = hist['Volume']

    # --- MA44 ---
    if len(hist) >= MA_SHORT:
        ma44_series    = close.rolling(MA_SHORT).mean()
        ma44_val       = ma44_series.iloc[-1]
        result['MA44'] = round(ma44_val, 2) if not np.isnan(ma44_val) else None

        # MA44 Ascending: is the MA44 higher now than N bars ago?
        # True if last value > value from up to 5 bars back (any 2 of last 5 ascending)
        recent_ma44 = ma44_series.dropna().tail(6)
        if len(recent_ma44) >= 2:
            result['MA44 Ascending'] = bool(recent_ma44.iloc[-1] > recent_ma44.iloc[0])
        else:
            result['MA44 Ascending'] = None

        # Close > MA44
        result['Close > MA44'] = bool(close.iloc[-1] > ma44_val) if not np.isnan(ma44_val) else None
    else:
        result['MA44'] = None
        result['MA44 Ascending'] = None
        result['Close > MA44'] = None

    # --- MA200 ---
    if len(hist) >= MA_LONG:
        ma200_val      = close.rolling(MA_LONG).mean().iloc[-1]
        result['MA200'] = round(ma200_val, 2) if not np.isnan(ma200_val) else None
        result['Close > MA200'] = bool(close.iloc[-1] > ma200_val) if not np.isnan(ma200_val) else None
    else:
        result['MA200'] = None
        result['Close > MA200'] = None

    # --- RSI14 (recompute from price) ---
    if len(hist) >= 15:
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi_val = (100 - 100 / (1 + rs)).iloc[-1]
        result['RSI14'] = round(rsi_val, 2) if not np.isnan(rsi_val) else None
    else:
        result['RSI14'] = None

    # --- MACD (12/26/9) ---
    if len(hist) >= 35:
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - macd_signal
        ml  = macd_line.iloc[-1]
        ms  = macd_signal.iloc[-1]
        mh  = macd_hist.iloc[-1]
        result['MACD Line']   = round(ml, 4) if not np.isnan(ml) else None
        result['MACD Signal'] = round(ms, 4) if not np.isnan(ms) else None
        result['MACD Hist']   = round(mh, 4) if not np.isnan(mh) else None
        result['MACD > Signal'] = bool(ml > ms) if (not np.isnan(ml) and not np.isnan(ms)) else None
    else:
        result['MACD Line']   = None
        result['MACD Signal'] = None
        result['MACD Hist']   = None
        result['MACD > Signal'] = None

    # --- Volume: Today > Yesterday ---
    if len(volume.dropna()) >= 2:
        vol_today = volume.iloc[-1]
        vol_yest  = volume.iloc[-2]
        result['Vol Today > Yesterday'] = bool(vol_today > vol_yest) if (
            not np.isnan(vol_today) and not np.isnan(vol_yest)) else None
        result['Vol Today']     = int(vol_today) if not np.isnan(vol_today) else None
        result['Vol Yesterday'] = int(vol_yest)  if not np.isnan(vol_yest)  else None
    else:
        result['Vol Today > Yesterday'] = None
        result['Vol Today']     = None
        result['Vol Yesterday'] = None

    # --- 52-week range (computed from price, not normalized_features) ---
    hist_52 = df[df.index <= signal_date].tail(252)
    if len(hist_52) >= 20:
        hi52  = hist_52['High'].max()
        lo52  = hist_52['Low'].min()
        cur   = close.iloc[-1]
        if hi52 > 0:
            result['52W High Norm (price)'] = round(cur / hi52 * 100, 2)
        else:
            result['52W High Norm (price)'] = None
        if cur > 0:
            result['52W Low Norm (price)']  = round(lo52 / cur * 100, 2)
        else:
            result['52W Low Norm (price)']  = None
        price_52w_ago = hist_52['Close'].iloc[0]
        if price_52w_ago and price_52w_ago != 0:
            result['52W Change % (price)'] = round((cur - price_52w_ago) / price_52w_ago * 100, 2)
        else:
            result['52W Change % (price)'] = None
    else:
        result['52W High Norm (price)'] = None
        result['52W Low Norm (price)']  = None
        result['52W Change % (price)']  = None

    return result

# ── Build feature rows ────────────────────────────────────────────────────────
print("Computing technical features per signal (this may take 1-2 mins)...")
tech_rows = []
total = len(signals)
for i, row in signals.iterrows():
    if (i + 1) % 5000 == 0:
        print(f"  {i+1:,}/{total:,}")
    feats = get_technical_features(row['Ticker'], row['Signal Date'])
    tech_rows.append(feats)

tech_df = pd.DataFrame(tech_rows)

# ── Merge everything ──────────────────────────────────────────────────────────
print("Merging all features...")
result = signals.copy()

# Merge technical features (computed per-row, no join needed)
result = pd.concat([result.reset_index(drop=True),
                    tech_df.reset_index(drop=True)], axis=1)

# Merge fundamental sheets on Ticker
result = result.merge(bs_df,  on='Ticker', how='left', suffixes=('', '_bs'))
result = result.merge(nf_df,  on='Ticker', how='left', suffixes=('', '_nf'))
result = result.merge(fs_df,  on='Ticker', how='left', suffixes=('', '_fs'))
result = result.merge(pf_df,  on='Ticker', how='left', suffixes=('', '_pf'))

# Drop any duplicate score columns from entry_filtered (already present)
for col in ['Fundamental Score_fs', 'Performer Score_pf', 'Base Score_bs', 'Base Status_bs']:
    if col in result.columns:
        result.drop(columns=col, inplace=True)

# ── Column ordering ────────────────────────────────────────────────────────────
# Identity + Signal info
ID_COLS = ['Ticker', 'Company Name', 'Sector', 'Industry',
           'Signal Date', 'Entry Date', 'Year', 'Pattern', 'Strength', 'Entry Type']

# Price at signal
PRICE_COLS = ['Sig Open', 'Sig High', 'Sig Low', 'Sig Close',
              'Entry Price', 'SL Price', 'Risk (%)']

# Technical (computed from daily prices at signal date)
TECH_COLS = [
    'MA44', 'MA44 Ascending', 'Close > MA44',
    'MA200', 'Close > MA200',
    'RSI14',
    '52W High Norm (price)', '52W Low Norm (price)', '52W Change % (price)',
    'Vol Today', 'Vol Yesterday', 'Vol Today > Yesterday',
    'MACD Line', 'MACD Signal', 'MACD Hist', 'MACD > Signal',
]

# Base Score fundamentals
BASE_COLS = [
    'Base Score', 'Base Status',
    'BS P/E', 'BS PEG', 'BS P/B', 'BS ROE (%)', 'BS Op Margin (%)',
    'BS D/E', 'BS Current Ratio', 'BS Int Coverage', 'BS Z-Score',
    'Valuation Pts', 'Profitability Pts', 'Health Pts',
]

# Normalized / Fundamental Score features
FUND_COLS = [
    'Fundamental Score', 'Performer Score',
    'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'P/B Ratio', 'P/S Ratio',
    'EV/EBITDA', 'EV/Revenue',
    'ROE (%)', 'Op. Margin (%)', 'Net Margin (%)',
    'Revenue Growth (%)', 'Earnings Growth (%)',
    'Revenue/MarketCap', 'Div. Yield (%)',
    'Debt/Equity', 'Current Ratio',
    'EPS (TTM)', 'EPS Forward',
    '52W High Norm', '52W Low Norm', '52W Change (%)',   # from normalized_features (static)
]

# Targets
TARGET_COLS = [
    '1D Result', '1D Return (%)',
    '3D Result', '3D Return (%)',
    '5D Result', '5D Return (%)',
]

# Build final column order (only keep cols that exist)
def keep(cols):
    return [c for c in cols if c in result.columns]

ordered_cols = (keep(ID_COLS) + keep(PRICE_COLS) + keep(TECH_COLS) +
                keep(BASE_COLS) + keep(FUND_COLS) + keep(TARGET_COLS))

# Add any leftover columns not explicitly listed
all_listed = set(ordered_cols)
extra = [c for c in result.columns if c not in all_listed]
ordered_cols = ordered_cols + extra

result = result[ordered_cols]

# ── Stats ──────────────────────────────────────────────────────────────────────
print(f"\nFeature table: {result.shape}")
print(f"  Columns: {len(result.columns)}")
print(f"  2024 rows: {(result['Year']=='2024').sum():,}")
print(f"  2025 rows: {(result['Year']=='2025').sum():,}")
print(f"\nSample columns: {list(result.columns[:20])}")

# Null rate per key column (quality check)
print("\nNull rates for key columns:")
check_cols = ['MA44','MA44 Ascending','Close > MA44','MA200','RSI14',
              'MACD Line','MACD Signal','MACD > Signal',
              'Vol Today > Yesterday','Base Score','Fundamental Score',
              'Performer Score','P/B Ratio','EV/EBITDA']
for c in check_cols:
    if c in result.columns:
        null_pct = result[c].isna().mean() * 100
        print(f"  {c:<30} {null_pct:.1f}% null")

# ── Column guide ───────────────────────────────────────────────────────────────
GUIDE = [
    # Identity
    ('Ticker',              'Identity',  'NSE ticker symbol'),
    ('Company Name',        'Identity',  'Company full name'),
    ('Sector',              'Identity',  'Sector (from yfinance)'),
    ('Industry',            'Identity',  'Industry (from yfinance)'),
    ('Signal Date',         'Signal',    'Date the candlestick pattern completed'),
    ('Entry Date',          'Signal',    'Date entry order was triggered (next 1-3 bars)'),
    ('Year',                'Signal',    '2024 or 2025'),
    ('Pattern',             'Signal',    'Candlestick pattern name'),
    ('Strength',            'Signal',    'Bullish or Strongly Bullish'),
    ('Entry Type',          'Signal',    'stop = buy-stop above high | limit = buy-limit at mid | market_next_open = next bar open'),
    # Price
    ('Sig Open',            'Price',     'Signal bar open price'),
    ('Sig High',            'Price',     'Signal bar high price'),
    ('Sig Low',             'Price',     'Signal bar low price'),
    ('Sig Close',           'Price',     'Signal bar close price'),
    ('Entry Price',         'Price',     'Actual price at which entry was triggered'),
    ('SL Price',            'Price',     'Stop-loss reference price (below signal bar low)'),
    ('Risk (%)',             'Price',     '(Entry - SL) / Entry * 100 — % loss if stopped out'),
    # Technical
    ('MA44',                'Technical', '44-day simple moving average at signal date'),
    ('MA44 Ascending',      'Technical', 'TRUE if MA44 today > MA44 from 5 bars ago (trending up)'),
    ('Close > MA44',        'Technical', 'TRUE if close price is above MA44'),
    ('MA200',               'Technical', '200-day simple moving average at signal date'),
    ('Close > MA200',       'Technical', 'TRUE if close price is above MA200'),
    ('RSI14',               'Technical', 'RSI(14) computed from daily closes up to signal date'),
    ('52W High Norm (price)','Technical','Current price as % of 52-week high (100=at high, 50=halfway)'),
    ('52W Low Norm (price)', 'Technical','52-week low as % of current price (100=at low, 50=halfway)'),
    ('52W Change % (price)', 'Technical','% change in close over last 252 trading days'),
    ('Vol Today',           'Technical', 'Volume on signal date'),
    ('Vol Yesterday',       'Technical', 'Volume on day before signal date'),
    ('Vol Today > Yesterday','Technical','TRUE if today volume > yesterday volume'),
    ('MACD Line',           'Technical', 'MACD line = EMA(12) - EMA(26) at signal date'),
    ('MACD Signal',         'Technical', 'MACD signal line = EMA(9) of MACD line'),
    ('MACD Hist',           'Technical', 'MACD histogram = MACD Line - Signal (positive = bullish momentum)'),
    ('MACD > Signal',       'Technical', 'TRUE if MACD Line is above Signal line (bullish crossover)'),
    # Base Score
    ('Base Score',          'Base Score','Composite financial health score 0-100'),
    ('Base Status',         'Base Score','Strong Buy / Good / Hold / AVOID'),
    ('BS P/E',              'Base Score','Trailing P/E from yfinance'),
    ('BS PEG',              'Base Score','PEG ratio (trailing)'),
    ('BS P/B',              'Base Score','Price-to-book ratio'),
    ('BS ROE (%)',          'Base Score','Return on equity %'),
    ('BS Op Margin (%)',    'Base Score','Operating margin %'),
    ('BS D/E',              'Base Score','Debt-to-equity ratio'),
    ('BS Current Ratio',    'Base Score','Current assets / current liabilities'),
    ('BS Int Coverage',     'Base Score','EBIT / interest expense (proxy)'),
    ('BS Z-Score',          'Base Score','Altman Z-Score proxy (>2.99=safe, 1.81-2.99=grey, <1.81=distress)'),
    ('Valuation Pts',       'Base Score','Points from valuation sub-score (max 30)'),
    ('Profitability Pts',   'Base Score','Points from profitability sub-score (max 30)'),
    ('Health Pts',          'Base Score','Points from financial health sub-score (max 40)'),
    # Fundamental Score
    ('Fundamental Score',   'Fund Score','Rank-weighted composite score 0-100 (12 features)'),
    ('Performer Score',     'Fund Score','Historical performer rank score'),
    ('P/E Ratio',           'Fund Score','Trailing P/E from normalized_features'),
    ('Forward P/E',         'Fund Score','Forward P/E (growth expectation)'),
    ('PEG Ratio',           'Fund Score','PEG ratio'),
    ('P/B Ratio',           'Fund Score','Price-to-book'),
    ('P/S Ratio',           'Fund Score','Price-to-sales'),
    ('EV/EBITDA',           'Fund Score','Enterprise value / EBITDA'),
    ('EV/Revenue',          'Fund Score','Enterprise value / revenue'),
    ('ROE (%)',              'Fund Score','Return on equity %'),
    ('Op. Margin (%)',       'Fund Score','Operating margin %'),
    ('Net Margin (%)',       'Fund Score','Net profit margin %'),
    ('Revenue Growth (%)',   'Fund Score','YoY revenue growth %'),
    ('Earnings Growth (%)',  'Fund Score','YoY earnings growth %'),
    ('Revenue/MarketCap',    'Fund Score','Revenue / market cap (value indicator)'),
    ('Div. Yield (%)',       'Fund Score','Dividend yield %'),
    ('Debt/Equity',          'Fund Score','Debt to equity'),
    ('Current Ratio',        'Fund Score','Current ratio'),
    ('EPS (TTM)',             'Fund Score','Trailing twelve month EPS'),
    ('EPS Forward',          'Fund Score','Forward EPS estimate'),
    ('52W High Norm',        'Fund Score','Price / 52W high * 100 (static, from normalized_features)'),
    ('52W Low Norm',         'Fund Score','52W low / price * 100 (static)'),
    ('52W Change (%)',       'Fund Score','52W price change % (static)'),
    # Targets
    ('1D Result',           'Target',    'Profit or Loss — 1 day after entry'),
    ('1D Return (%)',        'Target',    '% return 1 day after entry price'),
    ('3D Result',           'Target',    'Profit or Loss — 3 days after entry'),
    ('3D Return (%)',        'Target',    '% return 3 days after entry price'),
    ('5D Result',           'Target',    'Profit or Loss — 5 days after entry'),
    ('5D Return (%)',        'Target',    '% return 5 days after entry price'),
]
guide_df = pd.DataFrame(GUIDE, columns=['Column', 'Category', 'Description'])

# ── Excel output ──────────────────────────────────────────────────────────────
print(f"\nWriting Excel: {OUT}")

df_2024 = result[result['Year'] == '2024'].drop(columns='Year').reset_index(drop=True)
df_2025 = result[result['Year'] == '2025'].drop(columns='Year').reset_index(drop=True)

PROFIT_COLOR = '#C6EFCE'
LOSS_COLOR   = '#FFC7CE'

def write_feature_sheet(writer, df_sh, sheet_name, wb):
    df_sh.to_excel(writer, index=False, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]

    fmt_hdr   = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white',
                                'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fmt_num   = wb.add_format({'num_format': '#,##0.00'})
    fmt_int   = wb.add_format({'num_format': '#,##0'})
    fmt_bool_t= wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221', 'bold': True})
    fmt_bool_f= wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    fmt_profit= wb.add_format({'bg_color': PROFIT_COLOR, 'font_color': '#276221',
                                'num_format': '+0.00;-0.00'})
    fmt_loss  = wb.add_format({'bg_color': LOSS_COLOR,   'font_color': '#9C0006',
                                'num_format': '+0.00;-0.00'})
    fmt_lbl_p = wb.add_format({'bg_color': PROFIT_COLOR, 'font_color': '#276221'})
    fmt_lbl_l = wb.add_format({'bg_color': LOSS_COLOR,   'font_color': '#9C0006'})
    fmt_strong= wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221', 'bold': True})
    fmt_bull  = wb.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})

    COL_W = {
        'Ticker': 12, 'Company Name': 28, 'Sector': 18, 'Industry': 22,
        'Signal Date': 13, 'Entry Date': 13, 'Pattern': 24, 'Strength': 18,
        'Entry Type': 18,
        'Sig Open': 10, 'Sig High': 10, 'Sig Low': 10, 'Sig Close': 10,
        'Entry Price': 12, 'SL Price': 10, 'Risk (%)': 9,
        'MA44': 10, 'MA44 Ascending': 15, 'Close > MA44': 13,
        'MA200': 10, 'Close > MA200': 14,
        'RSI14': 8,
        '52W High Norm (price)': 18, '52W Low Norm (price)': 17, '52W Change % (price)': 18,
        'Vol Today': 12, 'Vol Yesterday': 13, 'Vol Today > Yesterday': 18,
        'MACD Line': 12, 'MACD Signal': 13, 'MACD Hist': 11, 'MACD > Signal': 14,
        'Base Score': 12, 'Base Status': 13,
        'BS P/E': 9, 'BS PEG': 9, 'BS P/B': 9,
        'Valuation Pts': 14, 'Profitability Pts': 16, 'Health Pts': 11,
        'Fundamental Score': 18, 'Performer Score': 16,
        '1D Return (%)': 13, '1D Result': 10,
        '3D Return (%)': 13, '3D Result': 10,
        '5D Return (%)': 13, '5D Result': 10,
    }

    BOOL_COLS   = {'MA44 Ascending', 'Close > MA44', 'Close > MA200',
                   'Vol Today > Yesterday', 'MACD > Signal'}
    INT_COLS    = {'Vol Today', 'Vol Yesterday', 'Valuation Pts', 'Profitability Pts', 'Health Pts'}
    SCORE_COLS  = {'Base Score', 'Fundamental Score', 'Performer Score'}

    for ci, cn in enumerate(df_sh.columns):
        w = COL_W.get(cn, max(len(cn) + 2, 10))
        ws.set_column(ci, ci, w, fmt_int if cn in INT_COLS else fmt_num)
        ws.write(0, ci, cn, fmt_hdr)
    ws.set_row(0, 32)

    # Score colour scales
    for sc in SCORE_COLS:
        if sc in df_sh.columns:
            ci = df_sh.columns.get_loc(sc)
            ws.conditional_format(1, ci, len(df_sh), ci, {
                'type': '3_color_scale',
                'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B',
            })

    # Base Status colours
    BASE_COLORS = {
        'Strong Buy': {'bg_color': '#C6EFCE', 'font_color': '#276221'},
        'Good':       {'bg_color': '#FFEB9C', 'font_color': '#9C6500'},
        'Hold':       {'bg_color': '#BDD7EE', 'font_color': '#1F4E79'},
        'AVOID':      {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
    }

    # Per-row coloring (bool, strength, result, base status)
    strength_ci = df_sh.columns.get_loc('Strength') if 'Strength' in df_sh.columns else None
    bs_ci       = df_sh.columns.get_loc('Base Status') if 'Base Status' in df_sh.columns else None

    result_pairs = []
    for ret_col, lbl_col in [('1D Return (%)','1D Result'),
                              ('3D Return (%)','3D Result'),
                              ('5D Return (%)','5D Result')]:
        if ret_col in df_sh.columns and lbl_col in df_sh.columns:
            result_pairs.append((df_sh.columns.get_loc(ret_col),
                                  df_sh.columns.get_loc(lbl_col),
                                  ret_col, lbl_col))

    bool_ci = {cn: df_sh.columns.get_loc(cn)
               for cn in BOOL_COLS if cn in df_sh.columns}

    for ri, row in df_sh.iterrows():
        er = ri + 1

        # Strength
        if strength_ci is not None:
            s = row.get('Strength', '')
            ws.write(er, strength_ci, s,
                     fmt_strong if s == 'Strongly Bullish' else fmt_bull)

        # Boolean columns
        for cn, ci in bool_ci.items():
            v = row[cn]
            if v is True or v == 'True' or v == 1:
                ws.write(er, ci, 'TRUE',  fmt_bool_t)
            elif v is False or v == 'False' or v == 0:
                ws.write(er, ci, 'FALSE', fmt_bool_f)

        # Result / return pairs
        for pct_ci, lbl_ci, pct_col, lbl_col in result_pairs:
            lbl = row[lbl_col]
            pct = row[pct_col]
            f_l = fmt_lbl_p if lbl == 'Profit' else fmt_lbl_l
            f_p = fmt_profit if lbl == 'Profit' else fmt_loss
            ws.write(er, lbl_ci, lbl if pd.notna(lbl) else '', f_l if pd.notna(lbl) else fmt_num)
            if pd.notna(pct):
                ws.write(er, pct_ci, pct, f_p)

        # Base Status
        if bs_ci is not None:
            status = row.get('Base Status', '')
            cfg = BASE_COLORS.get(status)
            if cfg:
                ws.write(er, bs_ci, status, wb.add_format({**cfg, 'bold': True}))

    ws.freeze_panes(1, 4)
    ws.autofilter(0, 0, len(df_sh), len(df_sh.columns) - 1)


def write_guide_sheet(writer, guide, wb):
    guide.to_excel(writer, index=False, sheet_name='Column Guide')
    ws = writer.sheets['Column Guide']
    fmt_hdr = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white',
                              'border': 1})
    cat_colors = {
        'Identity':   '#D9E1F2',
        'Signal':     '#DDEBF7',
        'Price':      '#FFF2CC',
        'Technical':  '#E2EFDA',
        'Base Score': '#FCE4D6',
        'Fund Score': '#EAF0FB',
        'Target':     '#C6EFCE',
    }
    fmt_cat = {k: wb.add_format({'bg_color': v, 'bold': True}) for k, v in cat_colors.items()}
    fmt_desc = wb.add_format({'text_wrap': True})

    ws.set_column(0, 0, 28); ws.set_column(1, 1, 14); ws.set_column(2, 2, 55, fmt_desc)
    ws.set_row(0, 24)
    for ci, cn in enumerate(guide.columns):
        ws.write(0, ci, cn, fmt_hdr)
    for ri, row in guide.iterrows():
        cat = row['Category']
        f   = fmt_cat.get(cat, wb.add_format({}))
        ws.write(ri + 1, 0, row['Column'], f)
        ws.write(ri + 1, 1, cat, f)
        ws.write(ri + 1, 2, row['Description'], fmt_desc)
        ws.set_row(ri + 1, 18)


with pd.ExcelWriter(OUT, engine='xlsxwriter',
                    engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
    wb = writer.book
    write_feature_sheet(writer, df_2024, '2024 Feature Table', wb)
    write_feature_sheet(writer, df_2025, '2025 Feature Table', wb)
    write_guide_sheet(writer, guide_df, wb)

print(f"\nSaved: {OUT}")
print(f"  2024 rows: {len(df_2024):,}  |  2025 rows: {len(df_2025):,}")
print(f"  Total columns: {len(result.columns)}")
print(f"\nColumn list:")
for i, c in enumerate(result.columns):
    print(f"  {i+1:>2}. {c}")
