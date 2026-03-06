"""
lstm_5d_eval.py
---------------
Load the trained LSTM model from lstm_5d_predictor.py and evaluate it on
ALL available tickers (2025+ test data). Outputs a ranked Excel report.

Usage:
    python lstm_5d_eval.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, datetime, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN      = 30
HORIZON      = 5
THRESH_UP    = 1.0
THRESH_DN    = -1.0
TRAIN_CUTOFF = pd.Timestamp('2025-01-01')
BATCH_SIZE   = 512
MIN_TEST_SEQ = 20     # skip tickers with fewer test sequences

BASE       = os.path.dirname(__file__)
SHEETS_DIR = os.path.join(BASE, 'sheets')
MODELS_DIR = os.path.join(BASE, 'models')
DAILY_DIR  = os.path.normpath(os.path.join(os.path.dirname(BASE), '..', 'daily_data'))
MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_5d.pth')
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'lstm_5d_all_tickers_{TODAY_STR}.xlsx')

# ── Model (must match lstm_5d_predictor.py architecture) ─────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, n_classes=2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc    = nn.Linear(32, n_classes)

    def forward(self, x):
        x, _      = self.lstm1(x)
        x         = self.drop1(x)
        _, (h, _) = self.lstm2(x)
        return self.fc(self.drop2(h[-1]))

class SeqDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]

# ── Indicators ────────────────────────────────────────────────────────────────
def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(close):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal

# ── Per-ticker data builder ───────────────────────────────────────────────────
def build_ticker_data(ticker):
    path = os.path.join(DAILY_DIR, f'{ticker}_daily.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=True).sort_index()
        if len(df) < SEQ_LEN + HORIZON + 50:
            return None

        macd_l, macd_s = compute_macd(df['Close'])
        rsi  = compute_rsi(df['Close'])
        ret1 = df['Close'].pct_change(1) * 100
        ret5 = df['Close'].pct_change(5) * 100

        feat = pd.DataFrame({
            'Open':   df['Open'],   'High':   df['High'],
            'Low':    df['Low'],    'Close':  df['Close'],
            'Volume': df['Volume'], 'MACD_l': macd_l,
            'MACD_s': macd_s,       'RSI':    rsi,
            'Ret1':   ret1,         'Ret5':   ret5,
        }).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        fwd5  = (df['Close'].shift(-HORIZON) - df['Close']) / df['Close'] * 100
        label = np.full(len(df), np.nan)
        label[fwd5 > THRESH_UP]  = 0   # Up
        label[fwd5 < THRESH_DN]  = 1   # Down

        train_mask = df.index < TRAIN_CUTOFF
        if train_mask.sum() < SEQ_LEN + 20:
            return None

        scaler = MinMaxScaler()
        scaler.fit(feat[train_mask].values)
        f_scaled = scaler.transform(feat.values)

        X, y, dates = [], [], []
        for i in range(SEQ_LEN - 1, len(df) - HORIZON):
            lbl = label[i]
            if np.isnan(lbl): continue
            if df.index[i] < TRAIN_CUTOFF: continue   # test set only
            X.append(f_scaled[i - SEQ_LEN + 1 : i + 1])
            y.append(int(lbl))
            dates.append(df.index[i])

        if len(X) < MIN_TEST_SEQ:
            return None

        return ticker, np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(dates)
    except Exception:
        return None

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run lstm_5d_predictor.py first to train and save the model.")
    sys.exit(1)

ckpt       = torch.load(MODEL_PATH, map_location='cpu')
n_features = ckpt['n_features']
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMClassifier(n_features=n_features).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"Loaded model from {MODEL_PATH}  (n_features={n_features})")
print(f"Using device: {device}")

# ── Discover & process all tickers ───────────────────────────────────────────
all_files   = sorted(os.listdir(DAILY_DIR))
all_tickers = [f.replace('_daily.csv', '') for f in all_files if f.endswith('_daily.csv')]
print(f"\nBuilding test sequences for {len(all_tickers)} tickers...")

results = Parallel(n_jobs=-1, prefer='threads')(
    delayed(build_ticker_data)(t) for t in all_tickers
)
results = [r for r in results if r is not None]
print(f"Valid tickers (≥{MIN_TEST_SEQ} test sequences): {len(results)}")

# ── Evaluate each ticker ──────────────────────────────────────────────────────
print(f"\nEvaluating...")
rows      = []
all_preds = []   # for aggregate stats

for ticker, X, y, dates in results:
    loader = DataLoader(SeqDataset(X), batch_size=BATCH_SIZE)
    probs  = []
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    preds = np.argmax(probs, axis=1)

    up_p  = precision_score(y, preds, pos_label=0, average='binary', zero_division=0)
    dn_p  = precision_score(y, preds, pos_label=1, average='binary', zero_division=0)
    up_r  = recall_score(y, preds, pos_label=0, average='binary', zero_division=0)
    dn_r  = recall_score(y, preds, pos_label=1, average='binary', zero_division=0)
    acc   = accuracy_score(y, preds)
    base  = max(np.bincount(y)) / len(y)
    n_up  = int((y == 0).sum())
    n_dn  = int((y == 1).sum())

    rows.append({
        'Ticker':         ticker,
        'N_Test':         len(y),
        'N_Up_Actual':    n_up,
        'N_Down_Actual':  n_dn,
        'N_Up_Pred':      int((preds == 0).sum()),
        'N_Down_Pred':    int((preds == 1).sum()),
        'UP_Precision':   round(up_p, 4),
        'UP_Recall':      round(up_r, 4),
        'DOWN_Precision': round(dn_p, 4),
        'DOWN_Recall':    round(dn_r, 4),
        'Accuracy':       round(acc, 4),
        'Baseline':       round(base, 4),
        'Edge_UP':        round(up_p - base, 4),
        'Edge_DOWN':      round(dn_p - base, 4),
    })
    all_preds.append((ticker, y, preds, probs, dates))

df_results = pd.DataFrame(rows)

# ── Aggregate stats ───────────────────────────────────────────────────────────
print(f"\n── Aggregate across {len(df_results)} tickers ─────────────────────────────")
print(f"  Median UP   Precision : {df_results['UP_Precision'].median():.1%}")
print(f"  Median DOWN Precision : {df_results['DOWN_Precision'].median():.1%}")
print(f"  Median Accuracy       : {df_results['Accuracy'].median():.1%}")
print(f"  Tickers UP_Prec ≥ 60% : {(df_results['UP_Precision'] >= 0.60).sum()}")
print(f"  Tickers UP_Prec ≥ 65% : {(df_results['UP_Precision'] >= 0.65).sum()}")
print(f"  Tickers UP_Prec ≥ 70% : {(df_results['UP_Precision'] >= 0.70).sum()}")

# ── Top 30 by UP Precision ────────────────────────────────────────────────────
print(f"\n── Top 30 by UP Precision ─────────────────────────────────────────────")
print(f"  {'Ticker':<14}  {'N':>5}  {'UP_Prec':>8}  {'UP_Rec':>7}  {'DN_Prec':>8}  {'Acc':>7}  {'Edge_UP':>8}")
top30 = df_results.sort_values('UP_Precision', ascending=False).head(30)
for _, r in top30.iterrows():
    print(f"  {r['Ticker']:<14}  {r['N_Test']:>5}  {r['UP_Precision']:>8.1%}  {r['UP_Recall']:>7.1%}  {r['DOWN_Precision']:>8.1%}  {r['Accuracy']:>7.1%}  {r['Edge_UP']:>+8.1%}")

# ── Build predictions sheet (all tickers combined) ────────────────────────────
pred_rows = []
for ticker, y, preds, probs, dates in all_preds:
    for i in range(len(y)):
        pred_rows.append({
            'Ticker':    ticker,
            'Date':      pd.Timestamp(dates[i]).date(),
            'Prob_Up':   round(float(probs[i, 0]), 4),
            'Prob_Down': round(float(probs[i, 1]), 4),
            'Predicted': 'Up' if preds[i] == 0 else 'Down',
            'Actual':    'Up' if y[i] == 0 else 'Down',
            'Correct':   bool(preds[i] == y[i]),
        })
pred_df = pd.DataFrame(pred_rows)

# ── Save Excel ────────────────────────────────────────────────────────────────
os.makedirs(SHEETS_DIR, exist_ok=True)

with pd.ExcelWriter(OUT, engine='xlsxwriter') as writer:
    wb = writer.book

    # Sheet 1: All tickers sorted by UP Precision
    sheet1 = df_results.sort_values('UP_Precision', ascending=False)
    sheet1.to_excel(writer, sheet_name='All_Tickers', index=False)

    # Sheet 2: Top tickers (UP_Precision >= 60% and N_Test >= 50)
    top_sheet = sheet1[(sheet1['UP_Precision'] >= 0.60) & (sheet1['N_Test'] >= 50)]
    top_sheet.to_excel(writer, sheet_name='Top_UP60pct', index=False)

    # Sheet 3: All predictions
    pred_df.to_excel(writer, sheet_name='All_Predictions', index=False)

    # Highlight UP_Precision column in All_Tickers
    ws    = writer.sheets['All_Tickers']
    green = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221'})
    red   = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    col_i = sheet1.columns.get_loc('UP_Precision')
    ws.conditional_format(1, col_i, len(sheet1), col_i,
                          {'type': 'cell', 'criteria': '>=', 'value': 0.65, 'format': green})
    ws.conditional_format(1, col_i, len(sheet1), col_i,
                          {'type': 'cell', 'criteria': '<', 'value': 0.45, 'format': red})

print(f"\nSaved: {OUT}")
