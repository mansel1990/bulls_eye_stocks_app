"""
lstm_5d_tune.py
---------------
Hyperparameter search for LSTM 5-day direction predictor.
Optimises UP Precision on a held-out validation set (2024 data).
Uses the same multi-ticker training approach as lstm_5d_predictor.py.

Search space:
  - seq_len:        [20, 30, 45, 60]
  - hidden:         [32, 64, 128]
  - dropout:        [0.2, 0.3, 0.4]
  - lr:             [5e-4, 1e-3, 2e-3]
  - thresh_up/dn:   [0.5, 1.0, 1.5, 2.0]
  - conf_threshold: [0.50, 0.55, 0.58, 0.60]  (inference confidence cutoff)

Strategy:
  Train split  : dates < 2024-01-01
  Val split    : 2024-01-01 <= dates < 2025-01-01  (full year, unseen)
  Metric       : UP Precision at chosen conf_threshold (min 10 predictions)

Saves best model to models/lstm_5d_best.pth and results to sheets/lstm_tune_{date}.xlsx

Usage:
    python lstm_5d_tune.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, datetime, warnings, itertools, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from joblib import Parallel, delayed

# ── Fixed config ──────────────────────────────────────────────────────────────
SEED          = 42
HORIZON       = 5
TRAIN_END     = pd.Timestamp('2024-01-01')   # train strictly before this
VAL_END       = pd.Timestamp('2025-01-01')   # val = [TRAIN_END, VAL_END)
MAX_TICKERS   = 500
BATCH_SIZE    = 512
MAX_EPOCHS    = 40
PATIENCE      = 8
MIN_UP_PREDS  = 10    # skip configs where model predicts < N Up signals in val

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE       = os.path.dirname(__file__)
SHEETS_DIR = os.path.join(BASE, 'sheets')
MODELS_DIR = os.path.join(BASE, 'models')
DAILY_DIR  = os.path.normpath(os.path.join(os.path.dirname(BASE), '..', 'daily_data'))
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'lstm_tune_{TODAY_STR}.xlsx')
BEST_PATH  = os.path.join(MODELS_DIR, 'lstm_5d_best.pth')

# ── Search space ──────────────────────────────────────────────────────────────
# Set QUICK=True for a fast ~48-config run to find promising region first,
# then set QUICK=False for the full 1152-config grid search.
QUICK = True

if QUICK:
    SEARCH = {
        'seq_len':        [20, 30, 45],
        'hidden':         [64, 128],
        'dropout':        [0.2, 0.3],
        'lr':             [5e-4, 1e-3],
        'thresh_up':      [1.0, 1.5],
        'conf_threshold': [0.55, 0.60],
    }
else:
    SEARCH = {
        'seq_len':        [20, 30, 45, 60],
        'hidden':         [64, 128],
        'dropout':        [0.2, 0.3, 0.4],
        'lr':             [5e-4, 1e-3, 2e-3],
        'thresh_up':      [0.5, 1.0, 1.5, 2.0],
        'conf_threshold': [0.50, 0.55, 0.58, 0.60],
    }

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

# ── Dataset ───────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]

# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.3, n_classes=2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden // 2, n_classes)

    def forward(self, x):
        x, _      = self.lstm1(x)
        x         = self.drop1(x)
        _, (h, _) = self.lstm2(x)
        return self.fc(self.drop2(h[-1]))

# ── Per-ticker raw data loader (cached, no seq building yet) ──────────────────
def load_ticker_raw(ticker):
    path = os.path.join(DAILY_DIR, f'{ticker}_daily.csv')
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=True).sort_index()
        if len(df) < 100: return None
        macd_l, macd_s = compute_macd(df['Close'])
        rsi  = compute_rsi(df['Close'])
        ret1 = df['Close'].pct_change(1) * 100
        ret5 = df['Close'].pct_change(5) * 100
        feat = pd.DataFrame({
            'Open': df['Open'], 'High': df['High'], 'Low': df['Low'],
            'Close': df['Close'], 'Volume': df['Volume'],
            'MACD_l': macd_l, 'MACD_s': macd_s, 'RSI': rsi,
            'Ret1': ret1, 'Ret5': ret5,
        }).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return ticker, df.index.values, feat.values, df['Close'].values
    except: return None

# ── Build sequences for given config ─────────────────────────────────────────
def build_sequences(raw_data, seq_len, thresh_up, thresh_dn):
    """Build (X, y, dates) arrays for train, val splits."""
    all_X_tr, all_y_tr = [], []
    all_X_val, all_y_val = [], []

    for ticker, dates, feat_vals, close in raw_data:
        n = len(dates)
        if n < seq_len + HORIZON + 20: continue

        # Labels
        fwd5  = np.full(n, np.nan)
        for i in range(n - HORIZON):
            fwd5[i] = (close[i + HORIZON] - close[i]) / close[i] * 100
        label = np.full(n, np.nan)
        label[fwd5 > thresh_up]   = 0   # Up
        label[fwd5 < -thresh_dn]  = 1   # Down

        # Fit scaler on data before TRAIN_END
        tr_mask = dates < np.datetime64(TRAIN_END)
        if tr_mask.sum() < seq_len + 20: continue
        scaler = MinMaxScaler()
        scaler.fit(feat_vals[tr_mask])
        f_scaled = scaler.transform(feat_vals)

        for i in range(seq_len - 1, n - HORIZON):
            lbl = label[i]
            if np.isnan(lbl): continue
            seq = f_scaled[i - seq_len + 1 : i + 1]
            d   = dates[i]
            if d < np.datetime64(TRAIN_END):
                all_X_tr.append(seq); all_y_tr.append(int(lbl))
            elif d < np.datetime64(VAL_END):
                all_X_val.append(seq); all_y_val.append(int(lbl))

    if not all_X_tr or not all_X_val: return None
    return (np.array(all_X_tr, dtype=np.float32), np.array(all_y_tr, dtype=np.int64),
            np.array(all_X_val, dtype=np.float32), np.array(all_y_val, dtype=np.int64))

# ── Train one config ──────────────────────────────────────────────────────────
def train_eval(cfg, data_pack, device):
    seq_len, hidden, dropout, lr, thresh_up, conf_thresh = (
        cfg['seq_len'], cfg['hidden'], cfg['dropout'],
        cfg['lr'], cfg['thresh_up'], cfg['conf_threshold']
    )
    result = build_sequences(data_pack, seq_len, thresh_up, thresh_up)
    if result is None: return None

    X_tr, y_tr, X_val, y_val = result
    n_features = X_tr.shape[2]

    # Balanced sampler for train
    cls_counts = np.bincount(y_tr)
    if len(cls_counts) < 2 or 0 in cls_counts: return None
    weights = 1.0 / cls_counts[y_tr]
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(y_tr), replacement=True)

    tr_loader  = DataLoader(SeqDataset(X_tr, y_tr),   batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(SeqDataset(X_val, y_val),  batch_size=BATCH_SIZE)

    model     = LSTMClassifier(n_features=n_features, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_state    = None
    wait          = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                v_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break

    # Evaluate with confidence threshold
    model.load_state_dict(best_state)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs)

    up_mask = probs[:, 0] >= conf_thresh
    n_up    = up_mask.sum()
    if n_up < MIN_UP_PREDS:
        up_prec = 0.0
    else:
        up_prec = (y_val[up_mask] == 0).mean()

    # Also compute overall val acc for reference
    preds   = np.argmax(probs, axis=1)
    val_acc = (preds == y_val).mean()

    return {
        **cfg,
        'up_prec':    round(float(up_prec), 4),
        'n_up_preds': int(n_up),
        'val_acc':    round(float(val_acc), 4),
        'n_train':    len(y_tr),
        'n_val':      len(y_val),
        'state_dict': best_state,
        'n_features': n_features,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading raw ticker data...")
all_files   = sorted(os.listdir(DAILY_DIR))
all_tickers = [f.replace('_daily.csv', '') for f in all_files if f.endswith('_daily.csv')]

# Top tickers by file size for speed
sizes    = [(t, os.path.getsize(os.path.join(DAILY_DIR, f'{t}_daily.csv'))) for t in all_tickers]
sizes.sort(key=lambda x: -x[1])
selected = [t for t, _ in sizes[:MAX_TICKERS]]

raw_data = Parallel(n_jobs=-1, prefer='threads')(
    delayed(load_ticker_raw)(t) for t in selected
)
raw_data = [r for r in raw_data if r is not None]
print(f"Raw data loaded for {len(raw_data)} tickers")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate all combinations
keys   = list(SEARCH.keys())
combos = list(itertools.product(*[SEARCH[k] for k in keys]))
cfgs   = [dict(zip(keys, c)) for c in combos]

# Deduplicate by (seq_len, thresh_up) to avoid redundant data builds
# Each unique (seq_len, thresh_up) combo is a unique data configuration
print(f"\nTotal configurations: {len(cfgs)}")
print(f"Starting search...\n")

results     = []
best_prec   = 0.0
best_cfg    = None
best_model  = None
t0          = time.time()

for i, cfg in enumerate(cfgs, 1):
    res = train_eval(cfg, raw_data, device)
    if res is None: continue

    state = res.pop('state_dict')
    n_f   = res.pop('n_features')
    results.append(res)

    elapsed = time.time() - t0
    eta     = elapsed / i * (len(cfgs) - i)
    print(f"[{i:4d}/{len(cfgs)}] "
          f"seq={cfg['seq_len']:2d} hid={cfg['hidden']:3d} drop={cfg['dropout']:.1f} "
          f"lr={cfg['lr']:.0e} thr={cfg['thresh_up']:.1f} conf={cfg['conf_threshold']:.2f} "
          f"=> UP_Prec={res['up_prec']:.1%} (n={res['n_up_preds']:3d}) "
          f"ValAcc={res['val_acc']:.1%}  ETA {eta/60:.1f}m", flush=True)

    if res['up_prec'] > best_prec and res['n_up_preds'] >= MIN_UP_PREDS:
        best_prec  = res['up_prec']
        best_cfg   = {**cfg, 'n_features': n_f}
        best_model = state
        print(f"  *** NEW BEST: UP_Prec={best_prec:.1%} ***")

# ── Save best model ───────────────────────────────────────────────────────────
if best_model is not None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'state_dict': best_model,
        'n_features': best_cfg['n_features'],
        'seq_len':    best_cfg['seq_len'],
        'hidden':     best_cfg['hidden'],
        'dropout':    best_cfg['dropout'],
        'thresh_up':  best_cfg['thresh_up'],
        'conf_threshold': best_cfg['conf_threshold'],
    }, BEST_PATH)
    print(f"\nBest model saved: {BEST_PATH}")
    print(f"Best config: {best_cfg}")
    print(f"Best UP Precision (val): {best_prec:.1%}")

# ── Save results to Excel ─────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values('up_prec', ascending=False)

os.makedirs(SHEETS_DIR, exist_ok=True)
with pd.ExcelWriter(OUT, engine='xlsxwriter') as writer:
    wb = writer.book

    # All results
    df.to_excel(writer, sheet_name='All_Configs', index=False)

    # Top 50 by UP precision
    df.head(50).to_excel(writer, sheet_name='Top50', index=False)

    # Best config per seq_len
    best_per_seq = df.groupby('seq_len').first().reset_index()
    best_per_seq.to_excel(writer, sheet_name='Best_Per_SeqLen', index=False)

    # Highlight UP precision
    ws  = writer.sheets['All_Configs']
    col = df.columns.get_loc('up_prec')
    fmt_green = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221'})
    fmt_red   = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    ws.conditional_format(1, col, len(df), col,
        {'type': 'cell', 'criteria': '>=', 'value': 0.70, 'format': fmt_green})
    ws.conditional_format(1, col, len(df), col,
        {'type': 'cell', 'criteria': '<',  'value': 0.55, 'format': fmt_red})

print(f"\nResults saved: {OUT}")
print(f"Total time: {(time.time()-t0)/60:.1f} minutes")
