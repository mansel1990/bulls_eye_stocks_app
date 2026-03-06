import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, datetime, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from joblib import Parallel, delayed

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
FOCUS_TICKER = 'RELIANCE'   # primary evaluation ticker
# 25 additional tickers to evaluate (large-cap NSE)
EVAL_TICKERS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'HINDUNILVR', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'KOTAKBANK',
    'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
    'SUNPHARMA', 'ULTRACEMCO', 'WIPRO', 'NESTLEIND', 'POWERGRID',
    'NTPC', 'ONGC', 'TATAMOTORS', 'TATASTEEL', 'ADANIENT',
]
SEQ_LEN      = 30
HORIZON      = 5
THRESH_UP    = 1.0
THRESH_DN    = -1.0
TRAIN_CUTOFF = pd.Timestamp('2025-01-01')
BATCH_SIZE   = 512
MAX_EPOCHS   = 60
PATIENCE     = 10
MAX_TICKERS  = 500          # top N tickers by data length for speed

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE       = os.path.dirname(__file__)
SHEETS_DIR = os.path.join(BASE, 'sheets')
DAILY_DIR  = os.path.normpath(os.path.join(os.path.dirname(BASE), '..', 'daily_data'))
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'lstm_5d_26tickers_{TODAY_STR}.xlsx')

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
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Model ─────────────────────────────────────────────────────────────────────
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

# ── Per-ticker sequence builder ───────────────────────────────────────────────
def process_ticker(ticker):
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
            'Open':   df['Open'],
            'High':   df['High'],
            'Low':    df['Low'],
            'Close':  df['Close'],
            'Volume': df['Volume'],
            'MACD_l': macd_l,
            'MACD_s': macd_s,
            'RSI':    rsi,
            'Ret1':   ret1,
            'Ret5':   ret5,
        }).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        # Labels
        fwd5  = (df['Close'].shift(-HORIZON) - df['Close']) / df['Close'] * 100
        label = np.full(len(df), np.nan)
        label[fwd5 > THRESH_UP]  = 0   # Up
        label[fwd5 < THRESH_DN]  = 1   # Down
        # Flat rows are skipped (nan)

        # Scaler fitted on train only
        train_mask = df.index < TRAIN_CUTOFF
        if train_mask.sum() < SEQ_LEN + 20:
            return None
        scaler = MinMaxScaler()
        scaler.fit(feat[train_mask].values)
        f_scaled = scaler.transform(feat.values)

        X, y, dates = [], [], []
        for i in range(SEQ_LEN - 1, len(df) - HORIZON):
            lbl = label[i]
            if np.isnan(lbl):
                continue
            X.append(f_scaled[i - SEQ_LEN + 1 : i + 1])
            y.append(int(lbl))
            dates.append(df.index[i])

        if len(X) < 10:
            return None

        return (np.array(X, dtype=np.float32),
                np.array(y, dtype=np.int64),
                np.array(dates),
                ticker)
    except Exception:
        return None

# ── Discover tickers ──────────────────────────────────────────────────────────
print("Scanning daily_data directory...")
all_files = sorted(os.listdir(DAILY_DIR))
all_tickers = [f.replace('_daily.csv', '') for f in all_files if f.endswith('_daily.csv')]

# Ensure all EVAL_TICKERS are always included
eval_set = set(t for t in EVAL_TICKERS if os.path.exists(os.path.join(DAILY_DIR, f'{t}_daily.csv')))
remaining = [t for t in all_tickers if t not in eval_set]

# Fill remaining slots with largest tickers by file size
sizes = [(t, os.path.getsize(os.path.join(DAILY_DIR, f'{t}_daily.csv'))) for t in remaining]
sizes.sort(key=lambda x: -x[1])
selected = list(eval_set) + [t for t, _ in sizes[:MAX_TICKERS - len(eval_set)]]
print(f"Processing {len(selected)} tickers (eval tickers: {len(eval_set)})...")

# ── Build sequences in parallel ───────────────────────────────────────────────
results = Parallel(n_jobs=-1, prefer='threads')(
    delayed(process_ticker)(t) for t in selected
)
results = [r for r in results if r is not None]
print(f"Valid tickers: {len(results)}")

# ── Split train / test ────────────────────────────────────────────────────────
all_X_tr, all_y_tr = [], []
# per-ticker test storage for all eval tickers
ticker_test = {}   # ticker -> (X, y, dates)

for X, y, dates, ticker in results:
    tr_m = dates < TRAIN_CUTOFF
    te_m = ~tr_m

    if tr_m.any():
        all_X_tr.append(X[tr_m])
        all_y_tr.append(y[tr_m])

    if ticker in eval_set and te_m.any():
        ticker_test[ticker] = (X[te_m], y[te_m], dates[te_m])

X_train = np.concatenate(all_X_tr)
y_train = np.concatenate(all_y_tr)

# Primary focus ticker test arrays (for backward compat)
X_test, y_test, d_test = ticker_test[FOCUS_TICKER]

print(f"\nTrain: {len(X_train):,} sequences across {len(results)} tickers")
print(f"Eval tickers with test data: {len(ticker_test)}")

for split, yy in [('Train', y_train), (f'Test({FOCUS_TICKER})', y_test)]:
    vals, cnts = np.unique(yy, return_counts=True)
    dist = dict(zip(vals.tolist(), cnts.tolist()))
    total = len(yy)
    print(f"  {split}: Up={dist.get(0,0)} ({dist.get(0,0)/total:.0%})  Down={dist.get(1,0)} ({dist.get(1,0)/total:.0%})")

N_FEATURES = X_train.shape[2]

# ── Training setup ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

val_n  = max(1, int(len(X_train) * 0.15))
X_tr   = X_train[:-val_n]; y_tr = y_train[:-val_n]
X_val  = X_train[-val_n:]; y_val = y_train[-val_n:]

# Balanced sampler
cls_counts = np.bincount(y_tr)
weights    = 1.0 / cls_counts[y_tr]
sampler    = WeightedRandomSampler(weights.tolist(), num_samples=len(y_tr), replacement=True)

tr_loader  = DataLoader(SeqDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, sampler=sampler)
vl_loader  = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE)

model     = LSTMClassifier(n_features=N_FEATURES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

best_loss  = float('inf')
wait       = 0
best_state = None

print(f"\nTraining (max {MAX_EPOCHS} epochs, patience={PATIENCE})...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss    += loss.item()
        tr_correct += (logits.argmax(1) == yb).sum().item()
        tr_total   += len(yb)
    tr_loss /= len(tr_loader)

    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in vl_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            v_loss += criterion(logits, yb).item()
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(yb)
    v_loss /= len(vl_loader)
    v_acc   = v_correct / v_total

    scheduler.step(v_loss)
    lr = optimizer.param_groups[0]['lr']
    print(f"  Epoch {epoch:3d} | TrainLoss {tr_loss:.4f} | ValLoss {v_loss:.4f} ValAcc {v_acc:.2%} | LR {lr:.2e}", flush=True)

    if v_loss < best_loss:
        best_loss  = v_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

# ── Save model ────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(BASE, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_5d.pth')
os.makedirs(MODEL_DIR, exist_ok=True)
model.load_state_dict(best_state)
torch.save({'state_dict': best_state, 'n_features': N_FEATURES,
            'seq_len': SEQ_LEN, 'hidden': 64}, MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()

te_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE)
all_probs = []
with torch.no_grad():
    for xb, _ in te_loader:
        logits = model(xb.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

probs = np.concatenate(all_probs)
preds = np.argmax(probs, axis=1)

print(f"\n── {FOCUS_TICKER} Test Results (2025+) {'─'*40}")
print(classification_report(y_test, preds, target_names=['Up', 'Down']))

up_prec   = precision_score(y_test, preds, pos_label=0, average='binary', zero_division=0)
down_prec = precision_score(y_test, preds, pos_label=1, average='binary', zero_division=0)
up_rec    = recall_score(y_test, preds, pos_label=0, average='binary', zero_division=0)
down_rec  = recall_score(y_test, preds, pos_label=1, average='binary', zero_division=0)
acc       = (preds == y_test).mean()
baseline  = max(np.bincount(y_test)) / len(y_test)

print(f"UP   Precision: {up_prec:.1%}  Recall: {up_rec:.1%}")
print(f"DOWN Precision: {down_prec:.1%}  Recall: {down_rec:.1%}")
print(f"Accuracy: {acc:.1%}  |  Majority-class baseline: {baseline:.1%}")

# ── Per-ticker evaluation ─────────────────────────────────────────────────────
print(f"\n── Per-Ticker Results (2025+) {'─'*40}")
print(f"  {'Ticker':<14}  {'N':>5}  {'UP_Prec':>8}  {'DN_Prec':>8}  {'Acc':>7}  {'Baseline':>9}")
ticker_rows = []
for tkr in sorted(ticker_test.keys()):
    Xt, yt, dt = ticker_test[tkr]
    if len(Xt) == 0:
        continue
    tl = DataLoader(SeqDataset(Xt, yt), batch_size=BATCH_SIZE)
    tp = []
    with torch.no_grad():
        for xb, _ in tl:
            tp.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    tp    = np.concatenate(tp)
    tpred = np.argmax(tp, axis=1)
    up_p  = precision_score(yt, tpred, pos_label=0, average='binary', zero_division=0)
    dn_p  = precision_score(yt, tpred, pos_label=1, average='binary', zero_division=0)
    tacc  = (tpred == yt).mean()
    tbase = max(np.bincount(yt)) / len(yt)
    print(f"  {tkr:<14}  {len(yt):>5}  {up_p:>8.1%}  {dn_p:>8.1%}  {tacc:>7.1%}  {tbase:>9.1%}")
    ticker_rows.append({
        'Ticker': tkr, 'N_Test': len(yt),
        'UP_Precision': round(up_p, 4), 'DOWN_Precision': round(dn_p, 4),
        'Accuracy': round(tacc, 4), 'Baseline': round(tbase, 4),
        'Edge_UP': round(up_p - tbase, 4),
    })
ticker_df = pd.DataFrame(ticker_rows).sort_values('UP_Precision', ascending=False)

# ── Confidence threshold sweep ────────────────────────────────────────────────
print(f"\n── Confidence Threshold Analysis ({'─'*30})")
print(f"  {'Threshold':>9}  {'UP_Prec':>8}  {'UP_N':>6}  {'DN_Prec':>8}  {'DN_N':>6}  {'Coverage':>9}")
thresh_rows = []
for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
    up_mask   = probs[:, 0] >= thresh
    dn_mask   = probs[:, 1] >= thresh
    up_n      = up_mask.sum()
    dn_n      = dn_mask.sum()
    coverage  = (up_mask | dn_mask).sum() / len(y_test)
    up_p = (y_test[up_mask] == 0).mean() if up_n > 0 else 0.0
    dn_p = (y_test[dn_mask] == 1).mean() if dn_n > 0 else 0.0
    print(f"  {thresh:>9.2f}  {up_p:>8.1%}  {up_n:>6}  {dn_p:>8.1%}  {dn_n:>6}  {coverage:>9.1%}")
    thresh_rows.append({'Threshold': thresh, 'UP_Precision': round(up_p, 4), 'UP_N': int(up_n),
                        'DOWN_Precision': round(dn_p, 4), 'DOWN_N': int(dn_n), 'Coverage': round(coverage, 4)})
thresh_df = pd.DataFrame(thresh_rows)

# ── Save Excel ────────────────────────────────────────────────────────────────
label_map = {0: 'Up', 1: 'Down'}
preds_df  = pd.DataFrame({
    'Date':      d_test,
    'Prob_Up':   probs[:, 0].round(4),
    'Prob_Down': probs[:, 1].round(4),
    'Predicted': [label_map[p] for p in preds],
    'Actual':    [label_map[a] for a in y_test],
    'Correct':   preds == y_test,
})

rows = []
for cls, name, pos in [(0, 'Up', 0), (1, 'Down', 1)]:
    mask = y_test == cls
    rows.append({
        'Class':         name,
        'N_Actual':      int(mask.sum()),
        'N_Pred':        int((preds == cls).sum()),
        'Precision':     round(precision_score(y_test, preds, pos_label=pos, average='binary', zero_division=0), 4),
        'Recall':        round(recall_score(y_test, preds, pos_label=pos, average='binary', zero_division=0), 4),
        'Freq_Baseline': round(mask.mean(), 4),
    })
metrics_df = pd.DataFrame(rows)

os.makedirs(SHEETS_DIR, exist_ok=True)
with pd.ExcelWriter(OUT, engine='xlsxwriter') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    ticker_df.to_excel(writer, sheet_name='Per_Ticker', index=False)
    thresh_df.to_excel(writer, sheet_name='Threshold_Sweep', index=False)
    preds_df.to_excel(writer, sheet_name='Predictions', index=False)

print(f"\nSaved: {OUT}")
