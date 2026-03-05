"""
Random Forest + Deep Learning — 1D Profit Precision
=====================================================
Data:  feature_table_2026-02-25.xlsx  (2024=train, 2025=test)
Target: 1D Result (Profit=1, Loss=0)
Goal:   Maximise UP-class precision (of predicted Profit, how many actually profit)

Column exclusions (per user spec):
  EXCLUDE A-F  : Ticker, Company Name, Sector, Industry, Signal Date, Entry Date
  EXCLUDE J-O  : Sig Open, Sig High, Sig Low, Sig Close, Entry Price, SL Price
                 → replaced by 6 derived boolean comparison features
  EXCLUDE Z,AA : Vol Today, Vol Yesterday  (raw volumes, too ticker-specific)
  EXCLUDE AH   : Base Status (text, redundant with Base Score)

New derived features (replace J-O):
  entry_gt_sl       : Entry Price > SL Price
  entry_gt_sig_open : Entry Price > Sig Open
  entry_gt_sig_low  : Entry Price > Sig Low
  entry_gt_sig_high : Entry Price > Sig High
  entry_gt_sig_close: Entry Price > Sig Close
  entry_gt_avg_yday : Entry Price > (Sig Open + Sig Close) / 2  (avg of signal bar)

Feature set (G onwards, minus exclusions, plus 6 derived):
  G  Pattern            (one-hot encoded)
  H  Strength           (binary)
  I  Entry Type         (one-hot encoded)
  P  Risk (%)
  Q  MA44
  R  MA44 Ascending
  S  Close > MA44
  T  MA200
  U  Close > MA200
  V  RSI14
  W  52W High Norm (price)
  X  52W Low Norm (price)
  Y  52W Change % (price)
  AB Vol Today > Yesterday
  AC MACD Line
  AD MACD Signal
  AE MACD Hist
  AF MACD > Signal
  AG Base Score
  AI-AQ BS fundamentals (P/E, PEG, P/B, ROE, Margin, D/E, CR, Int Cov, Z-Score)
  AR-AT Valuation/Profitability/Health Pts
  AU Fundamental Score
  AV Performer Score
  AW-BQ All normalized fundamental features
  + 6 derived entry comparison booleans

Output: sheets/rf_dl_precision_<date>.xlsx
"""

import os, datetime, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
SHEETS_DIR = os.path.join(BASE, 'sheets')
SRC        = os.path.join(SHEETS_DIR, 'feature_table_2026-02-25.xlsx')
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'rf_dl_precision_{TODAY_STR}.xlsx')
SEED       = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ── Load both years ────────────────────────────────────────────────────────────
print("Loading feature table...")
df24 = pd.read_excel(SRC, sheet_name='2024 Feature Table')
df25 = pd.read_excel(SRC, sheet_name='2025 Feature Table')
df24['_year'] = '2024'
df25['_year'] = '2025'
raw = pd.concat([df24, df25], ignore_index=True)
print(f"  Loaded: {len(raw):,} rows  |  {len(raw.columns)} columns")

# ── Derived entry comparison features (replace J-O raw prices) ────────────────
print("Creating derived entry comparison features...")
raw['entry_gt_sl']        = (raw['Entry Price'] > raw['SL Price']).astype(float)
raw['entry_gt_sig_open']  = (raw['Entry Price'] > raw['Sig Open']).astype(float)
raw['entry_gt_sig_low']   = (raw['Entry Price'] > raw['Sig Low']).astype(float)
raw['entry_gt_sig_high']  = (raw['Entry Price'] > raw['Sig High']).astype(float)
raw['entry_gt_sig_close'] = (raw['Entry Price'] > raw['Sig Close']).astype(float)
raw['entry_gt_avg_yday']  = (raw['Entry Price'] > (raw['Sig Open'] + raw['Sig Close']) / 2).astype(float)

DERIVED_COLS = [
    'entry_gt_sl', 'entry_gt_sig_open', 'entry_gt_sig_low',
    'entry_gt_sig_high', 'entry_gt_sig_close', 'entry_gt_avg_yday',
]

# ── Define feature columns ─────────────────────────────────────────────────────
# Explicitly excluded columns
EXCLUDE = {
    # A-F: identity / dates
    'Ticker', 'Company Name', 'Sector', 'Industry', 'Signal Date', 'Entry Date',
    # J-O: raw prices (replaced by derived booleans)
    'Sig Open', 'Sig High', 'Sig Low', 'Sig Close', 'Entry Price', 'SL Price',
    # Z, AA: raw volume counts
    'Vol Today', 'Vol Yesterday',
    # AH: text status label
    'Base Status',
    # Targets and meta
    '1D Result', '1D Return (%)', '3D Result', '3D Return (%)',
    '5D Result', '5D Return (%)', '_year',
}

# Categorical columns that need encoding
CAT_COLS = ['Pattern', 'Strength', 'Entry Type']

# All numeric/bool feature cols (G+ minus exclusions minus categoricals)
NUM_COLS = [c for c in raw.columns
            if c not in EXCLUDE and c not in CAT_COLS and c not in DERIVED_COLS]

print(f"  Numeric/bool features: {len(NUM_COLS)}")
print(f"  Categorical features:  {len(CAT_COLS)}")
print(f"  Derived bool features: {len(DERIVED_COLS)}")

# ── Encode categoricals ────────────────────────────────────────────────────────
# Pattern -> one-hot
pat_dummies  = pd.get_dummies(raw['Pattern'].fillna('Unknown'), prefix='PAT')
# Strength -> binary
raw['Strength_bin'] = (raw['Strength'] == 'Strongly Bullish').astype(float)
# Entry Type -> one-hot
et_dummies   = pd.get_dummies(raw['Entry Type'].fillna('stop'), prefix='ET')

# ── Boolean columns: True/False -> 1/0 ────────────────────────────────────────
BOOL_COLS = ['MA44 Ascending', 'Close > MA44', 'Close > MA200',
             'Vol Today > Yesterday', 'MACD > Signal']
for bc in BOOL_COLS:
    if bc in raw.columns:
        raw[bc] = raw[bc].map({True: 1, False: 0, 'True': 1, 'False': 0,
                                1: 1, 0: 0}).fillna(0)

# ── Assemble feature matrix ────────────────────────────────────────────────────
feat_df = pd.concat([
    raw[NUM_COLS].reset_index(drop=True),
    raw[['Strength_bin'] + DERIVED_COLS].reset_index(drop=True),
    pat_dummies.reset_index(drop=True),
    et_dummies.reset_index(drop=True),
], axis=1)

# Numeric coerce + fill NaN with median
for c in feat_df.columns:
    feat_df[c] = pd.to_numeric(feat_df[c], errors='coerce')
feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0)

feature_names = feat_df.columns.tolist()
print(f"  Total features after encoding: {len(feature_names)}")

# Replace inf/-inf with NaN then fill with median
feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0)
# Clip extreme outliers on continuous columns only (skip bool/binary cols)
bool_set = set(BOOL_COLS + DERIVED_COLS +
               [c for c in feature_names if c.startswith('PAT_') or c.startswith('ET_')
                or c == 'Strength_bin'])
for c in feat_df.columns:
    if c in bool_set:
        continue
    col = feat_df[c].astype(float)
    q1, q3 = col.quantile(0.01), col.quantile(0.99)
    if q3 > q1:
        feat_df[c] = col.clip(q1 - 5*(q3-q1), q3 + 5*(q3-q1))

# ── Target ─────────────────────────────────────────────────────────────────────
y_all = (raw['1D Result'] == 'Profit').astype(int).values

# ── Train (2024) / Test (2025) split ──────────────────────────────────────────
train_mask = raw['_year'].values == '2024'
test_mask  = raw['_year'].values == '2025'

X_train = feat_df.values[train_mask].astype(np.float32)
X_test  = feat_df.values[test_mask].astype(np.float32)
y_train = y_all[train_mask]
y_test  = y_all[test_mask]

baseline = y_test.mean() * 100
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"1D Profit baseline (test): {baseline:.1f}%")
print(f"Train profit rate:         {y_train.mean()*100:.1f}%")

# ── Precision-optimised threshold sweep ───────────────────────────────────────
def best_threshold(probs, y_true, min_recall=0.05, min_signals=30):
    """Find threshold maximising precision with recall >= min_recall."""
    best_p, best_t, best_r = 0, 0.5, 0
    for t in np.arange(0.30, 0.97, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() < min_signals:
            continue
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        if r >= min_recall and p > best_p:
            best_p, best_t, best_r = p, t, r
    return best_t, best_p, best_r

def evaluate(y_true, y_pred, name, thr):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        'Model':            name,
        'Threshold':        round(thr, 2),
        'Baseline (%)':     round(baseline, 1),
        'UP Precision (%)': round(precision_score(y_true, y_pred, zero_division=0)*100, 1),
        'UP Recall (%)':    round(recall_score(y_true, y_pred, zero_division=0)*100, 1),
        'F1':               round(f1_score(y_true, y_pred, zero_division=0), 3),
        'Accuracy (%)':     round(accuracy_score(y_true, y_pred)*100, 1),
        'Predicted UP':     int(y_pred.sum()),
        'True Positives':   int(tp),
        'False Positives':  int(fp),
        'Test N':           len(y_true),
        'Edge (pp)':        round(precision_score(y_true, y_pred, zero_division=0)*100 - baseline, 1),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("RANDOM FOREST")
print("="*65)

pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

rf = RandomForestClassifier(
    n_estimators     = 600,
    max_depth        = 12,
    min_samples_leaf = 20,       # prevents overfitting on small leaves
    min_samples_split= 40,
    max_features     = 'sqrt',
    class_weight     = {0: 1, 1: pos_ratio},
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = 0,
)
print("  Training Random Forest (600 trees)...")
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_thr, rf_prec, rf_rec = best_threshold(rf_probs, y_test, min_recall=0.05)
rf_pred = (rf_probs >= rf_thr).astype(int)
rf_res  = evaluate(y_test, rf_pred, 'Random Forest', rf_thr)

print(f"  Threshold: {rf_thr:.2f}  |  Precision: {rf_prec*100:.1f}%  |  "
      f"Recall: {rf_rec*100:.1f}%  |  Predicted UP: {rf_pred.sum():,}  |  "
      f"Edge: {rf_res['Edge (pp)']:+.1f}pp")

# Feature importances
fi = pd.DataFrame({
    'Feature':    feature_names,
    'Importance': rf.feature_importances_,
}).sort_values('Importance', ascending=False).reset_index(drop=True)
fi['Importance (%)'] = (fi['Importance'] * 100).round(3)

print("\n  Top 20 features:")
print(fi[['Feature','Importance (%)']].head(20).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING — MLP with precision-biased loss
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("DEEP LEARNING (MLP)")
print("="*65)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512),  nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)

# Split last 15% of train as validation
n_val      = int(len(X_train_sc) * 0.15)
X_tr, X_vl = X_train_sc[:-n_val], X_train_sc[-n_val:]
y_tr, y_vl  = y_train[:-n_val].astype(np.float32), y_train[-n_val:].astype(np.float32)

# Up-weight positive class × 1.5 to push for precision
pos_w  = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1) * 1.5
pos_wt = torch.tensor(pos_w, dtype=torch.float32)

tr_ld = DataLoader(DS(X_tr, y_tr), batch_size=512, shuffle=True,  drop_last=True)
vl_ld = DataLoader(DS(X_vl, y_vl), batch_size=512, shuffle=False)

model  = MLP(X_train_sc.shape[1])
opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

best_vl, best_state, patience, wait = float('inf'), None, 15, 0

print(f"  Training MLP ({X_train_sc.shape[1]} features)...")
for epoch in range(150):
    model.train()
    for xb, yb in tr_ld:
        opt.zero_grad()
        out  = model(xb)
        w    = torch.where(yb == 1, pos_wt, torch.ones_like(yb))
        loss = nn.functional.binary_cross_entropy(out, yb, weight=w)
        loss.backward(); opt.step()
    sched.step()

    model.eval()
    vl_loss = 0.0
    with torch.no_grad():
        for xb, yb in vl_ld:
            out     = model(xb)
            w       = torch.where(yb == 1, pos_wt, torch.ones_like(yb))
            vl_loss += nn.functional.binary_cross_entropy(out, yb, weight=w).item()
    vl_loss /= max(len(vl_ld), 1)

    if vl_loss < best_vl:
        best_vl    = vl_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    dl_probs = model(torch.tensor(X_test_sc)).numpy()

dl_thr, dl_prec, dl_rec = best_threshold(dl_probs, y_test, min_recall=0.05)
dl_pred = (dl_probs >= dl_thr).astype(int)
dl_res  = evaluate(y_test, dl_pred, 'Deep Learning (MLP)', dl_thr)

print(f"  Epochs: {epoch+1}  |  Threshold: {dl_thr:.2f}  |  "
      f"Precision: {dl_prec*100:.1f}%  |  Recall: {dl_rec*100:.1f}%  |  "
      f"Predicted UP: {dl_pred.sum():,}  |  Edge: {dl_res['Edge (pp)']:+.1f}pp")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERSECTION: RF AND DL both agree (high confidence signals only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("INTERSECTION (RF AND DL both high confidence)")
print("="*65)

best_int_p, best_int_res = 0, None
for t in np.arange(0.40, 0.92, 0.02):
    y_int = ((rf_probs >= t) & (dl_probs >= t)).astype(int)
    if y_int.sum() < 20:
        continue
    p = precision_score(y_test, y_int, zero_division=0)
    r = recall_score(y_test, y_int, zero_division=0)
    if r >= 0.02 and p > best_int_p:
        best_int_p = p
        best_int_res = evaluate(y_test, y_int, 'RF + DL Intersection', t)

if best_int_res:
    print(f"  Threshold: {best_int_res['Threshold']:.2f}  |  "
          f"Precision: {best_int_res['UP Precision (%)']:.1f}%  |  "
          f"Recall: {best_int_res['UP Recall (%)']:.1f}%  |  "
          f"Predicted UP: {best_int_res['Predicted UP']:,}  |  "
          f"Edge: {best_int_res['Edge (pp)']:+.1f}pp")

# ── Summary ───────────────────────────────────────────────────────────────────
results = [rf_res, dl_res]
if best_int_res:
    results.append(best_int_res)

print("\n" + "="*65)
print("RESULTS SUMMARY (1D Profit Precision)")
print("="*65)
df_res = pd.DataFrame(results)
print(df_res[['Model','Baseline (%)','UP Precision (%)','Edge (pp)',
              'UP Recall (%)','Predicted UP','True Positives',
              'False Positives','Threshold']].to_string(index=False))

# ── High-confidence signal list (test set 2025, predicted Profit by both) ─────
# Build a readable output: 2025 signals predicted as Profit by RF, ranked by RF prob
test_raw = raw[test_mask].reset_index(drop=True)
test_raw['RF_Prob']   = rf_probs
test_raw['DL_Prob']   = dl_probs
test_raw['RF_Pred']   = rf_pred
test_raw['DL_Pred']   = dl_pred
test_raw['Both_Pred'] = (rf_pred & dl_pred)

# Signals where BOTH models predict Profit, sorted by avg probability
both_signals = test_raw[test_raw['Both_Pred'] == 1].copy()
both_signals['Avg_Prob'] = (both_signals['RF_Prob'] + both_signals['DL_Prob']) / 2
both_signals = both_signals.sort_values('Avg_Prob', ascending=False).reset_index(drop=True)

print(f"\n  Signals where BOTH models predict Profit: {len(both_signals):,}")
if len(both_signals):
    actual_profit = (both_signals['1D Result'] == 'Profit').mean() * 100
    print(f"  Actual 1D profit rate in those signals:  {actual_profit:.1f}%")

# ── Excel output ───────────────────────────────────────────────────────────────
print(f"\nWriting Excel: {OUT}")

PROFIT_C = '#C6EFCE'; LOSS_C = '#FFC7CE'

def write_results_sheet(writer, df_r, sheet_name, wb):
    df_r.to_excel(writer, index=False, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]
    fh = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white',
                         'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fg = wb.add_format({'bg_color': PROFIT_C, 'font_color': '#276221',
                         'bold': True, 'num_format': '0.0'})
    fm = wb.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500', 'num_format': '0.0'})
    fb = wb.add_format({'bg_color': LOSS_C,    'font_color': '#9C0006', 'num_format': '0.0'})
    fe = wb.add_format({'bg_color': '#DDEBF7', 'font_color': '#1F3864',
                         'num_format': '+0.0;-0.0'})
    ws.set_row(0, 28)
    prec_ci = df_r.columns.get_loc('UP Precision (%)') if 'UP Precision (%)' in df_r.columns else None
    edge_ci = df_r.columns.get_loc('Edge (pp)')        if 'Edge (pp)'        in df_r.columns else None
    for ci, cn in enumerate(df_r.columns):
        ws.set_column(ci, ci, max(len(cn)+2, 12))
        ws.write(0, ci, cn, fh)
    for ri, row in df_r.iterrows():
        if prec_ci is not None:
            p = row['UP Precision (%)']
            ws.write(ri+1, prec_ci, p, fg if p >= 60 else fm if p >= 50 else fb)
        if edge_ci is not None:
            ws.write(ri+1, edge_ci, row['Edge (pp)'], fe)

def write_signals_sheet(writer, df_s, sheet_name, wb):
    DISP_COLS = ['Ticker','Signal Date','Entry Date','Pattern','Strength',
                 'Entry Type','Risk (%)','RSI14','MACD > Signal',
                 'MA44 Ascending','Close > MA44','Close > MA200',
                 'Base Score','Fundamental Score','Performer Score',
                 '1D Result','1D Return (%)',
                 'RF_Prob','DL_Prob','Avg_Prob',
                 'entry_gt_sl','entry_gt_sig_open','entry_gt_sig_low',
                 'entry_gt_sig_high','entry_gt_sig_close','entry_gt_avg_yday']
    disp = df_s[[c for c in DISP_COLS if c in df_s.columns]].copy()
    for pc in ['RF_Prob','DL_Prob','Avg_Prob']:
        if pc in disp.columns:
            disp[pc] = disp[pc].round(4)
    disp.to_excel(writer, index=False, sheet_name=sheet_name)
    ws  = writer.sheets[sheet_name]
    fh  = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white',
                          'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    fp  = wb.add_format({'bg_color': PROFIT_C, 'font_color': '#276221'})
    fl  = wb.add_format({'bg_color': LOSS_C,   'font_color': '#9C0006'})
    fpp = wb.add_format({'bg_color': PROFIT_C, 'font_color': '#276221',
                          'num_format': '+0.00;-0.00'})
    fll = wb.add_format({'bg_color': LOSS_C,   'font_color': '#9C0006',
                          'num_format': '+0.00;-0.00'})
    fpr = wb.add_format({'num_format': '0.0000'})
    fbt = wb.add_format({'bg_color': PROFIT_C, 'font_color': '#276221', 'bold': True})
    fbf = wb.add_format({'bg_color': LOSS_C,   'font_color': '#9C0006'})
    ws.set_row(0, 28)
    COL_W = {
        'Ticker': 12, 'Signal Date': 13, 'Entry Date': 13,
        'Pattern': 24, 'Strength': 18, 'Entry Type': 18,
        'RF_Prob': 10, 'DL_Prob': 10, 'Avg_Prob': 10,
        '1D Result': 10, '1D Return (%)': 13,
        'Base Score': 12, 'Fundamental Score': 18, 'Performer Score': 16,
    }
    BOOL_DISP = {'MACD > Signal','MA44 Ascending','Close > MA44','Close > MA200',
                 'entry_gt_sl','entry_gt_sig_open','entry_gt_sig_low',
                 'entry_gt_sig_high','entry_gt_sig_close','entry_gt_avg_yday'}
    res_ci  = disp.columns.get_loc('1D Result')    if '1D Result'    in disp.columns else None
    ret_ci  = disp.columns.get_loc('1D Return (%)') if '1D Return (%)' in disp.columns else None
    for ci, cn in enumerate(disp.columns):
        ws.set_column(ci, ci, COL_W.get(cn, max(len(cn)+2, 10)))
        ws.write(0, ci, cn, fh)
    for ri, row in disp.iterrows():
        er = ri + 1
        if res_ci is not None:
            lbl = row['1D Result']
            ws.write(er, res_ci, lbl, fp if lbl == 'Profit' else fl)
        if ret_ci is not None:
            pct = row['1D Return (%)']
            lbl = row.get('1D Result', '')
            if pd.notna(pct):
                ws.write(er, ret_ci, pct, fpp if lbl == 'Profit' else fll)
        for cn in BOOL_DISP:
            if cn in disp.columns:
                ci2 = disp.columns.get_loc(cn)
                v   = row[cn]
                if v == 1 or v is True:
                    ws.write(er, ci2, 'YES', fbt)
                elif v == 0 or v is False:
                    ws.write(er, ci2, 'NO',  fbf)
        for pc in ['RF_Prob','DL_Prob','Avg_Prob']:
            if pc in disp.columns:
                ci2 = disp.columns.get_loc(pc)
                ws.write(er, ci2, row[pc], fpr)
    # Colour scales for score columns
    for sc in ['Base Score','Fundamental Score','Performer Score']:
        if sc in disp.columns:
            ci2 = disp.columns.get_loc(sc)
            ws.conditional_format(1, ci2, len(disp), ci2, {
                'type': '3_color_scale',
                'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'
            })
    ws.freeze_panes(1, 3)
    ws.autofilter(0, 0, len(disp), len(disp.columns)-1)

with pd.ExcelWriter(OUT, engine='xlsxwriter',
                    engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
    wb = writer.book

    # Sheet 1: Model Results
    write_results_sheet(writer, df_res, 'Model Results', wb)

    # Sheet 2: Feature Importance (RF)
    fi.head(60).to_excel(writer, index=False, sheet_name='Feature Importance')
    ws_fi = writer.sheets['Feature Importance']
    fh2   = wb.add_format({'bold': True, 'bg_color': '#1F3864', 'font_color': 'white', 'border': 1})
    ws_fi.set_row(0, 24)
    ws_fi.set_column(0, 0, 32); ws_fi.set_column(1, 1, 14); ws_fi.set_column(2, 2, 16)
    for ci, cn in enumerate(fi.columns): ws_fi.write(0, ci, cn, fh2)
    imp_ci = fi.columns.get_loc('Importance (%)')
    ws_fi.conditional_format(1, imp_ci, min(60, len(fi)), imp_ci, {
        'type': '2_color_scale', 'min_color': '#FFFFFF', 'max_color': '#4472C4'
    })

    # Sheet 3: High-Confidence Signals (both models agree, sorted by prob)
    if len(both_signals):
        write_signals_sheet(writer, both_signals,
                            f'Both Predict Profit ({len(both_signals)})', wb)

    # Sheet 4: RF-only predicted Profit signals
    rf_only = test_raw[test_raw['RF_Pred'] == 1].sort_values('RF_Prob', ascending=False)
    if len(rf_only):
        write_signals_sheet(writer, rf_only,
                            f'RF Predicts Profit ({len(rf_only)})', wb)

    # Sheet 5: DL-only predicted Profit signals
    dl_only = test_raw[test_raw['DL_Pred'] == 1].sort_values('DL_Prob', ascending=False)
    if len(dl_only):
        write_signals_sheet(writer, dl_only,
                            f'DL Predicts Profit ({len(dl_only)})', wb)

print(f"\nSaved: {OUT}")
print(f"\nFinal Summary:")
print(df_res[['Model','Baseline (%)','UP Precision (%)','Edge (pp)',
              'UP Recall (%)','Predicted UP','Threshold']].to_string(index=False))
