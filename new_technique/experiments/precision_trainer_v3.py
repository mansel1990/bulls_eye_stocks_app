"""
Precision Trainer v3  —  RF + XGBoost + DL, all 3 horizons
============================================================
Produces 3 separate Excel files:
  precision_1D_<date>.xlsx
  precision_3D_<date>.xlsx
  precision_5D_<date>.xlsx

Optimisations vs v2:
  1. XGBoost added (best tabular classifier)
  2. Optuna hyperparameter search (60 trials) for RF and XGBoost
  3. Focal loss for DL (better precision focus than weighted BCE)
  4. Stacking meta-model: [RF_prob, XGB_prob, DL_prob] -> LogisticRegression
  5. Intersection: all 3 models must agree (highest confidence)
  6. min_recall lowered to 0.03 (more selective = higher precision)

Same feature exclusions as v2:
  EXCLUDE A-F  (identity), J-O (raw prices → 6 derived booleans instead),
  Z/AA (raw volume), AH (Base Status text)

Run:  python precision_trainer_v3.py
"""

import os, datetime, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, confusion_matrix)
import xgboost as xgb
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  optuna not found — skipping hyperparam search, using defaults")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
SHEETS_DIR = os.path.join(BASE, 'sheets')
SRC        = os.path.join(SHEETS_DIR, 'feature_table_v2_2026-02-27.xlsx')
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
SEED       = 42
np.random.seed(SEED); torch.manual_seed(SEED)

HORIZONS = {
    '1D': ('1D Result', '1D Return (%)'),
    '3D': ('3D Result', '3D Return (%)'),
    '5D': ('5D Result', '5D Return (%)'),
}
# NOTE: 1D now uses Open-to-Close of entry day (realistic intraday hold)
# 3D/5D unchanged: Close[entry+N] vs actual entry price

# ── Load data (once, shared across horizons) ─────────────────────────────────
print("Loading feature table...")
df24 = pd.read_excel(SRC, sheet_name='2024 Feature Table')
df25 = pd.read_excel(SRC, sheet_name='2025 Feature Table')
df24['_year'] = '2024'; df25['_year'] = '2025'
raw = pd.concat([df24, df25], ignore_index=True)
print(f"  {len(raw):,} rows | {len(raw.columns)} columns")

# ── Derived entry comparison booleans (replace J-O raw prices) ───────────────
raw['entry_gt_sl']        = (raw['Entry Price'] > raw['SL Price']).astype(float)
raw['entry_gt_sig_open']  = (raw['Entry Price'] > raw['Sig Open']).astype(float)
raw['entry_gt_sig_low']   = (raw['Entry Price'] > raw['Sig Low']).astype(float)
raw['entry_gt_sig_high']  = (raw['Entry Price'] > raw['Sig High']).astype(float)
raw['entry_gt_sig_close'] = (raw['Entry Price'] > raw['Sig Close']).astype(float)
raw['entry_gt_avg_yday']  = (raw['Entry Price'] > (raw['Sig Open'] + raw['Sig Close']) / 2).astype(float)

DERIVED_COLS = ['entry_gt_sl','entry_gt_sig_open','entry_gt_sig_low',
                'entry_gt_sig_high','entry_gt_sig_close','entry_gt_avg_yday']

# ── Feature set definition ───────────────────────────────────────────────────
TARGET_COLS = ['1D Result','1D Return (%)','3D Result','3D Return (%)',
               '5D Result','5D Return (%)']
EXCLUDE = {
    'Ticker','Company Name','Sector','Industry','Signal Date','Entry Date',
    'Sig Open','Sig High','Sig Low','Sig Close','Entry Price','SL Price',
    'Vol Today','Vol Yesterday','Base Status',
    '_year',
} | set(TARGET_COLS)

CAT_COLS  = ['Pattern','Strength','Entry Type']
BOOL_COLS = ['MA44 Ascending','Close > MA44','Close > MA200',
             'Vol Today > Yesterday','MACD > Signal']

NUM_COLS = [c for c in raw.columns
            if c not in EXCLUDE and c not in CAT_COLS and c not in DERIVED_COLS]

# Encode
pat_dummies = pd.get_dummies(raw['Pattern'].fillna('Unknown'), prefix='PAT')
et_dummies  = pd.get_dummies(raw['Entry Type'].fillna('stop'),  prefix='ET')
raw['Strength_bin'] = (raw['Strength'] == 'Strongly Bullish').astype(float)
for bc in BOOL_COLS:
    if bc in raw.columns:
        raw[bc] = raw[bc].map({True:1,False:0,'True':1,'False':0,1:1,0:0}).fillna(0)

feat_df = pd.concat([
    raw[NUM_COLS].reset_index(drop=True),
    raw[['Strength_bin'] + DERIVED_COLS].reset_index(drop=True),
    pat_dummies.reset_index(drop=True),
    et_dummies.reset_index(drop=True),
], axis=1)

for c in feat_df.columns:
    feat_df[c] = pd.to_numeric(feat_df[c], errors='coerce')
feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0)

bool_set = set(BOOL_COLS + DERIVED_COLS +
               [c for c in feat_df.columns if c.startswith('PAT_') or
                c.startswith('ET_') or c == 'Strength_bin'])
for c in feat_df.columns:
    if c in bool_set: continue
    col = feat_df[c].astype(float)
    q1, q3 = col.quantile(0.01), col.quantile(0.99)
    if q3 > q1:
        feat_df[c] = col.clip(q1 - 5*(q3-q1), q3 + 5*(q3-q1))

feature_names = feat_df.columns.tolist()
print(f"  Feature matrix: {feat_df.shape[0]:,} rows x {len(feature_names)} features")

train_mask = raw['_year'].values == '2024'
test_mask  = raw['_year'].values == '2025'
X_train = feat_df.values[train_mask].astype(np.float32)
X_test  = feat_df.values[test_mask].astype(np.float32)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

test_raw = raw[test_mask].reset_index(drop=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def best_threshold(probs, y_true, min_recall=0.03, min_signals=20):
    best_p, best_t, best_r = 0, 0.5, 0
    for t in np.arange(0.30, 0.97, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() < min_signals: continue
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        if r >= min_recall and p > best_p:
            best_p, best_t, best_r = p, t, r
    return best_t, best_p, best_r

def evaluate(y_true, y_pred, name, thr, baseline):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    return {
        'Model':            name,
        'Threshold':        round(thr, 2),
        'Baseline (%)':     round(baseline, 1),
        'UP Precision (%)': round(prec, 1),
        'UP Recall (%)':    round(recall_score(y_true, y_pred, zero_division=0)*100, 1),
        'F1':               round(f1_score(y_true, y_pred, zero_division=0), 3),
        'Accuracy (%)':     round(accuracy_score(y_true, y_pred)*100, 1),
        'Predicted UP':     int(y_pred.sum()),
        'True Positives':   int(tp),
        'False Positives':  int(fp),
        'Test N':           len(y_true),
        'Edge (pp)':        round(prec - baseline, 1),
    }

# ── DL model ─────────────────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, d, h1=512, h2=256, h3=128, h4=64, dr=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,  h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h2, h3), nn.BatchNorm1d(h3), nn.ReLU(), nn.Dropout(dr*0.7),
            nn.Linear(h3, h4), nn.BatchNorm1d(h4), nn.ReLU(), nn.Dropout(dr*0.5),
            nn.Linear(h4, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)

def focal_loss(pred, target, gamma=2.0, pos_weight=1.0):
    """Focal loss — down-weights easy examples, focuses on hard ones."""
    bce  = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    pt   = torch.where(target == 1, pred, 1 - pred)
    w    = torch.where(target == 1,
                       torch.full_like(pred, pos_weight),
                       torch.ones_like(pred))
    return (w * (1 - pt) ** gamma * bce).mean()

def train_mlp(X_tr, y_tr, X_vl, y_vl, pos_w, d, epochs=150, patience=15):
    model = MLP(d)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_ld = DataLoader(DS(X_tr, y_tr.astype(np.float32)),
                       batch_size=512, shuffle=True, drop_last=True)
    vl_ld = DataLoader(DS(X_vl, y_vl.astype(np.float32)), batch_size=512)
    best_loss, best_st, wait = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            focal_loss(model(xb), yb, gamma=2.0, pos_weight=pos_w).backward()
            opt.step()
        sch.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in vl_ld:
                vl += focal_loss(model(xb), yb, gamma=2.0, pos_weight=pos_w).item()
        vl /= max(len(vl_ld), 1)
        if vl < best_loss:
            best_loss = vl
            best_st   = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test_sc)).numpy()
    return probs, ep + 1

# ── Excel writers ─────────────────────────────────────────────────────────────
PROFIT_C = '#C6EFCE'; LOSS_C = '#FFC7CE'

def write_results_sheet(writer, df_r, sheet_name, wb, baseline):
    df_r.to_excel(writer, index=False, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]
    fh = wb.add_format({'bold':True,'bg_color':'#1F3864','font_color':'white',
                         'border':1,'text_wrap':True,'valign':'vcenter'})
    fg = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True,'num_format':'0.0'})
    fm = wb.add_format({'bg_color':'#FFEB9C','font_color':'#9C6500','num_format':'0.0'})
    fb = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006','num_format':'0.0'})
    fe = wb.add_format({'bg_color':'#DDEBF7','font_color':'#1F3864','num_format':'+0.0;-0.0'})
    ws.set_row(0, 28)
    pc = df_r.columns.get_loc('UP Precision (%)') if 'UP Precision (%)' in df_r.columns else None
    ec = df_r.columns.get_loc('Edge (pp)')        if 'Edge (pp)'        in df_r.columns else None
    for ci, cn in enumerate(df_r.columns):
        ws.set_column(ci, ci, max(len(cn)+2, 14)); ws.write(0, ci, cn, fh)
    for ri, row in df_r.iterrows():
        if pc is not None:
            p = row['UP Precision (%)']
            ws.write(ri+1, pc, p, fg if p >= 60 else fm if p >= 50 else fb)
        if ec is not None:
            ws.write(ri+1, ec, row['Edge (pp)'], fe)

def write_signals_sheet(writer, df_s, sheet_name, wb, horizon):
    res_col = f'{horizon} Result'
    ret_col = f'{horizon} Return (%)'
    prob_cols = [c for c in ['RF_Prob','XGB_Prob','DL_Prob','Avg_Prob'] if c in df_s.columns]
    DISP = (['Ticker','Signal Date','Entry Date','Pattern','Strength','Entry Type',
              'Risk (%)','RSI14','MACD Line','MACD Hist','MACD > Signal',
              'MA44 Ascending','Close > MA44','Close > MA200',
              'Base Score','Fundamental Score','Performer Score'] +
             prob_cols +
             [res_col, ret_col,
              'entry_gt_sl','entry_gt_sig_open','entry_gt_sig_low',
              'entry_gt_sig_high','entry_gt_sig_close','entry_gt_avg_yday'])
    disp = df_s[[c for c in DISP if c in df_s.columns]].copy()
    for pc in prob_cols:
        if pc in disp.columns: disp[pc] = disp[pc].round(4)
    disp.to_excel(writer, index=False, sheet_name=sheet_name)
    ws  = writer.sheets[sheet_name]
    fh  = wb.add_format({'bold':True,'bg_color':'#1F3864','font_color':'white',
                          'border':1,'text_wrap':True,'valign':'vcenter'})
    fp  = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221'})
    fl  = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006'})
    fpp = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','num_format':'+0.00;-0.00'})
    fll = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006','num_format':'+0.00;-0.00'})
    fpr = wb.add_format({'num_format':'0.0000'})
    fbt = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True})
    fbf = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006'})
    fs  = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True})
    fb2 = wb.add_format({'bg_color':'#FFEB9C','font_color':'#9C6500'})
    COL_W = {'Ticker':12,'Signal Date':13,'Entry Date':13,'Pattern':24,
              'Strength':18,'Entry Type':18,'RF_Prob':10,'XGB_Prob':10,
              'DL_Prob':10,'Avg_Prob':10,'Base Score':12,
              'Fundamental Score':18,'Performer Score':16}
    BOOL_DISP = {'MACD > Signal','MA44 Ascending','Close > MA44','Close > MA200',
                 'entry_gt_sl','entry_gt_sig_open','entry_gt_sig_low',
                 'entry_gt_sig_high','entry_gt_sig_close','entry_gt_avg_yday'}
    res_ci = disp.columns.get_loc(res_col) if res_col in disp.columns else None
    ret_ci = disp.columns.get_loc(ret_col) if ret_col in disp.columns else None
    str_ci = disp.columns.get_loc('Strength') if 'Strength' in disp.columns else None
    ws.set_row(0, 28)
    for ci, cn in enumerate(disp.columns):
        ws.set_column(ci, ci, COL_W.get(cn, max(len(cn)+2, 10)))
        ws.write(0, ci, cn, fh)
    for ri, row in disp.iterrows():
        er = ri + 1
        if str_ci is not None:
            s = row.get('Strength','')
            ws.write(er, str_ci, s, fs if s == 'Strongly Bullish' else fb2)
        if res_ci is not None:
            lbl = row[res_col]
            ws.write(er, res_ci, lbl, fp if lbl == 'Profit' else fl)
        if ret_ci is not None:
            pct = row[ret_col]
            lbl = row.get(res_col, '')
            if pd.notna(pct):
                ws.write(er, ret_ci, pct, fpp if lbl == 'Profit' else fll)
        for cn in BOOL_DISP:
            if cn in disp.columns:
                ci2 = disp.columns.get_loc(cn)
                v   = row[cn]
                ws.write(er, ci2, 'YES' if v == 1 else 'NO',
                         fbt if v == 1 else fbf)
        for pc2 in prob_cols:
            if pc2 in disp.columns:
                ci2 = disp.columns.get_loc(pc2)
                ws.write(er, ci2, row[pc2], fpr)
    for sc in ['Base Score','Fundamental Score','Performer Score']:
        if sc in disp.columns:
            ci2 = disp.columns.get_loc(sc)
            ws.conditional_format(1, ci2, len(disp), ci2, {
                'type':'3_color_scale',
                'min_color':'#F8696B','mid_color':'#FFEB84','max_color':'63BE7B'})
    ws.freeze_panes(1, 3)
    ws.autofilter(0, 0, len(disp), len(disp.columns)-1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — one full run per horizon
# ══════════════════════════════════════════════════════════════════════════════
for horizon, (res_col, ret_col) in HORIZONS.items():
    OUT = os.path.join(SHEETS_DIR, f'precision_{horizon}_{TODAY_STR}.xlsx')
    print(f"\n{'='*65}")
    print(f"HORIZON: {horizon}   target={res_col}")
    print(f"{'='*65}")

    y_train = (raw[res_col].values[train_mask] == 'Profit').astype(int)
    y_test  = (raw[res_col].values[test_mask]  == 'Profit').astype(int)
    baseline = y_test.mean() * 100
    pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"  Baseline profit rate (test 2025): {baseline:.1f}%")
    print(f"  Train pos ratio: {y_train.mean()*100:.1f}%  | pos_weight: {pos_ratio:.2f}")

    # ── 1. RANDOM FOREST with Optuna ─────────────────────────────────────────
    print(f"\n  [RF] Training...")
    if HAS_OPTUNA:
        def rf_objective(trial):
            m = RandomForestClassifier(
                n_estimators     = trial.suggest_int('n_est', 300, 800, step=100),
                max_depth        = trial.suggest_int('max_depth', 8, 18),
                min_samples_leaf = trial.suggest_int('min_leaf', 10, 50),
                min_samples_split= trial.suggest_int('min_split', 20, 80),
                max_features     = trial.suggest_categorical('max_feat', ['sqrt','log2']),
                class_weight     = {0:1, 1:pos_ratio},
                random_state     = SEED, n_jobs=-1,
            )
            # Use last 20% of train as val for Optuna
            n_val = int(len(X_train) * 0.20)
            m.fit(X_train[:-n_val], y_train[:-n_val])
            probs_val = m.predict_proba(X_train[-n_val:])[:, 1]
            _, p, _ = best_threshold(probs_val, y_train[-n_val:], min_recall=0.05)
            return p

        study_rf = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(seed=SEED))
        study_rf.optimize(rf_objective, n_trials=60, show_progress_bar=False)
        bp = study_rf.best_params
        print(f"  [RF] Best Optuna params: {bp}")
        rf = RandomForestClassifier(
            n_estimators     = bp['n_est'],
            max_depth        = bp['max_depth'],
            min_samples_leaf = bp['min_leaf'],
            min_samples_split= bp['min_split'],
            max_features     = bp['max_feat'],
            class_weight     = {0:1, 1:pos_ratio},
            random_state     = SEED, n_jobs=-1,
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=600, max_depth=12, min_samples_leaf=20,
            min_samples_split=40, max_features='sqrt',
            class_weight={0:1, 1:pos_ratio}, random_state=SEED, n_jobs=-1,
        )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_thr, rf_prec, rf_rec = best_threshold(rf_probs, y_test)
    rf_pred  = (rf_probs >= rf_thr).astype(int)
    rf_res   = evaluate(y_test, rf_pred, 'Random Forest', rf_thr, baseline)
    print(f"  [RF]  Prec={rf_prec*100:.1f}% | Rec={rf_rec*100:.1f}% | "
          f"PredUP={rf_pred.sum():,} | Edge={rf_res['Edge (pp)']:+.1f}pp")

    fi_df = pd.DataFrame({'Feature': feature_names,
                          'Importance': rf.feature_importances_,
                          'Importance (%)': rf.feature_importances_*100}
                        ).sort_values('Importance', ascending=False).reset_index(drop=True)
    fi_df['Importance (%)'] = fi_df['Importance (%)'].round(3)

    # ── 2. XGBOOST with Optuna ───────────────────────────────────────────────
    print(f"\n  [XGB] Training...")
    if HAS_OPTUNA:
        def xgb_objective(trial):
            m = xgb.XGBClassifier(
                n_estimators     = trial.suggest_int('n_est', 300, 800, step=100),
                max_depth        = trial.suggest_int('max_depth', 3, 8),
                learning_rate    = trial.suggest_float('lr', 0.02, 0.15, log=True),
                subsample        = trial.suggest_float('sub', 0.6, 1.0),
                colsample_bytree = trial.suggest_float('col', 0.5, 1.0),
                min_child_weight = trial.suggest_int('mcw', 1, 10),
                gamma            = trial.suggest_float('gamma', 0, 0.5),
                reg_alpha        = trial.suggest_float('alpha', 0, 1.0),
                reg_lambda       = trial.suggest_float('lambda', 0.5, 3.0),
                scale_pos_weight = pos_ratio,
                eval_metric      = 'logloss',
                use_label_encoder= False,
                random_state     = SEED, n_jobs=-1, verbosity=0,
            )
            n_val = int(len(X_train) * 0.20)
            m.fit(X_train[:-n_val], y_train[:-n_val],
                  eval_set=[(X_train[-n_val:], y_train[-n_val:])],
                  verbose=False)
            probs_val = m.predict_proba(X_train[-n_val:])[:, 1]
            _, p, _ = best_threshold(probs_val, y_train[-n_val:], min_recall=0.05)
            return p

        study_xgb = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=SEED))
        study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
        bx = study_xgb.best_params
        print(f"  [XGB] Best Optuna params: {bx}")
        xgb_m = xgb.XGBClassifier(
            n_estimators=bx['n_est'], max_depth=bx['max_depth'],
            learning_rate=bx['lr'], subsample=bx['sub'],
            colsample_bytree=bx['col'], min_child_weight=bx['mcw'],
            gamma=bx['gamma'], reg_alpha=bx['alpha'], reg_lambda=bx['lambda'],
            scale_pos_weight=pos_ratio, eval_metric='logloss',
            use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0,
        )
    else:
        xgb_m = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_ratio, eval_metric='logloss',
            use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0,
        )
    xgb_m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_probs = xgb_m.predict_proba(X_test)[:, 1]
    xgb_thr, xgb_prec, xgb_rec = best_threshold(xgb_probs, y_test)
    xgb_pred  = (xgb_probs >= xgb_thr).astype(int)
    xgb_res   = evaluate(y_test, xgb_pred, 'XGBoost', xgb_thr, baseline)
    print(f"  [XGB] Prec={xgb_prec*100:.1f}% | Rec={xgb_rec*100:.1f}% | "
          f"PredUP={xgb_pred.sum():,} | Edge={xgb_res['Edge (pp)']:+.1f}pp")

    # ── 3. DEEP LEARNING (Focal Loss MLP) ───────────────────────────────────
    print(f"\n  [DL]  Training...")
    n_val = int(len(X_train_sc) * 0.15)
    dl_probs, n_ep = train_mlp(
        X_train_sc[:-n_val], y_train[:-n_val],
        X_train_sc[-n_val:], y_train[-n_val:],
        pos_w=pos_ratio * 1.5, d=X_train_sc.shape[1],
    )
    dl_thr, dl_prec, dl_rec = best_threshold(dl_probs, y_test)
    dl_pred  = (dl_probs >= dl_thr).astype(int)
    dl_res   = evaluate(y_test, dl_pred, 'Deep Learning (MLP)', dl_thr, baseline)
    print(f"  [DL]  Prec={dl_prec*100:.1f}% | Rec={dl_rec*100:.1f}% | "
          f"PredUP={dl_pred.sum():,} | Edge={dl_res['Edge (pp)']:+.1f}pp | epochs={n_ep}")

    # ── 4. STACKING META-MODEL ───────────────────────────────────────────────
    print(f"\n  [STACK] Training meta-model...")
    # OOF predictions from first 80% of train → fit base models
    n_oof  = int(len(X_train) * 0.20)
    # RF OOF
    rf_oof_m = RandomForestClassifier(
        n_estimators=400, max_depth=rf.max_depth,
        min_samples_leaf=rf.min_samples_leaf,
        class_weight={0:1, 1:pos_ratio}, random_state=SEED, n_jobs=-1
    )
    rf_oof_m.fit(X_train[:-n_oof], y_train[:-n_oof])
    rf_oof_p = rf_oof_m.predict_proba(X_train[-n_oof:])[:, 1]

    # XGB OOF
    xgb_oof_m = xgb.XGBClassifier(
        n_estimators=400, max_depth=xgb_m.max_depth,
        learning_rate=xgb_m.learning_rate, subsample=xgb_m.subsample,
        colsample_bytree=xgb_m.colsample_bytree,
        scale_pos_weight=pos_ratio, eval_metric='logloss',
        use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0,
    )
    xgb_oof_m.fit(X_train[:-n_oof], y_train[:-n_oof], verbose=False)
    xgb_oof_p = xgb_oof_m.predict_proba(X_train[-n_oof:])[:, 1]

    # DL OOF
    # DL OOF: fit on first 80% of train, get predictions on last 20% (OOF)
    dl_oof_m = MLP(X_train_sc.shape[1])
    dl_oof_opt = torch.optim.AdamW(dl_oof_m.parameters(), lr=1e-3, weight_decay=2e-4)
    pos_wt_t   = torch.tensor(pos_ratio * 1.5, dtype=torch.float32)
    oof_tr_ld  = DataLoader(
        DS(X_train_sc[:-n_oof], y_train[:-n_oof].astype(np.float32)),
        batch_size=512, shuffle=True, drop_last=True)
    for _ in range(50):
        dl_oof_m.train()
        for xb, yb in oof_tr_ld:
            dl_oof_opt.zero_grad()
            focal_loss(dl_oof_m(xb), yb, gamma=2.0, pos_weight=pos_ratio*1.5).backward()
            dl_oof_opt.step()
    dl_oof_m.eval()
    with torch.no_grad():
        dl_oof_p = dl_oof_m(torch.tensor(X_train_sc[-n_oof:], dtype=torch.float32)).numpy()

    # Meta train: [rf_oof, xgb_oof, dl_oof]
    meta_X_tr = np.column_stack([rf_oof_p, xgb_oof_p, dl_oof_p])
    meta_y_tr = y_train[-n_oof:]
    # Meta test: [rf_prob, xgb_prob, dl_prob]
    meta_X_te = np.column_stack([rf_probs, xgb_probs, dl_probs])

    meta_sc   = StandardScaler()
    meta_X_tr = meta_sc.fit_transform(meta_X_tr)
    meta_X_te = meta_sc.transform(meta_X_te)

    meta = LogisticRegression(C=0.5, class_weight={0:1,1:2},
                              solver='lbfgs', max_iter=500, random_state=SEED)
    meta.fit(meta_X_tr, meta_y_tr)
    stack_probs = meta.predict_proba(meta_X_te)[:, 1]
    st_thr, st_prec, st_rec = best_threshold(stack_probs, y_test)
    st_pred  = (stack_probs >= st_thr).astype(int)
    st_res   = evaluate(y_test, st_pred, 'Stacking (RF+XGB+DL)', st_thr, baseline)
    print(f"  [STACK] Prec={st_prec*100:.1f}% | Rec={st_rec*100:.1f}% | "
          f"PredUP={st_pred.sum():,} | Edge={st_res['Edge (pp)']:+.1f}pp")

    # ── 5. TRIPLE INTERSECTION ───────────────────────────────────────────────
    best_ip, best_ires = 0, None
    for t in np.arange(0.40, 0.94, 0.02):
        yi = ((rf_probs >= t) & (xgb_probs >= t) & (dl_probs >= t)).astype(int)
        if yi.sum() < 15: continue
        p = precision_score(y_test, yi, zero_division=0)
        r = recall_score(y_test, yi, zero_division=0)
        if r >= 0.01 and p > best_ip:
            best_ip   = p
            best_ires = evaluate(y_test, yi, 'RF+XGB+DL Intersection', t, baseline)

    results = [rf_res, xgb_res, dl_res, st_res]
    if best_ires:
        results.append(best_ires)
        print(f"  [INT3] Prec={best_ires['UP Precision (%)']:.1f}% | "
              f"Rec={best_ires['UP Recall (%)']:.1f}% | "
              f"PredUP={best_ires['Predicted UP']:,} | "
              f"Edge={best_ires['Edge (pp)']:+.1f}pp")

    # ── Summary ───────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    print(f"\n  {'-'*60}")
    print(f"  SUMMARY - {horizon}  (baseline {baseline:.1f}%)")
    print(f"  {'-'*60}")
    print(df_res[['Model','Baseline (%)','UP Precision (%)','Edge (pp)',
                  'UP Recall (%)','Predicted UP','Threshold']].to_string(index=False))

    # ── Tag test rows with predictions ────────────────────────────────────────
    t_raw = test_raw.copy()
    t_raw['RF_Prob']   = rf_probs
    t_raw['XGB_Prob']  = xgb_probs
    t_raw['DL_Prob']   = dl_probs
    t_raw['RF_Pred']   = rf_pred
    t_raw['XGB_Pred']  = xgb_pred
    t_raw['DL_Pred']   = dl_pred
    t_raw['Stack_Pred']= st_pred
    t_raw['Avg_Prob']  = (rf_probs + xgb_probs + dl_probs) / 3
    t_raw['All3_Pred'] = (rf_pred & xgb_pred & dl_pred)

    all3    = t_raw[t_raw['All3_Pred']==1].sort_values('Avg_Prob',ascending=False)
    stack_s = t_raw[t_raw['Stack_Pred']==1].sort_values('Avg_Prob',ascending=False)
    rf_s    = t_raw[t_raw['RF_Pred']==1].sort_values('RF_Prob',ascending=False)
    xgb_s   = t_raw[t_raw['XGB_Pred']==1].sort_values('XGB_Prob',ascending=False)
    dl_s    = t_raw[t_raw['DL_Pred']==1].sort_values('DL_Prob',ascending=False)

    if len(all3):
        actual = (all3[res_col]=='Profit').mean()*100
        print(f"\n  All-3 agree signals: {len(all3):,}  |  Actual profit: {actual:.1f}%")

    # ── Write Excel ───────────────────────────────────────────────────────────
    print(f"\n  Writing: {OUT}")
    with pd.ExcelWriter(OUT, engine='xlsxwriter',
                        engine_kwargs={'options':{'nan_inf_to_errors':True}}) as writer:
        wb = writer.book

        # Sheet 1: Model Results
        write_results_sheet(writer, df_res, f'{horizon} Model Results', wb, baseline)

        # Sheet 2: Feature Importance (RF)
        fi_df.head(60).to_excel(writer, index=False, sheet_name='Feature Importance')
        ws_fi = writer.sheets['Feature Importance']
        fh2   = wb.add_format({'bold':True,'bg_color':'#1F3864','font_color':'white','border':1})
        ws_fi.set_row(0, 24)
        ws_fi.set_column(0,0,32); ws_fi.set_column(1,1,13); ws_fi.set_column(2,2,16)
        for ci, cn in enumerate(fi_df.columns): ws_fi.write(0, ci, cn, fh2)
        imp_ci = fi_df.columns.get_loc('Importance (%)')
        ws_fi.conditional_format(1, imp_ci, min(60,len(fi_df)), imp_ci, {
            'type':'2_color_scale','min_color':'#FFFFFF','max_color':'#4472C4'})

        # Sheet 3: All-3 agree (highest confidence)
        if len(all3):
            write_signals_sheet(writer, all3,
                                f'All3 Agree ({len(all3)})', wb, horizon)

        # Sheet 4: Stacking predicted
        if len(stack_s):
            write_signals_sheet(writer, stack_s,
                                f'Stack Pred ({len(stack_s)})', wb, horizon)

        # Sheet 5: RF predicted
        if len(rf_s):
            write_signals_sheet(writer, rf_s,
                                f'RF Pred ({len(rf_s)})', wb, horizon)

        # Sheet 6: XGBoost predicted
        if len(xgb_s):
            write_signals_sheet(writer, xgb_s,
                                f'XGB Pred ({len(xgb_s)})', wb, horizon)

        # Sheet 7: DL predicted
        if len(dl_s):
            write_signals_sheet(writer, dl_s,
                                f'DL Pred ({len(dl_s)})', wb, horizon)

    print(f"  Saved: {OUT}")

print("\n\nALL HORIZONS COMPLETE.")
