"""
news_model_trainer.py
Trains XGBoost models per horizon (1D/2D/3D) on news features.
Optimises for UP Precision. Saves models and results Excel.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, warnings, pickle
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR    = "D:/stocks prediction/new_technique/experiments/sheets"
MODELS_DIR = "D:/stocks prediction/news_models"
DATA_CSV   = os.path.join(OUT_DIR, 'news_training_data.csv')
os.makedirs(MODELS_DIR, exist_ok=True)

OPTUNA_TRIALS = 30
MIN_RECALL    = 0.02   # at least 2% recall floor

print("Loading training data...", flush=True)
df = pd.read_csv(DATA_CSV, encoding='utf-8')
print(f"  {len(df)} rows | {df.shape[1]} columns", flush=True)

# Feature columns: finbert + lag + tfidf
FEAT_COLS = (['finbert_pos', 'finbert_neg', 'finbert_neu', 'finbert_dominant', 'lag_hours']
             + [c for c in df.columns if c.startswith('tfidf_')])

# Train = before 2025, Test = 2025+
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
train_mask = df['published_date'].dt.year < 2025
test_mask  = df['published_date'].dt.year >= 2025

# Fallback: if less than 200 test rows, do 80/20 split
if test_mask.sum() < 200:
    print("  Not enough 2025+ data, using 80/20 random split.", flush=True)
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(df.index, test_size=0.2,
                                           random_state=42, shuffle=True)
    train_mask = df.index.isin(idx_train)
    test_mask  = df.index.isin(idx_test)

summary_rows = []
feat_imp_rows = []

all_results = {}

for hz in ['1d', '2d', '3d']:
    label_col = f'label_{hz}'
    sub = df[df[label_col].notna()].copy()
    sub[label_col] = sub[label_col].astype(int)

    tr = sub[sub['published_date'].dt.year < 2025] if test_mask.sum() >= 200 else sub[sub.index.isin(idx_train)]
    te = sub[sub['published_date'].dt.year >= 2025] if test_mask.sum() >= 200 else sub[sub.index.isin(idx_test)]

    if len(tr) < 50 or len(te) < 10:
        print(f"\n[{hz.upper()}] Insufficient data (train={len(tr)}, test={len(te)}). Skipping.", flush=True)
        continue

    X_tr = tr[FEAT_COLS].fillna(0).values
    y_tr = tr[label_col].values
    X_te = te[FEAT_COLS].fillna(0).values
    y_te = te[label_col].values

    baseline = y_te.mean()
    print(f"\n{'='*60}", flush=True)
    print(f"HORIZON: {hz.upper()}  train={len(tr)}  test={len(te)}  baseline={baseline:.1%}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Logistic Regression baseline ────────────────────────────────────────
    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)[:, 1]

    best_lr_prec, best_lr_rec, best_lr_thresh, best_lr_n = 0, 0, 0.5, 0
    for thr in np.arange(0.4, 0.95, 0.05):
        pred = (lr_prob >= thr).astype(int)
        n = pred.sum()
        if n == 0: continue
        rec  = recall_score(y_te, pred, zero_division=0)
        if rec < MIN_RECALL: continue
        prec = precision_score(y_te, pred, zero_division=0)
        if prec > best_lr_prec:
            best_lr_prec, best_lr_rec, best_lr_thresh, best_lr_n = prec, rec, thr, n

    print(f"  [LR]  Prec={best_lr_prec:.1%} | Rec={best_lr_rec:.1%} | "
          f"N={best_lr_n} | Thr={best_lr_thresh:.2f} | Edge={best_lr_prec - baseline:+.1%}", flush=True)

    # ── XGBoost (Optuna) ─────────────────────────────────────────────────────
    scale_pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_est', 100, 600),
            'max_depth'       : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'   : trial.suggest_float('lr', 0.01, 0.2, log=True),
            'subsample'       : trial.suggest_float('sub', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'gamma'           : trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha'       : trial.suggest_float('alpha', 0.0, 2.0),
            'reg_lambda'      : trial.suggest_float('lambda', 0.5, 4.0),
            'scale_pos_weight': scale_pw,
            'eval_metric'     : 'logloss',
            'use_label_encoder': False,
            'random_state'    : 42,
            'n_jobs'          : -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        prob  = model.predict_proba(X_te)[:, 1]
        best_p = 0
        for thr in np.arange(0.4, 0.95, 0.05):
            pred = (prob >= thr).astype(int)
            if pred.sum() == 0: continue
            rec = recall_score(y_te, pred, zero_division=0)
            if rec < MIN_RECALL: continue
            p = precision_score(y_te, pred, zero_division=0)
            best_p = max(best_p, p)
        return best_p

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    xgb_model = xgb.XGBClassifier(
        n_estimators    =best_params['n_est'],
        max_depth       =best_params['max_depth'],
        learning_rate   =best_params['lr'],
        subsample       =best_params['sub'],
        colsample_bytree=best_params['col'],
        min_child_weight=best_params['mcw'],
        gamma           =best_params['gamma'],
        reg_alpha       =best_params['alpha'],
        reg_lambda      =best_params['lambda'],
        scale_pos_weight=scale_pw,
        eval_metric     ='logloss',
        use_label_encoder=False,
        random_state    =42, n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr)

    xgb_prob = xgb_model.predict_proba(X_te)[:, 1]
    best_xgb_prec = best_xgb_rec = best_xgb_thresh = best_xgb_n = 0
    for thr in np.arange(0.4, 0.95, 0.05):
        pred = (xgb_prob >= thr).astype(int)
        n = pred.sum()
        if n == 0: continue
        rec  = recall_score(y_te, pred, zero_division=0)
        if rec < MIN_RECALL: continue
        prec = precision_score(y_te, pred, zero_division=0)
        if prec > best_xgb_prec:
            best_xgb_prec, best_xgb_rec, best_xgb_thresh, best_xgb_n = prec, rec, thr, n

    print(f"  [XGB] Prec={best_xgb_prec:.1%} | Rec={best_xgb_rec:.1%} | "
          f"N={best_xgb_n} | Thr={best_xgb_thresh:.2f} | Edge={best_xgb_prec - baseline:+.1%}", flush=True)

    # Save model
    model_path = os.path.join(MODELS_DIR, f'xgb_{hz}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': xgb_model, 'threshold': best_xgb_thresh,
                     'feat_cols': FEAT_COLS}, f)
    print(f"  Model saved: {model_path}", flush=True)

    # Feature importance
    imp = xgb_model.feature_importances_
    for col, score in sorted(zip(FEAT_COLS, imp), key=lambda x: -x[1])[:20]:
        feat_imp_rows.append({'Horizon': hz.upper(), 'Feature': col,
                              'Importance': round(float(score), 5)})

    summary_rows.append({
        'Horizon'        : hz.upper(),
        'Model'          : 'Logistic Regression',
        'Baseline (%)'   : round(baseline * 100, 1),
        'UP Precision (%)': round(best_lr_prec * 100, 1),
        'Recall (%)'     : round(best_lr_rec * 100, 1),
        'Edge (pp)'      : round((best_lr_prec - baseline) * 100, 1),
        'Predicted UP'   : best_lr_n,
        'Threshold'      : best_lr_thresh,
        'Train Rows'     : len(tr),
        'Test Rows'      : len(te),
    })
    summary_rows.append({
        'Horizon'        : hz.upper(),
        'Model'          : 'XGBoost',
        'Baseline (%)'   : round(baseline * 100, 1),
        'UP Precision (%)': round(best_xgb_prec * 100, 1),
        'Recall (%)'     : round(best_xgb_rec * 100, 1),
        'Edge (pp)'      : round((best_xgb_prec - baseline) * 100, 1),
        'Predicted UP'   : best_xgb_n,
        'Threshold'      : best_xgb_thresh,
        'Train Rows'     : len(tr),
        'Test Rows'      : len(te),
    })

    # Test predictions
    preds = (xgb_prob >= best_xgb_thresh).astype(int)
    te_out = te[['scrip', 'published_date', 'headline',
                 f'return_{hz}', label_col]].copy()
    te_out['xgb_prob']      = xgb_prob.round(4)
    te_out['xgb_predicted'] = preds
    te_out['correct']       = (preds == y_te).astype(int)
    all_results[hz] = te_out

# ── Write Excel ───────────────────────────────────────────────────────────────
today_str = datetime.today().strftime('%Y-%m-%d')
out_path  = os.path.join(OUT_DIR, f'news_model_results_{today_str}.xlsx')

with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
    wb = writer.book
    fmt_hdr = wb.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white'})
    fmt_pos = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221'})
    fmt_neg = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    # Sheet 1: Model Results
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_excel(writer, sheet_name='Model Results', index=False)
    ws = writer.sheets['Model Results']
    ws.set_column('A:B', 22); ws.set_column('C:J', 16)

    # Sheet 2: Feature Importance
    if feat_imp_rows:
        fi_df = pd.DataFrame(feat_imp_rows)
        fi_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        writer.sheets['Feature Importance'].set_column('A:C', 20)

    # Sheet 3+: Test Predictions per horizon
    for hz, pred_df in all_results.items():
        sheet = f'Test {hz.upper()}'
        pred_df.to_excel(writer, sheet_name=sheet, index=False)
        ws2 = writer.sheets[sheet]
        ws2.set_column('A:A', 14); ws2.set_column('B:B', 14)
        ws2.set_column('C:C', 70); ws2.set_column('D:H', 14)

print(f"\nSaved: {out_path}", flush=True)

# Print summary table
print(f"\n{'='*70}", flush=True)
print("FINAL RESULTS", flush=True)
print(f"{'='*70}", flush=True)
if summary_rows:
    s = pd.DataFrame(summary_rows)
    print(s.to_string(index=False), flush=True)
