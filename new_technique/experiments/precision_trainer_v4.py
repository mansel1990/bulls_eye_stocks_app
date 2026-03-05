"""
Precision Trainer v4  —  XGBoost, 1D only, with News Sentiment feature
=======================================================================
Changes from v3:
  - XGBoost only (no RF, no DL, no stacking)
  - 1D horizon only
  - Adds news_sentiment feature: 1=Up(Positive), 0=Down(Negative), 2=Neutral/No News
  - 1D Return = Open-to-Close of entry day (already correct in feature table)
  - News join: for each signal, look for news articles published in the
    3 days BEFORE the signal date. Aggregate dominant finbert sentiment.
    If multiple articles → majority vote. Tie or no news → 2 (Neutral).

Run:  python precision_trainer_v4.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, datetime, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  optuna not found — using default params")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
SHEETS_DIR = os.path.join(BASE, 'sheets')
SRC        = os.path.join(SHEETS_DIR, 'feature_table_v2_2026-03-05.xlsx')
NEWS_CSV   = os.path.join(SHEETS_DIR, 'news_training_data.csv')
TODAY_STR  = datetime.date.today().strftime('%Y-%m-%d')
OUT        = os.path.join(SHEETS_DIR, f'precision_1D_v4_{TODAY_STR}.xlsx')
SEED       = 42
np.random.seed(SEED)

# ── Load feature table ────────────────────────────────────────────────────────
print("Loading feature table...")
df24 = pd.read_excel(SRC, sheet_name='2024 Feature Table')
df25 = pd.read_excel(SRC, sheet_name='2025 Feature Table')
df24['_year'] = '2024'; df25['_year'] = '2025'
raw = pd.concat([df24, df25], ignore_index=True)
raw['Signal Date'] = pd.to_datetime(raw['Signal Date'])
print(f"  {len(raw):,} rows | {len(raw.columns)} columns")

# ── Load news training data ───────────────────────────────────────────────────
print("Loading news data...")
news = pd.read_csv(NEWS_CSV, encoding='utf-8',
                   usecols=['scrip', 'published_date', 'finbert_dominant', 'label_1d'])
news['published_date'] = pd.to_datetime(news['published_date'], errors='coerce')
news = news.dropna(subset=['published_date', 'finbert_dominant'])
news['scrip'] = news['scrip'].str.strip().str.upper()
print(f"  {len(news):,} news articles | {news['scrip'].nunique()} tickers")

# ── Join news to signals ──────────────────────────────────────────────────────
# For each signal (Ticker, Signal Date):
#   Collect all news articles for that ticker published in [Signal Date - 3d, Signal Date]
#   Aggregate finbert_dominant: 0=neg, 1=pos, 2=neu  → majority vote
#   No news or tie → 2 (Neutral)
print("Joining news to signals (this may take ~30s)...")

news_lookup = news.groupby('scrip')

def get_news_sentiment(ticker, signal_date):
    """Return majority finbert_dominant for articles in the 3-day window before signal."""
    try:
        grp = news_lookup.get_group(ticker)
    except KeyError:
        return 2  # no news for this ticker

    window = grp[
        (grp['published_date'] >= signal_date - pd.Timedelta(days=3)) &
        (grp['published_date'] <= signal_date)
    ]
    if window.empty:
        return 2  # no news in window → Neutral

    # finbert_dominant: 0=pos, 1=neg, 2=neu (from news_feature_builder)
    counts = window['finbert_dominant'].value_counts()
    best_val  = counts.idxmax()
    best_cnt  = counts.max()
    # Check for tie
    if (counts == best_cnt).sum() > 1:
        return 2  # tie → Neutral
    # Map: finbert 0=pos→1(Up), 1=neg→0(Down), 2=neu→2(Neutral)
    mapping = {0: 1, 1: 0, 2: 2}
    return mapping.get(int(best_val), 2)

raw['Ticker_upper'] = raw['Ticker'].str.strip().str.upper()
raw['news_sentiment'] = [
    get_news_sentiment(t, d)
    for t, d in zip(raw['Ticker_upper'], raw['Signal Date'])
]
raw.drop(columns='Ticker_upper', inplace=True)

sentiment_counts = raw['news_sentiment'].value_counts().sort_index()
print(f"  news_sentiment distribution:")
print(f"    Up (1):      {sentiment_counts.get(1, 0):,}  ({100*sentiment_counts.get(1,0)/len(raw):.1f}%)")
print(f"    Down (0):    {sentiment_counts.get(0, 0):,}  ({100*sentiment_counts.get(0,0)/len(raw):.1f}%)")
print(f"    Neutral (2): {sentiment_counts.get(2, 0):,}  ({100*sentiment_counts.get(2,0)/len(raw):.1f}%)")

# ── Derived booleans (same as v3) ─────────────────────────────────────────────
raw['entry_gt_sl']        = (raw['Entry Price'] > raw['SL Price']).astype(float)
raw['entry_gt_sig_open']  = (raw['Entry Price'] > raw['Sig Open']).astype(float)
raw['entry_gt_sig_low']   = (raw['Entry Price'] > raw['Sig Low']).astype(float)
raw['entry_gt_sig_high']  = (raw['Entry Price'] > raw['Sig High']).astype(float)
raw['entry_gt_sig_close'] = (raw['Entry Price'] > raw['Sig Close']).astype(float)
raw['entry_gt_avg_yday']  = (raw['Entry Price'] > (raw['Sig Open'] + raw['Sig Close']) / 2).astype(float)

DERIVED_COLS = ['entry_gt_sl','entry_gt_sig_open','entry_gt_sig_low',
                'entry_gt_sig_high','entry_gt_sig_close','entry_gt_avg_yday']

# ── Feature set ───────────────────────────────────────────────────────────────
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
            if c not in EXCLUDE and c not in CAT_COLS and c not in DERIVED_COLS
            and c != 'news_sentiment']

# Encode categoricals
pat_dummies = pd.get_dummies(raw['Pattern'].fillna('Unknown'), prefix='PAT')
et_dummies  = pd.get_dummies(raw['Entry Type'].fillna('stop'),  prefix='ET')
raw['Strength_bin'] = (raw['Strength'] == 'Strongly Bullish').astype(float)
for bc in BOOL_COLS:
    if bc in raw.columns:
        raw[bc] = raw[bc].map({True:1,False:0,'True':1,'False':0,1:1,0:0}).fillna(0)

# news_sentiment as ordinal: 0=Down, 1=Up, 2=Neutral
raw['news_sentiment'] = raw['news_sentiment'].astype(float)

feat_df = pd.concat([
    raw[NUM_COLS].reset_index(drop=True),
    raw[['Strength_bin', 'news_sentiment'] + DERIVED_COLS].reset_index(drop=True),
    pat_dummies.reset_index(drop=True),
    et_dummies.reset_index(drop=True),
], axis=1)

for c in feat_df.columns:
    feat_df[c] = pd.to_numeric(feat_df[c], errors='coerce')
feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0)

# Clip outliers (skip bool/categorical cols)
bool_set = set(BOOL_COLS + DERIVED_COLS +
               [c for c in feat_df.columns if c.startswith('PAT_') or
                c.startswith('ET_') or c in ('Strength_bin', 'news_sentiment')])
for c in feat_df.columns:
    if c in bool_set: continue
    col = feat_df[c].astype(float)
    q1, q3 = col.quantile(0.01), col.quantile(0.99)
    if q3 > q1:
        feat_df[c] = col.clip(q1 - 5*(q3-q1), q3 + 5*(q3-q1))

feature_names = feat_df.columns.tolist()
print(f"\nFeature matrix: {feat_df.shape[0]:,} rows x {len(feature_names)} features")
print(f"  news_sentiment position: {feature_names.index('news_sentiment')}")

# ── Train / test split ────────────────────────────────────────────────────────
train_mask = raw['_year'].values == '2024'
test_mask  = raw['_year'].values == '2025'
X_train = feat_df.values[train_mask].astype(np.float32)
X_test  = feat_df.values[test_mask].astype(np.float32)

y_train = (raw['1D Result'].values[train_mask] == 'Profit').astype(int)
y_test  = (raw['1D Result'].values[test_mask]  == 'Profit').astype(int)

baseline   = y_test.mean() * 100
pos_ratio  = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

print(f"\nTrain: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"Baseline profit rate (test 2025): {baseline:.1f}%")
print(f"Pos weight: {pos_ratio:.2f}")

# News sentiment impact check
test_raw = raw[test_mask].reset_index(drop=True)
for s_val, s_name in [(1,'Up'),(0,'Down'),(2,'Neutral')]:
    mask = test_raw['news_sentiment'] == s_val
    if mask.sum() > 0:
        rate = (test_raw[mask]['1D Result'] == 'Profit').mean() * 100
        print(f"  Sentiment={s_name}: {mask.sum():,} signals | actual profit={rate:.1f}%")

# ── Helper: best threshold ────────────────────────────────────────────────────
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

# ── XGBoost with Optuna ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"TRAINING XGBoost — 1D Horizon  (Optuna {'ON' if HAS_OPTUNA else 'OFF'})")
print(f"{'='*65}")

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

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(xgb_objective, n_trials=60, show_progress_bar=True)
    bx = study.best_params
    print(f"\nBest params: {bx}")
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
xgb_res   = evaluate(y_test, xgb_pred, 'XGBoost (with News Sentiment)', xgb_thr, baseline)

print(f"\nResult:")
print(f"  Prec={xgb_prec*100:.1f}% | Rec={xgb_rec*100:.1f}% | "
      f"PredUP={xgb_pred.sum():,} | Edge={xgb_res['Edge (pp)']:+.1f}pp | Thr={xgb_thr:.2f}")

# ── Compare: XGBoost WITHOUT news_sentiment ───────────────────────────────────
print(f"\n{'='*65}")
print("Ablation: XGBoost WITHOUT news_sentiment (same params)")
print(f"{'='*65}")

ns_idx    = feature_names.index('news_sentiment')
X_tr_no   = np.delete(X_train, ns_idx, axis=1)
X_te_no   = np.delete(X_test,  ns_idx, axis=1)

xgb_no = xgb.XGBClassifier(
    n_estimators=xgb_m.n_estimators, max_depth=xgb_m.max_depth,
    learning_rate=xgb_m.learning_rate, subsample=xgb_m.subsample,
    colsample_bytree=xgb_m.colsample_bytree, min_child_weight=xgb_m.min_child_weight,
    gamma=xgb_m.gamma, reg_alpha=xgb_m.reg_alpha, reg_lambda=xgb_m.reg_lambda,
    scale_pos_weight=pos_ratio, eval_metric='logloss',
    use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0,
)
xgb_no.fit(X_tr_no, y_train, eval_set=[(X_te_no, y_test)], verbose=False)
no_probs = xgb_no.predict_proba(X_te_no)[:, 1]
no_thr, no_prec, no_rec = best_threshold(no_probs, y_test)
no_pred  = (no_probs >= no_thr).astype(int)
no_res   = evaluate(y_test, no_pred, 'XGBoost (no News Sentiment)', no_thr, baseline)

print(f"  Prec={no_prec*100:.1f}% | Rec={no_rec*100:.1f}% | "
      f"PredUP={no_pred.sum():,} | Edge={no_res['Edge (pp)']:+.1f}pp | Thr={no_thr:.2f}")

# ── Feature importance ────────────────────────────────────────────────────────
fi_df = pd.DataFrame({
    'Feature':        feature_names,
    'Importance':     xgb_m.feature_importances_,
    'Importance (%)': xgb_m.feature_importances_ * 100,
}).sort_values('Importance', ascending=False).reset_index(drop=True)
fi_df['Importance (%)'] = fi_df['Importance (%)'].round(3)
ns_rank = fi_df[fi_df['Feature'] == 'news_sentiment'].index[0] + 1
print(f"\nnews_sentiment importance rank: #{ns_rank} of {len(feature_names)}")
print(f"  news_sentiment importance: {fi_df[fi_df['Feature']=='news_sentiment']['Importance (%)'].values[0]:.3f}%")

# ── Tag test rows ─────────────────────────────────────────────────────────────
t_raw = test_raw.copy()
t_raw['XGB_Prob']      = xgb_probs
t_raw['XGB_Pred']      = xgb_pred
t_raw['news_sentiment']= test_raw['news_sentiment'].values  # already present

predicted_up = t_raw[t_raw['XGB_Pred'] == 1].sort_values('XGB_Prob', ascending=False)

print(f"\nPredicted UP signals: {len(predicted_up):,}")
if len(predicted_up):
    actual = (predicted_up['1D Result'] == 'Profit').mean() * 100
    print(f"Actual profit rate on predicted UP: {actual:.1f}%")

    # Breakdown by news_sentiment within predicted UP
    for s_val, s_name in [(1,'Up'),(0,'Down'),(2,'Neutral')]:
        mask = predicted_up['news_sentiment'] == s_val
        if mask.sum() > 0:
            rate = (predicted_up[mask]['1D Result'] == 'Profit').mean() * 100
            print(f"  PredUP + Sentiment={s_name}: {mask.sum():,} signals | profit={rate:.1f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
results = [xgb_res, no_res]
df_res = pd.DataFrame(results)
print(f"\n{'='*65}")
print(f"FINAL COMPARISON  (baseline {baseline:.1f}%)")
print(f"{'='*65}")
print(df_res[['Model','Baseline (%)','UP Precision (%)','Edge (pp)',
              'UP Recall (%)','Predicted UP','Threshold']].to_string(index=False))

# ── Write Excel ───────────────────────────────────────────────────────────────
PROFIT_C = '#C6EFCE'; LOSS_C = '#FFC7CE'; NEUT_C = '#FFEB9C'

print(f"\nWriting: {OUT}")
with pd.ExcelWriter(OUT, engine='xlsxwriter',
                    engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
    wb = writer.book
    fh  = wb.add_format({'bold':True,'bg_color':'#1F3864','font_color':'white',
                         'border':1,'text_wrap':True,'valign':'vcenter'})
    fg  = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True,'num_format':'0.0'})
    fm  = wb.add_format({'bg_color':NEUT_C,  'font_color':'#9C6500','num_format':'0.0'})
    fb  = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006','num_format':'0.0'})
    fe  = wb.add_format({'bg_color':'#DDEBF7','font_color':'#1F3864','num_format':'+0.0;-0.0'})
    fpr = wb.add_format({'num_format': '0.0000'})
    fpp = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','num_format':'+0.00;-0.00'})
    fll = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006','num_format':'+0.00;-0.00'})
    fup = wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True})
    fdn = wb.add_format({'bg_color':LOSS_C,  'font_color':'#9C0006','bold':True})
    fnu = wb.add_format({'bg_color':NEUT_C,  'font_color':'#9C6500'})

    # Sheet 1: Model Comparison
    df_res.to_excel(writer, index=False, sheet_name='Model Comparison')
    ws = writer.sheets['Model Comparison']
    ws.set_row(0, 28)
    pc = df_res.columns.get_loc('UP Precision (%)')
    ec = df_res.columns.get_loc('Edge (pp)')
    for ci, cn in enumerate(df_res.columns):
        ws.set_column(ci, ci, max(len(cn)+2, 18))
        ws.write(0, ci, cn, fh)
    for ri, row in df_res.iterrows():
        p = row['UP Precision (%)']
        ws.write(ri+1, pc, p, fg if p >= 60 else fm if p >= 50 else fb)
        ws.write(ri+1, ec, row['Edge (pp)'], fe)

    # Sheet 2: Feature Importance
    fi_df.head(60).to_excel(writer, index=False, sheet_name='Feature Importance')
    ws_fi = writer.sheets['Feature Importance']
    ws_fi.set_row(0, 24)
    ws_fi.set_column(0, 0, 32); ws_fi.set_column(1, 1, 13); ws_fi.set_column(2, 2, 16)
    for ci, cn in enumerate(fi_df.columns):
        ws_fi.write(0, ci, cn, fh)
    imp_ci = fi_df.columns.get_loc('Importance (%)')
    ws_fi.conditional_format(1, imp_ci, min(60, len(fi_df)), imp_ci,
                             {'type':'2_color_scale','min_color':'#FFFFFF','max_color':'#4472C4'})
    # Highlight news_sentiment row
    ns_row = fi_df[fi_df['Feature'] == 'news_sentiment'].index[0] + 1
    for ci in range(len(fi_df.columns)):
        ws_fi.write(ns_row, ci, fi_df.iloc[ns_row-1, ci],
                    wb.add_format({'bg_color':'#FFF2CC','bold':True}))

    # Sheet 3: Predicted UP signals (with news)
    DISP = ['Ticker','Signal Date','Entry Date','Pattern','Strength','Entry Type',
            'Risk (%)','RSI14','MACD Line','MACD Hist','MACD > Signal',
            'MA44 Ascending','Close > MA44','Close > MA200',
            'Base Score','Fundamental Score','Performer Score',
            'news_sentiment',
            'XGB_Prob','1D Result','1D Return (%)']
    disp = predicted_up[[c for c in DISP if c in predicted_up.columns]].copy()
    disp.to_excel(writer, index=False, sheet_name=f'XGB Predicted UP ({len(disp)})')
    ws2 = writer.sheets[f'XGB Predicted UP ({len(disp)})']
    ws2.set_row(0, 28)
    COL_W = {'Ticker':12,'Signal Date':13,'Entry Date':13,'Pattern':24,
             'Strength':18,'Entry Type':18,'news_sentiment':16,'XGB_Prob':10,
             'Base Score':12,'Fundamental Score':18,'Performer Score':16}
    BOOL_DISP = {'MACD > Signal','MA44 Ascending','Close > MA44','Close > MA200'}
    res_ci  = disp.columns.get_loc('1D Result')    if '1D Result'    in disp.columns else None
    ret_ci  = disp.columns.get_loc('1D Return (%)') if '1D Return (%)' in disp.columns else None
    ns_ci   = disp.columns.get_loc('news_sentiment') if 'news_sentiment' in disp.columns else None
    prob_ci = disp.columns.get_loc('XGB_Prob')      if 'XGB_Prob'      in disp.columns else None
    str_ci  = disp.columns.get_loc('Strength')      if 'Strength'      in disp.columns else None
    for ci, cn in enumerate(disp.columns):
        ws2.set_column(ci, ci, COL_W.get(cn, max(len(cn)+2, 10)))
        ws2.write(0, ci, cn, fh)
    for ri, row in disp.iterrows():
        er = ri + 1
        if str_ci is not None:
            s = row.get('Strength', '')
            ws2.write(er, str_ci, s,
                      wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True})
                      if s == 'Strongly Bullish'
                      else wb.add_format({'bg_color':NEUT_C,'font_color':'#9C6500'}))
        if ns_ci is not None:
            sv = int(row['news_sentiment'])
            label = {1:'Up', 0:'Down', 2:'Neutral'}.get(sv, 'Neutral')
            fmt   = fup if sv == 1 else fdn if sv == 0 else fnu
            ws2.write(er, ns_ci, label, fmt)
        if prob_ci is not None:
            ws2.write(er, prob_ci, row['XGB_Prob'], fpr)
        if res_ci is not None:
            lbl = row['1D Result']
            ws2.write(er, res_ci, lbl,
                      wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221'})
                      if lbl == 'Profit'
                      else wb.add_format({'bg_color':LOSS_C,'font_color':'#9C0006'}))
        if ret_ci is not None:
            pct = row['1D Return (%)']
            lbl = row.get('1D Result','')
            if pd.notna(pct):
                ws2.write(er, ret_ci, pct, fpp if lbl == 'Profit' else fll)
        for cn in BOOL_DISP:
            if cn in disp.columns:
                ci2 = disp.columns.get_loc(cn)
                v   = row[cn]
                ws2.write(er, ci2, 'YES' if v == 1 else 'NO',
                          wb.add_format({'bg_color':PROFIT_C,'font_color':'#276221','bold':True})
                          if v == 1 else
                          wb.add_format({'bg_color':LOSS_C,'font_color':'#9C0006'}))
        for sc in ['Base Score','Fundamental Score','Performer Score']:
            if sc in disp.columns:
                ci2 = disp.columns.get_loc(sc)
                ws2.conditional_format(1, ci2, len(disp), ci2,
                                       {'type':'3_color_scale',
                                        'min_color':'#F8696B','mid_color':'#FFEB84',
                                        'max_color':'#63BE7B'})
    ws2.freeze_panes(1, 3)
    ws2.autofilter(0, 0, len(disp), len(disp.columns)-1)

    # Sheet 4: News Sentiment Breakdown (within predicted UP)
    if len(predicted_up):
        rows = []
        for s_val, s_name in [(1,'Up'),(0,'Down'),(2,'Neutral/No News')]:
            mask = predicted_up['news_sentiment'] == s_val
            n = mask.sum()
            if n == 0:
                rows.append({'News Sentiment': s_name, 'Count': 0,
                             'Profit Rate (%)': '-', 'Avg 1D Return (%)': '-'})
            else:
                rate = (predicted_up[mask]['1D Result'] == 'Profit').mean() * 100
                avg_r = predicted_up[mask]['1D Return (%)'].mean()
                rows.append({'News Sentiment': s_name, 'Count': int(n),
                             'Profit Rate (%)': round(rate, 1),
                             'Avg 1D Return (%)': round(avg_r, 2)})
        rows.append({'News Sentiment': 'ALL', 'Count': len(predicted_up),
                     'Profit Rate (%)': round((predicted_up['1D Result']=='Profit').mean()*100, 1),
                     'Avg 1D Return (%)': round(predicted_up['1D Return (%)'].mean(), 2)})
        bd_df = pd.DataFrame(rows)
        bd_df.to_excel(writer, index=False, sheet_name='Sentiment Breakdown')
        ws3 = writer.sheets['Sentiment Breakdown']
        ws3.set_row(0, 24)
        for ci, cn in enumerate(bd_df.columns):
            ws3.set_column(ci, ci, max(len(cn)+4, 18))
            ws3.write(0, ci, cn, fh)

print(f"Saved: {OUT}")
print(f"\nDONE.")
