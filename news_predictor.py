"""
news_predictor.py
Inference: fetch latest news for ticker(s), run FinBERT + TF-IDF,
predict direction and best horizon using saved XGBoost models.

Usage:
  python news_predictor.py RELIANCE TCS HDFCBANK
  python news_predictor.py --all          (runs all tickers from NSE Symbols CSV)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, warnings, pickle
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import feedparser
from transformers import pipeline

SYMBOLS_CSV = "D:/stocks prediction/NSE Symbols (1).csv"
MODELS_DIR  = "D:/stocks prediction/news_models"
OUT_DIR     = "D:/stocks prediction/new_technique/experiments/sheets"
NEWS_HOURS  = 72

# ── Load models ───────────────────────────────────────────────────────────────
def load_model(hz):
    path = os.path.join(MODELS_DIR, f'xgb_{hz}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

models = {hz: load_model(hz) for hz in ['1d', '2d', '3d']}
available = [hz for hz, m in models.items() if m is not None]
if not available:
    print("No trained models found. Run news_model_trainer.py first.", flush=True)
    sys.exit(1)

tfidf_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
if not os.path.exists(tfidf_path):
    print("TF-IDF vectorizer not found. Run news_feature_builder.py first.", flush=True)
    sys.exit(1)
with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

print(f"Loaded models for horizons: {available}", flush=True)

# ── Load FinBERT ──────────────────────────────────────────────────────────────
print("Loading FinBERT...", flush=True)
finbert = pipeline('text-classification', model='ProsusAI/finbert',
                   truncation=True, max_length=512, top_k=None)
print("Ready.\n", flush=True)

def run_finbert(text):
    if not text or not text.strip():
        return {'pos': 0.33, 'neg': 0.33, 'neu': 0.34}
    try:
        results = finbert(text[:512])[0]
        scores  = {r['label'].lower(): r['score'] for r in results}
        return {'pos': scores.get('positive', 0.0),
                'neg': scores.get('negative', 0.0),
                'neu': scores.get('neutral',  0.0)}
    except Exception:
        return {'pos': 0.33, 'neg': 0.33, 'neu': 0.34}

def fetch_news(scrip, company):
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=NEWS_HOURS)
    seen, rows = set(), []
    for query in [f"{scrip} {company} share", f"{company} stock NSE"]:
        url  = (f"https://news.google.com/rss/search"
                f"?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en")
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        for e in feed.entries:
            link = e.get('link', '')
            if link in seen: continue
            pub = None
            if hasattr(e, 'published_parsed') and e.published_parsed:
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            if pub and pub < cutoff: continue
            seen.add(link)
            rows.append({'headline': e.get('title', ''),
                         'summary' : e.get('summary', ''),
                         'source'  : (e.get('source', {}).get('title', '')
                                      if isinstance(e.get('source'), dict) else ''),
                         'published': pub.strftime('%Y-%m-%d %H:%M') if pub else 'unknown'})
    return rows

def predict_ticker(scrip, company):
    articles = fetch_news(scrip, company)
    if not articles:
        return None

    results = []
    for art in articles:
        text = f"{art['headline']}. {art['summary']}".strip()
        fb   = run_finbert(text)
        dom  = max(fb, key=fb.get)
        dom_enc = {'pos': 0, 'neg': 1, 'neu': 2}.get(dom, 2)

        tfidf_vec = tfidf.transform([art['headline']]).toarray()[0]
        feat_base = [fb['pos'], fb['neg'], fb['neu'], dom_enc, 72]  # lag=72 (fresh news)

        horizon_preds = {}
        for hz in available:
            m    = models[hz]
            feat = np.array(feat_base + list(tfidf_vec)).reshape(1, -1)
            prob = m['model'].predict_proba(feat)[0][1]
            pred = 'Positive' if prob >= m['threshold'] else 'Negative'
            horizon_preds[hz] = {'prob': round(prob, 3), 'pred': pred}

        # Best horizon = highest prob for predicted direction
        best_hz = max(available,
                      key=lambda h: horizon_preds[h]['prob']
                      if horizon_preds[h]['pred'] == 'Positive' else 0)

        results.append({
            'Scrip'         : scrip,
            'Headline'      : art['headline'],
            'Source'        : art['source'],
            'Published'     : art['published'],
            'FinBERT'       : f"{dom.upper()} ({max(fb.values()):.0%})",
            **{f'Pred_{hz.upper()}': f"{v['pred']} ({v['prob']:.0%})"
               for hz, v in horizon_preds.items()},
            'Best Horizon'  : best_hz.upper(),
            'Best Confidence': f"{horizon_preds[best_hz]['prob']:.0%}",
        })

    return results

# ── Parse CLI args ────────────────────────────────────────────────────────────
sym_df = pd.read_csv(SYMBOLS_CSV)
sym_df.columns = sym_df.columns.str.strip()
sym_map = dict(zip(sym_df['Scrip'].str.strip(), sym_df['Company Name'].str.strip()))

args = sys.argv[1:]
if '--all' in args:
    ticker_list = list(sym_map.items())
else:
    ticker_list = [(t, sym_map.get(t, t)) for t in args] if args else [
        ('RELIANCE', 'Reliance Industries Limited'),
        ('TCS', 'Tata Consultancy Services Limited'),
        ('HDFCBANK', 'HDFC Bank Limited'),
    ]

all_rows = []
for scrip, company in ticker_list:
    print(f"  [{scrip}] {company}...", end=' ', flush=True)
    rows = predict_ticker(scrip, company)
    if rows:
        all_rows.extend(rows)
        # Show top prediction
        top = rows[0]
        print(f"{len(rows)} articles | Best: {top['Best Horizon']} {top['Best Confidence']}", flush=True)
    else:
        print("No recent news.", flush=True)

if not all_rows:
    print("\nNo predictions generated.", flush=True)
    sys.exit(0)

# ── Output ────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(all_rows)
print(f"\n{out_df[['Scrip','Headline','Best Horizon','Best Confidence']].to_string(index=False)}", flush=True)

today_str = datetime.today().strftime('%Y-%m-%d')
out_path  = os.path.join(OUT_DIR, f'news_predictions_{today_str}.xlsx')
os.makedirs(OUT_DIR, exist_ok=True)

with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
    out_df.to_excel(writer, sheet_name='Predictions', index=False)
    wb = writer.book
    ws = writer.sheets['Predictions']
    ws.set_column('A:A', 14); ws.set_column('B:B', 75)
    ws.set_column('C:D', 20); ws.set_column('E:J', 22)
    fmt_pos = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221', 'bold': True})
    fmt_neg = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True})
    for r, val in enumerate(out_df.get('Best Horizon', []), start=1):
        # colour the Best Confidence cell
        conf = out_df['Best Confidence'].iloc[r-1]
        pred_cols = [c for c in out_df.columns if c.startswith('Pred_')]
        if pred_cols:
            best_pred_col = f"Pred_{val}"
            if best_pred_col in out_df.columns:
                cell_val = out_df[best_pred_col].iloc[r-1]
                fmt = fmt_pos if 'Positive' in str(cell_val) else fmt_neg
                ws.write(r, out_df.columns.get_loc('Best Confidence'), conf, fmt)

print(f"\nSaved: {out_path}", flush=True)
