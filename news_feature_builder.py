"""
news_feature_builder.py
Joins news cache to daily price data, labels movements, runs FinBERT + TF-IDF.
Outputs news_training_data.csv and news_training_data.xlsx
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, warnings, pickle
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

CACHE_DIR   = "D:/stocks prediction/news_cache"
DAILY_DIR   = "D:/stocks prediction/daily_data"
OUT_DIR     = "D:/stocks prediction/new_technique/experiments/sheets"
MODELS_DIR  = "D:/stocks prediction/news_models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Market open IST = UTC+5:30 = 03:45 UTC
MARKET_OPEN_UTC_H = 3
MARKET_OPEN_UTC_M = 45
FINBERT_BATCH     = 512   # headlines ~20 tokens, 512 is optimal for CPU

# ── Load FinBERT ──────────────────────────────────────────────────────────────
print("Loading FinBERT...", flush=True)
finbert = pipeline('text-classification', model='ProsusAI/finbert',
                   truncation=True, max_length=64, top_k=None,
                   batch_size=FINBERT_BATCH)
print("FinBERT ready.\n", flush=True)

def run_finbert_batch(texts: list) -> list:
    """Run FinBERT on a list of texts in one batched call.
    Returns list of dicts with pos/neg/neu scores."""
    default = {'pos': 0.33, 'neg': 0.33, 'neu': 0.34}
    if not texts:
        return []
    # Clean texts — use headline only (already short)
    cleaned = [t[:256] if t and t.strip() else 'neutral' for t in texts]
    try:
        results = finbert(cleaned)   # pipeline handles batching internally
        out = []
        for res in results:
            scores = {r['label'].lower(): r['score'] for r in res}
            out.append({
                'pos': scores.get('positive', 0.0),
                'neg': scores.get('negative', 0.0),
                'neu': scores.get('neutral',  0.0),
            })
        return out
    except Exception as e:
        print(f"  FinBERT batch error: {e}", flush=True)
        return [default] * len(texts)

def next_trading_day(price_df, date):
    """Return index of first row on or after date."""
    idx = price_df.index.searchsorted(pd.Timestamp(date))
    if idx >= len(price_df):
        return None
    return idx

def compute_return(price_df, base_idx, n_days):
    """return_Nd = (Close[base_idx + n_days] - Open[base_idx]) / Open[base_idx] * 100"""
    target_idx = base_idx + n_days
    if target_idx >= len(price_df):
        return None
    open_price  = price_df['Open'].iloc[base_idx]
    close_price = price_df['Close'].iloc[base_idx + n_days]
    if open_price == 0 or pd.isna(open_price) or pd.isna(close_price):
        return None
    return (close_price - open_price) / open_price * 100

def label(ret):
    if ret is None:
        return None
    if ret > 1.0:
        return 1
    if ret < -0.4:
        return 0
    return None   # ambiguous zone — skip

def lag_bucket(pub_dt):
    """Hours from article publish to next 9:15 AM IST, bucketed 24/48/72."""
    if pub_dt is None:
        return 72
    # next market open after pub_dt
    market_open = pub_dt.replace(hour=MARKET_OPEN_UTC_H, minute=MARKET_OPEN_UTC_M,
                                  second=0, microsecond=0)
    if market_open <= pub_dt:
        market_open += timedelta(days=1)
    diff_h = (market_open - pub_dt).total_seconds() / 3600
    if diff_h <= 24:   return 24
    if diff_h <= 48:   return 48
    return 72

# ── Collect all cache files ───────────────────────────────────────────────────
cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_news.csv')]
print(f"Found {len(cache_files)} cached ticker files.", flush=True)

all_rows = []
processed = skipped = 0

for fi, fname in enumerate(cache_files, 1):
    scrip = fname.replace('_news.csv', '')
    daily_path = os.path.join(DAILY_DIR, f"{scrip}_daily.csv")

    if not os.path.exists(daily_path):
        skipped += 1
        continue

    try:
        news_df  = pd.read_csv(os.path.join(CACHE_DIR, fname), encoding='utf-8')
        price_df = pd.read_csv(daily_path, index_col='Date', parse_dates=True)
    except Exception:
        skipped += 1
        continue

    if news_df.empty or price_df.empty:
        skipped += 1
        continue

    print(f"[{fi}/{len(cache_files)}] {scrip}  {len(news_df)} articles...", flush=True)

    ticker_rows = []
    for _, row in news_df.iterrows():
        # Parse publish date
        pub_dt = None
        if pd.notna(row.get('published_date')) and str(row['published_date']).strip():
            try:
                pub_dt = pd.Timestamp(row['published_date']).to_pydatetime()
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass

        pub_date = pub_dt.date() if pub_dt else None
        if pub_date is None:
            continue

        next_idx = next_trading_day(price_df, pub_date)
        if next_idx is None:
            continue

        r1 = compute_return(price_df, next_idx, 1)
        r2 = compute_return(price_df, next_idx, 2)
        r3 = compute_return(price_df, next_idx, 3)

        l1 = label(r1)
        l2 = label(r2)
        l3 = label(r3)

        if l1 is None and l2 is None and l3 is None:
            continue

        ticker_rows.append({
            'scrip'        : scrip,
            'published_date': str(pub_date),
            'headline'     : str(row.get('headline', '')),
            'lag_hours'    : lag_bucket(pub_dt),
            'return_1d'    : round(r1, 4) if r1 is not None else None,
            'return_2d'    : round(r2, 4) if r2 is not None else None,
            'return_3d'    : round(r3, 4) if r3 is not None else None,
            'label_1d'     : l1,
            'label_2d'     : l2,
            'label_3d'     : l3,
        })

    if ticker_rows:
        # Batch FinBERT on all headlines for this ticker at once
        headlines  = [r['headline'] for r in ticker_rows]
        fb_results = run_finbert_batch(headlines)
        for r, fb in zip(ticker_rows, fb_results):
            dominant  = max(fb, key=fb.get)
            r['finbert_pos']      = round(fb['pos'], 4)
            r['finbert_neg']      = round(fb['neg'], 4)
            r['finbert_neu']      = round(fb['neu'], 4)
            r['finbert_dominant'] = {'pos': 0, 'neg': 1, 'neu': 2}.get(dominant, 2)
        all_rows.extend(ticker_rows)

    processed += 1

print(f"\nProcessed: {processed}  Skipped: {skipped}", flush=True)
print(f"Total rows before TF-IDF: {len(all_rows)}", flush=True)

if not all_rows:
    print("No data collected. Exiting.", flush=True)
    sys.exit(1)

base_df = pd.DataFrame(all_rows)

# ── TF-IDF on headlines ───────────────────────────────────────────────────────
print("\nFitting TF-IDF (top 100 features)...", flush=True)
tfidf = TfidfVectorizer(max_features=100, stop_words='english',
                        ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf.fit_transform(base_df['headline'].fillna(''))
tfidf_cols   = [f"tfidf_{i:03d}" for i in range(tfidf_matrix.shape[1])]
tfidf_df     = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

# Save TF-IDF vectorizer for inference
with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
print(f"TF-IDF fitted: {len(tfidf_cols)} features. Vectorizer saved.", flush=True)

# ── Combine ───────────────────────────────────────────────────────────────────
final_df = pd.concat([base_df.reset_index(drop=True), tfidf_df], axis=1)

# ── Print class balance ───────────────────────────────────────────────────────
for hz in ['1d', '2d', '3d']:
    col = f'label_{hz}'
    valid = final_df[col].dropna()
    pos = (valid == 1).sum()
    neg = (valid == 0).sum()
    print(f"  {hz}: {len(valid)} labeled rows | Pos={pos} ({100*pos/len(valid):.1f}%)  "
          f"Neg={neg} ({100*neg/len(valid):.1f}%)", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
today_str = datetime.today().strftime('%Y-%m-%d')
csv_out   = os.path.join(OUT_DIR, 'news_training_data.csv')
xlsx_out  = os.path.join(OUT_DIR, f'news_training_data_{today_str}.xlsx')

final_df.to_csv(csv_out, index=False, encoding='utf-8')
print(f"\nSaved CSV : {csv_out}", flush=True)

with pd.ExcelWriter(xlsx_out, engine='xlsxwriter') as writer:
    final_df.head(5000).to_excel(writer, sheet_name='Training Data', index=False)
    ws = writer.sheets['Training Data']
    ws.set_column('A:A', 12); ws.set_column('B:B', 12)
    ws.set_column('C:C', 70); ws.set_column('D:N', 10)
print(f"Saved XLSX: {xlsx_out}", flush=True)
print(f"\nTotal rows in training data: {len(final_df)}", flush=True)
