"""
news_collector.py
Fetches Google News RSS for all 1524 active NSE tickers.
Saves raw articles to news_cache/{SCRIP}_news.csv (incremental).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os, time
from datetime import datetime, timezone, timedelta
import pandas as pd
import feedparser

SYMBOLS_CSV = "D:/stocks prediction/NSE Symbols (1).csv"
CACHE_DIR   = "D:/stocks prediction/news_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Generic noise patterns to always exclude
NOISE_PATTERNS = [
    'share price today', 'stock price today', 'live stock price',
    'nse/bse', 'stock price and chart', 'share price live',
    'stocks to buy', 'penny stocks', 'smallcap stocks', 'stocks below',
    '52-week high', '52-week low', 'unusual volume', 'ipo:', 'q1 results today',
    'q2 results today', 'q3 results today', 'q4 results today',
    'nse listed stocks', 'bom stocks', 'nse sme stocks',
]

def is_relevant(headline: str, scrip: str, company: str) -> bool:
    """Return True only if headline is specifically about this company."""
    h = headline.lower()

    # Reject generic noise
    if any(p in h for p in NOISE_PATTERNS):
        return False

    # Must mention ticker or meaningful part of company name
    scrip_lower   = scrip.lower()
    # Take first two meaningful words of company name (skip Ltd/Limited/Corp etc.)
    stop = {'limited', 'ltd', 'corporation', 'corp', 'india', 'industries',
            'company', 'enterprises', 'group', 'holdings', 'services', 'solutions'}
    name_words = [w for w in company.lower().split() if w not in stop]
    name_key   = name_words[0] if name_words else company.lower()[:6]

    if scrip_lower in h or name_key in h:
        return True
    return False

sym_df = pd.read_csv(SYMBOLS_CSV)
sym_df.columns = sym_df.columns.str.strip()
tickers = list(zip(sym_df['Scrip'].str.strip(), sym_df['Company Name'].str.strip()))

def fetch_rss(query: str) -> list:
    url  = (f"https://news.google.com/rss/search"
            f"?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en")
    feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
    rows = []
    for e in feed.entries:
        pub = None
        if hasattr(e, 'published_parsed') and e.published_parsed:
            pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        source = (e.get('source', {}).get('title', '')
                  if isinstance(e.get('source'), dict) else '')
        rows.append({
            'headline'      : e.get('title', ''),
            'summary'       : e.get('summary', ''),
            'published_date': pub.strftime('%Y-%m-%d %H:%M:%S') if pub else '',
            'source'        : source,
            'url'           : e.get('link', ''),
        })
    return rows

total = len(tickers)
collected = skipped = failed = 0

print(f"Collecting news for {total} tickers...\n", flush=True)

for i, (scrip, company) in enumerate(tickers, 1):
    out_path = os.path.join(CACHE_DIR, f"{scrip}_news.csv")

    if os.path.exists(out_path):
        print(f"[{i}/{total}] {scrip}  SKIP (cached)", flush=True)
        skipped += 1
        continue

    print(f"[{i}/{total}] {scrip}  {company[:40]}...", end=' ', flush=True)
    try:
        rows = []
        for query in [f"{scrip} {company} share", f"{company} stock NSE"]:
            rows.extend(fetch_rss(query))

        # deduplicate by URL and filter irrelevant articles
        seen = set()
        uniq = []
        for r in rows:
            if r['url'] not in seen:
                if is_relevant(r['headline'], scrip, company):
                    seen.add(r['url'])
                    r['scrip']   = scrip
                    r['company'] = company
                    uniq.append(r)

        if uniq:
            df = pd.DataFrame(uniq)[['scrip','company','headline','summary',
                                     'published_date','source','url']]
            df.to_csv(out_path, index=False, encoding='utf-8')
            print(f"{len(uniq)} articles", flush=True)
        else:
            # write empty file so we don't retry
            pd.DataFrame(columns=['scrip','company','headline','summary',
                                   'published_date','source','url']).to_csv(
                out_path, index=False, encoding='utf-8')
            print("0 articles", flush=True)

        collected += 1

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        failed += 1

    time.sleep(1.0)

print(f"\nDone.  Collected: {collected}  Skipped: {skipped}  Failed: {failed}", flush=True)
