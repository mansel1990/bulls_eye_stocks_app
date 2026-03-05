"""
NSE Stock News Sentiment Scanner
- Reads tickers + company names from NSE Symbols (1).csv
- Queries Google News RSS for each company (last 72h)
- Classifies each headline as Positive / Negative / Neutral w.r.t. stock
- Outputs Excel with one row per news item + summary sheet

To run on all tickers: set SAMPLE_OVERRIDE = [] and SAMPLE_N = None
"""

import os, time
from datetime import datetime, timezone, timedelta
import pandas as pd
import feedparser

SYMBOLS_CSV  = "D:/stocks prediction/NSE Symbols (1).csv"
OUT_DIR      = "D:/stocks prediction/new_technique/experiments/sheets"
NEWS_HOURS   = 72    # look-back window in hours
SAMPLE_N     = 20    # fallback sample size if SAMPLE_OVERRIDE is empty

# Well-known tickers for the initial 20-company sample test
# Set to [] to use the first SAMPLE_N rows from the CSV instead
SAMPLE_OVERRIDE = [
    ("RELIANCE",   "Reliance Industries Limited"),
    ("TCS",        "Tata Consultancy Services Limited"),
    ("HDFCBANK",   "HDFC Bank Limited"),
    ("INFY",       "Infosys Limited"),
    ("ICICIBANK",  "ICICI Bank Limited"),
    ("WIPRO",      "Wipro Limited"),
    ("ADANIENT",   "Adani Enterprises Limited"),
    ("TATAMOTORS", "Tata Motors Limited"),
    ("SBIN",       "State Bank of India"),
    ("AXISBANK",   "Axis Bank Limited"),
    ("BAJFINANCE", "Bajaj Finance Limited"),
    ("ONGC",       "Oil and Natural Gas Corporation"),
    ("MARUTI",     "Maruti Suzuki India Limited"),
    ("SUNPHARMA",  "Sun Pharmaceutical Industries"),
    ("LTIM",       "LTIMindtree Limited"),
    ("KOTAKBANK",  "Kotak Mahindra Bank Limited"),
    ("HINDUNILVR", "Hindustan Unilever Limited"),
    ("ITC",        "ITC Limited"),
    ("POWERGRID",  "Power Grid Corporation of India"),
    ("BHARTIARTL", "Bharti Airtel Limited"),
]

# ── Sentiment word lists (stock-context) ──────────────────────────────────────
POS_WORDS = [
    "profit", "growth", "surge", "rally", "gain", "beat", "record",
    "expansion", "upgrade", "buy", "outperform", "bullish", "dividend",
    "strong", "rise", "jumped", "soared", "positive", "turnaround",
    "deal", "contract", "order", "acquisition", "merger", "launch",
    "revenue", "robust", "improved", "approved", "wins", "award",
    "recovery", "upside", "momentum", "high", "all-time", "increase",
]
NEG_WORDS = [
    "loss", "decline", "fall", "drop", "crash", "sell", "downgrade",
    "bearish", "debt", "default", "fraud", "scam", "probe", "penalty",
    "fine", "lawsuit", "miss", "weak", "slump", "lower", "cut", "layoff",
    "bankruptcy", "insolvency", "negative", "risk", "warning", "recall",
    "investigation", "raid", "seized", "halt", "suspend", "concern",
    "disappointing", "slowdown", "pressure", "underperform", "reduce",
]

def classify(text: str) -> str:
    t   = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos > neg:   return "Positive"
    if neg > pos:   return "Negative"
    return "Neutral"

def score_detail(text: str):
    t = text.lower()
    return [w for w in POS_WORDS if w in t], [w for w in NEG_WORDS if w in t]

def fetch_news(scrip: str, company: str) -> list:
    """Return list of news dicts within NEWS_HOURS for this stock."""
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=NEWS_HOURS)
    seen    = set()
    results = []

    queries = [
        f"{scrip} {company} share",
        f"{company} stock NSE",
    ]

    for query in queries:
        url  = (f"https://news.google.com/rss/search"
                f"?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en")
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})

        for entry in feed.entries:
            link = entry.get("link", "")
            if link in seen:
                continue

            pub = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if pub and pub < cutoff:
                continue

            title     = entry.get("title", "")
            source    = (entry.get("source", {}).get("title", "")
                         if isinstance(entry.get("source"), dict) else "")
            full_text = title + " " + entry.get("summary", "")

            sentiment        = classify(full_text)
            pos_hits, neg_hits = score_detail(full_text)

            seen.add(link)
            results.append({
                "Scrip"         : scrip,
                "Company"       : company,
                "Headline"      : title,
                "Source"        : source,
                "Published"     : pub.strftime("%Y-%m-%d %H:%M") if pub else "unknown",
                "Sentiment"     : sentiment,
                "Positive Words": ", ".join(pos_hits),
                "Negative Words": ", ".join(neg_hits),
                "URL"           : link,
            })

    return results

# ── Load ticker list ──────────────────────────────────────────────────────────
if SAMPLE_OVERRIDE:
    sym_df = pd.DataFrame(SAMPLE_OVERRIDE, columns=["Scrip", "Company Name"])
else:
    sym_df = pd.read_csv(SYMBOLS_CSV)
    sym_df.columns = sym_df.columns.str.strip()
    if SAMPLE_N:
        sym_df = sym_df.head(SAMPLE_N)

all_rows = []
summary  = []

print(f"Scanning {len(sym_df)} tickers for news (last {NEWS_HOURS}h)...\n", flush=True)

for _, row in sym_df.iterrows():
    scrip   = str(row["Scrip"]).strip()
    company = str(row["Company Name"]).strip()
    print(f"  [{scrip}] {company}...", end=" ", flush=True)

    items = fetch_news(scrip, company)
    count = len(items)
    pos   = sum(1 for r in items if r["Sentiment"] == "Positive")
    neg   = sum(1 for r in items if r["Sentiment"] == "Negative")
    neu   = count - pos - neg

    if count == 0:
        overall = "No News"
    elif pos > neg:
        overall = "Positive"
    elif neg > pos:
        overall = "Negative"
    else:
        overall = "Neutral"

    print(f"{count} articles | {pos}+ {neg}- {neu}~ => {overall}", flush=True)

    all_rows.extend(items)
    summary.append({
        "Scrip"            : scrip,
        "Company"          : company,
        "Total Articles"   : count,
        "Positive"         : pos,
        "Negative"         : neg,
        "Neutral"          : neu,
        "Overall Sentiment": overall,
    })
    time.sleep(1.0)

# ── Write Excel ───────────────────────────────────────────────────────────────
today_str = datetime.today().strftime("%Y-%m-%d")
out_path  = os.path.join(OUT_DIR, f"news_sentiment_{today_str}.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    wb = writer.book

    fmt_pos = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#276221", "bold": True})
    fmt_neg = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006", "bold": True})
    fmt_neu = wb.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})
    fmt_non = wb.add_format({"bg_color": "#F2F2F2", "font_color": "#666666", "italic": True})

    def sentiment_fmt(val):
        return {"Positive": fmt_pos, "Negative": fmt_neg,
                "Neutral": fmt_neu}.get(val, fmt_non)

    # ── Summary sheet ─────────────────────────────────────────────────────────
    sum_df = pd.DataFrame(summary)
    sum_df.to_excel(writer, sheet_name="Summary", index=False)
    ws = writer.sheets["Summary"]
    ws.set_column("A:A", 14); ws.set_column("B:B", 42)
    ws.set_column("C:G", 12); ws.set_column("H:H", 16)
    for r, val in enumerate(sum_df["Overall Sentiment"], start=1):
        ws.write(r, 7, val, sentiment_fmt(val))

    # ── All News sheet ────────────────────────────────────────────────────────
    if all_rows:
        news_df = pd.DataFrame(all_rows)
        news_df.to_excel(writer, sheet_name="All News", index=False)
        wn = writer.sheets["All News"]
        wn.set_column("A:A", 14); wn.set_column("B:B", 36)
        wn.set_column("C:C", 72); wn.set_column("D:D", 24)
        wn.set_column("E:E", 16); wn.set_column("F:F", 12)
        wn.set_column("G:H", 36); wn.set_column("I:I", 60)
        for r, val in enumerate(news_df["Sentiment"], start=1):
            wn.write(r, 5, val, sentiment_fmt(val))

print(f"\nSaved  : {out_path}")
print(f"Articles: {len(all_rows)}  across {len(summary)} tickers")
