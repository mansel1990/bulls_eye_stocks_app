import os
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
DAILY_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'daily_data')
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), 'performers_2024_2025.xlsx')
NEUTRAL_PCT    = 4.0   # ±4% → Neutral

# 8 quarters in order oldest → newest
QUARTERS = [
    ('Q1-2024', '2024-01-01', '2024-03-31'),
    ('Q2-2024', '2024-04-01', '2024-06-30'),
    ('Q3-2024', '2024-07-01', '2024-09-30'),
    ('Q4-2024', '2024-10-01', '2024-12-31'),
    ('Q1-2025', '2025-01-01', '2025-03-31'),
    ('Q2-2025', '2025-04-01', '2025-06-30'),
    ('Q3-2025', '2025-07-01', '2025-09-30'),
    ('Q4-2025', '2025-10-01', '2025-12-31'),
]
N = len(QUARTERS)  # 8

# ── Scoring tables ───────────────────────────────────────────────────────────
# Consecutive-run bonus (applied once per run length across all quarters)
PROFIT_RUN  = {1: 8,  2: 15, 3: 30, 4: 50}
NEUTRAL_RUN = {1: 5,  2: 10, 3: 15, 4: 20}
LOSS_RUN    = {1: -5, 2: -15, 3: -30, 4: -50}

# Per-quarter recency weight: latest (index 7) → +8/-8, oldest (index 0) → +1/-1
# weight[i] = i+1  where i=0 is Q1-2024, i=7 is Q4-2025


def classify(pct_change):
    if pct_change > NEUTRAL_PCT:
        return 'Profit'
    elif pct_change < -NEUTRAL_PCT:
        return 'Loss'
    else:
        return 'Neutral'


def run_score(results):
    """Score consecutive runs of same category, capped at runs of 4."""
    score = 0
    i = 0
    while i < len(results):
        cat = results[i]
        run = 1
        while i + run < len(results) and results[i + run] == cat and run < 4:
            run += 1
        if cat == 'Profit':
            score += PROFIT_RUN[run]
        elif cat == 'Neutral':
            score += NEUTRAL_RUN[run]
        else:
            score += LOSS_RUN[run]
        i += run
    return score


def recency_score(results):
    """Per-quarter recency bonus: latest quarter has highest weight."""
    score = 0
    for i, cat in enumerate(results):
        weight = i + 1   # 1 for oldest, 8 for latest
        if cat == 'Profit':
            score += weight
        elif cat == 'Loss':
            score -= weight
        # Neutral → 0
    return score


def process_ticker(ticker, df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    row = {'Ticker': ticker}
    results = []   # category per quarter in order

    for label, start, end in QUARTERS:
        mask = (df.index >= start) & (df.index <= end)
        q_df = df.loc[mask]

        if q_df.empty:
            row[label] = 'N/A'
            results.append(None)
            continue

        open_price  = q_df['Close'].iloc[0]
        close_price = q_df['Close'].iloc[-1]

        if open_price == 0 or pd.isna(open_price):
            row[label] = 'N/A'
            results.append(None)
            continue

        pct = (close_price - open_price) / open_price * 100
        cat = classify(pct)
        row[label] = cat
        results.append(cat)

    # ── Score ────────────────────────────────────────────────────────────────
    valid = [r for r in results if r is not None]
    if not valid:
        row['Score'] = 50
        return row

    score = 50
    score += run_score(valid)
    score += recency_score(valid)
    row['Score'] = score
    return row


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    files = sorted(f for f in os.listdir(DAILY_DATA_DIR) if f.endswith('_daily.csv'))
    print(f"Processing {len(files)} tickers...")

    records = []
    for i, fname in enumerate(files):
        ticker = fname.replace('_daily.csv', '')
        path   = os.path.join(DAILY_DATA_DIR, fname)
        try:
            df = pd.read_csv(path, index_col='Date')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            row = process_ticker(ticker, df)
            records.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)} done")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    result_df = pd.DataFrame(records)
    # Order columns
    cols = ['Ticker'] + [q[0] for q in QUARTERS] + ['Score']
    result_df = result_df[cols]
    result_df = result_df.sort_values('Score', ascending=False).reset_index(drop=True)

    # ── Excel output with colour coding ──────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Performers')

        wb  = writer.book
        ws  = writer.sheets['Performers']

        fmt_profit  = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#276221'})  # green
        fmt_loss    = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})  # red
        fmt_neutral = wb.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})  # amber
        fmt_header  = wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
        fmt_score   = wb.add_format({'bold': True})

        # Header
        for col_num, col_name in enumerate(result_df.columns):
            ws.write(0, col_num, col_name, fmt_header)

        # Data rows
        q_labels = [q[0] for q in QUARTERS]
        q_col_indices = {label: result_df.columns.get_loc(label) for label in q_labels}
        score_col = result_df.columns.get_loc('Score')

        for row_num, row in result_df.iterrows():
            excel_row = row_num + 1
            for label, col_idx in q_col_indices.items():
                val = row[label]
                if val == 'Profit':
                    ws.write(excel_row, col_idx, val, fmt_profit)
                elif val == 'Loss':
                    ws.write(excel_row, col_idx, val, fmt_loss)
                elif val == 'Neutral':
                    ws.write(excel_row, col_idx, val, fmt_neutral)
                else:
                    ws.write(excel_row, col_idx, val)
            ws.write(excel_row, score_col, row['Score'], fmt_score)

        # Column widths
        ws.set_column(0, 0, 14)                          # Ticker
        ws.set_column(1, len(q_labels), 10)              # Quarter cols
        ws.set_column(score_col, score_col, 8)           # Score

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Rows: {len(result_df)}  |  Top 5:")
    print(result_df.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
