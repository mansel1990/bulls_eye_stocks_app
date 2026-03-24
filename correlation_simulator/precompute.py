"""
precompute.py
─────────────
Runs the full simulation from 2026-01-01 to today, day by day,
storing all signals, trades, and portfolio snapshots into PostgreSQL.

Run once (or re-run after reset):
    python precompute.py

Progress is printed to console. Safe to interrupt and resume —
it picks up from sim.state.sim_date automatically.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
from datetime import date, timedelta

from db import get_conn, init_db
from engine import (
    load_all_tickers, _build_price_df, _get_peers_map,
    compute_signals_for_date, get_exit_signals,
)

SIM_START  = date(2026, 1, 1)
SIM_END    = date.today()
INVEST_PER = 10_000.0


# ── helpers ───────────────────────────────────────────────────────────────────

def next_business_day(d: date) -> date:
    d2 = d + timedelta(days=1)
    while d2.weekday() >= 5:
        d2 += timedelta(days=1)
    return d2


def business_days_between(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d < end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def get_state(cur) -> dict:
    cur.execute("SELECT sim_date, last_run_date FROM sim.state WHERE id=1")
    row = cur.fetchone()
    return {"current_date": row[0], "last_run_date": row[1]}


def set_state(cur, sim_date: date, last_run: date):
    cur.execute(
        "UPDATE sim.state SET sim_date=%s, last_run_date=%s WHERE id=1",
        (sim_date, last_run),
    )


def get_open_positions(cur) -> list:
    cur.execute("""
        SELECT id, ticker, peers, entry_date, entry_price, hold_days, amount_invested
        FROM sim.stock_suggestions WHERE status='OPEN'
    """)
    cols = ["id","ticker","peers","entry_date","entry_price","hold_days","amount_invested"]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / total) if total else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Correlation Simulator — Pre-compute all days")
    print("=" * 60)

    init_db()

    # ── Warm all caches first (done once) ─────────────────────────────────────
    print("\nLoading tickers & building caches…", end=" ", flush=True)
    t0 = time.time()
    td  = load_all_tickers()
    pdf = _build_price_df(td)
    _get_peers_map(td, pdf)
    print(f"done in {time.time()-t0:.1f}s  ({len(td)} tickers)\n")

    # ── Determine start point ─────────────────────────────────────────────────
    conn = get_conn()
    cur  = conn.cursor()
    state = get_state(cur)
    start = state["current_date"]   # resume from where we left off

    if start >= SIM_END:
        print("Simulation already complete up to today. Reset first to rerun.")
        cur.close(); conn.close()
        return

    all_days   = business_days_between(start, SIM_END)
    total_days = len(all_days)
    print(f"Simulating {total_days} business days: {start} → {SIM_END}\n")

    wall_start = time.time()
    days_done  = 0

    # last_trade_dates: tracks most recent entry date per ticker for f_days_gap feature
    last_trade_dates = {}
    cur.execute("SELECT ticker, MAX(entry_date) FROM sim.stock_suggestions GROUP BY ticker")
    for row in cur.fetchall():
        if row[1]:
            last_trade_dates[row[0]] = row[1]

    for current_date in all_days:
        t_day = time.time()

        # ── Exits ─────────────────────────────────────────────────────────────
        open_pos     = get_open_positions(cur)
        exits        = get_exit_signals(open_pos, current_date)
        exit_tickers = {e["ticker"] for e in exits}

        for ex in exits:
            pos = next((p for p in open_pos if p["ticker"] == ex["ticker"]), None)
            if not pos:
                continue
            cur.execute("""
                UPDATE sim.stock_suggestions
                SET status='SOLD', exit_date=%s, exit_price=%s,
                    pnl_pct=%s, hold_days=%s, exit_reason=%s
                WHERE id=%s
            """, (current_date, ex["exit_price"], ex["pnl_pct"],
                  ex["hold_days"], ex["exit_reason"], pos["id"]))

        if exit_tickers:
            cur.execute(
                "UPDATE sim.stock_suggestions SET hold_days=hold_days+1 "
                "WHERE status='OPEN' AND ticker NOT IN %s",
                (tuple(exit_tickers),))
        else:
            cur.execute(
                "UPDATE sim.stock_suggestions SET hold_days=hold_days+1 WHERE status='OPEN'")

        # ── Buys ──────────────────────────────────────────────────────────────
        signals  = compute_signals_for_date(current_date, last_trade_dates)
        buys     = signals.get("buys", [])

        cur.execute("SELECT ticker FROM sim.stock_suggestions WHERE status='OPEN'")
        already_open = {r[0] for r in cur.fetchall()}
        new_buys = [b for b in buys if b["ticker"] not in already_open]

        for b in new_buys:
            cur.execute("""
                INSERT INTO sim.stock_suggestions
                  (signal_date, ticker, peers, entry_z, peer_slope_pct,
                   entry_date, entry_price, status, amount_invested)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'OPEN',%s)
                ON CONFLICT (signal_date, ticker) DO NOTHING
            """, (current_date, b["ticker"], b["peers"], b["entry_z"],
                  b["peer_slope_pct"], current_date, b["entry_price"], INVEST_PER))
            last_trade_dates[b["ticker"]] = current_date

        # ── Snapshot ──────────────────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM sim.stock_suggestions WHERE status='OPEN'")
        open_count = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(amount_invested),0) FROM sim.stock_suggestions WHERE status='SOLD'")
        closed_inv = float(cur.fetchone()[0])

        cur.execute("SELECT COALESCE(SUM(pnl_pct*amount_invested/100),0) FROM sim.stock_suggestions WHERE status='SOLD'")
        realised_pnl = float(cur.fetchone()[0])

        cur.execute("SELECT COALESCE(SUM(amount_invested),0) FROM sim.stock_suggestions WHERE status='OPEN'")
        open_inv = float(cur.fetchone()[0])

        cur.execute("""
            INSERT INTO sim.daily_portfolio
              (snap_date, open_positions, bought, sold, total_invested, realised_pnl, unrealised_pnl)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (snap_date) DO UPDATE SET
              open_positions=EXCLUDED.open_positions,
              bought=EXCLUDED.bought,
              sold=EXCLUDED.sold,
              total_invested=EXCLUDED.total_invested,
              realised_pnl=EXCLUDED.realised_pnl
        """, (current_date, open_count, len(new_buys), len(exits),
              closed_inv + open_inv, realised_pnl, 0))

        next_date = next_business_day(current_date)
        set_state(cur, next_date, current_date)
        conn.commit()

        # ── Progress ──────────────────────────────────────────────────────────
        days_done += 1
        elapsed   = time.time() - wall_start
        avg_s     = elapsed / days_done
        remaining = avg_s * (total_days - days_done)
        pct       = days_done / total_days * 100

        eta_str = (
            f"{int(remaining//60)}m {int(remaining%60)}s"
            if remaining > 60
            else f"{remaining:.0f}s"
        )
        day_ms = (time.time() - t_day) * 1000

        print(
            f"\r{bar(days_done, total_days)} "
            f"{pct:5.1f}%  {current_date}  "
            f"buy={len(new_buys):2d} sell={len(exits):2d} open={open_count:3d}  "
            f"{day_ms:5.0f}ms/day  ETA {eta_str}   ",
            end="", flush=True,
        )

    print(f"\n\n{'='*60}")
    print(f"  DONE — {days_done} days simulated in {time.time()-wall_start:.1f}s")

    # ── Final summary ─────────────────────────────────────────────────────────
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status='SOLD' AND pnl_pct > 0) AS wins,
            COUNT(*) FILTER (WHERE status='SOLD' AND pnl_pct <= 0) AS losses,
            COUNT(*) FILTER (WHERE status='SOLD') AS total,
            COALESCE(AVG(pnl_pct) FILTER (WHERE status='SOLD'), 0) AS avg_pnl,
            COALESCE(SUM(pnl_pct*amount_invested/100) FILTER (WHERE status='SOLD'), 0) AS realised,
            COUNT(*) FILTER (WHERE status='OPEN') AS open_now
        FROM sim.stock_suggestions
    """)
    r = cur.fetchone()
    wins, losses, total, avg_pnl, realised, open_now = r
    wr = wins / total * 100 if total else 0

    print(f"  Closed trades : {total}  (Win rate: {wr:.1f}%)")
    print(f"  Avg PnL       : {avg_pnl:+.2f}%")
    print(f"  Realised P&L  : ₹{float(realised):,.0f}")
    print(f"  Still open    : {open_now} positions")
    print(f"{'='*60}\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
