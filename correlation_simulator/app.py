"""
app.py — FastAPI backend for the Correlation Mean-Reversion Simulator
Run: uvicorn app:app --reload --port 5050
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import psycopg2
import psycopg2.extras

import pandas as pd
from db import get_conn, init_db
from engine import compute_signals_for_date, get_exit_signals, load_all_tickers

_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Correlation Simulator")

# ── CORS ──────────────────────────────────────────────────────────────────────
_allowed_origin = os.getenv("ALLOWED_ORIGIN", "https://www.mcubetechstudio.com")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin, "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API Key auth ───────────────────────────────────────────────────────────────
_API_KEY = os.getenv("API_KEY", "")

@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if request.url.path.startswith("/api"):
        # Support header OR query param (query param needed for SSE via EventSource)
        key = request.headers.get("x-api-key") or request.query_params.get("key")
        if _API_KEY and key != _API_KEY:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)

SIM_START  = date(2026, 1, 1)
SIM_END    = date.today()
INVEST_PER = 10_000.0


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    init_db()
    # Warmup is disabled at startup to keep idle RAM low (~80MB vs ~500MB).
    # Call GET /api/warmup manually before the first advance to pre-load caches.


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_state(cur) -> dict:
    cur.execute("SELECT sim_date, last_run_date, total_invested, total_value FROM sim.state WHERE id=1")
    row = cur.fetchone()
    return {
        "current_date"  : row[0],
        "last_run_date" : row[1],
        "total_invested": float(row[2] or 0),
        "total_value"   : float(row[3] or 0),
    }


def set_state(cur, current_date, last_run_date=None, total_invested=None, total_value=None):
    cur.execute("""
        UPDATE sim.state
        SET sim_date       = %s,
            last_run_date  = COALESCE(%s, last_run_date),
            total_invested = COALESCE(%s, total_invested),
            total_value    = COALESCE(%s, total_value)
        WHERE id = 1
    """, (current_date, last_run_date, total_invested, total_value))


def get_open_positions(cur) -> list:
    cur.execute("""
        SELECT id, ticker, peers, entry_date, entry_price, hold_days, amount_invested
        FROM   sim.stock_suggestions
        WHERE  status = 'OPEN'
    """)
    cols = ["id", "ticker", "peers", "entry_date", "entry_price", "hold_days", "amount_invested"]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def get_close_prices_on(tickers: list, on_date: date) -> dict:
    """Return {ticker: close_price} for a list of tickers on a specific date.
    Falls back to the nearest prior date if exact date has no data."""
    td = load_all_tickers()
    ts = pd.Timestamp(on_date)
    result = {}
    for ticker in tickers:
        df = td.get(ticker)
        if df is None:
            continue
        subset = df[df["Date"] <= ts]
        if subset.empty:
            continue
        result[ticker] = float(subset.iloc[-1]["Close"])
    return result


def next_business_day(d: date) -> date:
    d2 = d + timedelta(days=1)
    while d2.weekday() >= 5:
        d2 += timedelta(days=1)
    return d2


# ── Static UI ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status(as_of: str = None) -> dict[str, Any]:
    conn = get_conn()
    cur  = conn.cursor()
    state = get_state(cur)
    last_run = state.get("last_run_date")

    # If as_of date provided, show snapshot of that date
    if as_of:
        try:
            as_of_date = date.fromisoformat(as_of)
        except ValueError:
            as_of_date = None
    else:
        as_of_date = None

    # ── Open positions (as of the requested date, or current) ─────────────────
    if as_of_date:
        # Positions that were OPEN on as_of_date: entered on or before, not exited before
        cur.execute("""
            SELECT ticker, signal_date, entry_date, entry_price, peers,
                   entry_z, peer_slope_pct,
                   (%s::date - entry_date) AS hold_days,
                   amount_invested
            FROM   sim.stock_suggestions
            WHERE  entry_date <= %s
              AND  (exit_date IS NULL OR exit_date > %s)
            ORDER  BY entry_date ASC, ticker
        """, (as_of_date, as_of_date, as_of_date))
    else:
        cur.execute("""
            SELECT ticker, signal_date, entry_date, entry_price, peers,
                   entry_z, peer_slope_pct, hold_days, amount_invested
            FROM   sim.stock_suggestions
            WHERE  status = 'OPEN'
            ORDER  BY entry_date ASC, ticker
        """)
    cols = ["ticker","signal_date","entry_date","entry_price","peers",
            "entry_z","peer_slope_pct","hold_days","amount_invested"]
    open_pos = [dict(zip(cols, r)) for r in cur.fetchall()]
    for p in open_pos:
        for k in ["signal_date","entry_date"]:
            if p[k]: p[k] = str(p[k])
        for k in ["entry_price","entry_z","peer_slope_pct","amount_invested"]:
            if p[k] is not None: p[k] = float(p[k])
        p["hold_days"] = int(p["hold_days"] or 0)

    # Enrich with current price + unrealised P&L as of the viewed date
    price_date = as_of_date if as_of_date else last_run
    if price_date and open_pos:
        cur_prices = get_close_prices_on([p["ticker"] for p in open_pos], price_date)
        for p in open_pos:
            cp = cur_prices.get(p["ticker"])
            if cp and p["entry_price"]:
                p["current_price"] = round(cp, 2)
                p["unreal_pnl_pct"] = round((cp - p["entry_price"]) / p["entry_price"] * 100, 2)
                p["unreal_pnl_rs"]  = round((cp - p["entry_price"]) / p["entry_price"] * p["amount_invested"], 0)
            else:
                p["current_price"]  = None
                p["unreal_pnl_pct"] = None
                p["unreal_pnl_rs"]  = None

    # ── Buys on the viewed date ────────────────────────────────────────────────
    view_date = as_of_date if as_of_date else last_run
    todays_buys = []
    if view_date:
        cur.execute("""
            SELECT ticker, entry_price, entry_z, peer_slope_pct, peers
            FROM   sim.stock_suggestions
            WHERE  entry_date = %s
            ORDER  BY ticker
        """, (view_date,))
        bc = ["ticker","entry_price","entry_z","peer_slope_pct","peers"]
        for r in cur.fetchall():
            d = dict(zip(bc, r))
            for k in ["entry_price","entry_z","peer_slope_pct"]:
                if d[k] is not None: d[k] = float(d[k])
            todays_buys.append(d)

    # ── Sells on the viewed date ───────────────────────────────────────────────
    todays_sells = []
    if view_date:
        cur.execute("""
            SELECT ticker, entry_price, exit_price, pnl_pct, hold_days, exit_reason
            FROM   sim.stock_suggestions
            WHERE  exit_date = %s AND status = 'SOLD'
            ORDER  BY ticker
        """, (view_date,))
        sc = ["ticker","entry_price","exit_price","pnl_pct","hold_days","exit_reason"]
        for r in cur.fetchall():
            d = dict(zip(sc, r))
            for k in ["entry_price","exit_price","pnl_pct"]:
                if d[k] is not None: d[k] = float(d[k])
            d["hold_days"] = int(d["hold_days"] or 0)
            todays_sells.append(d)

    # ── Closed trade history (sorted by exit_date desc) ────────────────────────
    if as_of_date:
        cur.execute("""
            SELECT ticker, entry_date, exit_date, entry_price, exit_price,
                   pnl_pct, exit_reason, hold_days, amount_invested, status
            FROM   sim.stock_suggestions
            WHERE  exit_date <= %s
            ORDER  BY exit_date DESC, ticker
            LIMIT  500
        """, (as_of_date,))
    else:
        cur.execute("""
            SELECT ticker, entry_date, exit_date, entry_price, exit_price,
                   pnl_pct, exit_reason, hold_days, amount_invested, status
            FROM   sim.stock_suggestions
            WHERE  status IN ('SOLD','EXPIRED')
            ORDER  BY exit_date DESC, ticker
            LIMIT  500
        """)
    cols2 = ["ticker","entry_date","exit_date","entry_price","exit_price",
             "pnl_pct","exit_reason","hold_days","amount_invested","status"]
    history = [dict(zip(cols2, r)) for r in cur.fetchall()]
    for h in history:
        for k in ["entry_date","exit_date"]:
            if h[k]: h[k] = str(h[k])
        for k in ["entry_price","exit_price","pnl_pct","amount_invested"]:
            if h[k] is not None: h[k] = float(h[k])
        h["hold_days"] = int(h["hold_days"] or 0)

    if as_of_date:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE exit_date <= %s AND pnl_pct > 0)  AS wins,
                COUNT(*) FILTER (WHERE exit_date <= %s AND pnl_pct <= 0) AS losses,
                COUNT(*) FILTER (WHERE exit_date <= %s)                   AS total_closed,
                COALESCE(SUM(pnl_pct * amount_invested / 100) FILTER (WHERE exit_date <= %s), 0) AS realised_pnl,
                COALESCE(SUM(amount_invested) FILTER (WHERE exit_date <= %s), 0)                  AS closed_invested,
                COALESCE(SUM(amount_invested) FILTER (WHERE entry_date <= %s AND (exit_date IS NULL OR exit_date > %s)), 0) AS open_invested
            FROM sim.stock_suggestions
        """, (as_of_date, as_of_date, as_of_date, as_of_date, as_of_date, as_of_date, as_of_date))
    else:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status='SOLD' AND pnl_pct > 0)  AS wins,
                COUNT(*) FILTER (WHERE status='SOLD' AND pnl_pct <= 0) AS losses,
                COUNT(*) FILTER (WHERE status='SOLD')                   AS total_closed,
                COALESCE(SUM(pnl_pct * amount_invested / 100) FILTER (WHERE status='SOLD'), 0) AS realised_pnl,
                COALESCE(SUM(amount_invested) FILTER (WHERE status='SOLD'), 0)                  AS closed_invested,
                COALESCE(SUM(amount_invested) FILTER (WHERE status='OPEN'), 0)                  AS open_invested
            FROM sim.stock_suggestions
        """)
    r = cur.fetchone()
    wins, losses, total_closed, realised_pnl, closed_invested, open_invested = (
        int(r[0] or 0), int(r[1] or 0), int(r[2] or 0),
        float(r[3] or 0), float(r[4] or 0), float(r[5] or 0),
    )
    total_inv = closed_invested + open_invested
    win_rate  = round(wins / total_closed * 100, 1) if total_closed > 0 else 0
    gain_pct  = round(realised_pnl / total_inv * 100, 2) if total_inv > 0 else 0

    cur.execute("""
        SELECT snap_date, open_positions, bought, sold,
               total_invested, realised_pnl, unrealised_pnl
        FROM   sim.daily_portfolio
        ORDER  BY snap_date
    """)
    chart_rows = cur.fetchall()
    chart = {
        "dates"          : [str(r[0]) for r in chart_rows],
        "open_positions" : [int(r[1] or 0) for r in chart_rows],
        "bought"         : [int(r[2] or 0) for r in chart_rows],
        "sold"           : [int(r[3] or 0) for r in chart_rows],
        "total_invested" : [float(r[4] or 0) for r in chart_rows],
        "realised_pnl"   : [float(r[5] or 0) for r in chart_rows],
    }

    cur.close(); conn.close()

    return {
        "current_date"  : str(state["current_date"]),
        "last_run_date" : str(last_run) if last_run else None,
        "viewed_date"   : str(as_of_date) if as_of_date else (str(last_run) if last_run else str(state["current_date"])),
        "sim_end"       : str(SIM_END),
        "open_positions": open_pos,
        "todays_buys"   : todays_buys,
        "todays_sells"  : todays_sells,
        "history"       : history,
        "metrics"       : {
            "total_invested": round(total_inv, 2),
            "realised_pnl"  : round(realised_pnl, 2),
            "open_invested" : round(open_invested, 2),
            "win_rate"      : win_rate,
            "wins"          : wins,
            "losses"        : losses,
            "total_closed"  : total_closed,
            "gain_pct"      : gain_pct,
        },
        "chart": chart,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_advance_sync(current_date: date, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Runs entirely in a ThreadPoolExecutor worker.
    Pushes SSE strings into `queue` so the async generator can yield them immediately.
    """
    t0 = time.time()

    def push(event: str, data: dict):
        asyncio.run_coroutine_threadsafe(queue.put(_sse(event, data)), loop)

    def elapsed():
        return round(time.time() - t0, 2)

    try:
        push("progress", {"pct": 5,  "step": "Connecting to DB…", "elapsed": elapsed()})

        conn = get_conn()
        cur  = conn.cursor()

        push("progress", {"pct": 12, "step": "Loading open positions…", "elapsed": elapsed()})

        open_pos     = get_open_positions(cur)
        exits        = get_exit_signals(open_pos, current_date)
        exit_tickers = {e["ticker"] for e in exits}

        push("progress", {"pct": 25, "elapsed": elapsed(),
            "step": f"Processing {len(exits)} exit(s) from {len(open_pos)} open position(s)…"})

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
                "UPDATE sim.stock_suggestions SET hold_days=hold_days+1 WHERE status='OPEN' AND ticker NOT IN %s",
                (tuple(exit_tickers),))
        else:
            cur.execute("UPDATE sim.stock_suggestions SET hold_days=hold_days+1 WHERE status='OPEN'")

        push("progress", {"pct": 38, "step": "Scanning buy signals across all tickers…", "elapsed": elapsed()})

        # Build last_trade_dates for f_days_gap ML feature
        cur.execute("SELECT ticker, MAX(entry_date) FROM sim.stock_suggestions GROUP BY ticker")
        last_trade_dates = {r[0]: r[1] for r in cur.fetchall() if r[1]}

        signals  = compute_signals_for_date(current_date, last_trade_dates)   # ← slow part
        buys     = signals.get("buys", [])

        push("progress", {"pct": 82, "elapsed": elapsed(),
            "step": f"Found {len(buys)} signal(s) — writing to DB…"})

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

        push("progress", {"pct": 92, "step": "Saving daily snapshot…", "elapsed": elapsed()})

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
              open_positions=EXCLUDED.open_positions, bought=EXCLUDED.bought,
              sold=EXCLUDED.sold, total_invested=EXCLUDED.total_invested,
              realised_pnl=EXCLUDED.realised_pnl
        """, (current_date, open_count, len(new_buys), len(exits),
              closed_inv + open_inv, realised_pnl, 0))

        next_date = next_business_day(current_date)
        set_state(cur, next_date, last_run_date=current_date)
        conn.commit()
        cur.close()
        conn.close()

        push("done", {
            "pct"         : 100,
            "step"        : "Complete",
            "elapsed"     : elapsed(),
            "advanced_to" : str(next_date),
            "buys_today"  : len(new_buys),
            "sells_today" : len(exits),
            "buy_tickers" : [b["ticker"] for b in new_buys],
            "sell_details": exits,
        })

    except Exception as e:
        push("error", {"pct": 0, "step": f"Error: {e}", "elapsed": elapsed()})

    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel → stop generator


@app.get("/api/advance/stream")
async def api_advance_stream():
    conn = get_conn()
    cur  = conn.cursor()
    state = get_state(cur)
    current_date = state["current_date"]
    cur.close(); conn.close()

    if current_date >= SIM_END:
        raise HTTPException(status_code=400, detail="Simulation already at today's date.")

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # Kick off the heavy work in a thread — it pushes events into the queue
    loop.run_in_executor(_executor, _run_advance_sync, current_date, queue, loop)

    async def event_stream():
        while True:
            msg = await queue.get()
            if msg is None:   # sentinel = finished
                break
            yield msg

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/advance")
def api_advance() -> dict[str, Any]:
    """Non-streaming fallback (kept for compatibility)."""
    conn = get_conn()
    cur  = conn.cursor()
    state = get_state(cur)
    current_date = state["current_date"]

    if current_date >= SIM_END:
        cur.close(); conn.close()
        raise HTTPException(status_code=400, detail="Simulation already at today's date.")

    open_pos     = get_open_positions(cur)
    exits        = get_exit_signals(open_pos, current_date)
    exit_tickers = {e["ticker"] for e in exits}

    for ex in exits:
        pos = next((p for p in open_pos if p["ticker"] == ex["ticker"]), None)
        if not pos: continue
        cur.execute("""
            UPDATE sim.stock_suggestions
            SET status='SOLD', exit_date=%s, exit_price=%s,
                pnl_pct=%s, hold_days=%s, exit_reason=%s WHERE id=%s
        """, (current_date, ex["exit_price"], ex["pnl_pct"],
              ex["hold_days"], ex["exit_reason"], pos["id"]))

    if exit_tickers:
        cur.execute("UPDATE sim.stock_suggestions SET hold_days=hold_days+1 WHERE status='OPEN' AND ticker NOT IN %s",
                    (tuple(exit_tickers),))
    else:
        cur.execute("UPDATE sim.stock_suggestions SET hold_days=hold_days+1 WHERE status='OPEN'")

    cur.execute("SELECT ticker, MAX(entry_date) FROM sim.stock_suggestions GROUP BY ticker")
    last_trade_dates = {r[0]: r[1] for r in cur.fetchall() if r[1]}
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
          open_positions=EXCLUDED.open_positions, bought=EXCLUDED.bought,
          sold=EXCLUDED.sold, total_invested=EXCLUDED.total_invested,
          realised_pnl=EXCLUDED.realised_pnl
    """, (current_date, open_count, len(new_buys), len(exits),
          closed_inv + open_inv, realised_pnl, 0))

    next_date = next_business_day(current_date)
    set_state(cur, next_date, last_run_date=current_date)
    conn.commit(); cur.close(); conn.close()

    return {
        "advanced_to": str(next_date), "buys_today": len(new_buys),
        "sells_today": len(exits), "buy_tickers": [b["ticker"] for b in new_buys],
        "sell_details": exits,
    }


@app.post("/api/reset")
def api_reset() -> dict[str, Any]:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM sim.stock_suggestions")
    cur.execute("DELETE FROM sim.daily_portfolio")
    cur.execute("UPDATE sim.state SET sim_date='2026-01-01', last_run_date=NULL, total_invested=0, total_value=0 WHERE id=1")
    conn.commit()
    cur.close(); conn.close()
    return {"status": "reset", "current_date": "2026-01-01"}


@app.get("/api/warmup")
def api_warmup() -> dict[str, Any]:
    count = len(load_all_tickers())
    return {"loaded_tickers": count}


@app.post("/api/sleep")
def api_sleep() -> dict[str, Any]:
    """Clear all in-memory caches to drop RAM from ~500MB back to ~80MB.
    Call this when done using the app. Call /api/warmup again before next use."""
    import engine
    engine._ticker_data_cache = None
    engine._price_df_cache    = None
    engine._peers_cache       = None
    engine._corr_strength_cache = None
    engine._zscore_cache      = {}
    engine._open_prices_cache = {}
    engine._ml_model          = None
    engine._ml_features       = None
    engine._fund_cache        = None
    engine._mkt_regime_cache  = None
    return {"status": "sleeping", "message": "Caches cleared. Call /api/warmup to wake up."}
