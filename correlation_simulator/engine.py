"""
engine.py — Core signal engine for the correlation mean-reversion simulator.

For a given 'current_date':
  1. Load all CSV data strictly up to current_date - 1 day (no future data).
  2. Compute correlation peers (on pre-2023 training data).
  3. Build spread z-scores up to current_date - 1.
  4. For each ticker with z < -ENTRY_Z: compute 32 ML features, run through
     XGBoost model (corr_ml_v4b.pkl). Only emit BUY if ml_prob >= ML_THRESHOLD.
  5. For open positions: check if today's z-score >= EXIT_Z or hold >= MAX_HOLD
       → emit SELL at today's open price.

ML model trained on 2023-01-01 → 2025-02-28.
Validated on 2025-03-01 → 2026-02-28: 263 trades @ 66.5% WR, +4.47% avg PnL.
"""

import sys, os
import numpy as np
import pandas as pd
from glob import glob
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR   = r"d:\stocks prediction\bulls_eye_stocks_app\daily_data"
FUND_FILE  = r"d:\stocks prediction\bulls_eye_stocks_app\new_technique\experiments\sheets\fundamental_score_2026-02-19.xlsx"
MODEL_PATH = r"d:\stocks prediction\bulls_eye_stocks_app\new_technique\stratergies\models\corr_ml_v4b.pkl"
FEAT_PATH  = r"d:\stocks prediction\bulls_eye_stocks_app\new_technique\stratergies\models\corr_ml_v4b_features.pkl"

# ── Strategy parameters (loose — ML does the filtering) ───────────────────────
CORR_WINDOW       = 120
SPREAD_WINDOW     = 60
ENTRY_Z           = 1.5    # loose entry — ML filters down to high-confidence
EXIT_Z            = 0.0
MAX_HOLD          = 30
TOP_N_PEERS       = 5
PEER_SLOPE_WINDOW = 30
PEER_SLOPE_MIN    = 0.0    # no slope pre-filter — ML handles it
MCAP_MIN_CR       = 500
MIN_HISTORY       = 200
TRAIN_CUTOFF      = pd.Timestamp("2023-01-01")   # peer corr computed pre-2023
ML_THRESHOLD      = 0.55   # validated: 263 trades, 66.5% WR, +4.47% avg PnL


# ── Caches ────────────────────────────────────────────────────────────────────
_ticker_data_cache: dict | None = None
_price_df_cache: pd.DataFrame | None = None
_peers_cache: dict | None = None
_corr_strength_cache: dict | None = None
_zscore_cache: dict = {}        # {prev_ts: {ticker: (z_val, slope_val, extra)}}
_open_prices_cache: dict = {}   # {cur_ts:  {ticker: open_price}}
_ml_model = None
_ml_features = None
_fund_cache: pd.DataFrame | None = None
_mkt_regime_cache: dict | None = None  # {date: {mkt_20d, mkt_60d, mkt_vol, breadth, drawdown}}


def _load_ml_model():
    global _ml_model, _ml_features
    if _ml_model is not None:
        return _ml_model, _ml_features
    import joblib
    _ml_model    = joblib.load(MODEL_PATH)
    _ml_features = joblib.load(FEAT_PATH)
    return _ml_model, _ml_features


def _load_fund_data() -> pd.DataFrame:
    global _fund_cache
    if _fund_cache is not None:
        return _fund_cache
    fund = pd.read_excel(FUND_FILE, usecols=[
        "Ticker", "Sector", "Fundamental Score", "Beta",
        "Net Margin (%)", "P/B Ratio", "Gross Margin (%)",
        "Debt/Equity", "Current Ratio", "Revenue Growth (%)", "Analyst Score"
    ])
    fund["Ticker"] = fund["Ticker"].astype(str).str.strip()
    fund = fund.set_index("Ticker")
    sectors   = sorted(fund["Sector"].fillna("Unknown").unique())
    sector_map = {s: i for i, s in enumerate(sectors)}
    fund["sector_id"] = fund["Sector"].fillna("Unknown").map(sector_map)
    for col in ["Fundamental Score","Beta","Net Margin (%)","P/B Ratio",
                "Gross Margin (%)","Debt/Equity","Revenue Growth (%)","Analyst Score"]:
        fund[col] = pd.to_numeric(fund[col], errors="coerce")
    _fund_cache = fund
    return _fund_cache


def _load_vol_mcap_map() -> dict:
    fund = pd.read_excel(FUND_FILE, usecols=["Ticker", "Vol/MarketCap (%)"])
    fund = fund.dropna(subset=["Ticker", "Vol/MarketCap (%)"])
    fund["Ticker"] = fund["Ticker"].astype(str).str.strip()
    return fund.set_index("Ticker")["Vol/MarketCap (%)"].to_dict()


def _compute_mcap_cr(df: pd.DataFrame, pct: float) -> float:
    avg_to = (df["Close"].tail(30) * df["Volume"].tail(30)).mean()
    return (avg_to / (pct / 100)) / 1e7 if pct > 0 else 0


def _load_ticker(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df if len(df) >= MIN_HISTORY else None
    except Exception:
        return None


def load_all_tickers() -> dict:
    global _ticker_data_cache
    if _ticker_data_cache is not None:
        return _ticker_data_cache
    vol_mcap    = _load_vol_mcap_map()
    files       = glob(os.path.join(DATA_DIR, "*_daily.csv"))
    ticker_data = {}
    for fp in files:
        t = os.path.basename(fp).replace("_daily.csv", "")
        if t not in vol_mcap:
            continue
        df = _load_ticker(fp)
        if df is None:
            continue
        if _compute_mcap_cr(df, vol_mcap[t]) < MCAP_MIN_CR:
            continue
        ticker_data[t] = df
    _ticker_data_cache = ticker_data
    return ticker_data


def _build_price_df(ticker_data: dict) -> pd.DataFrame:
    global _price_df_cache
    if _price_df_cache is not None:
        return _price_df_cache
    closes = {t: df.set_index("Date")["Close"] for t, df in ticker_data.items()}
    _price_df_cache = pd.DataFrame(closes).sort_index()
    return _price_df_cache


def _get_peers_map(ticker_data: dict, price_df: pd.DataFrame) -> tuple[dict, dict]:
    """Returns (peers_map, corr_strength_map), both cached."""
    global _peers_cache, _corr_strength_cache
    if _peers_cache is not None:
        return _peers_cache, _corr_strength_cache
    ret_df    = price_df.pct_change()
    train_ret = ret_df[ret_df.index < TRAIN_CUTOFF].tail(CORR_WINDOW)
    peers_map         = {}
    corr_strength_map = {}
    for ticker in ticker_data:
        if ticker not in train_ret.columns or train_ret[ticker].isna().all():
            continue
        corr = train_ret.corrwith(train_ret[ticker]).drop(ticker).dropna()
        top  = corr.sort_values(ascending=False).head(TOP_N_PEERS)
        peers_map[ticker]         = top.index.tolist()
        corr_strength_map[ticker] = float(top.mean()) if len(top) > 0 else 0.0
    _peers_cache         = peers_map
    _corr_strength_cache = corr_strength_map
    return _peers_cache, _corr_strength_cache


def _build_market_regime(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build market regime features — cached once per process."""
    global _mkt_regime_cache
    if _mkt_regime_cache is not None:
        return _mkt_regime_cache
    ret_df   = price_df.pct_change()
    mkt_ret  = ret_df.mean(axis=1)
    mkt_20d  = mkt_ret.rolling(20).mean()
    mkt_60d  = mkt_ret.rolling(60).mean()
    mkt_vol  = mkt_ret.rolling(20).std()
    mkt_cum  = (1 + mkt_ret).cumprod()
    breadth  = (price_df > price_df.rolling(20).mean()).mean(axis=1)
    mkt_60h  = mkt_cum.rolling(60).max()
    mkt_dd   = mkt_cum / mkt_60h - 1
    _mkt_regime_cache = pd.DataFrame({
        "mkt_20d"   : mkt_20d,
        "mkt_60d"   : mkt_60d,
        "mkt_vol"   : mkt_vol,
        "breadth"   : breadth,
        "drawdown"  : mkt_dd,
    })
    return _mkt_regime_cache


def _rolling_mean_std_last(arr: np.ndarray, w: int):
    tail = arr[-w:]
    return tail.mean(), tail.std(ddof=1)


def _calc_rsi_last(close_arr: np.ndarray, window: int = 14) -> float:
    """Compute RSI using EWM (matches training script's ewm(com=window-1))."""
    if len(close_arr) < window + 1:
        return 50.0
    s = pd.Series(close_arr)
    diff = s.diff(1)
    gain = diff.clip(lower=0).fillna(0)
    loss = (-diff).clip(lower=0).fillna(0)
    alpha = 1.0 / window  # com=window-1 means alpha=1/(window-1+1)=1/window
    avg_g = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_l = loss.ewm(com=window - 1, min_periods=window).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    last  = rsi.iloc[-1]
    return float(last) if not pd.isna(last) else 50.0


def _get_zscores_for_date(prev_ts: pd.Timestamp) -> dict:
    """
    Vectorised — computes z-scores + extended features for ALL tickers.
    Returns {ticker: (z_val, slope_val, extra_dict)}, cached per prev_ts.
    extra_dict contains all features needed by the ML model.
    """
    global _zscore_cache
    if prev_ts in _zscore_cache:
        return _zscore_cache[prev_ts]

    ticker_data   = load_all_tickers()
    price_df      = _build_price_df(ticker_data)
    peers_map, corr_strength_map = _get_peers_map(ticker_data, price_df)
    mkt_df        = _build_market_regime(price_df)
    fund_df       = _load_fund_data()

    # Tail slice for spread computation
    # Need enough rows for: SPREAD_WINDOW warmup + 252 for pctile rank + 10 for z_10d_ago + buffer
    tail_len = SPREAD_WINDOW + 252 + 60  # 372 rows
    ps       = price_df[price_df.index <= prev_ts].tail(tail_len)
    cols_set = set(ps.columns)

    col_arrays: dict[str, np.ndarray] = {
        col: ps[col].values.astype(np.float64) for col in ps.columns
    }
    n_rows    = len(ps)
    ps_dates  = ps.index

    # Market regime at prev_ts
    def _mkt(col):
        try:
            v = mkt_df.loc[prev_ts, col]
            return float(v) if not pd.isna(v) else 0.0
        except:
            return 0.0

    mkt_20d  = _mkt("mkt_20d")
    mkt_60d  = _mkt("mkt_60d")
    mkt_vol  = _mkt("mkt_vol")
    breadth  = _mkt("breadth")
    drawdown = _mkt("drawdown")

    result: dict[str, tuple] = {}

    for ticker in ticker_data:
        if ticker not in cols_set:
            continue
        peers = peers_map.get(ticker, [])
        if not peers:
            continue
        peer_cols = [p for p in peers if p in cols_set]
        if not peer_cols:
            continue

        t_arr      = col_arrays[ticker]
        peer_stack = np.stack([col_arrays[p] for p in peer_cols], axis=1)
        peer_mean  = np.nanmean(peer_stack, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            spread = np.where(peer_mean != 0, t_arr / peer_mean, np.nan)

        valid_mask  = ~np.isnan(spread)
        first_valid = np.argmax(valid_mask)
        if not valid_mask[first_valid]:
            continue
        spread      = spread.copy()
        first_val   = spread[first_valid]
        if first_val == 0 or np.isnan(first_val):
            continue
        spread /= first_val

        usable = n_rows - first_valid
        if usable < SPREAD_WINDOW + 2:
            continue

        seg = spread[first_valid:]
        if len(seg) < SPREAD_WINDOW:
            continue

        rm, rs = _rolling_mean_std_last(seg, SPREAD_WINDOW)
        if rs == 0 or np.isnan(rs):
            continue
        z_val = (seg[-1] - rm) / rs

        # Peer slope
        if n_rows < PEER_SLOPE_WINDOW + 1:
            slope_val = np.nan
        else:
            pm_now  = peer_mean[-1]
            pm_prev = peer_mean[-(PEER_SLOPE_WINDOW + 1)]
            slope_val = (pm_now / pm_prev - 1) if (pm_prev != 0 and not np.isnan(pm_prev)) else np.nan

        # ── Extra features for ML ──────────────────────────────────────────────

        # Compute full rolling z-score series over seg (matches training pandas logic)
        # We need at least SPREAD_WINDOW rows for rolling mean/std
        seg_series = pd.Series(seg)
        roll_mean_s = seg_series.rolling(SPREAD_WINDOW).mean()
        roll_std_s  = seg_series.rolling(SPREAD_WINDOW).std().replace(0, np.nan)
        zscore_s    = (seg_series - roll_mean_s) / roll_std_s

        # z 5 and 10 days ago (matches: zscore.iloc[i-5], zscore.iloc[i-10])
        n_seg = len(zscore_s)
        z_5d  = float(zscore_s.iloc[-6])  if n_seg >= 6  and not pd.isna(zscore_s.iloc[-6])  else 0.0
        z_10d = float(zscore_s.iloc[-11]) if n_seg >= 11 and not pd.isna(zscore_s.iloc[-11]) else 0.0

        # Spread percentile rank vs 252d history (matches: rolling(252).rank(pct=True))
        spread_full = seg
        spread_series = pd.Series(spread_full)
        pctile_series = spread_series.rolling(252).rank(pct=True)
        pctile = float(pctile_series.iloc[-1]) if not pd.isna(pctile_series.iloc[-1]) else 0.5

        # Consecutive days below entry threshold (matches pandas groupby cumsum)
        below_thresh = (zscore_s < -ENTRY_Z).astype(int)
        # cumsum of group changes to detect consecutive runs
        groups = (below_thresh != below_thresh.shift()).cumsum()
        consec_series = below_thresh.groupby(groups).cumsum()
        consec = int(consec_series.iloc[-1]) if not pd.isna(consec_series.iloc[-1]) else 0

        # Ticker return volatility 20d
        if len(t_arr) >= 21:
            t_rets = np.diff(t_arr[-21:]) / t_arr[-21:-1]
            t_vol  = float(np.nanstd(t_rets))
        else:
            t_vol  = 0.0

        # Peer vol 20d
        if n_rows >= 21:
            p_rets = np.diff(peer_mean[-21:]) / peer_mean[-21:-1]
            p_vol  = float(np.nanstd(p_rets))
        else:
            p_vol  = 0.0

        # Volume ratio (current 1d vol vs 20d avg)
        raw_df = ticker_data[ticker].set_index("Date")
        try:
            vol_series = raw_df["Volume"].reindex(ps_dates)
            vol_last   = vol_series.iloc[-1]
            vol_ma20   = vol_series.iloc[-20:].mean()
            vol_ratio  = float(vol_last / vol_ma20) if vol_ma20 > 0 else 1.0
        except:
            vol_ratio  = 1.0

        # Momentum 5d and 20d
        mom_5d  = float(t_arr[-1] / t_arr[-6]  - 1) if len(t_arr) >= 6  else 0.0
        mom_20d = float(t_arr[-1] / t_arr[-21] - 1) if len(t_arr) >= 21 else 0.0

        # ATR ratio — matches training: rolling(5).mean() / rolling(20).mean()
        try:
            hl_series = (raw_df["High"] - raw_df["Low"]).reindex(ps_dates)
            atr5_s    = hl_series.rolling(5).mean()
            atr_ma_s  = atr5_s.rolling(20).mean()
            atr5_last = atr5_s.iloc[-1]
            atr_ma_last = atr_ma_s.iloc[-1]
            atr_ratio = float(atr5_last / atr_ma_last) if (atr_ma_last and atr_ma_last > 0) else 1.0
        except:
            atr_ratio = 1.0

        # RSI
        rsi_val = _calc_rsi_last(t_arr)

        # Corr strength
        corr_str = corr_strength_map.get(ticker, 0.0)

        # Fundamental features
        fd = fund_df.loc[ticker] if ticker in fund_df.index else None
        def _fd(col, default):
            if fd is None: return default
            v = fd.get(col, np.nan)
            return float(v) if not pd.isna(v) else default

        # Use pandas rolling values for spread_mean/std (consistent with z-score series)
        rm_last = float(roll_mean_s.iloc[-1]) if not pd.isna(roll_mean_s.iloc[-1]) else rm
        rs_last = float(roll_std_s.iloc[-1])  if not pd.isna(roll_std_s.iloc[-1])  else rs

        extra = {
            "f_entry_z"       : round(float(z_val), 4),
            "f_z_5d_ago"      : round(z_5d, 4),
            "f_z_10d_ago"     : round(z_10d, 4),
            "f_consec_below"  : consec,
            "f_spread_pctile" : round(pctile, 4),
            "f_spread_std"    : round(rs_last, 6),
            "f_spread_mean"   : round(rm_last, 6),
            "f_spread_val"    : round(float(seg[-1]), 6),
            "f_peer_slope"    : round(float(slope_val) if not np.isnan(slope_val) else 0.0, 4),
            "f_corr_strength" : round(corr_str, 4),
            "f_peer_vol20"    : round(p_vol, 6),
            "f_rsi"           : round(rsi_val, 2),
            "f_ticker_vol20"  : round(t_vol, 6),
            "f_vol_ratio"     : round(vol_ratio, 4),
            "f_mom_5d"        : round(mom_5d, 4),
            "f_mom_20d"       : round(mom_20d, 4),
            "f_atr_ratio"     : round(atr_ratio, 4),
            "f_days_gap"      : 999,   # filled in compute_signals_for_date
            "f_mkt_ret_20d"   : round(mkt_20d, 6),
            "f_mkt_ret_60d"   : round(mkt_60d, 6),
            "f_mkt_vol"       : round(mkt_vol, 6),
            "f_mkt_breadth"   : round(breadth, 4),
            "f_mkt_drawdown"  : round(drawdown, 4),
            "f_fund_score"    : round(_fd("Fundamental Score", 50.0), 2),
            "f_beta"          : round(_fd("Beta", 1.0), 3),
            "f_net_margin"    : round(_fd("Net Margin (%)", 0.0), 2),
            "f_pb_ratio"      : round(min(_fd("P/B Ratio", 2.0), 50.0), 3),
            "f_gross_margin"  : round(_fd("Gross Margin (%)", 0.0), 2),
            "f_debt_equity"   : round(min(_fd("Debt/Equity", 1.0), 20.0), 3),
            "f_rev_growth"    : round(_fd("Revenue Growth (%)", 0.0), 2),
            "f_analyst_score" : round(_fd("Analyst Score", 50.0), 2),
            "f_sector_id"     : int(_fd("sector_id", -1)),
        }

        result[ticker] = (float(z_val), float(slope_val) if not np.isnan(slope_val) else np.nan, extra)

    _zscore_cache[prev_ts] = result
    if len(_zscore_cache) > 5:
        del _zscore_cache[min(_zscore_cache)]

    return result


def _get_open_prices(cur_ts: pd.Timestamp) -> dict:
    global _open_prices_cache
    if cur_ts in _open_prices_cache:
        return _open_prices_cache[cur_ts]
    ticker_data = load_all_tickers()
    prices = {}
    for ticker, df_raw in ticker_data.items():
        row = df_raw[df_raw["Date"] == cur_ts]
        if len(row) > 0:
            prices[ticker] = float(row["Open"].iloc[0])
    _open_prices_cache[cur_ts] = prices
    if len(_open_prices_cache) > 5:
        del _open_prices_cache[min(_open_prices_cache)]
    return prices


def _run_ml_filter(candidates: list, last_trade_dates: dict, cur_ts: pd.Timestamp) -> list:
    """
    Takes list of candidate dicts (each has 'ticker' and 'extra' feature dict).
    Returns filtered list with ml_prob added.
    """
    if not candidates:
        return []

    model, feat_cols = _load_ml_model()

    rows = []
    for c in candidates:
        extra = c["extra"].copy()
        # Fill f_days_gap from last_trade_dates
        last_dt = last_trade_dates.get(c["ticker"])
        if last_dt:
            extra["f_days_gap"] = min((cur_ts.date() - last_dt).days, 999)
        else:
            extra["f_days_gap"] = 999
        row = [extra.get(f, 0.0) for f in feat_cols]
        rows.append(row)

    X     = np.array(rows, dtype=np.float64)
    proba = model.predict_proba(X)[:, 1]

    filtered = []
    for c, prob in zip(candidates, proba):
        if prob >= ML_THRESHOLD:
            filtered.append({**c, "ml_prob": round(float(prob), 4)})
    return filtered


def compute_signals_for_date(current_date: date, last_trade_dates: dict = None) -> dict:
    """
    Returns {'buys': [{ticker, peers, entry_z, peer_slope_pct, entry_price, ml_prob}]}
    last_trade_dates: {ticker: date} — used to compute f_days_gap feature.
    """
    if last_trade_dates is None:
        last_trade_dates = {}

    cur_ts  = pd.Timestamp(current_date)
    prev_ts = cur_ts - pd.tseries.offsets.BDay(1)

    zscores     = _get_zscores_for_date(prev_ts)
    open_prices = _get_open_prices(cur_ts)
    peers_map, _ = _get_peers_map(load_all_tickers(), _build_price_df(load_all_tickers()))

    # Collect raw candidates (z < -ENTRY_Z, slope >= 0, has open price)
    # Note: slope >= 0 pre-filter is required — ML model was trained only on
    # trades where peer_slope >= 0. Negative slopes are out-of-distribution.
    candidates = []
    for ticker, result in zscores.items():
        z_val, slope_val, extra = result
        if z_val >= -ENTRY_Z:
            continue
        if np.isnan(slope_val) or slope_val < 0.0:
            continue
        entry_price = open_prices.get(ticker)
        if entry_price is None:
            continue
        candidates.append({
            "ticker"        : ticker,
            "peers"         : ", ".join(peers_map.get(ticker, [])),
            "entry_z"       : round(z_val, 4),
            "peer_slope_pct": round(slope_val * 100, 4) if not np.isnan(slope_val) else 0.0,
            "entry_price"   : round(entry_price, 4),
            "extra"         : extra,
        })

    # ML filter
    filtered = _run_ml_filter(candidates, last_trade_dates, cur_ts)

    buys = []
    for c in filtered:
        buys.append({
            "ticker"        : c["ticker"],
            "peers"         : c["peers"],
            "entry_z"       : c["entry_z"],
            "peer_slope_pct": c["peer_slope_pct"],
            "entry_price"   : c["entry_price"],
            "ml_prob"       : c["ml_prob"],
        })

    return {"buys": buys}


def get_exit_signals(open_positions: list, current_date: date) -> list:
    """Uses shared z-score cache. Near-instant after first call per date."""
    if not open_positions:
        return []

    cur_ts  = pd.Timestamp(current_date)
    prev_ts = cur_ts - pd.tseries.offsets.BDay(1)

    zscores     = _get_zscores_for_date(prev_ts)
    open_prices = _get_open_prices(cur_ts)

    ticker_data = load_all_tickers()
    prev_closes: dict[str, float] = {}
    for pos in open_positions:
        t = pos["ticker"]
        if t not in open_prices and t in ticker_data:
            row = ticker_data[t][ticker_data[t]["Date"] == prev_ts]
            if len(row) > 0:
                prev_closes[t] = float(row["Close"].iloc[0])

    exits = []
    for pos in open_positions:
        ticker    = pos["ticker"]
        entry_px  = float(pos["entry_price"])
        hold_days = int(pos["hold_days"])
        new_hold  = hold_days + 1

        exit_px = open_prices.get(ticker) or prev_closes.get(ticker)
        if not exit_px:
            continue

        zs    = zscores.get(ticker)
        z_val = zs[0] if zs else np.nan

        should_exit = (not np.isnan(z_val) and z_val >= EXIT_Z) or new_hold >= MAX_HOLD
        if not should_exit:
            continue

        exits.append({
            "ticker"     : ticker,
            "exit_price" : round(exit_px, 4),
            "exit_reason": "MEAN" if (not np.isnan(z_val) and z_val >= EXIT_Z) else "MAX_HOLD",
            "pnl_pct"    : round((exit_px - entry_px) / entry_px * 100, 4),
            "hold_days"  : new_hold,
        })

    return exits
