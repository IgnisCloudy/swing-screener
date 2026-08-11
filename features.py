"""
features.py — vectorised, strictly causal feature and score computation
======================================================================

THE ONE RULE
------------
Every column produced here at row t must depend only on data at or before t.
No `.max()` over a whole series, no centred windows, no `.shift(-n)`.

This is what makes the identical code safe for live screening AND for
historical backtesting. If the backtest scored stocks differently from the
live screener, it would validate nothing. So both import from here.

Causality is not assumed — test_engine.py asserts that the score computed
here at date T equals the score computed on data truncated at T, for every
component, across hundreds of random dates.

Incorporates the v3 review changes:
  1. Upper-circuit rejection regardless of volume
  2. Volume anomaly cap (reject above 15x)
  3. Rejection-wick penalty on the candle component
  4. Multi-year-high bonus on the 52-week-high component
  5. Gap-up guard on the entry trigger
  (6. market breadth and 7. earnings filter live in regime.py / events.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters — single source of truth. The backtest reads these, so changing
# a number here changes both what you trade and what you measure.
# ---------------------------------------------------------------------------
PARAMS = {
    # ---- hard filters ----
    "min_price": 20.0,
    "min_turnover": 5e7,            # ₹5 Cr, 20-day average
    "atr_pct_min": 1.5,
    "atr_pct_max": 8.0,
    "min_market_cap": 5e9,          # ₹500 Cr
    "min_history": 150,
    "circuit_tol": 0.0002,          # close within 0.02% of high => treat as locked
    "max_vol_ratio": 15.0,          # v3: reject bulk-deal volume anomalies
    "max_trigger_gap_pct": 3.0,     # v3: reject if trigger sits >3% above close

    # ---- volume component (20) ----
    "vol_strong": 2.0, "vol_mid": 1.5, "vol_weak": 1.2,
    "vol_pts_strong": 16.0, "vol_pts_mid": 11.0, "vol_pts_weak": 6.0,
    "vol_proxy": 2.5, "vol_bonus": 4.0,

    # ---- squeeze component (20) ----
    "bbw_ratio_max": 0.75, "atr_ratio_max": 0.80,
    "squeeze_recency_tol": 1.02,
    "base_lookback": 15, "base_max_range_pct": 12.0, "base_pts": 4.0,

    # ---- sector component (15) ----
    "sector_ema_pts": 8.0, "sector_rs_pts": 7.0, "sector_neutral": 7.5,

    # ---- trend component (15) ----
    "trend_full": 15.0, "trend_partial": 9.0, "trend_weak": 4.0,

    # ---- candle component (10) ----
    "candle_top": 0.80, "candle_mid": 0.65,
    "candle_pts_top": 10.0, "candle_pts_mid": 5.0,
    "max_upper_wick_frac": 0.50,    # v3: wick > 50% of range => 0

    # ---- high-proximity component (10 + 2 bonus) ----
    "hi_band_lo": 2.0, "hi_band_hi": 5.0,
    "hi_pts_band": 10.0, "hi_pts_extended": 7.0, "hi_pts_near": 5.0,
    "ath_bonus": 2.0,               # v3: multi-year high bonus
    "ath_min_history": 750,         # ~3y before we call it a multi-year high

    # ---- momentum component (10) ----
    "rsi_lo": 55.0, "rsi_hi": 72.0,
    "rsi_pts_full": 10.0, "rsi_pts_partial": 5.0,

    # ---- trade plan ----
    "entry_trigger_mult": 1.005,
    "atr_stop_normal": 1.5,
    "atr_stop_caution": 1.25,
    "atr_stop_strict": 1.0,
    "atr_t1": 2.0,
}

# Raw maximum is 102 (100 + the 2-point multi-year-high bonus). Scores are
# normalised back to 100 for display. Normalisation is monotonic, so it has
# no effect on ranking — it only preserves the "out of 100" framing.
RAW_MAX = 102.0

COMPONENT_MAX = {
    "volume": 20.0, "squeeze": 20.0, "sector": 15.0, "trend": 15.0,
    "candle": 10.0, "near_high": 12.0, "momentum": 10.0,
}


# ===========================================================================
# Causal indicator primitives
# ===========================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    """Exponentially weighted mean. adjust=False makes it strictly causal."""
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bb_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (2 * k * sd) / mid * 100


# ===========================================================================
# Feature frame
# ===========================================================================
def compute_features(df: pd.DataFrame, p: dict | None = None) -> pd.DataFrame:
    """
    Build every derived column needed for filtering and scoring.
    Input: OHLCV indexed by date, ascending.
    Output: same index, one column per feature. All causal.
    """
    p = p or PARAMS
    f = pd.DataFrame(index=df.index)

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    # NB: raw volume is `vol_raw`, not `volume` — `volume` is the name of the
    # score component, and a duplicate column silently turns every lookup into
    # a Series instead of a scalar.
    f["open"], f["high"], f["low"], f["close"], f["vol_raw"] = o, h, l, c, v

    # --- liquidity / volatility ---
    f["turnover_20d"] = (c * v).rolling(20).mean()
    f["atr"] = atr(df)
    f["atr_pct"] = f["atr"] / c * 100

    # prior-20-session average volume, excluding today (shift(1) is what
    # keeps today's own volume out of its own benchmark)
    f["avg_vol_20"] = v.shift(1).rolling(20).mean()
    f["vol_ratio"] = v / f["avg_vol_20"]

    # --- moving averages ---
    f["ema9"] = ema(c, 9)
    f["ema20"] = ema(c, 20)
    f["ema50"] = ema(c, 50)
    f["ema200"] = ema(c, 200)
    f["rsi"] = rsi(c)

    # --- highs ---
    f["hi_52w"] = h.rolling(252, min_periods=60).max()
    f["dist_52w_pct"] = (f["hi_52w"] - c) / f["hi_52w"] * 100
    # expanding max = highest price in all history seen SO FAR (causal)
    f["hi_alltime"] = h.expanding().max()
    f["bars_seen"] = np.arange(1, len(df) + 1)
    # "At a multi-year high" means today PRINTED the highest high in all
    # available history. Comparing the close to the running max of highs
    # instead would almost never fire, because a day's close is normally a
    # little below its own high.
    f["at_multiyear_high"] = (
        (h >= f["hi_alltime"] * 0.999) & (f["bars_seen"] >= p["ath_min_history"])
    )

    # --- candle geometry ---
    rng = (h - l).replace(0, np.nan)
    f["close_pos"] = (c - l) / rng
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    f["upper_wick_frac"] = (h - body_top) / rng          # v3
    f["is_circuit_locked"] = (h - c).abs() < (p["circuit_tol"] * c)   # v3

    # --- volatility squeeze / VCP ---
    bbw = bb_width(c)
    f["bbw"] = bbw
    bbw_prior = bbw.shift(1)
    f["bbw_min10"] = bbw_prior.rolling(10).min()
    f["bbw_min20"] = bbw_prior.rolling(20).min()
    f["bbw_base"] = bbw.rolling(126, min_periods=60).median()
    f["bbw_compression"] = f["bbw_min10"] / f["bbw_base"]

    atrp_prior = f["atr_pct"].shift(1)
    f["atrp_min10"] = atrp_prior.rolling(10).min()
    f["atrp_min20"] = atrp_prior.rolling(20).min()
    f["atrp_base"] = f["atr_pct"].rolling(126, min_periods=60).median()
    f["atrp_compression"] = f["atrp_min10"] / f["atrp_base"]

    # --- consolidation base ---
    n = p["base_lookback"]
    base_hi = h.shift(1).rolling(n).max()
    base_lo = l.shift(1).rolling(n).min()
    f["base_hi"], f["base_lo"] = base_hi, base_lo
    f["base_range_pct"] = (base_hi - base_lo) / base_lo * 100
    f["breakout"] = (c > base_hi) & (f["base_range_pct"] < p["base_max_range_pct"])

    # --- returns ---
    f["ret_5d_pct"] = (c / c.shift(5) - 1) * 100
    f["ret_20d_pct"] = (c / c.shift(20) - 1) * 100

    # --- trade plan levels (known at the close of the signal bar) ---
    f["entry_trigger"] = h * p["entry_trigger_mult"]
    f["trigger_gap_pct"] = (f["entry_trigger"] - c) / c * 100

    return f


# ===========================================================================
# Hard filters
# ===========================================================================
def compute_filters(f: pd.DataFrame, market_cap_series: pd.Series | None = None,
                    p: dict | None = None) -> pd.DataFrame:
    """
    One boolean column per filter, plus `passes` = all of them.
    Keeping them separate lets the backtest report which filter rejected what.
    """
    p = p or PARAMS
    out = pd.DataFrame(index=f.index)

    out["f_price"] = f["close"] >= p["min_price"]
    out["f_turnover"] = f["turnover_20d"] >= p["min_turnover"]
    out["f_atr"] = (f["atr_pct"] >= p["atr_pct_min"]) & (f["atr_pct"] <= p["atr_pct_max"])
    out["f_history"] = f["bars_seen"] >= p["min_history"]

    # v3: any close locked at the high is rejected, whatever the volume.
    # A lock means unfilled demand carries into tomorrow's open, which gaps
    # the price past the entry trigger and destroys the ATR-based stop maths.
    out["f_not_circuit"] = ~f["is_circuit_locked"].fillna(False)

    # v3: absurd volume is a bulk deal or corporate action, not accumulation
    out["f_vol_sane"] = (f["vol_ratio"] <= p["max_vol_ratio"]) | f["vol_ratio"].isna()

    # v3: if the trigger already sits far above the close, the move has happened
    out["f_trigger_near"] = f["trigger_gap_pct"] <= p["max_trigger_gap_pct"]

    if market_cap_series is not None:
        out["f_mcap"] = market_cap_series.reindex(f.index).ffill() >= p["min_market_cap"]
    else:
        out["f_mcap"] = True

    out["passes"] = out[[c for c in out.columns if c.startswith("f_")]].all(axis=1)
    return out


# ===========================================================================
# Scoring
# ===========================================================================
def compute_scores(f: pd.DataFrame, sector_score: pd.Series | None = None,
                   delivery_pct: pd.Series | None = None,
                   p: dict | None = None) -> pd.DataFrame:
    """
    Vectorised component scores. Returns one column per component plus
    `score_raw` (out of 102) and `score` (normalised to 100).

    sector_score : per-date 0-15 series for this stock's sector. If None,
                   the neutral value is used — which is what happens live
                   when a sector index fails to load.
    delivery_pct : per-date NSE delivery percentage, or None. Backtests
                   almost always pass None (see METHODOLOGY.md).
    """
    p = p or PARAMS
    s = pd.DataFrame(index=f.index)

    # ---------------- 1. Volume surge (20) ----------------
    vr = f["vol_ratio"]
    vol = pd.Series(0.0, index=f.index)
    vol = vol.mask(vr >= p["vol_weak"], p["vol_pts_weak"])
    vol = vol.mask(vr >= p["vol_mid"], p["vol_pts_mid"])
    vol = vol.mask(vr >= p["vol_strong"], p["vol_pts_strong"])
    # anomalies score zero rather than maximum (they are also hard-filtered,
    # but scoring them zero keeps the two mechanisms independent)
    vol = vol.mask(vr > p["max_vol_ratio"], 0.0)

    if delivery_pct is not None:
        dly = delivery_pct.reindex(f.index)
        bonus_ok = (dly > 50) | (dly.isna() & (vr >= p["vol_proxy"]))
    else:
        bonus_ok = vr >= p["vol_proxy"]
    bonus_ok = bonus_ok & (vr <= p["max_vol_ratio"])
    vol = vol + bonus_ok.astype(float) * p["vol_bonus"]
    s["volume"] = vol.clip(upper=COMPONENT_MAX["volume"]).fillna(0.0)

    # ---------------- 2. Volatility squeeze / VCP (20) ----------------
    sq_bbw = (f["bbw_min10"] <= f["bbw_min20"] * p["squeeze_recency_tol"]) & \
             (f["bbw_compression"] <= p["bbw_ratio_max"])
    bbw_pts = pd.Series(
        np.interp(f["bbw_compression"].fillna(9), [0.45, p["bbw_ratio_max"]], [11.0, 7.0]),
        index=f.index)
    bbw_pts = bbw_pts.where(sq_bbw.fillna(False), 0.0)

    sq_atr = (f["atrp_min10"] <= f["atrp_min20"] * p["squeeze_recency_tol"]) & \
             (f["atrp_compression"] <= p["atr_ratio_max"])
    atr_pts = pd.Series(
        np.interp(f["atrp_compression"].fillna(9), [0.50, p["atr_ratio_max"]], [5.0, 3.0]),
        index=f.index)
    atr_pts = atr_pts.where(sq_atr.fillna(False), 0.0)

    brk_pts = f["breakout"].fillna(False).astype(float) * p["base_pts"]
    s["squeeze"] = (bbw_pts + atr_pts + brk_pts).clip(upper=COMPONENT_MAX["squeeze"])
    s["_squeeze_flag"] = (sq_bbw | sq_atr).fillna(False)

    # ---------------- 3. Sector relative strength (15) ----------------
    if sector_score is not None:
        s["sector"] = sector_score.reindex(f.index).fillna(p["sector_neutral"])
    else:
        s["sector"] = p["sector_neutral"]

    # ---------------- 4. Trend alignment (15) ----------------
    c = f["close"]
    perfect = (c > f["ema20"]) & (f["ema20"] > f["ema50"]) & (f["ema50"] > f["ema200"])
    partial = (c > f["ema20"]) & (f["ema20"] > f["ema50"])
    weak = c > f["ema20"]
    trend = pd.Series(0.0, index=f.index)
    trend = trend.mask(weak, p["trend_weak"])
    trend = trend.mask(partial, p["trend_partial"])
    trend = trend.mask(perfect, p["trend_full"])
    s["trend"] = trend.fillna(0.0)
    s["_perfect_trend"] = perfect.fillna(False)

    # ---------------- 5. Candle strength (10), wick-penalised ----------------
    cp = f["close_pos"]
    cand = pd.Series(0.0, index=f.index)
    cand = cand.mask(cp >= p["candle_mid"], p["candle_pts_mid"])
    cand = cand.mask(cp >= p["candle_top"], p["candle_pts_top"])
    # v3: a long upper wick is distribution — selling into the rally — however
    # strong the close looks on a close-position basis.
    rejection = f["upper_wick_frac"] > p["max_upper_wick_frac"]
    cand = cand.mask(rejection.fillna(False), 0.0)
    s["candle"] = cand.fillna(0.0)
    s["_rejection_wick"] = rejection.fillna(False)

    # ---------------- 6. High proximity (10) + multi-year bonus (2) --------
    d = f["dist_52w_pct"]
    nh = pd.Series(0.0, index=f.index)
    nh = nh.mask(d <= 10.0, p["hi_pts_near"])
    nh = nh.mask((d >= p["hi_band_lo"]) & (d <= p["hi_band_hi"]), p["hi_pts_band"])
    nh = nh.mask(d < p["hi_band_lo"], p["hi_pts_extended"])
    nh = nh + f["at_multiyear_high"].fillna(False).astype(float) * p["ath_bonus"]
    s["near_high"] = nh.clip(upper=COMPONENT_MAX["near_high"]).fillna(0.0)

    # ---------------- 7. Momentum (10) ----------------
    r = f["rsi"]
    mom = pd.Series(0.0, index=f.index)
    mom = mom.mask(((r >= 50) & (r < p["rsi_lo"])) | ((r > p["rsi_hi"]) & (r <= 78)),
                   p["rsi_pts_partial"])
    mom = mom.mask((r >= p["rsi_lo"]) & (r <= p["rsi_hi"]), p["rsi_pts_full"])
    s["momentum"] = mom.fillna(0.0)

    comps = ["volume", "squeeze", "sector", "trend", "candle", "near_high", "momentum"]
    s["score_raw"] = s[comps].sum(axis=1)
    s["score"] = s["score_raw"] * 100.0 / RAW_MAX
    return s


def build_stock_frame(df: pd.DataFrame, sector_score=None, market_cap_series=None,
                      delivery_pct=None, p: dict | None = None) -> pd.DataFrame:
    """features + filters + scores in one frame, one row per date."""
    f = compute_features(df, p)
    flt = compute_filters(f, market_cap_series, p)
    sc = compute_scores(f, sector_score, delivery_pct, p)
    return pd.concat([f, flt, sc], axis=1)
