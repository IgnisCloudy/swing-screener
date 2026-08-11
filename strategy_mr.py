"""
strategy_mr.py — RSI mean-reversion for range-bound regimes
===========================================================

Structurally the opposite trade to the breakout system, for the regime where
the breakout system bleeds. In a range, moving averages flatten and sit in
the middle of the chop; buying a pullback to a flat 20 EMA is buying the
middle of the range at poor risk-reward. Fading the extremes is the trade
that matches the structure.

THE FALLING-KNIFE PROBLEM
-------------------------
Indian mid-caps can hold RSI below 30 for weeks during a broad correction.
Buying on oversold alone is how you catch the knife. Three guards, all
mandatory, all confirmed at the close of the signal bar:

  1. REVERSAL CONFIRMATION — the stock must close back above the previous
     day's high while still oversold. This is the "someone stepped in"
     signal. Without it, oversold is just a falling stock.

  2. STRUCTURAL HEALTH — either price above the 200 EMA, OR the 200 EMA
     itself not falling more than 3% over 20 sessions.

     Note the "or". Demanding price above the 200 EMA is a TREND test, and
     in a range-bound market the 200 EMA flattens and price oscillates across
     it continuously — the same structural error as buying a pullback to a
     flat 20 EMA. Requiring it outright rejects most legitimate range setups
     (measured: it took the strategy to zero qualifying trades). What we
     actually need to exclude is a genuine downtrend, and a 200 EMA in
     sustained decline is the test for that.

  3. NOT IN FREEFALL — the 20-day return floor rejects names that have
     collapsed rather than pulled back, and the down-day streak cap rejects
     relentless one-way selling with no attempt at a bid.

Volume climax is scored rather than required: a capitulation bar is good
evidence, but demanding it as a filter would cut the sample too thin.

TRADE MANAGEMENT IS DIFFERENT
-----------------------------
Mean reversion does not trend. It takes profit into the move back toward the
mean and gets out. A 2xATR target and a 9 EMA trail — correct for breakouts —
would hand back most of a mean-reversion gain. So: closer target, full exit,
short time stop. See MR_PLAN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import ema, rsi, atr, PARAMS

MR_PARAMS = {
    # ---- hard filters (shared liquidity floor with the breakout system) ----
    "min_price": 20.0,
    "min_turnover": 5e7,
    "atr_pct_min": 1.5,
    "atr_pct_max": 8.0,
    "min_market_cap": 5e9,
    "min_history": 250,

    # ---- oversold definition ----
    "rsi_entry_max": 40.0,      # must be at or below this on the signal bar
    "rsi_deep": 25.0,           # full marks for depth here
    "rsi_shallow": 40.0,

    # ---- falling-knife guards ----
    "require_reversal": True,   # close > previous day's high
    "require_structural_health": True,
    # A 200 EMA is heavily smoothed, so even a 1.5% fall over 20 sessions
    # requires price sustained well below it. This threshold is a reasoned
    # starting point, not a measured one — it is included in the sensitivity
    # sweep so the data can argue with it.
    "max_ema200_decline_pct": -1.5,
    "min_ret_20d": -25.0,       # a pullback, not a collapse
    "max_down_streak": 6,       # relentless selling with no bid attempt
    "max_dist_below_20ema": 25.0,   # stretched, not detached

    # ---- scoring weights (100 total) ----
    "w_rsi_depth": 20.0,
    "w_reversal": 20.0,
    "w_stretch": 15.0,
    "w_trend_intact": 15.0,
    "w_support": 15.0,
    "w_vol_climax": 15.0,

    # ---- trade plan ----
    "entry_trigger_mult": 1.003,   # tighter than breakout: 0.3% above prev high
    "stop_atr_mult": 1.5,
    "stop_below_swing_low": True,  # use the lower of the two stops
    "swing_low_lookback": 5,
    "target_atr_mult": 1.5,        # closer than the breakout's 2.0
    "target_cap_at_20ema": True,   # the mean IS the target
    "max_holding_days": 8,         # mean reversion resolves fast or fails
}

MR_COMPONENT_MAX = {
    "rsi_depth": 20.0, "reversal": 20.0, "stretch": 15.0,
    "trend_intact": 15.0, "support": 15.0, "vol_climax": 15.0,
}
MR_RAW_MAX = 100.0

def mr_plan(p: dict | None = None) -> dict:
    """
    Build the trade-management spec FRESH from current parameters.

    This is a function, not a module-level constant, and that matters: a dict
    built at import time freezes whatever MR_PARAMS held at import, so a
    sensitivity sweep that mutates MR_PARAMS would silently change nothing.
    In the v4 run that produced identical results for every value of
    target_atr_mult and max_holding_days — the sweep looked like a flat
    plateau when in fact it had never varied at all.
    """
    p = p or MR_PARAMS
    return {
        "name": "mean_reversion",
        "entry_trigger_mult": p["entry_trigger_mult"],
        "stop_atr_mult": p["stop_atr_mult"],
        "targets": [(p["target_atr_mult"], 1.0)],   # full exit at target
        "trail": None,                               # no trailing
        "max_holding_days": p["max_holding_days"],
        "order_valid_days": 1,
        "max_gap_over_trigger": 1.5,
    }


# Backwards-compatible constant for callers that do not sweep.
MR_PLAN = mr_plan()


def compute_mr_features(df: pd.DataFrame, p: dict | None = None) -> pd.DataFrame:
    """Causal features for the mean-reversion setup."""
    p = p or MR_PARAMS
    f = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    f["open"], f["high"], f["low"], f["close"], f["vol_raw"] = o, h, l, c, v
    f["bars_seen"] = np.arange(1, len(df) + 1)
    f["turnover_20d"] = (c * v).rolling(20).mean()
    f["atr"] = atr(df)
    f["atr_pct"] = f["atr"] / c * 100
    f["rsi"] = rsi(c)

    f["ema20"] = ema(c, 20)
    f["ema50"] = ema(c, 50)
    f["ema200"] = ema(c, 200)
    f["sma50_slope"] = c.rolling(50).mean().diff(10)

    f["avg_vol_20"] = v.shift(1).rolling(20).mean()
    f["vol_ratio"] = v / f["avg_vol_20"]

    # --- guard inputs ---
    # reversal: closed back above yesterday's high while oversold
    f["reversal_bar"] = c > h.shift(1)
    f["above_200ema"] = c > f["ema200"]
    # 20-session change in the 200 EMA: flat-to-rising means the structure is
    # intact even when price is temporarily beneath it, which is normal in a
    # range. Sustained decline means a real downtrend — no mean reversion there.
    f["ema200_slope_pct"] = (f["ema200"] / f["ema200"].shift(20) - 1) * 100
    f["structurally_healthy"] = (
        f["above_200ema"] | (f["ema200_slope_pct"] >= p["max_ema200_decline_pct"]))
    f["ret_20d_pct"] = (c / c.shift(20) - 1) * 100

    down = (c < c.shift(1)).astype(int)
    # consecutive down closes ending today
    streak = down.groupby((down != down.shift()).cumsum()).cumsum()
    f["down_streak"] = streak.where(down == 1, 0)

    f["dist_below_20ema_pct"] = (f["ema20"] - c) / f["ema20"] * 100

    # --- support structure ---
    f["low_20d"] = l.shift(1).rolling(20).min()
    f["dist_above_low20_pct"] = (c - f["low_20d"]) / f["low_20d"] * 100
    f["swing_low"] = l.rolling(p["swing_low_lookback"]).min()

    # --- candle geometry ---
    rng = (h - l).replace(0, np.nan)
    f["close_pos"] = (c - l) / rng
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    f["upper_wick_frac"] = (h - body_top) / rng
    f["is_circuit_locked"] = (h - c).abs() < (0.0002 * c)

    # entry trigger for tomorrow
    f["entry_trigger"] = h * p["entry_trigger_mult"]
    f["trigger_gap_pct"] = (f["entry_trigger"] - c) / c * 100

    return f


def compute_mr_filters(f: pd.DataFrame, market_cap_series=None,
                       p: dict | None = None) -> pd.DataFrame:
    p = p or MR_PARAMS
    out = pd.DataFrame(index=f.index)

    out["f_price"] = f["close"] >= p["min_price"]
    out["f_turnover"] = f["turnover_20d"] >= p["min_turnover"]
    out["f_atr"] = (f["atr_pct"] >= p["atr_pct_min"]) & (f["atr_pct"] <= p["atr_pct_max"])
    out["f_history"] = f["bars_seen"] >= p["min_history"]
    out["f_not_circuit"] = ~f["is_circuit_locked"].fillna(False)

    # --- the oversold condition itself ---
    out["f_oversold"] = f["rsi"] <= p["rsi_entry_max"]

    # --- guard 1: reversal confirmation ---
    out["f_reversal"] = (f["reversal_bar"].fillna(False)
                         if p["require_reversal"] else True)

    # --- guard 2: structural health (not a genuine downtrend) ---
    out["f_uptrend"] = (f["structurally_healthy"].fillna(False)
                        if p["require_structural_health"] else True)

    # --- guard 3: pullback, not collapse ---
    out["f_not_freefall"] = (
        (f["ret_20d_pct"] >= p["min_ret_20d"])
        & (f["down_streak"] <= p["max_down_streak"])
        & (f["dist_below_20ema_pct"] <= p["max_dist_below_20ema"])
    ).fillna(False)

    if market_cap_series is not None:
        out["f_mcap"] = market_cap_series.reindex(f.index).ffill() >= p["min_market_cap"]
    else:
        out["f_mcap"] = True

    out["passes"] = out[[c for c in out.columns if c.startswith("f_")]].all(axis=1)
    return out


def compute_mr_scores(f: pd.DataFrame, sector_score=None,
                      p: dict | None = None) -> pd.DataFrame:
    p = p or MR_PARAMS
    s = pd.DataFrame(index=f.index)

    # --- 1. oversold depth (20) — deeper is better, to a point ---
    depth = np.interp(f["rsi"].fillna(100),
                      [p["rsi_deep"], p["rsi_shallow"]],
                      [p["w_rsi_depth"], p["w_rsi_depth"] * 0.4])
    depth = pd.Series(depth, index=f.index)
    # below the "deep" line we stop rewarding: that is distress, not a dip
    depth = depth.mask(f["rsi"] < 15.0, p["w_rsi_depth"] * 0.5)
    s["rsi_depth"] = depth.where(f["rsi"] <= p["rsi_shallow"], 0.0).fillna(0.0)

    # --- 2. reversal quality (20) ---
    rev = pd.Series(0.0, index=f.index)
    rev = rev.mask(f["reversal_bar"].fillna(False), p["w_reversal"] * 0.6)
    strong_close = f["close_pos"] >= 0.7
    rev = rev + (f["reversal_bar"].fillna(False) & strong_close.fillna(False)
                 ).astype(float) * (p["w_reversal"] * 0.4)
    # an upper wick on the reversal bar means sellers were still there
    rev = rev.mask(f["upper_wick_frac"].fillna(0) > 0.5, p["w_reversal"] * 0.3)
    s["reversal"] = rev.clip(upper=p["w_reversal"])

    # --- 3. stretch from the mean (15) — the reversion fuel ---
    d = f["dist_below_20ema_pct"]
    stretch = pd.Series(0.0, index=f.index)
    stretch = stretch.mask(d >= 2.0, p["w_stretch"] * 0.4)
    stretch = stretch.mask(d >= 5.0, p["w_stretch"] * 0.75)
    stretch = stretch.mask(d >= 8.0, p["w_stretch"])
    # too far is detachment, not stretch
    stretch = stretch.mask(d > p["max_dist_below_20ema"], 0.0)
    s["stretch"] = stretch.fillna(0.0)

    # --- 4. trend intact (15) ---
    ti = pd.Series(0.0, index=f.index)
    ti = ti.mask(f["structurally_healthy"].fillna(False), p["w_trend_intact"] * 0.4)
    ti = ti.mask(f["above_200ema"].fillna(False), p["w_trend_intact"] * 0.6)
    ti = ti + ((f["close"] > f["ema50"]).fillna(False)).astype(float) * (
        p["w_trend_intact"] * 0.2)
    ti = ti + ((f["sma50_slope"] > 0).fillna(False)).astype(float) * (
        p["w_trend_intact"] * 0.2)
    s["trend_intact"] = ti.clip(upper=p["w_trend_intact"])

    # --- 5. support proximity (15) — bouncing off the range floor ---
    dl = f["dist_above_low20_pct"]
    sup = pd.Series(0.0, index=f.index)
    sup = sup.mask(dl <= 12.0, p["w_support"] * 0.4)
    sup = sup.mask(dl <= 6.0, p["w_support"] * 0.7)
    sup = sup.mask(dl <= 3.0, p["w_support"])
    s["support"] = sup.fillna(0.0)

    # --- 6. volume climax (15) — capitulation then the bid returns ---
    vr = f["vol_ratio"]
    vc = pd.Series(0.0, index=f.index)
    vc = vc.mask(vr >= 1.3, p["w_vol_climax"] * 0.4)
    vc = vc.mask(vr >= 1.8, p["w_vol_climax"] * 0.7)
    vc = vc.mask(vr >= 2.5, p["w_vol_climax"])
    vc = vc.mask(vr > 15.0, 0.0)     # bulk deal, not capitulation
    s["vol_climax"] = vc.fillna(0.0)

    comps = list(MR_COMPONENT_MAX)
    s["score_raw"] = s[comps].sum(axis=1)
    s["score"] = s["score_raw"] * 100.0 / MR_RAW_MAX

    # trade-plan levels, all known at the signal close
    s["mr_target_hint"] = f["ema20"]      # the mean is the natural target
    s["mr_swing_low"] = f["swing_low"]
    return s


def build_mr_frame(df: pd.DataFrame, sector_score=None, market_cap_series=None,
                   p: dict | None = None) -> pd.DataFrame:
    f = compute_mr_features(df, p)
    flt = compute_mr_filters(f, market_cap_series, p)
    sc = compute_mr_scores(f, sector_score, p)
    return pd.concat([f, flt, sc], axis=1)
