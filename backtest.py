"""
backtest.py — event-driven backtest with realistic execution
============================================================

Design principles, in order of importance:

1. NO LOOKAHEAD. A signal is generated from the close of day T using only
   data through T. Execution happens on T+1 or later. The scoring code is
   imported from features.py — the same module the live screener uses — so
   the backtest cannot drift from what you actually trade.

2. WHEN IN DOUBT, ASSUME THE WORSE OUTCOME. Daily bars cannot tell you
   whether the high or the low came first. If a bar spans both the stop and
   the target, this engine books the stop. Real fills are worse than
   optimistic backtests, not better.

3. COSTS ARE NOT OPTIONAL. Indian delivery trades pay STT both sides, stamp
   duty, exchange and SEBI charges, GST on brokerage, plus slippage on a
   stop-triggered entry. Default assumption is 0.30% round trip in charges
   plus 0.30% round trip in slippage. Set them to zero and every number in
   the output improves; that does not make it true.

4. RESULTS ARE REPORTED IN R-MULTIPLES. R = (exit - entry) / (entry - stop).
   One R is one unit of the risk you chose to take. This normalises across
   ₹40 stocks and ₹40,000 stocks and across volatility regimes, and it makes
   the results independent of position-sizing assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import PARAMS, build_stock_frame

# --- execution assumptions -------------------------------------------------
COSTS = {
    "charge_pct_per_side": 0.15,    # STT + stamp + exchange + SEBI + GST'd brokerage
    "slippage_pct_per_side": 0.15,  # stop-triggered entries fill worse than limit
}

EXEC = {
    "order_valid_days": 1,
    "max_gap_over_trigger": 2.0,
    "max_holding_days": 15,
}

# Trade-management plans. Breakouts and mean reversion need different exits:
# a breakout is trying to ride a trend, so it books half and trails the rest.
# Mean reversion is trying to catch a snap back to the mean, so it takes the
# whole position off at a closer target and does not hang around — a 2xATR
# target and a 9 EMA trail would hand back most of a reversion gain.
BREAKOUT_PLAN = {
    "name": "breakout",
    "targets": [(2.0, 0.50)],       # (ATR multiple, fraction booked)
    "trail": "ema9",
    "max_holding_days": 15,
    "order_valid_days": 1,
    "max_gap_over_trigger": 2.0,
}


def simulate_trade(px, ema_series, signal_idx, trigger, atr_val, stop_mult,
                   costs=None, ex=None, plan=None, stop_floor=None):
    """
    Walk one candidate forward bar by bar from the signal.

    plan       : trade-management spec (BREAKOUT_PLAN or strategy_mr.MR_PLAN)
    stop_floor : optional absolute stop level (e.g. a swing low). When given,
                 the TIGHTER of it and the ATR stop is used.

    Returns a trade dict, or None if the order never filled.
    """
    costs = costs or COSTS
    plan = plan or BREAKOUT_PLAN
    cfg = dict(EXEC)
    cfg.update({k: v for k, v in plan.items()
                if k in ("order_valid_days", "max_gap_over_trigger",
                         "max_holding_days")})
    if ex:
        cfg.update(ex)
    n = len(px)

    o = px["Open"].to_numpy(); h = px["High"].to_numpy()
    l = px["Low"].to_numpy();  c = px["Close"].to_numpy()
    e_trail = ema_series.to_numpy()

    # ---------------- entry ----------------
    entry_i = entry_px = None
    for k in range(1, cfg["order_valid_days"] + 1):
        i = signal_idx + k
        if i >= n:
            return None
        if o[i] >= trigger:
            # gapped through the trigger: you fill at the open, not the trigger
            if o[i] > trigger * (1 + cfg["max_gap_over_trigger"] / 100):
                return None          # gapped too far — skip, do not chase
            entry_i, entry_px = i, o[i]
            break
        if h[i] >= trigger:
            entry_i, entry_px = i, trigger
            break
    if entry_i is None:
        return None

    slip = costs["slippage_pct_per_side"] / 100
    chg = costs["charge_pct_per_side"] / 100
    entry_cost = entry_px * (1 + slip) * (1 + chg)

    stop = entry_px - stop_mult * atr_val
    if stop_floor is not None and np.isfinite(stop_floor):
        stop = max(stop, float(stop_floor))      # the tighter of the two
    stop_initial = stop
    targets = [(entry_px + m * atr_val, frac) for m, frac in plan["targets"]]
    risk = entry_px - stop
    if risk <= 0:
        return None

    # ---------------- management ----------------
    frac_open, booked = 1.0, 0.0
    hit_targets, t1_hit = 0, False
    exit_i = exit_reason = None

    for i in range(entry_i, min(n, entry_i + cfg["max_holding_days"] + 1)):
        day_low, day_high, day_close, day_open = l[i], h[i], c[i], o[i]

        gap_stop = day_open <= stop and i > entry_i
        hit_stop = day_low <= stop or gap_stop

        # Conservative tie-break: a bar covering both stop and target is
        # booked as a stop. Daily data cannot resolve the sequence.
        if hit_stop:
            px_out = min(day_open, stop) if gap_stop else stop
            booked += frac_open * px_out * (1 - chg) * (1 - slip)
            frac_open = 0.0
            exit_i, exit_reason = i, ("stop_after_t1" if t1_hit else "stop")
            break

        if hit_targets < len(targets):
            tgt_px, tgt_frac = targets[hit_targets]
            if day_high >= tgt_px:
                take = min(tgt_frac, frac_open)
                booked += take * tgt_px * (1 - chg) * (1 - slip)
                frac_open -= take
                hit_targets += 1
                t1_hit = True
                if frac_open <= 1e-9:
                    exit_i, exit_reason = i, "target"
                    break
                stop = max(stop, entry_px)      # remainder to breakeven
                continue

        if t1_hit and plan.get("trail") and day_close < e_trail[i]:
            booked += frac_open * day_close * (1 - chg) * (1 - slip)
            frac_open = 0.0
            exit_i, exit_reason = i, "trail_9ema"
            break

    if frac_open > 0:                       # time stop
        exit_i = min(n - 1, entry_i + cfg["max_holding_days"])
        booked += frac_open * c[exit_i] * (1 - chg) * (1 - slip)
        exit_reason = exit_reason or "time_stop"
        frac_open = 0.0

    pnl_per_unit = booked - entry_cost
    return {
        "signal_i": signal_idx, "entry_i": entry_i, "exit_i": exit_i,
        "entry_date": px.index[entry_i], "exit_date": px.index[exit_i],
        "entry_px": round(float(entry_px), 2),
        "stop_px": round(float(stop_initial), 2),
        "t1_px": round(float(targets[0][0]), 2),
        "risk_per_unit": float(risk),
        "pnl_pct": float(pnl_per_unit / entry_cost * 100),
        "r_multiple": float(pnl_per_unit / risk),
        "t1_hit": bool(t1_hit), "exit_reason": exit_reason,
        "holding_days": int(exit_i - entry_i),
        "strategy": plan["name"],
    }


# ===========================================================================
# Portfolio-level backtest
# ===========================================================================
def run_backtest(frames: dict[str, pd.DataFrame], px_data: dict[str, pd.DataFrame],
                 regime_df: pd.DataFrame, start=None, end=None,
                 top_n: int = 5, min_score: float = 0.0,
                 max_concurrent: int = 5, costs=None, ex=None,
                 selection: str = "score") -> pd.DataFrame:
    """
    frames    : symbol -> per-date frame from build_stock_frame()
    px_data   : symbol -> raw OHLCV
    regime_df : per-date regime state (from regime.compute_regime)
    selection : 'score'  — rank by the model, take the top N
                'random' — take N at random from the same filtered set
                           (the baseline that tells you whether the SCORING
                            adds anything beyond the FILTERS)

    Returns one row per executed trade.
    """
    costs = costs or COSTS
    ex = ex or EXEC
    rng = np.random.default_rng(12345)

    all_dates = regime_df.index
    if start is not None:
        all_dates = all_dates[all_dates >= pd.Timestamp(start)]
    if end is not None:
        all_dates = all_dates[all_dates <= pd.Timestamp(end)]

    # position lookup so we can index forward cheaply
    pos = {s: {d: i for i, d in enumerate(df.index)} for s, df in px_data.items()}
    trades = []
    open_until: dict[str, pd.Timestamp] = {}

    for d in all_dates:
        if d not in regime_df.index:
            continue
        stop_mult = regime_df.at[d, "atr_stop_mult"]
        if not np.isfinite(stop_mult):
            continue

        # ---- candidates known as of the close of d ----
        cands = []
        for sym, fr in frames.items():
            if d not in fr.index:
                continue
            row = fr.loc[d]
            if not bool(row["passes"]):
                continue
            if row["score"] < min_score:
                continue
            if not np.isfinite(row["atr"]) or not np.isfinite(row["entry_trigger"]):
                continue
            # don't stack a second position in a name already held
            if sym in open_until and open_until[sym] >= d:
                continue
            cands.append((sym, float(row["score"]), row))

        if not cands:
            continue

        if selection == "random":
            idx = rng.permutation(len(cands))[:top_n]
            chosen = [cands[i] for i in idx]
        else:
            cands.sort(key=lambda x: -x[1])
            chosen = cands[:top_n]

        # respect a cap on simultaneous exposure
        live = sum(1 for s, u in open_until.items() if u >= d)
        room = max(0, max_concurrent - live)
        chosen = chosen[:room]

        for sym, score, row in chosen:
            px = px_data[sym]
            si = pos[sym].get(d)
            if si is None:
                continue
            tr = simulate_trade(px, frames[sym]["ema9"], si,
                                float(row["entry_trigger"]), float(row["atr"]),
                                float(stop_mult), costs, ex)
            if tr is None:
                continue
            tr["symbol"] = sym
            tr["signal_date"] = d
            tr["score"] = score
            tr["regime"] = regime_df.at[d, "state"]
            tr["stop_mult"] = stop_mult
            for comp in ["volume", "squeeze", "sector", "trend",
                         "candle", "near_high", "momentum"]:
                tr[f"c_{comp}"] = float(row[comp])
            tr["vol_ratio"] = float(row["vol_ratio"]) if np.isfinite(row["vol_ratio"]) else np.nan
            tr["atr_pct"] = float(row["atr_pct"])
            tr["dist_52w_pct"] = float(row["dist_52w_pct"]) if np.isfinite(row["dist_52w_pct"]) else np.nan
            trades.append(tr)
            open_until[sym] = tr["exit_date"]

    return pd.DataFrame(trades)


# ===========================================================================
# Statistics
# ===========================================================================
def summarise(trades: pd.DataFrame, label: str = "") -> dict:
    """Headline statistics. Expectancy in R is the number that matters."""
    if trades is None or trades.empty:
        return {"label": label, "n": 0}

    r = trades["r_multiple"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()

    equity = r.cumsum()
    dd = (equity - equity.cummax()).min()

    # Standard error on expectancy — the honest way to say whether a
    # difference between two configurations means anything at all.
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan

    return {
        "label": label,
        "n": int(len(r)),
        "hit_rate_pct": round(float((r > 0).mean() * 100), 1),
        "expectancy_R": round(float(r.mean()), 3),
        "expectancy_SE": round(float(se), 3) if np.isfinite(se) else None,
        "avg_win_R": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_R": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "payoff_ratio": round(float(wins.mean() / -losses.mean()), 2)
                        if len(wins) and len(losses) and losses.mean() != 0 else None,
        "profit_factor": round(float(gross_win / gross_loss), 2) if gross_loss > 0 else None,
        "total_R": round(float(r.sum()), 1),
        "max_dd_R": round(float(dd), 1),
        "avg_hold_days": round(float(trades["holding_days"].mean()), 1),
        "t1_hit_pct": round(float(trades["t1_hit"].mean() * 100), 1),
        "avg_pnl_pct": round(float(trades["pnl_pct"].mean()), 2),
    }


def component_attribution(trades: pd.DataFrame, min_bucket: int = 30) -> pd.DataFrame:
    """
    For each scoring component, split trades by the points that component
    awarded and report mean R per bucket.

    This is the most useful output of the whole exercise. If mean R rises with
    a component's score, that component is carrying information. If it is flat
    or inverted, the component is noise and its weight should come down —
    regardless of how sensible the reasoning behind it sounded.

    Buckets thinner than `min_bucket` trades are reported but flagged, because
    a mean R computed from nine trades tells you nothing.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()

    rows = []
    for comp in ["volume", "squeeze", "sector", "trend", "candle",
                 "near_high", "momentum"]:
        col = f"c_{comp}"
        if col not in trades.columns:
            continue
        vals = trades[col]
        # squeeze is a continuous 0-20 score, so grouping on the raw value
        # produces ~130 buckets of one trade each — noise, not signal. Bin the
        # continuous components into tiers; the discrete ones group as-is.
        if comp in ("squeeze",):
            binned = pd.cut(vals, bins=[-0.01, 0.01, 5, 10, 15, 20.01],
                            labels=["0", "0-5", "5-10", "10-15", "15-20"])
            grouper = binned
            is_binned = True
        else:
            grouper = vals
            is_binned = False
        for v, grp in trades.groupby(grouper, observed=True):
            rows.append({
                "component": comp,
                "points": str(v) if is_binned else round(float(v), 1),
                "n": len(grp),
                "mean_R": round(float(grp["r_multiple"].mean()), 3),
                "hit_rate_pct": round(float((grp["r_multiple"] > 0).mean() * 100), 1),
                "reliable": len(grp) >= min_bucket,
            })
    out = pd.DataFrame(rows)
    # sort binned components by their own tier order, others numerically
    return out.reset_index(drop=True)


def score_decile_analysis(trades: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """
    Does a higher total score actually produce a higher mean R?
    If this table is flat, the 100-point model is decoration.
    """
    if trades is None or trades.empty or len(trades) < bins * 5:
        return pd.DataFrame()
    t = trades.copy()
    try:
        t["bucket"] = pd.qcut(t["score"], bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = t.groupby("bucket", observed=True)["r_multiple"]
    return pd.DataFrame({
        "n": g.size(),
        "mean_R": g.mean().round(3),
        "hit_rate_pct": (t.groupby("bucket", observed=True)["r_multiple"]
                         .apply(lambda s: (s > 0).mean() * 100).round(1)),
    }).reset_index()


def regime_split(trades: pd.DataFrame) -> pd.DataFrame:
    """Performance by market regime — does the gate actually earn its place?"""
    if trades is None or trades.empty or "regime" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for state, grp in trades.groupby("regime"):
        s = summarise(grp, state)
        rows.append(s)
    return pd.DataFrame(rows)


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    rows = []
    for reason, grp in trades.groupby("exit_reason"):
        rows.append({
            "exit_reason": reason,
            "n": len(grp),
            "share_pct": round(len(grp) / len(trades) * 100, 1),
            "mean_R": round(float(grp["r_multiple"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
