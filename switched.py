"""
switched.py — regime-switched portfolio backtest
================================================

Runs whichever strategy the regime calls for on each date:

    trending -> breakout       (features.py)
    range    -> mean reversion (strategy_mr.py)
    bear     -> no trades

The comparison that matters is not "is the switched system profitable" in
isolation, but whether it beats three alternatives on the SAME dates:

    1. breakout always      (the v3 system, which measured -0.407R)
    2. mean reversion always
    3. random from whichever filtered set the regime selected

If switching does not beat running one strategy continuously, the regime
classifier is adding complexity without adding information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import simulate_trade, summarise, COSTS, BREAKOUT_PLAN
from strategy_mr import mr_plan


def run_switched(bo_frames: dict, mr_frames: dict, px_data: dict,
                 regime_df: pd.DataFrame, start=None, end=None,
                 top_n: int = 5, min_score: float = 0.0,
                 max_concurrent: int = 5, costs=None,
                 force_strategy: str | None = None,
                 selection: str = "score", seed: int = 12345) -> pd.DataFrame:
    """
    force_strategy : None follows the regime. 'breakout' or 'mean_reversion'
                     ignores the regime and always runs that strategy — this
                     is how the comparison arms are produced.
    """
    costs = costs or COSTS
    rng = np.random.default_rng(seed)

    dates = regime_df.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    pos = {s: {d: i for i, d in enumerate(df.index)} for s, df in px_data.items()}
    trades = []
    open_until: dict[str, pd.Timestamp] = {}

    for d in dates:
        strat = force_strategy or regime_df.at[d, "strategy"]
        if strat == "cash" or strat is None:
            continue
        stop_mult = float(regime_df.at[d, "atr_stop_mult"])
        if not np.isfinite(stop_mult):
            continue

        if strat == "breakout":
            frames, plan, trail_col = bo_frames, BREAKOUT_PLAN, "ema9"
        else:
            # rebuilt per call so sensitivity sweeps actually take effect
            frames, plan, trail_col = mr_frames, mr_plan(), "ema20"

        cands = []
        for sym, fr in frames.items():
            if d not in fr.index:
                continue
            row = fr.loc[d]
            if not bool(row["passes"]) or row["score"] < min_score:
                continue
            if not np.isfinite(row["atr"]) or not np.isfinite(row["entry_trigger"]):
                continue
            if sym in open_until and open_until[sym] >= d:
                continue
            cands.append((sym, float(row["score"]), row))
        if not cands:
            continue

        if selection == "random":
            chosen = [cands[i] for i in rng.permutation(len(cands))[:top_n]]
        else:
            cands.sort(key=lambda x: -x[1])
            chosen = cands[:top_n]

        live = sum(1 for s, u in open_until.items() if u >= d)
        chosen = chosen[:max(0, max_concurrent - live)]

        for sym, score, row in chosen:
            si = pos[sym].get(d)
            if si is None:
                continue
            floor = None
            if strat == "mean_reversion" and "mr_swing_low" in row:
                sl = row["mr_swing_low"]
                floor = float(sl) if np.isfinite(sl) else None
            tr = simulate_trade(px_data[sym], frames[sym][trail_col], si,
                                float(row["entry_trigger"]), float(row["atr"]),
                                stop_mult, costs, plan=plan, stop_floor=floor)
            if tr is None:
                continue
            tr.update({"symbol": sym, "signal_date": d, "score": score,
                       "regime": regime_df.at[d, "state"], "stop_mult": stop_mult})
            trades.append(tr)
            open_until[sym] = tr["exit_date"]

    return pd.DataFrame(trades)


def compare_arms(bo_frames, mr_frames, px, reg, start=None, end=None,
                 top_n=5, label="") -> pd.DataFrame:
    """The switched system against the alternatives it must beat."""
    arms = {
        "switched (regime-aware)": dict(force_strategy=None),
        "breakout always (v3)": dict(force_strategy="breakout"),
        "mean-reversion always": dict(force_strategy="mean_reversion"),
        "switched, random picks": dict(force_strategy=None, selection="random"),
    }
    rows = []
    for name, kw in arms.items():
        tr = run_switched(bo_frames, mr_frames, px, reg, start=start, end=end,
                          top_n=top_n, **kw)
        s = summarise(tr, f"{label} {name}".strip())
        rows.append(s)
    return pd.DataFrame(rows)


def per_strategy_split(trades: pd.DataFrame) -> pd.DataFrame:
    """Which half of the switched system is carrying it (or dragging it)."""
    if trades is None or trades.empty or "strategy" not in trades.columns:
        return pd.DataFrame()
    return pd.DataFrame([summarise(g, k) for k, g in trades.groupby("strategy")])
