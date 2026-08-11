"""
regime.py — market regime gate and sector relative strength
===========================================================

v3 change: the regime is no longer decided by the Nifty 50 alone.

The Nifty 50 is capitalisation-weighted and a handful of mega-caps can hold
it above its 20 EMA while the mid- and small-cap names that most swing setups
come from are already rolling over. So the gate now combines the index with
market breadth — the share of the liquid universe trading above its own
50 EMA — which is a direct measurement of the market the trades actually
live in.

Both series are causal: breadth on date t counts only stocks whose price on
date t is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import PARAMS, ema

# v4 regime thresholds. Three states, each with its OWN strategy rather than
# merely its own stop multiple — that was the v3 mistake. A tighter stop does
# not fix a strategy that is structurally wrong for the market it is in.
BREADTH_TRENDING = 55.0
BREADTH_BEAR = 35.0

REGIME_STRATEGY = {
    "trending": "breakout",       # momentum works when there is momentum
    "range": "mean_reversion",    # fade the extremes of the channel
    "bear": "cash",               # do not trade
}


def compute_breadth(closes: pd.DataFrame, span: int = 50) -> pd.Series:
    """
    closes : wide frame, one column per stock, indexed by date.
    Returns % of stocks with data on that date trading above their own EMA.

    Stocks not yet listed contribute NaN and are excluded from both numerator
    and denominator, so the figure is not diluted by absent names.
    """
    e = closes.apply(lambda s: ema(s.dropna(), span).reindex(closes.index))
    above = (closes > e)
    valid = closes.notna() & e.notna()
    return (above & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan) * 100


def compute_regime(nifty_close: pd.Series, breadth: pd.Series | None = None,
                   p: dict | None = None) -> pd.DataFrame:
    """
    Three-state classification driving WHICH STRATEGY trades, not just how
    wide the stop is.

      TRENDING : Nifty above its 20 EMA AND breadth > 55%  -> breakout
      BEAR     : Nifty below its 20 EMA AND breadth < 35%  -> cash
      RANGE    : everything in between                     -> mean reversion

    The v3 backtest measured a 0.43R spread between its best and worst regime
    on the same strategy, which is the evidence this split rests on. Bear is
    cash rather than a tighter stop because the measured expectancy there was
    -0.268R: the correct size in that regime is zero.
    """
    p = p or PARAMS
    out = pd.DataFrame(index=nifty_close.index)
    e20 = ema(nifty_close, 20)
    out["nifty_close"] = nifty_close
    out["nifty_ema20"] = e20
    out["nifty_above"] = nifty_close > e20
    out["gap_pct"] = (nifty_close / e20 - 1) * 100

    if breadth is None:
        # Without breadth the index alone decides, and we never claim the
        # bear state — a cash call needs more evidence than one index line.
        b = pd.Series(np.nan, index=out.index)
        trending = out["nifty_above"]
        bear = pd.Series(False, index=out.index)
    else:
        b = breadth.reindex(out.index)
        trending = out["nifty_above"] & (b > BREADTH_TRENDING)
        bear = (~out["nifty_above"]) & (b < BREADTH_BEAR)
    out["breadth"] = b

    state = pd.Series("range", index=out.index, dtype=object)
    state = state.mask(trending.fillna(False), "trending")
    state = state.mask(bear.fillna(False), "bear")
    out["state"] = state
    out["strategy"] = out["state"].map(REGIME_STRATEGY)

    out["atr_stop_mult"] = out["state"].map({
        "trending": p["atr_stop_normal"],
        "range": p["atr_stop_caution"],
        "bear": p["atr_stop_strict"],
    }).astype(float)
    return out


def compute_sector_scores(sector_closes: dict[str, pd.Series],
                          nifty_close: pd.Series,
                          p: dict | None = None) -> dict[str, pd.Series]:
    """
    Per-sector 0-15 score series: 8 points for the sector index above its own
    20 EMA, 7 for outperforming the Nifty over the trailing 10 sessions.
    """
    p = p or PARAMS
    nifty_10d = nifty_close / nifty_close.shift(10) - 1
    out = {}
    for name, close in sector_closes.items():
        close = close.reindex(nifty_close.index).ffill()
        above = close > ema(close, 20)
        rs10 = close / close.shift(10) - 1
        outperf = rs10 > nifty_10d
        score = (above.fillna(False).astype(float) * p["sector_ema_pts"]
                 + outperf.fillna(False).astype(float) * p["sector_rs_pts"])
        # before enough history exists, fall back to neutral rather than zero
        score = score.where(close.notna() & ema(close, 20).notna(), p["sector_neutral"])
        out[name] = score
    return out
