"""
paper_log.py — captures every pick, resolves outcomes causally
==============================================================

Design principles

1. ONE LOG. Trending picks and sit-out picks land in the same CSV with a
   `regime` column. Separate logs would drift; one log with a filter cannot.

2. RESOLUTION IS CAUSAL. Each nightly scan walks OPEN positions forward
   against the day just closed, using only the bar that just landed. It never
   looks at "today's chart" from the future. This mirrors backtest.simulate_trade
   exactly — same tie-breaks, same cost model, same behaviour on gaps — so a
   trade closed here is directly comparable to a trade the backtest would have
   closed on the same data.

3. THE HONEST TIE-BREAK. When a bar spans both stop and target, the stop
   wins. Daily bars cannot resolve intraday sequence, and assuming the
   favourable order is the single most common way a paper log flatters
   itself.

4. NO REVISIONS. Once a row is closed with an exit price, it is never
   reopened. Data providers do sometimes revise a historical bar; the log
   uses whichever bar was in front of the scan on the day of decision, and
   sticks with it. That is what live trading feels like.
"""

from __future__ import annotations

import csv
import json
import os
import datetime as dt

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE, "data", "paper_log.csv")

# Cost model — matches backtest.COSTS. If either moves, the other must.
COST_PER_SIDE = 0.15 / 100
SLIP_PER_SIDE = 0.15 / 100
MAX_HOLDING_DAYS = 15

COLUMNS = [
    "signal_date", "symbol", "name", "sector",
    "regime", "regime_state", "score", "score_components_json",
    "prev_high", "atr", "atr_pct",
    "entry_trigger", "stop_planned", "target1_planned",
    # execution state, filled once/twice as the trade unfolds
    "entry_date", "entry_px", "stop_px", "target1_px",
    "t1_hit", "exit_date", "exit_px", "exit_reason",
    "r_multiple", "pnl_pct", "holding_days",
    "status",   # open / filled_open / closed / expired
    "notes",
]


def ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)


def _read_log() -> pd.DataFrame:
    ensure_log()
    # Load with all columns as object so downstream .at[i, col] = value writes
    # never fail on dtype coercion — pandas 2.x refuses to write a bool into a
    # float column, which is what happened when t1_hit and status were being
    # inferred from an empty file.
    df = pd.read_csv(LOG_PATH, dtype=object)
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS]


def _write_log(df: pd.DataFrame):
    df.to_csv(LOG_PATH, index=False)


# ===========================================================================
# Append new picks from tonight's scan
# ===========================================================================
def append_picks(picks: list[dict], regime_state: str, scan_date: dt.date):
    """
    Every pick from tonight's scan enters the log with status='open' — meaning
    the trigger has been set but the market has not yet acted on it. Tomorrow's
    scan will look for a fill.

    Same-symbol de-duplication: if this symbol already has an unclosed row, we
    do not add another. In live trading you would not stack a second breakout
    signal on top of an unfilled one from the previous day.
    """
    if not picks:
        return 0

    df = _read_log()
    active_syms = set()
    if not df.empty:
        active_syms = set(df.loc[df["status"].isin(["open", "filled_open"]),
                                 "symbol"].astype(str))

    added = 0
    new_rows = []
    signal_date_str = scan_date.isoformat()

    for p in picks:
        sym = p.get("symbol")
        if not sym or sym in active_syms:
            continue

        row = {c: None for c in COLUMNS}
        row.update({
            "signal_date": signal_date_str,
            "symbol": sym,
            "name": p.get("name"),
            "sector": p.get("sector"),
            # 'regime' is the label used for filtering the log downstream:
            # trending picks are 'trade', everything else is 'paper'.
            "regime": "trade" if regime_state == "trending" else "paper",
            "regime_state": regime_state,
            "score": p.get("score"),
            "score_components_json": json.dumps(p.get("components") or {}),
            "prev_high": p.get("prev_high"),
            "atr": p.get("atr") if p.get("atr") is not None
                   else (p.get("risk_pct", 0) * p.get("price", 0) / 100 / 1.5
                         if p.get("price") and p.get("risk_pct") else None),
            "atr_pct": p.get("atr_pct"),
            "entry_trigger": p.get("entry"),
            "stop_planned": p.get("stop"),
            "target1_planned": p.get("target1"),
            "status": "open",
            "notes": "",
        })
        new_rows.append(row)
        added += 1

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        _write_log(df)
    return added


# ===========================================================================
# Resolve open positions against the latest bar
# ===========================================================================
def _apply_costs_exit(px_raw: float) -> float:
    """Sell side: pay charges and slippage."""
    return px_raw * (1 - COST_PER_SIDE) * (1 - SLIP_PER_SIDE)


def _apply_costs_entry(px_raw: float) -> float:
    """Buy side: pay slippage and charges."""
    return px_raw * (1 + SLIP_PER_SIDE) * (1 + COST_PER_SIDE)


def resolve_open(price_data: dict[str, pd.DataFrame], today: dt.date):
    """
    Walk every open/filled_open row forward against the latest bar in price_data.

    price_data : symbol -> OHLCV frame the scan has just downloaded.

    The traversal has to be bar-by-bar (not "look at today's bar in isolation")
    because a row may have been sitting untouched for several days — a stock
    the scan does not report on every day (limit runs, small caps that dropped
    out of the universe temporarily, weekends). We advance the row's state
    through every bar strictly AFTER its signal_date up to and including today,
    and only commit the final state.
    """
    df = _read_log()
    if df.empty:
        return {"resolved": 0, "filled": 0, "still_open": 0, "expired": 0}

    resolved = filled = expired = 0
    today_str = today.isoformat()

    def _f(v):
        """Parse a possibly-string cell to float, treating blanks as NaN."""
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" or v == "nan":
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def _b(v):
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")

    for i, row in df.iterrows():
        if row["status"] not in ("open", "filled_open"):
            continue

        sym = row["symbol"]
        bars = price_data.get(sym)
        if bars is None or bars.empty:
            continue

        signal_date = pd.to_datetime(row["signal_date"]).date()
        # only bars strictly after the signal are eligible for execution
        forward = bars[bars.index.date > signal_date]
        if forward.empty:
            continue

        trigger = _f(row["entry_trigger"])
        stop_planned = _f(row["stop_planned"])
        target = _f(row["target1_planned"])
        if not np.isfinite(trigger) or not np.isfinite(stop_planned) or not np.isfinite(target):
            continue

        state = row["status"]
        # entry_i is the row position of the entry bar within `forward`. On a
        # continuing filled_open row we recover it from the stored entry_date
        # rather than leaving it None — otherwise bars_since_entry = 0 forever
        # and the time stop can never fire on a position that spans several
        # scan runs.
        entry_i = None
        if state == "filled_open" and row["entry_date"] not in (None, "", float("nan")):
            try:
                ent_date = pd.to_datetime(row["entry_date"]).date()
                match = np.where(forward.index.date == ent_date)[0]
                if len(match):
                    entry_i = int(match[0])
            except Exception:
                pass
        entry_px = _f(row["entry_px"]) if row["entry_px"] not in (None, "", float("nan")) else None
        if entry_px is not None and not np.isfinite(entry_px):
            entry_px = None
        exit_px = None
        exit_reason = None
        t1_hit = _b(row["t1_hit"])
        booked = 0.0
        frac_open = 1.0 if state == "filled_open" else 0.0
        sp = _f(row["stop_px"])
        stop_dynamic = sp if np.isfinite(sp) else stop_planned

        # ---------- ENTRY search (state == 'open') ----------
        forward_arr = forward.reset_index()
        for j, bar in forward_arr.iterrows():
            bar_date = bar["Date"].date() if "Date" in bar else bar[forward.index.name or "index"].date()

            if state == "open":
                # order valid one day; skip if opened too far above trigger (per backtest)
                op = float(bar["Open"])
                hi = float(bar["High"])
                if op >= trigger:
                    if op > trigger * 1.02:
                        # gapped too far — order expires unfilled
                        state = "expired"
                        exit_reason = "gap_too_far"
                        expired += 1
                        break
                    entry_px = op
                elif hi >= trigger:
                    entry_px = trigger
                else:
                    # not filled today; day-1 order expires
                    state = "expired"
                    exit_reason = "unfilled"
                    expired += 1
                    break

                entry_i = j
                stop_dynamic = entry_px - (entry_px - stop_planned)  # keep planned distance
                state = "filled_open"
                frac_open = 1.0
                filled += 1
                # fall through to manage this same bar if it also hit target/stop

            if state == "filled_open":
                op = float(bar["Open"]); hi = float(bar["High"])
                lo = float(bar["Low"]);  cl = float(bar["Close"])
                bars_since_entry = j - (entry_i if entry_i is not None else j)

                gap_stop = op <= stop_dynamic and j > (entry_i or -1)
                hit_stop = lo <= stop_dynamic or gap_stop
                hit_t1 = (not t1_hit) and hi >= target

                if hit_stop:
                    px_out = min(op, stop_dynamic) if gap_stop else stop_dynamic
                    booked += frac_open * _apply_costs_exit(px_out)
                    exit_px = px_out
                    exit_reason = "stop_after_t1" if t1_hit else "stop"
                    state = "closed"
                    break

                if hit_t1:
                    booked += 0.5 * _apply_costs_exit(target)
                    frac_open = 0.5
                    t1_hit = True
                    stop_dynamic = max(stop_dynamic, entry_px)   # to breakeven

                if bars_since_entry >= MAX_HOLDING_DAYS:
                    booked += frac_open * _apply_costs_exit(cl)
                    exit_px = cl
                    exit_reason = "time_stop"
                    state = "closed"
                    break

        # ---------- write back ----------
        if state in ("closed", "expired", "filled_open"):
            entry_bar_date = None
            if entry_i is not None:
                entry_bar_date = forward.index[entry_i].date().isoformat()

            df.at[i, "entry_date"] = entry_bar_date
            df.at[i, "entry_px"] = round(entry_px, 2) if entry_px else None
            df.at[i, "stop_px"] = round(stop_dynamic, 2) if entry_px else None
            df.at[i, "target1_px"] = round(target, 2)
            df.at[i, "t1_hit"] = bool(t1_hit)

            if state == "expired":
                df.at[i, "status"] = "expired"
                df.at[i, "exit_reason"] = exit_reason
                df.at[i, "notes"] = f"Order not filled; expired {today_str}"

            elif state == "closed":
                entry_cost = _apply_costs_entry(entry_px)
                pnl_per_unit = booked - entry_cost
                risk = entry_px - float(row["stop_planned"])
                r_mult = pnl_per_unit / risk if risk > 0 else None
                pnl_pct = pnl_per_unit / entry_cost * 100
                holding = None
                if entry_i is not None:
                    exit_index = forward.index[min(j, len(forward) - 1)]
                    entry_index = forward.index[entry_i]
                    holding = (exit_index - entry_index).days

                df.at[i, "status"] = "closed"
                df.at[i, "exit_date"] = forward.index[min(j, len(forward) - 1)].date().isoformat()
                df.at[i, "exit_px"] = round(exit_px, 2) if exit_px else None
                df.at[i, "exit_reason"] = exit_reason
                df.at[i, "r_multiple"] = round(float(r_mult), 3) if r_mult is not None else None
                df.at[i, "pnl_pct"] = round(float(pnl_pct), 2)
                df.at[i, "holding_days"] = holding
                resolved += 1

            elif state == "filled_open":
                df.at[i, "status"] = "filled_open"

    _write_log(df)
    still_open = int((df["status"].isin(["open", "filled_open"])).sum())
    return {"resolved": resolved, "filled": filled,
            "still_open": still_open, "expired": expired}


# ===========================================================================
# Summary — the honest scoreboard
# ===========================================================================
def summary(min_age_days: int = 0) -> dict:
    """
    Aggregate stats separated by regime label ('trade' vs 'paper').
    min_age_days lets you exclude very recent trades whose outcomes may still be pending.
    """
    df = _read_log()
    out = {"total_rows": int(len(df)),
           "open": int((df["status"].isin(["open", "filled_open"])).sum()),
           "closed": int((df["status"] == "closed").sum()),
           "expired": int((df["status"] == "expired").sum()),
           "by_regime": {}}

    if df.empty or (df["status"] == "closed").sum() == 0:
        return out

    closed = df[df["status"] == "closed"].copy()
    if min_age_days > 0:
        cutoff = (dt.date.today() - dt.timedelta(days=min_age_days)).isoformat()
        closed = closed[closed["signal_date"] <= cutoff]

    for label, grp in closed.groupby("regime"):
        r = pd.to_numeric(grp["r_multiple"], errors="coerce").dropna()
        if r.empty:
            continue
        wins = r[r > 0]; losses = r[r <= 0]
        se = float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
        out["by_regime"][label] = {
            "n": int(len(r)),
            "hit_rate_pct": round(float((r > 0).mean() * 100), 1),
            "expectancy_R": round(float(r.mean()), 3),
            "expectancy_SE": round(se, 3) if np.isfinite(se) else None,
            "total_R": round(float(r.sum()), 1),
            "avg_win_R": round(float(wins.mean()), 2) if len(wins) else 0.0,
            "avg_loss_R": round(float(losses.mean()), 2) if len(losses) else 0.0,
        }
    return out


def read_all() -> pd.DataFrame:
    return _read_log()
