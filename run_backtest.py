"""
run_backtest.py — the validation protocol
=========================================

Order of operations matters here. The out-of-sample period is opened ONCE,
at the end, after every decision has been made. Everything before that runs
on the training period only.

    python run_backtest.py --years 5

Stages
------
  0  NULL TEST      engine on shuffled returns; expectancy must not be
                    positive. Catches lookahead before anything else runs.
  1  TRAIN          headline stats, component attribution, score deciles,
                    regime split — all in-sample. Hypotheses only.
  2  BASELINES      random selection from the same filtered set, and a
                    filters-only run. Separates "the filters work" from
                    "the scoring works".
  3  SENSITIVITY    each key threshold swept. Plateaus are real; spikes are
                    curve-fitting.
  4  WALK-FORWARD   sequential train/test windows, out-of-sample only.
  5  HOLDOUT        the final untouched period. Read once. This is the number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

import features as F
from features import build_stock_frame, PARAMS
from regime import compute_regime, compute_breadth, compute_sector_scores
import backtest as B
from backtest import (run_backtest, summarise, component_attribution,
                      score_decile_analysis, regime_split, exit_reason_breakdown)

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
OUT = os.path.join(BASE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

NIFTY = "^NSEI"
SECTOR_INDICES = {
    "Bank": "^NSEBANK", "IT": "^CNXIT", "Pharma": "^CNXPHARMA",
    "Auto": "^CNXAUTO", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY", "Realty": "^CNXREALTY",
    "PSU Bank": "^CNXPSUBANK", "Infrastructure": "^CNXINFRA",
    "Media": "^CNXMEDIA",
}


# ===========================================================================
# Data
# ===========================================================================
def load_universe(path=None, limit=None) -> pd.DataFrame:
    path = path or os.path.join(BASE, "data", "nse_universe.csv")
    if not os.path.exists(path):
        sys.exit(f"Universe file not found at {path}.\n"
                 "Run fetch_universe.py from an Indian connection first.")
    u = pd.read_csv(path)
    return u.head(limit) if limit else u


def download(symbols, years, tag) -> dict:
    """Batch download with an on-disk cache — refetching 2,000 stocks hurts."""
    import yfinance as yf
    cache = os.path.join(CACHE, f"bt_{tag}_{years}y.parquet")
    if os.path.exists(cache):
        flat = pd.read_parquet(cache)
        return {s: g.droplevel(0) for s, g in flat.groupby(level=0)}

    out = {}
    tickers = [s if s.startswith("^") else s + ".NS" for s in symbols]
    step = 100
    for i in range(0, len(tickers), step):
        chunk = tickers[i:i + step]
        try:
            data = yf.download(chunk, period=f"{years}y", interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=True, progress=False)
        except Exception as e:
            print(f"  chunk {i//step} failed: {e}")
            continue
        for t in chunk:
            try:
                d = data[t].dropna(how="all") if len(chunk) > 1 else data.dropna(how="all")
            except (KeyError, TypeError):
                continue
            if d is None or d.empty or len(d) < 250:
                continue
            key = t[:-3] if t.endswith(".NS") else t
            out[key] = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        print(f"  {min(i+step, len(tickers))}/{len(tickers)}", flush=True)

    if out:
        pd.concat(out, names=["symbol", "date"]).to_parquet(cache)
    return out


def approx_market_cap(px: dict, symbols) -> dict:
    """
    Historical market cap approximated as (current shares outstanding) x
    (historical price). Shares outstanding is fetched once per symbol.

    This is an approximation: it ignores issuance, buybacks and splits over
    the test window, so a company that doubled its share count looks smaller
    in the past than it was. The alternative — applying today's market cap to
    a 2022 signal — is outright lookahead, which is worse. Documented in
    METHODOLOGY.md as a known limitation.
    """
    import yfinance as yf
    caps = {}
    for i, s in enumerate(symbols):
        try:
            info = yf.Ticker(s + ".NS").info
            mc, pr = info.get("marketCap"), info.get("currentPrice")
            if mc and pr and pr > 0 and s in px:
                shares = mc / pr
                caps[s] = px[s]["Close"] * shares
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  market cap {i+1}/{len(symbols)}", flush=True)
    return caps


# ===========================================================================
# Stage 0 — null test
# ===========================================================================
def null_test(px: dict, reg: pd.DataFrame, n_symbols=60, seed=0) -> dict:
    """
    Destroy the time-ordering of returns while keeping their distribution,
    then rerun. Any real signal disappears; any lookahead survives.

    A materially positive expectancy here means the engine is reading the
    future and every other number in this report is worthless.
    """
    rng = np.random.default_rng(seed)
    syms = list(px)[:n_symbols]
    shuffled = {}
    for s in syms:
        d = px[s]
        r = d["Close"].pct_change().dropna().to_numpy().copy()
        rng.shuffle(r)
        newc = d["Close"].iloc[0] * np.cumprod(1 + r)
        newc = np.concatenate([[d["Close"].iloc[0]], newc])
        scale = newc / d["Close"].to_numpy()
        nd = d.copy()
        for col in ["Open", "High", "Low", "Close"]:
            nd[col] = nd[col].to_numpy() * scale
        shuffled[s] = nd

    frames = {s: build_stock_frame(shuffled[s]) for s in syms}
    tr = run_backtest(frames, shuffled, reg, top_n=5, max_concurrent=5)
    return summarise(tr, "NULL (shuffled returns)")


# ===========================================================================
# Stage 3 — sensitivity
# ===========================================================================
def sensitivity(frames_builder, px, reg, start, end, param, values,
                top_n=5) -> pd.DataFrame:
    """
    Sweep one threshold and report out-of-sample expectancy at each value.

    Read the SHAPE, not the maximum. A parameter that works across a
    contiguous band of values is describing something real. One that spikes
    at a single value and collapses either side of it has been fitted to
    noise in this particular sample, and will not survive live.
    """
    rows = []
    original = PARAMS[param]
    try:
        for v in values:
            PARAMS[param] = v
            frames = frames_builder()
            tr = run_backtest(frames, px, reg, start=start, end=end, top_n=top_n)
            s = summarise(tr, f"{param}={v}")
            s["param"], s["value"] = param, v
            rows.append(s)
    finally:
        PARAMS[param] = original
    return pd.DataFrame(rows)


# ===========================================================================
# Stage 4 — walk-forward
# ===========================================================================
def walk_forward(frames, px, reg, first_test_start, n_windows=4,
                 window_months=6, top_n=5) -> pd.DataFrame:
    """
    Sequential out-of-sample windows. Each window is evaluated with rules
    that were fixed before that window began, so consistency across windows
    is evidence the result is not one lucky period.
    """
    rows = []
    start = pd.Timestamp(first_test_start)
    for w in range(n_windows):
        s = start + pd.DateOffset(months=window_months * w)
        e = s + pd.DateOffset(months=window_months) - pd.Timedelta(days=1)
        tr = run_backtest(frames, px, reg, start=s, end=e, top_n=top_n)
        r = summarise(tr, f"W{w+1} {s.date()}→{e.date()}")
        r["start"], r["end"] = str(s.date()), str(e.date())
        rows.append(r)
    return pd.DataFrame(rows)


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size (useful for a fast first run)")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--holdout-months", type=int, default=9,
                    help="final untouched out-of-sample period")
    ap.add_argument("--skip-mcap", action="store_true",
                    help="skip market-cap filter (much faster; see METHODOLOGY)")
    args = ap.parse_args()

    print("=" * 70)
    print("SWING SCREENER BACKTEST")
    print("=" * 70)

    uni = load_universe(limit=args.limit)
    syms = uni["symbol"].tolist()
    print(f"\nUniverse: {len(syms)} symbols | {args.years}y of daily data")

    print("\nDownloading stock data…")
    px = download(syms, args.years, "stocks")
    print(f"  usable: {len(px)} symbols")
    if len(px) < 50:
        sys.exit("Too few symbols downloaded to draw any conclusion.")

    print("Downloading indices…")
    idx = download([NIFTY] + list(SECTOR_INDICES.values()), args.years, "idx")
    if NIFTY not in idx:
        sys.exit("Nifty 50 data unavailable — the regime gate cannot be built.")
    nifty_close = idx[NIFTY]["Close"]

    caps = {}
    if not args.skip_mcap:
        print("Fetching shares outstanding for market-cap approximation…")
        caps = approx_market_cap(px, list(px))
        print(f"  market cap available for {len(caps)} symbols")

    print("Computing breadth and regime…")
    closes_wide = pd.DataFrame({s: px[s]["Close"] for s in px})
    breadth = compute_breadth(closes_wide)
    reg = compute_regime(nifty_close.reindex(closes_wide.index).ffill(), breadth)
    print(f"  regime days: {reg['state'].value_counts().to_dict()}")

    sector_closes = {k: idx[v]["Close"] for k, v in SECTOR_INDICES.items() if v in idx}
    sect = compute_sector_scores(sector_closes, nifty_close) if sector_closes else {}
    print(f"  sector indices available: {len(sect)}")

    # sector assignment is fetched live in the screener; for the backtest we
    # leave it neutral unless a mapping file is supplied, and say so.
    smap_path = os.path.join(BASE, "data", "sector_map.csv")
    smap = {}
    if os.path.exists(smap_path):
        sm = pd.read_csv(smap_path)
        smap = dict(zip(sm["symbol"], sm["sector"]))
        print(f"  sector map loaded for {len(smap)} symbols")
    else:
        print("  no sector_map.csv — sector component held at neutral 7.5 "
              "(see METHODOLOGY.md)")

    def build_all():
        out = {}
        for s, d in px.items():
            ss = sect.get(smap.get(s)) if smap else None
            out[s] = build_stock_frame(d, sector_score=ss,
                                       market_cap_series=caps.get(s))
        return out

    print("Building feature frames…")
    frames = build_all()

    all_dates = reg.index
    holdout_start = all_dates[-1] - pd.DateOffset(months=args.holdout_months)
    train_start = all_dates[0] + pd.DateOffset(months=9)   # warm-up for EMAs
    print(f"\n  TRAIN   {train_start.date()} → {holdout_start.date()}")
    print(f"  HOLDOUT {holdout_start.date()} → {all_dates[-1].date()}  (opened once)")

    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "universe": len(px), "years": args.years,
              "train_start": str(train_start.date()),
              "holdout_start": str(holdout_start.date())}

    # ---------------- stage 0 ----------------
    print("\n[0] NULL TEST — shuffled returns, expectancy must not be positive")
    nt = null_test(px, reg)
    report["null_test"] = nt
    print(f"    {nt}")
    if nt.get("n", 0) > 30 and nt.get("expectancy_R", 0) > 0.10:
        print("    *** WARNING: positive expectancy on shuffled data. ***")
        print("    *** Suspect lookahead. Do not trust anything below.  ***")

    # ---------------- stage 1 ----------------
    print("\n[1] TRAIN — in-sample. Hypotheses only, not conclusions.")
    tr_train = run_backtest(frames, px, reg, start=train_start, end=holdout_start,
                            top_n=args.top_n)
    s_train = summarise(tr_train, "train")
    report["train"] = s_train
    print(f"    {s_train}")

    if not tr_train.empty:
        att = component_attribution(tr_train)
        dec = score_decile_analysis(tr_train)
        rsp = regime_split(tr_train)
        exr = exit_reason_breakdown(tr_train)
        att.to_csv(os.path.join(OUT, "component_attribution.csv"), index=False)
        dec.to_csv(os.path.join(OUT, "score_deciles.csv"), index=False)
        rsp.to_csv(os.path.join(OUT, "regime_split.csv"), index=False)
        tr_train.to_csv(os.path.join(OUT, "trades_train.csv"), index=False)
        print("\n    Component attribution (does each component separate winners?)")
        print(att.to_string(index=False))
        print("\n    Score deciles (does a higher score mean a higher R?)")
        print(dec.to_string(index=False))
        print("\n    By regime")
        print(rsp.to_string(index=False))
        print("\n    Exits")
        print(exr.to_string(index=False))

    # ---------------- stage 2 ----------------
    print("\n[2] BASELINES")
    rnd = run_backtest(frames, px, reg, start=train_start, end=holdout_start,
                       top_n=args.top_n, selection="random")
    s_rnd = summarise(rnd, "random from filtered set")
    report["baseline_random"] = s_rnd
    print(f"    {s_rnd}")
    if s_train.get("n", 0) and s_rnd.get("n", 0):
        edge = s_train["expectancy_R"] - s_rnd["expectancy_R"]
        se = np.sqrt((s_train.get("expectancy_SE") or 0) ** 2
                     + (s_rnd.get("expectancy_SE") or 0) ** 2)
        print(f"    scoring edge over random: {edge:+.3f}R "
              f"({edge/se:.1f} standard errors)" if se else "")
        print("    If this is near zero, the FILTERS are doing the work and "
              "the 100-point model is decoration.")

    # ---------------- stage 3 ----------------
    print("\n[3] SENSITIVITY — look for plateaus, not peaks")
    sweeps = {
        "vol_strong": [1.6, 1.8, 2.0, 2.2, 2.5],
        "rsi_lo": [45, 50, 55, 60],
        "max_trigger_gap_pct": [2.0, 3.0, 4.0, 6.0],
        "max_vol_ratio": [8.0, 15.0, 25.0, 1e9],
    }
    sens_all = []
    for param, vals in sweeps.items():
        sdf = sensitivity(build_all, px, reg, train_start, holdout_start,
                          param, vals, top_n=args.top_n)
        sens_all.append(sdf)
        print(f"\n    {param}")
        print(sdf[["value", "n", "hit_rate_pct", "expectancy_R",
                   "expectancy_SE"]].to_string(index=False))
    if sens_all:
        pd.concat(sens_all).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)

    # ---------------- stage 4 ----------------
    print("\n[4] WALK-FORWARD")
    wf_start = train_start + pd.DateOffset(months=12)
    wf = walk_forward(frames, px, reg, wf_start,
                      n_windows=max(2, int((holdout_start - wf_start).days / 182)),
                      top_n=args.top_n)
    report["walk_forward"] = wf.to_dict("records")
    print(wf[["label", "n", "hit_rate_pct", "expectancy_R", "total_R",
              "max_dd_R"]].to_string(index=False))
    wf.to_csv(os.path.join(OUT, "walk_forward.csv"), index=False)

    # ---------------- stage 5 ----------------
    print("\n[5] HOLDOUT — untouched until now. This is the number that counts.")
    tr_hold = run_backtest(frames, px, reg, start=holdout_start, top_n=args.top_n)
    s_hold = summarise(tr_hold, "holdout")
    report["holdout"] = s_hold
    print(f"    {s_hold}")
    if not tr_hold.empty:
        tr_hold.to_csv(os.path.join(OUT, "trades_holdout.csv"), index=False)

    if s_train.get("n") and s_hold.get("n"):
        drop = s_train["expectancy_R"] - s_hold["expectancy_R"]
        print(f"\n    train {s_train['expectancy_R']:+.3f}R  →  "
              f"holdout {s_hold['expectancy_R']:+.3f}R   (decay {drop:+.3f}R)")
        print("    Some decay is normal. A collapse to zero or negative means "
              "the in-sample result was fitted, not found.")

    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nWritten to {OUT}/")
    print("\nInterpretation guide: METHODOLOGY.md, section 6.")


if __name__ == "__main__":
    main()
