"""
run_v4.py — regime-switched validation on a clean data split
============================================================

THE DATA HYGIENE PROBLEM THIS SOLVES
------------------------------------
The v3 holdout (Nov 2025 - Aug 2026) has now been looked at four times. Every
look burns it. Anything developed against it and then tested on it would be
measuring memory, not edge.

So v4 pulls TEN years and splits by era:

    DEVELOP  2016-2021   never examined. Regime thresholds, mean-reversion
                         parameters and sensitivity sweeps are read here.
    VERIFY   2022-2026   opened ONCE at the end. This spans the 2022 bear,
                         the 2023-24 trend, and the 2024-25 chop, so it tests
                         all three regimes rather than whichever one happened
                         to dominate a short window.

The 2016-2021 era also contains 2018's grind and the 2020 crash and recovery,
which is the point: the regime classifier cannot be validated on data that
only contains one regime.

    python run_v4.py --years 10
    python run_v4.py --years 10 --limit 400 --skip-mcap   # faster first pass

WHAT WOULD COUNT AS SUCCESS
---------------------------
Not "the switched system made money". The bar is that switching beats the
alternatives on the SAME dates:

  * switched > breakout-always      (regime awareness earns its complexity)
  * switched > mean-reversion-always
  * switched > random picks         (the scoring earns its place)
  * VERIFY expectancy positive after costs, with a usable sample

If switching only ties breakout-always, the regime classifier is decoration
and the honest answer is to trade breakouts and sit out the rest.
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
from features import build_stock_frame
from strategy_mr import build_mr_frame, MR_PARAMS
from regime import compute_regime, compute_breadth, compute_sector_scores
from backtest import summarise, component_attribution, exit_reason_breakdown
from switched import run_switched, compare_arms, per_strategy_split
from run_backtest import (load_universe, download, approx_market_cap,
                          NIFTY, SECTOR_INDICES)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results_v4")
os.makedirs(OUT, exist_ok=True)


def null_test_switched(px, bo, mr, reg, n_symbols=60, seed=0):
    """Shuffled returns must not produce edge. Same guard as v3."""
    rng = np.random.default_rng(seed)
    syms = list(px)[:n_symbols]
    sh = {}
    for s in syms:
        d = px[s]
        r = d["Close"].pct_change().dropna().to_numpy().copy()
        rng.shuffle(r)
        newc = np.concatenate([[d["Close"].iloc[0]],
                               d["Close"].iloc[0] * np.cumprod(1 + r)])
        scale = newc / d["Close"].to_numpy()
        nd = d.copy()
        for col in ["Open", "High", "Low", "Close"]:
            nd[col] = nd[col].to_numpy() * scale
        sh[s] = nd
    b = {s: build_stock_frame(sh[s]) for s in syms}
    m = {s: build_mr_frame(sh[s]) for s in syms}
    tr = run_switched(b, m, sh, reg, top_n=5)
    return summarise(tr, "NULL (shuffled)")


def sweep(param_dict, param, values, rebuild, px, reg, start, end, top_n=5,
          strategy=None):
    """One-parameter sensitivity. Read plateaus, not peaks."""
    rows = []
    original = param_dict[param]
    try:
        for v in values:
            param_dict[param] = v
            bo, mr = rebuild()
            tr = run_switched(bo, mr, px, reg, start=start, end=end,
                              top_n=top_n, force_strategy=strategy)
            s = summarise(tr, f"{param}={v}")
            s["param"], s["value"] = param, v
            rows.append(s)
    finally:
        param_dict[param] = original
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--skip-mcap", action="store_true")
    ap.add_argument("--verify-start", default="2022-01-01",
                    help="start of the untouched verification era")
    args = ap.parse_args()

    print("=" * 70)
    print("SWING SCREENER v4 — REGIME-SWITCHED VALIDATION")
    print("=" * 70)

    uni = load_universe(limit=args.limit)
    syms = uni["symbol"].tolist()
    print(f"\nUniverse: {len(syms)} symbols | {args.years}y")

    print("\nDownloading stocks…")
    px = download(syms, args.years, f"v4stocks{args.limit or 'all'}")
    print(f"  usable: {len(px)}")
    if len(px) < 50:
        sys.exit("Too few symbols to conclude anything.")

    print("Downloading indices…")
    idx = download([NIFTY] + list(SECTOR_INDICES.values()), args.years, "v4idx")
    if NIFTY not in idx:
        sys.exit("No Nifty data — the regime gate cannot be built.")

    caps = {}
    if not args.skip_mcap:
        print("Market-cap approximation…")
        caps = approx_market_cap(px, list(px))
        print(f"  available for {len(caps)}")

    print("Breadth and regime…")
    wide = pd.DataFrame({s: px[s]["Close"] for s in px})
    breadth = compute_breadth(wide)
    reg = compute_regime(idx[NIFTY]["Close"].reindex(wide.index).ffill(), breadth)
    print(f"  regime days: {reg['state'].value_counts().to_dict()}")

    smap_path = os.path.join(BASE, "data", "sector_map.csv")
    smap = {}
    if os.path.exists(smap_path):
        sm = pd.read_csv(smap_path)
        smap = dict(zip(sm["symbol"], sm["sector"]))
        print(f"  sector map: {len(smap)} symbols")
    sector_closes = {k: idx[v]["Close"] for k, v in SECTOR_INDICES.items() if v in idx}
    sect = compute_sector_scores(sector_closes, idx[NIFTY]["Close"]) if sector_closes else {}

    def rebuild():
        bo, mr = {}, {}
        for s, d in px.items():
            ss = sect.get(smap.get(s)) if smap else None
            bo[s] = build_stock_frame(d, sector_score=ss, market_cap_series=caps.get(s))
            mr[s] = build_mr_frame(d, market_cap_series=caps.get(s))
        return bo, mr

    print("Building frames (both strategies)…")
    bo_frames, mr_frames = rebuild()

    dates = reg.index
    dev_start = dates[0] + pd.DateOffset(months=12)   # EMA warm-up
    verify_start = pd.Timestamp(args.verify_start)
    if verify_start <= dev_start:
        sys.exit(f"Not enough history: develop era would be empty. "
                 f"Data starts {dates[0].date()}; need --years 10.")
    print(f"\n  DEVELOP {dev_start.date()} → {verify_start.date()}")
    print(f"  VERIFY  {verify_start.date()} → {dates[-1].date()}   (opened once)")

    rep = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "universe": len(px), "years": args.years,
           "develop": [str(dev_start.date()), str(verify_start.date())],
           "verify": [str(verify_start.date()), str(dates[-1].date())],
           "regime_days": {k: int(v) for k, v in reg["state"].value_counts().items()}}

    # ---------------- 0. null ----------------
    print("\n[0] NULL TEST")
    nt = null_test_switched(px, bo_frames, mr_frames, reg)
    rep["null"] = nt
    print(f"    {nt}")
    if nt.get("n", 0) > 30 and nt.get("expectancy_R", 0) > 0.10:
        print("    *** POSITIVE ON SHUFFLED DATA — LOOKAHEAD. STOP. ***")

    # ---------------- 1. develop ----------------
    print("\n[1] DEVELOP era — hypotheses only")
    arms_dev = compare_arms(bo_frames, mr_frames, px, reg,
                            start=dev_start, end=verify_start, top_n=args.top_n)
    rep["develop_arms"] = arms_dev.to_dict("records")
    print(arms_dev[["label", "n", "hit_rate_pct", "expectancy_R",
                    "expectancy_SE", "total_R", "max_dd_R"]].to_string(index=False))

    tr_dev = run_switched(bo_frames, mr_frames, px, reg,
                          start=dev_start, end=verify_start, top_n=args.top_n)
    if not tr_dev.empty:
        sp = per_strategy_split(tr_dev)
        print("\n    Which strategy is carrying it:")
        print(sp[["label", "n", "hit_rate_pct", "expectancy_R",
                  "expectancy_SE"]].to_string(index=False))
        rep["develop_by_strategy"] = sp.to_dict("records")
        print("\n    Exits:")
        print(exit_reason_breakdown(tr_dev).to_string(index=False))
        tr_dev.to_csv(os.path.join(OUT, "trades_develop.csv"), index=False)

    # ---------------- 2. sensitivity ----------------
    print("\n[2] SENSITIVITY on the develop era — plateaus, not peaks")
    sweeps = []
    for param, vals in [("rsi_entry_max", [30, 35, 40, 45]),
                        ("max_ema200_decline_pct", [-0.5, -1.5, -3.0, -99.0]),
                        ("target_atr_mult", [1.0, 1.5, 2.0]),
                        ("max_holding_days", [5, 8, 12])]:
        sdf = sweep(MR_PARAMS, param, vals, rebuild, px, reg, dev_start,
                    verify_start, args.top_n, strategy="mean_reversion")
        sweeps.append(sdf)
        print(f"\n    MR {param}")
        print(sdf[["value", "n", "hit_rate_pct", "expectancy_R",
                   "expectancy_SE"]].to_string(index=False))
    if sweeps:
        pd.concat(sweeps).to_csv(os.path.join(OUT, "sensitivity_mr.csv"), index=False)

    # ---------------- 3. verify ----------------
    print("\n[3] VERIFY era — untouched until now. This is the answer.")
    arms_ver = compare_arms(bo_frames, mr_frames, px, reg,
                            start=verify_start, top_n=args.top_n)
    rep["verify_arms"] = arms_ver.to_dict("records")
    print(arms_ver[["label", "n", "hit_rate_pct", "expectancy_R",
                    "expectancy_SE", "total_R", "max_dd_R"]].to_string(index=False))

    tr_ver = run_switched(bo_frames, mr_frames, px, reg,
                          start=verify_start, top_n=args.top_n)
    if not tr_ver.empty:
        sp = per_strategy_split(tr_ver)
        print("\n    By strategy:")
        print(sp[["label", "n", "hit_rate_pct", "expectancy_R",
                  "expectancy_SE"]].to_string(index=False))
        rep["verify_by_strategy"] = sp.to_dict("records")
        print("\n    By regime:")
        rows = [summarise(g, k) for k, g in tr_ver.groupby("regime")]
        print(pd.DataFrame(rows)[["label", "n", "hit_rate_pct",
                                  "expectancy_R"]].to_string(index=False))
        tr_ver.to_csv(os.path.join(OUT, "trades_verify.csv"), index=False)

    # ---------------- verdict ----------------
    print("\n" + "=" * 70)
    def get(df, key):
        m = df[df["label"].str.contains(key, regex=False)]
        return float(m["expectancy_R"].iloc[0]) if len(m) else float("nan")

    sw = get(arms_ver, "switched (regime-aware)")
    bo_only = get(arms_ver, "breakout always")
    rnd = get(arms_ver, "random")
    print(f"VERIFY: switched {sw:+.3f}R | breakout-always {bo_only:+.3f}R | "
          f"random {rnd:+.3f}R")
    if np.isfinite(sw):
        if sw <= 0:
            print("VERDICT: no edge out of sample. Switching did not rescue it.")
        elif np.isfinite(bo_only) and sw <= bo_only + 0.05:
            print("VERDICT: switching adds nothing over breakout-always.")
        elif np.isfinite(rnd) and sw <= rnd + 0.05:
            print("VERDICT: scoring adds nothing over random within the regime.")
        else:
            print("VERDICT: switched system beat every alternative out of sample. "
                  "Forward-test on paper before committing capital.")
    rep["verdict"] = {"switched": sw, "breakout_always": bo_only, "random": rnd}

    with open(os.path.join(OUT, "report_v4.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\nWritten to {OUT}/")


if __name__ == "__main__":
    main()
