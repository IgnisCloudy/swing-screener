"""
build_sector_map.py — one-time sector assignment for the backtest
=================================================================

The live scanner fetches each stock's sector during Stage 2. The backtest
needs the same mapping up front, for the whole universe, so the 15-point
sector component actually functions instead of sitting at the neutral 7.5.

Writes data/sector_map.csv (symbol,sector). Resumable: rerun after an
interruption and it continues from where it stopped. This is slow — one HTTP
request per symbol, rate-limited by Yahoo — so budget 15-30 minutes for the
full universe. It only needs to run once; sectors rarely change.

    python build_sector_map.py               # whole universe
    python build_sector_map.py --limit 300   # match a --limit backtest

The sector labels here MUST match the keys in regime.SECTOR_INDICES, or the
sector score silently falls back to neutral. The override lists and the
yfinance-label map are copied from scan_job_v3.py so the two agree.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "sector_map.csv")

# --- must match scan_job_v3.py exactly ---
YF_SECTOR_MAP = {
    "Financial Services": "Bank", "Financial": "Bank", "Technology": "IT",
    "Healthcare": "Pharma", "Consumer Cyclical": "Auto",
    "Consumer Defensive": "FMCG", "Basic Materials": "Metal",
    "Energy": "Energy", "Utilities": "Energy", "Real Estate": "Realty",
    "Industrials": "Infrastructure", "Communication Services": "Media",
}
OVERRIDES = {s: "Sugar" for s in
             ["BALRAMCHIN", "SHREERENUKA", "DHAMPURSUG", "DALMIASUG", "TRIVENI",
              "AVADHSUGAR", "DWARKESH", "UTTAMSUGAR", "MAWANASUG", "BAJAJHIND",
              "RAJSHREE", "KMSUGAR", "MAGADSUGAR"]}
OVERRIDES.update({s: "PSU Bank" for s in
                  ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "INDIANB",
                   "CENTRALBK", "IOB", "UCOBANK", "MAHABANK", "PSB"]})
OVERRIDES.update({s: "Energy" for s in
                  ["RELIANCE", "ONGC", "IOC", "BPCL", "HPCL", "GAIL", "OIL",
                   "NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "JSWENERGY",
                   "NHPC", "SJVN", "COALINDIA", "PETRONET", "IGL", "MGL"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import yfinance as yf

    uni = pd.read_csv(os.path.join(DATA, "nse_universe.csv"))
    if args.limit:
        uni = uni.head(args.limit)
    symbols = uni["symbol"].tolist()

    done = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = dict(zip(prev["symbol"], prev["sector"]))
        print(f"resuming — {len(done)} already mapped")

    rows = dict(done)
    for i, sym in enumerate(symbols):
        if sym in rows:
            continue
        if sym in OVERRIDES:
            rows[sym] = OVERRIDES[sym]
        else:
            try:
                raw = yf.Ticker(sym + ".NS").info.get("sector")
                rows[sym] = YF_SECTOR_MAP.get(raw, raw) if raw else "Unmapped"
            except Exception:
                rows[sym] = "Unmapped"

        # checkpoint every 25 so an interruption loses almost nothing
        if (i + 1) % 25 == 0:
            pd.DataFrame([{"symbol": k, "sector": v} for k, v in rows.items()]) \
              .to_csv(OUT, index=False)
            print(f"  {i+1}/{len(symbols)} mapped", flush=True)
            time.sleep(0.3)   # be polite to Yahoo

    pd.DataFrame([{"symbol": k, "sector": v} for k, v in rows.items()]) \
      .to_csv(OUT, index=False)
    counts = pd.Series(list(rows.values())).value_counts()
    print(f"\nwrote {OUT} — {len(rows)} symbols")
    print("sector distribution:")
    print(counts.to_string())
    unmapped = counts.get("Unmapped", 0)
    if unmapped:
        print(f"\n{unmapped} unmapped ({unmapped/len(rows)*100:.0f}%) — these "
              "get the neutral 7.5 in the backtest, same as live.")


if __name__ == "__main__":
    main()
