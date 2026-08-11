"""
scan_job_v5.py — nightly live scan, breakout + regime gate only
===============================================================

WHAT CHANGED FROM v3/v4, AND WHY

Backtested over 2017-2026 across two independent eras, the measured result was:

    breakout in a TRENDING regime   develop +0.011R   verify +0.095R
    mean reversion in a RANGE       develop -0.234R   verify -0.242R

Mean reversion lost money in both eras at almost identical magnitude. It is
removed entirely — not tuned, removed. The reasoning behind it was sound;
the market did not pay for it, twice, and that is the answer.

What survived is the regime gate. Breakout-always measured -0.165R in the
verify era; breakout-only-when-trending measured +0.095R. That 0.26R gap is
the whole edge, and it comes from NOT TRADING rather than from picking better.

So this scanner does something the earlier versions did not: on non-trending
days it returns no picks at all. Not a shorter list, not wider stops — none.
Sitting out is the strategy.

HONEST CALIBRATION: +0.095R carries SE 0.093, so the confidence interval
includes zero, and survivorship bias in the backtest universe means the true
figure is lower still. This is a paper-trading candidate, not a system to
size up on. The app says so on every screen.

Scoring is imported from features.py — the same module the backtest uses. If
this file had its own copy of the rules, the backtest would be validating
something you are not trading.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import datetime as dt
import traceback

import numpy as np
import pandas as pd
import requests

from features import build_stock_frame, PARAMS
from regime import compute_regime, compute_breadth, compute_sector_scores
import paper_log

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
CACHE = os.path.join(BASE, "cache")
HIST = os.path.join(DATA, "history")
for d in (DATA, CACHE, HIST):
    os.makedirs(d, exist_ok=True)

LATEST = os.path.join(DATA, "latest.json")
NSE_LIST = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

NIFTY = "^NSEI"
SECTOR_INDICES = {
    "Bank": "^NSEBANK", "Financial Services": "^NSEBANK", "IT": "^CNXIT",
    "Pharma": "^CNXPHARMA", "Healthcare": "^CNXPHARMA", "Auto": "^CNXAUTO",
    "FMCG": "^CNXFMCG", "Sugar": "^CNXFMCG", "Consumer": "^CNXFMCG",
    "Metal": "^CNXMETAL", "Energy": "^CNXENERGY", "Oil & Gas": "^CNXENERGY",
    "Power": "^CNXENERGY", "Realty": "^CNXREALTY", "PSU Bank": "^CNXPSUBANK",
    "Infrastructure": "^CNXINFRA", "Media": "^CNXMEDIA",
}
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

STAGE2_N = 150
TOP_CHARTS = 8
CHART_BARS = 130


def get_universe() -> pd.DataFrame:
    cache = os.path.join(CACHE, "nse_universe.csv")
    repo = os.path.join(DATA, "nse_universe.csv")
    if os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) / 86400 < 7:
        return pd.read_csv(cache)
    try:
        r = requests.get(NSE_LIST, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].str.strip() == "EQ"][["SYMBOL", "NAME OF COMPANY"]]
        df.columns = ["symbol", "name"]
        df["symbol"] = df["symbol"].str.strip()
        df.to_csv(cache, index=False)
        return df
    except Exception:
        if os.path.exists(repo):
            return pd.read_csv(repo)
        raise RuntimeError("No NSE list and no committed fallback at "
                           "data/nse_universe.csv")


def download(symbols, period="2y", chunk=120, tag="px"):
    import yfinance as yf
    cache = os.path.join(CACHE, f"{tag}_{dt.date.today()}.parquet")
    if os.path.exists(cache):
        flat = pd.read_parquet(cache)
        return {s: g.droplevel(0) for s, g in flat.groupby(level=0)}
    out = {}
    tick = [s if s.startswith("^") else s + ".NS" for s in symbols]
    for i in range(0, len(tick), chunk):
        part = tick[i:i + chunk]
        try:
            data = yf.download(part, period=period, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        for t in part:
            try:
                d = data[t].dropna(how="all") if len(part) > 1 else data.dropna(how="all")
            except (KeyError, TypeError):
                continue
            if d is None or d.empty or len(d) < PARAMS["min_history"]:
                continue
            out[t[:-3] if t.endswith(".NS") else t] = d[["Open", "High", "Low",
                                                         "Close", "Volume"]].copy()
        print(f"  {min(i+chunk, len(tick))}/{len(tick)}", flush=True)
    if out:
        pd.concat(out, names=["symbol", "date"]).to_parquet(cache)
    return out


def fundamentals(sym):
    import yfinance as yf
    try:
        info = yf.Ticker(sym + ".NS").info
        return info.get("marketCap"), info.get("sector")
    except Exception:
        return None, None


def delivery_pct(sym):
    try:
        from jugaad_data.nse import stock_df
        end = dt.date.today()
        d = stock_df(symbol=sym, from_date=end - dt.timedelta(days=12),
                     to_date=end, series="EQ")
        for col in ("%DLY QT TO TRADED QTY", "DELIV_PER", "%DELIVERBLE"):
            if col in d.columns:
                v = pd.to_numeric(d[col], errors="coerce").dropna()
                if len(v):
                    return float(v.iloc[-1])
    except Exception:
        pass
    try:
        from nsepython import nse_eq
        v = nse_eq(sym).get("securityWiseDP", {}).get("deliveryToTradedQuantity")
        if v is not None:
            return float(v)
    except Exception:
        pass
    return None


def js(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, dt.date, dt.datetime)):
        return str(v)
    return v


def main():
    started = dt.datetime.now(dt.timezone.utc)
    print(f"[{started:%Y-%m-%d %H:%M} UTC] v5 scan starting", flush=True)
    payload = {"generated_utc": started.isoformat(timespec="seconds"),
               "version": "v5", "status": "error", "error": None,
               "regime": None, "picks": [], "universe_count": 0,
               "passed_count": 0, "charts": {}}
    try:
        uni = get_universe()
        payload["universe_count"] = len(uni)
        print(f"universe: {len(uni)}", flush=True)

        print("downloading stocks…", flush=True)
        px = download(uni["symbol"].tolist(), tag="px")
        print(f"  usable {len(px)}", flush=True)

        print("downloading indices…", flush=True)
        idx = download([NIFTY] + sorted(set(SECTOR_INDICES.values())),
                       period="1y", tag="idx")

        names = dict(zip(uni["symbol"], uni["name"]))
        wide = pd.DataFrame({s: px[s]["Close"] for s in px})
        breadth = compute_breadth(wide)

        if NIFTY in idx:
            nifty = idx[NIFTY]["Close"].reindex(wide.index).ffill()
            reg = compute_regime(nifty, breadth)
            last = reg.iloc[-1]
            state = str(last["state"])
            payload["regime"] = {
                "available": True, "state": state,
                "uptrend": bool(last["nifty_above"]),
                "atr_stop_mult": float(last["atr_stop_mult"]),
                "nifty_close": js(last["nifty_close"]),
                "nifty_ema20": js(last["nifty_ema20"]),
                "gap_pct": js(last["gap_pct"]),
                "breadth_pct": js(last["breadth"]),
                "tradeable": state == "trending",
            }
            stop_mult = float(last["atr_stop_mult"])
        else:
            # No index data means no regime call, and no regime call means no
            # trading. The gate IS the edge; running blind without it is the
            # configuration that measured negative.
            payload["regime"] = {"available": False, "state": "unknown",
                                 "atr_stop_mult": PARAMS["atr_stop_normal"],
                                 "breadth_pct": js(breadth.iloc[-1]),
                                 "tradeable": False}
            stop_mult = PARAMS["atr_stop_normal"]
            reg = None

        # ---- THE GATE ----
        # On sit-out days we still (a) resolve open paper positions against
        # today's bar so we can measure how they turned out, and (b) generate
        # picks marked as research so we can eventually compare their outcomes
        # against the tradeable-day picks. The gate closes trading, not
        # learning.
        research_mode = not payload["regime"].get("tradeable")
        if research_mode:
            payload["status"] = "research"
            payload["mode"] = "research"
            payload["reason"] = (
                f"Regime is '{payload['regime'].get('state')}', not trending. "
                "Breakouts measured negative expectancy outside a trending "
                "regime; picks below are RESEARCH ONLY — do not trade.")
            print(f"GATE CLOSED — research mode: {payload['reason']}", flush=True)
        else:
            payload["mode"] = "trade"
        print(f"regime: {payload['regime'].get('state')} "
              f"breadth={payload['regime'].get('breadth_pct')}", flush=True)

        sector_closes = {k: idx[v]["Close"] for k, v in SECTOR_INDICES.items()
                         if v in idx}
        sect = (compute_sector_scores(sector_closes, idx[NIFTY]["Close"])
                if sector_closes and NIFTY in idx else {})

        # ---- stage 1: technicals on everything ----
        print("stage 1: scoring…", flush=True)
        stage1 = []
        frames = {}
        for sym, d in px.items():
            try:
                fr = build_stock_frame(d)
            except Exception:
                continue
            if fr.empty:
                continue
            row = fr.iloc[-1]
            if not bool(row["passes"]):
                continue
            frames[sym] = fr
            stage1.append((sym, float(row["score"]), row))
        stage1.sort(key=lambda x: -x[1])
        print(f"  passed filters: {len(stage1)}", flush=True)
        payload["passed_count"] = len(stage1)

        # ---- stage 2: enrich the leaders ----
        print(f"stage 2: enriching top {STAGE2_N}…", flush=True)
        picks = []
        for sym, score, row in stage1[:STAGE2_N]:
            mcap, raw_sector = fundamentals(sym)
            if mcap is not None and mcap < PARAMS["min_market_cap"]:
                continue
            sector = OVERRIDES.get(sym) or YF_SECTOR_MAP.get(raw_sector, raw_sector)
            ss = sect.get(sector)
            sector_pts = (float(ss.iloc[-1]) if ss is not None and len(ss)
                          else PARAMS["sector_neutral"])

            dly = delivery_pct(sym)
            vr = float(row["vol_ratio"]) if np.isfinite(row["vol_ratio"]) else np.nan
            vol_pts = float(row["volume"])
            if dly is not None and dly > 50 and vol_pts < 20:
                vol_pts = min(20.0, vol_pts + PARAMS["vol_bonus"])
                quality = f"delivery {dly:.0f}%"
            elif dly is None and np.isfinite(vr) and vr >= PARAMS["vol_proxy"]:
                quality = f"proxy {vr:.1f}x"
            else:
                quality = "—"

            comps = {c: float(row[c]) for c in
                     ["volume", "squeeze", "sector", "trend", "candle",
                      "near_high", "momentum"]}
            comps["volume"] = vol_pts
            comps["sector"] = sector_pts
            raw = sum(comps.values())
            final = raw * 100.0 / 102.0

            atr_v = float(row["atr"])
            entry = round(float(row["entry_trigger"]), 2)
            stop = round(entry - stop_mult * atr_v, 2)
            t1 = round(entry + PARAMS["atr_t1"] * atr_v, 2)
            risk = entry - stop

            picks.append({
                "symbol": sym, "name": names.get(sym, sym), "sector": sector or "Unmapped",
                "score": round(final, 1), "components": comps,
                "price": js(row["close"]), "prev_high": js(row["high"]),
                "entry": entry, "stop": stop, "target1": t1,
                "risk_pct": round(risk / entry * 100, 2),
                "t1_pct": round((t1 - entry) / entry * 100, 2),
                "rr": round((t1 - entry) / risk, 2) if risk > 0 else None,
                "trail_ema9": js(row["ema9"]),
                "rsi": js(row["rsi"]), "vol_ratio": js(vr),
                "delivery_pct": js(dly), "volume_quality": quality,
                "atr_pct": js(row["atr_pct"]),
                "dist_52w_pct": js(row["dist_52w_pct"]),
                "at_multiyear_high": js(row["at_multiyear_high"]),
                "market_cap_cr": round(mcap / 1e7) if mcap else None,
                "turnover_cr": js(row["turnover_20d"] / 1e7),
                "breakout": js(row["breakout"]),
                "squeeze": js(row["_squeeze_flag"]),
            })

        picks.sort(key=lambda x: -x["score"])
        payload["picks"] = picks
        payload["status"] = "ok" if picks else "empty"

        for p in picks[:TOP_CHARTS]:
            d = px[p["symbol"]].tail(CHART_BARS)
            fr = frames[p["symbol"]].tail(CHART_BARS)
            payload["charts"][p["symbol"]] = {
                "dates": [str(x.date()) for x in d.index],
                "open": [round(float(x), 2) for x in d["Open"]],
                "high": [round(float(x), 2) for x in d["High"]],
                "low": [round(float(x), 2) for x in d["Low"]],
                "close": [round(float(x), 2) for x in d["Close"]],
                "ema20": [round(float(x), 2) for x in fr["ema20"]],
                "ema50": [round(float(x), 2) for x in fr["ema50"]],
            }

        if picks:
            csv = os.path.join(HIST, f"scan_{dt.date.today()}.csv")
            pd.DataFrame([{k: v for k, v in p.items() if k != "components"}
                          for p in picks]).to_csv(csv, index=False)
            payload["csv_file"] = os.path.relpath(csv, BASE)
        print(f"picks: {len(picks)} | top "
              f"{picks[0]['score'] if picks else 0}", flush=True)

    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    # -------- paper log: resolve open positions, then append tonight's picks --
    # This runs in BOTH modes. Sit-out days still write the log so that when
    # we look back in 90 days we have parallel records for trade and research
    # regimes, gathered under identical conditions.
    try:
        paper_log.ensure_log()
        stats = paper_log.resolve_open(px, dt.date.today())
        print(f"paper log: resolved {stats['resolved']} closed, "
              f"{stats['filled']} filled today, {stats['still_open']} still open, "
              f"{stats['expired']} expired", flush=True)

        state_for_log = payload["regime"].get("state", "unknown")
        picks_for_log = payload.get("picks") or []
        # Attach ATR since resolve_open needs it and the payload does not carry it.
        for p in picks_for_log:
            if p.get("atr") is None:
                p["atr"] = None  # left None; not needed post-fill, only pre-fill trigger matters
        added = paper_log.append_picks(picks_for_log, state_for_log, dt.date.today())
        print(f"paper log: appended {added} new picks (regime={state_for_log})", flush=True)

        payload["paper_log"] = {
            "appended": added,
            **stats,
            "summary": paper_log.summary(),
        }
    except Exception as e:
        print(f"paper log update failed: {e}", flush=True)
        payload["paper_log_error"] = str(e)

    with open(LATEST, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), default=str)
    print(f"wrote {LATEST} ({os.path.getsize(LATEST)/1024:.0f} KB) "
          f"status={payload['status']}", flush=True)
    return 1 if payload["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
