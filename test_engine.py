"""
test_engine.py — correctness tests

The engine is validated on synthetic data where the correct answer is known
by construction. That is deliberately more useful than running it once on
real data: a backtest that silently peeks at the future still produces a
beautiful equity curve, and you cannot tell by looking at it.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from features import build_stock_frame, compute_features, compute_scores, PARAMS
from regime import compute_regime, compute_breadth
from backtest import simulate_trade, run_backtest, summarise, component_attribution

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def synth(n=600, seed=0, base=500.0, drift=0.03, vol=1.2):
    rng = np.random.default_rng(seed)
    c = np.cumsum(rng.normal(drift, vol, n)) + base
    c = np.maximum(c, 25.0)
    o = c * (1 - rng.uniform(0, .01, n))
    hi = np.maximum(c, o) * (1 + rng.uniform(.002, .015, n))
    lo = np.minimum(c, o) * (1 - rng.uniform(.002, .015, n))
    v = rng.uniform(5e5, 1.5e6, n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c, "Volume": v},
                        index=idx)


# ===========================================================================
print("\n=== 1. CAUSALITY — the test that everything else depends on ===")
# ===========================================================================
# If a feature computed on the full series differs from the same feature
# computed on data truncated at date T, the backtest is seeing the future.

df = synth(700, seed=7)
full = build_stock_frame(df)

cols_to_check = ["atr", "atr_pct", "vol_ratio", "ema20", "ema50", "ema200", "ema9",
                 "rsi", "hi_52w", "dist_52w_pct", "close_pos", "upper_wick_frac",
                 "bbw", "bbw_min10", "bbw_min20", "bbw_base", "bbw_compression",
                 "atrp_compression", "base_hi", "base_lo", "breakout",
                 "entry_trigger", "trigger_gap_pct", "hi_alltime", "turnover_20d"]
score_cols = ["volume", "squeeze", "trend", "candle", "near_high", "momentum",
              "score_raw", "score", "passes"]

rng = np.random.default_rng(3)
test_positions = sorted(rng.choice(np.arange(300, 700), size=60, replace=False))
max_dev = {}
for tp in test_positions:
    cut = df.iloc[: tp + 1]
    trunc = build_stock_frame(cut)
    d = df.index[tp]
    for col in cols_to_check + score_cols:
        a, b = full.at[d, col], trunc.at[d, col]
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            dev = 0.0 if bool(a) == bool(b) else 1.0
        elif pd.isna(a) and pd.isna(b):
            dev = 0.0
        elif pd.isna(a) or pd.isna(b):
            dev = 1.0
        else:
            denom = max(abs(float(a)), 1e-9)
            dev = abs(float(a) - float(b)) / denom
        max_dev[col] = max(max_dev.get(col, 0.0), dev)

worst = max(max_dev.values())
worst_col = max(max_dev, key=max_dev.get)
check("every feature is causal (full == truncated)", worst < 1e-9,
      f"worst deviation {worst:.2e} on '{worst_col}' across {len(test_positions)} dates")

# expanding max must never see a future high
c = df["Close"].to_numpy()
ath = full["hi_alltime"].to_numpy()
manual = np.maximum.accumulate(df["High"].to_numpy())
check("all-time high uses expanding (not global) max", np.allclose(ath, manual))

# an obvious lookahead bug would be caught by this
bad = df["High"].max()
check("52w high is not the global max", full["hi_52w"].iloc[300] < bad or
      abs(full["hi_52w"].iloc[300] - bad) > 1e-9)


# ===========================================================================
print("\n=== 2. v3 FILTERS ===")
# ===========================================================================
d2 = synth(400, seed=11).copy()

# --- upper circuit: close exactly at high, HEAVY volume (old rule let this
#     through; the v3 rule must reject it) ---
d2c = d2.copy()
i = len(d2c) - 1
d2c.iloc[i, d2c.columns.get_loc("High")] = d2c.iloc[i]["Close"]
d2c.iloc[i, d2c.columns.get_loc("Volume")] = d2c["Volume"].iloc[-21:-1].mean() * 5
fc = build_stock_frame(d2c)
check("circuit lock rejected even on heavy volume",
      bool(fc["is_circuit_locked"].iloc[-1]) and not bool(fc["passes"].iloc[-1]))

# --- volume anomaly cap ---
d2v = d2.copy()
d2v.iloc[-1, d2v.columns.get_loc("Volume")] = d2v["Volume"].iloc[-21:-1].mean() * 40
fv = build_stock_frame(d2v)
check("volume anomaly (40x) rejected by hard filter",
      not bool(fv["f_vol_sane"].iloc[-1]) and not bool(fv["passes"].iloc[-1]),
      f"vol_ratio={fv['vol_ratio'].iloc[-1]:.1f}")
check("volume anomaly also scores 0, not 20", fv["volume"].iloc[-1] == 0.0)

# a 3x surge must still be rewarded
d2s = d2.copy()
d2s.iloc[-1, d2s.columns.get_loc("Volume")] = d2s["Volume"].iloc[-21:-1].mean() * 3
fs = build_stock_frame(d2s)
check("legitimate 3x surge still scores full marks",
      fs["volume"].iloc[-1] == 20.0, f"pts={fs['volume'].iloc[-1]}")

# --- rejection wick ---
d2w = d2.copy()
j = len(d2w) - 1
op, cl = d2w.iloc[j]["Open"], d2w.iloc[j]["Close"]
body_top = max(op, cl)
body_low = min(op, cl)
# construct a shooting star: tiny body, huge upper wick, close near body top
d2w.iloc[j, d2w.columns.get_loc("Low")] = body_low * 0.999
d2w.iloc[j, d2w.columns.get_loc("High")] = body_top * 1.05
fw = build_stock_frame(d2w)
check("shooting star scores 0 on candle despite strong-looking close",
      fw["candle"].iloc[-1] == 0.0 and fw["upper_wick_frac"].iloc[-1] > 0.5,
      f"wick_frac={fw['upper_wick_frac'].iloc[-1]:.2f}")

# --- trigger gap guard ---
d2g = d2.copy()
d2g.iloc[-1, d2g.columns.get_loc("High")] = d2g.iloc[-1]["Close"] * 1.09
fg = build_stock_frame(d2g)
check("trigger 9% above close is rejected",
      not bool(fg["f_trigger_near"].iloc[-1]),
      f"gap={fg['trigger_gap_pct'].iloc[-1]:.1f}%")
d2g2 = d2.copy()
d2g2.iloc[-1, d2g2.columns.get_loc("High")] = d2g2.iloc[-1]["Close"] * 1.015
fg2 = build_stock_frame(d2g2)
check("trigger 2% above close is accepted", bool(fg2["f_trigger_near"].iloc[-1]))

# --- multi-year high bonus ---
d2a = synth(900, seed=5, drift=0.30).copy()
fa = build_stock_frame(d2a)
at_ath = fa["at_multiyear_high"]
check("multi-year-high bonus reachable and capped at 12",
      fa["near_high"].max() <= 12.0 and at_ath.any(),
      f"max near_high={fa['near_high'].max()}, ath days={int(at_ath.sum())}")

# --- score bounds across a wide random sweep ---
viol = 0
maxes = {"volume": 20, "squeeze": 20, "sector": 15, "trend": 15,
         "candle": 10, "near_high": 12, "momentum": 10}
for s in range(40):
    fr = build_stock_frame(synth(400, seed=100 + s, drift=float(rng.uniform(-.2, .3))))
    for comp, mx in maxes.items():
        if fr[comp].max() > mx + 1e-9 or fr[comp].min() < -1e-9:
            viol += 1
    if fr["score"].max() > 100 + 1e-6:
        viol += 1
check("all component budgets respected over 40 random series", viol == 0,
      f"violations={viol}")


# ===========================================================================
print("\n=== 3. EXECUTION SIMULATION ===")
# ===========================================================================
def make_path(closes, highs=None, lows=None, opens=None):
    n = len(closes)
    closes = np.array(closes, float)
    opens = np.array(opens, float) if opens is not None else closes.copy()
    highs = np.array(highs, float) if highs is not None else np.maximum(opens, closes)
    lows = np.array(lows, float) if lows is not None else np.minimum(opens, closes)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                         "Close": closes, "Volume": [1e6] * n}, index=idx)

ATRV = 10.0

# --- (a) trigger never touched -> no trade ---
p = make_path([100, 100, 100, 100], highs=[100, 100.5, 100.5, 100.5])
t = simulate_trade(p, pd.Series([100.0] * 4, index=p.index), 0,
                   trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("no fill when the trigger is never reached", t is None)

# --- (b) clean fill at trigger ---
p = make_path([100, 103, 104, 105], highs=[100, 105, 106, 107],
              lows=[99, 100, 103, 104], opens=[100, 100.2, 103, 104])
t = simulate_trade(p, pd.Series([90.0] * 4, index=p.index), 0,
                   trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("fills at the trigger when the bar trades through it",
      t is not None and abs(t["entry_px"] - 101.0) < 1e-6)

# --- (c) gap beyond tolerance -> skipped, not chased ---
p = make_path([100, 110, 112], highs=[100, 113, 114], lows=[99, 109, 111],
              opens=[100, 110, 112])
t = simulate_trade(p, pd.Series([90.0] * 3, index=p.index), 0,
                   trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("gap far above the trigger is skipped, not filled at the open", t is None)

# --- (d) small gap -> fills at the open, not the (better) trigger ---
p = make_path([100, 102, 104, 106], highs=[100, 104, 106, 108],
              lows=[99, 101.5, 103, 105], opens=[100, 102, 104, 106])
t = simulate_trade(p, pd.Series([90.0] * 4, index=p.index), 0,
                   trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("small gap fills at the open (worse than the trigger)",
      t is not None and abs(t["entry_px"] - 102.0) < 1e-6,
      f"entry={t['entry_px'] if t else None}")

# --- (e) stop hit -> R is about -1 before costs, worse after ---
p = make_path([100] + [101, 90, 88], highs=[100, 102, 101, 90],
              lows=[99, 100.5, 84, 86], opens=[100, 100.5, 100, 89])
t = simulate_trade(p, pd.Series([200.0] * 4, index=p.index), 0,
                   trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("stop-out lands near -1R and never better than -1R",
      t is not None and -1.6 < t["r_multiple"] < -0.95,
      f"R={t['r_multiple']:.3f} reason={t['exit_reason']}")

# --- (f) ambiguous bar (spans both stop and target) -> stop assumed ---
p = make_path([100, 101, 105], highs=[100, 102, 130], lows=[99, 100.5, 80],
              opens=[100, 100.5, 101])
t = simulate_trade(p, pd.Series([200.0] * 3, index=p.index), 0,
                   trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("bar covering both stop and target is booked as a stop",
      t is not None and t["r_multiple"] < 0 and "stop" in t["exit_reason"],
      f"R={t['r_multiple']:.3f} reason={t['exit_reason']}")

# --- (g) T1 then trail ---
# after T1 the stop moves to breakeven (101), so the trail must be tested on
# a bar that closes below the 9 EMA WITHOUT dipping through breakeven —
# otherwise the engine correctly books the breakeven stop instead
closes = [100, 101, 118, 125, 130, 128, 120]
highs  = [100, 102, 125, 128, 132, 130, 129]
lows   = [99, 100.5, 115, 122, 127, 125, 119]
opens  = [100, 100.5, 116, 124, 129, 129, 128]
p = make_path(closes, highs, lows, opens)
e9 = pd.Series([90, 90, 100, 110, 118, 124, 126], index=p.index, dtype=float)
t = simulate_trade(p, e9, 0, trigger=101.0, atr_val=ATRV, stop_mult=1.5)
check("T1 booked then remainder trails out on a close below the 9 EMA",
      t is not None and t["t1_hit"] and t["exit_reason"] == "trail_9ema"
      and t["r_multiple"] > 0.5,
      f"R={t['r_multiple']:.2f} reason={t['exit_reason']}")

# --- (h) costs must reduce the result ---
free = simulate_trade(p, e9, 0, 101.0, ATRV, 1.5,
                      costs={"charge_pct_per_side": 0, "slippage_pct_per_side": 0})
check("costs and slippage reduce returns", t["r_multiple"] < free["r_multiple"],
      f"with costs {t['r_multiple']:.3f} vs frictionless {free['r_multiple']:.3f}")

# --- (i) time stop ---
flat_c = [100, 101] + [102] * 30
flat_h = [100, 102] + [103] * 30
flat_l = [99, 100.5] + [101] * 30
p = make_path(flat_c, flat_h, flat_l, [100, 100.5] + [102] * 30)
t = simulate_trade(p, pd.Series([50.0] * 32, index=p.index), 0, 101.0, ATRV, 1.5)
check("a position that goes nowhere is closed by the time stop",
      t is not None and t["exit_reason"] == "time_stop"
      and t["holding_days"] <= 15, f"held {t['holding_days']}d")

# --- (j) tighter stop really is tighter ---
# entry 101, ATR 10 -> normal stop 86, strict stop 91.
# The low must land between them for the two to differ.
p = make_path([100, 101, 92, 93], highs=[100, 102, 101, 95],
              lows=[99, 100.5, 88, 90], opens=[100, 100.5, 100, 92])
t15 = simulate_trade(p, pd.Series([200.0] * 4, index=p.index), 0, 101.0, ATRV, 1.5)
t10 = simulate_trade(p, pd.Series([200.0] * 4, index=p.index), 0, 101.0, ATRV, 1.0)
check("strict regime stop triggers where the normal stop survives",
      t10["exit_reason"] == "stop" and t15["exit_reason"] != "stop",
      f"1.0x -> {t10['exit_reason']}, 1.5x -> {t15['exit_reason']}")


# ===========================================================================
print("\n=== 4. REGIME + BREADTH ===")
# ===========================================================================
n = 300
idx = pd.bdate_range("2023-01-02", periods=n)
up = pd.Series(np.linspace(18000, 24000, n), index=idx)
dn = pd.Series(np.linspace(24000, 18000, n), index=idx)

# breadth strong / weak
wide_up = pd.DataFrame({f"s{i}": np.linspace(100, 200, n) for i in range(20)}, index=idx)
wide_dn = pd.DataFrame({f"s{i}": np.linspace(200, 100, n) for i in range(20)}, index=idx)
b_up = compute_breadth(wide_up)
b_dn = compute_breadth(wide_dn)
check("breadth ~100% when everything trends up", b_up.iloc[-1] > 95,
      f"{b_up.iloc[-1]:.0f}%")
check("breadth ~0% when everything trends down", b_dn.iloc[-1] < 5,
      f"{b_dn.iloc[-1]:.0f}%")

r_trend = compute_regime(up, b_up)
r_bear = compute_regime(dn, b_dn)
r_range = compute_regime(up, b_dn)     # index up, breadth weak — divergence
check("index up + breadth>55 -> trending, runs breakout",
      r_trend["state"].iloc[-1] == "trending"
      and r_trend["strategy"].iloc[-1] == "breakout")
check("index down + breadth<35 -> bear, goes to CASH",
      r_bear["state"].iloc[-1] == "bear"
      and r_bear["strategy"].iloc[-1] == "cash")
check("divergence -> range, switches to mean reversion",
      r_range["state"].iloc[-1] == "range"
      and r_range["strategy"].iloc[-1] == "mean_reversion")
check("without breadth the bear/cash call is never made",
      (compute_regime(up, None)["state"] != "bear").all()
      and (compute_regime(dn, None)["state"] != "bear").all())


# ===========================================================================
print("\n=== 5. PORTFOLIO BACKTEST + BASELINE ===")
# ===========================================================================
syms = [f"T{i:02d}" for i in range(30)]
px_data, frames = {}, {}
for k, s in enumerate(syms):
    d = synth(500, seed=200 + k, drift=float(rng.uniform(-0.1, 0.35)))
    px_data[s] = d
    frames[s] = build_stock_frame(d)

nifty = pd.Series(np.linspace(19000, 23000, 500),
                  index=px_data[syms[0]].index)
closes_wide = pd.DataFrame({s: px_data[s]["Close"] for s in syms})
breadth = compute_breadth(closes_wide)
reg = compute_regime(nifty, breadth)

tr = run_backtest(frames, px_data, reg, start="2022-09-01", top_n=5, max_concurrent=5)
check("backtest produces trades", not tr.empty, f"n={len(tr)}")

if not tr.empty:
    # no trade may be entered before its own signal
    check("entry always strictly after the signal bar",
          bool((tr["entry_i"] > tr["signal_i"]).all()))
    check("exit never before entry", bool((tr["exit_i"] >= tr["entry_i"]).all()))
    # no overlapping positions in the same symbol
    ov = 0
    for sym, g in tr.groupby("symbol"):
        g = g.sort_values("entry_date")
        prev_exit = None
        for _, row in g.iterrows():
            if prev_exit is not None and row["signal_date"] <= prev_exit:
                ov += 1
            prev_exit = row["exit_date"]
    check("no overlapping positions in the same symbol", ov == 0, f"overlaps={ov}")

    st = summarise(tr, "model")
    check("summary statistics computed", st["n"] > 0 and "expectancy_R" in st)
    print(f"      -> {st['n']} trades, hit {st['hit_rate_pct']}%, "
          f"expectancy {st['expectancy_R']}R (SE {st['expectancy_SE']})")

    rnd = run_backtest(frames, px_data, reg, start="2022-09-01", top_n=5,
                       max_concurrent=5, selection="random")
    check("random baseline runs for comparison", not rnd.empty, f"n={len(rnd)}")

    att = component_attribution(tr, min_bucket=10)
    check("component attribution table produced", not att.empty,
          f"{att['component'].nunique()} components bucketed")

print("\n" + "=" * 62)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("All correctness tests passed.")
