"""
test_v4.py — mean reversion, trade plans, and regime switching.
Run alongside test_engine.py; both must pass before any v4 result is read.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from strategy_mr import build_mr_frame, MR_PARAMS, MR_PLAN, MR_COMPONENT_MAX
from features import build_stock_frame
from regime import compute_regime, compute_breadth
from backtest import simulate_trade, BREAKOUT_PLAN
from switched import run_switched, compare_arms

PASS, FAIL = [], []
def check(n, c, d=""):
    (PASS if c else FAIL).append(n)
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))


def pullback(n=400, seed=1, depth=0.12, collapse=False, reversal=True,
             downtrend=False):
    """Uptrend then a pullback, with an optional reversal bar at the end."""
    r = np.random.default_rng(seed)
    drift = -.6 if downtrend else .5
    up = np.cumsum(r.normal(drift, 0.8, n - 25)) + (900 if downtrend else 400)
    peak = up[-1]
    dip = np.linspace(peak, peak * (0.5 if collapse else 1 - depth), 24)
    c = np.maximum(np.concatenate([up, dip, [dip[-1]]]), 20)
    o = c * (1 - r.uniform(0, .006, n))
    h = np.maximum(c, o) * (1 + r.uniform(.002, .01, n))
    l = np.minimum(c, o) * (1 - r.uniform(.002, .01, n))
    v = r.uniform(6e5, 1.2e6, n)
    df = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v},
                      index=pd.bdate_range('2023-01-02', periods=n))
    if reversal:
        pv = df.iloc[-2]['High']
        for col, m in [('Close', 1.02), ('High', 1.025), ('Low', .995), ('Open', 1.0)]:
            df.iloc[-1, df.columns.get_loc(col)] = pv * m
        df.iloc[-1, df.columns.get_loc('Volume')] = df['Volume'].iloc[-21:-1].mean() * 3
    return df


def plain(n=400, seed=0):
    r = np.random.default_rng(seed)
    c = np.maximum(np.cumsum(r.normal(.15, 1.0, n)) + 500, 50)
    o = c * (1 - r.uniform(0, .008, n))
    h = np.maximum(c, o) * (1 + r.uniform(.002, .012, n))
    l = np.minimum(c, o) * (1 - r.uniform(.002, .012, n))
    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c,
                         'Volume': r.uniform(6e5, 1.2e6, n)},
                        index=pd.bdate_range('2023-01-02', periods=n))


print("\n=== 1. MEAN REVERSION IS CAUSAL ===")
d = pullback(500, seed=3)
full = build_mr_frame(d)
cols = ['rsi', 'atr', 'ema20', 'ema200', 'ema200_slope_pct', 'structurally_healthy',
        'vol_ratio', 'dist_below_20ema_pct', 'down_streak', 'ret_20d_pct',
        'dist_above_low20_pct', 'swing_low', 'entry_trigger', 'reversal_bar',
        'rsi_depth', 'reversal', 'stretch', 'trend_intact', 'support',
        'vol_climax', 'score', 'passes']
worst = 0.0
for tp in [300, 350, 400, 450, 480]:
    tr = build_mr_frame(d.iloc[:tp + 1]); dt = d.index[tp]
    for col in cols:
        a, b = full.at[dt, col], tr.at[dt, col]
        if isinstance(a, (bool, np.bool_)):
            dev = 0.0 if bool(a) == bool(b) else 1.0
        elif pd.isna(a) and pd.isna(b):
            dev = 0.0
        elif pd.isna(a) or pd.isna(b):
            dev = 1.0
        else:
            dev = abs(float(a) - float(b)) / max(abs(float(a)), 1e-9)
        worst = max(worst, dev)
check("every MR feature and score is causal", worst < 1e-9, f"worst dev {worst:.1e}")

print("\n=== 2. FALLING-KNIFE GUARDS ===")
good = build_mr_frame(pullback()).iloc[-1]
check("valid pullback + reversal passes", bool(good['passes']),
      f"rsi={good['rsi']:.0f} score={good['score']:.0f}")
check("passes while BELOW the 200 EMA (range behaviour)",
      not bool(good['above_200ema']) and bool(good['structurally_healthy']),
      f"200ema slope {good['ema200_slope_pct']:+.1f}%")
check("oversold without a reversal bar is rejected",
      not bool(build_mr_frame(pullback(reversal=False)).iloc[-1]['f_reversal']))
check("collapse rejected as freefall, not bought as a dip",
      not bool(build_mr_frame(pullback(collapse=True)).iloc[-1]['f_not_freefall']))
dn = build_mr_frame(pullback(downtrend=True)).iloc[-1]
check("genuine downtrend rejected by structural health",
      not bool(dn['f_uptrend']), f"200ema slope {dn['ema200_slope_pct']:+.1f}%")
check("non-oversold stock rejected",
      not bool(build_mr_frame(plain(400, 9)).iloc[-1]['f_oversold']))
n_ok = sum(1 for s in range(40) if bool(build_mr_frame(pullback(seed=s)).iloc[-1]['passes']))
check("guards are not so tight the strategy never trades", n_ok > 15,
      f"{n_ok}/40 qualify")

print("\n=== 3. MR SCORE BUDGETS ===")
viol = 0
for s in range(30):
    fr = build_mr_frame(pullback(seed=s, depth=float(np.random.default_rng(s).uniform(.05, .2))))
    for comp, mx in MR_COMPONENT_MAX.items():
        if fr[comp].max() > mx + 1e-9 or fr[comp].min() < -1e-9:
            viol += 1
    if fr['score'].max() > 100 + 1e-6:
        viol += 1
check("all MR component budgets respected", viol == 0, f"violations={viol}")

print("\n=== 4. TRADE PLANS DIFFER BY STRATEGY ===")
idx = pd.bdate_range('2024-01-01', periods=7)
p = pd.DataFrame({'Open': [100, 100.5, 105, 111, 117, 119, 118],
                  'High': [100, 102, 108, 115, 120, 122, 119],
                  'Low': [99, 100.5, 104, 110, 116, 118, 114],
                  'Close': [100, 101, 106, 112, 118, 120, 116],
                  'Volume': [1e6] * 7}, index=idx)
e = pd.Series([90.0] * 7, index=idx)
bo = simulate_trade(p, e, 0, 101.0, 5.0, 1.5, plan=BREAKOUT_PLAN)
mr = simulate_trade(p, e, 0, 101.0, 5.0, 1.5, plan=MR_PLAN)
check("breakout targets 2xATR and books half", bo['t1_px'] == 111.0,
      f"T1={bo['t1_px']} exit={bo['exit_reason']}")
check("MR targets a closer 1.5xATR and exits fully",
      mr['t1_px'] == 108.5 and mr['exit_reason'] == 'target',
      f"T1={mr['t1_px']} exit={mr['exit_reason']}")
check("MR does not trail", MR_PLAN.get('trail') is None)
check("MR time stop is shorter",
      MR_PLAN['max_holding_days'] < BREAKOUT_PLAN['max_holding_days'],
      f"{MR_PLAN['max_holding_days']}d vs {BREAKOUT_PLAN['max_holding_days']}d")
loose = simulate_trade(p, e, 0, 101.0, 5.0, 3.0, plan=MR_PLAN)
tight = simulate_trade(p, e, 0, 101.0, 5.0, 3.0, plan=MR_PLAN, stop_floor=99.0)
check("swing-low floor tightens a too-wide ATR stop",
      tight['stop_px'] > loose['stop_px'], f"{loose['stop_px']} -> {tight['stop_px']}")

print("\n=== 5. THREE-STATE REGIME SWITCHING ===")
n = 300
ix = pd.bdate_range('2023-01-02', periods=n)
up = pd.Series(np.linspace(18000, 24000, n), index=ix)
dwn = pd.Series(np.linspace(24000, 18000, n), index=ix)
wu = pd.DataFrame({f's{i}': np.linspace(100, 200, n) for i in range(20)}, index=ix)
wd = pd.DataFrame({f's{i}': np.linspace(200, 100, n) for i in range(20)}, index=ix)
bu, bd = compute_breadth(wu), compute_breadth(wd)
check("trending -> breakout",
      compute_regime(up, bu)['strategy'].iloc[-1] == 'breakout')
check("bear -> cash (not merely a tighter stop)",
      compute_regime(dwn, bd)['strategy'].iloc[-1] == 'cash')
check("range -> mean reversion",
      compute_regime(up, bd)['strategy'].iloc[-1] == 'mean_reversion')
check("no breadth -> never claims bear",
      (compute_regime(dwn, None)['state'] != 'bear').all())

print("\n=== 6. SWITCHED PORTFOLIO ===")
syms = [f"Z{i:02d}" for i in range(25)]
px = {s: (pullback(500, seed=60 + i) if i % 3 == 0 else plain(500, 60 + i))
      for i, s in enumerate(syms)}
bof = {s: build_stock_frame(d) for s, d in px.items()}
mrf = {s: build_mr_frame(d) for s, d in px.items()}
nif = pd.Series(np.linspace(19000, 23000, 500), index=px[syms[0]].index)
reg = compute_regime(nif, compute_breadth(pd.DataFrame({s: px[s]['Close'] for s in syms})))
tr = run_switched(bof, mrf, px, reg, start='2023-09-01', top_n=5)
check("switched backtest runs", not tr.empty, f"n={len(tr)}")
if not tr.empty:
    check("entry strictly after signal", bool((tr['entry_i'] > tr['signal_i']).all()))
    check("trades tagged with their strategy", 'strategy' in tr.columns,
          str(tr['strategy'].value_counts().to_dict()))
    cash_days = set(reg.index[reg['state'] == 'bear'])
    check("no trades signalled on cash days",
          len(set(tr['signal_date']) & cash_days) == 0)
arms = compare_arms(bof, mrf, px, reg, start='2023-09-01')
check("all four comparison arms produced", len(arms) == 4)

print("\n" + "=" * 60)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
for f in FAIL:
    print("   -", f)
if FAIL:
    sys.exit(1)
print("v4 correctness tests passed.")
