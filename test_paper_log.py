"""
test_paper_log.py — lifecycle and correctness

Covers:
 * append + dedup + strict CSV round-trip
 * causal fill (never before signal_date)
 * gap-through-trigger handled correctly (fill at open or skip)
 * gap-through-stop books at open, not stop
 * bar-covering-both books stop (conservative tie-break, matches backtest)
 * T1 then trail-out sequence
 * time stop
 * costs applied both sides
 * summary segregates trade vs paper
 * multi-day walk-forward: a stock that goes days without a bar in the log
   still gets its outcome computed correctly once bars land
"""
import sys, os, shutil, tempfile, datetime as dt
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

import paper_log

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def use_temp_log():
    tmp = tempfile.mkdtemp()
    paper_log.LOG_PATH = os.path.join(tmp, "paper_log.csv")
    paper_log.ensure_log()
    return tmp


def bars(rows, start="2024-01-02"):
    """rows = list of (o,h,l,c) — volume filled in."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(
        {"Open":[r[0] for r in rows], "High":[r[1] for r in rows],
         "Low":[r[2] for r in rows],  "Close":[r[3] for r in rows],
         "Volume":[1e6]*len(rows)}, index=idx)


def pick(symbol, trigger, stop, target, atr=5.0, name="Co"):
    return {
        "symbol": symbol, "name": name, "sector": "Test",
        "score": 70.0, "components": {"volume":16},
        "prev_high": trigger/1.005, "atr": atr, "atr_pct": 2.0,
        "entry": trigger, "stop": stop, "target1": target,
        "price": trigger/1.005,
    }


print("\n=== 1. APPEND + DEDUP ===")
use_temp_log()
n1 = paper_log.append_picks([pick("ACME", 105, 97.5, 115)], "trending", dt.date(2024,1,1))
check("first pick lands", n1 == 1)
n2 = paper_log.append_picks([pick("ACME", 106, 98, 116)], "trending", dt.date(2024,1,2))
check("duplicate symbol NOT re-added while original is open", n2 == 0,
      "would double-book in live trading")
n3 = paper_log.append_picks([pick("WIDGET", 200, 190, 220)], "range", dt.date(2024,1,2))
check("new symbol added regardless", n3 == 1)
df = paper_log.read_all()
check("regime label reflects state", set(df["regime"]) == {"trade", "paper"},
      str(df["regime"].tolist()))


print("\n=== 2. CAUSAL FILL ===")
use_temp_log()
paper_log.append_picks([pick("CAUS", 105, 97.5, 115)], "trending", dt.date(2024,1,10))
# provide bars INCLUDING bars before the signal date; if resolve peeks at them we've broken causality
data = {"CAUS": bars([
    (110, 112, 108, 111),   # 2024-01-02 — pre-signal, would trigger a fill if peeked
    (110, 112, 108, 111),
    (110, 112, 108, 111),
    (110, 112, 108, 111),
    (110, 112, 108, 111),
    (110, 112, 108, 111),   # 2024-01-09 — still pre-signal
    (110, 112, 108, 111),   # 2024-01-10 — signal date itself, still not eligible
    (100, 104, 99, 103),    # 2024-01-11 — first eligible bar; trigger NOT crossed
    (104, 106, 103, 105),   # 2024-01-12 — trigger 105 crossed on the high
], start="2024-01-02")}
stats = paper_log.resolve_open(data, dt.date(2024,1,12))
row = paper_log.read_all().iloc[0]
# order valid ONE day; must have expired after 2024-01-11 (first post-signal bar didn't fill)
check("order that misses on day+1 expires (does not silently backfill)",
      row["status"] == "expired" and row["exit_reason"] == "unfilled",
      f"status={row['status']} reason={row['exit_reason']}")


print("\n=== 3. GAP HANDLING ===")
# 3a. small gap fills at the open
use_temp_log()
paper_log.append_picks([pick("GAPA", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
data = {"GAPA": bars([
    (100.5, 102, 99, 101),      # signal day
    (101,   103, 100, 102),     # T+1: opens at 101, but trigger is 100 -> gap fill at 101
    (102, 108, 101, 107),
    (108, 112, 107, 111),
], start="2024-01-10")}
paper_log.resolve_open(data, dt.date(2024,1,15))
row = paper_log.read_all().iloc[0]
check("small gap fills at the OPEN (worse than trigger)",
      row["entry_px"] and abs(float(row["entry_px"]) - 101.0) < 0.01,
      f"entry_px={row['entry_px']}")

# 3b. big gap up expires the order rather than chasing
use_temp_log()
paper_log.append_picks([pick("GAPB", 100, 92.5, 110)], "trending", dt.date(2024,1,10))
data = {"GAPB": bars([
    (100, 101, 99, 100.5),   # signal
    (105, 108, 104, 106),    # T+1: opens 5% above trigger -> refuse to chase
], start="2024-01-10")}
paper_log.resolve_open(data, dt.date(2024,1,12))
row = paper_log.read_all().iloc[0]
check("gap > 2% above trigger expires (does not chase)",
      row["status"] == "expired" and row["exit_reason"] == "gap_too_far",
      f"status={row['status']} reason={row['exit_reason']}")


print("\n=== 4. TIE BREAKS ===")
# 4a. Bar that covers both stop and target -> STOP (conservative, matches backtest)
use_temp_log()
paper_log.append_picks([pick("TIE", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
data = {"TIE": bars([
    (100, 101, 99, 100.5),         # signal
    (100.5, 101, 100, 100.8),      # entry bar: fills at 100
    (101, 112, 88, 105),           # covers 92.5 stop AND 110 target — daily bar can't say order
], start="2024-01-10")}
paper_log.resolve_open(data, dt.date(2024,1,15))
row = paper_log.read_all().iloc[0]
check("bar covering both stop and target books the STOP",
      row["status"] == "closed" and row["exit_reason"] in ("stop","stop_after_t1"),
      f"reason={row['exit_reason']} R={row['r_multiple']}")

# 4b. Gap straight through stop fills at OPEN not stop
use_temp_log()
paper_log.append_picks([pick("GAPD", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
data = {"GAPD": bars([
    (100, 101, 99, 100.5),
    (100.5, 101, 100, 100.8),      # entry at 100
    (85, 88, 83, 86),              # gaps below 92.5 stop; fills at 85, not 92.5
], start="2024-01-10")}
paper_log.resolve_open(data, dt.date(2024,1,15))
row = paper_log.read_all().iloc[0]
check("gap-down through stop fills at OPEN (worse than stop)",
      row["exit_px"] and float(row["exit_px"]) <= 85.5,
      f"exit_px={row['exit_px']} R={row['r_multiple']}")


print("\n=== 5. T1 BOOKED + TIME STOP ON REMAINDER ===")
use_temp_log()
paper_log.append_picks([pick("WIN", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
data = {"WIN": bars([
    (100, 101, 99, 100.5),
    (100.5, 101, 100, 100.8),      # entry at 100
    (101, 112, 100, 111),          # hits T1 (110), remainder to breakeven (100)
    (111, 112, 100.5, 108),        # stays above breakeven
    (108, 109, 101, 105),
    (105, 106, 101, 103),
    (103, 104, 101, 102),          # remainder time-stops eventually
    (102, 103, 101, 101.5),
    (101.5, 102, 100.5, 101),
    (101, 101.5, 100.5, 100.8),
    (100.8, 101, 100.5, 100.5),
    (100.5, 101, 100, 100.5),
    (100.5, 101, 100, 100.5),
    (100.5, 101, 100, 100.5),
    (100.5, 101, 100, 100.5),
    (100.5, 101, 100, 100.5),      # day 15 from entry -> time stop hits
    (100.5, 101, 100, 100.5),
], start="2024-01-10")}
paper_log.resolve_open(data, dt.date(2024,2,10))
row = paper_log.read_all().iloc[0]
check("T1 booked at target",
      bool(row["t1_hit"]) and abs(float(row["target1_px"]) - 110) < 0.1,
      f"t1_hit={row['t1_hit']}")
check("remainder time-stops (position closes)",
      row["status"] == "closed" and row["exit_reason"] in ("time_stop","stop_after_t1"),
      f"reason={row['exit_reason']} R={row['r_multiple']} pnl%={row['pnl_pct']}")
check("R multiple is positive (T1 was hit)",
      float(row["r_multiple"]) > 0, f"R={row['r_multiple']}")


print("\n=== 6. COSTS ARE APPLIED ===")
# The same trade with fee-free costs should show a HIGHER R than with real costs.
use_temp_log()
paper_log.append_picks([pick("COST", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
data = {"COST": bars([
    (100, 101, 99, 100.5),
    (100.5, 101, 100, 100.8),
    (101, 112, 100.5, 110.5),     # target hit cleanly
    (110.5, 111, 100, 100),        # stops at breakeven
], start="2024-01-10")}
paper_log.resolve_open(data, dt.date(2024,1,20))
row_with_costs = paper_log.read_all().iloc[0]
r_with = float(row_with_costs["r_multiple"])

# rerun with zero costs
use_temp_log()
orig_cost, orig_slip = paper_log.COST_PER_SIDE, paper_log.SLIP_PER_SIDE
paper_log.COST_PER_SIDE = 0.0; paper_log.SLIP_PER_SIDE = 0.0
paper_log.append_picks([pick("COST", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
paper_log.resolve_open(data, dt.date(2024,1,20))
row_no_costs = paper_log.read_all().iloc[0]
r_no = float(row_no_costs["r_multiple"])
paper_log.COST_PER_SIDE, paper_log.SLIP_PER_SIDE = orig_cost, orig_slip

check("R with real costs is worse than R with zero costs",
      r_with < r_no, f"with_costs={r_with:.3f} zero_costs={r_no:.3f}")


print("\n=== 7. SUMMARY SEGREGATES TRADE vs PAPER ===")
use_temp_log()
# feed one clean winner as 'trade' and one clean loser as 'paper'
paper_log.append_picks([pick("WINR", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
paper_log.append_picks([pick("LOSR", 200, 185, 230, atr=10.0)], "range", dt.date(2024,1,10))
data = {
    "WINR": bars(
        [(100,101,99,100.5),(100.5,101,100,100.8),(101,115,101,112),
         # remainder rides down and eventually hits the breakeven stop
         (112,113,108,110),(110,111,105,106),(106,107,100.5,101),(101,102,99,100)],
        start="2024-01-10"),
    "LOSR": bars([(200,201,199,200.5),(200.5,201,200,200.8),(201,202,180,182),(182,183,180,181)],
                 start="2024-01-10"),
}
paper_log.resolve_open(data, dt.date(2024,1,25))
summ = paper_log.summary()
check("summary reports both regimes", set(summ["by_regime"]) == {"trade","paper"},
      str(list(summ["by_regime"])))
check("trade winner has positive R", summ["by_regime"]["trade"]["expectancy_R"] > 0)
check("paper loser has negative R", summ["by_regime"]["paper"]["expectancy_R"] < 0)


print("\n=== 8. MULTI-DAY GAP IN DATA ===")
# A pick sits for a week with no scan; when scan runs, log advances the state correctly
use_temp_log()
paper_log.append_picks([pick("SLOW", 100, 92.5, 110, atr=5.0)], "trending", dt.date(2024,1,10))
# first resolve with only signal bar available -> stays open
data_partial = {"SLOW": bars([(100,101,99,100.5)], start="2024-01-10")}
stats = paper_log.resolve_open(data_partial, dt.date(2024,1,10))
row = paper_log.read_all().iloc[0]
check("no forward bars -> stays open", row["status"] == "open",
      f"status={row['status']}")

# now resolve with the full sequence; should catch up
data_full = {"SLOW": bars([
    (100,101,99,100.5),(100.5,101,100,100.8),(101,113,100,112),(112,113,110,111)],
    start="2024-01-10")}
paper_log.resolve_open(data_full, dt.date(2024,1,20))
row = paper_log.read_all().iloc[0]
# Correct outcome: T1 hit, remainder on breakeven stop. Position may still be
# open (that's fine — it will resolve on a future scan run) or fully closed.
check("catches up when later bars arrive",
      row["status"] in ("closed", "filled_open") and bool(row["t1_hit"]),
      f"status={row['status']} t1_hit={row['t1_hit']}")


print("\n" + "="*58)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
for f in FAIL: print("   -", f)
if FAIL: sys.exit(1)
print("paper log tests passed.")
