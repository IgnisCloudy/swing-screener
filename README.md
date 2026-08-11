# Swing Screener v5 — breakout + regime gate

## What the backtest actually established

Ten years, 2017–2026, split into two independent eras with the second opened
only once:

| | Develop 2017–22 | Verify 2022–26 |
|---|---|---|
| Breakout in a **trending** regime | +0.011R | **+0.095R** |
| Mean reversion in a **range** | −0.234R | −0.242R |
| Breakout **always** (v3) | −0.093R | −0.165R |

Two conclusions, both acted on:

**Mean reversion is removed.** It lost money in both eras at almost identical
magnitude. Not tuned — removed. The structural reasoning was sound; the
market did not pay for it, twice.

**The regime gate is the edge.** Breakout-always measured −0.165R in the
verify era. Breakout-only-when-trending measured +0.095R. That 0.26R gap
comes from *not trading*, not from picking better. So v5 issues **zero picks**
on non-trending days — not a shorter list, not wider stops. None.

## Honest calibration

+0.095R carries SE 0.093, so the confidence interval **includes zero**. The
backtest universe is today's listed stocks, so survivorship bias means the
true figure is lower still. This is a paper-trading candidate, not a system
to size up on. The app states this on every screen.

## Files

| File | Purpose |
|---|---|
| `features.py` | Causal features, filters, scores. Single source of truth. |
| `regime.py` | Three-state regime from Nifty 20 EMA + breadth. |
| `backtest.py` | Execution simulation and statistics. |
| `switched.py` | Regime-switched portfolio runner and comparison arms. |
| `strategy_mr.py` | Mean reversion — **retained for reference, not traded**. |
| `run_backtest.py` / `run_v4.py` | Validation protocols. |
| `scan_job_v5.py` | Nightly live scan → `data/latest.json`. |
| `app.py` | Phone viewer, including the sit-out screen. |
| `test_engine.py` / `test_v4.py` | 58 correctness tests. |
| `METHODOLOGY.md` | **Read before trusting any number.** |

## Running

```bash
pip install -r requirements.txt
python test_engine.py && python test_v4.py     # both must pass
python fetch_universe.py                       # once, from an Indian connection
python build_sector_map.py                     # once
python run_v4.py --years 10                    # full validation
python scan_job_v5.py                          # a live scan
```

## Known bug fixed in v5

The v4 sensitivity sweep for `target_atr_mult` and `max_holding_days` was
inert: `MR_PLAN` was a module-level dict built at import time, so mutating
`MR_PARAMS` changed nothing and every value returned identical results. It
looked like a flat plateau when it had never varied. Now built by
`mr_plan()` per call. The MR *entry* filters were tested correctly; the MR
*exit* parameters never were — worth knowing if you ever revisit that code.

## What to do now

1. Deploy v5. On most days it will say SIT OUT. That is the design.
2. Log every trending-day signal on paper for 3–6 months.
3. Compare live results to the +0.095R expectation. Forward data has no
   survivorship bias, which makes it the only clean test remaining.
4. Only then consider real capital, and size so a full stop-out costs a
   fixed small share of it.

---

# v5.1 — research mode + paper log

## What changed and why

v5 hid picks on non-trending days. The upside: no temptation to trade.
The downside: you had no way to *see* what the algorithm was doing on those
days, and no way to eventually settle whether the gate was right on any
specific day. v5.1 opens that window without opening the trade.

**Three regime states now:**

| State | Screen | Log behaviour |
|---|---|---|
| Trending | Green banner + normal picks | Logged as `trade` |
| Range / Bear (with picks) | **Red banner: RESEARCH MODE — DO NOT TRADE**, picks shown with grey borders and PAPER tag | Logged as `paper` |
| Any regime with no picks | Blue SIT OUT screen | Nothing to log |

The research banner is deliberately visually distinct — not green, not
comfortable-looking, sticky at the top of the screen. Design intent: reading
it should feel like watching a match you did not bet on.

## Paper log (`paper_log.py`)

Every pick — trade *or* paper — is appended to `data/paper_log.csv` with
entry, stop, target, all component scores and the regime it was issued in.
Every subsequent scan walks the open rows forward against the latest bar,
resolving each one exactly the way `backtest.simulate_trade` would: gap-fill
at the open, stop-first tie-break, T1 booked and remainder to breakeven,
9-EMA trail (breakout) or straight target (research), 15-day time stop, and
full 0.15% costs both sides.

Nothing about this requires you to do anything. It runs unattended.

## The scoreboard

The app's Paper Log Scoreboard expander shows measured expectancy of the two
regime labels side by side. After ~30 closed trades in each it prints one of
three verdicts:

* **Gate confirmed** — trending picks meaningfully outperform paper picks.
  Leave the gate closed.
* **Reconsider the gate** — paper picks outperformed trending picks. Worth
  investigating before the next model change.
* **Within noise so far** — keep logging.

The comparison uses standard errors, not raw differences. A 0.05R "edge" on
15 trades either side means nothing; the app says so.

## The deal that makes this work

The design assumes you **do not act on the research picks**. If you take a
few that "look good," those become the ones you remember, the log records
outcomes that don't match your actual behaviour, and the scoreboard
becomes garbage.

Read the research picks. Do not trade them. Come back in 90 days.
