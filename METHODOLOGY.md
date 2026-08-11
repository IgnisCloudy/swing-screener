# Backtest methodology

This document explains how the screener is validated and, more importantly,
how the validation is designed to avoid fooling you. Read section 6 before
you act on any output.

---

## 1. The two ways a backtest lies

Almost every backtest that looks good and then fails live has done one of two
things.

**It saw the future.** A single careless line — using a full-series maximum,
a centred rolling window, or today's market cap to filter a signal from two
years ago — is enough to produce a beautiful, entirely fictional equity
curve. You cannot detect this by looking at the results, because a
lookahead bug makes the results look *better*, not stranger.

**It was fitted to noise.** With seven components, several thresholds each,
and one historical sample, you can always find a combination that performed
well. That combination describes the sample, not the market. The more
parameters you tune and the more variants you try, the more certain it is
that the best one is an accident.

Everything below exists to make those two failures visible.

---

## 2. Structural defences against lookahead

### One scoring implementation
`features.py` is imported by both the live screener and the backtest. There
is no second copy of the scoring logic. If they were separate, the backtest
would validate code you are not actually trading — a common and quietly
fatal arrangement.

### Every feature is causal by construction
Each column produced in `features.py` at row *t* depends only on data at or
before *t*. Exponential averages use `adjust=False`. Highs use
`rolling(252).max()` and `expanding().max()`, never a global maximum.
Volume ratios benchmark today against `shift(1).rolling(20)` so that today's
own volume is excluded from the average it is being compared to.

### Causality is asserted, not assumed
`test_engine.py` computes the full feature and score set on the complete
series, then recomputes it on the series truncated at each of 60 randomly
chosen dates, and asserts the values match. Current result: **maximum
deviation 0.00e+00 across every feature and every component score.** If any
future-peeking is introduced later, this test fails immediately.

### The null test
Stage 0 of the run shuffles each stock's daily returns — destroying all time
structure while preserving the return distribution — and reruns the entire
pipeline. Any genuine signal disappears. Any lookahead survives, because a
bug that reads tomorrow's price still reads it after shuffling.

On shuffled data the engine returns roughly **−0.45R** expectancy: no edge,
just costs. That is the correct answer. If this number ever comes back
positive, stop and find the bug before reading anything else.

---

## 3. Execution assumptions

A backtest's honesty lives here more than anywhere else.

| Situation | This engine's assumption | Why |
|---|---|---|
| Signal timing | Generated from the close of day T, executed on T+1 | You cannot trade on a close you have not yet seen |
| Order life | Valid one day; cancelled if unfilled | A stale breakout trigger is not the same trade three days later |
| Price gaps above the trigger | Fill at the open, not the trigger | You do not get yesterday's price |
| Gap more than 2% above the trigger | Skip the trade entirely | Chasing a gap widens the stop past the risk you sized for |
| Bar spans both stop and target | **Book the stop** | Daily bars cannot resolve sequence; assuming the favourable order is the single most common way a backtest flatters itself |
| Price gaps through the stop | Fill at the open, not the stop | Stops do not honour their own price on a gap |
| Costs | 0.15% per side | STT both sides, stamp duty, exchange and SEBI charges, GST'd brokerage |
| Slippage | 0.15% per side | Stop-triggered entries fill worse than limit orders |
| Position never resolves | Time stop at 15 sessions | Prevents a "swing" trade quietly becoming a one-year hold |

Total friction is about 0.6% round trip. On a trade targeting roughly 6% at
T1, friction consumes around a tenth of the gross move. Setting costs to
zero improves every number in the report and makes none of them true.

### Results are reported in R-multiples

R = (exit − entry) ÷ (entry − stop). One R is one unit of the risk you chose.

This normalises across a ₹40 stock and a ₹40,000 stock, across volatile and
quiet regimes, and it removes position sizing from the measurement entirely.
**Expectancy in R is the single number that matters**: the average R per
trade. Positive expectancy after costs is a system; a high hit rate with
negative expectancy is not.

---

## 4. Structural defences against overfitting

### Test hypotheses, not parameter combinations
Searching a seven-dimensional weight space against one sample is a guaranteed
overfit. This protocol never does that. Instead it asks a much lower-
dimensional question, once per component:

> Sorted by this component's score, do the trades separate?

That is `component_attribution()`. It buckets every trade by the points a
component awarded and reports mean R per bucket. If mean R rises with the
score, the component carries information. If it is flat, the component is
noise — regardless of how sensible the reasoning behind it sounded when it
was written. If it is *inverted*, the component is actively harmful.

This produces a defensible reason to change a weight, rather than a number
that happened to work.

### Time-based holdout, opened once
The final months of data are never touched until stage 5. Every decision —
which components matter, which thresholds to prefer — is made on the training
period. Random train/test splitting would be wrong here: shuffling dates
leaks information across time in a way that flatters any trend-following
system.

### Walk-forward
Sequential out-of-sample windows. Consistency across several windows is
evidence that a result is not one lucky quarter. A system that produces
+0.3R, +0.28R, +0.31R across three windows is far more believable than one
producing +2.1R, −0.4R, +0.2R with the same average.

### Read plateaus, not peaks
The sensitivity sweep varies one threshold at a time and reports expectancy
at each value.

- A parameter that performs well across a **contiguous band** of values is
  describing something real about the market.
- A parameter that **spikes at one value** and collapses either side of it
  has been fitted to noise in this particular sample. It will not survive.

Always choose the middle of a plateau over the top of a spike, even when the
spike shows a better number. This is the most useful discipline in the whole
document.

### The baseline that matters
Stage 2 runs an identical backtest that selects **at random from the same
filtered set**. This separates two claims that are easy to confuse:

- Model beats random → the **scoring** adds information.
- Model ≈ random, both positive → the **hard filters** are doing the work
  and the 100-point model is decoration.
- Both negative → the setup itself does not work, filters included.

Each is a completely different conclusion, and without this baseline you
cannot tell them apart.

### Sample size and the standard error
Every summary reports the standard error of expectancy. A difference between
two configurations smaller than about two standard errors is not a result.
Component buckets thinner than 30 trades are flagged `reliable=False` — a
mean R computed from nine trades tells you nothing at all.

Note also the multiple-comparisons problem: sweeping four parameters across
five values each means twenty tests, and roughly one in twenty will look
significant by chance at conventional thresholds. Treat a single standout
value with suspicion unless it sits on a plateau.

---

## 5. Known limitations — read before trusting any number

**Survivorship bias.** The universe file lists stocks trading *today*.
Companies delisted during the test window are absent, so the backtest never
buys a breakout in a company that subsequently collapsed. This biases results
optimistically, and it cannot be fixed with free data. Point-in-time
constituent history is a paid product. Treat this as the largest single
caveat on any positive result.

**Data quality.** Free daily data for NSE symbols contains occasional bad
ticks and split-adjustment errors. A mis-adjusted split can manufacture a
fake breakout or a fake stop-out.

**Delivery percentage is unavailable historically.** The volume component's
delivery bonus falls back to the ≥2.5× surge proxy throughout the backtest.
So the backtest validates the proxy, not the delivery signal — and the live
screener, when it can reach NSE, is running a slightly different volume rule
from the one tested.

**Market cap is approximated.** Historical market cap is reconstructed as
today's shares outstanding × historical price. This ignores issuance,
buybacks and splits over the window. It is an approximation chosen because
the alternative — applying today's market cap to a 2022 signal — would be
outright lookahead.

**Sector scores default to neutral.** Without a `data/sector_map.csv`, every
stock receives the neutral 7.5/15. The sector component is therefore
untested unless you supply that mapping, and its attribution row should be
ignored if it is constant.

**The earnings filter is not implemented.** The review is right that holding
a breakout into an earnings print is a binary event that overrides the
technicals. It is omitted because free historical earnings dates are
incomplete and patchy going back several years — an unreliable filter applied
inconsistently would corrupt the backtest more than leaving it out. The
honest options are a paid corporate-actions feed, or applying the rule
manually at the point of trading: **before entering any pick, check whether
results are due within five sessions, and skip it if so.** That takes ten
seconds per trade and is what I would actually recommend.

---

## 6. How to read the output

Work through the report in this order.

**1. Null test.** Expectancy must not be positive. If it is, stop — there is
a lookahead bug and nothing else in the report means anything.

**2. Holdout expectancy.** This is the headline. Positive after costs means
the system had an edge in a period it never saw. Compare it to training
expectancy: some decay is normal and expected; a collapse to zero or negative
means the in-sample result was fitted rather than found.

**3. The random baseline.** Determines whether your 100-point model is doing
anything, or whether the hard filters alone are responsible.

**4. Component attribution.** The action list. Components whose mean R rises
with their score deserve their weight or more. Flat components should have
their weight reduced. Inverted components should be removed or reversed.
Ignore any bucket flagged unreliable.

**5. Score deciles.** If the top score bucket does not out-earn the bottom
bucket, the ranking has no predictive content even if individual components
do — which usually means the weighting is wrong rather than the components.

**6. Regime split.** Tests whether the market-regime gate earns its place.
If `strict` and `normal` regimes show similar expectancy, the gate is
costing you trades for nothing. If `strict` is much worse, consider not
trading in that regime at all rather than merely tightening stops.

**7. Walk-forward.** Consistency check. Look at the spread across windows,
not the average.

### What would justify trading this

- Null test flat or negative ✔
- Holdout expectancy positive after costs, with n ≥ 100 trades
- Holdout within roughly one standard error of training expectancy
- Model expectancy meaningfully above the random baseline
- At least three of seven components showing monotonic attribution
- Walk-forward positive in the majority of windows
- Max drawdown in R that you could actually sit through

### What would justify not trading it

Any of: positive null test; holdout expectancy near zero or negative; model
indistinguishable from random; fewer than about 50 holdout trades; or
walk-forward results swinging wildly in sign.

**A negative result here is worth more than the tool.** Finding out over a
weekend that these parameters have no edge is enormously cheaper than finding
out over six months with real money.

---

## 7. What to do after the first run

Resist the urge to tune everything until the numbers improve — that is
precisely the overfitting the protocol exists to prevent.

The disciplined loop is:

1. Run once. Read the attribution table.
2. Change **at most two** things, each justified by an attribution row, not
   by a hunch and not by a number that happened to look good.
3. Re-run the training period only.
4. Open the holdout again **only after several such iterations**, and
   understand that each additional look at it erodes its value as an
   independent test. Three or four looks and it is no longer a holdout.
5. When you run out of holdout credibility, the honest next step is forward
   testing on paper — new data that did not exist when the rules were set.

Paper-trade in parallel regardless. Live conditions add slippage, partial
fills, and the psychological difficulty of taking a stop, none of which any
backtest captures.
