"""
app.py — v5 mobile viewer
=========================

Two states that matter:

  TRADEABLE   regime is trending -> today's picks
  SIT OUT     regime is range or bear -> no picks, and the screen says why

The sit-out screen is not an error or a degraded mode. It is the product.
Backtested 2017-2026, breakout-always measured -0.165R while
breakout-only-when-trending measured +0.095R. The entire measured edge comes
from not trading on the other days, so the app has to make a blank screen
feel like the system working rather than the system broken.
"""

import json
import os
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import paper_log as pl

BASE = os.path.dirname(os.path.abspath(__file__))
LATEST = os.path.join(BASE, "data", "latest.json")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

st.set_page_config(page_title="Swing Screener", page_icon="▲",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
  html, body, [class*="css"] { font-family:'JetBrains Mono', ui-monospace, monospace !important; }
  .stApp { background:#07090c; }
  .block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:820px; }

  .hd { border-left:3px solid #39ff88; padding-left:12px; margin-bottom:2px; }
  .hd h1 { font-size:1.15rem; letter-spacing:.14em; margin:0; color:#e6edf3; font-weight:700; }
  .hd .s { color:#6e7681; font-size:.7rem; letter-spacing:.05em; }

  .banner { padding:12px 15px; border-radius:3px; margin:13px 0;
            font-size:.78rem; line-height:1.6; border-left:4px solid; }
  .b-go   { background:#0d1f14; border-color:#39ff88; color:#7ee8a8; }
  .b-sit  { background:#151a21; border-color:#58a6ff; color:#9fc6f0; }
  .b-na   { background:#17191c; border-color:#6e7681; color:#9da5b4; }
  .b-warn { background:#241f10; border-color:#e8c33d; color:#e8c33d; }
  .b-research { background:#22161a; border-color:#c74560; color:#f4a3b8;
                position:sticky; top:0; z-index:100; padding:14px 16px; }
  .b-research b { color:#ffcbd7; letter-spacing:.08em; }
  .card-research { border-left:3px solid #6e7681 !important; opacity:0.92;
                   position:relative; }
  .card-research::before { content:"PAPER"; position:absolute; top:8px; right:12px;
    font-size:.55rem; letter-spacing:.15em; color:#8b949e;
    background:#161b22; padding:2px 6px; border-radius:2px; }

  .sitout { text-align:center; padding:34px 20px; background:#0b0f14;
            border:1px solid #1c2128; border-radius:4px; margin:18px 0; }
  .sitout .big { font-size:1.5rem; color:#58a6ff; letter-spacing:.16em;
                 font-weight:700; margin-bottom:10px; }
  .sitout .sub { color:#8b949e; font-size:.8rem; line-height:1.75; }

  .card { background:#0b0f14; border:1px solid #1c2128; border-left:3px solid #39ff88;
          border-radius:3px; padding:13px 15px; margin-bottom:11px; }
  .card .top { display:flex; justify-content:space-between; align-items:baseline; }
  .card .sym { font-size:1rem; font-weight:700; color:#e6edf3; letter-spacing:.05em; }
  .card .sc  { font-size:1rem; font-weight:700; color:#39ff88; }
  .card .co  { font-size:.68rem; color:#6e7681; margin:2px 0 9px; }
  .lvl { display:flex; gap:7px; margin:9px 0 4px; }
  .lvl div { flex:1; background:#11161d; border-radius:3px; padding:7px 4px;
             text-align:center; border:1px solid #21262d; }
  .lvl .k { font-size:.56rem; letter-spacing:.09em; color:#6e7681; display:block; }
  .lvl .v { font-size:.83rem; font-weight:700; display:block; margin-top:2px; }
  .g{color:#39ff88} .r{color:#ff6b6b} .y{color:#e8c33d} .b{color:#58a6ff}
  .meta { font-size:.65rem; color:#6e7681; line-height:1.65; margin-top:7px; }
  .chip { display:inline-block; padding:2px 7px; margin:2px 4px 2px 0; border-radius:2px;
          font-size:.6rem; background:#11161d; color:#8b949e; border:1px solid #21262d; }
  .chip-hot { background:#0d1f14; color:#39ff88; border-color:#1a4d2e; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hd">
  <h1>SWING SCREENER</h1>
  <div class="s">NSE breakout · trades only in a trending regime · paper-test stage</div>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load(mtime: float):
    with open(LATEST) as f:
        return json.load(f)


if not os.path.exists(LATEST):
    st.error("No scan results yet. Trigger the workflow from the Actions tab "
             "on GitHub, or wait for tonight's scheduled run.")
    st.stop()

p = load(os.path.getmtime(LATEST))

if p.get("status") == "error":
    st.error(f"The last scan failed: {p.get('error')}")
    st.caption(f"Attempted {p.get('generated_utc')}")
    st.stop()

gen = dt.datetime.fromisoformat(p["generated_utc"])
if gen.tzinfo is None:
    gen = gen.replace(tzinfo=dt.timezone.utc)
gen_ist = gen.astimezone(IST)
age_h = (dt.datetime.now(dt.timezone.utc) - gen).total_seconds() / 3600

st.caption(f"Last scan: {gen_ist:%a %d %b, %H:%M} IST · "
           f"{p.get('universe_count', 0):,} stocks screened")

if age_h > 30:
    st.markdown(f"<div class='banner b-warn'>⚠ These results are "
                f"{age_h/24:.1f} days old — the nightly job may have failed. "
                f"Check the Actions tab.</div>", unsafe_allow_html=True)

reg = p.get("regime") or {}
state = reg.get("state", "unknown")
breadth = reg.get("breadth_pct")

# ---------------------------------------------------------------- gate --
# Three states now, not two:
#   TRADEABLE  regime trending, mode=trade         -> green banner + normal picks
#   RESEARCH   regime not trending, picks present  -> red banner + picks tagged PAPER
#   SIT OUT    regime not trending, no picks       -> blue static screen
#
# The research state is new. It preserves the entire measured edge (no trades
# taken) while still letting the user see what the algorithm sees, and every
# pick is logged so the paper log can eventually settle whether the gate was
# right on this specific day.
mode = p.get("mode", "trade")
research_mode = (mode == "research") or (p.get("status") in ("no_trade", "research"))
picks_all = p.get("picks", [])

# ---- SIT OUT: research mode with no picks (legacy no_trade, or empty scan) ----
if research_mode and not picks_all:
    label = {"range": "RANGE-BOUND", "bear": "BEAR",
             "unknown": "REGIME UNKNOWN"}.get(state, state.upper())
    detail = ""
    if reg.get("available"):
        detail = (f"Nifty 50 {reg.get('nifty_close', 0):,.0f} "
                  f"({reg.get('gap_pct', 0):+.2f}% vs its 20 EMA)"
                  + (f" · breadth {breadth:.0f}%" if breadth is not None else ""))
    else:
        detail = "Index data unavailable, so no regime call could be made."

    st.markdown(f"""
<div class="sitout">
  <div class="big">SIT OUT</div>
  <div class="sub">
    Regime: <b>{label}</b><br>{detail}<br><br>
    Breakouts measured <b>negative</b> expectancy outside a trending regime
    across 2017&#8209;2026. No picks are issued today.<br><br>
    <span style="color:#6e7681">A blank screen is the system working, not failing.
    Not trading on days like this is where the entire measured edge comes from.</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='banner b-sit'>
<b>What flips this back on</b><br>
Nifty 50 back above its 20 EMA <b>and</b> market breadth above 55%.
The scan re-checks every weekday evening — no action needed from you.
</div>
""", unsafe_allow_html=True)
    st.caption("Educational tool, paper-testing stage. Not investment advice.")
    st.stop()

# ------------------------------------------------------------- RESEARCH mode --
if research_mode:
    label = {"range": "RANGE-BOUND", "bear": "BEAR",
             "unknown": "REGIME UNKNOWN"}.get(state, state.upper())
    detail = ""
    if reg.get("available"):
        detail = (f"Nifty 50 {reg.get('nifty_close', 0):,.0f} "
                  f"({reg.get('gap_pct', 0):+.2f}% vs 20 EMA)"
                  + (f" · breadth {breadth:.0f}%" if breadth is not None else ""))
    st.markdown(f"""
<div class="banner b-research">
<b>🔬 RESEARCH MODE — DO NOT TRADE</b><br>
Regime: <b>{label}</b>. {detail}<br>
Breakouts measured <b>negative</b> expectancy outside a trending regime
across 2017–2026. Picks below are being <b>logged as paper trades</b> so we
can measure in 90 days whether the gate should stay closed.<br><br>
<span style="color:#e8b4c2; font-size:.72rem">
Treat these like watching a match you did not bet on — informative, not actionable.
Acting on research picks corrupts the very data that would let us relax the gate.
</span></div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------- TRADEABLE mode --
else:
    breadth_txt = f" · breadth {breadth:.0f}%" if breadth is not None else ""
    st.markdown(f"""
<div class='banner b-go'><b>REGIME: TRENDING — breakouts active</b><br>
Nifty 50 {reg.get('nifty_close', 0):,.0f} ({reg.get('gap_pct', 0):+.2f}% above its
20 EMA){breadth_txt} · stops at {reg.get('atr_stop_mult', 1.5)}×ATR
</div>""", unsafe_allow_html=True)
if not picks_all:
    st.warning("Regime is trending, but nothing cleared the filters today. "
               "Thin breadth within a rising index — no trade is a position.")
    st.stop()

c1, c2 = st.columns(2)
top_n = c1.selectbox("Picks", [3, 5, 8], index=1)
min_score = c2.selectbox("Min score", [0, 50, 60, 70], index=2)

picks = [x for x in picks_all if (x.get("score") or 0) >= min_score][:top_n]
if not picks:
    best = max((x.get("score") or 0) for x in picks_all)
    st.warning(f"Nothing scored ≥ {min_score}; best today is {best:.0f}. "
               f"Showing the top {top_n} as watchlist only.")
    picks = picks_all[:top_n]

st.markdown(f"**TOP {len(picks)}** · {p.get('passed_count', 0)} passed filters")

MAXES = {"volume": 20, "squeeze": 20, "sector": 15, "trend": 15,
         "candle": 10, "near_high": 12, "momentum": 10}
charts = p.get("charts", {})
card_cls = "card card-research" if research_mode else "card"


def fmt(v, d=2, dash="—"):
    return dash if v is None else f"{v:,.{d}f}"


for x in picks:
    sym = x["symbol"]
    comp = x.get("components", {})
    chips = "".join(
        f"<span class='chip {'chip-hot' if (comp.get(k) or 0) >= mx*.75 else ''}'>"
        f"{k.upper()} {comp.get(k, 0):.0f}/{mx}</span>" for k, mx in MAXES.items())
    dly = x.get("delivery_pct")

    st.markdown(f"""
<div class="{card_cls}">
  <div class="top"><span class="sym">{sym}</span>
    <span class="sc">{x.get('score', 0):.0f}<span style="font-size:.62rem;color:#6e7681">/100</span></span></div>
  <div class="co">{x.get('name','')} · {x.get('sector','—')}</div>
  <div class="lvl">
    <div><span class="k">ENTRY</span><span class="v g">{fmt(x.get('entry'))}</span></div>
    <div><span class="k">STOP</span><span class="v r">{fmt(x.get('stop'))}</span></div>
    <div><span class="k">T1</span><span class="v y">{fmt(x.get('target1'))}</span></div>
    <div><span class="k">R:R</span><span class="v b">{fmt(x.get('rr'),2)}</span></div>
  </div>
  <div class="meta">
    Trigger 0.5% above prev high {fmt(x.get('prev_high'))} · risk {fmt(x.get('risk_pct'))}% ·
    T1 +{fmt(x.get('t1_pct'))}%<br>
    <b style="color:#8b949e">At T1</b> book 50%, stop to breakeven ·
    <b style="color:#8b949e">Rest</b> trails the 9 EMA on a closing basis (now {fmt(x.get('trail_ema9'))})<br>
    RSI {fmt(x.get('rsi'),1)} · VOL {fmt(x.get('vol_ratio'),1)}× ·
    DLY {fmt(dly,0,'n/a')}{'%' if dly is not None else ''} ·
    ATR {fmt(x.get('atr_pct'),1)}% · →52wH {fmt(x.get('dist_52w_pct'),1)}%<br>{chips}
  </div>
</div>
""", unsafe_allow_html=True)

    ch = charts.get(sym)
    if ch:
        with st.expander(f"Chart · {sym}"):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=ch["dates"], open=ch["open"], high=ch["high"], low=ch["low"],
                close=ch["close"], name=sym,
                increasing_line_color="#39ff88", decreasing_line_color="#ff5c5c"))
            for key, col, nm in [("ema20", "#58a6ff", "EMA20"), ("ema50", "#e8c33d", "EMA50")]:
                if key in ch:
                    fig.add_trace(go.Scatter(x=ch["dates"], y=ch[key],
                                             line=dict(width=1.2, color=col), name=nm))
            for lvl, nm, col in [(x.get("entry"), "ENTRY", "#39ff88"),
                                 (x.get("stop"), "STOP", "#ff5c5c"),
                                 (x.get("target1"), "T1", "#e8c33d")]:
                if lvl:
                    fig.add_hline(y=lvl, line_dash="dot", line_width=1, line_color=col,
                                  annotation_text=nm, annotation_position="right",
                                  annotation_font_color=col, annotation_font_size=9)
            fig.update_layout(height=330, margin=dict(l=4, r=4, t=6, b=4),
                              paper_bgcolor="#07090c", plot_bgcolor="#0b0f14",
                              font=dict(family="JetBrains Mono, monospace",
                                        color="#8b949e", size=9),
                              xaxis_rangeslider_visible=False, showlegend=False,
                              xaxis=dict(gridcolor="#161b22"),
                              yaxis=dict(gridcolor="#161b22"))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

st.divider()
df = pd.DataFrame([{k: v for k, v in x.items() if k != "components"} for x in picks_all])
st.download_button("⬇ Download full ranked list (CSV)", df.to_csv(index=False),
                   file_name=f"swing_scan_{gen_ist:%Y-%m-%d}.csv",
                   mime="text/csv", width='stretch')

st.markdown("""
<div class='banner b-warn'>
<b>Paper-testing stage.</b> Measured expectancy in a trending regime was
+0.095R with a standard error of 0.093 — the confidence interval includes
zero, and survivorship bias in the backtest means the true figure is lower.
Log these signals on paper for three to six months before committing capital.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------- scoreboard --
st.divider()
with st.expander("📊 Paper log scoreboard", expanded=False):
    try:
        summ = pl.summary()
        st.caption(f"{summ['total_rows']} total signals · {summ['open']} open · "
                   f"{summ['closed']} closed · "
                   f"{summ['expired']} expired without fill")
        by = summ.get("by_regime", {})
        if not by:
            st.info("Not enough closed trades yet. The scoreboard fills in as "
                    "positions resolve — expect meaningful numbers after ~90 days.")
        else:
            rows = []
            for label, stats in sorted(by.items()):
                rows.append({
                    "regime": ("🟢 TRADE (trending)" if label == "trade"
                               else "🔬 PAPER (sit-out)"),
                    "n trades": stats["n"],
                    "hit rate": f"{stats['hit_rate_pct']}%",
                    "expectancy": f"{stats['expectancy_R']:+.3f}R",
                    "SE": (f"±{stats['expectancy_SE']:.3f}"
                           if stats.get("expectancy_SE") else "—"),
                    "total R": f"{stats['total_R']:+.1f}",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            t = by.get("trade"); pp = by.get("paper")
            if t and pp and t["n"] >= 30 and pp["n"] >= 30:
                gap = t["expectancy_R"] - pp["expectancy_R"]
                se = t.get("expectancy_SE") or 0.2
                if abs(gap) > 2 * se:
                    verdict = (
                        "**Gate confirmed** — trending picks meaningfully "
                        "outperform sit-out picks. Keep the gate closed."
                        if gap > 0 else
                        "**Reconsider the gate** — sit-out picks are "
                        "outperforming trending picks. Worth investigating "
                        "before the next model iteration.")
                else:
                    verdict = "Difference is within noise so far. Keep logging."
                st.info(f"Trade vs Paper gap: {gap:+.3f}R — {verdict}")
            else:
                st.caption("Need at least 30 closed trades in each regime for "
                           "a meaningful comparison.")
    except Exception as e:
        st.error(f"Scoreboard error: {e}")

st.caption("Entry levels are triggers, not market orders. If price never crosses "
           "the entry there is no trade; if it gaps far above at the open, skip it. "
           "Not investment advice.")
