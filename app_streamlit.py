"""
Web version:  streamlit run app_streamlit.py
Same pipeline as run.py, with charts.
"""

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai
from run import StrategyError, test_strategy, understand
from strategy_spec import describe_rules

st.set_page_config(page_title="Strategy in Plain English", page_icon="📈", layout="wide")

EXAMPLES = [
    "Buy SPY when RSI(2) drops below 10 while price is above the 200 day moving average. Sell when the close goes above the 5 day moving average.",
    "Buy Bitcoin when it breaks above its 20-day high, stop 2 ATR below, take profit at 3R.",
    "Golden cross on the Nasdaq: buy QQQ when the 50 EMA crosses above the 200 EMA, sell when it crosses back below.",
    "Short gold when the close falls below the lower Bollinger band (20), exit after 5 days, 3% stop.",
    "Buy Apple when MACD crosses above its signal line and RSI(14) is under 60, exit when MACD crosses back below. 15 years.",
]

GRADE_COLOUR = {"Robust": "#1a9e5c", "Fragile": "#d98a00", "Doesn't hold up": "#d1435b"}


def md(text: str) -> str:
    """Stop Streamlit treating $ signs in AI text as maths formulas."""
    return text.replace("$", "\\$")


def pct(x, signed=True):
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def price_chart(report) -> go.Figure:
    trades = report.result.trades
    close = report.result.prices["Close"]
    long = report.strategy.direction == "long"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name=report.strategy.symbol,
                             line=dict(color="#8a94a6", width=1)))
    fig.add_trace(go.Scatter(
        x=[t.entry_date for t in trades], y=[t.entry_price for t in trades], mode="markers", name="Entry",
        marker=dict(symbol="triangle-up" if long else "triangle-down", size=9, color="#2f6fed")))
    fig.add_trace(go.Scatter(
        x=[t.exit_date for t in trades], y=[t.exit_price for t in trades], mode="markers", name="Exit",
        marker=dict(symbol="x", size=8, color=["#1a9e5c" if t.return_pct > 0 else "#d1435b" for t in trades]),
        text=[f"{t.exit_reason}: {t.return_pct:+.2f}%" for t in trades], hoverinfo="text+x"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), title="Every trade on the chart",
                      legend=dict(orientation="h", y=-0.12))
    return fig


def equity_chart(report) -> go.Figure:
    fig = go.Figure()
    for series, colour in ((report.result.equity, "#2f6fed"), (report.result.buy_hold, "#8a94a6")):
        fig.add_trace(go.Scatter(x=series.index, y=series * 10_000, name=series.name,
                                 line=dict(color=colour, width=2 if colour == "#2f6fed" else 1.5)))
    fig.add_vline(x=pd.Timestamp(report.oos["split_date"]), line_dash="dot", line_color="#d98a00")
    fig.add_annotation(x=pd.Timestamp(report.oos["split_date"]), y=1, yref="paper", showarrow=False,
                       text="unseen data →", xanchor="left", font=dict(color="#d98a00"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log", yaxis_tickprefix="$",
                      title="Growth of $10,000 vs buy & hold (log scale)", legend=dict(orientation="h", y=-0.12))
    return fig


def monte_carlo_chart(report) -> go.Figure:
    bands = report.mc["bands"]
    x = bands.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bands["p95"] * 10_000, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p5"] * 10_000, fill="tonexty", fillcolor="rgba(47,111,237,0.18)",
                             line=dict(width=0), name="90% of shuffles"))
    fig.add_trace(go.Scatter(x=x, y=bands["median"] * 10_000, line=dict(color="#2f6fed", width=2), name="Median"))
    fig.add_hline(y=10_000, line_color="#8a94a6", line_dash="dot")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log", yaxis_tickprefix="$",
                      xaxis_title="Trade number", title=f"Luck check: {report.mc['runs']:,} reshuffled histories",
                      legend=dict(orientation="h", y=-0.3))
    return fig


# --------------------------------------------------------------------------- page
st.title("📈 Strategy in Plain English")
st.caption("Describe a trading idea in your own words. AI turns it into exact rules, a tested engine backtests it "
           "on years of real prices, and robustness checks tell you whether it's an edge or luck.")

with st.sidebar:
    st.header("Try an example")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.idea = example
    st.divider()
    st.caption("Daily bars from Yahoo Finance. Fees of 0.05% per side. Signals at the close, fills at the next "
               "open, stop assumed first if stop and target hit the same day. Research only - not financial advice.")

idea = st.text_area("Your strategy", key="idea", height=100,
                    placeholder="e.g. Buy Bitcoin when it breaks its 20-day high, stop 2 ATR below, target 3R")
go_clicked = st.button("Backtest it", type="primary", disabled=not idea.strip())

if go_clicked:
    try:
        with st.status("Working...", expanded=True) as status:
            st.write("🧠 Reading your strategy...")
            strategy = understand(idea)
            st.write(f"📥 Downloading {strategy.years} years of {strategy.symbol}, then backtesting...")
            report = test_strategy(strategy)
            st.write("🎲 Ran the luck check and the unseen-data test. ✍️ Wrote the verdict.")
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.report = report
    except (StrategyError, ai.AIError, ai.MissingAPIKeyError) as error:
        st.error(str(error))
        st.session_state.pop("report", None)
    except anthropic.AuthenticationError:
        st.error("Your API key was rejected. Check ANTHROPIC_API_KEY in the .env file.")
    except anthropic.APIConnectionError:
        st.error("Couldn't reach Claude. Check your internet connection.")

report = st.session_state.get("report")
if report:
    s, m, sc = report.strategy, report.result.metrics, report.score

    left, right = st.columns([3, 2])
    with left:
        st.subheader(s.name)
        st.markdown("**What the AI understood** - check this before trusting any number:")
        st.markdown("\n".join(f"- {line}" for line in describe_rules(s)))
        if s.assumptions:
            with st.expander(f"{len(s.assumptions)} assumptions it made"):
                st.markdown("\n".join(f"- {a}" for a in s.assumptions))
        if s.unsupported:
            st.warning("Not modelled: " + " ".join(s.unsupported))
    with right:
        colour = GRADE_COLOUR[sc["grade"]]
        st.markdown(
            f"<div style='border:2px solid {colour};border-radius:12px;padding:14px 18px;text-align:center'>"
            f"<div style='font-size:13px;opacity:.75'>ROBUSTNESS SCORE</div>"
            f"<div style='font-size:46px;font-weight:700;color:{colour};line-height:1.1'>{sc['score']}"
            f"<span style='font-size:20px;opacity:.6'>/100</span></div>"
            f"<div style='font-size:18px;font-weight:600;color:{colour}'>{sc['grade']}</div></div>",
            unsafe_allow_html=True)
        for name, points in sc["parts"].items():
            st.progress(points / 20, text=f"{name}: {points:g}/20")

    if not report.result.trades:
        st.info("The rules never triggered a trade in this period. Try looser conditions or a longer history.")
        st.stop()

    cols = st.columns(3) + st.columns(3)
    cols[0].metric("Return", pct(m["total_return_pct"]), f"buy & hold {pct(m['buy_hold_return_pct'])}", delta_color="off")
    cols[1].metric("Per year", pct(m["cagr_pct"]), f"hold {pct(m['buy_hold_cagr_pct'])}", delta_color="off")
    cols[2].metric("Max drawdown", pct(m["max_drawdown_pct"]), f"hold {pct(m['buy_hold_max_drawdown_pct'])}", delta_color="off")
    cols[3].metric("Trades", m["trades"], f"{m['time_in_market_pct']:.0f}% time in market", delta_color="off")
    cols[4].metric("Win rate", pct(m["win_rate_pct"], signed=False))
    cols[5].metric("Profit factor", f"{m['profit_factor']:.2f}")

    v = report.verdict
    if v:
        st.markdown(f"### 🧐 {md(v.headline)}")
        c1, c2, c3 = st.columns(3)
        for col, title, items in ((c1, "✅ What works", v.what_works), (c2, "⚠️ Why it could fail", v.why_it_could_fail),
                                  (c3, "🧪 Try next", v.try_next)):
            with col:
                st.markdown(f"**{title}**")
                st.markdown("\n".join(f"- {md(i)}" for i in items) or "_Nothing stood out._")

    st.plotly_chart(equity_chart(report), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(price_chart(report), use_container_width=True, config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(monte_carlo_chart(report), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"In {report.mc['prob_loss_pct']:.0f}% of reshuffles the strategy lost money. "
                   f"A 1-in-20 bad run saw a {report.mc['worst_drawdown_p95_pct']:.0f}% drawdown.")
    with c2:
        o, i = report.oos["out_of_sample"], report.oos["in_sample"]
        st.markdown(f"**Before vs after {report.oos['split_date']}** (the last 30% acts as unseen data)")
        st.dataframe(pd.DataFrame({
            "First 70%": [i["trades"], pct(i["win_rate_pct"], False), f"{i['profit_factor']:.2f}", pct(i["total_return_pct"])],
            "Last 30% (unseen)": [o["trades"], pct(o["win_rate_pct"], False), f"{o['profit_factor']:.2f}", pct(o["total_return_pct"])],
        }, index=["Trades", "Win rate", "Profit factor", "Return"]), use_container_width=True)

    with st.expander(f"All {len(report.result.trades)} trades"):
        st.dataframe(pd.DataFrame([{
            "Entry": t.entry_date.date(), "Entry price": round(t.entry_price, 4),
            "Exit": t.exit_date.date(), "Exit price": round(t.exit_price, 4),
            "Why": t.exit_reason, "Days": t.bars_held, "Return %": round(t.return_pct, 2),
            "R": None if t.r_multiple is None else round(t.r_multiple, 2),
        } for t in report.result.trades]), use_container_width=True, hide_index=True)

    st.caption("Research only - not financial advice. Past results don't predict future ones.")
