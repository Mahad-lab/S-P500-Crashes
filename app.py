"""
AI sell-off vs. past S&P 500 crashes — a Streamlit recreation.

Every number on the chart is recomputed from raw S&P 500 daily closes
(^GSPC, price index, no dividends) so you can check the original chart's claims.

Run:
    pip install streamlit yfinance plotly pandas numpy
    streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TICKER = "^GSPC"
START = "1950-01-01"
DEFAULT_END = dt.date(2026, 11, 5)  # same cut-off as the original chart

# Each crash is identified by the record close the slide started from.
# label: where the name goes relative to the trough dot (short crashes only).
EPISODES = [
    dict(name="1973 oil shock", peak="1973-01-11", color="#7b3f98"),
    dict(name="1987 crash", peak="1987-08-25", color="#4c4ba5", label="right"),
    dict(name="Dot-com", peak="2000-03-24", color="#11957a"),
    dict(name="2008 crisis", peak="2007-10-09", color="#1d3461"),
    dict(name="Covid", peak="2020-02-19", color="#1f77a8", label="below"),
    dict(name="2022", peak="2022-01-03", color="#3fa9d8", label="right"),
    dict(name="April 2025", peak="2025-02-19", color="#9b8fc7", label="right"),
]
AI_COLOR = "#c8590b"
GRID = "rgba(0,0,0,0.07)"


# ---------------------------------------------------------------- data ----
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading S&P 500 closes from Yahoo Finance…")
def load_from_yahoo(end: dt.date) -> pd.Series:
    import yfinance as yf

    # yfinance's `end` is exclusive, so add a day to include the end date.
    df = yf.download(
        TICKER,
        start=START,
        end=(end + dt.timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns MultiIndex columns
        close = close.iloc[:, 0]
    return _clean(close)


def load_from_csv(file, end: dt.date) -> pd.Series:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date") or df.columns[0]
    close_col = cols.get("close") or df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[close_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
    )
    s = s[s.index <= pd.Timestamp(end)]
    return _clean(s)


def _clean(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index >= pd.Timestamp(START)].astype(float).rename("close")


# ------------------------------------------------------------ analysis ----
def find_slides(close: pd.Series) -> pd.DataFrame:
    """Split history into slides. A slide starts at a record close and lasts
    until the next record close; its depth is the lowest close in between."""
    values = close.to_numpy()
    running_max = np.maximum.accumulate(values)
    record_pos = np.flatnonzero(values >= running_max)

    rows = []
    for i, start in enumerate(record_pos):
        end = record_pos[i + 1] if i + 1 < len(record_pos) else len(values)
        if end - start < 2:  # next day was a record too: no slide
            continue
        seg = values[start:end]
        t = int(np.argmin(seg))
        rows.append(
            dict(
                peak_date=close.index[start],
                peak_pos=int(start),
                peak_close=seg[0],
                trough_date=close.index[start + t],
                days_to_trough=t,
                depth=seg[t] / seg[0] - 1,
                ongoing=end == len(values),
            )
        )
    return pd.DataFrame(rows)


def match_episode(slides: pd.DataFrame, peak: str) -> pd.Series | None:
    """Find the slide that started at `peak` (nearest record within 10 days)."""
    deep = slides[slides.depth <= -0.15]
    if deep.empty:
        return None
    gap = (deep.peak_date - pd.Timestamp(peak)).abs()
    if gap.min() > pd.Timedelta(days=10):
        return None
    return deep.loc[gap.idxmin()]


def slide_path(close: pd.Series, peak_pos: int, n_days: int) -> pd.DataFrame:
    seg = close.iloc[peak_pos : peak_pos + n_days + 1]
    return pd.DataFrame(
        dict(day=np.arange(len(seg)), pct=(seg.to_numpy() / seg.iloc[0] - 1) * 100, date=seg.index)
    )


def fmt_pct(x: float) -> str:
    return f"\u2212{abs(x):.0f}%" if x < 0 else f"{x:.0f}%"


# --------------------------------------------------------------- chart ----
def make_figure(close, slides, chosen, ai_name, cutoff):
    fig = go.Figure()
    hover = "%{customdata|%d %b %Y}<br>Day %{x}: %{y:.1f}% below record<extra>%{meta}</extra>"
    long_ones, lows = [], [0.0]

    for ep in chosen:
        row = match_episode(slides, ep["peak"])
        if row is None:
            continue
        path = slide_path(close, row.peak_pos, row.days_to_trough)
        trough = path.pct.iloc[-1]
        lows.append(trough)
        drawn = path[path.day <= cutoff]
        fig.add_trace(
            go.Scatter(
                x=drawn.day, y=drawn.pct, customdata=drawn.date, meta=ep["name"],
                mode="lines", line=dict(color=ep["color"], width=1.6),
                opacity=0.75, hovertemplate=hover, showlegend=False,
            )
        )
        if row.days_to_trough <= cutoff:
            fig.add_trace(
                go.Scatter(
                    x=[path.day.iloc[-1]], y=[trough], mode="markers",
                    marker=dict(color=ep["color"], size=9), hoverinfo="skip", showlegend=False,
                )
            )
            below = ep.get("label") == "below"
            fig.add_annotation(
                x=path.day.iloc[-1], y=trough,
                text=f"<b>{ep['name']}: {fmt_pct(trough)}</b>",
                showarrow=False, font=dict(color=ep["color"], size=13),
                xanchor="center" if below else "left",
                yanchor="top" if below else "middle",
                xshift=0 if below else 8, yshift=-8 if below else 0,
            )
        else:
            long_ones.append((ep, row, drawn.pct.iloc[-1], trough))

    # Crashes that bottomed after the cut-off: dotted drop to their trough.
    long_ones.sort(key=lambda r: -r[3])  # shallowest first
    label_x = cutoff + 6 + 3 * len(long_ones)
    prev_label_y = None
    for k, (ep, row, y_cut, trough) in enumerate(long_ones):
        xk = cutoff + 3 * (k + 1)
        fig.add_trace(
            go.Scatter(
                x=[cutoff, xk, xk], y=[y_cut, y_cut, trough], mode="lines",
                line=dict(color=ep["color"], width=1.4, dash="dot"),
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[xk], y=[trough], mode="markers", marker=dict(color=ep["color"], size=9),
                hovertemplate=f"{ep['name']}: trough {row.trough_date:%d %b %Y}<extra></extra>",
                showlegend=False,
            )
        )
        label_y = trough if prev_label_y is None else min(trough, prev_label_y - 5.5)
        prev_label_y = label_y
        fig.add_trace(
            go.Scatter(
                x=[xk, label_x - 1], y=[trough, label_y], mode="lines",
                line=dict(color=ep["color"], width=0.8), hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_annotation(
            x=label_x, y=label_y, xanchor="left", showarrow=False, align="left",
            text=(f"<b>{ep['name']}: {fmt_pct(trough)}</b><br>"
                  f"<span style='font-size:12px'>after {row.days_to_trough} days</span>"),
            font=dict(color=ep["color"], size=13),
        )

    # The current slide (the one still running at the last close).
    current = slides[slides.ongoing]
    ai_text = None
    if not current.empty:
        row = current.iloc[0]
        n_days = len(close) - 1 - row.peak_pos
        path = slide_path(close, row.peak_pos, n_days)
        latest = path.pct.iloc[-1]
        lows.append(path.pct.min())
        shown = path[path.day <= cutoff]
        fig.add_trace(
            go.Scatter(
                x=shown.day, y=shown.pct, customdata=shown.date, meta=ai_name,
                mode="lines", line=dict(color=AI_COLOR, width=3.5),
                hovertemplate=hover, showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[shown.day.iloc[-1]], y=[shown.pct.iloc[-1]], mode="markers",
                marker=dict(color=AI_COLOR, size=10), hoverinfo="skip", showlegend=False,
            )
        )
        ai_text = f"{ai_name}: {fmt_pct(latest)} after {n_days} days"
        fig.add_annotation(
            x=shown.day.iloc[-1], y=shown.pct.iloc[-1], ax=shown.day.iloc[-1], ay=7,
            axref="x", ayref="y", xanchor="left", yanchor="bottom",
            text=f"<b>{ai_text}</b>", font=dict(color=AI_COLOR, size=14),
            showarrow=True, arrowhead=0, arrowwidth=2, arrowcolor=AI_COLOR,
        )

    # Axes styled like the original.
    y_floor = min(-62, np.floor(min(lows) / 10) * 10 - 2)
    y_ticks = list(range(0, int(y_floor) - 1, -10))
    x_ticks = list(range(50, cutoff + 1, 50))
    x_text = [str(t) for t in x_ticks]
    if x_ticks:  # "trading days" sits to the right of the last tick, like the original
        fig.add_annotation(
            x=x_ticks[-1], y=0, xref="x", yref="paper", yanchor="top", xanchor="left",
            yshift=-6, xshift=16, showarrow=False, text="trading days",
            font=dict(size=13, color="#444"),
        )

    fig.add_hline(y=0, line=dict(color="#b5b5b5", width=1))
    fig.update_layout(
        template="plotly_white", height=680,
        margin=dict(l=70, r=20, t=40, b=40),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color="#333"),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(
        range=[-2, label_x + 60], tickvals=x_ticks, ticktext=x_text,
        showgrid=False, zeroline=False, tickfont=dict(family="monospace"),
    )
    fig.update_yaxes(
        range=[y_floor, 9], tickvals=y_ticks,
        ticktext=["Record" if t == 0 else f"\u2212{abs(t)}%" for t in y_ticks],
        gridcolor=GRID, zeroline=False,
    )
    return fig, ai_text


# ------------------------------------------------------------------ UI ----
def main():
    st.set_page_config(page_title="AI sell-off vs past crashes", layout="wide")

    with st.sidebar:
        st.header("Settings")
        end = st.date_input("Data up to", DEFAULT_END, max_value=dt.date.today())
        upload = st.file_uploader("…or use your own CSV (Date, Close)", type="csv")
        ai_name = st.text_input("Name for the current slide", "AI sell-off")
        cutoff = st.slider("Cut the axis after (trading days)", 100, 700, 250, 50)
        names = [e["name"] for e in EPISODES]
        picked = st.multiselect("Crashes to draw", names, default=names)

    try:
        close = load_from_csv(upload, end) if upload else load_from_yahoo(end)
    except Exception as exc:  # network or parsing problems
        st.error(f"Couldn't load data: {exc}")
        st.stop()
    if close.empty:
        st.error("No data returned.")
        st.stop()

    slides = find_slides(close)
    n6 = int((slides.depth <= -0.06).sum())
    n15 = int((slides.depth <= -0.15).sum())
    chosen = [e for e in EPISODES if e["name"] in picked]

    st.markdown(
        f"<h2 style='margin-bottom:0'>The {ai_name} compared to other stock market crashes</h2>"
        f"<p style='color:#555;font-size:1.05rem;margin-top:.3rem'>How far the S&amp;P 500 fell "
        f"below a record close. Since {close.index[0].year}, {n6} slides reached 6% and {n15} went "
        f"past 15%. The {len(chosen)} best known of those {n15} are drawn.</p>",
        unsafe_allow_html=True,
    )

    fig, _ = make_figure(close, slides, chosen, ai_name, cutoff)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"S&P 500 daily closes to {close.index[-1]:%d %b %Y}, without dividends · Source: Yahoo Finance"
    )

    with st.expander("Check the numbers"):
        table = slides[slides.depth <= -0.15].copy()
        lookup = {}
        for e in EPISODES:
            r = match_episode(slides, e["peak"])
            if r is not None:
                lookup[r.peak_date] = e["name"]
        table.insert(0, "crash", table.peak_date.map(lookup).fillna(""))
        table["depth"] = (table.depth * 100).round(1).astype(str) + "%"
        table["peak_date"] = table.peak_date.dt.date
        table["trough_date"] = table.trough_date.dt.date
        st.dataframe(
            table[["crash", "peak_date", "trough_date", "days_to_trough", "depth"]]
            .sort_values("peak_date", ascending=False),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "A slide runs from one record close to the next. Depth = lowest close in that "
            "stretch vs. the record. Counts can differ slightly from the original chart if "
            "it defined slides differently."
        )


if __name__ == "__main__":
    main()