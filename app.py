"""
Inflation Tracker — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Inflation Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0d0f14; }

.metric-card {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0;
}
.metric-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a6070;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 26px;
    font-weight: 600;
    color: #e8eaf0;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.metric-delta-up   { font-size: 13px; color: #f06c6c; margin-top: 4px; }
.metric-delta-down { font-size: 13px; color: #5eca8a; margin-top: 4px; }
.metric-delta-flat { font-size: 13px; color: #5a6070; margin-top: 4px; }

.section-title {
    font-size: 13px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3a4055;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #1a1e28;
    padding-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
}
.badge-food     { background: #1a2e1a; color: #5eca8a; }
.badge-energy   { background: #2e1a0e; color: #f0936c; }
.badge-housing  { background: #1a1a2e; color: #8c6cf0; }
.badge-transport{ background: #1a2530; color: #6cb8f0; }
.badge-health   { background: #2e1a1a; color: #f06c6c; }
.badge-other    { background: #1e1e1e; color: #888; }

[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1a1e28;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a6070 !important;
}

.stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_forecasts():
    path = Path("data/forecasts.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["ds"])
    return df


@st.cache_data(ttl=3600)
def load_metrics():
    path = Path("data/model_metrics.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="item")


@st.cache_data(ttl=3600)
def load_raw():
    path = Path("data/prices.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_value(v, unit):
    if unit and "index" in unit.lower():
        return f"{v:.1f}"
    if v > 500:
        return f"${v:,.0f}"
    return f"${v:.2f}"


def delta_color(pct):
    if pct > 2:   return "metric-delta-up"
    if pct < -2:  return "metric-delta-down"
    return "metric-delta-flat"


def delta_arrow(pct):
    if pct > 0: return "▲"
    if pct < 0: return "▼"
    return "—"


CATEGORY_COLORS = {
    "Food & Beverages":     "#5eca8a",
    "Energy":               "#f0936c",
    "Housing":              "#8c6cf0",
    "Transportation":       "#6cb8f0",
    "Healthcare":           "#f06c6c",
    "Education":            "#f0d96c",
    "Apparel":              "#f06cc8",
    "Recreation":           "#6cf0e8",
    "Food Away From Home":  "#a8f06c",
    "Other":                "#888888",
}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df_fc  = load_forecasts()
    df_raw = load_raw()

    # ── Demo mode if no data ───────────────────────────────────────────────────
    if df_fc is None or df_raw is None:
        st.warning("⚠️ No data found. Run `python data_pipeline.py` then `python forecast_model.py` first.")
        st.info("Showing demo mode with simulated data.")
        df_fc, df_raw = generate_demo_data()

    items      = sorted(df_fc["item"].unique())
    categories = sorted(df_fc["category"].unique())

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 Inflation Tracker")
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)

        selected_cat = st.selectbox("Category", ["All"] + categories)

        filtered_items = items if selected_cat == "All" else [
            i for i in items
            if df_fc.loc[df_fc["item"] == i, "category"].iloc[0] == selected_cat
        ]

        selected_item = st.selectbox("Item", filtered_items)

        years_back = st.slider("History (years)", 1, 10, 5)

        st.markdown("<div class='section-title'>About</div>", unsafe_allow_html=True)
        st.caption("Data: BLS Consumer Price Index (CPI-U)")
        st.caption("Forecast: Facebook Prophet · 95% CI")
        st.caption("Updated monthly")

    # ── Item data ──────────────────────────────────────────────────────────────
    item_fc   = df_fc[df_fc["item"] == selected_item].copy()
    item_unit = item_fc["item_unit"].iloc[0] if "item_unit" in item_fc.columns else \
                df_raw.loc[df_raw["item"] == selected_item, "unit"].iloc[0] \
                if df_raw is not None else ""

    cutoff    = item_fc["ds"].max() - pd.DateOffset(years=years_back)
    item_fc   = item_fc[item_fc["ds"] >= cutoff]

    hist      = item_fc[~item_fc["is_forecast"]]
    fcast     = item_fc[item_fc["is_forecast"]]

    # ── Metrics ────────────────────────────────────────────────────────────────
    cur_val   = hist["actual"].dropna().iloc[-1]  if not hist.empty else np.nan
    prev_12   = hist["actual"].dropna().iloc[-13] if len(hist) >= 13 else hist["actual"].dropna().iloc[0]
    yoy_pct   = (cur_val - prev_12) / prev_12 * 100 if prev_12 else 0

    prev_60   = hist["actual"].dropna().iloc[-61] if len(hist) >= 61 else hist["actual"].dropna().iloc[0]
    five_pct  = (cur_val - prev_60) / prev_60 * 100 if prev_60 else 0

    fc_end    = fcast["yhat"].iloc[-1]  if not fcast.empty else np.nan
    fc_pct    = (fc_end - cur_val) / cur_val * 100 if cur_val else 0

    cat_color = CATEGORY_COLORS.get(item_fc["category"].iloc[0], "#888")

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='padding: 1.5rem 0 0.5rem;'>
        <span style='font-size:28px; font-weight:600; color:#e8eaf0;'>{selected_item}</span>
        <span style='margin-left:12px; font-size:13px; color:{cat_color};
              background: {cat_color}22; padding:3px 10px; border-radius:20px;'>
          {item_fc["category"].iloc[0]}
        </span>
    </div>
    <div style='font-size:13px; color:#5a6070; margin-bottom:1.5rem;'>
        {item_unit} · Source: BLS CPI-U · Forecast: Prophet ML model
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta, suffix in [
        (c1, "Current price",    cur_val,  None,     ""),
        (c2, "1-year change",    yoy_pct,  yoy_pct,  "%"),
        (c3, "5-year change",    five_pct, five_pct, "%"),
        (c4, "12-mo forecast",   fc_end,   fc_pct,   ""),
    ]:
        with col:
            if val is None or np.isnan(val):
                display = "—"
            elif suffix == "%":
                display = f"{val:+.1f}%"
            else:
                display = fmt_value(val, item_unit)

            delta_html = ""
            if delta is not None and not np.isnan(delta):
                cls = delta_color(delta)
                delta_html = f"<div class='{cls}'>{delta_arrow(delta)} {abs(delta):.1f}% vs prior period</div>"

            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{display}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart ──────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Confidence band
    if not fcast.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fcast["ds"], fcast["ds"].iloc[::-1]]),
            y=pd.concat([fcast["yhat_upper"], fcast["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(239,147,108,0.12)",
            line=dict(width=0),
            name="95% confidence",
            hoverinfo="skip",
        ))

    # Forecast line
    if not fcast.empty:
        # Connect last historical point to first forecast
        connect_x = [hist["ds"].iloc[-1], fcast["ds"].iloc[0]] if not hist.empty else fcast["ds"].tolist()[:1]
        connect_y = [hist["actual"].iloc[-1], fcast["yhat"].iloc[0]] if not hist.empty else [fcast["yhat"].iloc[0]]
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode="lines",
            line=dict(color="#f0936c", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fcast["ds"], y=fcast["yhat"],
            mode="lines",
            name="Forecast (Prophet)",
            line=dict(color="#f0936c", width=2.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>Forecast: %{y:.2f}<extra></extra>",
        ))

    # Actual line
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["ds"], y=hist["actual"],
            mode="lines",
            name="Actual price",
            line=dict(color=cat_color, width=2.5),
            hovertemplate="%{x|%b %Y}<br>Price: %{y:.2f}<extra></extra>",
        ))

    # Today marker
    fig.add_vline(
        x=hist["ds"].max().timestamp() * 1000 if not hist.empty else 0,
        line_dash="dash", line_color="#2a2e3e", line_width=1,
        annotation_text="Today", annotation_font_color="#3a4055",
        annotation_font_size=11,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d0f14",
        font=dict(family="DM Sans", color="#5a6070"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="#1a1e28", showgrid=False,
            tickfont=dict(size=11), zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#1a1e28", showgrid=True,
            tickfont=dict(size=11), zeroline=False,
            tickprefix="" if "index" in item_unit.lower() else "$",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Model metrics + decomposition ─────────────────────────────────────────
    df_metrics = load_metrics()
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("<div class='section-title'>Model performance</div>", unsafe_allow_html=True)
        if df_metrics is not None and selected_item in df_metrics.index:
            row = df_metrics.loc[selected_item]
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    st.metric(k, f"{v:.4f}" if k != "MAPE_%" else f"{v:.2f}%")
        else:
            st.caption("Run forecast_model.py to see metrics.")

    with col_b:
        st.markdown("<div class='section-title'>Trend decomposition</div>", unsafe_allow_html=True)
        if "trend" in item_fc.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=item_fc["ds"], y=item_fc["trend"],
                mode="lines", name="Trend",
                line=dict(color=cat_color, width=2),
            ))
            if "yearly" in item_fc.columns:
                fig2.add_trace(go.Scatter(
                    x=item_fc["ds"], y=item_fc["yearly"],
                    mode="lines", name="Seasonal",
                    line=dict(color="#8c6cf0", width=1.5, dash="dot"),
                    yaxis="y2",
                ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d0f14",
                font=dict(family="DM Sans", color="#5a6070"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=200,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                            font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(showgrid=False, tickfont=dict(size=10), zeroline=False),
                yaxis=dict(gridcolor="#1a1e28", showgrid=True, tickfont=dict(size=10), zeroline=False),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=dict(size=10)),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Category overview heatmap ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Category inflation overview — 1-year change (%)</div>",
                unsafe_allow_html=True)

    last_vals = []
    for it in df_fc["item"].unique():
        sub = df_fc[(df_fc["item"] == it) & (~df_fc["is_forecast"])].dropna(subset=["actual"])
        if len(sub) >= 13:
            cur  = sub["actual"].iloc[-1]
            prev = sub["actual"].iloc[-13]
            pct  = (cur - prev) / prev * 100
            cat  = sub["category"].iloc[0]
            last_vals.append({"item": it, "category": cat, "yoy_pct": round(pct, 1)})

    if last_vals:
        ov = pd.DataFrame(last_vals).sort_values("yoy_pct", ascending=True)
        fig3 = go.Figure(go.Bar(
            x=ov["yoy_pct"],
            y=ov["item"],
            orientation="h",
            marker_color=[
                "#f06c6c" if v > 5 else "#f0936c" if v > 2 else "#5eca8a" if v < 0 else "#888"
                for v in ov["yoy_pct"]
            ],
            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d0f14",
            font=dict(family="DM Sans", color="#5a6070"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(300, len(last_vals) * 22),
            xaxis=dict(showgrid=True, gridcolor="#1a1e28", zeroline=True,
                       zerolinecolor="#2a2e3e", ticksuffix="%", tickfont=dict(size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── Demo data generator (runs without real data) ──────────────────────────────
def generate_demo_data():
    import random
    random.seed(42)
    items_meta = [
        ("Eggs (dozen)", "Food & Beverages", "per dozen", 2.5, 0.06),
        ("Gasoline, regular (gallon)", "Energy", "per gallon", 3.0, 0.15),
        ("Milk (whole, gallon)", "Food & Beverages", "per gallon", 3.8, 0.04),
        ("Ground beef (lb)", "Food & Beverages", "per lb", 4.2, 0.05),
        ("Rent of primary residence", "Housing", "monthly index", 340, 2.0),
        ("New cars", "Transportation", "monthly index", 280, 3.0),
        ("Medical care (overall)", "Healthcare", "monthly index", 520, 1.5),
        ("Airline fares", "Transportation", "monthly index", 310, 8.0),
    ]
    dates = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    fc_dates = pd.date_range("2026-01-01", periods=12, freq="MS")

    rows_raw, rows_fc = [], []
    for name, cat, unit, base, vol in items_meta:
        v = base
        hist_vals = []
        for d in dates:
            v += (random.random() - 0.47) * vol + 0.012 * v
            v = max(v, base * 0.3)
            rows_raw.append({"item": name, "category": cat, "unit": unit, "date": d, "value": round(v, 2)})
            hist_vals.append(round(v, 2))
            rows_fc.append({
                "item": name, "category": cat, "ds": d,
                "actual": round(v, 2), "yhat": round(v, 2),
                "yhat_upper": round(v * 1.05, 2), "yhat_lower": round(v * 0.95, 2),
                "trend": round(v * 0.98, 2), "yearly": round((random.random() - 0.5) * vol * 0.5, 3),
                "is_forecast": False,
            })
        for d in fc_dates:
            v += (random.random() - 0.45) * vol + 0.015 * v
            rows_fc.append({
                "item": name, "category": cat, "ds": d,
                "actual": None, "yhat": round(v, 2),
                "yhat_upper": round(v * 1.06, 2), "yhat_lower": round(v * 0.94, 2),
                "trend": round(v * 0.99, 2), "yearly": round((random.random() - 0.5) * vol * 0.5, 3),
                "is_forecast": True,
            })

    df_raw = pd.DataFrame(rows_raw)
    df_fc  = pd.DataFrame(rows_fc)
    df_fc["ds"] = pd.to_datetime(df_fc["ds"])
    return df_fc, df_raw


if __name__ == "__main__":
    main()
