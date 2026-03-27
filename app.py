"""Streamlit app for Indian Mutual Fund Analytics."""

import sys
import os

# Point matplotlib at a writable cache dir before any import triggers it
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BENCHMARKS, RISK_FREE_RATE, ROLLING_WINDOWS, TAG_BENCHMARK
from db import (
    get_benchmark_data,
    get_db_stats,
    get_fund_nav,
    get_funds_with_nav,
    init_db,
)
from analytics import (
    annualized_volatility,
    average_drawdown,
    batting_average,
    cagr,
    calendar_year_returns,
    calmar_ratio,
    compute_comparison_table,
    compute_fund_analytics,
    downside_capture,
    drawdown_series,
    martin_ratio,
    pct_positive_periods,
    pct_time_underwater,
    rolling_alpha,
    rolling_beta,
    rolling_returns,
    rolling_sharpe,
    rolling_sip_xirr,
    rolling_sortino,
    rolling_volatility,
    sharpe_ratio,
    sip_xirr,
    sortino_ratio,
    trailing_returns,
    ulcer_index,
    upside_capture,
)
from fetcher import (
    download_nav_for_schemes,
    fetch_benchmark_data,
    fetch_nse_index_data,
    get_scheme_codes_for_equity,
    refresh_scheme_list,
)
from config import NSE_DIRECT_INDICES
from tagger import (
    auto_tag_all_funds,
    get_all_tags,
    get_fund_tags,
    get_funds_by_tag,
    init_tags_table,
    set_fund_tags,
    CATEGORY_TAG,
    THEME_RULES,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MF Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
init_tags_table()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("MF Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Fund Analysis", "Fund Comparison",
     "Sector & Style", "Data Management"],
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def _signed_gradient_col(col):
    """Per-column color: green = positive, red = negative, yellow = zero.

    Normalizes each column independently around 0 so that the sign of the
    value — not its rank within the column — drives the color.
    """
    import matplotlib.cm as _cm
    max_abs = col.dropna().abs().max()
    if pd.isna(max_abs) or max_abs == 0:
        return [""] * len(col)
    cmap = _cm.RdYlGn
    styles = []
    for v in col:
        if pd.isna(v):
            styles.append("")
        else:
            scaled = float(np.clip((v / max_abs + 1) / 2, 0, 1))
            r, g, b, _ = cmap(scaled)
            styles.append(
                f"background-color: rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
            )
    return styles


def fmt_num(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def load_nav_as_series(scheme_code: int) -> pd.Series:
    df = get_fund_nav(scheme_code)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df["nav"]


@st.cache_data(show_spinner=False)
def _cached_fund_analytics(nav: pd.Series, bench: pd.Series | None) -> dict:
    """Cached wrapper around compute_fund_analytics."""
    return compute_fund_analytics(nav, bench)


@st.cache_data(show_spinner=False)
def _cached_rolling_sip(scheme_code: int, yrs: int) -> pd.Series:
    """Cached rolling SIP XIRR computation."""
    return rolling_sip_xirr(load_nav_as_series(scheme_code), yrs)


def load_benchmark_as_series(index_name: str) -> pd.Series:
    df = get_benchmark_data(index_name)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df["close_price"]


def add_range_buttons(fig: go.Figure, height: int = None,
                      rangeslider: bool = True) -> go.Figure:
    """Add period selector buttons + range slider + unified crosshair to a figure."""
    buttons = [
        dict(count=1,  label="1M",  step="month", stepmode="backward"),
        dict(count=3,  label="3M",  step="month", stepmode="backward"),
        dict(count=6,  label="6M",  step="month", stepmode="backward"),
        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
        dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
        dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
        dict(step="all", label="Max"),
    ]
    fig.update_xaxes(
        rangeselector=dict(buttons=buttons, bgcolor="#eef0f2",
                           activecolor="#c0c8d0"),
        rangeslider=dict(visible=rangeslider, thickness=0.06),
        type="date",
    )
    fig.update_layout(hovermode="x unified")
    if height is not None:
        fig.update_layout(height=height)
    return fig


def fund_selector(funds_df: pd.DataFrame, key_prefix: str = "fund",
                  multiselect: bool = False):
    if funds_df.empty:
        st.warning("No funds with NAV data. Go to Data Management to download data.")
        return [] if multiselect else None

    categories = ["All"] + sorted(funds_df["scheme_category"].dropna().unique().tolist())
    cat = st.selectbox("Category", categories, key=f"{key_prefix}_cat")
    filtered = funds_df if cat == "All" else funds_df[funds_df["scheme_category"] == cat]

    houses = ["All"] + sorted(filtered["fund_house"].dropna().unique().tolist())
    house = st.selectbox("Fund House", houses, key=f"{key_prefix}_house")
    if house != "All":
        filtered = filtered[filtered["fund_house"] == house]

    search = st.text_input("Search fund name", key=f"{key_prefix}_search")
    if search:
        filtered = filtered[filtered["scheme_name"].str.contains(search, case=False, na=False)]

    if filtered.empty:
        st.info("No funds match the filters.")
        return [] if multiselect else None

    options = filtered[["scheme_code", "scheme_name"]].set_index("scheme_code")["scheme_name"]

    if multiselect:
        return st.multiselect(
            "Select funds",
            options=options.index.tolist(),
            format_func=lambda x: options.get(x, str(x)),
            key=f"{key_prefix}_select",
        )
    else:
        return st.selectbox(
            "Select fund",
            options=options.index.tolist(),
            format_func=lambda x: options.get(x, str(x)),
            key=f"{key_prefix}_select",
        )


# ---------------------------------------------------------------------------
# PAGE: Dashboard
# ---------------------------------------------------------------------------
def render_dashboard():
    st.header("Dashboard")

    stats = get_db_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Funds in DB", stats["total_funds"])
    c2.metric("Funds with NAV", stats["funds_with_nav"])
    c3.metric("NAV Records", f"{stats['total_nav_records']:,}")
    c4.metric("Last Update", stats["last_update"] or "Never")

    if stats["funds_with_nav"] == 0:
        st.info("No NAV data yet. Go to **Data Management** to download fund data.")
        return

    funds = get_funds_with_nav()
    if funds.empty:
        return

    # ── Sidebar filters ──────────────────────────────────────────────────────
    all_tags = [t for t, _ in get_all_tags()]
    all_cats = sorted(funds["scheme_category"].dropna().unique().tolist())

    with st.sidebar:
        st.subheader("Filters")

        # 1. Tag filter (sector / style)
        sel_tags = st.multiselect("Sector / Style", all_tags, key="dash_tags",
                                  placeholder="All tags")

        # 2. Category filter
        sel_cats = st.multiselect("SEBI Category", all_cats, key="dash_cats",
                                  placeholder="All categories")

        # 3. Fund picker / excluder
        filter_mode = st.radio(
            "Fund filter", ["Show all", "Exclude funds", "Pick funds"],
            key="dash_filter_mode", horizontal=True,
        )

        # Build filtered fund set: tags → category → name filter
        if sel_tags:
            tagged_codes: set[int] = set()
            for tag in sel_tags:
                tagged_codes.update(
                    f["scheme_code"] for f in get_funds_by_tag(tag)
                )
            tag_filtered = funds[funds["scheme_code"].isin(tagged_codes)]
        else:
            tag_filtered = funds

        cat_filtered = (
            tag_filtered[tag_filtered["scheme_category"].isin(sel_cats)]
            if sel_cats else tag_filtered
        )

        filtered_names = sorted(cat_filtered["scheme_name"].tolist())
        if filter_mode == "Exclude funds":
            excluded = st.multiselect("Exclude from view", filtered_names,
                                      key="dash_exclude")
            include_names = [n for n in filtered_names if n not in excluded]
        elif filter_mode == "Pick funds":
            picked = st.multiselect("Show only", filtered_names, key="dash_pick")
            include_names = picked if picked else filtered_names
        else:
            include_names = filtered_names

    funds_view = funds[funds["scheme_name"].isin(include_names)]

    st.subheader("Category Distribution")
    cat_counts = funds_view["scheme_category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig_cat = px.bar(cat_counts, x="Category", y="Count", color="Category",
                     hover_data={"Count": True})
    fig_cat.update_layout(showlegend=False, height=380,
                          xaxis_tickangle=-30)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Top Performing Funds — CAGR since inception")
    perf_data = []
    for _, row in funds_view.iterrows():
        nav = load_nav_as_series(row["scheme_code"])
        if len(nav) > 252:
            c_val = cagr(nav)
            v_val = annualized_volatility(nav)
            if not np.isnan(c_val) and not np.isnan(v_val):
                perf_data.append({
                    "Fund": row["scheme_name"],
                    "Category": row["scheme_category"],
                    "CAGR": c_val,
                    "Volatility": v_val,
                    "Since": str(row["first_date"])[:10],
                })

    if perf_data:
        perf_df = pd.DataFrame(perf_data).sort_values("CAGR", ascending=False)

        # Top-20 table (all funds if fewer than 20)
        top_df = perf_df.head(20).copy()
        top_df["CAGR_pct"] = top_df["CAGR"] * 100
        top_df["Vol_pct"] = top_df["Volatility"] * 100
        display_top = top_df[["Fund", "Category", "CAGR_pct", "Vol_pct", "Since"]].rename(
            columns={"CAGR_pct": "CAGR (%)", "Vol_pct": "Volatility (%)"}
        )
        st.dataframe(
            display_top.style.apply(_signed_gradient_col, axis=0, subset=["CAGR (%)"])
                             .format({"CAGR (%)": "{:.2f}%", "Volatility (%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

        # Risk-return scatter
        st.subheader("Risk-Return Overview")
        fig_rr = px.scatter(
            perf_df.dropna(subset=["CAGR", "Volatility"]),
            x="Volatility", y="CAGR",
            color="Category", hover_name="Fund",
            labels={"Volatility": "Annualized Volatility", "CAGR": "CAGR (since inception)"},
            custom_data=["Since"],
        )
        fig_rr.update_traces(
            marker=dict(size=8, opacity=0.75),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "CAGR: %{y:.2%}<br>"
                "Volatility: %{x:.2%}<br>"
                "Since: %{customdata[0]}<extra></extra>"
            ),
        )
        fig_rr.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_rr.update_layout(height=500, yaxis_tickformat=".0%",
                              xaxis_tickformat=".0%")
        st.plotly_chart(fig_rr, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Fund Analysis
# ---------------------------------------------------------------------------
def render_fund_analysis():
    st.header("Fund Analysis")

    funds = get_funds_with_nav()

    with st.sidebar:
        st.subheader("Fund Selection")
        code = fund_selector(funds, key_prefix="analysis")

    if code is None:
        return

    nav_full = load_nav_as_series(code)
    if nav_full.empty:
        st.warning("No NAV data for this fund.")
        return

    fund_info = funds[funds["scheme_code"] == code].iloc[0]
    st.subheader(fund_info["scheme_name"])

    # Fund summary row (uses full unfiltered data)
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Current NAV", f"₹{nav_full.iloc[-1]:.4f}")
    ic2.metric("Category", fund_info["scheme_category"])
    ic3.metric("Data From", str(nav_full.index[0].date()))
    ic4.metric("Data To", str(nav_full.index[-1].date()))

    # Controls row: date range + benchmark
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        date_start = st.date_input(
            "Analysis start",
            value=nav_full.index[0].date(),
            min_value=nav_full.index[0].date(),
            max_value=nav_full.index[-1].date(),
            key="analysis_date_start",
        )
    with ctrl2:
        date_end = st.date_input(
            "Analysis end",
            value=nav_full.index[-1].date(),
            min_value=nav_full.index[0].date(),
            max_value=nav_full.index[-1].date(),
            key="analysis_date_end",
        )
    with ctrl3:
        stats = get_db_stats()
        bench_options = ["None"] + stats.get("benchmarks_loaded", [])
        bench_name = st.selectbox("Benchmark", bench_options, key="analysis_bench")

    if pd.Timestamp(date_start) >= pd.Timestamp(date_end):
        st.error("Start date must be before end date.")
        return

    # Apply date filter
    nav = nav_full[
        (nav_full.index >= pd.Timestamp(date_start)) &
        (nav_full.index <= pd.Timestamp(date_end))
    ]
    if len(nav) < 5:
        st.warning("Too few data points in selected range.")
        return

    bench_prices = None
    if bench_name != "None":
        bp = load_benchmark_as_series(bench_name)
        if not bp.empty:
            bench_prices = bp[
                (bp.index >= pd.Timestamp(date_start)) &
                (bp.index <= pd.Timestamp(date_end))
            ]
            if bench_prices.empty:
                bench_prices = None

    metrics = _cached_fund_analytics(nav, bench_prices)

    # Tabs
    tab_nav, tab_returns, tab_risk, tab_dd, tab_bench = st.tabs(
        ["NAV Chart", "Returns", "Risk Metrics", "Drawdown", "Benchmark"]
    )

    # ── NAV Chart ────────────────────────────────────────────────────────────
    with tab_nav:
        log_scale = st.checkbox("Log scale (Y-axis)", value=False, key="nav_log")

        daily_chg = nav.pct_change().fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values,
            name="NAV",
            customdata=daily_chg.values * 100,
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "NAV: ₹%{y:.4f}<br>"
                "Daily chg: %{customdata:+.2f}%"
                "<extra></extra>"
            ),
            line=dict(width=1.5),
        ))

        if bench_prices is not None:
            bp_aligned = bench_prices[bench_prices.index >= nav.index[0]]
            if len(bp_aligned) > 0:
                bp_norm = bp_aligned / bp_aligned.iloc[0] * nav.iloc[0]
                fig.add_trace(go.Scatter(
                    x=bp_norm.index, y=bp_norm.values,
                    name=bench_name,
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        f"{bench_name}: ₹%{{y:.4f}}"
                        "<extra></extra>"
                    ),
                    line=dict(width=1.5, dash="dash"),
                ))

        fig.update_layout(
            title="NAV History",
            xaxis_title="Date",
            yaxis_title="NAV (₹)" + (" — log scale" if log_scale else ""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        add_range_buttons(fig, height=520, rangeslider=not log_scale)

        if log_scale:
            fig.update_yaxes(type="log")
            fig.update_xaxes(rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        # Quick stats below chart
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("CAGR", fmt_pct(metrics.get("cagr")))
        s2.metric("Total Return", fmt_pct(metrics.get("absolute_return")))
        s3.metric("Volatility", fmt_pct(metrics.get("volatility")))
        s4.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown")))

    # ── Returns ───────────────────────────────────────────────────────────────
    with tab_returns:
        st.markdown("#### Trailing Returns")
        tr = trailing_returns(nav)
        labels = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "SI"]
        ret_cols = st.columns(len(labels))
        for i, label in enumerate(labels):
            val = tr.get(label, np.nan)
            delta = None
            if bench_prices is not None:
                btr = trailing_returns(bench_prices)
                bval = btr.get(label, np.nan)
                if not np.isnan(val) and not np.isnan(bval):
                    delta = fmt_pct(val - bval) + " vs bench"
            ret_cols[i].metric(label, fmt_pct(val), delta)

        st.markdown("#### Calendar Year Returns")
        cy_fund = calendar_year_returns(nav)
        if not cy_fund.empty:
            fig_cy = go.Figure()
            fig_cy.add_trace(go.Bar(
                x=cy_fund.index.astype(str), y=cy_fund.values * 100,
                name="Fund",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                marker_color=[
                    "rgba(220,50,50,0.85)" if v < 0 else "rgba(50,160,90,0.85)"
                    for v in cy_fund.values
                ],
            ))
            if bench_prices is not None:
                cy_bench = calendar_year_returns(bench_prices)
                common = cy_fund.index.intersection(cy_bench.index)
                if len(common) > 0:
                    fig_cy.add_trace(go.Bar(
                        x=cy_bench[common].index.astype(str),
                        y=cy_bench[common].values * 100,
                        name=bench_name,
                        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                        marker_color="rgba(100,120,180,0.65)",
                    ))
            fig_cy.update_layout(
                barmode="group", height=400,
                title="Calendar Year Returns",
                yaxis_title="Return (%)",
                xaxis_title="Year",
            )
            fig_cy.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_cy, use_container_width=True)
            st.caption("First and last bars may represent partial calendar years.")
        else:
            st.info("Not enough data for calendar year breakdown.")

        st.markdown("#### Rolling Returns")
        window_choice = st.selectbox("Rolling window", list(ROLLING_WINDOWS.keys()),
                                     index=1, key="rolling_w")
        window_days = ROLLING_WINDOWS[window_choice]
        rr = rolling_returns(nav, window_days)
        if not rr.empty:
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(
                x=rr.index, y=rr.values * 100, name="Fund",
                hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                line=dict(width=1.5),
            ))
            if bench_prices is not None:
                brr = rolling_returns(bench_prices, window_days)
                if not brr.empty:
                    fig_rr.add_trace(go.Scatter(
                        x=brr.index, y=brr.values * 100,
                        name=bench_name,
                        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                        line=dict(width=1.5, dash="dash"),
                    ))
            fig_rr.update_layout(
                title=f"Rolling {window_choice} Returns",
                yaxis_title="Return (%)",
            )
            fig_rr.add_hline(y=0, line_dash="dot", line_color="gray")
            add_range_buttons(fig_rr, height=460)
            st.plotly_chart(fig_rr, use_container_width=True)

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Mean", fmt_pct(rr.mean()))
            rc2.metric("Median", fmt_pct(rr.median()))
            rc3.metric("Min (worst)", fmt_pct(rr.min()))
            rc4.metric("Max (best)", fmt_pct(rr.max()))
        else:
            st.info(f"Not enough data for {window_choice} rolling returns.")

    # ── Risk Metrics ──────────────────────────────────────────────────────────
    with tab_risk:
        st.markdown("#### Risk Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Volatility (ann.)", fmt_pct(metrics.get("volatility")))
        r2.metric("Downside Deviation", fmt_pct(metrics.get("downside_deviation")))
        r3.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown")))
        r4.metric("Skewness", fmt_num(metrics.get("skewness")))

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Kurtosis", fmt_num(metrics.get("kurtosis")))
        r6.metric("VaR 95%", fmt_pct(metrics.get("var_95")))
        r7.metric("CVaR 95%", fmt_pct(metrics.get("cvar_95")))
        r8.metric("VaR 99%", fmt_pct(metrics.get("var_99")))

        st.markdown("#### Extended Drawdown Metrics")
        ed1, ed2, ed3 = st.columns(3)
        ed1.metric("Ulcer Index", fmt_pct(metrics.get("ulcer_index")))
        ed2.metric("Avg Drawdown", fmt_pct(metrics.get("average_drawdown")))
        ed3.metric("% Time Underwater", fmt_pct(metrics.get("pct_time_underwater")))

        st.markdown("#### Risk-Adjusted Returns")
        ra1, ra2, ra3, ra4 = st.columns(4)
        ra1.metric("Sharpe Ratio", fmt_num(metrics.get("sharpe_ratio")))
        ra2.metric("Sortino Ratio", fmt_num(metrics.get("sortino_ratio")))
        ra3.metric("Calmar Ratio", fmt_num(metrics.get("calmar_ratio")))
        ra4.metric("Martin Ratio", fmt_num(metrics.get("martin_ratio")))

        ra5, ra6 = st.columns(2)
        ra5.metric("% Positive 1Y Periods", fmt_pct(metrics.get("pct_positive_1y")))
        ra6.metric("CVaR 99%", fmt_pct(metrics.get("cvar_99")))

        st.markdown("#### SIP Returns (XIRR)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("1Y SIP XIRR", fmt_pct(metrics.get("sip_xirr_1y")))
        s2.metric("3Y SIP XIRR", fmt_pct(metrics.get("sip_xirr_3y")))
        s3.metric("5Y SIP XIRR", fmt_pct(metrics.get("sip_xirr_5y")))
        s4.metric("10Y SIP XIRR", fmt_pct(metrics.get("sip_xirr_10y")))
        st.caption("Monthly SIP of ₹1 invested each month; XIRR of the investment + redemption at period end.")

        with st.expander("Rolling SIP Distribution"):
            sip_horizon = st.selectbox("SIP horizon (years)", [1, 3, 5],
                                       key="fund_sip_hist_years")
            sip_series = _cached_rolling_sip(code, sip_horizon)
            if not sip_series.empty:
                fig_sip = go.Figure()
                fig_sip.add_trace(go.Histogram(
                    x=sip_series.values * 100, nbinsx=40,
                    name=f"{sip_horizon}Y SIP XIRR",
                    marker_color="steelblue", opacity=0.8,
                ))
                fig_sip.add_vline(x=0, line_dash="dot", line_color="red")
                fig_sip.add_vline(x=sip_series.mean() * 100,
                                  line_dash="dash", line_color="green",
                                  annotation_text=f"Avg {sip_series.mean()*100:.1f}%",
                                  annotation_position="top right")
                fig_sip.update_layout(
                    xaxis_title=f"{sip_horizon}Y SIP XIRR (%)",
                    yaxis_title="# Windows",
                    height=380,
                )
                st.plotly_chart(fig_sip, use_container_width=True)
                pct_pos = (sip_series > 0).sum() / len(sip_series)
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("% Positive Windows", fmt_pct(pct_pos))
                sc2.metric("Median XIRR", fmt_pct(sip_series.median()))
                sc3.metric("Worst XIRR", fmt_pct(sip_series.min()))
            else:
                st.info(f"Insufficient history for {sip_horizon}Y SIP analysis.")

        if bench_prices is not None:
            st.markdown(f"#### vs {bench_name}")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Beta", fmt_num(metrics.get("beta")))
            b2.metric("Alpha (Jensen's)", fmt_pct(metrics.get("alpha")))
            b3.metric("Tracking Error", fmt_pct(metrics.get("tracking_error")))
            b4.metric("Information Ratio", fmt_num(metrics.get("information_ratio")))
            b5, b6, b7, b8 = st.columns(4)
            b5.metric("Treynor Ratio", fmt_num(metrics.get("treynor_ratio")))
            b6.metric("Upside Capture", fmt_num(metrics.get("upside_capture")))
            b7.metric("Downside Capture", fmt_num(metrics.get("downside_capture")))
            b8.metric("Batting Average", fmt_pct(metrics.get("batting_average")))

            st.markdown("#### Rolling Alpha & Beta")
            rab_window = st.selectbox("Window", list(ROLLING_WINDOWS.keys()),
                                      index=1, key="fund_rab_window")
            rab_days = ROLLING_WINDOWS[rab_window]
            ra_series = rolling_alpha(nav, bench_prices, rab_days)
            rb_series = rolling_beta(nav, bench_prices, rab_days)
            col_ra, col_rb = st.columns(2)
            with col_ra:
                if not ra_series.empty:
                    fig_ra = go.Figure()
                    fig_ra.add_trace(go.Scatter(
                        x=ra_series.index, y=ra_series.values * 100,
                        name="Alpha",
                        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                        line=dict(width=1.5),
                    ))
                    fig_ra.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig_ra.update_layout(
                        title=f"Rolling {rab_window} Alpha",
                        yaxis_title="Alpha (%)", height=360,
                    )
                    add_range_buttons(fig_ra, height=360, rangeslider=False)
                    st.plotly_chart(fig_ra, use_container_width=True)
                else:
                    st.info("Not enough data for rolling alpha.")
            with col_rb:
                if not rb_series.empty:
                    fig_rb = go.Figure()
                    fig_rb.add_trace(go.Scatter(
                        x=rb_series.index, y=rb_series.values,
                        name="Beta",
                        hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
                        line=dict(width=1.5, color="darkorange"),
                    ))
                    fig_rb.add_hline(y=1, line_dash="dot", line_color="gray")
                    fig_rb.update_layout(
                        title=f"Rolling {rab_window} Beta",
                        yaxis_title="Beta", height=360,
                    )
                    add_range_buttons(fig_rb, height=360, rangeslider=False)
                    st.plotly_chart(fig_rb, use_container_width=True)
                else:
                    st.info("Not enough data for rolling beta.")

        with st.expander("All metrics (raw values)"):
            display_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            st.json({
                k: round(v, 6) if isinstance(v, float) and not np.isnan(v) else str(v)
                for k, v in display_metrics.items()
            })

        st.divider()

        # Rolling Volatility chart
        st.markdown("#### Rolling Volatility")
        rv_window = st.selectbox("Window", list(ROLLING_WINDOWS.keys()),
                                 index=1, key="risk_vol_window")
        rv = rolling_volatility(nav, ROLLING_WINDOWS[rv_window])
        if not rv.empty:
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=rv.index, y=rv.values * 100, name="Fund",
                hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                line=dict(width=1.5),
            ))
            if bench_prices is not None:
                brv = rolling_volatility(bench_prices, ROLLING_WINDOWS[rv_window])
                if not brv.empty:
                    fig_rv.add_trace(go.Scatter(
                        x=brv.index, y=brv.values * 100, name=bench_name,
                        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                        line=dict(width=1.5, dash="dash"),
                    ))
            fig_rv.update_layout(
                title=f"Rolling {rv_window} Volatility (Annualized)",
                yaxis_title="Volatility (%)",
            )
            add_range_buttons(fig_rv, height=420)
            st.plotly_chart(fig_rv, use_container_width=True)
        else:
            st.info(f"Not enough data for {rv_window} rolling volatility.")

        # Rolling Sharpe chart
        st.markdown("#### Rolling Sharpe Ratio")
        rs_window = st.selectbox("Window", list(ROLLING_WINDOWS.keys()),
                                 index=1, key="risk_sharpe_window")
        rs = rolling_sharpe(nav, ROLLING_WINDOWS[rs_window])
        if not rs.empty:
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(
                x=rs.index, y=rs.values, name="Fund",
                hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
                line=dict(width=1.5),
            ))
            if bench_prices is not None:
                brs = rolling_sharpe(bench_prices, ROLLING_WINDOWS[rs_window])
                if not brs.empty:
                    fig_rs.add_trace(go.Scatter(
                        x=brs.index, y=brs.values, name=bench_name,
                        hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
                        line=dict(width=1.5, dash="dash"),
                    ))
            fig_rs.update_layout(
                title=f"Rolling {rs_window} Sharpe Ratio",
                yaxis_title="Sharpe Ratio",
            )
            fig_rs.add_hline(y=0, line_dash="dot", line_color="gray")
            add_range_buttons(fig_rs, height=420)
            st.plotly_chart(fig_rs, use_container_width=True)
        else:
            st.info(f"Not enough data for {rs_window} rolling Sharpe.")

    # ── Drawdown ──────────────────────────────────────────────────────────────
    with tab_dd:
        dd = drawdown_series(nav)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            fill="tozeroy", name="Drawdown",
            hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
            line=dict(width=1, color="crimson"),
            fillcolor="rgba(220,20,60,0.25)",
        ))

        # Annotate top-5 worst drawdown periods as shaded regions
        dd_periods = metrics.get("_drawdown_periods", [])
        shade_colors = [
            "rgba(180,0,0,0.10)", "rgba(220,80,0,0.08)",
            "rgba(200,140,0,0.07)", "rgba(150,150,0,0.06)",
            "rgba(120,120,120,0.05)",
        ]
        for i, p in enumerate(dd_periods[:5]):
            x0 = p["start"]
            x1 = p["end"] if p["end"] is not None else dd.index[-1]
            fig_dd.add_vrect(
                x0=x0, x1=x1,
                fillcolor=shade_colors[i],
                layer="below", line_width=0,
                annotation_text=f"#{i+1} {p['depth']*100:.1f}%",
                annotation_position="top left",
                annotation_font_size=10,
            )

        fig_dd.update_layout(title="Drawdown Chart", yaxis_title="Drawdown (%)")
        add_range_buttons(fig_dd, height=480)
        st.plotly_chart(fig_dd, use_container_width=True)

        if dd_periods:
            st.markdown("#### Worst Drawdown Periods")
            dd_table = []
            for p in dd_periods[:10]:
                dd_table.append({
                    "Rank": dd_periods.index(p) + 1,
                    "Start": str(p["start"])[:10],
                    "Trough": str(p["trough_date"])[:10],
                    "Recovery": str(p["end"])[:10] if p["end"] else "Ongoing",
                    "Depth": f"{p['depth']*100:.2f}%",
                    "Duration (days)": p["duration_days"],
                    "Recovery (days)": p["recovery_days"] if p["recovery_days"] else "—",
                })
            st.dataframe(pd.DataFrame(dd_table), use_container_width=True, hide_index=True)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    with tab_bench:
        if bench_prices is None:
            st.info("Select a benchmark above to see comparison metrics.")
        else:
            st.markdown(f"#### Fund vs {bench_name} — Growth of ₹100")

            fp_aligned = nav[nav.index >= bench_prices.index[0]]
            bp_aligned = bench_prices[bench_prices.index >= nav.index[0]]
            if len(fp_aligned) > 1 and len(bp_aligned) > 1:
                common_start = max(fp_aligned.index[0], bp_aligned.index[0])
                fp_trim = fp_aligned[fp_aligned.index >= common_start]
                bp_trim = bp_aligned[bp_aligned.index >= common_start]

                fp_norm = fp_trim / fp_trim.iloc[0] * 100
                bp_norm = bp_trim / bp_trim.iloc[0] * 100
                alpha_vs = (fp_norm.iloc[-1] - bp_norm.iloc[-1])

                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(
                    x=fp_norm.index, y=fp_norm.values, name="Fund",
                    hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                    line=dict(width=1.8),
                ))
                fig_b.add_trace(go.Scatter(
                    x=bp_norm.index, y=bp_norm.values, name=bench_name,
                    hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                    line=dict(width=1.8, dash="dash"),
                ))
                fig_b.add_hline(y=100, line_dash="dot", line_color="gray",
                                annotation_text="Start: ₹100")
                fig_b.update_layout(
                    yaxis_title="Value of ₹100 invested",
                    title=f"Outperformance over period: ₹{alpha_vs:+.1f} per ₹100",
                )
                add_range_buttons(fig_b, height=480)
                st.plotly_chart(fig_b, use_container_width=True)

            # Capture ratio quadrant
            uc = metrics.get("upside_capture")
            dc = metrics.get("downside_capture")
            if uc is not None and dc is not None and not np.isnan(uc) and not np.isnan(dc):
                st.markdown("#### Capture Ratio Quadrant")
                # Quadrant labels
                fig_quad = go.Figure()
                # Quadrant backgrounds
                for (x0, x1, y0, y1, color, label) in [
                    (0, 100, 100, 200, "rgba(0,180,0,0.05)", "Best: High up, Low down"),
                    (100, 200, 100, 200, "rgba(255,200,0,0.05)", "Ok: High up, High down"),
                    (0, 100, 0, 100, "rgba(255,200,0,0.05)", "Ok: Low up, Low down"),
                    (100, 200, 0, 100, "rgba(220,0,0,0.05)", "Worst: Low up, High down"),
                ]:
                    fig_quad.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                                       fillcolor=color, line_width=0)
                    fig_quad.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label,
                                            showarrow=False, font=dict(size=9, color="gray"))

                fig_quad.add_trace(go.Scatter(
                    x=[dc], y=[uc], mode="markers+text",
                    text=[fund_info["scheme_name"][:35]],
                    textposition="top center",
                    marker=dict(size=16, color="royalblue",
                                line=dict(color="white", width=2)),
                    hovertemplate=(
                        f"Upside Capture: {uc:.1f}%<br>"
                        f"Downside Capture: {dc:.1f}%<extra></extra>"
                    ),
                ))
                fig_quad.add_hline(y=100, line_dash="dot", line_color="gray")
                fig_quad.add_vline(x=100, line_dash="dot", line_color="gray")
                fig_quad.update_layout(
                    xaxis_title="Downside Capture (%)",
                    yaxis_title="Upside Capture (%)",
                    height=420,
                    xaxis=dict(range=[max(0, dc - 40), dc + 40]),
                    yaxis=dict(range=[max(0, uc - 40), uc + 40]),
                )
                st.plotly_chart(fig_quad, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Fund Comparison
# ---------------------------------------------------------------------------
def render_comparison():
    st.header("Fund Comparison")

    funds = get_funds_with_nav()

    with st.sidebar:
        st.subheader("Select Funds to Compare")
        codes = fund_selector(funds, key_prefix="compare", multiselect=True)

    if not codes or len(codes) < 2:
        st.info("Select at least 2 funds from the sidebar to compare.")
        return

    # Benchmark
    stats = get_db_stats()
    bench_options = ["None"] + stats.get("benchmarks_loaded", [])
    top1, top2 = st.columns([2, 2])
    bench_name = top1.selectbox("Benchmark", bench_options, key="compare_bench")
    bench_prices_full = None
    if bench_name != "None":
        bp = load_benchmark_as_series(bench_name)
        if not bp.empty:
            bench_prices_full = bp

    # Load full NAV data
    fund_dict_full = {}
    for code in codes:
        info = funds[funds["scheme_code"] == code]
        name = info.iloc[0]["scheme_name"] if not info.empty else str(code)
        short_name = name[:45] + "…" if len(name) > 45 else name
        nav = load_nav_as_series(code)
        if not nav.empty:
            fund_dict_full[short_name] = nav

    if len(fund_dict_full) < 2:
        st.warning("Need at least 2 funds with data to compare.")
        return

    # Date range control
    all_starts = [s.index[0] for s in fund_dict_full.values()]
    all_ends   = [s.index[-1] for s in fund_dict_full.values()]
    auto_start = max(all_starts)
    auto_end   = min(all_ends)

    st.markdown("#### Date Range")
    align_mode = st.radio(
        "Alignment",
        ["Common start (auto)", "Custom range"],
        horizontal=True,
        key="compare_align_mode",
    )

    if align_mode == "Common start (auto)":
        compare_start = auto_start
        compare_end   = auto_end
        st.caption(
            f"Common period: **{compare_start.date()}** → **{compare_end.date()}** "
            f"({(compare_end - compare_start).days} days)"
        )
    else:
        dr1, dr2 = st.columns(2)
        compare_start = pd.Timestamp(dr1.date_input(
            "Start", value=auto_start.date(), key="compare_custom_start"
        ))
        compare_end = pd.Timestamp(dr2.date_input(
            "End", value=auto_end.date(), key="compare_custom_end"
        ))
        if compare_start >= compare_end:
            st.error("Start date must be before end date.")
            return

    # Apply date filter
    fund_dict = {
        name: nav[(nav.index >= compare_start) & (nav.index <= compare_end)]
        for name, nav in fund_dict_full.items()
    }
    fund_dict = {k: v for k, v in fund_dict.items() if len(v) > 10}

    if len(fund_dict) < 2:
        st.warning("Fewer than 2 funds have data in the selected range.")
        return

    bench_prices = None
    if bench_prices_full is not None:
        bench_prices = bench_prices_full[
            (bench_prices_full.index >= compare_start) &
            (bench_prices_full.index <= compare_end)
        ]
        if bench_prices.empty:
            bench_prices = None

    # Tabs
    tab_chart, tab_returns_cmp, tab_rolling, tab_table, tab_risk = st.tabs(
        ["Price Charts", "Returns Comparison", "Rolling Metrics", "Metrics Table", "Risk Comparison"]
    )

    # ── Price Charts ──────────────────────────────────────────────────────────
    with tab_chart:
        # Growth of ₹100
        fig_growth = go.Figure()
        for name, nav in fund_dict.items():
            normed = nav / nav.iloc[0] * 100
            fig_growth.add_trace(go.Scatter(
                x=normed.index, y=normed.values, name=name,
                hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                line=dict(width=1.8),
            ))
        if bench_prices is not None and len(bench_prices) > 0:
            bp_norm = bench_prices / bench_prices.iloc[0] * 100
            fig_growth.add_trace(go.Scatter(
                x=bp_norm.index, y=bp_norm.values, name=bench_name,
                hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                line=dict(width=2, dash="dash", color="gray"),
            ))
        fig_growth.add_hline(y=100, line_dash="dot", line_color="lightgray")
        fig_growth.update_layout(
            title=f"Growth of ₹100 — {compare_start.date()} to {compare_end.date()}",
            yaxis_title="Value (₹)",
        )
        add_range_buttons(fig_growth, height=520)
        st.plotly_chart(fig_growth, use_container_width=True)

        # Drawdown comparison
        st.markdown("#### Drawdown Comparison")
        fig_dd = go.Figure()
        for name, nav in fund_dict.items():
            dd = drawdown_series(nav)
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values * 100, name=name,
                hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                line=dict(width=1.4),
            ))
        fig_dd.update_layout(yaxis_title="Drawdown (%)")
        fig_dd.add_hline(y=0, line_dash="dot", line_color="gray")
        add_range_buttons(fig_dd, height=420)
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Returns Comparison ────────────────────────────────────────────────────
    with tab_returns_cmp:
        periods = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]

        # Trailing returns grouped bar
        st.markdown("#### Trailing Returns")
        fig_tr = go.Figure()
        for fname, fnav in fund_dict.items():
            tr = trailing_returns(fnav)
            y_vals = [
                tr.get(p, np.nan) * 100 if not np.isnan(tr.get(p, np.nan)) else None
                for p in periods
            ]
            fig_tr.add_trace(go.Bar(
                name=fname, x=periods, y=y_vals,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
        if bench_prices is not None:
            btr = trailing_returns(bench_prices)
            y_bench = [
                btr.get(p, np.nan) * 100 if not np.isnan(btr.get(p, np.nan)) else None
                for p in periods
            ]
            fig_tr.add_trace(go.Bar(
                name=bench_name, x=periods, y=y_bench,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                marker=dict(color="rgba(100,100,120,0.65)", pattern_shape="/"),
            ))
        fig_tr.update_layout(barmode="group", height=400,
                             yaxis_title="Return (%)")
        fig_tr.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_tr, use_container_width=True)

        # Calendar year returns comparison
        st.markdown("#### Calendar Year Returns")
        all_years = set()
        cy_data = {}
        for fname, fnav in fund_dict.items():
            cy = calendar_year_returns(fnav)
            cy_data[fname] = cy
            all_years.update(cy.index.tolist())
        if bench_prices is not None:
            cy_bench = calendar_year_returns(bench_prices)
            cy_data[bench_name] = cy_bench
            all_years.update(cy_bench.index.tolist())

        all_years_sorted = sorted(all_years)
        fig_cy_cmp = go.Figure()
        for fname, cy in cy_data.items():
            y_vals = [cy.get(y, np.nan) * 100 if y in cy.index else None
                      for y in all_years_sorted]
            is_bench = (fname == bench_name)
            fig_cy_cmp.add_trace(go.Bar(
                name=fname,
                x=[str(y) for y in all_years_sorted],
                y=y_vals,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                marker=dict(
                    pattern_shape="/" if is_bench else "",
                    opacity=0.7 if is_bench else 0.9,
                ),
            ))
        fig_cy_cmp.update_layout(barmode="group", height=420,
                                 yaxis_title="Return (%)", xaxis_title="Year")
        fig_cy_cmp.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_cy_cmp, use_container_width=True)
        st.caption("First/last bars may be partial years.")

    # ── Rolling Metrics ───────────────────────────────────────────────────────
    with tab_rolling:
        st.markdown("#### Rolling Metric Comparison")
        rm_col1, rm_col2 = st.columns(2)
        metric_choice = rm_col1.selectbox(
            "Metric",
            ["Volatility", "Sharpe", "Sortino", "Returns", "Alpha", "Beta"],
            key="compare_rolling_metric",
        )
        roll_window_key = rm_col2.selectbox(
            "Window", list(ROLLING_WINDOWS.keys()),
            index=1, key="compare_rolling_window",
        )
        roll_days = ROLLING_WINDOWS[roll_window_key]

        needs_bench = metric_choice in ("Alpha", "Beta")
        if needs_bench and bench_prices is None:
            st.warning(f"Select a benchmark to view rolling {metric_choice}.")
        else:
            fig_rm = go.Figure()
            scale, y_label = 1, metric_choice
            for fname, fnav in fund_dict.items():
                if metric_choice == "Volatility":
                    series = rolling_volatility(fnav, roll_days)
                    scale, y_label = 100, "Volatility (%)"
                elif metric_choice == "Sharpe":
                    series = rolling_sharpe(fnav, roll_days)
                    scale, y_label = 1, "Sharpe Ratio"
                elif metric_choice == "Sortino":
                    series = rolling_sortino(fnav, roll_days)
                    scale, y_label = 1, "Sortino Ratio"
                elif metric_choice == "Alpha":
                    series = rolling_alpha(fnav, bench_prices, roll_days)
                    scale, y_label = 100, "Alpha (%)"
                elif metric_choice == "Beta":
                    series = rolling_beta(fnav, bench_prices, roll_days)
                    scale, y_label = 1, "Beta"
                else:
                    series = rolling_returns(fnav, roll_days)
                    scale, y_label = 100, "Return (%)"

                if not series.empty:
                    fig_rm.add_trace(go.Scatter(
                        x=series.index, y=series.values * scale, name=fname,
                        hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
                        line=dict(width=1.5),
                    ))

            fig_rm.update_layout(
                title=f"Rolling {roll_window_key} {metric_choice}",
                yaxis_title=y_label,
            )
            if metric_choice in ("Sharpe", "Sortino", "Alpha"):
                fig_rm.add_hline(y=0, line_dash="dot", line_color="gray")
            if metric_choice == "Beta":
                fig_rm.add_hline(y=1, line_dash="dot", line_color="gray")
            add_range_buttons(fig_rm, height=500)
            st.plotly_chart(fig_rm, use_container_width=True)
        st.caption(
            f"Rolling {metric_choice} over trailing {roll_window_key} window. "
            "Gaps indicate insufficient data."
        )

    # ── Metrics Table ─────────────────────────────────────────────────────────
    with tab_table:
        comp_df = compute_comparison_table(fund_dict, bench_prices)
        if not comp_df.empty:
            display_df = comp_df.copy()
            pct_rows = [r for r in display_df.index
                        if any(k in r for k in [
                            "return", "cagr", "volatility", "deviation",
                            "drawdown", "alpha", "tracking", "var", "cvar",
                            "underwater", "batting", "sip_xirr", "positive",
                        ])]
            for col in display_df.columns:
                for idx in display_df.index:
                    val = display_df.loc[idx, col]
                    if pd.notna(val) and np.isfinite(float(val)):
                        fval = float(val)
                        display_df.loc[idx, col] = (
                            f"{fval*100:.2f}%" if idx in pct_rows
                            else f"{fval:.3f}"
                        )
                    else:
                        display_df.loc[idx, col] = "—"
            st.dataframe(display_df, use_container_width=True, height=720)

    # ── Risk Comparison ───────────────────────────────────────────────────────
    with tab_risk:
        # Risk-return scatter
        rr_data = []
        for fname, fnav in fund_dict.items():
            v = annualized_volatility(fnav)
            c = cagr(fnav)
            if not np.isnan(v) and not np.isnan(c):
                rr_data.append({"Fund": fname, "Volatility": v * 100, "CAGR": c * 100})
        if rr_data:
            rr_df = pd.DataFrame(rr_data)
            fig_scatter = px.scatter(
                rr_df, x="Volatility", y="CAGR", text="Fund",
                labels={"Volatility": "Volatility (%)", "CAGR": "CAGR (%)"},
            )
            fig_scatter.update_traces(
                textposition="top center", marker=dict(size=12),
                hovertemplate="<b>%{text}</b><br>CAGR: %{y:.2f}%<br>Vol: %{x:.2f}%<extra></extra>",
            )
            fig_scatter.update_layout(title="Risk-Return Profile", height=480)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Risk-adjusted ratios bar chart
        st.markdown("#### Risk-Adjusted Ratios")
        fund_names = list(fund_dict.keys())
        fig_rar = go.Figure()
        rar_metrics = {
            "Sharpe":  [sharpe_ratio(fund_dict[fn]) for fn in fund_names],
            "Sortino": [sortino_ratio(fund_dict[fn]) for fn in fund_names],
            "Calmar":  [calmar_ratio(fund_dict[fn]) for fn in fund_names],
        }
        for mname, vals in rar_metrics.items():
            fig_rar.add_trace(go.Bar(
                name=mname,
                x=fund_names,
                y=[v if not np.isnan(v) else None for v in vals],
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            ))
        fig_rar.update_layout(
            barmode="group", height=420,
            yaxis_title="Ratio",
            xaxis_tickangle=-25,
        )
        fig_rar.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_rar, use_container_width=True)

        # Capture ratio scatter (if benchmark)
        if bench_prices is not None:
            cap_data = []
            for fname, fnav in fund_dict.items():
                uc = upside_capture(fnav, bench_prices)
                dc = downside_capture(fnav, bench_prices)
                if not np.isnan(uc) and not np.isnan(dc):
                    cap_data.append({"Fund": fname, "Upside": uc, "Downside": dc})
            if cap_data:
                cap_df = pd.DataFrame(cap_data)
                fig_cap = px.scatter(
                    cap_df, x="Downside", y="Upside", text="Fund",
                    labels={"Downside": "Downside Capture (%)",
                            "Upside": "Upside Capture (%)"},
                )
                fig_cap.update_traces(
                    textposition="top center", marker=dict(size=12),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Upside: %{y:.1f}%<br>"
                        "Downside: %{x:.1f}%<extra></extra>"
                    ),
                )
                fig_cap.add_hline(y=100, line_dash="dot")
                fig_cap.add_vline(x=100, line_dash="dot")
                fig_cap.update_layout(
                    title=f"Capture Ratios vs {bench_name} (top-left = ideal)",
                    height=480,
                )
                st.plotly_chart(fig_cap, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Data Management
# ---------------------------------------------------------------------------
def render_data_management():
    st.header("Data Management")

    stats = get_db_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Funds", stats["total_funds"])
    c2.metric("Funds with NAV", stats["funds_with_nav"])
    c3.metric("Total NAV Records", f"{stats['total_nav_records']:,}")
    c4.metric("Benchmarks", ", ".join(stats.get("benchmarks_loaded", [])) or "None")

    st.divider()

    st.markdown("#### Step 1: Download Fund List from AMFI")
    st.caption("Fetches the master list of equity mutual fund schemes from AMFI India.")
    if st.button("Refresh Scheme List", key="refresh_list"):
        with st.spinner("Fetching scheme list from AMFI..."):
            try:
                total, equity = refresh_scheme_list()
                st.success(f"Found {total:,} total schemes, saved {equity:,} equity schemes.")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    st.markdown("#### Step 2: Download Historical NAV Data")
    st.caption(
        "Downloads full NAV history for all equity schemes from mfapi.in. "
        "Progress is saved — you can stop and resume."
    )
    scheme_codes = get_scheme_codes_for_equity()
    st.write(f"Equity schemes in DB: **{len(scheme_codes)}**")
    max_schemes = st.number_input(
        "Max schemes to fetch (0 = all)", min_value=0, value=0, step=50, key="max_schemes"
    )
    if st.button("Download NAV Data", key="download_nav"):
        codes = scheme_codes[:int(max_schemes)] if max_schemes > 0 else scheme_codes
        if not codes:
            st.warning("No equity schemes found. Refresh the scheme list first.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            def progress_cb(current, total, code, msg):
                if total > 0:
                    progress_bar.progress(current / total)
                status_text.text(f"[{current}/{total}] {msg}")
            updated = download_nav_for_schemes(codes, progress_callback=progress_cb)
            progress_bar.progress(1.0)
            st.success(f"Updated NAV data for {updated} schemes.")

    st.divider()

    st.markdown("#### Step 3: Download Benchmark Index Data")
    st.caption("Downloads index price data from Yahoo Finance.")
    bench_selection = st.multiselect(
        "Select benchmarks",
        options=list(BENCHMARKS.keys()),
        default=list(BENCHMARKS.keys()),
        key="bench_select",
    )
    if st.button("Download Benchmark Data", key="download_bench"):
        selected = {k: BENCHMARKS[k] for k in bench_selection}
        if not selected:
            st.warning("Select at least one benchmark.")
        else:
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()
            def bench_cb(current, total, name, msg):
                if total > 0:
                    progress_bar2.progress(current / total)
                status_text2.text(msg)
            fetch_benchmark_data(selected, progress_callback=bench_cb)
            progress_bar2.progress(1.0)
            st.success("Benchmark data updated.")

    st.divider()

    st.markdown("#### Step 3b: Download NSE Smallcap / Microcap Indices")
    st.caption(
        "Fetches from NSE daily archive files (archives.nseindia.com). "
        "Available from Jan 2013. First run downloads ~3,200 files — allow ~5 min."
    )
    nse_selection = st.multiselect(
        "Select NSE indices",
        options=list(NSE_DIRECT_INDICES.keys()),
        default=list(NSE_DIRECT_INDICES.keys()),
        key="nse_index_select",
    )
    if st.button("Download NSE Index Data", key="download_nse"):
        if not nse_selection:
            st.warning("Select at least one index.")
        else:
            selected_nse = {k: NSE_DIRECT_INDICES[k] for k in nse_selection}
            nse_progress = st.progress(0)
            nse_status = st.empty()
            def nse_cb(done, total, date_str, msg):
                if total > 0:
                    nse_progress.progress(min(done / total, 1.0))
                nse_status.text(msg)
            counts = fetch_nse_index_data(selected_nse, progress_callback=nse_cb)
            nse_progress.progress(1.0)
            summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
            st.success(f"NSE index data updated. {summary or 'All up to date.'}")
            st.cache_data.clear()

    st.divider()

    st.markdown("#### Daily Update")
    st.caption("Incremental update: refreshes scheme list + fetches only new NAV records.")
    if st.button("Run Daily Update", key="daily_update"):
        with st.spinner("Running daily update..."):
            try:
                refresh_scheme_list()
                codes = get_scheme_codes_for_equity()
                progress_bar3 = st.progress(0)
                status3 = st.empty()
                def daily_cb(current, total, code, msg):
                    if total > 0:
                        progress_bar3.progress(current / total)
                    status3.text(f"[{current}/{total}] {msg}")
                updated = download_nav_for_schemes(codes, progress_callback=daily_cb)
                progress_bar3.progress(1.0)
                fetch_benchmark_data()
                fetch_nse_index_data()
                st.success(f"Daily update complete. Updated {updated} schemes + benchmarks.")
            except Exception as e:
                st.error(f"Update failed: {e}")


# ---------------------------------------------------------------------------
# PAGE: Sector & Style
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_tag_navs(tag: str) -> dict[str, pd.Series]:
    """Load NAV series for all funds in a tag. Cached per tag."""
    funds = get_funds_by_tag(tag)
    result = {}
    for f in funds:
        nav = load_nav_as_series(f["scheme_code"])
        if len(nav) > 60:
            # short label: strip boilerplate suffixes
            name = f["scheme_name"]
            for suffix in [
                " - Direct Plan - Growth", "-Direct Plan-Growth",
                " - Direct - Growth", "- Direct - Growth",
                " Direct Plan Growth", " Direct Growth",
                " - Direct Plan - Growth Option",
                " Direct Plan - Growth Option",
                "-Direct-Growth", " -Direct Plan- Growth Option",
                " - Growth - Direct Plan", "-Direct Plan-Growth Option",
                " Growth Option - Direct Plan", "-Direct Plan -Growth Option",
                " Direct Plan Growth Plan - Growth Option",
                "- Direct Plan- Growth", "- Direct-Growth",
                " Growth Option- Direct", "-Growth Option- Direct",
                " - Growth Option - Direct Plan",
            ]:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            # also strip leading AMC prefix patterns like "XYZ Fund - "
            result[name.strip()] = nav
    return result


@st.cache_data(show_spinner=False)
def _compute_tag_metrics(tag: str, bench_name: str) -> pd.DataFrame:
    """Compute full metrics table for all funds in a tag. Cached."""
    fund_navs = _load_tag_navs(tag)
    bench = load_benchmark_as_series(bench_name) if bench_name != "None" else None
    return compute_comparison_table(fund_navs, bench)


def render_sector_style():
    st.header("Sector & Style Explorer")

    tags = get_all_tags()
    if not tags:
        st.warning("No tags found. Run **Data Management → Rebuild Tags** first.")
        return

    # ── Sidebar: tag picker ──────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Select Sector / Style")

        # Group tags into Style and Sector for readability
        style_tags  = [t for t, _ in tags if t in set(CATEGORY_TAG.values())]
        sector_tags = [t for t, _ in tags if t not in set(CATEGORY_TAG.values())]

        group = st.radio("Group", ["Style / Cap", "Sector / Theme"], horizontal=True,
                         key="tag_group")
        tag_pool = style_tags if group == "Style / Cap" else sector_tags
        tag_options = [f"{t}  ({n})" for t, n in tags if t in tag_pool]
        tag_map = {f"{t}  ({n})": t for t, n in tags if t in tag_pool}

        if not tag_options:
            st.info("No tags in this group.")
            return

        selected_label = st.selectbox("Tag", tag_options, key="tag_select")
        selected_tag = tag_map[selected_label]

        st.divider()
        stats = get_db_stats()
        bench_options = ["None"] + stats.get("benchmarks_loaded", [])
        suggested_bench = TAG_BENCHMARK.get(selected_tag, "Nifty 50")
        default_bench_idx = (
            bench_options.index(suggested_bench)
            if suggested_bench in bench_options else 0
        )
        bench_name = st.selectbox("Benchmark", bench_options,
                                  index=default_bench_idx, key="tag_bench")

        # Fund deselection
        st.markdown("**Funds in tag**")
        fund_list = get_funds_by_tag(selected_tag)

    if not fund_list:
        st.info("No funds with NAV data for this tag.")
        return

    # ── Fund filter (exclude or pick) ───────────────────────────────────────
    all_names = list(_load_tag_navs(selected_tag).keys())
    with st.sidebar:
        filter_mode = st.radio(
            "Fund filter", ["Show all", "Exclude funds", "Pick funds"],
            key="tag_filter_mode", horizontal=True,
        )
        if filter_mode == "Exclude funds":
            deselect = st.multiselect("Exclude from view", all_names, key="tag_deselect")
            include = [n for n in all_names if n not in deselect]
        elif filter_mode == "Pick funds":
            include = st.multiselect("Show only", all_names, key="tag_pick")
            if not include:
                st.caption("No funds selected — showing all.")
                include = all_names
        else:
            include = all_names

    fund_navs_all = _load_tag_navs(selected_tag)
    fund_navs = {k: v for k, v in fund_navs_all.items() if k in include}

    if len(fund_navs) < 1:
        st.warning("All funds deselected.")
        return

    bench_prices = (
        load_benchmark_as_series(bench_name) if bench_name != "None" else None
    )

    n_funds = len(fund_navs)
    st.markdown(
        f"### {selected_tag} — {n_funds} fund{'s' if n_funds != 1 else ''}"
    )

    # ── Date range control ───────────────────────────────────────────────────
    all_starts = [v.index[0] for v in fund_navs.values()]
    all_ends   = [v.index[-1] for v in fund_navs.values()]

    # Robust auto-period: use max(all_ends) so discontinued funds don't pull
    # the end backward; use a percentile start so brand-new funds (short history)
    # don't push the start forward past most funds' available data.
    auto_end = max(all_ends)
    _starts_sorted = sorted(all_starts)
    # 80th-percentile start date → 80 % of funds have data from here or earlier
    _p80_idx = min(int(len(_starts_sorted) * 0.80), len(_starts_sorted) - 1)
    auto_start = _starts_sorted[_p80_idx]
    if auto_start >= auto_end:
        auto_start = min(all_starts)  # fallback: earliest available

    dr1, dr2, dr3 = st.columns(3)
    align = dr1.radio("Period", ["Common start", "Custom"], horizontal=True,
                      key="tag_align")
    if align == "Custom":
        t_start = pd.Timestamp(dr2.date_input("From", value=auto_start.date(),
                                               key="tag_start"))
        t_end   = pd.Timestamp(dr3.date_input("To",   value=auto_end.date(),
                                               key="tag_end"))
    else:
        t_start, t_end = auto_start, auto_end
        dr2.metric("Common start", str(t_start.date()))
        dr3.metric("Common end",   str(t_end.date()))

    if t_start >= t_end:
        st.error("Start must be before end.")
        return

    # Slice to date range
    fund_navs = {
        k: v[(v.index >= t_start) & (v.index <= t_end)]
        for k, v in fund_navs.items()
        if len(v[(v.index >= t_start) & (v.index <= t_end)]) > 10
    }
    if bench_prices is not None:
        bench_prices = bench_prices[
            (bench_prices.index >= t_start) & (bench_prices.index <= t_end)
        ]
        if bench_prices.empty:
            bench_prices = None

    # ── Tabs ─────────────────────────────────────────────────────────────────
    (tab_growth, tab_returns, tab_rolling,
     tab_risk, tab_metrics, tab_manage) = st.tabs([
        "Growth Chart", "Returns", "Rolling Metrics",
        "Risk & Ratios", "Full Metrics", "Manage Tags",
    ])

    # ── Growth of ₹100 ───────────────────────────────────────────────────────
    with tab_growth:
        fig = go.Figure()
        # Per-fund lines
        normed_list = []
        for name, nav in fund_navs.items():
            normed = nav / nav.iloc[0] * 100
            normed_list.append(normed)
            fig.add_trace(go.Scatter(
                x=normed.index, y=normed.values, name=name,
                hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                line=dict(width=1.6),
            ))
        # Category average overlay
        if len(normed_list) > 1:
            normed_df = pd.concat(normed_list, axis=1).ffill(limit=5)
            cat_avg = normed_df.mean(axis=1).dropna()
            if len(cat_avg) > 1:
                fig.add_trace(go.Scatter(
                    x=cat_avg.index, y=cat_avg.values,
                    name=f"{selected_tag} Avg",
                    hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                    line=dict(width=2.5, dash="dot", color="darkorange"),
                ))
        if bench_prices is not None and len(bench_prices) > 0:
            bp = bench_prices / bench_prices.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=bp.index, y=bp.values, name=bench_name,
                hovertemplate="%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>",
                line=dict(width=2.5, dash="dash", color="black"),
            ))
        fig.add_hline(y=100, line_dash="dot", line_color="lightgray")
        fig.update_layout(
            title=f"Growth of ₹100 — {selected_tag}",
            yaxis_title="Value (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        add_range_buttons(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown comparison below
        st.markdown("#### Drawdown")
        fig_dd = go.Figure()
        for name, nav in fund_navs.items():
            dd = drawdown_series(nav)
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values * 100, name=name,
                hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
                line=dict(width=1.3),
            ))
        fig_dd.update_layout(yaxis_title="Drawdown (%)")
        fig_dd.add_hline(y=0, line_dash="dot", line_color="gray")
        add_range_buttons(fig_dd, height=380)
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Returns ───────────────────────────────────────────────────────────────
    with tab_returns:
        periods = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]

        # Trailing returns heatmap
        st.markdown("#### Trailing Returns — All Funds")
        tr_rows = []
        for name, nav in fund_navs.items():
            tr = trailing_returns(nav)
            row = {"Fund": name}
            for p in periods:
                val = tr.get(p, np.nan)
                row[p] = val * 100 if pd.notna(val) else np.nan
            row["Since Inception"] = cagr(nav) * 100
            tr_rows.append(row)

        tr_df = pd.DataFrame(tr_rows).set_index("Fund")
        # Color: green = positive return, red = negative; symmetric per column
        styled = tr_df.style.apply(
            _signed_gradient_col, axis=0
        ).format("{:.2f}%", na_rep="—")
        st.dataframe(styled, use_container_width=True, height=min(600, 40 + 35 * len(tr_df)))

        # Grouped bar chart
        st.markdown("#### Trailing Returns — Grouped Bar")
        fig_tr = go.Figure()
        for name, nav in fund_navs.items():
            tr = trailing_returns(nav)
            y_vals = [tr.get(p, np.nan) * 100 if pd.notna(tr.get(p, np.nan))
                      else None for p in periods]
            fig_tr.add_trace(go.Bar(
                name=name, x=periods, y=y_vals,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
        if bench_prices is not None:
            btr = trailing_returns(bench_prices)
            y_b = [btr.get(p, np.nan) * 100 if pd.notna(btr.get(p, np.nan))
                   else None for p in periods]
            fig_tr.add_trace(go.Bar(
                name=bench_name, x=periods, y=y_b,
                marker=dict(color="rgba(80,80,80,0.6)", pattern_shape="/"),
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
        fig_tr.update_layout(barmode="group", yaxis_title="Return (%)", height=440)
        fig_tr.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_tr, use_container_width=True)

        # Calendar year returns
        st.markdown("#### Calendar Year Returns")
        all_years: set[int] = set()
        cy_data: dict[str, pd.Series] = {}
        for name, nav in fund_navs.items():
            cy = calendar_year_returns(nav)
            cy_data[name] = cy
            all_years.update(cy.index.tolist())
        if bench_prices is not None:
            cy_b = calendar_year_returns(bench_prices)
            cy_data[bench_name] = cy_b
            all_years.update(cy_b.index.tolist())

        years_sorted = sorted(all_years)
        fig_cy = go.Figure()
        for name, cy in cy_data.items():
            is_b = name == bench_name
            y_vals = [cy.get(y, np.nan) * 100 if y in cy.index else None
                      for y in years_sorted]
            fig_cy.add_trace(go.Bar(
                name=name, x=[str(y) for y in years_sorted], y=y_vals,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                marker=dict(opacity=0.65 if is_b else 0.9,
                            pattern_shape="/" if is_b else ""),
            ))
        fig_cy.update_layout(barmode="group", height=440,
                             yaxis_title="Return (%)", xaxis_title="Year")
        fig_cy.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_cy, use_container_width=True)
        st.caption("First/last bars may be partial years.")

    # ── Rolling Metrics ───────────────────────────────────────────────────────
    with tab_rolling:
        rc1, rc2 = st.columns(2)
        rm_metric = rc1.selectbox(
            "Metric",
            ["Volatility", "Sharpe", "Sortino", "Returns", "Alpha", "Beta"],
            key="tag_rolling_metric",
        )
        rm_window = rc2.selectbox(
            "Window", list(ROLLING_WINDOWS.keys()), index=1,
            key="tag_rolling_window",
        )
        rm_days = ROLLING_WINDOWS[rm_window]

        needs_bench = rm_metric in ("Alpha", "Beta")
        if needs_bench and bench_prices is None:
            st.warning(f"Select a benchmark in the sidebar to view rolling {rm_metric}.")
        else:
            fig_rm = go.Figure()
            scale, ylabel = 1, rm_metric
            for name, nav in fund_navs.items():
                if rm_metric == "Volatility":
                    s = rolling_volatility(nav, rm_days)
                    scale, ylabel = 100, "Volatility (%)"
                elif rm_metric == "Sharpe":
                    s = rolling_sharpe(nav, rm_days)
                    scale, ylabel = 1, "Sharpe Ratio"
                elif rm_metric == "Sortino":
                    s = rolling_sortino(nav, rm_days)
                    scale, ylabel = 1, "Sortino Ratio"
                elif rm_metric == "Alpha":
                    s = rolling_alpha(nav, bench_prices, rm_days)
                    scale, ylabel = 100, "Alpha (%)"
                elif rm_metric == "Beta":
                    s = rolling_beta(nav, bench_prices, rm_days)
                    scale, ylabel = 1, "Beta"
                else:
                    s = rolling_returns(nav, rm_days)
                    scale, ylabel = 100, "Return (%)"

                if not s.empty:
                    fig_rm.add_trace(go.Scatter(
                        x=s.index, y=s.values * scale, name=name,
                        hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
                        line=dict(width=1.4),
                    ))

            if bench_prices is not None and rm_metric not in ("Alpha", "Beta"):
                if rm_metric == "Volatility":
                    bs = rolling_volatility(bench_prices, rm_days)
                elif rm_metric == "Sharpe":
                    bs = rolling_sharpe(bench_prices, rm_days)
                elif rm_metric == "Sortino":
                    bs = rolling_sortino(bench_prices, rm_days)
                else:
                    bs = rolling_returns(bench_prices, rm_days)
                if not bs.empty:
                    fig_rm.add_trace(go.Scatter(
                        x=bs.index, y=bs.values * scale, name=bench_name,
                        hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
                        line=dict(width=2.5, dash="dash", color="black"),
                    ))

            fig_rm.update_layout(
                title=f"Rolling {rm_window} {rm_metric} — {selected_tag}",
                yaxis_title=ylabel,
            )
            if rm_metric in ("Sharpe", "Sortino", "Alpha"):
                fig_rm.add_hline(y=0, line_dash="dot", line_color="gray")
            if rm_metric == "Beta":
                fig_rm.add_hline(y=1, line_dash="dot", line_color="gray")
            add_range_buttons(fig_rm, height=540)
            st.plotly_chart(fig_rm, use_container_width=True)

    # ── Risk & Ratios ─────────────────────────────────────────────────────────
    with tab_risk:
        # Risk-return scatter
        rr_rows = []
        for name, nav in fund_navs.items():
            v = annualized_volatility(nav)
            c = cagr(nav)
            sr = sharpe_ratio(nav)
            mdd = abs(nav.pipe(lambda p: ((p - p.cummax()) / p.cummax()).min()))
            if not np.isnan(v) and not np.isnan(c):
                rr_rows.append({
                    "Fund": name, "Volatility": v * 100,
                    "CAGR": c * 100, "Sharpe": sr,
                    "Max Drawdown": mdd * 100,
                })
        if rr_rows:
            rr_df = pd.DataFrame(rr_rows)

            st.markdown("#### Risk-Return Scatter")
            color_col = st.selectbox(
                "Color by", ["Sharpe", "Max Drawdown", "CAGR"],
                key="tag_scatter_color",
            )
            fig_rr = px.scatter(
                rr_df, x="Volatility", y="CAGR", color=color_col,
                text="Fund", size_max=14,
                color_continuous_scale="RdYlGn",
                hover_data={"Sharpe": ":.2f", "Max Drawdown": ":.1f%",
                            "Volatility": ":.1f%", "CAGR": ":.1f%"},
                labels={"Volatility": "Volatility (%)", "CAGR": "CAGR (%)"},
            )
            fig_rr.update_traces(
                textposition="top center",
                marker=dict(size=12, line=dict(color="white", width=1)),
            )
            fig_rr.update_layout(height=520)
            st.plotly_chart(fig_rr, use_container_width=True)

        # Risk-adjusted ratios bar chart
        st.markdown("#### Risk-Adjusted Ratios")
        fund_names_list = list(fund_navs.keys())
        fig_rar = go.Figure()
        for mname, fn in [("Sharpe", sharpe_ratio),
                           ("Sortino", sortino_ratio),
                           ("Calmar", calmar_ratio)]:
            y_vals = [fn(fund_navs[n]) for n in fund_names_list]
            fig_rar.add_trace(go.Bar(
                name=mname, x=fund_names_list,
                y=[v if (v is not None and np.isfinite(v)) else None for v in y_vals],
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            ))
        fig_rar.update_layout(
            barmode="group", height=440, yaxis_title="Ratio",
            xaxis_tickangle=-30,
        )
        fig_rar.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_rar, use_container_width=True)

        # Capture ratios (if benchmark)
        if bench_prices is not None:
            cap_rows = []
            for name, nav in fund_navs.items():
                uc = upside_capture(nav, bench_prices)
                dc = downside_capture(nav, bench_prices)
                if not np.isnan(uc) and not np.isnan(dc):
                    cap_rows.append({"Fund": name, "Upside": uc, "Downside": dc})
            if cap_rows:
                st.markdown(f"#### Capture Ratios vs {bench_name}")
                cap_df = pd.DataFrame(cap_rows)
                fig_cap = px.scatter(
                    cap_df, x="Downside", y="Upside", text="Fund",
                    labels={"Downside": "Downside Capture (%)",
                            "Upside": "Upside Capture (%)"},
                )
                fig_cap.update_traces(
                    textposition="top center",
                    marker=dict(size=11, line=dict(color="white", width=1)),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Upside: %{y:.1f}%  Downside: %{x:.1f}%"
                        "<extra></extra>"
                    ),
                )
                fig_cap.add_hline(y=100, line_dash="dot")
                fig_cap.add_vline(x=100, line_dash="dot")
                fig_cap.update_layout(height=500,
                    title="Top-left = ideal (High upside, Low downside)")
                st.plotly_chart(fig_cap, use_container_width=True)

    # ── Full Metrics Table ────────────────────────────────────────────────────
    with tab_metrics:
        with st.spinner(f"Computing metrics for {len(fund_navs)} funds…"):
            comp_df = compute_comparison_table(fund_navs, bench_prices)

        if not comp_df.empty:
            pct_rows = [r for r in comp_df.index
                        if any(k in r for k in [
                            "return", "cagr", "volatility", "deviation",
                            "drawdown", "alpha", "tracking", "var", "cvar",
                            "underwater", "batting", "sip_xirr", "positive",
                        ])]
            display_df = comp_df.copy().astype(object)
            for col in display_df.columns:
                for idx in display_df.index:
                    val = comp_df.loc[idx, col]
                    if pd.notna(val) and np.isfinite(float(val)):
                        fval = float(val)
                        display_df.loc[idx, col] = (
                            f"{fval*100:.2f}%" if idx in pct_rows
                            else f"{fval:.3f}"
                        )
                    else:
                        display_df.loc[idx, col] = "—"

            st.dataframe(display_df, use_container_width=True, height=750)

            # ── Within-category Quartile Ranking ─────────────────────────────
            st.markdown("#### Within-Category Quartile Ranking")
            st.caption("Q1 = best in category for each metric. "
                       "Lower-is-better metrics (risk, drawdown, VaR) are inverted.")

            _lower_is_better = {
                "volatility", "downside_deviation", "max_drawdown",
                "ulcer_index", "average_drawdown", "pct_time_underwater",
                "tracking_error", "var_95", "var_99", "cvar_95", "cvar_99",
                "worst_drawdown_depth", "worst_drawdown_duration",
                "downside_capture",
            }
            numeric_df = comp_df.apply(pd.to_numeric, errors="coerce")
            q_df = pd.DataFrame(index=comp_df.index, columns=comp_df.columns,
                                 dtype=object)
            n_funds = comp_df.shape[1]
            for idx in comp_df.index:
                row = numeric_df.loc[idx].dropna()
                if len(row) < 2:
                    q_df.loc[idx] = "—"
                    continue
                ascending = idx in _lower_is_better
                ranks = row.rank(ascending=ascending, method="min")
                for col in comp_df.columns:
                    if col not in ranks.index:
                        q_df.loc[idx, col] = "—"
                        continue
                    r = ranks[col] / n_funds
                    if r <= 0.25:
                        q_df.loc[idx, col] = "Q1"
                    elif r <= 0.50:
                        q_df.loc[idx, col] = "Q2"
                    elif r <= 0.75:
                        q_df.loc[idx, col] = "Q3"
                    else:
                        q_df.loc[idx, col] = "Q4"

            def _color_quartile(val):
                colors = {"Q1": "background-color:#c6efce;color:#276221",
                          "Q2": "background-color:#ffeb9c;color:#9c6500",
                          "Q3": "background-color:#ffcc99;color:#7a3000",
                          "Q4": "background-color:#ffc7ce;color:#9c0006"}
                return colors.get(val, "")

            st.dataframe(
                q_df.style.map(_color_quartile),
                use_container_width=True,
                height=750,
            )

            # Download button
            csv = comp_df.to_csv()
            st.download_button(
                "Download metrics as CSV",
                data=csv,
                file_name=f"{selected_tag.replace(' ', '_')}_metrics.csv",
                mime="text/csv",
            )

    # ── Manage Tags ───────────────────────────────────────────────────────────
    with tab_manage:
        st.markdown("#### Tag Management")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Rebuild all tags from rules**")
            st.caption(
                "Re-runs auto-tagging for all funds. Use after adding new funds "
                "or updating tagger rules."
            )
            if st.button("Rebuild All Tags", key="rebuild_tags"):
                with st.spinner("Tagging all funds…"):
                    counts = auto_tag_all_funds()
                _load_tag_navs.clear()
                _compute_tag_metrics.clear()
                st.success(
                    f"Tagged {sum(counts.values())} assignments across "
                    f"{len(counts)} tags."
                )
                st.rerun()

        with col_b:
            st.markdown("**Override tags for a specific fund**")
            all_funds_df = get_funds_with_nav()
            if not all_funds_df.empty:
                fund_opts = all_funds_df[["scheme_code","scheme_name"]].set_index("scheme_code")["scheme_name"]
                edit_code = st.selectbox(
                    "Fund", fund_opts.index.tolist(),
                    format_func=lambda x: fund_opts.get(x, str(x)),
                    key="tag_edit_fund",
                )
                current_tags = get_fund_tags(edit_code)
                all_possible = sorted({t for t, _ in get_all_tags()})
                new_tags = st.multiselect(
                    "Tags", all_possible, default=current_tags,
                    key="tag_edit_tags",
                )
                if st.button("Save Tags", key="tag_save"):
                    set_fund_tags(edit_code, new_tags)
                    _load_tag_navs.clear()
                    st.success("Tags updated.")
                    st.rerun()

        st.divider()
        st.markdown("**Current tag distribution**")
        tag_counts_df = pd.DataFrame(get_all_tags(), columns=["Tag", "Funds"])
        fig_tags = px.bar(
            tag_counts_df.sort_values("Funds", ascending=True),
            x="Funds", y="Tag", orientation="h",
            color="Funds", color_continuous_scale="Blues",
        )
        fig_tags.update_layout(height=max(400, 22 * len(tag_counts_df)),
                               showlegend=False, yaxis_title="")
        st.plotly_chart(fig_tags, use_container_width=True)


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------
if page == "Dashboard":
    render_dashboard()
elif page == "Fund Analysis":
    render_fund_analysis()
elif page == "Fund Comparison":
    render_comparison()
elif page == "Sector & Style":
    render_sector_style()
elif page == "Data Management":
    render_data_management()
