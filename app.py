"""Streamlit dashboard for the Campaign Performance Evaluation System.

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the package importable when deployed without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cpe.data_generator import generate_campaigns
from cpe.forecasting import train, what_if
from cpe.metrics import aggregate
from cpe.report import evaluate

st.set_page_config(page_title="Campaign Performance Evaluator", layout="wide")
st.title("Digital Marketing Campaign Performance Evaluation")

# ---------- Sidebar: data source and settings ----------
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload campaign CSV", type="csv")
    st.caption("Required columns: date, campaign_id, channel, impressions, "
               "clicks, conversions, spend, revenue")
    target_roas = st.slider("Target ROAS", 1.0, 6.0, 2.0, 0.25)


@st.cache_data
def load(file) -> pd.DataFrame:
    return pd.read_csv(file) if file is not None else generate_campaigns()


try:
    ev = evaluate(load(upload), target_roas=target_roas)
except ValueError as err:
    st.error(str(err))
    st.stop()

daily = ev.daily
with st.sidebar:
    st.header("Filters")
    channels = st.multiselect("Channels", sorted(daily["channel"].unique()),
                              default=sorted(daily["channel"].unique()))
    lo, hi = daily["date"].min().date(), daily["date"].max().date()
    date_range = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)

if len(date_range) == 2 and channels:
    mask = (daily["channel"].isin(channels)
            & daily["date"].dt.date.between(date_range[0], date_range[1]))
    if mask.sum() != len(daily):
        ev = evaluate(daily[mask], target_roas=target_roas)
        daily = ev.daily
elif not channels:
    st.warning("Select at least one channel.")
    st.stop()

# ---------- KPI tiles ----------
o = ev.overview
cols = st.columns(6)
cols[0].metric("Spend", f"{o['spend']:,.0f}")
cols[1].metric("Revenue", f"{o['revenue']:,.0f}")
cols[2].metric("ROAS", f"{o['roas']:.2f}x")
cols[3].metric("ROI", f"{o['roi']:.1%}")
cols[4].metric("CVR", f"{o['cvr']:.2%}")
cols[5].metric("CPA", f"{o['cpa']:,.2f}")

tab_score, tab_channel, tab_trend, tab_anom, tab_budget, tab_forecast = st.tabs(
    ["Scorecard", "Channels", "Trends", "Anomalies", "Budget", "Forecast"])

with tab_score:
    st.subheader("Campaign scorecard")
    show = ev.campaigns[["campaign_id", "campaign_name", "channel", "spend", "revenue",
                         "roas", "ctr", "cvr", "cpa", "performance_score", "grade",
                         "recommendation"]]
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "roas": st.column_config.NumberColumn(format="%.2fx"),
        "performance_score": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                             format="%.1f"),
    })
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(ev.campaigns, x="grade", color="grade",
                                 category_orders={"grade": list("ABCDF")},
                                 title="Grade distribution"), width="stretch")
    c2.plotly_chart(px.scatter(ev.campaigns, x="spend", y="roas", size="conversions",
                               color="channel", hover_name="campaign_name",
                               title="Spend vs ROAS (bubble = conversions)"),
                    width="stretch")

with tab_channel:
    st.subheader("Channel comparison")
    metric = st.selectbox("Metric", ["roas", "roi", "ctr", "cvr", "cpc", "cpa", "cpm"])
    st.plotly_chart(px.bar(ev.channels, x="channel", y=metric, color="channel",
                           title=f"{metric.upper()} by channel"), width="stretch")
    st.plotly_chart(px.pie(ev.channels, names="channel", values="spend",
                           title="Spend share", hole=0.5), width="stretch")

with tab_trend:
    st.subheader("Weekly trends")
    weekly = aggregate(daily.assign(week=daily["date"].dt.to_period("W").dt.start_time),
                       ["week", "channel"])
    kpi = st.selectbox("KPI", ["roas", "spend", "revenue", "conversions", "ctr", "cvr", "cpa"])
    st.plotly_chart(px.line(weekly, x="week", y=kpi, color="channel", markers=True),
                    width="stretch")

with tab_anom:
    st.subheader(f"Detected anomalies ({len(ev.anomalies)})")
    st.caption("Days where a KPI deviates from the campaign's median by a robust "
               "z-score of 3.5 or more.")
    st.dataframe(ev.anomalies, width="stretch", hide_index=True)

with tab_budget:
    st.subheader("Suggested budget reallocation")
    st.caption("Same total budget, redistributed by performance score x ROAS. "
               "Paused campaigns receive nothing. Projections assume ROAS holds.")
    b = ev.budget
    st.metric("Projected revenue at same total spend",
              f"{b['projected_revenue'].sum():,.0f}",
              f"{b['projected_revenue'].sum() - o['revenue']:,.0f} vs actual")
    st.plotly_chart(px.bar(b, x="campaign_id", y="change", color="channel",
                           title="Budget change per campaign"), width="stretch")
    st.dataframe(b, width="stretch", hide_index=True)

with tab_forecast:
    st.subheader("What-if spend forecast")


    @st.cache_resource
    def get_model(data: pd.DataFrame):
        return train(data)

    result = get_model(daily)
    st.caption(f"Random forest, 14-day time holdout: MAE {result.mae:.2f} "
               f"conversions/day, R² {result.r2:.3f}")
    camp = st.selectbox("Campaign", ev.campaigns["campaign_id"])
    avg = daily[daily["campaign_id"] == camp]["spend"].mean()
    levels = [round(avg * m, 2) for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    st.dataframe(what_if(result, daily, camp, levels), hide_index=True)
    st.plotly_chart(px.bar(result.feature_importance.head(10)[::-1], orientation="h",
                           title="Top feature importances"), width="stretch")

st.download_button("Download scorecard CSV", ev.campaigns.to_csv(index=False),
                   "campaign_scores.csv", "text/csv")
