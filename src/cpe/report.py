"""Builds the full evaluation and writes CSV + HTML reports."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd

from .anomalies import detect_anomalies
from .metrics import aggregate, summary, validate
from .scoring import budget_reallocation, score_campaigns


@dataclass
class Evaluation:
    daily: pd.DataFrame
    overview: dict
    campaigns: pd.DataFrame
    channels: pd.DataFrame
    anomalies: pd.DataFrame
    budget: pd.DataFrame


def evaluate(raw: pd.DataFrame, target_roas: float = 2.0) -> Evaluation:
    daily = validate(raw)
    campaigns = score_campaigns(aggregate(daily, "campaign_id"), target_roas=target_roas)
    channels = aggregate(daily, "channel").sort_values("roas", ascending=False)
    return Evaluation(
        daily=daily,
        overview=summary(daily),
        campaigns=campaigns,
        channels=channels.reset_index(drop=True),
        anomalies=detect_anomalies(daily),
        budget=budget_reallocation(campaigns),
    )


def write_reports(ev: Evaluation, out_dir: str | Path = "reports") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "campaign_scores.csv": ev.campaigns,
        "channel_summary.csv": ev.channels,
        "anomalies.csv": ev.anomalies,
        "budget_recommendation.csv": ev.budget,
    }
    paths = []
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
        paths.append(out / name)
    html_path = out / "report.html"
    html_path.write_text(_html(ev), encoding="utf-8")
    paths.append(html_path)
    return paths


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for c in cols:
        if c in ("ctr", "cvr", "roi", "change_pct"):
            view[c] = view[c].map("{:.2%}".format)
        elif c in ("spend", "revenue", "cpa", "cpc", "profit", "suggested_budget", "change"):
            view[c] = view[c].map("{:,.2f}".format)
        elif c == "roas":
            view[c] = view[c].map("{:.2f}x".format)
    return view.to_html(index=False, border=0, classes="t", escape=True)


def _html(ev: Evaluation) -> str:
    o = ev.overview
    tiles = [("Spend", f"{o['spend']:,.0f}"), ("Revenue", f"{o['revenue']:,.0f}"),
             ("ROAS", f"{o['roas']:.2f}x"), ("ROI", f"{o['roi']:.1%}"),
             ("CTR", f"{o['ctr']:.2%}"), ("CVR", f"{o['cvr']:.2%}"),
             ("CPA", f"{o['cpa']:,.2f}"), ("Conversions", f"{o['conversions']:,.0f}")]
    tile_html = "".join(f"<div class='tile'><span>{escape(k)}</span><b>{v}</b></div>"
                        for k, v in tiles)
    start, end = ev.daily["date"].min().date(), ev.daily["date"].max().date()
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Campaign Performance Report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:32px;color:#1f2933;background:#fff}}
h1{{margin-bottom:4px}} .muted{{color:#6b7280}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:20px 0}}
.tile{{border:1px solid #e5e7eb;border-radius:8px;padding:12px}}
.tile span{{display:block;color:#6b7280;font-size:13px}} .tile b{{font-size:22px}}
table.t{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:28px}}
table.t th,table.t td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}}
table.t th{{background:#f9fafb}}
</style></head><body>
<h1>Campaign Performance Report</h1>
<p class="muted">{start} to {end} &middot; {ev.campaigns.shape[0]} campaigns &middot;
{ev.channels.shape[0]} channels</p>
<div class="tiles">{tile_html}</div>
<h2>Campaign scorecard</h2>
{_fmt(ev.campaigns, ["campaign_id", "campaign_name", "channel", "spend", "revenue", "roas",
                     "ctr", "cvr", "cpa", "performance_score", "grade", "recommendation"])}
<h2>Channel comparison</h2>
{_fmt(ev.channels, ["channel", "spend", "revenue", "roas", "roi", "ctr", "cvr", "cpc", "cpa"])}
<h2>Budget reallocation</h2>
{_fmt(ev.budget, ["campaign_id", "channel", "spend", "suggested_budget", "change", "change_pct", "roas"])}
<h2>Anomalies ({len(ev.anomalies)})</h2>
{_fmt(ev.anomalies, ["date", "campaign_id", "kpi", "direction", "z_score", "likely_cause"])
 if len(ev.anomalies) else "<p>No anomalies detected.</p>"}
</body></html>"""
