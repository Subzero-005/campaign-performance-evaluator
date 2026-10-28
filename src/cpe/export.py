"""Export an evaluation as a single JSON file for a static web dashboard.

The JSON holds everything the Streamlit dashboard shows (KPIs, scorecard,
channels, weekly trends, anomalies, budget plan and spend forecasts), so the
results can be published on any static host without running Python there.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .forecasting import train, what_if
from .metrics import aggregate
from .report import Evaluation

SPEND_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _records(df: pd.DataFrame) -> list[dict]:
    # to_json turns NaN into null and timestamps into ISO strings
    return json.loads(df.to_json(orient="records", date_format="iso", double_precision=6))


def _daily_by_campaign(daily: pd.DataFrame) -> dict:
    """Compact per-campaign daily series for drill-down charts (columns, not rows)."""
    cols = ["impressions", "clicks", "conversions", "spend", "revenue"]
    out = {}
    for camp, g in daily.sort_values("date").groupby("campaign_id"):
        series = {"date": g["date"].dt.strftime("%Y-%m-%d").tolist()}
        for c in cols:
            series[c] = [round(float(v), 2) for v in g[c]]
        out[str(camp)] = series
    return out


def build_payload(ev: Evaluation, target_roas: float, with_forecast: bool = True) -> dict:
    daily = ev.daily
    weekly = aggregate(daily.assign(week=daily["date"].dt.to_period("W").dt.start_time),
                       ["week", "channel"])
    weekly["week"] = weekly["week"].dt.strftime("%Y-%m-%d")

    anomalies = ev.anomalies.copy()
    if "date" in anomalies:
        anomalies["date"] = pd.to_datetime(anomalies["date"]).dt.strftime("%Y-%m-%d")

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_roas": target_roas,
        "period": {
            "start": daily["date"].min().strftime("%Y-%m-%d"),
            "end": daily["date"].max().strftime("%Y-%m-%d"),
            "campaigns": int(daily["campaign_id"].nunique()),
            "channels": int(daily["channel"].nunique()),
        },
        "overview": {k: float(v) for k, v in ev.overview.items()},
        "campaigns": _records(ev.campaigns),
        "channels": _records(ev.channels),
        "weekly": _records(weekly),
        "anomalies": _records(anomalies),
        "budget": _records(ev.budget),
        "daily": _daily_by_campaign(daily),
    }

    if with_forecast:
        result = train(daily)
        whatif = {}
        for camp in ev.campaigns["campaign_id"]:
            avg = daily.loc[daily["campaign_id"] == camp, "spend"].mean()
            levels = [round(avg * m, 2) for m in SPEND_MULTIPLIERS]
            whatif[camp] = _records(what_if(result, daily, camp, levels))
        payload["forecast"] = {
            "mae": result.mae,
            "r2": result.r2,
            "feature_importance": [
                {"feature": name, "importance": float(v)}
                for name, v in result.feature_importance.head(10).items()
            ],
            "what_if": whatif,
        }
    return payload


def export_json(ev: Evaluation, path: str | Path, target_roas: float, with_forecast: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_payload(ev, target_roas, with_forecast), separators=(",", ":")))
    return path
