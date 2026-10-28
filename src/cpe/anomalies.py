"""Daily anomaly detection per campaign using robust z-scores."""
from __future__ import annotations

import pandas as pd

from .metrics import add_kpis


def detect_anomalies(daily: pd.DataFrame, threshold: float = 3.5,
                     kpis: tuple[str, ...] = ("ctr", "cvr", "cpc", "roas")) -> pd.DataFrame:
    """Flag days whose KPI deviates strongly from that campaign's median.

    Uses the modified z-score (median / MAD), which is not skewed by the
    very outliers it is trying to detect.
    """
    df = add_kpis(daily)
    found = []
    for kpi in kpis:
        grouped = df.groupby("campaign_id")[kpi]
        median = grouped.transform("median")
        mad = grouped.transform(lambda s: (s - s.median()).abs().median())
        z = 0.6745 * (df[kpi] - median) / mad.where(mad > 0)
        hits = df[z.abs() >= threshold].copy()
        if hits.empty:
            continue
        hits["kpi"] = kpi
        hits["value"] = hits[kpi]
        hits["expected"] = median[hits.index]
        hits["z_score"] = z[hits.index].round(2)
        hits["direction"] = hits["z_score"].apply(lambda v: "spike" if v > 0 else "drop")
        found.append(hits)

    if not found:
        return pd.DataFrame(columns=["date", "campaign_id", "campaign_name", "kpi",
                                     "value", "expected", "z_score", "direction", "likely_cause"])
    out = pd.concat(found)
    out["likely_cause"] = out.apply(_likely_cause, axis=1)
    cols = ["date", "campaign_id", "campaign_name", "channel", "kpi", "value",
            "expected", "z_score", "direction", "likely_cause"]
    return out[cols].sort_values(["date", "campaign_id"]).reset_index(drop=True)


def _likely_cause(row: pd.Series) -> str:
    if row["kpi"] in ("cvr", "roas") and row["direction"] == "drop" and row["value"] == 0:
        return "Possible conversion-tracking outage"
    if row["kpi"] == "ctr" and row["direction"] == "spike":
        return "Possible bot / invalid traffic"
    if row["kpi"] == "cpc" and row["direction"] == "spike":
        return "Auction competition or bid change"
    if row["kpi"] == "cpc" and row["direction"] == "drop":
        return "Cheap clicks - check traffic quality"
    return "Investigate"
