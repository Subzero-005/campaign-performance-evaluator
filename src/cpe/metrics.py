"""Core marketing KPI calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "campaign_id", "channel", "impressions",
                    "clicks", "conversions", "spend", "revenue"]

KPI_COLUMNS = ["ctr", "cpc", "cpm", "cvr", "cpa", "roas", "roi"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and clean obviously bad values."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    numeric = ["impressions", "clicks", "conversions", "spend", "revenue"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    if "campaign_name" not in df.columns:
        df["campaign_name"] = df["campaign_id"]
    return df


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add KPI columns computed from summed raw metrics.

    Works on daily rows or on any aggregated frame, as long as the raw
    columns are sums (never average a ratio).
    """
    df = df.copy()
    df["ctr"] = _safe_div(df["clicks"], df["impressions"])
    df["cpc"] = _safe_div(df["spend"], df["clicks"])
    df["cpm"] = _safe_div(df["spend"], df["impressions"]) * 1000
    df["cvr"] = _safe_div(df["conversions"], df["clicks"])
    df["cpa"] = _safe_div(df["spend"], df["conversions"])
    df["roas"] = _safe_div(df["revenue"], df["spend"])
    df["roi"] = _safe_div(df["revenue"] - df["spend"], df["spend"])
    df["profit"] = df["revenue"] - df["spend"]
    return df


def aggregate(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Sum raw metrics by the given key(s) and recompute KPIs."""
    keys = [by] if isinstance(by, str) else list(by)
    sums = ["impressions", "clicks", "conversions", "spend", "revenue"]
    extra = {}
    if "campaign_id" in keys:
        for col in ("campaign_name", "channel", "objective", "region"):
            if col in df.columns and col not in keys:
                extra[col] = "first"
    agg = df.groupby(keys, as_index=False).agg({**{c: "sum" for c in sums}, **extra})
    return add_kpis(agg)


def summary(df: pd.DataFrame) -> dict:
    """Portfolio-level totals and KPIs."""
    totals = df[["impressions", "clicks", "conversions", "spend", "revenue"]].sum()
    kpis = add_kpis(totals.to_frame().T).iloc[0]
    return {**totals.to_dict(), **{k: float(kpis[k]) for k in KPI_COLUMNS + ["profit"]}}
