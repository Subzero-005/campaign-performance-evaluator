"""Composite campaign performance scoring, grading and recommendations."""
from __future__ import annotations

import pandas as pd

# KPI -> (weight, higher_is_better)
DEFAULT_WEIGHTS = {
    "roas": (0.35, True),
    "cvr": (0.20, True),
    "ctr": (0.15, True),
    "cpa": (0.20, False),
    "cpc": (0.10, False),
}

GRADE_BANDS = [(80, "A"), (65, "B"), (50, "C"), (35, "D"), (0, "F")]


def _percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Rank-based 0-100 score; robust to outliers and different KPI scales."""
    if series.nunique() <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = series.rank(pct=True, ascending=higher_is_better)
    return (ranks - ranks.min()) / (ranks.max() - ranks.min()) * 100


def score_campaigns(campaigns: pd.DataFrame,
                    weights: dict[str, tuple[float, bool]] | None = None,
                    target_roas: float = 2.0) -> pd.DataFrame:
    """Score aggregated campaign rows (output of metrics.aggregate)."""
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(w for w, _ in weights.values())
    df = campaigns.copy()

    # Campaigns with zero conversions have cpa=0; treat that as worst, not best.
    cpa = df["cpa"].where(df["conversions"] > 0, df["cpa"].max() * 2 + 1)

    score = pd.Series(0.0, index=df.index)
    for kpi, (w, higher) in weights.items():
        values = cpa if kpi == "cpa" else df[kpi]
        df[f"{kpi}_score"] = _percentile_score(values, higher)
        score += df[f"{kpi}_score"] * w / total_w

    df["performance_score"] = score.round(1)
    df["grade"] = df["performance_score"].apply(_grade)
    df["recommendation"] = df.apply(_recommend, axis=1, target_roas=target_roas)
    return df.sort_values("performance_score", ascending=False).reset_index(drop=True)


def _grade(score: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def _recommend(row: pd.Series, target_roas: float) -> str:
    if row["roas"] >= target_roas * 1.5 and row["performance_score"] >= 65:
        return "Scale: increase budget"
    if row["roas"] >= target_roas and row["performance_score"] >= 50:
        return "Maintain: performing on target"
    if row["roas"] >= 1.0:
        if row["ctr_score"] < 40:
            return "Optimize: refresh creatives / targeting (low CTR)"
        if row["cvr_score"] < 40:
            return "Optimize: improve landing page (low CVR)"
        return "Optimize: tighten bidding to lower CPA"
    return "Pause: spend exceeds revenue"


def budget_reallocation(scored: pd.DataFrame, total_budget: float | None = None,
                        min_share: float = 0.02, returns_exponent: float = 1.0) -> pd.DataFrame:
    """Suggest a budget split proportional to performance score x ROAS.

    Paused campaigns get nothing; every other campaign keeps at least
    `min_share` of the budget so it can still gather data.

    `returns_exponent` controls how revenue responds to the new spend:
    revenue = current_revenue * (new_spend / current_spend) ** returns_exponent.
    1.0 assumes ROAS holds at any spend (optimistic); values below 1 model
    diminishing returns, e.g. 0.7 means doubling spend raises revenue by ~62%.
    """
    if not 0 < returns_exponent <= 1:
        raise ValueError("returns_exponent must be in (0, 1]")
    df = scored[["campaign_id", "campaign_name", "channel", "spend",
                 "roas", "performance_score", "recommendation"]].copy()
    total_budget = total_budget if total_budget is not None else df["spend"].sum()
    active = ~df["recommendation"].str.startswith("Pause")
    weight = (df["performance_score"] * df["roas"]).where(active, 0)

    if weight.sum() == 0:
        df["suggested_budget"] = df["spend"]
    else:
        share = weight / weight.sum()
        share = share.where(~active, share.clip(lower=min_share))
        share = share / share.sum()
        df["suggested_budget"] = (share * total_budget).round(2)

    df["change"] = df["suggested_budget"] - df["spend"]
    df["change_pct"] = (df["change"] / df["spend"].where(df["spend"] > 0)).fillna(0)
    current_revenue = df["spend"] * df["roas"]
    ratio = (df["suggested_budget"] / df["spend"].where(df["spend"] > 0)).fillna(0)
    df["projected_revenue"] = current_revenue * ratio ** returns_exponent
    return df.sort_values("suggested_budget", ascending=False).reset_index(drop=True)
