"""Synthetic daily campaign data generator.

Produces realistic multi-channel marketing data so the system can be
demonstrated without access to a real ad account.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# channel -> (base CTR, base CVR, base CPC, avg order value)
CHANNEL_PROFILES = {
    "Google Search": (0.045, 0.060, 1.20, 85.0),
    "Meta Ads": (0.012, 0.025, 0.65, 60.0),
    "Instagram": (0.009, 0.020, 0.70, 55.0),
    "LinkedIn": (0.006, 0.040, 4.50, 220.0),
    "YouTube": (0.004, 0.012, 0.35, 70.0),
    "Email": (0.030, 0.030, 0.40, 75.0),
}

OBJECTIVES = ["Awareness", "Traffic", "Conversions", "Retargeting"]
REGIONS = ["North", "South", "East", "West"]


def generate_campaigns(n_campaigns: int = 24, days: int = 90,
                       start: str = "2026-01-01", seed: int = 42) -> pd.DataFrame:
    """Return one row per campaign per day."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    channels = list(CHANNEL_PROFILES)
    rows = []

    for i in range(n_campaigns):
        channel = channels[i % len(channels)]
        ctr, cvr, cpc, aov = CHANNEL_PROFILES[channel]
        objective = rng.choice(OBJECTIVES)
        region = rng.choice(REGIONS)
        # campaign-level quality multiplier: some campaigns are simply better
        quality = rng.lognormal(0, 0.35)
        daily_budget = rng.uniform(80, 900)

        for day_idx, date in enumerate(dates):
            weekend = 0.85 if date.day_of_week >= 5 else 1.0
            fatigue = max(0.6, 1 - 0.003 * day_idx)  # creative fatigue over time
            spend = daily_budget * rng.uniform(0.8, 1.1) * weekend
            day_cpc = cpc * rng.uniform(0.85, 1.2) / quality ** 0.3
            clicks = max(int(spend / day_cpc), 0)
            day_ctr = ctr * quality * fatigue * rng.uniform(0.8, 1.2)
            impressions = max(int(clicks / max(day_ctr, 1e-4)), clicks)
            day_cvr = min(cvr * quality * rng.uniform(0.7, 1.3), 0.5)
            conversions = int(rng.binomial(clicks, day_cvr)) if clicks else 0
            revenue = conversions * aov * rng.uniform(0.8, 1.25)

            rows.append({
                "date": date,
                "campaign_id": f"CMP-{i + 1:03d}",
                "campaign_name": f"{channel} {objective} {region} #{i + 1}",
                "channel": channel,
                "objective": objective,
                "region": region,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": round(spend, 2),
                "revenue": round(revenue, 2),
            })

    df = pd.DataFrame(rows)
    _inject_anomalies(df, rng)
    return df


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator, n: int = 6) -> None:
    """Simulate tracking outages and bot-traffic spikes for the detector to find."""
    for idx in rng.choice(df.index, size=n, replace=False):
        if rng.random() < 0.5:  # tracking outage
            df.loc[idx, ["conversions", "revenue"]] = 0
        else:  # click spike with no conversions (bot traffic)
            df.loc[idx, "clicks"] *= 6
            df.loc[idx, "impressions"] = max(df.loc[idx, "impressions"], df.loc[idx, "clicks"])
            df.loc[idx, ["conversions", "revenue"]] = 0


if __name__ == "__main__":
    out = generate_campaigns()
    out.to_csv("data/campaigns.csv", index=False)
    print(f"Wrote {len(out)} rows to data/campaigns.csv")
