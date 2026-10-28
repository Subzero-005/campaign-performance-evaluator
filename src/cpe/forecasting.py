"""ML model that predicts daily conversions from spend and context.

Lets marketers ask "what happens if I spend X on campaign Y tomorrow?".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["channel", "objective", "region"]
NUMERIC = ["spend", "day_of_week", "days_running", "hist_cvr", "hist_cpc"]


@dataclass
class ForecastResult:
    model: Pipeline
    mae: float
    r2: float
    feature_importance: pd.Series


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["campaign_id", "date"]).copy()
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "Unknown"
    df["day_of_week"] = df["date"].dt.day_of_week
    df["days_running"] = df.groupby("campaign_id").cumcount()
    # Historical (lagged, expanding) efficiency: only uses past days, no leakage.
    g = df.groupby("campaign_id")
    past_clicks = g["clicks"].transform(lambda s: s.shift().expanding().sum())
    past_conv = g["conversions"].transform(lambda s: s.shift().expanding().sum())
    past_spend = g["spend"].transform(lambda s: s.shift().expanding().sum())
    df["hist_cvr"] = (past_conv / past_clicks).replace([np.inf, -np.inf], np.nan)
    df["hist_cpc"] = (past_spend / past_clicks).replace([np.inf, -np.inf], np.nan)
    df[["hist_cvr", "hist_cpc"]] = df[["hist_cvr", "hist_cpc"]].fillna(
        df[["hist_cvr", "hist_cpc"]].median())
    return df


def train(daily: pd.DataFrame, holdout_days: int = 14, seed: int = 42) -> ForecastResult:
    """Train on all but the last `holdout_days`, evaluate on those (time split)."""
    df = build_features(daily).dropna(subset=["hist_cvr", "hist_cpc"])
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train_df, test_df = df[df["date"] <= cutoff], df[df["date"] > cutoff]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                     random_state=seed, n_jobs=-1)),
    ])
    # Target is conversions per unit of spend. Trees cannot extrapolate, so
    # predicting efficiency and multiplying by spend keeps what-if answers
    # sensible for budgets outside a campaign's historical range.
    model.fit(train_df[CATEGORICAL + NUMERIC], _efficiency(train_df))

    pred = predict_conversions(model, test_df)
    names = model.named_steps["pre"].get_feature_names_out()
    importance = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
    return ForecastResult(
        model=model,
        mae=float(mean_absolute_error(test_df["conversions"], pred)),
        r2=float(r2_score(test_df["conversions"], pred)),
        feature_importance=importance.sort_values(ascending=False),
    )


def _efficiency(df: pd.DataFrame) -> pd.Series:
    return (df["conversions"] / df["spend"].where(df["spend"] > 0)).fillna(0)


def predict_conversions(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X[CATEGORICAL + NUMERIC]), 0, None) * X["spend"].to_numpy(float)


def what_if(result: ForecastResult, daily: pd.DataFrame, campaign_id: str,
            spend_levels: list[float]) -> pd.DataFrame:
    """Predict next-day conversions for a campaign at several spend levels."""
    feats = build_features(daily)
    last = feats[feats["campaign_id"] == campaign_id].iloc[-1]
    rows = []
    for spend in spend_levels:
        row = last[CATEGORICAL + NUMERIC].copy()
        row["spend"] = spend
        row["days_running"] = last["days_running"] + 1
        row["day_of_week"] = (last["day_of_week"] + 1) % 7
        rows.append(row)
    X = pd.DataFrame(rows)
    out = pd.DataFrame({"spend": spend_levels,
                        "predicted_conversions": predict_conversions(result.model, X).round(1)})
    out["predicted_cpa"] = (out["spend"] / out["predicted_conversions"].replace(0, np.nan)).round(2)
    return out
