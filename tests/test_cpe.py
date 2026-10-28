import pandas as pd
import pytest

from cpe.anomalies import detect_anomalies
from cpe.data_generator import generate_campaigns
from cpe.forecasting import train, what_if
from cpe.metrics import add_kpis, aggregate, summary, validate
from cpe.report import evaluate, write_reports
from cpe.scoring import budget_reallocation, score_campaigns


@pytest.fixture(scope="module")
def daily():
    return validate(generate_campaigns(n_campaigns=12, days=60, seed=7))


def test_kpi_formulas():
    df = pd.DataFrame({"impressions": [10_000], "clicks": [200], "conversions": [10],
                       "spend": [400.0], "revenue": [1_200.0]})
    k = add_kpis(df).iloc[0]
    assert k["ctr"] == pytest.approx(0.02)
    assert k["cpc"] == pytest.approx(2.0)
    assert k["cpm"] == pytest.approx(40.0)
    assert k["cvr"] == pytest.approx(0.05)
    assert k["cpa"] == pytest.approx(40.0)
    assert k["roas"] == pytest.approx(3.0)
    assert k["roi"] == pytest.approx(2.0)


def test_zero_denominators_do_not_crash():
    df = pd.DataFrame({"impressions": [0], "clicks": [0], "conversions": [0],
                       "spend": [0.0], "revenue": [0.0]})
    k = add_kpis(df).iloc[0]
    assert k[["ctr", "cpc", "cvr", "cpa", "roas"]].eq(0).all()


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="spend"):
        validate(pd.DataFrame({"date": [], "campaign_id": [], "channel": [],
                               "impressions": [], "clicks": [], "conversions": [],
                               "revenue": []}))


def test_aggregate_recomputes_ratios_from_sums():
    df = pd.DataFrame({"campaign_id": ["A", "A"], "impressions": [100, 900],
                       "clicks": [50, 10], "conversions": [0, 0],
                       "spend": [1.0, 1.0], "revenue": [0.0, 0.0]})
    # mean of daily CTRs would be 0.505; correct pooled CTR is 60/1000
    assert aggregate(df, "campaign_id")["ctr"].iloc[0] == pytest.approx(0.06)


def test_scores_are_bounded_and_sorted(daily):
    scored = score_campaigns(aggregate(daily, "campaign_id"))
    assert scored["performance_score"].between(0, 100).all()
    assert scored["performance_score"].is_monotonic_decreasing
    assert set(scored["grade"]) <= set("ABCDF")


def test_unprofitable_campaign_is_paused(daily):
    camps = aggregate(daily, "campaign_id")
    camps.loc[0, "revenue"] = camps.loc[0, "spend"] * 0.5
    camps = add_kpis(camps)
    scored = score_campaigns(camps)
    row = scored[scored["campaign_id"] == camps.loc[0, "campaign_id"]].iloc[0]
    assert row["recommendation"].startswith("Pause")


def test_budget_reallocation_conserves_total(daily):
    scored = score_campaigns(aggregate(daily, "campaign_id"))
    plan = budget_reallocation(scored)
    assert plan["suggested_budget"].sum() == pytest.approx(scored["spend"].sum(), rel=1e-4)
    paused = plan["recommendation"].str.startswith("Pause")
    assert (plan.loc[paused, "suggested_budget"] == 0).all()


def test_detects_injected_tracking_outage(daily):
    df = daily.copy()
    idx = df[(df["campaign_id"] == "CMP-001") & (df["clicks"] > 50)].index[10]
    df.loc[idx, ["conversions", "revenue"]] = 0
    found = detect_anomalies(df)
    hit = found[(found["campaign_id"] == "CMP-001") & (found["date"] == df.loc[idx, "date"])]
    assert not hit.empty
    assert "tracking" in " ".join(hit["likely_cause"]).lower()


def test_forecast_quality_and_monotonic_spend(daily):
    result = train(daily)
    assert result.r2 > 0.5
    out = what_if(result, daily, "CMP-001", [100, 200, 400])
    assert out["predicted_conversions"].is_monotonic_increasing


def test_end_to_end_reports(tmp_path, daily):
    ev = evaluate(daily)
    assert summary(daily)["spend"] == pytest.approx(daily["spend"].sum())
    paths = write_reports(ev, tmp_path)
    assert all(p.exists() for p in paths)
    assert "Campaign scorecard" in (tmp_path / "report.html").read_text()
