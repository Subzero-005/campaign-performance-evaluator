# Digital Marketing Campaign Performance Evaluation System

**▶ [Open the live dashboard](https://subzero-005.github.io/campaigns/)**: scorecard, channels, trends, anomalies, budget plan and spend forecasts for the sample dataset. You can re-weight the scoring model live, simulate budgets with diminishing returns, drag a spend slider on the forecast, and click any campaign for its daily chart. The browser reproduces the Python scores, grades, recommendations and budgets exactly at default settings.

A Python system that takes daily ad-campaign data from any channel (Google Search, Meta, Instagram, LinkedIn, YouTube, Email), works out standard marketing KPIs, gives each campaign a score and grade, finds unusual days, suggests how to split the budget, and predicts conversions at different spend levels.

It comes with a CLI that writes CSV and HTML reports, and an interactive Streamlit dashboard.

## Features

| Module | What it does |
|---|---|
| `metrics.py` | Checks input data and computes **CTR, CPC, CPM, CVR, CPA, ROAS, ROI, profit**. Ratios are always calculated from summed raw metrics, never by averaging daily ratios. |
| `scoring.py` | Builds a **composite performance score (0–100)** from weighted, rank-normalised KPIs, gives an **A–F grade**, and makes a rule-based **recommendation** (Scale / Maintain / Optimize with a reason / Pause). It also suggests a **budget reallocation** that keeps the total budget the same, with an optional **diminishing-returns** model for the revenue projection. |
| `anomalies.py` | Uses a **robust z-score (median/MAD)** for each campaign to flag unusual days, with a likely cause such as a tracking outage, bot traffic or an auction change. |
| `forecasting.py` | A **Random Forest** model predicts conversions per unit of spend from channel, objective, region, day of week, campaign age and past efficiency (lagged, so there is no leakage). It is checked on a 14-day time-based holdout and supports **what-if spend analysis**. |
| `report.py` | Runs the whole pipeline and exports the CSVs and an HTML report. |
| `export.py` | Writes all results, including per-campaign forecasts, to one JSON file that a static web dashboard can load. This is what powers the live dashboard. |
| `data_generator.py` | Generates realistic synthetic data per channel, including creative fatigue, weekend effects and injected anomalies. |
| `app.py` | Streamlit dashboard with KPI tiles, scorecard, channel comparison, weekly trends, anomalies, budget plan and forecasts. You can upload your own CSV. |

## Scoring model

| KPI | Weight | Better when |
|---|---|---|
| ROAS | 35% | higher |
| CPA | 20% | lower |
| CVR | 20% | higher |
| CTR | 15% | higher |
| CPC | 10% | lower |

Each KPI is turned into a percentile rank across campaigns, so KPIs measured on different scales can be compared fairly. Campaigns with zero conversions get the worst CPA instead of a misleading CPA of 0.

Grades: A ≥ 80, B ≥ 65, C ≥ 50, D ≥ 35, F < 35.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # or: pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate a sample dataset (24 campaigns x 90 days)
cpe generate --campaigns 24 --days 90

# 2. Evaluate campaigns and write reports to reports/
cpe evaluate data/campaigns.csv --target-roas 2.0

# 3. What-if forecast for one campaign
cpe forecast data/campaigns.csv --campaign CMP-001 --spend 100 250 500 1000

# 4. Interactive dashboard
streamlit run app.py

# 5. Export results as JSON for a static web dashboard
cpe export data/campaigns.csv --out web/data.json

# Tests
pytest
```

### Input format

One row per campaign per day:

```
date,campaign_id,campaign_name,channel,objective,region,impressions,clicks,conversions,spend,revenue
2026-01-01,CMP-001,Brand Search North,Google Search,Conversions,North,12000,540,32,648.00,2720.00
```

`campaign_name`, `objective` and `region` are optional.

### Outputs (`reports/`)

- `campaign_scores.csv`: KPIs, score, grade and recommendation for each campaign
- `channel_summary.csv`: KPIs for each channel
- `anomalies.csv`: flagged campaign-days with likely causes
- `budget_recommendation.csv`: current vs suggested budget
- `report.html`: a single shareable report

`cpe export` writes `web/data.json`, which holds everything the dashboard shows.

## Project structure

```
campaign-performance-evaluator/
├── app.py                  # Streamlit dashboard
├── src/cpe/
│   ├── cli.py              # `cpe` command
│   ├── data_generator.py
│   ├── metrics.py
│   ├── scoring.py
│   ├── anomalies.py
│   ├── forecasting.py
│   ├── report.py
│   └── export.py           # JSON for the static web dashboard
├── tests/test_cpe.py
├── data/                   # input CSVs
└── reports/                # generated outputs
```

## Limitations

- By default the budget projection assumes each campaign's ROAS stays the same at the new spend level. In practice returns usually fall as spend grows, so pass `returns_exponent` below 1 to `budget_reallocation` (e.g. `0.7`: doubling spend raises revenue by about 62%) for a more conservative projection. The live dashboard has a slider for this.
- The forecast predicts conversion efficiency for each campaign. It is reliable inside a campaign's usual spend range and less so far outside it.
- The sample data is synthetic. Replace it with an export from your ad platforms for real analysis.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, pytest
