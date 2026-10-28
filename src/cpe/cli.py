"""Command-line entry point.

Examples:
    cpe generate --campaigns 30 --days 120
    cpe evaluate data/campaigns.csv --target-roas 2.5
    cpe forecast data/campaigns.csv --campaign CMP-001 --spend 200 400 800
    cpe export data/campaigns.csv --out web/data.json
"""
from __future__ import annotations

import argparse

import pandas as pd

from .data_generator import generate_campaigns
from .export import export_json
from .forecasting import train, what_if
from .metrics import validate
from .report import evaluate, write_reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cpe", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a synthetic dataset")
    g.add_argument("--campaigns", type=int, default=24)
    g.add_argument("--days", type=int, default=90)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/campaigns.csv")

    e = sub.add_parser("evaluate", help="score campaigns and write reports")
    e.add_argument("csv")
    e.add_argument("--target-roas", type=float, default=2.0)
    e.add_argument("--out-dir", default="reports")

    f = sub.add_parser("forecast", help="train model and run a what-if on spend")
    f.add_argument("csv")
    f.add_argument("--campaign", required=True)
    f.add_argument("--spend", type=float, nargs="+", default=[100, 250, 500, 1000])

    x = sub.add_parser("export", help="write a JSON file for the static web dashboard")
    x.add_argument("csv")
    x.add_argument("--target-roas", type=float, default=2.0)
    x.add_argument("--out", default="web/data.json")
    x.add_argument("--no-forecast", action="store_true", help="skip training the forecast model")

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        df = generate_campaigns(args.campaigns, args.days, seed=args.seed)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df):,} rows to {args.out}")

    elif args.cmd == "evaluate":
        ev = evaluate(pd.read_csv(args.csv), target_roas=args.target_roas)
        o = ev.overview
        print(f"Spend {o['spend']:,.0f} | Revenue {o['revenue']:,.0f} | "
              f"ROAS {o['roas']:.2f}x | ROI {o['roi']:.1%} | CPA {o['cpa']:,.2f}\n")
        print(ev.campaigns[["campaign_id", "channel", "roas", "performance_score",
                            "grade", "recommendation"]].to_string(index=False))
        print(f"\n{len(ev.anomalies)} anomalies detected")
        for p in write_reports(ev, args.out_dir):
            print(f"  wrote {p}")

    elif args.cmd == "forecast":
        daily = validate(pd.read_csv(args.csv))
        result = train(daily)
        print(f"Model holdout: MAE {result.mae:.2f} conversions/day, R² {result.r2:.3f}\n")
        print(what_if(result, daily, args.campaign, args.spend).to_string(index=False))

    elif args.cmd == "export":
        ev = evaluate(pd.read_csv(args.csv), target_roas=args.target_roas)
        path = export_json(ev, args.out, args.target_roas, with_forecast=not args.no_forecast)
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
