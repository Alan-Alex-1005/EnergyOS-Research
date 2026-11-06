#!/usr/bin/env python3
"""
run_real_demo.py — Thin usability wrapper around run_demo.py

Purpose: make it easy to analyze a REAL dataset while recording where it
came from (name / provider / source URL). This wrapper does NOT
implement any research logic of its own — it only builds a JSON config
(reusing run_demo.py's existing --config mechanism) and delegates
straight to run_demo.main(). All forecasting, anomaly detection, agent,
and collaboration logic is exactly what run_demo.py already runs.

Example:
    python run_real_demo.py --input "path/to/energy.csv" \\
        --name "My Building Dataset" \\
        --provider "My Utility Company" \\
        --source "https://example.org/dataset"

With a real, documented tariff/carbon factor (optional — omit if you
don't have one; the Cost/Carbon agents will honestly report
UNAVAILABLE rather than guessing):
    python run_real_demo.py --input data.csv --name "X" --provider "Y" --source "Z" \\
        --tariff-per-kwh 0.18 --tariff-currency USD --carbon-factor-kg-per-kwh 0.42

See `python run_real_demo.py --help` for all options, and README.md
for a full walkthrough.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import run_demo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_real_demo.py",
        description=(
            "Analyze a real CSV/XLSX energy dataset end-to-end (data understanding -> forecast -> "
            "anomaly -> specialized agents -> multi-agent collaboration -> AI Energy Engineer), "
            "recording where the data came from. This is a thin wrapper around run_demo.py --analyze "
            "— it adds no new analysis logic."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to the real CSV or XLSX file to analyze.")
    parser.add_argument("--name", required=True, help="A human-readable name for this dataset (for the record; shown in the terminal banner).")
    parser.add_argument("--provider", required=True, help="Who publishes/maintains this dataset (utility, agency, research group, etc.).")
    parser.add_argument("--source", required=True, help="URL or citation for the dataset's origin.")
    parser.add_argument("--sheet", default=None, help="Sheet name, for XLSX input with multiple sheets.")
    parser.add_argument("--series", default=None, help="Explicit analytical series_id to analyze, if the dataset has more than one.")
    parser.add_argument("--config", default=None, help="Optional existing JSON config file to extend (mapping overrides, analysis settings).")
    parser.add_argument("--tariff-per-kwh", type=float, default=None, help="A REAL, documented tariff rate (currency per kWh). Omit if you don't have one — do not guess.")
    parser.add_argument("--tariff-currency", default="USD", help="Currency code for --tariff-per-kwh (default: USD).")
    parser.add_argument("--carbon-factor-kg-per-kwh", type=float, default=None, help="A REAL, documented grid emission factor (kg CO2e per kWh). Omit if you don't have one.")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG/results.json generation (still prints the terminal report).")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    if args.tariff_per_kwh is not None or args.carbon_factor_kg_per_kwh is not None:
        impact = config.get("impact", {})
        if args.tariff_per_kwh is not None:
            impact["tariff_per_kwh"] = args.tariff_per_kwh
            impact["tariff_currency"] = args.tariff_currency
        if args.carbon_factor_kg_per_kwh is not None:
            impact["carbon_factor_kg_per_kwh"] = args.carbon_factor_kg_per_kwh
        config["impact"] = impact

    print("=" * 60)
    print("REAL-DATA PROVENANCE")
    print("=" * 60)
    print(f"Dataset name:  {args.name}")
    print(f"Provider:      {args.provider}")
    print(f"Source:        {args.source}")
    print(f"File:          {args.input}")
    print("=" * 60)
    print()

    config_path = args.config
    tmp_path = None
    if config and not args.config:
        fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="run_real_demo_config_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f)
        config_path = tmp_path
    elif config and args.config:
        fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="run_real_demo_config_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f)
        config_path = tmp_path

    demo_argv = [args.input, "--analyze"]
    if args.sheet:
        demo_argv += ["--sheet", args.sheet]
    if args.series:
        demo_argv += ["--series", args.series]
    if config_path:
        demo_argv += ["--config", config_path]
    if args.no_figures:
        demo_argv += ["--no-figures"]

    try:
        return run_demo.main(demo_argv)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    sys.exit(main())
