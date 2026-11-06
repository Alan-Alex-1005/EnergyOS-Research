#!/usr/bin/env python3
"""
run_demo.py — EnergyOS Research Demo v2 — CLI

Usage (Day 1 — data understanding only):
    python run_demo.py data/example.csv
    python run_demo.py data/example.xlsx --sheet Sheet1
    python run_demo.py data/example.csv --config configs/example.json

Usage (Day 2 — forecasting/anomaly/impact analysis):
    python run_demo.py data/example.csv --analyze
    python run_demo.py data/example.csv --analyze --series POWER:Main_Power_kW
    python run_demo.py data/example.csv --analyze --config configs/example.json
"""

from __future__ import annotations

import argparse
import json
import sys

from src.analysis_pipeline import AnalysisConfigError, run_day2_analysis, write_results_json
from src.analysis_terminal import render_agents_report, render_analysis_report
from src.agents.impact_config import ImpactConfigError
from src.mapping import MappingConfigError
from src.orchestration.energy_engineer import generate_narrative
from src.orchestration.orchestrator import run_orchestration
from src.orchestration.results import build_collaboration_dict, generate_collaboration_figures
from src.orchestration.terminal import render_collaboration_report, render_energy_engineer_report
from src.pipeline import PipelineError, run_pipeline
from src.terminal import render_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_demo.py",
        description="EnergyOS Research Demo v2 — Data Understanding (Day 1) + Analysis (Day 2)",
    )
    parser.add_argument("input_file", help="Path to a CSV or XLSX file.")
    parser.add_argument("--sheet", default=None, help="Sheet name for XLSX input (required if the workbook has multiple sheets).")
    parser.add_argument("--config", default=None, help="Path to a JSON config file (mapping overrides + optional 'analysis' block).")
    parser.add_argument("--analyze", action="store_true", help="Run Day 2 forecasting/anomaly/impact analysis after data understanding.")
    parser.add_argument("--series", default=None, help="Explicit analytical series_id to analyze (see 'ANALYTICAL SERIES AVAILABLE' listing).")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG/results.json generation (still prints the terminal report).")
    return parser


def _load_config(config_path: str | None) -> dict | None:
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(args.input_file, sheet=args.sheet, config_path=args.config)
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — last line of defense, never show a raw traceback
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1

    report = render_report(
        load_result=result.load_result,
        profile=result.profile,
        timestamp_info=result.timestamp_info,
        mapping_result=result.mapping_result,
        cleaning_report=result.cleaning_report,
        quality_result=result.quality_result,
        capability_report=result.capability_report,
        long_format_mapping=result.long_format_mapping,
    )
    print(report)

    if not args.analyze:
        return 0

    print("")
    try:
        analysis_config = _load_config(args.config)
        run = run_day2_analysis(
            args.input_file,
            sheet=args.sheet,
            day1_config_path=args.config,
            analysis_config=analysis_config,
            series_id_override=args.series,
            generate_figures=not args.no_figures,
            day1_result=result,
        )
    except (PipelineError, MappingConfigError, AnalysisConfigError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1

    print(render_analysis_report(run))

    print("")
    try:
        collab = run_orchestration(run, analysis_config)
    except ImpactConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(render_agents_report(collab.agents))
    print("")
    print(render_collaboration_report(collab))
    print("")
    narrative = generate_narrative(run, collab)
    print(render_energy_engineer_report(narrative))

    if run.output_dir:
        generate_collaboration_figures(run.output_dir, collab)
        results_path = write_results_json(run, extra={"collaboration": build_collaboration_dict(collab, narrative)})
        if results_path:
            print(f"\nResults written: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
