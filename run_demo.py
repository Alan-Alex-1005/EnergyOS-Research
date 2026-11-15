
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

Forecast evidence export:
    When Day 2 analysis generates an output directory,
    EnergyOS also attempts to create forecast_evidence.csv
    containing timestamp-aligned forecast and anomaly data.

    This export is read-only with respect to analytical results.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.analysis_pipeline import (
    AnalysisConfigError,
    run_day2_analysis,
    write_results_json,
)
from src.analysis_terminal import (
    render_agents_report,
    render_analysis_report,
)
from src.agents.impact_config import ImpactConfigError
from src.mapping import MappingConfigError
from src.orchestration.energy_engineer import generate_narrative
from src.orchestration.orchestrator import run_orchestration
from src.orchestration.results import (
    build_collaboration_dict,
    generate_collaboration_figures,
)
from src.orchestration.terminal import (
    render_collaboration_report,
    render_energy_engineer_report,
)
from src.pipeline import PipelineError, run_pipeline
from src.terminal import render_report

# NEW: Export the already-calculated forecast and anomaly evidence.
# This module must exist in the project root.
from energyos_evidence_export import export_forecast_evidence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_demo.py",
        description=(
            "EnergyOS Research Demo v2 — "
            "Data Understanding (Day 1) + Analysis (Day 2)"
        ),
    )

    parser.add_argument(
        "input_file",
        help="Path to a CSV or XLSX file.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help=(
            "Sheet name for XLSX input "
            "(required if the workbook has multiple sheets)."
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON config file "
            "(mapping overrides + optional 'analysis' block)."
        ),
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Run Day 2 forecasting/anomaly/impact "
            "analysis after data understanding."
        ),
    )

    parser.add_argument(
        "--series",
        default=None,
        help=(
            "Explicit analytical series_id to analyze "
            "(see 'ANALYTICAL SERIES AVAILABLE' listing)."
        ),
    )

    parser.add_argument(
        "--no-figures",
        action="store_true",
        help=(
            "Skip PNG/results.json generation "
            "(still prints the terminal report)."
        ),
    )

    return parser


def _load_config(
    config_path: str | None,
) -> dict | None:
    """
    Load the optional existing EnergyOS configuration.

    No configuration is modified.
    """

    if not config_path:
        return None

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main(
    argv: list[str] | None = None,
) -> int:

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # --------------------------------------------------
    # DAY 1 — DATA INTELLIGENCE
    # --------------------------------------------------

    try:
        result = run_pipeline(
            args.input_file,
            sheet=args.sheet,
            config_path=args.config,
        )

    except PipelineError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    except Exception as error:
        # Last line of defense.
        # Preserve the existing CLI error behavior.
        print(
            f"UNEXPECTED ERROR: {error}",
            file=sys.stderr,
        )
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

    # Day 1 only.
    if not args.analyze:
        return 0

    # --------------------------------------------------
    # DAY 2 — FORECASTING AND ANOMALY ANALYSIS
    # --------------------------------------------------

    print("")

    try:
        analysis_config = _load_config(
            args.config
        )

        run = run_day2_analysis(
            args.input_file,
            sheet=args.sheet,
            day1_config_path=args.config,
            analysis_config=analysis_config,
            series_id_override=args.series,
            generate_figures=not args.no_figures,
            day1_result=result,
        )

    except (
        PipelineError,
        MappingConfigError,
        AnalysisConfigError,
        json.JSONDecodeError,
    ) as error:

        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        return 1

    except Exception as error:
        print(
            f"UNEXPECTED ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        render_analysis_report(run)
    )

    # --------------------------------------------------
    # SPECIALIZED ENERGY AGENTS
    # --------------------------------------------------

    print("")

    try:
        collab = run_orchestration(
            run,
            analysis_config,
        )

    except ImpactConfigError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        render_agents_report(
            collab.agents
        )
    )

    # --------------------------------------------------
    # MULTI-AGENT COLLABORATION
    # --------------------------------------------------

    print("")

    print(
        render_collaboration_report(
            collab
        )
    )

    # --------------------------------------------------
    # AI ENERGY ENGINEER
    # --------------------------------------------------

    print("")

    narrative = generate_narrative(
        run,
        collab,
    )

    print(
        render_energy_engineer_report(
            narrative
        )
    )

    # --------------------------------------------------
    # RESEARCH OUTPUT GENERATION
    # --------------------------------------------------

    if run.output_dir:

        # Preserve the existing multi-agent figures.
        generate_collaboration_figures(
            run.output_dir,
            collab,
        )

        # Preserve the existing results.json structure.
        results_path = write_results_json(
            run,
            extra={
                "collaboration": build_collaboration_dict(
                    collab,
                    narrative,
                )
            },
        )

        if results_path:
            print(
                f"\nResults written: {results_path}"
            )

        # --------------------------------------------------
        # NEW — STRUCTURED FORECAST EVIDENCE EXPORT
        # --------------------------------------------------
        #
        # This exports existing forecast/anomaly observations.
        #
        # It does not:
        #   - retrain forecasting models
        #   - change model selection
        #   - recalculate TrustAI scores
        #   - modify analytical results
        #
        # Export failures are reported without interrupting
        # the otherwise successful EnergyOS demonstration.
        # --------------------------------------------------

        try:

            evidence_path = export_forecast_evidence(
                run
            )

            if evidence_path is not None:

                print(
                    "Forecast evidence exported:",
                    evidence_path,
                )

            else:

                print(
                    "Forecast evidence unavailable "
                    "for this run."
                )

        except (ValueError, OSError, AttributeError) as error:

            print(
                "WARNING: Forecast evidence export failed:",
                error,
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
