"""
evaluation/real_data_runner.py — Day 5: Real-Data Evaluation Runner

This is the FINAL RESEARCH runner. It accepts REAL dataset file paths
supplied by the caller — it never generates or injects synthetic
measurements, spikes, equipment, schedules, tariffs, or carbon factors.
`scenarios.py` / `scenario_runner.py`'s synthetic-fixture machinery
exists ONLY for automated software tests and is never invoked here.

Required per-dataset metadata (cannot be inferred, so it's mandatory):
  dataset_name, source_provider, source_reference (URL/citation).
A dataset without these is rejected.

Cost/Carbon agents only ever run if the caller supplies a real,
documented tariff/emission-factor value via `analysis_config`. No
default or example value is ever substituted here.

The 5 target categories (normal operation / natural anomaly /
rich-context / insufficient data / LONG format) are populated only from
whatever REAL datasets are actually supplied. A category with no
supplied dataset is reported as "NOT_AVAILABLE" rather than being
filled with a manufactured scenario.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..visualization import research_plots as viz
from .comparison import build_comparison_rows, write_comparison_csv
from .provenance import build_provenance
from .robustness import build_robustness_matrix
from .scenario_runner import run_scenario

REQUIRED_CATEGORIES = (
    "normal_operation",
    "natural_anomaly",
    "rich_context",
    "insufficient_data",
    "long_format",
)


class RealDataError(Exception):
    """Raised when a real-data evaluation input is missing required provenance."""


@dataclass
class RealDatasetSpec:
    scenario_id: str
    dataset_name: str
    source_provider: Optional[str]
    source_reference: Optional[str]
    category: str
    path: str
    day1_config_path: Optional[str] = None
    analysis_config: Optional[dict] = None
    series_id_override: Optional[str] = None
    long_format_transform: Optional[dict] = None
    description: str = ""

    def __post_init__(self):
        if not self.dataset_name or not self.dataset_name.strip():
            raise RealDataError(f"Dataset for scenario '{self.scenario_id}' is missing a dataset_name.")
        if not self.source_provider or not self.source_provider.strip():
            raise RealDataError(f"Dataset '{self.dataset_name}' is missing source_provider (who publishes this data).")
        if not self.source_reference or not self.source_reference.strip():
            raise RealDataError(f"Dataset '{self.dataset_name}' is missing source_reference (URL or citation).")
        if not os.path.exists(self.path):
            raise RealDataError(f"Dataset '{self.dataset_name}' file not found: {self.path}")


def apply_long_format_transform(input_path: str, output_path: str, timestamp_col: str, value_cols: list) -> str:
    """
    Representation-ONLY WIDE -> LONG reshape for format-robustness testing.
    Every original (timestamp, value) pair is preserved exactly — this is
    `pandas.melt`, not a data-generating transformation.
    """
    df = pd.read_csv(input_path)
    if timestamp_col not in df.columns:
        raise RealDataError(f"long_format_transform: timestamp column '{timestamp_col}' not found in {input_path}.")
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise RealDataError(f"long_format_transform: value column(s) {missing} not found in {input_path}.")

    long_df = df.melt(id_vars=[timestamp_col], value_vars=value_cols, var_name="metric", value_name="value")
    expected_rows = len(df) * len(value_cols)
    if len(long_df) != expected_rows:
        raise RealDataError("long_format_transform produced an unexpected row count — refusing to proceed.")

    long_df.to_csv(output_path, index=False)
    return output_path


def _build_research_summary_lines(scenario_results, comparison_rows) -> list:
    n = len(scenario_results)
    mean_cov = next(r for r in comparison_rows if r["dimension"] == "MEAN_EXPLANATION_COVERAGE_PCT")
    mean_div = next(r for r in comparison_rows if r["dimension"] == "MEAN_EVIDENCE_DIVERSITY_COUNT")
    lines = [
        f"Datasets evaluated: {n}",
        "",
        "Mean Explanation Coverage:",
        f"  Level 1: {mean_cov['level1_isolated_analytics']:.1f}%   "
        f"Level 2: {mean_cov['level2_independent_agents']:.1f}%   "
        f"Level 3: {mean_cov['level3_collaborative_framework']:.1f}%",
        "",
        "Mean Evidence Diversity (distinct categories):",
        f"  Level 1: {mean_div['level1_isolated_analytics']:.2f}   "
        f"Level 2: {mean_div['level2_independent_agents']:.2f}   "
        f"Level 3: {mean_div['level3_collaborative_framework']:.2f}",
        "",
        "Per-dataset result:",
    ]
    for sr in scenario_results:
        n_inc = len(sr.collab.incidents)
        conf = f"{sr.collab.confidence.score:.1f}" if sr.collab.confidence else "n/a"
        lines.append(f"  {sr.scenario_id}: incidents={n_inc}, L3 coverage={sr.level3_coverage.percentage:.0f}%, confidence={conf}")
    lines.append("")
    lines.append("This is a framework coverage/diversity study, NOT a prediction-accuracy benchmark.")
    return lines


def run_real_data_evaluation(dataset_specs: list, output_root: str) -> dict:
    if not dataset_specs:
        raise RealDataError(
            "No real dataset specs were supplied. This runner does not generate synthetic "
            "measurements for final research results — provide at least one real dataset."
        )

    scenario_results = []
    provenance_records = []
    transform_dir = os.path.join(output_root, "_real_data_transforms")

    for spec in dataset_specs:
        path = spec.path
        transformations = []
        if spec.long_format_transform:
            os.makedirs(transform_dir, exist_ok=True)
            out_path = os.path.join(transform_dir, f"{spec.scenario_id}_LONG.csv")
            path = apply_long_format_transform(
                spec.path, out_path, spec.long_format_transform["timestamp_col"], spec.long_format_transform["value_cols"]
            )
            transformations.append(
                f"WIDE->LONG representation-only reshape (pandas.melt) of columns "
                f"{spec.long_format_transform['value_cols']}; timestamps and values unchanged."
            )

        result = run_scenario(
            spec.scenario_id,
            spec.description or f"Real dataset: {spec.dataset_name}",
            "REAL",
            "LONG" if spec.long_format_transform else "WIDE",
            path,
            day1_config_path=spec.day1_config_path,
            analysis_config=spec.analysis_config,
            series_id_override=spec.series_id_override,
            output_root=os.path.join(output_root, "_real_data_runs"),
        )
        scenario_results.append(result)
        provenance_records.append(
            build_provenance(result.run, spec.dataset_name, spec.source_provider, spec.source_reference, transformations)
        )

    covered_categories = {spec.category for spec in dataset_specs}
    category_status = {
        cat: ("AVAILABLE" if cat in covered_categories else "NOT_AVAILABLE — no real dataset supplied for this category")
        for cat in REQUIRED_CATEGORIES
    }

    robustness_rows = build_robustness_matrix(scenario_results)
    comparison_rows = build_comparison_rows(scenario_results)

    eval_dir = os.path.join(output_root, "evaluation_real")
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = write_comparison_csv(scenario_results, os.path.join(eval_dir, "research_comparison.csv"))

    figures = {
        "14": viz.plot_framework_comparison(eval_dir, comparison_rows),
        "15": viz.plot_explanation_coverage(eval_dir, scenario_results),
        "16": viz.plot_evidence_diversity(eval_dir, scenario_results),
        "17": viz.plot_scenario_robustness(eval_dir, robustness_rows),
    }
    summary_lines = _build_research_summary_lines(scenario_results, comparison_rows)
    summary_lines = (
        ["REAL-DATA EVALUATION — no synthetic measurements used.", ""]
        + summary_lines
        + ["", "Category coverage:"]
        + [f"  {cat}: {status}" for cat, status in category_status.items()]
    )
    figures["18"] = viz.plot_research_summary(eval_dir, summary_lines)

    evaluation_dict = {
        "data_kind": "REAL",
        "category_coverage": category_status,
        "datasets": [
            {
                "scenario_id": sr.scenario_id,
                "provenance": {
                    "dataset_name": prov.dataset_name,
                    "source_provider": prov.source_provider,
                    "source_reference": prov.source_reference,
                    "filename": prov.filename,
                    "input_format": prov.input_format,
                    "row_count": prov.row_count,
                    "selected_series": prov.selected_series,
                    "measurement_role": prov.measurement_role,
                    "unit": prov.unit,
                    "time_start": prov.time_start,
                    "time_end": prov.time_end,
                    "preprocessing_performed": prov.preprocessing_performed,
                    "transformations_performed": prov.transformations_performed,
                    "context_available": prov.context_available,
                },
            }
            for sr, prov in zip(scenario_results, provenance_records)
        ],
        "robustness_matrix": robustness_rows,
        "comparison_table": comparison_rows,
        "research_summary": summary_lines,
        "not_a_benchmark_of": [
            "forecasting accuracy",
            "anomaly-detection precision/recall",
            "real-world cost or carbon savings",
            "root-cause diagnostic accuracy",
        ],
    }
    json_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_dict, f, indent=2, default=str)

    return {
        "scenario_results": scenario_results,
        "provenance_records": provenance_records,
        "category_status": category_status,
        "robustness_rows": robustness_rows,
        "comparison_rows": comparison_rows,
        "csv_path": csv_path,
        "json_path": json_path,
        "figures": figures,
        "eval_dir": eval_dir,
    }
