"""
evaluation/scenarios.py — Day 5 Part 6: Evaluation Scenarios

Defines the 5 required scenarios. Synthetic datasets used here are
clearly labeled CONTROLLED/SYNTHETIC — never presented as real-world
validation. Scenario 1 reuses the actual demo dataset (`data/example.csv`,
itself synthetic but not scenario-engineered); Scenario 5 reuses an
existing Day-1 LONG-format test fixture. Scenarios 2-4 generate small,
purpose-built CONTROLLED datasets to demonstrate specific, named
conditions (spike / rich context / sparsity).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ScenarioSpec:
    scenario_id: str
    description: str
    data_kind: str
    input_format: str
    path: str
    day1_config_path: Optional[str] = None
    analysis_config: Optional[dict] = None
    series_id_override: Optional[str] = None


def _write_spike_dataset(fixtures_dir: str) -> str:
    """Scenario 2: CONTROLLED/SYNTHETIC — sustained unexplained spike, NO equipment context."""
    rng = np.random.default_rng(101)
    n = 24 * 20
    period = 24
    pattern = 40 + 15 * np.sin(np.linspace(0, 2 * np.pi, period))
    values = np.tile(pattern, n // period) + rng.normal(0, 1.5, n)
    spike_start = int(n * 0.90)
    values[spike_start : spike_start + 6] += 120
    ts = pd.date_range("2026-07-01", periods=n, freq="h")
    path = os.path.join(fixtures_dir, "eval_scenario2_spike_SYNTHETIC.csv")
    pd.DataFrame({"timestamp": ts, "power_kw": np.round(values, 2)}).to_csv(path, index=False)
    return path


def _write_rich_context_dataset(fixtures_dir: str) -> str:
    """Scenario 3: CONTROLLED/SYNTHETIC — explicit equipment/meter identifier + a sustained spike."""
    rng = np.random.default_rng(102)
    n = 24 * 20
    period = 24
    pattern = 30 + 12 * np.sin(np.linspace(0, 2 * np.pi, period))
    values = np.tile(pattern, n // period) + rng.normal(0, 1.2, n)
    spike_start = int(n * 0.90)
    values[spike_start : spike_start + 8] += 90
    ts = pd.date_range("2026-08-01", periods=n, freq="h")
    path = os.path.join(fixtures_dir, "eval_scenario3_rich_context_SYNTHETIC.csv")
    pd.DataFrame({"timestamp": ts, "meter_id": "AHU-3", "power_kw": np.round(values, 2)}).to_csv(path, index=False)
    return path


def _write_sparse_dataset(fixtures_dir: str) -> str:
    """Scenario 4: CONTROLLED/SYNTHETIC — deliberately insufficient sample count."""
    ts = pd.date_range("2026-01-01", periods=5, freq="h")
    path = os.path.join(fixtures_dir, "eval_scenario4_sparse_SYNTHETIC.csv")
    pd.DataFrame({"timestamp": ts, "power_kw": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(path, index=False)
    return path


def build_scenarios(fixtures_dir: str, repo_root: str) -> list:
    os.makedirs(fixtures_dir, exist_ok=True)

    spike_path = _write_spike_dataset(fixtures_dir)
    rich_path = _write_rich_context_dataset(fixtures_dir)
    sparse_path = _write_sparse_dataset(fixtures_dir)
    long_path = os.path.join(repo_root, "tests", "fixtures", "long_format_meter.csv")
    example_path = os.path.join(repo_root, "data", "example.csv")

    return [
        ScenarioSpec(
            scenario_id="scenario_1_normal_operation",
            description="Normal operation — no significant incident expected.",
            data_kind="SYNTHETIC_DEMO_DATASET",
            input_format="WIDE",
            path=example_path,
        ),
        ScenarioSpec(
            scenario_id="scenario_2_unexplained_spike",
            description="CONTROLLED/SYNTHETIC — sustained unexplained consumption spike, no equipment context.",
            data_kind="CONTROLLED_SYNTHETIC",
            input_format="WIDE",
            path=spike_path,
        ),
        ScenarioSpec(
            scenario_id="scenario_3_rich_context_incident",
            description="CONTROLLED/SYNTHETIC — explicit equipment/meter identifier, tariff, and carbon factor configured.",
            data_kind="CONTROLLED_SYNTHETIC",
            input_format="WIDE",
            path=rich_path,
            analysis_config={"impact": {"tariff_per_kwh": 0.22, "tariff_currency": "USD", "carbon_factor_kg_per_kwh": 0.38}},
        ),
        ScenarioSpec(
            scenario_id="scenario_4_insufficient_data",
            description="CONTROLLED/SYNTHETIC — deliberately sparse dataset; honest degradation expected.",
            data_kind="CONTROLLED_SYNTHETIC",
            input_format="WIDE",
            path=sparse_path,
        ),
        ScenarioSpec(
            scenario_id="scenario_5_long_format",
            description="Day-1 LONG-format ingestion/mapping path (existing test fixture), explicit series selection.",
            data_kind="TEST_FIXTURE",
            input_format="LONG",
            path=long_path,
            series_id_override="POWER:power_kw:meter_id=M1",
        ),
    ]
