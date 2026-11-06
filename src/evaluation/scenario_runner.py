"""
evaluation/scenario_runner.py — Day 5: Per-Scenario Evaluation

Runs ONE dataset/config through the existing pipeline exactly once
(`run_day2_analysis`), then derives all three levels from that SAME
run — Level 2 via `run_all_agents`, Level 3 via `run_orchestration`
(which itself calls `run_all_agents` once more internally — unavoidable
without changing Day 3/4 code, but zero NEW forecasting/anomaly math is
performed; agents are pure re-derivations from the same AnalysisRunResult).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..agents.impact_config import parse_impact_config
from ..agents.runner import run_all_agents
from ..analysis_pipeline import run_day2_analysis
from ..orchestration.energy_engineer import generate_narrative
from ..orchestration.orchestrator import run_orchestration
from .coverage import compute_explanation_coverage
from .diversity import compute_evidence_diversity
from .dimensions import applicable_dimensions
from .levels import level1_isolated_analytics, level2_independent_agents, level3_collaborative_framework
from .recommendation_support import classify_recommendation_support


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    data_kind: str
    input_format: str
    run: object
    agents: dict
    collab: object
    narrative: object
    applicable: list
    level1_support: dict
    level2_support: dict
    level3_support: dict
    level1_coverage: object
    level2_coverage: object
    level3_coverage: object
    level1_diversity: object
    level2_diversity: object
    level3_diversity: object
    level1_recommendation: str
    level2_recommendation: str
    level3_recommendation: str


def run_scenario(
    scenario_id: str,
    description: str,
    data_kind: str,
    input_format: str,
    path: str,
    day1_config_path: Optional[str] = None,
    analysis_config: Optional[dict] = None,
    series_id_override: Optional[str] = None,
    output_root: str = "outputs",
) -> ScenarioResult:
    run = run_day2_analysis(
        path,
        day1_config_path=day1_config_path,
        analysis_config=analysis_config,
        series_id_override=series_id_override,
        output_root=output_root,
        generate_figures=False,
    )

    impact_cfg = parse_impact_config(analysis_config)
    tariff_configured = impact_cfg.tariff_per_kwh is not None
    carbon_configured = impact_cfg.carbon_factor_kg_per_kwh is not None
    applicable = applicable_dimensions(tariff_configured, carbon_configured)

    agents = run_all_agents(run, analysis_config)
    collab = run_orchestration(run, analysis_config)
    narrative = generate_narrative(run, collab)

    l1 = level1_isolated_analytics(run)
    l2 = level2_independent_agents(run, agents)
    l3 = level3_collaborative_framework(run, collab)

    return ScenarioResult(
        scenario_id=scenario_id,
        description=description,
        data_kind=data_kind,
        input_format=input_format,
        run=run,
        agents=agents,
        collab=collab,
        narrative=narrative,
        applicable=applicable,
        level1_support=l1,
        level2_support=l2,
        level3_support=l3,
        level1_coverage=compute_explanation_coverage(l1, applicable),
        level2_coverage=compute_explanation_coverage(l2, applicable),
        level3_coverage=compute_explanation_coverage(l3, applicable),
        level1_diversity=compute_evidence_diversity(l1, applicable),
        level2_diversity=compute_evidence_diversity(l2, applicable),
        level3_diversity=compute_evidence_diversity(l3, applicable),
        level1_recommendation=classify_recommendation_support(None),
        level2_recommendation=classify_recommendation_support(None),
        level3_recommendation=classify_recommendation_support(collab.recommendations),
    )
