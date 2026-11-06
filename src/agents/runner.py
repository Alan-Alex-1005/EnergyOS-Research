"""
agents/runner.py — Day 3: Run All Specialized Agents

Capability-aware execution: an agent that has nothing to work with
reports UNAVAILABLE/INSUFFICIENT_DATA honestly rather than being skipped
silently or fabricating a finding.
"""

from __future__ import annotations

from ..analysis_pipeline import AnalysisRunResult
from .anomaly_agent import run_anomaly_agent
from .base import STATUS_UNAVAILABLE, AgentResult
from .carbon_agent import run_carbon_agent
from .cost_agent import run_cost_agent
from .equipment_agent import run_equipment_agent
from .forecast_agent import run_forecast_agent
from .impact_config import ImpactConfig, parse_impact_config


def run_all_agents(run: AnalysisRunResult, impact_config_raw: dict | None = None) -> dict[str, AgentResult]:
    impact_config = parse_impact_config(impact_config_raw)

    if run.primary is None:
        no_series = AgentResult(
            agent_id="n/a",
            status=STATUS_UNAVAILABLE,
            finding="No analytical series was available in this dataset.",
            limitations=[run.selection_reason],
        )
        return {
            "forecast_agent": AgentResult(agent_id="forecast_agent", status=STATUS_UNAVAILABLE, finding=no_series.finding, limitations=no_series.limitations),
            "anomaly_agent": AgentResult(agent_id="anomaly_agent", status=STATUS_UNAVAILABLE, finding=no_series.finding, limitations=no_series.limitations),
            "equipment_agent": AgentResult(agent_id="equipment_agent", status=STATUS_UNAVAILABLE, finding=no_series.finding, limitations=no_series.limitations),
            "cost_agent": AgentResult(agent_id="cost_agent", status=STATUS_UNAVAILABLE, finding=no_series.finding, limitations=no_series.limitations),
            "carbon_agent": AgentResult(agent_id="carbon_agent", status=STATUS_UNAVAILABLE, finding=no_series.finding, limitations=no_series.limitations),
        }

    series_result = run.primary
    return {
        "forecast_agent": run_forecast_agent(series_result),
        "anomaly_agent": run_anomaly_agent(series_result),
        "equipment_agent": run_equipment_agent(series_result, run.day1.capability_report),
        "cost_agent": run_cost_agent(series_result, impact_config),
        "carbon_agent": run_carbon_agent(series_result, impact_config),
    }
