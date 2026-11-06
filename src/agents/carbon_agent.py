"""
agents/carbon_agent.py — Day 3: Carbon Agent

    estimated carbon impact = estimated excess kWh x emission factor

Only computed when BOTH a valid excess-energy result AND an explicitly
configured emission factor exist. Never invents a factor.
"""

from __future__ import annotations

from ..analysis_pipeline import SeriesAnalysisResult
from .base import STATUS_DONE, STATUS_UNAVAILABLE, AgentResult
from .impact_config import ImpactConfig

AGENT_ID = "carbon_agent"


def run_carbon_agent(series_result: SeriesAnalysisResult, impact_config: ImpactConfig) -> AgentResult:
    excess = series_result.excess_energy
    if excess is None or not excess.available:
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_UNAVAILABLE,
            finding="Carbon impact requires a valid excess-energy estimate, which is not available for this series.",
            limitations=[excess.reason if excess else "no excess-energy result was computed."],
        )

    if impact_config.carbon_factor_kg_per_kwh is None:
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_UNAVAILABLE,
            finding="No emission factor is configured; carbon impact cannot be estimated.",
            evidence={"excess_energy_kwh": excess.excess_energy_kwh},
            limitations=["Set config 'impact.carbon_factor_kg_per_kwh' to enable this agent."],
        )

    carbon_kg = excess.excess_energy_kwh * impact_config.carbon_factor_kg_per_kwh
    finding = (
        f"Estimated carbon impact: {carbon_kg:.2f} kg CO2e "
        f"({excess.excess_energy_kwh:.1f} kWh x {impact_config.carbon_factor_kg_per_kwh:.4g} kg/kWh)."
    )

    return AgentResult(
        agent_id=AGENT_ID,
        status=STATUS_DONE,
        finding=finding,
        evidence={
            "excess_energy_kwh": excess.excess_energy_kwh,
            "carbon_factor_kg_per_kwh": impact_config.carbon_factor_kg_per_kwh,
        },
        metrics={"estimated_carbon_kg": round(carbon_kg, 2)},
        provenance={"carbon_factor": "CONFIGURED", "excess_energy": "ESTIMATED", "carbon_impact": "CALCULATED"},
    )
