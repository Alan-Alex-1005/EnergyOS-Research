"""
agents/cost_agent.py — Day 3: Cost Agent

    estimated cost impact = estimated excess kWh x tariff

Only computed when BOTH a valid excess-energy result AND an explicitly
configured tariff exist. Never invents a tariff. Day 3 does not
implement full cost analysis (that's Day 3's Cost Agent scope, distinct
from a Day-4/5 "final cost conclusion") — this is a single, transparent
multiplication of two already-known quantities.
"""

from __future__ import annotations

from ..analysis_pipeline import SeriesAnalysisResult
from .base import STATUS_DONE, STATUS_UNAVAILABLE, AgentResult
from .impact_config import ImpactConfig

AGENT_ID = "cost_agent"


def run_cost_agent(series_result: SeriesAnalysisResult, impact_config: ImpactConfig) -> AgentResult:
    excess = series_result.excess_energy
    if excess is None or not excess.available:
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_UNAVAILABLE,
            finding="Cost impact requires a valid excess-energy estimate, which is not available for this series.",
            limitations=[excess.reason if excess else "no excess-energy result was computed."],
        )

    if impact_config.tariff_per_kwh is None:
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_UNAVAILABLE,
            finding="No tariff is configured; cost impact cannot be estimated.",
            evidence={"excess_energy_kwh": excess.excess_energy_kwh},
            limitations=["Set config 'impact.tariff_per_kwh' to enable this agent."],
        )

    cost = excess.excess_energy_kwh * impact_config.tariff_per_kwh
    finding = (
        f"Estimated cost impact: {impact_config.tariff_currency} {cost:.2f} "
        f"({excess.excess_energy_kwh:.1f} kWh x {impact_config.tariff_per_kwh:.4g}/kWh)."
    )

    return AgentResult(
        agent_id=AGENT_ID,
        status=STATUS_DONE,
        finding=finding,
        evidence={
            "excess_energy_kwh": excess.excess_energy_kwh,
            "tariff_per_kwh": impact_config.tariff_per_kwh,
            "currency": impact_config.tariff_currency,
        },
        metrics={"estimated_cost": round(cost, 2)},
        provenance={"tariff": "CONFIGURED", "excess_energy": "ESTIMATED", "cost_impact": "CALCULATED"},
    )
