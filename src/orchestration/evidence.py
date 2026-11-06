"""
orchestration/evidence.py — Day 4 Part 2: Shared Evidence Model

Converts the EXISTING AgentResult objects (Day 3) into a common,
structured evidence representation the orchestrator can correlate. No
new analytical computation happens here — every value is read straight
out of an agent's already-computed evidence/metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents.base import AgentResult
from ..analysis_pipeline import SeriesAnalysisResult


@dataclass
class SharedEvidence:
    source_agent: str
    evidence_type: str  # FORECAST_DEVIATION / ANOMALY / EQUIPMENT_CONTEXT / ENERGY_IMPACT / COST_IMPACT / CARBON_IMPACT
    series_id: str
    time_start: Optional[str]
    time_end: Optional[str]
    metric: str
    value: Optional[float]
    unit: Optional[str]
    provenance: str  # OBSERVED / CALCULATED / ESTIMATED / INFERRED
    availability: str  # mirrors the source agent's AgentResult.status
    description: str
    detail: dict = field(default_factory=dict)


def extract_evidence(agents: dict[str, AgentResult], series_result: Optional[SeriesAnalysisResult]) -> list[SharedEvidence]:
    """Build the shared evidence list from this run's agent results. Never
    invents a series_id/time window — both come from the actual analysis."""
    if series_result is None:
        return []

    series_id = series_result.series.series_id
    time_start = series_result.split.test_start if series_result.split else None
    time_end = series_result.split.test_end if series_result.split else None

    evidence: list[SharedEvidence] = []

    fa = agents.get("forecast_agent")
    if fa and fa.status == "DONE":
        mean_residual = fa.evidence.get("mean_residual")
        evidence.append(
            SharedEvidence(
                source_agent="forecast_agent",
                evidence_type="FORECAST_DEVIATION",
                series_id=series_id,
                time_start=time_start,
                time_end=time_end,
                metric="mean_residual",
                value=mean_residual,
                unit=series_result.series.unit,
                provenance="CALCULATED",
                availability=fa.status,
                description=fa.finding,
                detail={"model": fa.evidence.get("selected_model"), "rmse": fa.metrics.get("rmse")},
            )
        )

    aa = agents.get("anomaly_agent")
    if aa and aa.status == "DONE":
        evidence.append(
            SharedEvidence(
                source_agent="anomaly_agent",
                evidence_type="ANOMALY",
                series_id=series_id,
                time_start=time_start,
                time_end=time_end,
                metric="anomalies_detected",
                value=float(aa.evidence.get("anomalies_detected", 0)),
                unit="count",
                provenance="INFERRED",
                availability=aa.status,
                description=aa.finding,
                detail={"events": aa.evidence.get("events", []), "peak_score": aa.metrics.get("peak_score")},
            )
        )

    ea = agents.get("equipment_agent")
    if ea and ea.status in ("DONE", "LIMITED"):
        evidence.append(
            SharedEvidence(
                source_agent="equipment_agent",
                evidence_type="EQUIPMENT_CONTEXT",
                series_id=series_id,
                time_start=None,
                time_end=None,
                metric="identifier_context",
                value=None,
                unit=None,
                provenance="MEASURED" if ea.status == "DONE" else "OBSERVED",
                availability=ea.status,
                description=ea.finding,
                detail={"identifier_context": ea.evidence.get("identifier_context", {})},
            )
        )

    if series_result.excess_energy and series_result.excess_energy.available:
        ee = series_result.excess_energy
        evidence.append(
            SharedEvidence(
                source_agent="excess_energy",
                evidence_type="ENERGY_IMPACT",
                series_id=series_id,
                time_start=time_start,
                time_end=time_end,
                metric="excess_energy_kwh",
                value=ee.excess_energy_kwh,
                unit="kWh",
                provenance="ESTIMATED",
                availability="DONE",
                description=ee.basis or "Estimated excess energy above forecast baseline.",
            )
        )

    ca = agents.get("cost_agent")
    if ca and ca.status == "DONE":
        evidence.append(
            SharedEvidence(
                source_agent="cost_agent",
                evidence_type="COST_IMPACT",
                series_id=series_id,
                time_start=time_start,
                time_end=time_end,
                metric="estimated_cost",
                value=ca.metrics.get("estimated_cost"),
                unit=ca.evidence.get("currency"),
                provenance="CALCULATED",
                availability=ca.status,
                description=ca.finding,
            )
        )

    cb = agents.get("carbon_agent")
    if cb and cb.status == "DONE":
        evidence.append(
            SharedEvidence(
                source_agent="carbon_agent",
                evidence_type="CARBON_IMPACT",
                series_id=series_id,
                time_start=time_start,
                time_end=time_end,
                metric="estimated_carbon_kg",
                value=cb.metrics.get("estimated_carbon_kg"),
                unit="kg CO2e",
                provenance="CALCULATED",
                availability=cb.status,
                description=cb.finding,
            )
        )

    return evidence
