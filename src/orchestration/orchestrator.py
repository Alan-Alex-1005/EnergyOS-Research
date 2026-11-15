
"""
EnergyOS multi-agent orchestration.

Coordinates specialist agents and applies forecast reliability
enforcement before generating incident recommendations.

Raw anomaly results remain available for research and auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents.runner import run_all_agents
from ..analysis_pipeline import AnalysisRunResult
from ..forecasting.reliability import assess_forecast_reliability

from .agreement import AgreementResult, compute_agreement
from .completeness import CompletenessResult, compute_completeness
from .confidence import ConfidenceResult, compute_confidence
from .correlation import correlate_incidents
from .evidence import extract_evidence
from .recommendations import generate_recommendations


@dataclass
class CollaborationResult:
    agents: dict
    available_agents: list
    unavailable_agents: list
    evidence: list = field(default_factory=list)
    incidents: list = field(default_factory=list)
    agreement: Optional[AgreementResult] = None
    completeness: Optional[CompletenessResult] = None
    data_quality: Optional[float] = None
    data_quality_status: Optional[str] = None
    confidence: Optional[ConfidenceResult] = None
    recommendations: list = field(default_factory=list)

    # TrustAI safety information
    reliability_status: str = "NOT_ASSESSED"
    decision_supported: bool = False
    reliability_reasons: list[str] = field(default_factory=list)
    withheld_incident_count: int = 0


def run_orchestration(
    run: AnalysisRunResult,
    impact_config_raw: Optional[dict] = None,
) -> CollaborationResult:

    agents = run_all_agents(run, impact_config_raw)

    available_agents = [
        name
        for name, agent in agents.items()
        if agent.status in ("DONE", "LIMITED")
    ]

    unavailable_agents = [
        name
        for name, agent in agents.items()
        if agent.status not in ("DONE", "LIMITED")
    ]

    # Preserve all raw evidence for research and audit.
    evidence = extract_evidence(agents, run.primary)

    agreement = compute_agreement(agents)
    completeness = compute_completeness(agents, run.primary)

    data_quality = run.day1.quality_result.overall
    data_quality_status = run.day1.quality_result.status

    # This is the original collaboration score.
    # It is NOT sufficient by itself to approve a decision.
    confidence = compute_confidence(
        data_quality,
        agreement.score,
        completeness.score,
    )

    reliability_status = "NOT_ASSESSED"
    reliability_reasons = [
        "No usable forecast was available for reliability assessment."
    ]
    decision_supported = False

    if (
        run.primary is not None
        and run.primary.forecast is not None
        and run.primary.forecast.forecast_result is not None
    ):
        assessment = assess_forecast_reliability(
            run.primary.forecast.forecast_result
        )

        reliability_status = assessment.status
        reliability_reasons = list(assessment.reasons)

        decision_supported = (
            assessment.suitable_for_anomaly_evidence
        )

    # Calculate candidate incidents for auditing.
    candidate_incidents = correlate_incidents(evidence)

    if decision_supported:
        incidents = candidate_incidents
        recommendations = generate_recommendations(
            incidents,
            agents,
        )
        withheld_incident_count = 0
    else:
        # Do not present forecast-dependent candidates
        # as actionable incidents.
        incidents = []
        recommendations = []
        withheld_incident_count = len(candidate_incidents)

    return CollaborationResult(
        agents=agents,
        available_agents=available_agents,
        unavailable_agents=unavailable_agents,
        evidence=evidence,
        incidents=incidents,
        agreement=agreement,
        completeness=completeness,
        data_quality=data_quality,
        data_quality_status=data_quality_status,
        confidence=confidence,
        recommendations=recommendations,
        reliability_status=reliability_status,
        decision_supported=decision_supported,
        reliability_reasons=reliability_reasons,
        withheld_incident_count=withheld_incident_count,
    )
