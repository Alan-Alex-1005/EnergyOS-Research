"""
orchestration/orchestrator.py — Day 4 Part 1: Multi-Agent Orchestrator

Receives the EXISTING AgentResult objects (Day 3) and the Day-2/Day-1
results they wrap, and coordinates their evidence into one
CollaborationResult: which agents are available, correlated incidents,
agreement, completeness, reused data quality, decision-support
confidence, and deterministic recommendations. Computes NO new
forecasting/anomaly/impact mathematics — that stays exactly where Day 2/3
put it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents.runner import run_all_agents
from ..analysis_pipeline import AnalysisRunResult
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


def run_orchestration(run: AnalysisRunResult, impact_config_raw: Optional[dict] = None) -> CollaborationResult:
    agents = run_all_agents(run, impact_config_raw)

    available_agents = [k for k, v in agents.items() if v.status in ("DONE", "LIMITED")]
    unavailable_agents = [k for k, v in agents.items() if v.status not in ("DONE", "LIMITED")]

    evidence = extract_evidence(agents, run.primary)
    incidents = correlate_incidents(evidence)

    agreement = compute_agreement(agents)
    completeness = compute_completeness(agents, run.primary)

    data_quality = run.day1.quality_result.overall
    data_quality_status = run.day1.quality_result.status

    confidence = compute_confidence(data_quality, agreement.score, completeness.score)

    recommendations = generate_recommendations(incidents, agents)

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
    )
