"""
agents/anomaly_agent.py — Day 3: Anomaly Agent

Wraps the EXISTING Day-2 AnomalyEngineResult. Computes nothing new.
"""

from __future__ import annotations

from ..analysis_pipeline import SeriesAnalysisResult
from .base import STATUS_DONE, STATUS_INSUFFICIENT_DATA, STATUS_UNAVAILABLE, AgentResult

AGENT_ID = "anomaly_agent"


def run_anomaly_agent(series_result: SeriesAnalysisResult) -> AgentResult:
    if series_result.sufficiency.status != "AVAILABLE":
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_INSUFFICIENT_DATA,
            finding=f"Insufficient data to run anomaly detection on '{series_result.series.series_id}'.",
            evidence={"sample_count": series_result.sufficiency.sample_count},
            limitations=[series_result.sufficiency.reason],
        )

    an = series_result.anomaly
    if an is None or not an.applicable:
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_UNAVAILABLE,
            finding="Anomaly detection was not available for this series.",
            limitations=[an.reason if an else "anomaly detection was not attempted."],
        )

    events_summary = [
        {
            "event_id": e.event_id,
            "start_time": e.start_time,
            "end_time": e.end_time,
            "duration_seconds": e.duration_seconds,
            "peak_score": e.peak_score,
            "severity": e.severity,
            "direction": e.direction,
            "observation_count": e.observation_count,
            "persistence_class": e.persistence_class,
        }
        for e in an.events
    ]

    if an.anomalies_detected == 0:
        finding = f"No anomalies detected across {an.observations_analyzed} observations (peak score {an.peak_score:.2f})."
    else:
        finding = (
            f"{an.anomalies_detected} anomalous observation(s) across {len(an.events)} event(s); "
            f"highest severity {an.highest_severity}, peak score {an.peak_score:.2f}."
        )

    limitations = list(an.warnings)

    return AgentResult(
        agent_id=AGENT_ID,
        status=STATUS_DONE,
        finding=finding,
        evidence={
            "observations_analyzed": an.observations_analyzed,
            "anomalies_detected": an.anomalies_detected,
            "events": events_summary,
            "base_threshold": an.base_threshold,
        },
        metrics={
            "anomaly_rate_pct": an.anomaly_rate,
            "peak_score": an.peak_score,
        },
        provenance={"anomaly_classification": "INFERRED", "counts": "CALCULATED"},
        limitations=limitations,
    )
