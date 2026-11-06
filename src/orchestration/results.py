"""
orchestration/results.py — Day 4 Part 12: results.json Integration

Builds the "collaboration" section merged into the same results.json Day
2 already writes — terminal, PNGs, and JSON all read from the same
CollaborationResult/EnergyEngineerNarrative objects, never recomputed
independently per output format.
"""

from __future__ import annotations

from ..analysis_pipeline import _json_safe
from ..visualization import research_plots as viz
from .energy_engineer import EnergyEngineerNarrative
from .orchestrator import CollaborationResult


def generate_collaboration_figures(output_dir: str, collab: CollaborationResult) -> dict:
    return {
        "09": viz.plot_agent_status(output_dir, collab.agents),
        "10": viz.plot_agent_agreement(output_dir, collab.agreement),
        "11": viz.plot_evidence_completeness(output_dir, collab.completeness),
        "12": viz.plot_incident_summary(output_dir, collab.incidents),
        "13": viz.plot_decision_confidence(output_dir, collab.confidence),
    }


def build_collaboration_dict(collab: CollaborationResult, narrative: EnergyEngineerNarrative) -> dict:
    result = {
        "agents": {
            key: {
                "status": a.status,
                "finding": a.finding,
                "evidence": a.evidence,
                "metrics": a.metrics,
                "provenance": a.provenance,
                "limitations": a.limitations,
                "warnings": a.warnings,
            }
            for key, a in collab.agents.items()
        },
        "available_agents": collab.available_agents,
        "unavailable_agents": collab.unavailable_agents,
        "evidence": [
            {
                "source_agent": e.source_agent,
                "evidence_type": e.evidence_type,
                "series_id": e.series_id,
                "time_start": e.time_start,
                "time_end": e.time_end,
                "metric": e.metric,
                "value": e.value,
                "unit": e.unit,
                "provenance": e.provenance,
                "availability": e.availability,
                "description": e.description,
            }
            for e in collab.evidence
        ],
        "incidents": [
            {
                "incident_id": inc.incident_id,
                "series_id": inc.series_id,
                "time_start": inc.time_start,
                "time_end": inc.time_end,
                "evidence_types_present": sorted(inc.evidence_types_present),
                "trigger_evidence_count": len(inc.trigger_evidence),
                "supporting_evidence_count": len(inc.supporting_evidence),
            }
            for inc in collab.incidents
        ],
        "agreement": (
            {
                "score": collab.agreement.score,
                "status": collab.agreement.status,
                "participants": collab.agreement.participants,
                "forecast_direction": collab.agreement.forecast_direction,
                "anomaly_direction": collab.agreement.anomaly_direction,
                "notes": collab.agreement.notes,
            }
            if collab.agreement
            else None
        ),
        "evidence_completeness": (
            {
                "score": collab.completeness.score,
                "category_credit": collab.completeness.category_credit,
                "category_status": collab.completeness.category_status,
                "weights": collab.completeness.weights,
                "notes": collab.completeness.notes,
            }
            if collab.completeness
            else None
        ),
        "data_quality": {"score": collab.data_quality, "status": collab.data_quality_status},
        "decision_confidence": (
            {
                "score": collab.confidence.score,
                "label": collab.confidence.label,
                "components": collab.confidence.components,
                "weights_used": collab.confidence.weights_used,
                "notes": collab.confidence.notes,
            }
            if collab.confidence
            else None
        ),
        "recommendations": [
            {
                "finding": r.finding,
                "possible_explanation": r.possible_explanation,
                "recommended_action": r.recommended_action,
                "evidence_refs": r.evidence_refs,
            }
            for r in collab.recommendations
        ],
        "ai_energy_engineer": {
            "what": narrative.what,
            "where": narrative.where,
            "when": narrative.when,
            "why": narrative.why,
            "energy_impact": narrative.energy_impact,
            "cost_impact": narrative.cost_impact,
            "carbon_impact": narrative.carbon_impact,
            "evidence": narrative.evidence,
            "missing_context": narrative.missing_context,
            "confidence": narrative.confidence,
            "root_cause": narrative.root_cause,
            "recommended_action": narrative.recommended_action,
        },
    }
    return _json_safe(result)
