"""
evaluation/levels.py — Day 5 Part 1: Three Evaluation Levels

Each level function takes objects ALREADY computed elsewhere
(AnalysisRunResult from Day 2, the agents dict from Day 3, the
CollaborationResult from Day 4) and returns a dict[dimension] -> bool.
No forecasting/anomaly/impact/collaboration math is repeated here — this
module only asks "does this level's existing output actually contain
evidence X".
"""

from __future__ import annotations

from ..analysis_pipeline import AnalysisRunResult
from ..orchestration.orchestrator import CollaborationResult
from .dimensions import DIMENSIONS


def _base_dims() -> dict:
    return {d: False for d in DIMENSIONS}


def level1_isolated_analytics(run: AnalysisRunResult) -> dict:
    """LEVEL 1: raw Day-2 forecasting/anomaly/excess-energy only — no agents at all."""
    dims = _base_dims()
    p = run.primary
    if p is None or p.skipped_reason:
        return dims

    fc = p.forecast
    an = p.anomaly
    ee = p.excess_energy

    dims["abnormal_behavior_detection"] = bool(an and an.applicable)
    dims["forecast_context"] = bool(fc and fc.forecast_result is not None)
    dims["anomaly_event_information"] = bool(an and an.applicable)
    dims["temporal_persistence"] = bool(an and an.applicable)  # events carry persistence_class
    dims["excess_energy_impact"] = bool(ee and ee.available)
    # No agents exist at Level 1 -> no equipment/cost/carbon evidence, ever.
    dims["equipment_context_evidence"] = False
    dims["cost_impact"] = False
    dims["carbon_impact"] = False
    dims["provenance"] = bool(fc and fc.forecast_result is not None) or bool(ee and ee.available)
    # No agent layer -> no cross-agent concepts of any kind.
    dims["cross_agent_evidence_correlation"] = False
    dims["agreement_assessment"] = False
    dims["evidence_completeness_assessment"] = False
    dims["data_quality_awareness"] = False  # quality score exists (Day 1) but isn't USED in this level's output
    dims["decision_support_confidence"] = False
    dims["evidence_backed_recommendation"] = False
    # Each component discloses its own unavailability reason (skipped_reason,
    # mape_note, excess-energy reason) — genuinely present, just not unified.
    dims["missing_context_disclosure"] = True
    dims["unified_explanation"] = False  # three separate results, no single narrative
    return dims


def level2_independent_agents(run: AnalysisRunResult, agents: dict) -> dict:
    """LEVEL 2: the 5 specialized agents, run independently — no orchestration."""
    dims = _base_dims()
    if run.primary is None or run.primary.skipped_reason:
        return dims

    fa = agents.get("forecast_agent")
    aa = agents.get("anomaly_agent")
    ea = agents.get("equipment_agent")
    ca = agents.get("cost_agent")
    cb = agents.get("carbon_agent")

    dims["abnormal_behavior_detection"] = bool(aa and aa.status == "DONE")
    dims["forecast_context"] = bool(fa and fa.status == "DONE")
    dims["anomaly_event_information"] = bool(aa and aa.status == "DONE")
    dims["temporal_persistence"] = bool(aa and aa.status == "DONE")
    dims["excess_energy_impact"] = bool(run.primary.excess_energy and run.primary.excess_energy.available)
    dims["equipment_context_evidence"] = bool(ea and ea.status in ("DONE", "LIMITED"))
    dims["cost_impact"] = bool(ca and ca.status == "DONE")
    dims["carbon_impact"] = bool(cb and cb.status == "DONE")
    dims["provenance"] = any(a and a.provenance for a in (fa, aa, ea, ca, cb))
    # Agents run independently by construction — no correlation/agreement/
    # completeness/confidence/aggregated-recommendation exists at this level.
    dims["cross_agent_evidence_correlation"] = False
    dims["agreement_assessment"] = False
    dims["evidence_completeness_assessment"] = False
    dims["data_quality_awareness"] = False
    dims["decision_support_confidence"] = False
    dims["evidence_backed_recommendation"] = False  # agents report findings, not a Recommendation object
    dims["missing_context_disclosure"] = any(a and a.limitations for a in (fa, aa, ea, ca, cb) if a)
    dims["unified_explanation"] = False  # 5 separate findings, never merged
    return dims


def level3_collaborative_framework(run: AnalysisRunResult, collab: CollaborationResult) -> dict:
    """LEVEL 3: the complete Day-4 collaborative system."""
    dims = level2_independent_agents(run, collab.agents)  # Level 3 has everything Level 2 has, plus:
    # "Cross-agent" requires evidence from at least 2 DISTINCT agents —
    # a single agent's evidence item is not a cross-agent correlation.
    distinct_agents = {e.source_agent for e in collab.evidence}
    dims["cross_agent_evidence_correlation"] = len(distinct_agents) >= 2
    dims["agreement_assessment"] = collab.agreement is not None
    dims["evidence_completeness_assessment"] = collab.completeness is not None
    dims["data_quality_awareness"] = collab.data_quality is not None  # feeds directly into confidence
    dims["decision_support_confidence"] = collab.confidence is not None
    # A recommendation only counts as "evidence-backed" if at least one of
    # its evidence_refs actually corresponds to a real evidence item that
    # was produced — not merely a generic default reference.
    real_evidence_types = {e.evidence_type for e in collab.evidence}
    dims["evidence_backed_recommendation"] = any(
        any(ref in real_evidence_types for ref in r.evidence_refs) for r in collab.recommendations
    )
    dims["unified_explanation"] = True  # AI Energy Engineer always produces one narrative
    return dims
