"""
evaluation/dimensions.py — Day 5 Part 2: Measurable Research Dimensions

Eighteen objective, boolean-valued dimensions, each measuring whether a
given evaluation level's OWN runtime output actually provides that kind
of evidence/capability — never an accuracy or quality judgment.

`schedule_evidence` is ALWAYS not-applicable (N/A) system-wide: no
schedule model exists anywhere in this codebase (Day 3's Equipment Agent
explicitly refuses to infer one), so it is excluded from every level's
denominator rather than silently scored False for one level and not
another.

`cost_impact` and `carbon_impact` are OPTIONAL, config-dependent
dimensions: whether they are "applicable" for a given scenario depends
on whether that scenario configured a tariff/carbon factor at all — NOT
on which level is being evaluated. This keeps the denominator identical
across Level 1/2/3 for the same scenario (spec: "do not manipulate the
denominator to favor the collaborative system").
"""

from __future__ import annotations

DIMENSIONS = [
    "abnormal_behavior_detection",
    "forecast_context",
    "anomaly_event_information",
    "temporal_persistence",
    "excess_energy_impact",
    "equipment_context_evidence",
    "schedule_evidence",
    "cost_impact",
    "carbon_impact",
    "provenance",
    "cross_agent_evidence_correlation",
    "agreement_assessment",
    "evidence_completeness_assessment",
    "data_quality_awareness",
    "decision_support_confidence",
    "evidence_backed_recommendation",
    "missing_context_disclosure",
    "unified_explanation",
]

# System-wide: never applicable, at any level, in this codebase.
ALWAYS_NA_DIMENSIONS = {"schedule_evidence"}

# Scenario-dependent (config-driven), NOT level-dependent.
OPTIONAL_CONFIG_DIMENSIONS = {"cost_impact", "carbon_impact"}

# Maps each evidence-category-like dimension to a short evidence-diversity
# label (Part 4). Only dimensions that represent a distinct EVIDENCE
# CATEGORY (not a meta-capability like "agreement_assessment") are listed.
EVIDENCE_CATEGORY_DIMENSIONS = {
    "forecast_context": "forecast",
    "anomaly_event_information": "anomaly",
    "temporal_persistence": "temporal",
    "equipment_context_evidence": "equipment",
    "schedule_evidence": "schedule",
    "excess_energy_impact": "energy",
    "cost_impact": "cost",
    "carbon_impact": "carbon",
}


def applicable_dimensions(tariff_configured: bool, carbon_configured: bool) -> list:
    """
    The set of dimensions that are meaningful to score for a given
    scenario's configuration — identical across all three levels for
    that scenario (only the boolean support values differ by level).
    """
    applicable = [d for d in DIMENSIONS if d not in ALWAYS_NA_DIMENSIONS]
    if not tariff_configured:
        applicable = [d for d in applicable if d != "cost_impact"]
    if not carbon_configured:
        applicable = [d for d in applicable if d != "carbon_impact"]
    return applicable
