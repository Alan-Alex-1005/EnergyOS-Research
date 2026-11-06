"""
evaluation/recommendation_support.py — Day 5 Part 5: Recommendation Support

Deterministic classification into NONE / GENERIC / CONTEXTUAL /
IMPACT_AWARE, based entirely on the `evidence_refs` already attached to
each Recommendation (Day 4) — never inferred or guessed. Levels 1 and 2
have no Recommendation Engine at all (Day 2 produces no recommended
action; Day 3 agents produce findings, not recommendations), so they are
NONE by construction — not because we're penalizing them, but because
that object genuinely doesn't exist at those levels.
"""

from __future__ import annotations

NONE_ = "NONE"
GENERIC = "GENERIC"
CONTEXTUAL = "CONTEXTUAL"
IMPACT_AWARE = "IMPACT_AWARE"


def classify_recommendation_support(recommendations) -> str:
    if not recommendations:
        return NONE_

    all_refs = set()
    for r in recommendations:
        all_refs.update(r.evidence_refs)

    if all_refs & {"COST_IMPACT", "CARBON_IMPACT", "ENERGY_IMPACT"}:
        return IMPACT_AWARE
    if "EQUIPMENT_CONTEXT" in all_refs:
        return CONTEXTUAL
    return GENERIC
