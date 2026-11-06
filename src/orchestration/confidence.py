"""
orchestration/confidence.py — Day 4 Part 7: Decision-Support Confidence

NOT a statistical probability. A transparent, weighted blend of three
already-computed, independently-documented components:

    confidence = 45% * Data Quality (Day 1's existing score, reused verbatim)
               + 30% * Agent Agreement (Part 4)
               + 25% * Evidence Completeness (Part 5)

If Agent Agreement is not computable (0 or 1 available participants —
see agreement.py), its weight is proportionally redistributed across
the remaining two components rather than treated as 0 (an unavailable
component should never silently drag confidence down as if it were bad
news; it's simply missing information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

WEIGHTS = {
    "data_quality": 0.45,
    "agent_agreement": 0.30,
    "evidence_completeness": 0.25,
}

LABEL = "Decision-Support Confidence (not a statistical probability)"


@dataclass
class ConfidenceResult:
    score: float
    label: str = LABEL
    components: dict = field(default_factory=dict)
    weights_used: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def compute_confidence(data_quality: float, agreement_score: Optional[float], completeness_score: float) -> ConfidenceResult:
    components = {"data_quality": data_quality, "evidence_completeness": completeness_score}
    weights_used = {"data_quality": WEIGHTS["data_quality"], "evidence_completeness": WEIGHTS["evidence_completeness"]}
    notes = []

    if agreement_score is not None:
        components["agent_agreement"] = agreement_score
        weights_used["agent_agreement"] = WEIGHTS["agent_agreement"]
    else:
        notes.append(
            "Agent Agreement is not computable (fewer than 2 available participants); its weight "
            f"({WEIGHTS['agent_agreement']*100:.0f}%) was redistributed proportionally to the remaining components."
        )

    total_weight = sum(weights_used.values())
    normalized_weights = {k: v / total_weight for k, v in weights_used.items()}
    score = sum(components[k] * normalized_weights[k] for k in components)

    return ConfidenceResult(
        score=round(score, 1),
        components=components,
        weights_used={k: round(v, 4) for k, v in normalized_weights.items()},
        notes=notes,
    )
