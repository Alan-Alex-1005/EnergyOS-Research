"""
evaluation/diversity.py — Day 5 Part 4: Evidence Diversity

A transparent count of distinct EVIDENCE CATEGORIES (not meta-capabilities
like "agreement_assessment") available to a level, using the same
support dict levels.py already produced. Measures DIVERSITY of evidence,
never correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dimensions import EVIDENCE_CATEGORY_DIMENSIONS


@dataclass
class DiversityResult:
    categories: list
    count: int


def compute_evidence_diversity(support: dict, applicable: list) -> DiversityResult:
    categories = [
        label
        for dim, label in EVIDENCE_CATEGORY_DIMENSIONS.items()
        if dim in applicable and support.get(dim) is True
    ]
    return DiversityResult(categories=sorted(categories), count=len(categories))
