"""
evaluation/coverage.py — Day 5 Part 3: Explanation Coverage Metric

    Explanation Coverage = supported dimensions / applicable dimensions

`applicable dimensions` come from dimensions.applicable_dimensions() —
identical across levels for a given scenario (schedule_evidence always
excluded; cost/carbon excluded only if that scenario didn't configure
them). Only `supported` (the numerator) varies by level. This is a
framework COVERAGE metric — never prediction accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoverageResult:
    numerator: int
    denominator: int
    percentage: float
    supported_dimensions: list
    applicable_dimensions: list


def compute_explanation_coverage(support: dict, applicable: list) -> CoverageResult:
    supported = [d for d in applicable if support.get(d) is True]
    denominator = len(applicable)
    numerator = len(supported)
    percentage = round(100.0 * numerator / denominator, 1) if denominator else 0.0
    return CoverageResult(
        numerator=numerator,
        denominator=denominator,
        percentage=percentage,
        supported_dimensions=supported,
        applicable_dimensions=list(applicable),
    )
