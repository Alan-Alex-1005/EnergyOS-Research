"""
evaluation/comparison.py — Day 5 Part 8: Research Comparison Table

Machine-produced Level 1 vs Level 2 vs Level 3 comparison, aggregated
across ALL scenarios. Rows are the measurable dimensions/metrics defined
in dimensions.py/coverage.py/diversity.py/recommendation_support.py.
Values are actual counts from runtime evaluation — nothing invented.
"""

from __future__ import annotations

import csv

from .dimensions import DIMENSIONS


def build_comparison_rows(scenario_results: list) -> list:
    rows = []
    for dim in DIMENSIONS:
        applicable_count = sum(1 for sr in scenario_results if dim in sr.applicable)
        l1_count = sum(1 for sr in scenario_results if dim in sr.applicable and sr.level1_support.get(dim) is True)
        l2_count = sum(1 for sr in scenario_results if dim in sr.applicable and sr.level2_support.get(dim) is True)
        l3_count = sum(1 for sr in scenario_results if dim in sr.applicable and sr.level3_support.get(dim) is True)
        rows.append(
            {
                "dimension": dim,
                "applicable_scenarios": applicable_count,
                "level1_isolated_analytics": l1_count,
                "level2_independent_agents": l2_count,
                "level3_collaborative_framework": l3_count,
            }
        )

    n = len(scenario_results) or 1
    rows.append(
        {
            "dimension": "MEAN_EXPLANATION_COVERAGE_PCT",
            "applicable_scenarios": n,
            "level1_isolated_analytics": round(sum(sr.level1_coverage.percentage for sr in scenario_results) / n, 1),
            "level2_independent_agents": round(sum(sr.level2_coverage.percentage for sr in scenario_results) / n, 1),
            "level3_collaborative_framework": round(sum(sr.level3_coverage.percentage for sr in scenario_results) / n, 1),
        }
    )
    rows.append(
        {
            "dimension": "MEAN_EVIDENCE_DIVERSITY_COUNT",
            "applicable_scenarios": n,
            "level1_isolated_analytics": round(sum(sr.level1_diversity.count for sr in scenario_results) / n, 2),
            "level2_independent_agents": round(sum(sr.level2_diversity.count for sr in scenario_results) / n, 2),
            "level3_collaborative_framework": round(sum(sr.level3_diversity.count for sr in scenario_results) / n, 2),
        }
    )
    return rows


def write_comparison_csv(scenario_results: list, path: str) -> str:
    rows = build_comparison_rows(scenario_results)
    fieldnames = ["dimension", "applicable_scenarios", "level1_isolated_analytics", "level2_independent_agents", "level3_collaborative_framework"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
