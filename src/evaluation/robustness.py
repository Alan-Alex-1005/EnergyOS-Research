"""
evaluation/robustness.py — Day 5 Part 7: Robustness Matrix

One row per scenario, columns = actual runtime statuses already present
on the ScenarioResult (agents dict, collab). No new computation.
"""

from __future__ import annotations


def build_robustness_matrix(scenario_results: list) -> list:
    rows = []
    for sr in scenario_results:
        agents = sr.agents
        collab = sr.collab

        def status(key):
            a = agents.get(key)
            return a.status if a else "UNAVAILABLE"

        if sr.run.primary is None:
            result_label = "NO_SERIES"
        elif sr.run.primary.skipped_reason:
            result_label = "INSUFFICIENT_DATA"
        elif collab.incidents:
            result_label = f"{len(collab.incidents)} INCIDENT(S)"
        else:
            result_label = "NORMAL"

        rows.append(
            {
                "scenario": sr.scenario_id,
                "input_format": sr.input_format,
                "forecast": status("forecast_agent"),
                "anomaly": status("anomaly_agent"),
                "equipment": status("equipment_agent"),
                "cost": status("cost_agent"),
                "carbon": status("carbon_agent"),
                "collaboration": "DONE" if sr.run.primary and not sr.run.primary.skipped_reason else "UNAVAILABLE",
                "explanation_coverage_l3_pct": sr.level3_coverage.percentage,
                "result": result_label,
            }
        )
    return rows
