"""
orchestration/completeness.py — Day 4 Part 5: Evidence Completeness Score

TRANSPARENT, weighted, documented. Six evidence categories, split into
two tiers:

  CORE (data-dependent, 85% combined weight) — whether this evidence
  exists depends on the DATA ITSELF (enough samples, mapped columns):
    forecast            0.25
    anomaly             0.25
    equipment_context   0.20
    energy_impact       0.15

  OPTIONAL (config-dependent, 15% combined weight) — whether this
  evidence exists depends on a CONFIGURATION CHOICE (a tariff/carbon
  factor the user may not have supplied), not a data limitation. Given
  much less weight so a dataset isn't misleadingly "punished" for a
  choice the analyst made, not a data quality issue (spec requirement):
    cost                0.075
    carbon              0.075

Per-category credit: 1.0 if the corresponding agent/result status is
DONE, 0.5 if LIMITED (partial evidence), 0.0 if UNAVAILABLE/
INSUFFICIENT_DATA/ERROR/not run.

completeness = 100 * sum(weight_i * credit_i)  (weights already sum to 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents.base import AgentResult
from ..analysis_pipeline import SeriesAnalysisResult

WEIGHTS = {
    "forecast": 0.25,
    "anomaly": 0.25,
    "equipment_context": 0.20,
    "energy_impact": 0.15,
    "cost": 0.075,
    "carbon": 0.075,
}
CORE_CATEGORIES = ("forecast", "anomaly", "equipment_context", "energy_impact")
OPTIONAL_CATEGORIES = ("cost", "carbon")

_CREDIT = {"DONE": 1.0, "LIMITED": 0.5}


@dataclass
class CompletenessResult:
    score: float
    category_credit: dict = field(default_factory=dict)
    category_status: dict = field(default_factory=dict)
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))
    notes: list = field(default_factory=list)


def _status_of(agent: Optional[AgentResult]) -> str:
    return agent.status if agent is not None else "UNAVAILABLE"


def compute_completeness(agents: dict, series_result: Optional[SeriesAnalysisResult]) -> CompletenessResult:
    statuses = {
        "forecast": _status_of(agents.get("forecast_agent")),
        "anomaly": _status_of(agents.get("anomaly_agent")),
        "equipment_context": _status_of(agents.get("equipment_agent")),
        "energy_impact": "DONE" if (series_result and series_result.excess_energy and series_result.excess_energy.available) else "UNAVAILABLE",
        "cost": _status_of(agents.get("cost_agent")),
        "carbon": _status_of(agents.get("carbon_agent")),
    }
    credit = {k: _CREDIT.get(v, 0.0) for k, v in statuses.items()}
    score = 100.0 * sum(WEIGHTS[k] * credit[k] for k in WEIGHTS)

    notes = [
        f"Core categories (data-dependent, {sum(WEIGHTS[k] for k in CORE_CATEGORIES)*100:.0f}% combined weight): "
        + ", ".join(f"{k}={statuses[k]}" for k in CORE_CATEGORIES),
        f"Optional categories (config-dependent, {sum(WEIGHTS[k] for k in OPTIONAL_CATEGORIES)*100:.0f}% combined weight — "
        f"missing tariff/carbon-factor config is not treated as a data defect): "
        + ", ".join(f"{k}={statuses[k]}" for k in OPTIONAL_CATEGORIES),
    ]

    return CompletenessResult(
        score=round(score, 1),
        category_credit=credit,
        category_status=statuses,
        notes=notes,
    )
