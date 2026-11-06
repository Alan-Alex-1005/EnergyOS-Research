"""
orchestration/correlation.py — Day 4 Part 3: Incident Correlation

Correlates evidence into an Incident ONLY when there's a genuine trigger
(anomalies_detected > 0 for that series). Everything else that shares
the same series_id and overlapping time window is attached as
SUPPORTING evidence — never presented as a proven cause. No incident is
created for a series with zero detected anomalies (that's "normal
operation", Scenario A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .evidence import SharedEvidence


@dataclass
class Incident:
    incident_id: str
    series_id: str
    time_start: Optional[str]
    time_end: Optional[str]
    trigger_evidence: list = field(default_factory=list)
    supporting_evidence: list = field(default_factory=list)

    @property
    def all_evidence(self) -> list:
        return self.trigger_evidence + self.supporting_evidence

    @property
    def evidence_types_present(self) -> set:
        return {e.evidence_type for e in self.all_evidence}


def _windows_overlap(a_start, a_end, b_start, b_end) -> bool:
    """True if two time windows overlap. A missing (None) window (e.g.
    equipment context has no specific time range) is treated as always
    overlapping — it's dataset-level context, not time-bound."""
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return True
    return not (a_end < b_start or b_end < a_start)


def correlate_incidents(evidence_list: list) -> list:
    by_series: dict = {}
    for e in evidence_list:
        by_series.setdefault(e.series_id, []).append(e)

    incidents: list = []
    for series_id, items in by_series.items():
        triggers = [e for e in items if e.evidence_type == "ANOMALY" and (e.value or 0) > 0]
        if not triggers:
            continue  # no genuine trigger for this series -> no incident (never fabricated)

        trigger = triggers[0]
        supporting = [
            e
            for e in items
            if e is not trigger
            and _windows_overlap(trigger.time_start, trigger.time_end, e.time_start, e.time_end)
        ]

        incidents.append(
            Incident(
                incident_id=f"incident:{series_id}",
                series_id=series_id,
                time_start=trigger.time_start,
                time_end=trigger.time_end,
                trigger_evidence=[trigger],
                supporting_evidence=supporting,
            )
        )

    return incidents
