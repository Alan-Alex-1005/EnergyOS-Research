"""
agents/equipment_agent.py — Day 3: Equipment / Context Agent

Reports ONLY equipment/location/status/occupancy/schedule context that
actually exists in the mapped dataset (Day 1's capability report + the
selected series' own context fields). Never invents equipment, a
location, an occupancy pattern, a schedule, or a root cause. Schedule
deviation is only ever reported if a genuine schedule is known — since
no schedule configuration currently exists anywhere in this system, this
agent NEVER claims a schedule deviation (that would be fabrication); it
reports schedule-relevant context as INSUFFICIENT_EVIDENCE instead.
"""

from __future__ import annotations

from ..analysis_pipeline import SeriesAnalysisResult
from .base import STATUS_DONE, STATUS_LIMITED, STATUS_UNAVAILABLE, AgentResult

AGENT_ID = "equipment_agent"

_CONTEXT_FIELDS = ("equipment_id", "device_id", "meter_id", "site", "building", "floor", "zone", "room")


def run_equipment_agent(series_result: SeriesAnalysisResult, capability_report) -> AgentResult:
    series = series_result.series
    present_context = {f: getattr(series, f) for f in _CONTEXT_FIELDS if getattr(series, f) is not None}

    caps = capability_report.as_dict()
    equipment_cap = caps.get("Equipment Analysis")
    schedule_cap = caps.get("Schedule Analysis")
    occupancy_cap = caps.get("Occupancy Context")
    weather_cap = caps.get("Weather Context")

    evidence = {
        "identifier_context": present_context,
        "equipment_capability": equipment_cap.status if equipment_cap else None,
        "schedule_capability": schedule_cap.status if schedule_cap else None,
        "occupancy_capability": occupancy_cap.status if occupancy_cap else None,
        "weather_capability": weather_cap.status if weather_cap else None,
    }
    limitations = []
    for cap in (equipment_cap, schedule_cap, occupancy_cap, weather_cap):
        if cap and cap.status != "AVAILABLE":
            limitations.append(f"{cap.name}: {cap.reason}")

    # Schedule deviation is NEVER inferred here — no schedule definition
    # exists anywhere in this system yet (Day 3 explicitly forbids
    # fabricating one). We only ever report whether schedule-relevant
    # context (status/occupancy + timestamp) exists at all.
    limitations.append(
        "No explicit schedule definition exists in this system yet; schedule DEVIATION is never inferred, "
        "only whether schedule-relevant context (status/occupancy) is present."
    )

    if present_context:
        finding = f"Identifier context present: {present_context}."
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_DONE,
            finding=finding,
            evidence=evidence,
            provenance={"identifier_context": "MEASURED"},
            limitations=limitations,
        )

    if equipment_cap and equipment_cap.status == "LIMITED":
        finding = f"No equipment/device/meter identifier mapped; treated as a single asset ({equipment_cap.reason})."
        return AgentResult(
            agent_id=AGENT_ID,
            status=STATUS_LIMITED,
            finding=finding,
            evidence=evidence,
            limitations=limitations,
        )

    return AgentResult(
        agent_id=AGENT_ID,
        status=STATUS_UNAVAILABLE,
        finding="No equipment, location, status, or occupancy context is available in this dataset.",
        evidence=evidence,
        limitations=limitations,
    )
