"""
orchestration/terminal.py — Day 4 Part 10: Terminal Experience

Renders MULTI-AGENT COLLABORATION and AI ENERGY ENGINEER sections
directly from CollaborationResult / EnergyEngineerNarrative — no
recomputation, matching every other terminal renderer in this codebase.
"""

from __future__ import annotations

from .energy_engineer import EnergyEngineerNarrative
from .orchestrator import CollaborationResult

LINE = "=" * 60

_AGENT_LABELS = {
    "forecast_agent": "Forecast Agent",
    "anomaly_agent": "Anomaly Agent",
    "equipment_agent": "Equipment Agent",
    "cost_agent": "Cost Agent",
    "carbon_agent": "Carbon Agent",
}


def render_collaboration_report(collab: CollaborationResult) -> str:
    lines = [LINE, "              MULTI-AGENT COLLABORATION", LINE, ""]

    for key, label in _AGENT_LABELS.items():
        agent = collab.agents.get(key)
        status = agent.status if agent else "UNAVAILABLE"
        dots = "." * max(1, 22 - len(label))
        lines.append(f"{label} {dots} {status}")
    lines.append("")

    lines.append(f"Evidence correlated: {len(collab.evidence)} item(s)")
    lines.append(f"Incident candidates: {len(collab.incidents)}")
    for incident in collab.incidents:
        lines.append(
            f"  - {incident.incident_id}: series={incident.series_id}, "
            f"evidence={sorted(incident.evidence_types_present)}"
        )
    lines.append("")

    ag = collab.agreement
    if ag and ag.score is not None:
        agreement_str = f"{ag.score:.1f}% ({ag.status})"
    elif ag:
        agreement_str = f"n/a ({ag.status})"
    else:
        agreement_str = "n/a"
    lines.append(f"Agent Agreement:       {agreement_str}")

    comp = collab.completeness
    lines.append(f"Evidence Completeness: {comp.score:.1f}%" if comp else "Evidence Completeness: n/a")

    if collab.data_quality is not None:
        lines.append(f"Data Quality:          {collab.data_quality:.1f}/100 ({collab.data_quality_status})")
    else:
        lines.append("Data Quality: n/a")

    conf = collab.confidence
    lines.append(f"Decision Confidence:   {conf.score:.1f}/100" if conf else "Decision Confidence: n/a")
    lines.append("")
    lines.append(LINE)
    return "\n".join(lines)


def render_energy_engineer_report(narrative: EnergyEngineerNarrative) -> str:
    lines = [LINE, "                  AI ENERGY ENGINEER", LINE, ""]
    lines.append("WHAT:")
    lines.append(narrative.what)
    lines.append("")
    lines.append("WHERE:")
    lines.append(narrative.where)
    lines.append("")
    lines.append("WHEN:")
    lines.append(narrative.when)
    lines.append("")
    lines.append("WHY:")
    lines.append(narrative.why)
    lines.append("")
    lines.append("ENERGY IMPACT:")
    lines.append(narrative.energy_impact)
    lines.append("")
    lines.append("COST IMPACT:")
    lines.append(narrative.cost_impact)
    lines.append("")
    lines.append("CARBON IMPACT:")
    lines.append(narrative.carbon_impact)
    lines.append("")
    lines.append("EVIDENCE:")
    if narrative.evidence:
        for e in narrative.evidence:
            lines.append(f"  \u2022 {e}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("MISSING CONTEXT:")
    if narrative.missing_context:
        for m in narrative.missing_context:
            lines.append(f"  \u2022 {m}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("CONFIDENCE:")
    lines.append(narrative.confidence)
    lines.append("")
    lines.append("ROOT CAUSE:")
    lines.append(narrative.root_cause)
    lines.append("")
    lines.append("RECOMMENDED ACTION:")
    lines.append(narrative.recommended_action)
    lines.append(LINE)
    return "\n".join(lines)
