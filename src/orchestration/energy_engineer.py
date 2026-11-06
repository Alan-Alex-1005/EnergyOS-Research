"""
orchestration/energy_engineer.py — Day 4 Part 9: AI Energy Engineer

Deterministic — no external LLM. Every field is assembled directly from
CollaborationResult / AnalysisRunResult data that already exists; nothing
is generated freeform. If root cause isn't proven (it never is in this
system — no schedule model exists), that is stated explicitly rather
than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..analysis_pipeline import AnalysisRunResult
from .orchestrator import CollaborationResult

ROOT_CAUSE_UNCONFIRMED = "Root cause is not confirmed by available evidence."


@dataclass
class EnergyEngineerNarrative:
    what: str
    where: str
    when: str
    why: str
    energy_impact: str
    cost_impact: str
    carbon_impact: str
    evidence: list = field(default_factory=list)
    missing_context: list = field(default_factory=list)
    confidence: str = ""
    root_cause: str = ROOT_CAUSE_UNCONFIRMED
    recommended_action: str = ""


def generate_narrative(run: AnalysisRunResult, collab: CollaborationResult) -> EnergyEngineerNarrative:
    if run.primary is None:
        return EnergyEngineerNarrative(
            what="No analytical series was available in this dataset.",
            where="n/a",
            when="n/a",
            why="Not applicable — no POWER/ENERGY/DEMAND series could be analyzed.",
            energy_impact="Unavailable.",
            cost_impact="Unavailable.",
            carbon_impact="Unavailable.",
            evidence=[],
            missing_context=[run.selection_reason],
            confidence="Not computed — no series available.",
            root_cause=ROOT_CAUSE_UNCONFIRMED,
            recommended_action="Provide a dataset with a mapped POWER/ENERGY/DEMAND series and a timestamp column.",
        )

    series = run.primary.series
    where = series.label()

    if run.primary.skipped_reason:
        # Insufficient data — NEVER report this as "normal operation" (that
        # would misrepresent absence-of-analysis as a confirmed finding).
        return EnergyEngineerNarrative(
            what=f"Analysis could not be performed: {run.primary.skipped_reason}",
            where=where,
            when="n/a",
            why="Insufficient data — no forecast or anomaly analysis was run for this series.",
            energy_impact="Unavailable — analysis did not run.",
            cost_impact="Unavailable — analysis did not run.",
            carbon_impact="Unavailable — analysis did not run.",
            evidence=[f"[{e.source_agent}] {e.description}" for e in collab.evidence],
            missing_context=[run.primary.skipped_reason],
            confidence=(
                f"{collab.confidence.score:.1f}/100 — {collab.confidence.label} (reflects the lack of analysis, not a real result)"
                if collab.confidence
                else "not computed"
            ),
            root_cause=ROOT_CAUSE_UNCONFIRMED,
            recommended_action="Collect more data for this series before drawing any conclusion.",
        )

    if not collab.incidents:
        an = collab.agents.get("anomaly_agent")
        what = "No anomalous consumption was detected; the series behaved within the range the forecast model expected."
        when = (
            f"Test period {run.primary.split.test_start} to {run.primary.split.test_end}"
            if run.primary.split
            else "n/a"
        )
        why = "Normal operation — no deviation warranting investigation was found."
    else:
        incident = collab.incidents[0]
        an_evidence = incident.trigger_evidence[0]
        what = an_evidence.description
        when = f"{incident.time_start} to {incident.time_end}" if incident.time_start else "n/a"
        why = (
            f"{an_evidence.detail.get('events', []).__len__()} anomaly event(s) were statistically flagged "
            f"against the series' expected (forecast) behavior."
        )

    energy_ev = next((e for e in collab.evidence if e.evidence_type == "ENERGY_IMPACT"), None)
    energy_impact = f"{energy_ev.value:.1f} {energy_ev.unit} ({energy_ev.provenance})" if energy_ev else "Unavailable."

    cost_ev = next((e for e in collab.evidence if e.evidence_type == "COST_IMPACT"), None)
    cost_impact = f"{cost_ev.unit or ''} {cost_ev.value:.2f} ({cost_ev.provenance})" if cost_ev else "Unavailable — no tariff configured."

    carbon_ev = next((e for e in collab.evidence if e.evidence_type == "CARBON_IMPACT"), None)
    carbon_impact = f"{carbon_ev.value:.2f} {carbon_ev.unit} ({carbon_ev.provenance})" if carbon_ev else "Unavailable — no carbon factor configured."

    evidence_lines = [f"[{e.source_agent}] {e.description}" for e in collab.evidence]

    missing_context = []
    for agent_id in collab.unavailable_agents:
        agent = collab.agents.get(agent_id)
        if agent and agent.limitations:
            missing_context.append(f"{agent_id}: {agent.limitations[0]}")
        elif agent:
            missing_context.append(f"{agent_id}: {agent.finding}")

    confidence_str = (
        f"{collab.confidence.score:.1f}/100 — {collab.confidence.label}" if collab.confidence else "not computed"
    )

    recommended_action = collab.recommendations[0].recommended_action if collab.recommendations else "No action recommended."

    return EnergyEngineerNarrative(
        what=what,
        where=where,
        when=when,
        why=why,
        energy_impact=energy_impact,
        cost_impact=cost_impact,
        carbon_impact=carbon_impact,
        evidence=evidence_lines,
        missing_context=missing_context,
        confidence=confidence_str,
        root_cause=ROOT_CAUSE_UNCONFIRMED,  # never proven in this system — always explicit
        recommended_action=recommended_action,
    )
