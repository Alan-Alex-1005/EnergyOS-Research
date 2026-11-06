"""
orchestration/recommendations.py — Day 4 Part 8: Recommendation Engine

Deterministic, evidence-referencing recommendations. Every Recommendation
keeps three things separate:
  - finding: what the evidence actually shows
  - possible_explanation: a labeled HYPOTHESIS, only offered when there's
    at least contextual grounding for it (never a fabricated root cause)
  - recommended_action: what to do next

No recommendation is emitted without a concrete evidence item backing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Recommendation:
    finding: str
    possible_explanation: Optional[str]  # None if no defensible hypothesis exists
    recommended_action: str
    evidence_refs: list = field(default_factory=list)  # evidence_type strings this recommendation cites


def generate_recommendations(incidents: list, agents: dict) -> list:
    if not incidents:
        return [
            Recommendation(
                finding="No anomalies were detected for the analyzed series during the test period.",
                possible_explanation=None,
                recommended_action="No investigation required; consumption is within the range the forecast model expected.",
                evidence_refs=["ANOMALY"],
            )
        ]

    recommendations: list = []
    for incident in incidents:
        anomaly_evidence = incident.trigger_evidence[0] if incident.trigger_evidence else None
        events = anomaly_evidence.detail.get("events", []) if anomaly_evidence else []
        sustained = any(e.get("persistence_class") == "SUSTAINED" for e in events)

        if sustained:
            finding = f"A sustained anomalous consumption pattern was detected on series '{incident.series_id}'."
            action = "Investigate the sustained unexpected consumption — check for equipment left running or a process change during this window."
        else:
            finding = f"An anomalous consumption event was detected on series '{incident.series_id}'."
            action = "Investigate the anomalous consumption event; confirm whether it corresponds to a known operational change."

        equipment_evidence = next((e for e in incident.supporting_evidence if e.evidence_type == "EQUIPMENT_CONTEXT"), None)
        possible_explanation = None
        evidence_refs = ["ANOMALY"]
        if equipment_evidence and equipment_evidence.availability == "DONE" and equipment_evidence.detail.get("identifier_context"):
            ids = equipment_evidence.detail["identifier_context"]
            possible_explanation = (
                f"HYPOTHESIS (unconfirmed): the deviation may relate to the identified equipment/meter {ids}. "
                f"This is a candidate for investigation, not a confirmed cause."
            )
            action += f" Start with the identified equipment/meter: {ids}."
            evidence_refs.append("EQUIPMENT_CONTEXT")

        recommendations.append(
            Recommendation(finding=finding, possible_explanation=possible_explanation, recommended_action=action, evidence_refs=evidence_refs)
        )

        energy_evidence = next((e for e in incident.supporting_evidence if e.evidence_type == "ENERGY_IMPACT"), None)
        if energy_evidence and energy_evidence.value:
            recommendations.append(
                Recommendation(
                    finding=f"Estimated excess energy of {energy_evidence.value:.1f} {energy_evidence.unit} is associated with this incident's test period.",
                    possible_explanation=None,
                    recommended_action=f"Review the excess-energy impact ({energy_evidence.value:.1f} {energy_evidence.unit}) against operational records.",
                    evidence_refs=["ENERGY_IMPACT"],
                )
            )

        cost_evidence = next((e for e in incident.supporting_evidence if e.evidence_type == "COST_IMPACT"), None)
        if cost_evidence and cost_evidence.value:
            recommendations.append(
                Recommendation(
                    finding=f"Estimated cost impact of {cost_evidence.unit or ''} {cost_evidence.value:.2f} is associated with this incident.",
                    possible_explanation=None,
                    recommended_action="Review the estimated cost impact against the billing period.",
                    evidence_refs=["COST_IMPACT"],
                )
            )

        carbon_evidence = next((e for e in incident.supporting_evidence if e.evidence_type == "CARBON_IMPACT"), None)
        if carbon_evidence and carbon_evidence.value:
            recommendations.append(
                Recommendation(
                    finding=f"Estimated carbon impact of {carbon_evidence.value:.2f} {carbon_evidence.unit} is associated with this incident.",
                    possible_explanation=None,
                    recommended_action="Review the estimated carbon impact as part of sustainability reporting.",
                    evidence_refs=["CARBON_IMPACT"],
                )
            )

        if not equipment_evidence or equipment_evidence.availability != "DONE":
            recommendations.append(
                Recommendation(
                    finding="No equipment/meter identifier context was available for this series.",
                    possible_explanation=None,
                    recommended_action="Consider adding an equipment/device/meter identifier column to enable per-asset investigation in future runs.",
                    evidence_refs=["EQUIPMENT_CONTEXT"],
                )
            )

    return recommendations
