
"""
orchestration/energy_engineer.py — EnergyOS AI Energy Engineer

Deterministic, evidence-grounded narrative generation.
No external LLM and no fabricated findings.

TrustAI Safety Enforcement:
- Withholds forecast-dependent conclusions when reliability is poor.
- Never treats a blocked incident as proof of normal operation.
- Preserves raw evidence for auditing and research.
- Keeps collaboration scores visible without presenting them as
  validated decision confidence.
- Does not invent equipment faults or root causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..analysis_pipeline import AnalysisRunResult
from .orchestrator import CollaborationResult


ROOT_CAUSE_UNCONFIRMED = (
    "Root cause is not confirmed by available evidence."
)


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


def generate_narrative(
    run: AnalysisRunResult,
    collab: CollaborationResult,
) -> EnergyEngineerNarrative:

    # ---------------------------------------------------------
    # 1. No analytical series
    # ---------------------------------------------------------

    if run.primary is None:
        return EnergyEngineerNarrative(
            what=(
                "No analytical series was available "
                "in this dataset."
            ),
            where="n/a",
            when="n/a",
            why=(
                "Not applicable — no POWER/ENERGY/DEMAND "
                "series could be analyzed."
            ),
            energy_impact="Unavailable.",
            cost_impact="Unavailable.",
            carbon_impact="Unavailable.",
            evidence=[],
            missing_context=[run.selection_reason],
            confidence="Not computed — no series available.",
            root_cause=ROOT_CAUSE_UNCONFIRMED,
            recommended_action=(
                "Provide a dataset with a mapped "
                "POWER/ENERGY/DEMAND series and "
                "a timestamp column."
            ),
        )

    series = run.primary.series
    where = series.label()

    evidence_lines = [
        f"[{e.source_agent}] {e.description}"
        for e in collab.evidence
    ]

    missing_context = []

    for agent_id in collab.unavailable_agents:
        agent = collab.agents.get(agent_id)

        if agent and agent.limitations:
            missing_context.append(
                f"{agent_id}: {agent.limitations[0]}"
            )
        elif agent:
            missing_context.append(
                f"{agent_id}: {agent.finding}"
            )

    # ---------------------------------------------------------
    # 2. Analysis skipped due to insufficient data
    # ---------------------------------------------------------

    if run.primary.skipped_reason:
        return EnergyEngineerNarrative(
            what=(
                "Analysis could not be performed: "
                f"{run.primary.skipped_reason}"
            ),
            where=where,
            when="n/a",
            why=(
                "Insufficient data — no forecast or "
                "anomaly analysis was run for this series."
            ),
            energy_impact=(
                "Unavailable — analysis did not run."
            ),
            cost_impact=(
                "Unavailable — analysis did not run."
            ),
            carbon_impact=(
                "Unavailable — analysis did not run."
            ),
            evidence=evidence_lines,
            missing_context=[
                run.primary.skipped_reason,
                *missing_context,
            ],
            confidence="Not validated — analysis was skipped.",
            root_cause=ROOT_CAUSE_UNCONFIRMED,
            recommended_action=(
                "Collect more data for this series "
                "before drawing any conclusion."
            ),
        )

    # ---------------------------------------------------------
    # 3. TrustAI Forecast Reliability Safety Enforcement
    # ---------------------------------------------------------

    decision_supported = getattr(
        collab,
        "decision_supported",
        None,
    )

    reliability_status = getattr(
        collab,
        "reliability_status",
        "NOT_ASSESSED",
    )

    reliability_reasons = list(
        getattr(
            collab,
            "reliability_reasons",
            [],
        )
    )

    withheld_incident_count = getattr(
        collab,
        "withheld_incident_count",
        0,
    )

    # If orchestration has not supplied a reliability verdict,
    # independently assess the available forecast.
    if decision_supported is None:
        from ..forecasting.reliability import (
            assess_forecast_reliability,
        )

        forecast_engine = getattr(
            run.primary,
            "forecast",
            None,
        )

        forecast_result = (
            getattr(
                forecast_engine,
                "forecast_result",
                None,
            )
            if forecast_engine is not None
            else None
        )

        if forecast_result is not None:
            assessment = assess_forecast_reliability(
                forecast_result
            )

            decision_supported = (
                assessment.suitable_for_anomaly_evidence
            )

            reliability_status = assessment.status
            reliability_reasons = list(
                assessment.reasons
            )

        else:
            decision_supported = False
            reliability_status = "NOT_ASSESSED"
            reliability_reasons = [
                "No usable forecast was available "
                "for reliability assessment."
            ]

    # ---------------------------------------------------------
    # 4. Withhold unreliable forecast-dependent conclusions
    # ---------------------------------------------------------

    if not decision_supported:

        if not reliability_reasons:
            reliability_reasons = [
                "Forecast reliability could not "
                "be established."
            ]

        test_period = (
            f"Test period "
            f"{run.primary.split.test_start} "
            f"to {run.primary.split.test_end}"
            if run.primary.split
            else "n/a"
        )

        safety_evidence = [
            (
                "TrustAI reliability gate: "
                f"{reliability_status}."
            ),
            (
                f"{withheld_incident_count} "
                "incident candidate(s) withheld "
                "from actionable conclusions."
            ),
            (
                "Raw statistical findings are retained "
                "for research and independent review."
            ),
        ]

        # Preserve the numerical collaboration score for
        # compatibility and research transparency.
        # Explicitly distinguish it from decision confidence.
        if collab.confidence:
            confidence_str = (
                f"{collab.confidence.score:.1f}/100 — "
                f"{collab.confidence.label} "
                "(raw collaboration score only; "
                "NOT validated decision confidence). "
                "WITHHELD — forecast reliability "
                "requirements were not met."
            )
        else:
            confidence_str = (
                "No collaboration score available. "
                "WITHHELD — forecast reliability "
                "requirements were not met."
            )

        return EnergyEngineerNarrative(
            what=(
                "Forecast-dependent energy conclusions "
                "are WITHHELD pending reliability review."
            ),
            where=where,
            when=test_period,
            why=(
                "TrustAI rejected the forecast as "
                "sufficient evidence for actionable "
                "anomaly conclusions. "
                + " ".join(reliability_reasons)
            ),
            energy_impact=(
                "Not validated for decision-making."
            ),
            cost_impact=(
                "Not validated for decision-making."
            ),
            carbon_impact=(
                "Not validated for decision-making."
            ),
            evidence=[
                *safety_evidence,
                *evidence_lines,
            ],
            missing_context=[
                *reliability_reasons,
                *missing_context,
            ],
            confidence=confidence_str,
            root_cause=ROOT_CAUSE_UNCONFIRMED,
            recommended_action=(
                "Review forecasting accuracy, meter "
                "semantics, operating conditions, "
                "data quality, and alternative models "
                "before issuing energy anomaly "
                "recommendations."
            ),
        )

    # ---------------------------------------------------------
    # 5. Forecast passed preliminary reliability checks
    # ---------------------------------------------------------

    if not collab.incidents:

        what = (
            "No statistically significant consumption "
            "deviation was detected against the "
            "selected forecasting baseline."
        )

        when = (
            f"Test period "
            f"{run.primary.split.test_start} "
            f"to {run.primary.split.test_end}"
            if run.primary.split
            else "n/a"
        )

        why = (
            "No forecast-based incident candidate "
            "met the current detection criteria. "
            "This does not prove normal equipment "
            "operation."
        )

    else:
        incident = collab.incidents[0]

        if incident.trigger_evidence:
            an_evidence = incident.trigger_evidence[0]
            what = an_evidence.description

            event_details = (
                an_evidence.detail or {}
            )

            event_count = len(
                event_details.get(
                    "events",
                    [],
                )
            )

            why = (
                f"{event_count} anomaly event(s) "
                "were statistically flagged against "
                "the forecast baseline. Physical "
                "root cause is not confirmed."
            )

        else:
            what = (
                "A statistical incident candidate "
                "was identified."
            )

            why = (
                "The collaboration engine identified "
                "an incident candidate, but no "
                "trigger evidence was attached."
            )

        when = (
            f"{incident.time_start} "
            f"to {incident.time_end}"
            if incident.time_start
            else "n/a"
        )

    # ---------------------------------------------------------
    # 6. Evidence-grounded impact calculations
    # ---------------------------------------------------------

    energy_ev = next(
        (
            e
            for e in collab.evidence
            if e.evidence_type == "ENERGY_IMPACT"
        ),
        None,
    )

    energy_impact = (
        f"{energy_ev.value:.1f} "
        f"{energy_ev.unit} "
        f"({energy_ev.provenance})"
        if energy_ev is not None
        and energy_ev.value is not None
        else "Unavailable."
    )

    cost_ev = next(
        (
            e
            for e in collab.evidence
            if e.evidence_type == "COST_IMPACT"
        ),
        None,
    )

    cost_impact = (
        f"{cost_ev.unit or ''} "
        f"{cost_ev.value:.2f} "
        f"({cost_ev.provenance})"
        if cost_ev is not None
        and cost_ev.value is not None
        else "Unavailable — no validated cost impact."
    )

    carbon_ev = next(
        (
            e
            for e in collab.evidence
            if e.evidence_type == "CARBON_IMPACT"
        ),
        None,
    )

    carbon_impact = (
        f"{carbon_ev.value:.2f} "
        f"{carbon_ev.unit} "
        f"({carbon_ev.provenance})"
        if carbon_ev is not None
        and carbon_ev.value is not None
        else "Unavailable — no validated carbon impact."
    )

    # ---------------------------------------------------------
    # 7. Confidence and recommendations
    # ---------------------------------------------------------

    confidence_str = (
        f"{collab.confidence.score:.1f}/100 — "
        f"{collab.confidence.label} "
        "(collaboration score; not a statistical probability)"
        if collab.confidence
        else "not computed"
    )

    recommended_action = (
        collab.recommendations[0].recommended_action
        if collab.recommendations
        else "No forecast-based action recommended."
    )

    # ---------------------------------------------------------
    # 8. Return evidence-grounded narrative
    # ---------------------------------------------------------

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
        root_cause=ROOT_CAUSE_UNCONFIRMED,
        recommended_action=recommended_action,
    )
