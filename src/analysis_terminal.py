"""
analysis_terminal.py — Day 2 Part S: Terminal Output for --analyze

Renders the Day-2 analysis exactly from AnalysisRunResult / the objects
it wraps — no recomputation, matching Part H/R's "one source of truth"
requirement. Appended after (not replacing) the Day-1 report.
"""

from __future__ import annotations

from .analysis_pipeline import AnalysisRunResult
from .agents.base import AgentResult

LINE = "=" * 60
SUB_LINE = "-" * 60


def _section(title: str) -> str:
    return f"{title}\n{SUB_LINE}"


def _row(label: str, value: str, width: int = 24) -> str:
    return f"{label:<{width}} {value}"


def render_agents_report(agents: dict[str, AgentResult]) -> str:
    lines = [LINE, "                SPECIALIZED ENERGY AGENTS", LINE, ""]

    labels = {
        "forecast_agent": "Forecast Agent",
        "anomaly_agent": "Anomaly Agent",
        "equipment_agent": "Equipment Agent",
        "cost_agent": "Cost Agent",
        "carbon_agent": "Carbon Agent",
    }
    for key, label in labels.items():
        agent = agents.get(key)
        if agent is None:
            continue
        lines.append(f"{label} {'.' * max(1, 22 - len(label))} {agent.status}")
        lines.append(f"Finding: {agent.finding}")
        if agent.metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in agent.metrics.items() if v is not None)
            if metrics_str:
                lines.append(f"Metrics: {metrics_str}")
        if agent.limitations:
            for lim in agent.limitations[:2]:
                lines.append(f"  \u2022 {lim}")
        lines.append("")

    lines.append(LINE)
    return "\n".join(lines)


def render_analysis_report(run: AnalysisRunResult) -> str:
    lines = []
    lines.append(LINE)
    lines.append("        ENERGYOS RESEARCH DEMO v2 — ANALYSIS (DAY 2)")
    lines.append(LINE)
    lines.append("")

    lines.append(_section("ANALYTICAL SERIES AVAILABLE"))
    if not run.all_series:
        lines.append("No POWER/ENERGY/DEMAND series were found in this dataset.")
    else:
        for s, suf in zip(run.all_series, run.all_sufficiency):
            marker = " <= selected" if s.series_id == run.selected_series_id else ""
            lines.append(f"{s.series_id:<40} samples={suf.sample_count:<6} {suf.status}{marker}")
    lines.append("")
    lines.append(f"Selection: {run.selection_reason}")
    lines.append("")

    if run.primary is None:
        lines.append("No series was analyzed.")
        for w in run.warnings:
            lines.append(f"\u2022 {w}")
        lines.append(LINE)
        return "\n".join(lines)

    p = run.primary
    lines.append(_section("ANALYSIS SERIES"))
    lines.append(_row("Series", p.series.series_id))
    lines.append(_row("Role", p.series.role))
    lines.append(_row("Unit", p.series.unit or "n/a"))
    lines.append(_row("Samples", str(p.series.sample_count)))
    lines.append("")

    if p.skipped_reason:
        lines.append(_section("FORECAST EVALUATION"))
        lines.append(f"SKIPPED — {p.skipped_reason}")
        lines.append("")
        lines.append(LINE)
        return "\n".join(lines)

    lines.append(_row("Sampling", p.prepared.sampling_interval_label or "irregular/unknown"))
    lines.append("")

    fc = p.forecast
    lines.append(_section("FORECAST EVALUATION"))
    if fc and fc.forecast_result:
        fr = fc.forecast_result
        lines.append(_row("Model", fc.selected_model))
        lines.append(_row("Selection basis", fc.selection_criterion))
        lines.append("")
        lines.append(_row("Training samples", str(fr.train_samples)))
        lines.append(_row("Testing samples", str(fr.test_samples)))
        lines.append("")
        lines.append(_row("MAE", f"{fr.mae:.4g}" if fr.mae is not None else "n/a"))
        lines.append(_row("RMSE", f"{fr.rmse:.4g}" if fr.rmse is not None else "n/a"))
        lines.append(_row("MAPE", f"{fr.mape:.2f}%" if fr.mape is not None else f"unavailable ({fr.mape_note})" if fr.mape_note else "n/a"))
        lines.append(_row("R2", f"{fr.r2:.4f}" if fr.r2 is not None else f"unavailable ({fr.r2_note})" if fr.r2_note else "n/a"))
        lines.append("")
        lines.append(_section("MODEL COMPARISON"))
        for e in fc.comparison:
            if not e.attempted:
                lines.append(f"{e.model_name:<20} SKIPPED — {e.reason}")
            else:
                lines.append(
                    f"{e.model_name:<20} validation RMSE={e.validation_rmse:.4g}  test RMSE={e.rmse:.4g}"
                )
        lines.append("")
        lines.append(_row("Selected", fc.selected_model))
    else:
        lines.append("No forecast could be produced." + (f" ({fc.selection_reason})" if fc else ""))
    lines.append("")

    lines.append(_section("ANOMALY ANALYSIS"))
    an = p.anomaly
    if an and an.applicable:
        lines.append(_row("Observations analysed", str(an.observations_analyzed)))
        lines.append(_row("Anomalies detected", str(an.anomalies_detected)))
        lines.append(_row("Anomaly rate", f"{an.anomaly_rate:.2f} %"))
        lines.append(_row("Peak statistical score", f"{an.peak_score:.2f}" if an.peak_score is not None else "n/a"))
        lines.append(_row("Highest severity", an.highest_severity))
        lines.append(_row("Anomaly events", str(len(an.events))))
        for evt in an.events[:5]:
            lines.append(
                f"  - {evt.event_id}: {evt.observation_count} obs, {evt.severity}, "
                f"{evt.direction.lower()}, {evt.persistence_class}"
            )
        if len(an.events) > 5:
            lines.append(f"  ... and {len(an.events) - 5} more (see results.json)")
    else:
        lines.append(f"Unavailable — {an.reason if an else 'no anomaly analysis was run.'}")
    lines.append("")

    lines.append(_section("ENERGY IMPACT"))
    ee = p.excess_energy
    if ee and ee.available:
        lines.append(_row("Estimated excess", f"{ee.excess_energy_kwh:.1f} kWh"))
        lines.append(_row("Basis", ee.basis))
        lines.append(_row("Provenance", ee.provenance))
    else:
        lines.append("Unavailable")
        lines.append(f"Reason: {ee.reason if ee else 'no excess-energy calculation was attempted.'}")
    lines.append("")

    if p.figures:
        lines.append(_section("RESEARCH OUTPUTS"))
        for key in sorted(p.figures):
            f = p.figures[key]
            mark = "\u2713" if f.generated else "\u2717"
            extra = "" if f.generated else f" (skipped — {f.reason})"
            lines.append(f"{mark} {f.filename}{extra}")
        lines.append("")

    if run.output_dir:
        lines.append(f"Output directory: {run.output_dir}")
    lines.append(LINE)

    return "\n".join(lines)
