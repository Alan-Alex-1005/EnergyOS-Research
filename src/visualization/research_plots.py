"""
visualization/research_plots.py — Day 2 Part P/Q: Research PNG Figures

Matplotlib only (no seaborn), scientific/research style, not dashboards.
Every figure function returns a FigureResult — if the data can't
honestly support a figure, it's SKIPPED with a reason, never rendered
as a misleading empty/fabricated chart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless — no display server in this environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureResult:
    name: str
    filename: str
    generated: bool
    path: Optional[str] = None
    reason: Optional[str] = None


def _skip(name: str, filename: str, reason: str) -> FigureResult:
    return FigureResult(name=name, filename=filename, generated=False, reason=reason)


def _save(fig, output_dir: str, filename: str, name: str) -> FigureResult:
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return _skip(name, filename, "figure failed to write a non-empty file")
    return FigureResult(name=name, filename=filename, generated=True, path=path)


def plot_dataset_profile(output_dir: str, profile_summary: dict) -> FigureResult:
    """01_dataset_profile.png — quantitative summary, not a decorative dashboard."""
    filename = "01_dataset_profile.png"
    name = "Dataset Profile"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    lines = [
        f"File: {profile_summary.get('file', 'n/a')}",
        f"Rows: {profile_summary.get('rows', 'n/a')}",
        f"Mapped series: {profile_summary.get('mapped_series', 'n/a')}",
        f"Time range: {profile_summary.get('time_start', 'n/a')} to {profile_summary.get('time_end', 'n/a')}",
        f"Sampling interval: {profile_summary.get('sampling_interval', 'n/a')}",
        f"Missing-data ratio: {profile_summary.get('missing_ratio', 'n/a')}",
        f"Data quality score: {profile_summary.get('quality_score', 'n/a')}",
    ]
    ax.text(0.02, 0.95, "Dataset Profile", fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(
        0.02, 0.80, "\n".join(lines), fontsize=11, va="top", family="monospace", transform=ax.transAxes
    )
    return _save(fig, output_dir, filename, name)


def plot_energy_timeseries(output_dir: str, timestamps: pd.Series, values: pd.Series, series_label: str, unit: str) -> FigureResult:
    """02_energy_timeseries.png — actual energy/power over time, clearly labeled."""
    filename = "02_energy_timeseries.png"
    name = "Energy Time Series"
    if len(values.dropna()) == 0:
        return _skip(name, filename, "no non-null observations to plot.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(timestamps, values, linewidth=0.8, color="tab:blue")
    ax.set_title(f"{series_label} over time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"{series_label} ({unit})" if unit else series_label)
    fig.autofmt_xdate()
    return _save(fig, output_dir, filename, name)


def plot_daily_load_profile(
    output_dir: str, timestamps: pd.Series, values: pd.Series, series_label: str, unit: str, is_regular: bool
) -> FigureResult:
    """03_daily_load_profile.png — mean by time-of-day. Only when meaningful (regular sampling, enough days)."""
    filename = "03_daily_load_profile.png"
    name = "Daily Load Profile"
    if not is_regular:
        return _skip(name, filename, "sampling is irregular; a time-of-day aggregate would not be meaningful.")

    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "value": values}).dropna()
    if df.empty:
        return _skip(name, filename, "no non-null observations to aggregate.")

    n_days = df["ts"].dt.date.nunique()
    if n_days < 2:
        return _skip(name, filename, f"only {n_days} distinct day(s) of data; need at least 2 for a daily profile.")

    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    profile = df.groupby("minute_of_day")["value"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = profile.index / 60.0
    ax.plot(hours, profile.values, color="tab:orange")
    ax.set_title(f"Daily Load Profile — mean {series_label} by time of day ({n_days} days)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(f"Mean {series_label} ({unit})" if unit else f"Mean {series_label}")
    ax.set_xlim(0, 24)
    return _save(fig, output_dir, filename, name)


def plot_forecast_vs_actual(
    output_dir: str, test_timestamps, actual, predicted, model_name: str, series_label: str, unit: str
) -> FigureResult:
    """04_forecast_vs_actual.png — one of the most important figures."""
    filename = "04_forecast_vs_actual.png"
    name = "Forecast vs Actual"
    if actual is None or predicted is None or len(actual) == 0:
        return _skip(name, filename, "no forecast was produced for this series.")

    ts = pd.to_datetime(pd.Series(test_timestamps))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts, actual, label="Actual", color="tab:blue", linewidth=1.2)
    ax.plot(ts, predicted, label="Predicted", color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_title(f"Forecast vs Actual — model: {model_name}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"{series_label} ({unit})" if unit else series_label)
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, output_dir, filename, name)


def plot_forecast_residuals(output_dir: str, test_timestamps, residuals, unit: str) -> FigureResult:
    """05_forecast_residuals.png — e_t = actual - expected, with a zero reference."""
    filename = "05_forecast_residuals.png"
    name = "Forecast Residuals"
    if residuals is None or len(residuals) == 0:
        return _skip(name, filename, "no residuals available (no forecast was produced).")

    ts = pd.to_datetime(pd.Series(test_timestamps))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, residuals, color="tab:purple", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("Forecast Residuals (actual - expected)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"Residual ({unit})" if unit else "Residual")
    fig.autofmt_xdate()
    return _save(fig, output_dir, filename, name)


def plot_anomaly_score_timeline(output_dir: str, observations, base_threshold: float) -> FigureResult:
    """06_anomaly_score_timeline.png — statistical score over time with threshold reference."""
    filename = "06_anomaly_score_timeline.png"
    name = "Anomaly Score Timeline"
    if not observations:
        return _skip(name, filename, "no anomaly observations available.")

    ts = pd.to_datetime(pd.Series([o.timestamp for o in observations]))
    z = np.array([o.z_score for o in observations])
    is_anom = np.array([o.is_anomalous for o in observations])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts, np.abs(z), color="tab:gray", linewidth=0.9, label="|robust z-score|")
    if is_anom.any():
        ax.scatter(ts[is_anom], np.abs(z)[is_anom], color="tab:red", s=18, zorder=3, label="Anomalous")
    ax.axhline(base_threshold, color="tab:red", linestyle="--", linewidth=1.0, label=f"threshold ({base_threshold:g})")
    ax.set_title("Anomaly Statistical Score Timeline")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("|robust z-score|")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, output_dir, filename, name)


def plot_anomaly_distribution(output_dir: str, observations) -> FigureResult:
    """07_anomaly_distribution.png — residual/score distribution, explains why points were flagged."""
    filename = "07_anomaly_distribution.png"
    name = "Anomaly Distribution"
    if not observations:
        return _skip(name, filename, "no anomaly observations available.")

    residuals = np.array([o.residual for o in observations])
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 2:
        return _skip(name, filename, "not enough residual observations for a distribution plot.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=min(30, max(5, len(residuals) // 3)), color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Residual Distribution (test period)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    return _save(fig, output_dir, filename, name)


def plot_excess_energy_analysis(
    output_dir: str, test_timestamps, actual, predicted, excess_energy_kwh: Optional[float], unit: str
) -> FigureResult:
    """08_excess_energy_analysis.png — actual vs expected with excess region shaded."""
    filename = "08_excess_energy_analysis.png"
    name = "Excess Energy Analysis"
    if excess_energy_kwh is None or actual is None or len(actual) == 0:
        return _skip(name, filename, "excess energy was not available for this series (see results.json for reason).")

    ts = pd.to_datetime(pd.Series(test_timestamps))
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts, actual, label="Actual", color="tab:blue", linewidth=1.1)
    ax.plot(ts, predicted, label="Expected (forecast baseline)", color="tab:red", linestyle="--", linewidth=1.1)
    ax.fill_between(
        ts, predicted, actual, where=(actual > predicted), color="tab:orange", alpha=0.4, label="Excess"
    )
    ax.set_title(f"Excess Energy Analysis — estimated excess: {excess_energy_kwh:.1f} kWh")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"Power ({unit})" if unit else "Power")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, output_dir, filename, name)


# ---------------------------------------------------------------------------
# Day 4: Multi-agent collaboration figures
# ---------------------------------------------------------------------------


def plot_agent_status(output_dir: str, agents: dict) -> FigureResult:
    """09_agent_status.png — which agents ran and their capability-aware status."""
    filename = "09_agent_status.png"
    name = "Agent Status"
    if not agents:
        return _skip(name, filename, "no agents were run.")

    status_colors = {"DONE": "tab:green", "LIMITED": "tab:orange", "UNAVAILABLE": "tab:gray", "INSUFFICIENT_DATA": "tab:red", "ERROR": "tab:red"}
    names = list(agents.keys())
    statuses = [agents[n].status for n in names]
    colors = [status_colors.get(s, "tab:gray") for s in statuses]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, [1] * len(names), color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    for i, s in enumerate(statuses):
        ax.text(0.5, i, s, ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Specialized Agent Status")
    return _save(fig, output_dir, filename, name)


def plot_agent_agreement(output_dir: str, agreement) -> FigureResult:
    """10_agent_agreement.png — agreement score with participant/status context."""
    filename = "10_agent_agreement.png"
    name = "Agent Agreement"
    if agreement is None or agreement.score is None:
        reason = f"agreement not computable ({agreement.status})" if agreement else "no agreement result available."
        return _skip(name, filename, reason)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Agent Agreement"], [agreement.score], color="tab:blue", width=0.5)
    ax.set_ylim(0, 100)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Agreement (%)")
    ax.set_title(f"Agent Agreement — {agreement.status}\nParticipants: {', '.join(agreement.participants)}")
    ax.text(0, agreement.score + 3, f"{agreement.score:.1f}%", ha="center", fontweight="bold")
    return _save(fig, output_dir, filename, name)


def plot_evidence_completeness(output_dir: str, completeness) -> FigureResult:
    """11_evidence_completeness.png — per-category credit bar chart."""
    filename = "11_evidence_completeness.png"
    name = "Evidence Completeness"
    if completeness is None or not completeness.category_credit:
        return _skip(name, filename, "no completeness result available.")

    categories = list(completeness.category_credit.keys())
    credits = [completeness.category_credit[c] * 100 for c in categories]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(categories, credits, color="teal")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Credit (%)")
    ax.set_title(f"Evidence Completeness — overall {completeness.score:.1f}%")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, output_dir, filename, name)


def plot_incident_summary(output_dir: str, incidents: list) -> FigureResult:
    """12_incident_summary.png — evidence-type coverage per correlated incident."""
    filename = "12_incident_summary.png"
    name = "Incident Summary"
    if not incidents:
        return _skip(name, filename, "no incidents were correlated (no anomalies detected — normal operation).")

    all_types = sorted({t for inc in incidents for t in inc.evidence_types_present})
    fig, ax = plt.subplots(figsize=(9, max(3, len(incidents) * 0.8 + 1)))
    for i, inc in enumerate(incidents):
        for j, t in enumerate(all_types):
            present = t in inc.evidence_types_present
            ax.scatter(j, i, s=200, color="tab:green" if present else "lightgray", marker="s")
    ax.set_yticks(range(len(incidents)))
    ax.set_yticklabels([inc.incident_id for inc in incidents])
    ax.set_xticks(range(len(all_types)))
    ax.set_xticklabels(all_types, rotation=30, ha="right")
    ax.set_title("Incident Evidence Coverage")
    return _save(fig, output_dir, filename, name)


def plot_decision_confidence(output_dir: str, confidence) -> FigureResult:
    """13_decision_confidence.png — the weighted components behind the final confidence score."""
    filename = "13_decision_confidence.png"
    name = "Decision Confidence"
    if confidence is None or not confidence.components:
        return _skip(name, filename, "no confidence result available.")

    labels = list(confidence.components.keys())
    values = [confidence.components[k] for k in labels]
    weights = [confidence.weights_used.get(k, 0) * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, values, width=0.4, label="Component score", color="tab:blue")
    ax.bar(x + 0.2, weights, width=0.4, label="Weight used (%)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", " ").title() for l in labels])
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title(f"Decision-Support Confidence — {confidence.score:.1f}/100 (not a statistical probability)")
    return _save(fig, output_dir, filename, name)


# ---------------------------------------------------------------------------
# Day 5: Research evaluation figures
# ---------------------------------------------------------------------------


def plot_framework_comparison(output_dir: str, comparison_rows: list) -> FigureResult:
    """14_framework_comparison.png — Level 1/2/3 dimension-support counts, side by side."""
    filename = "14_framework_comparison.png"
    name = "Framework Comparison"
    dim_rows = [r for r in comparison_rows if r["applicable_scenarios"] > 0 and not r["dimension"].startswith("MEAN_")]
    if not dim_rows:
        return _skip(name, filename, "no applicable dimensions to compare.")

    labels = [r["dimension"].replace("_", " ") for r in dim_rows]
    l1 = [r["level1_isolated_analytics"] for r in dim_rows]
    l2 = [r["level2_independent_agents"] for r in dim_rows]
    l3 = [r["level3_collaborative_framework"] for r in dim_rows]

    fig, ax = plt.subplots(figsize=(10, max(5, len(dim_rows) * 0.35)))
    y = np.arange(len(labels))
    h = 0.25
    ax.barh(y - h, l1, height=h, label="Level 1: Isolated Analytics", color="tab:gray")
    ax.barh(y, l2, height=h, label="Level 2: Independent Agents", color="tab:blue")
    ax.barh(y + h, l3, height=h, label="Level 3: Collaborative Framework", color="tab:green")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Scenarios supporting this dimension (of those where applicable)")
    ax.set_title("Framework Comparison — Level 1 vs 2 vs 3")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, output_dir, filename, name)


def plot_explanation_coverage(output_dir: str, scenario_results: list) -> FigureResult:
    """15_explanation_coverage.png — Explanation Coverage % across the three levels, per scenario."""
    filename = "15_explanation_coverage.png"
    name = "Explanation Coverage"
    if not scenario_results:
        return _skip(name, filename, "no scenario results available.")

    labels = [sr.scenario_id.replace("scenario_", "S") for sr in scenario_results]
    l1 = [sr.level1_coverage.percentage for sr in scenario_results]
    l2 = [sr.level2_coverage.percentage for sr in scenario_results]
    l3 = [sr.level3_coverage.percentage for sr in scenario_results]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, l1, width=w, label="Level 1", color="tab:gray")
    ax.bar(x, l2, width=w, label="Level 2", color="tab:blue")
    ax.bar(x + w, l3, width=w, label="Level 3", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Explanation Coverage (%)")
    ax.set_title("Explanation Coverage by Scenario and Level")
    ax.legend()
    return _save(fig, output_dir, filename, name)


def plot_evidence_diversity(output_dir: str, scenario_results: list) -> FigureResult:
    """16_evidence_diversity.png — distinct evidence-category count across the three levels, per scenario."""
    filename = "16_evidence_diversity.png"
    name = "Evidence Diversity"
    if not scenario_results:
        return _skip(name, filename, "no scenario results available.")

    labels = [sr.scenario_id.replace("scenario_", "S") for sr in scenario_results]
    l1 = [sr.level1_diversity.count for sr in scenario_results]
    l2 = [sr.level2_diversity.count for sr in scenario_results]
    l3 = [sr.level3_diversity.count for sr in scenario_results]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, l1, width=w, label="Level 1", color="tab:gray")
    ax.bar(x, l2, width=w, label="Level 2", color="tab:blue")
    ax.bar(x + w, l3, width=w, label="Level 3", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Distinct evidence categories")
    ax.set_title("Evidence Diversity by Scenario and Level")
    ax.legend()
    return _save(fig, output_dir, filename, name)


def plot_scenario_robustness(output_dir: str, robustness_rows: list) -> FigureResult:
    """17_scenario_robustness.png — actual per-scenario agent statuses as a color grid."""
    filename = "17_scenario_robustness.png"
    name = "Scenario Robustness"
    if not robustness_rows:
        return _skip(name, filename, "no robustness data available.")

    status_colors = {"DONE": 2, "LIMITED": 1, "UNAVAILABLE": 0, "INSUFFICIENT_DATA": 0, "ERROR": 0}
    cols = ["forecast", "anomaly", "equipment", "cost", "carbon", "collaboration"]
    grid = np.array([[status_colors.get(row[c], 0) for c in cols] for row in robustness_rows])

    fig, ax = plt.subplots(figsize=(8, max(3, len(robustness_rows) * 0.6)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(robustness_rows)))
    ax.set_yticklabels([row["scenario"].replace("scenario_", "S") for row in robustness_rows])
    for i, row in enumerate(robustness_rows):
        for j, c in enumerate(cols):
            ax.text(j, i, row[c][:4], ha="center", va="center", fontsize=7)
    ax.set_title("Scenario Robustness — actual runtime statuses")
    return _save(fig, output_dir, filename, name)


def plot_research_summary(output_dir: str, summary_lines: list) -> FigureResult:
    """18_research_summary.png — clean final research summary figure."""
    filename = "18_research_summary.png"
    name = "Research Summary"
    if not summary_lines:
        return _skip(name, filename, "no summary content available.")

    fig, ax = plt.subplots(figsize=(9, min(10, 1 + len(summary_lines) * 0.4)))
    ax.axis("off")
    ax.text(0.02, 0.97, "Research Summary", fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.88, "\n".join(summary_lines), fontsize=10, va="top", family="monospace", transform=ax.transAxes)
    return _save(fig, output_dir, filename, name)
