"""
terminal.py — Day 1: Terminal Experience

Renders the Day-1 pipeline results as clean, readable CLI output.
No fake values are ever printed — every field comes directly from the
upstream modules' real results (spec section 14).
"""

from __future__ import annotations

from typing import Optional

from .canonical import CONTEXT_ROLES
from .capabilities import CapabilityReport
from .cleaning import CleaningReport
from .loader import LoadResult
from .long_format import LongFormatMappingResult
from .mapping import MappingResult
from .profiler import Profile
from .quality import QualityResult
from .schema import TimestampInfo

LINE = "=" * 60
SUB_LINE = "-" * 60


def _section(title: str) -> str:
    return f"{title}\n{SUB_LINE}"


def _row(label: str, value: str, width: int = 20) -> str:
    return f"{label:<{width}} {value}"


def render_report(
    load_result: LoadResult,
    profile: Profile,
    timestamp_info: TimestampInfo,
    mapping_result: MappingResult,
    cleaning_report: CleaningReport,
    quality_result: QualityResult,
    capability_report: CapabilityReport,
    long_format_mapping: Optional[LongFormatMappingResult] = None,
) -> str:
    lines = []
    lines.append(LINE)
    lines.append("        ENERGYOS RESEARCH DEMO v2 \u2014 DATA INTELLIGENCE")
    lines.append(LINE)
    lines.append("")

    # Input
    lines.append(_section("Input"))
    lines.append(_row("File", load_result.source_filename))
    lines.append(_row("Type", load_result.file_type))
    if load_result.selected_sheet:
        lines.append(_row("Sheet", load_result.selected_sheet))
    lines.append(_row("Rows", f"{profile.row_count:,}"))
    lines.append(_row("Columns", f"{profile.column_count:,}"))
    lines.append(_row("Format", profile.format_hint))
    lines.append("")

    # Time Series
    lines.append(_section("Time Series"))
    if timestamp_info.detected:
        lines.append(_row("Timestamp", timestamp_info.column))
        lines.append(_row("Confidence", f"{timestamp_info.confidence * 100:.0f}%"))
        lines.append(_row("Start", str(timestamp_info.start)))
        lines.append(_row("End", str(timestamp_info.end)))
        lines.append(_row("Sampling Interval", timestamp_info.sampling_interval or "n/a (irregular or too few points)"))
        lines.append(_row("Monotonic", str(timestamp_info.is_monotonic)))
        lines.append(_row("Duplicate Timestamps", str(timestamp_info.duplicate_timestamp_count)))
    else:
        lines.append("No timestamp column detected.")
    lines.append("")

    # Detected Roles
    lines.append(_section("Detected Roles"))
    for col, m in mapping_result.mapped.items():
        unit_str = f" ({m.unit})" if m.unit else ""
        lines.append(f"{col:<20} {m.role + unit_str:<18} {m.confidence * 100:.0f}%  [{m.source}]")
    for col in mapping_result.unresolved:
        lines.append(f"{col:<20} {'UNKNOWN':<18} unresolved")
    lines.append("")

    # Mapping
    lines.append(_section("Mapping"))
    lines.append(_row("Mapped", str(mapping_result.mapped_count)))
    lines.append(_row("Unresolved", str(mapping_result.unresolved_count)))
    lines.append("")

    # Long-Format Metric Mapping (only shown for LONG-format datasets)
    if long_format_mapping is not None and long_format_mapping.applicable:
        lines.append(_section("Long-Format Metric Mapping"))
        lines.append(_row("Metric Column", str(long_format_mapping.metric_column)))
        lines.append(_row("Value Column", str(long_format_mapping.value_column)))
        if long_format_mapping.unit_column:
            lines.append(_row("Unit Column", long_format_mapping.unit_column))
        if long_format_mapping.identifier_columns:
            lines.append(_row("Grouped By", ", ".join(long_format_mapping.identifier_columns)))
        lines.append("")
        for s in long_format_mapping.series:
            keys = " ".join(f"{k}={v}" for k, v in s.group_keys.items())
            label = f"{s.metric_value} {keys}".strip()
            if s.role == "UNKNOWN":
                lines.append(f"{label:<32} {'UNKNOWN':<12} unresolved  ({s.row_count} rows)")
            else:
                unit_str = f" ({s.unit})" if s.unit else ""
                lines.append(
                    f"{label:<32} {s.role + unit_str:<18} {s.confidence * 100:.0f}%  "
                    f"[{s.source}]  ({s.row_count} rows)"
                )
        lines.append("")

    # Cleaning
    lines.append(_section("Cleaning"))
    lines.append(_row("Rows Before", f"{cleaning_report.rows_before:,}"))
    lines.append(_row("Rows After", f"{cleaning_report.rows_after:,}"))
    for action, count in cleaning_report.action_counts.items():
        if count:
            lines.append(_row(action, str(count)))
    lines.append("")

    # Data Quality
    lines.append(_section("Data Quality"))
    lines.append(_row("Overall", f"{quality_result.overall:.1f} / 100"))
    lines.append(_row("Status", quality_result.status))
    for factor, value in quality_result.factors.items():
        label = factor.replace("_", " ").title()
        lines.append(_row(f"  {label}", f"{value:.0f}%"))
    lines.append("")

    # Capabilities
    lines.append(_section("Capabilities"))
    for cap in capability_report.capabilities:
        if cap.status == "AVAILABLE":
            lines.append(f"{cap.name:<20} AVAILABLE")
        else:
            lines.append(f"{cap.name:<20} {cap.status} \u2014 {cap.reason}")
    lines.append("")

    # Warnings
    all_warnings = list(load_result.warnings) + list(timestamp_info.warnings) + list(mapping_result.warnings)
    if long_format_mapping is not None:
        all_warnings += list(long_format_mapping.warnings)
    if all_warnings:
        lines.append(_section("Warnings"))
        for w in all_warnings:
            lines.append(f"\u2022 {w}")
        lines.append("")

    lines.append("Day 1 pipeline complete.")
    return "\n".join(lines)
