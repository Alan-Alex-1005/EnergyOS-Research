"""
evaluation/provenance.py — Day 5: Dataset Provenance for Real-Data Evaluation

Two kinds of fields:
  - USER-SUPPLIED (cannot be inferred from a CSV): dataset_name,
    source_provider, source_reference (URL/citation). The runner refuses
    to proceed without these for a "real data" evaluation.
  - DERIVED from the actual computed run (never fabricated): filename,
    format, row count, selected series, role, unit, time range,
    preprocessing actions (from Day-1 CleaningReport + Day-2 prep
    warnings), and which context fields were genuinely present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetProvenance:
    dataset_name: str
    source_provider: Optional[str]
    source_reference: Optional[str]
    filename: str
    input_format: str
    row_count: int
    selected_series: Optional[str]
    measurement_role: Optional[str]
    unit: Optional[str]
    time_start: Optional[str]
    time_end: Optional[str]
    preprocessing_performed: list = field(default_factory=list)
    transformations_performed: list = field(default_factory=list)
    context_available: dict = field(default_factory=dict)


def build_provenance(
    run,
    dataset_name: str,
    source_provider: Optional[str],
    source_reference: Optional[str],
    transformations_performed: Optional[list] = None,
) -> DatasetProvenance:
    day1 = run.day1
    preprocessing = []
    for action, count in day1.cleaning_report.action_counts.items():
        if count:
            preprocessing.append(f"{action}: {count}")

    p = run.primary
    context_available = {}
    selected_series = None
    role = None
    unit = None
    time_start = time_end = None

    if p is not None:
        selected_series = p.series.series_id
        role = p.series.role
        unit = p.series.unit
        context_available = {
            f: getattr(p.series, f)
            for f in ("equipment_id", "device_id", "meter_id", "site", "building", "floor", "zone", "room")
            if getattr(p.series, f) is not None
        }
        if p.prepared is not None:
            preprocessing.extend(p.prepared.warnings)
            if len(p.prepared.timestamps):
                time_start = str(p.prepared.timestamps.iloc[0])
                time_end = str(p.prepared.timestamps.iloc[-1])

    return DatasetProvenance(
        dataset_name=dataset_name,
        source_provider=source_provider,
        source_reference=source_reference,
        filename=day1.load_result.source_filename,
        input_format=day1.profile.format_hint,
        row_count=day1.profile.row_count,
        selected_series=selected_series,
        measurement_role=role,
        unit=unit,
        time_start=time_start,
        time_end=time_end,
        preprocessing_performed=preprocessing,
        transformations_performed=transformations_performed or [],
        context_available=context_available,
    )
