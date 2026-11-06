"""
pipeline.py — Day 1: Pipeline Orchestrator

Wires together: File Loader -> Profiler -> Timestamp Detection ->
Role Detection -> Mapping -> Cleaning -> Canonical Model -> Quality
Score -> Capability Report.

Used by both run_demo.py (CLI) and the test suite, so the CLI and tests
never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import capabilities as capabilities_mod
from . import cleaning as cleaning_mod
from . import canonical as canonical_mod
from . import loader as loader_mod
from . import long_format as long_format_mod
from . import mapping as mapping_mod
from . import profiler as profiler_mod
from . import quality as quality_mod
from . import role_detection as role_detection_mod
from . import schema as schema_mod


class PipelineError(Exception):
    """Raised for any pipeline failure. Message is CLI-safe."""


@dataclass
class PipelineResult:
    load_result: loader_mod.LoadResult
    profile: profiler_mod.Profile
    timestamp_info: schema_mod.TimestampInfo
    role_suggestions: list
    mapping_result: mapping_mod.MappingResult
    cleaned_df: pd.DataFrame
    cleaning_report: cleaning_mod.CleaningReport
    canonical_df: pd.DataFrame
    quality_result: quality_mod.QualityResult
    capability_report: capabilities_mod.CapabilityReport
    long_format_mapping: Optional[long_format_mod.LongFormatMappingResult] = None


def run_pipeline(
    path: str, sheet: Optional[str] = None, config_path: Optional[str] = None
) -> PipelineResult:
    try:
        load_result = loader_mod.load_file(path, sheet=sheet)
    except loader_mod.LoaderError as e:
        raise PipelineError(str(e)) from e

    df = load_result.dataframe

    config = None
    if config_path:
        try:
            config = mapping_mod.load_config(config_path)
        except mapping_mod.MappingConfigError as e:
            raise PipelineError(str(e)) from e

    profile = profiler_mod.profile_dataframe(df)
    timestamp_info = schema_mod.detect_timestamp(df)

    non_ts_df_columns = [c for c in df.columns if c != timestamp_info.column]
    non_ts_profiles = {c: profile.columns[c] for c in non_ts_df_columns}
    role_suggestions = role_detection_mod.detect_roles(non_ts_profiles)

    try:
        mapping_result = mapping_mod.build_mapping(
            all_columns=list(df.columns),
            timestamp_column=timestamp_info.column,
            role_suggestions=role_suggestions,
            config=config,
        )
    except mapping_mod.MappingConfigError as e:
        raise PipelineError(str(e)) from e

    # If the config designated a different timestamp column than the one
    # auto-detected, re-run timestamp parsing against that column so
    # sampling interval / start / end reflect the CONFIGURED column.
    if mapping_result.timestamp_column and mapping_result.timestamp_column != timestamp_info.column:
        timestamp_info = _detect_specific_timestamp(df, mapping_result.timestamp_column, timestamp_info)

    cleaned_df, cleaning_report = cleaning_mod.clean_dataset(df, timestamp_info, mapping_result)

    canonical_df = canonical_mod.build_canonical(cleaned_df, mapping_result, load_result.source_filename)

    try:
        long_format_result = long_format_mod.build_long_format_mapping(
            cleaned_df, profile, mapping_result, config=config
        )
    except mapping_mod.MappingConfigError as e:
        raise PipelineError(str(e)) from e

    # LONG-format datasets resolve their measured series at the metric-value
    # level (see long_format.py) rather than the physical-column level —
    # merge those canonical rows in so forecasting/anomaly/etc. downstream
    # consumers see one unified canonical dataset regardless of source shape.
    if long_format_result.applicable and not long_format_result.canonical_df.empty:
        canonical_df = (
            long_format_result.canonical_df
            if canonical_df.empty
            else pd.concat([canonical_df, long_format_result.canonical_df], ignore_index=True)
        )

    quality_result = quality_mod.compute_quality(
        profile, timestamp_info, mapping_result, cleaning_report, long_format_mapping=long_format_result
    )

    extra_roles = long_format_result.resolved_roles() if long_format_result.applicable else set()
    capability_report = capabilities_mod.assess_capabilities(
        mapping_result, timestamp_info, cleaning_report.rows_after, extra_roles=extra_roles
    )

    return PipelineResult(
        load_result=load_result,
        profile=profile,
        timestamp_info=timestamp_info,
        role_suggestions=role_suggestions,
        mapping_result=mapping_result,
        cleaned_df=cleaned_df,
        cleaning_report=cleaning_report,
        canonical_df=canonical_df,
        quality_result=quality_result,
        capability_report=capability_report,
        long_format_mapping=long_format_result,
    )


def _detect_specific_timestamp(df: pd.DataFrame, column: str, fallback: schema_mod.TimestampInfo) -> schema_mod.TimestampInfo:
    """Re-run timestamp parsing pinned to a config-designated column."""
    if column not in df.columns:
        return fallback
    single_col_df = df[[column]]
    result = schema_mod.detect_timestamp(single_col_df)
    if result.detected:
        result.confidence = 1.0
        result.evidence = ["explicitly configured as timestamp_column"] + result.evidence
        return result
    return fallback
