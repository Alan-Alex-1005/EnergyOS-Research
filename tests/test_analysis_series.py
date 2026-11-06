import pandas as pd

from src.analysis_series import (
    AnalysisSeries,
    MIN_SAMPLES_FOR_FORECAST,
    assess_series_sufficiency,
    build_analysis_series,
    select_primary_series,
)
from src.pipeline import run_pipeline
from tests.conftest import fixture_path


def test_wide_dataset_produces_one_series_per_measure_column():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    series_list = build_analysis_series(result.canonical_df)
    roles = {s.role for s in series_list}
    assert roles == {"POWER", "ENERGY"}
    for s in series_list:
        assert s.sample_count == 200
        assert s.meter_id is None  # never fabricated — wide_energy.csv has no meter column


def test_long_dataset_produces_one_series_per_meter_per_metric():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    series_list = build_analysis_series(result.canonical_df)
    # 2 metrics (power_kw, energy_kwh) x 2 meters (M1, M2) = 4 series
    assert len(series_list) == 4
    meters = {s.meter_id for s in series_list}
    assert meters == {"M1", "M2"}
    roles = {s.role for s in series_list}
    assert roles == {"POWER", "ENERGY"}
    for s in series_list:
        assert s.sample_count == 50


def test_series_never_mixes_units_or_meters():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    series_list = build_analysis_series(result.canonical_df)
    for s in series_list:
        assert s.meter_id in ("M1", "M2")
        assert s.unit in ("kW", "kWh")


def test_sufficiency_available_for_well_populated_series():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    series_list = build_analysis_series(result.canonical_df)
    power_series = next(s for s in series_list if s.role == "POWER")
    sufficiency = assess_series_sufficiency(power_series)
    assert sufficiency.status == "AVAILABLE"
    assert sufficiency.sample_count == 200


def test_sufficiency_insufficient_for_short_series():
    short_series = AnalysisSeries(
        series_id="POWER:test",
        role="POWER",
        unit="kW",
        timestamps=pd.Series(pd.date_range("2026-01-01", periods=5, freq="h")),
        values=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        used_normalized=True,
        sample_count=5,
    )
    sufficiency = assess_series_sufficiency(short_series)
    assert sufficiency.status == "INSUFFICIENT_DATA"
    assert sufficiency.sample_count == 5
    assert str(MIN_SAMPLES_FOR_FORECAST) in sufficiency.reason


def test_sufficiency_ignores_nan_padding():
    """A series with lots of rows but mostly-missing values still isn't sufficient."""
    values = [1.0, 2.0, 3.0] + [None] * 30
    series = AnalysisSeries(
        series_id="POWER:sparse",
        role="POWER",
        unit="kW",
        timestamps=pd.Series(pd.date_range("2026-01-01", periods=33, freq="h")),
        values=pd.Series(values),
        used_normalized=True,
        sample_count=33,
    )
    sufficiency = assess_series_sufficiency(series)
    assert sufficiency.status == "INSUFFICIENT_DATA"
    assert sufficiency.sample_count == 3


def test_per_series_sufficiency_independent_across_meters():
    """
    Part A3's exact example: many rows overall doesn't mean every series
    has enough history — sufficiency must be judged per series, using a
    threshold that varies the outcome independently of total row count.
    """
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    series_list = build_analysis_series(result.canonical_df)
    assert len(series_list) == 4  # sanity: plenty of total rows across series
    for s in series_list:
        assert assess_series_sufficiency(s, min_samples=60).status == "INSUFFICIENT_DATA"
    for s in series_list:
        assert assess_series_sufficiency(s, min_samples=10).status == "AVAILABLE"


def test_select_primary_series_prefers_power_over_energy():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    series_list = build_analysis_series(result.canonical_df)
    chosen, reason = select_primary_series(series_list)
    assert chosen.role == "POWER"
    assert "role preference" in reason


def test_select_primary_series_explicit_request():
    result = run_pipeline(fixture_path("long_format_meter.csv"))
    series_list = build_analysis_series(result.canonical_df)
    target = series_list[0].series_id
    chosen, reason = select_primary_series(series_list, requested_series_id=target)
    assert chosen.series_id == target
    assert "explicitly requested" in reason


def test_select_primary_series_unknown_request_reports_available_ids():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    series_list = build_analysis_series(result.canonical_df)
    chosen, reason = select_primary_series(series_list, requested_series_id="NOT_A_REAL_SERIES")
    assert chosen is None
    assert "not found" in reason
    assert "Available series" in reason


def test_select_primary_series_empty_list():
    chosen, reason = select_primary_series([])
    assert chosen is None
    assert "no POWER/ENERGY/DEMAND" in reason


def test_no_energy_data_produces_no_series():
    result = run_pipeline(fixture_path("no_energy_data.csv"))
    series_list = build_analysis_series(result.canonical_df)
    assert series_list == []


def test_non_null_helper_drops_missing_observations():
    result = run_pipeline(fixture_path("missing_values.csv"))
    series_list = build_analysis_series(result.canonical_df)
    power_series = next(s for s in series_list if s.role == "POWER")
    assert power_series.values.isna().sum() > 0
    ts, vals = power_series.non_null()
    assert vals.isna().sum() == 0
    assert len(ts) == len(vals) == power_series.values.notna().sum()
