import pandas as pd
import pytest

from src.analysis_series import AnalysisSeries
from src.timeseries import (
    DEFAULT_TRAIN_RATIO,
    TimeSeriesPrepError,
    chronological_train_test_split,
    prepare_series,
)


def _make_series(timestamps, values, series_id="POWER:test", role="POWER", unit="kW"):
    return AnalysisSeries(
        series_id=series_id,
        role=role,
        unit=unit,
        timestamps=pd.Series(pd.to_datetime(timestamps)),
        values=pd.Series(values, dtype=float),
        used_normalized=True,
        sample_count=len(values),
    )


def test_prepare_series_regular_hourly_data():
    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    series = _make_series(ts, list(range(48)))
    prepared = prepare_series(series)
    assert prepared.is_regular is True
    assert prepared.sampling_interval_label == "1 hours"
    assert prepared.gap_count == 0
    assert prepared.prepared_sample_count == 48
    assert prepared.sufficient is True


def test_prepare_series_detects_gap():
    ts = list(pd.date_range("2026-01-01", periods=20, freq="h"))
    # remove a chunk in the middle to create one large gap
    ts = ts[:10] + ts[15:]
    series = _make_series(ts, list(range(len(ts))))
    prepared = prepare_series(series)
    assert prepared.gap_count >= 1
    assert prepared.largest_gap_seconds > 3600


def test_prepare_series_detects_irregular_sampling():
    # wildly inconsistent gaps between points
    ts = ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 03:00", "2026-01-01 03:02", "2026-01-02 12:00"]
    series = _make_series(ts, [1, 2, 3, 4, 5])
    prepared = prepare_series(series)
    assert prepared.is_regular is False


def test_prepare_series_reports_missing_values_without_filling():
    ts = pd.date_range("2026-01-01", periods=30, freq="h")
    values = [float(i) for i in range(30)]
    values[5] = None
    values[10] = None
    series = _make_series(ts, values)
    prepared = prepare_series(series)
    assert prepared.missing_value_count == 2
    assert prepared.prepared_sample_count == 28  # dropped for modeling, never filled
    assert prepared.original_sample_count == 30


def test_prepare_series_deduplicates_true_duplicate_timestamps():
    ts = list(pd.date_range("2026-01-01", periods=10, freq="h"))
    ts_with_dupe = ts + [ts[3]]
    values = list(range(10)) + [999]
    series = _make_series(ts_with_dupe, values)
    prepared = prepare_series(series)
    assert prepared.duplicate_timestamp_count == 1
    assert prepared.prepared_sample_count == 10
    # kept FIRST occurrence, not the duplicate's value
    assert 999 not in prepared.values.values


def test_prepare_series_insufficient_data_flagged():
    ts = pd.date_range("2026-01-01", periods=5, freq="h")
    series = _make_series(ts, [1, 2, 3, 4, 5])
    prepared = prepare_series(series, min_samples=20)
    assert prepared.sufficient is False
    assert any("minimum required" in w for w in prepared.warnings)


def test_chronological_split_default_80_20():
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    series = _make_series(ts, list(range(100)))
    prepared = prepare_series(series)
    split = chronological_train_test_split(prepared)
    assert split.train_ratio == DEFAULT_TRAIN_RATIO
    assert split.train_samples == 80
    assert split.test_samples == 20
    # train is strictly earlier than test — no leakage
    assert split.train_timestamps.iloc[-1] < split.test_timestamps.iloc[0]
    assert list(split.train_values) == list(range(80))
    assert list(split.test_values) == list(range(80, 100))


def test_chronological_split_never_shuffles():
    """
    Part D (non-negotiable): the split must be a pure chronological cut,
    not a random sample. Verify train values are exactly the first N in
    original order and test values are exactly the last M.
    """
    ts = pd.date_range("2026-01-01", periods=50, freq="h")
    values = list(range(50))
    series = _make_series(ts, values)
    prepared = prepare_series(series)
    split = chronological_train_test_split(prepared, train_ratio=0.6)
    assert list(split.train_values) == list(range(30))
    assert list(split.test_values) == list(range(30, 50))
    assert split.train_samples + split.test_samples == 50


def test_chronological_split_configurable_ratio():
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    series = _make_series(ts, list(range(100)))
    prepared = prepare_series(series)
    split = chronological_train_test_split(prepared, train_ratio=0.9)
    assert split.train_samples == 90
    assert split.test_samples == 10


def test_chronological_split_rejects_invalid_ratio():
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    series = _make_series(ts, list(range(20)))
    prepared = prepare_series(series)
    with pytest.raises(TimeSeriesPrepError, match="train_ratio"):
        chronological_train_test_split(prepared, train_ratio=1.5)
    with pytest.raises(TimeSeriesPrepError, match="train_ratio"):
        chronological_train_test_split(prepared, train_ratio=0.0)


def test_chronological_split_rejects_too_few_samples():
    ts = pd.date_range("2026-01-01", periods=1, freq="h")
    series = _make_series(ts, [1.0])
    prepared = prepare_series(series, min_samples=1)
    with pytest.raises(TimeSeriesPrepError, match="too few"):
        chronological_train_test_split(prepared)


def test_split_preserves_timestamps_exactly():
    ts = pd.date_range("2026-01-01", periods=40, freq="30min")
    series = _make_series(ts, list(range(40)))
    prepared = prepare_series(series)
    split = chronological_train_test_split(prepared, train_ratio=0.75)
    assert split.train_start == str(ts[0])
    assert split.test_end == str(ts[39])


def test_multiple_series_prepared_independently():
    """LONG-format style: two series with different lengths/intervals must not interfere."""
    ts_a = pd.date_range("2026-01-01", periods=30, freq="h")
    ts_b = pd.date_range("2026-02-01", periods=10, freq="30min")
    series_a = _make_series(ts_a, list(range(30)), series_id="POWER:meter=A")
    series_b = _make_series(ts_b, list(range(10)), series_id="POWER:meter=B")
    prepared_a = prepare_series(series_a, min_samples=20)
    prepared_b = prepare_series(series_b, min_samples=20)
    assert prepared_a.sufficient is True
    assert prepared_b.sufficient is False
    assert prepared_a.sampling_interval_label == "1 hours"
    assert prepared_b.sampling_interval_label == "30 minutes"
