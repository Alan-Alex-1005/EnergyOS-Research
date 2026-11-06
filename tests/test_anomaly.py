import numpy as np
import pandas as pd
import pytest

from src.anomaly.engine import run_anomaly_engine
from src.anomaly.events import AnomalyObservation, group_anomaly_events
from src.anomaly.robust import (
    DEFAULT_BASE_THRESHOLD,
    classify_severity,
    fit_robust_stats,
    robust_z_scores,
    severity_cutoffs,
)
from src.analysis_series import AnalysisSeries
from src.forecasting.engine import run_forecast_engine
from src.timeseries import chronological_train_validation_test_split, prepare_series


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def test_fit_robust_stats_known_example():
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is an outlier
    stats = fit_robust_stats(residuals)
    assert stats.scale_method == "MAD"
    assert stats.median == 3.5
    # abs devs from 3.5: 2.5,1.5,0.5,0.5,1.5,96.5 -> median = 1.5
    assert abs(stats.mad - 1.5) < 1e-9


def test_robust_z_scores_known_example():
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = fit_robust_stats(residuals)  # median=3, mad=median(|.-3|)=median([2,1,0,1,2])=1
    z = robust_z_scores(np.array([3.0, 5.0]), stats)
    assert abs(z[0] - 0.0) < 1e-9
    assert abs(z[1] - 0.6745 * 2 / 1) < 1e-9


def test_fit_robust_stats_mad_zero_falls_back_to_std():
    # many identical residuals, one outlier -> MAD is 0
    residuals = np.array([5.0] * 10 + [50.0])
    stats = fit_robust_stats(residuals)
    assert stats.scale_method == "MAD=0 fallback to std-based scale"
    assert stats.mad > 0
    assert stats.note is not None


def test_fit_robust_stats_fully_degenerate():
    residuals = np.array([5.0] * 10)  # zero spread entirely
    stats = fit_robust_stats(residuals)
    assert stats.mad == 0.0
    assert "degenerate" in stats.scale_method
    z = robust_z_scores(np.array([5.0, 6.0]), stats)
    assert z[0] == 0.0
    assert abs(z[1]) == pytest.approx(10.0)  # documented sentinel, not inf/NaN
    assert np.isfinite(z).all()


def test_fit_robust_stats_empty_input():
    stats = fit_robust_stats(np.array([]))
    assert stats.mad == 0.0
    assert stats.scale_method == "undefined"


def test_severity_cutoffs_scale_with_base_threshold():
    default_cutoffs = severity_cutoffs(3.0)
    scaled_cutoffs = severity_cutoffs(6.0)
    for label in default_cutoffs:
        assert abs(scaled_cutoffs[label] - default_cutoffs[label] * 2) < 1e-9


def test_severity_cutoffs_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        severity_cutoffs(0)
    with pytest.raises(ValueError):
        severity_cutoffs(-1)


def test_classify_severity_bands():
    assert classify_severity(0.5) == "NORMAL"
    assert classify_severity(3.0) == "LOW"
    assert classify_severity(4.0) == "MEDIUM"
    assert classify_severity(5.5) == "HIGH"
    assert classify_severity(8.0) == "CRITICAL"


def test_classify_severity_configurable_threshold():
    # with a stricter (lower) base threshold, the same z becomes more severe
    strict = classify_severity(3.5, base_threshold=1.0)
    loose = classify_severity(3.5, base_threshold=DEFAULT_BASE_THRESHOLD)
    assert strict != "NORMAL"
    assert loose == "LOW"


# ---------------------------------------------------------------------------
# Anomaly events
# ---------------------------------------------------------------------------

def _obs(index, ts, residual, z, severity):
    return AnomalyObservation(
        index=index, timestamp=ts, residual=residual, z_score=z, severity=severity, is_anomalous=severity != "NORMAL"
    )


def test_isolated_spike_forms_single_event():
    obs = [
        _obs(0, "2026-01-01 00:00:00", 0.1, 0.2, "NORMAL"),
        _obs(1, "2026-01-01 01:00:00", 10.0, 5.0, "HIGH"),
        _obs(2, "2026-01-01 02:00:00", 0.1, 0.3, "NORMAL"),
    ]
    events = group_anomaly_events(obs, series_id="POWER:test")
    assert len(events) == 1
    assert events[0].observation_count == 1
    assert events[0].persistence_class == "ISOLATED_SPIKE"
    assert events[0].direction == "ABOVE_EXPECTED"


def test_consecutive_anomalies_grouped_into_one_sustained_event():
    obs = [_obs(0, "2026-01-01 00:00:00", 0.1, 0.1, "NORMAL")]
    for i in range(1, 6):
        obs.append(_obs(i, f"2026-01-01 0{i}:00:00", 8.0, 4.5, "HIGH"))
    obs.append(_obs(6, "2026-01-01 06:00:00", 0.1, 0.1, "NORMAL"))
    events = group_anomaly_events(obs, series_id="POWER:test")
    assert len(events) == 1
    assert events[0].observation_count == 5
    assert events[0].persistence_class == "SUSTAINED"


def test_separated_anomalies_create_separate_events():
    obs = [
        _obs(0, "2026-01-01 00:00:00", 8.0, 4.0, "MEDIUM"),
        _obs(1, "2026-01-01 01:00:00", 0.1, 0.1, "NORMAL"),
        _obs(2, "2026-01-01 02:00:00", 0.1, 0.1, "NORMAL"),
        _obs(3, "2026-01-01 03:00:00", -8.0, 4.0, "MEDIUM"),
    ]
    events = group_anomaly_events(obs, series_id="POWER:test")
    assert len(events) == 2
    assert events[0].direction == "ABOVE_EXPECTED"
    assert events[1].direction == "BELOW_EXPECTED"


def test_event_duration_computed_from_timestamps():
    obs = [
        _obs(0, "2026-01-01 00:00:00", 8.0, 4.0, "MEDIUM"),
        _obs(1, "2026-01-01 01:00:00", 8.0, 4.0, "MEDIUM"),
        _obs(2, "2026-01-01 02:00:00", 8.0, 4.0, "MEDIUM"),
    ]
    events = group_anomaly_events(obs, series_id="POWER:test")
    assert events[0].duration_seconds == 7200.0


def test_no_anomalies_produces_no_events():
    obs = [_obs(i, f"2026-01-01 0{i}:00:00", 0.1, 0.1, "NORMAL") for i in range(5)]
    events = group_anomaly_events(obs, series_id="POWER:test")
    assert events == []


# ---------------------------------------------------------------------------
# Full anomaly engine — controlled synthetic scenarios
# ---------------------------------------------------------------------------

def _make_series(timestamps, values, series_id="POWER:test"):
    return AnalysisSeries(
        series_id=series_id,
        role="POWER",
        unit="kW",
        timestamps=pd.Series(pd.to_datetime(timestamps)),
        values=pd.Series(values, dtype=float),
        used_normalized=True,
        sample_count=len(values),
    )


def test_anomaly_engine_normal_periodic_data_low_activity():
    """TEST/SYNTHETIC: clean periodic load with tiny noise should trigger few/no anomalies."""
    rng = np.random.default_rng(1)
    n = 24 * 12
    period = 24
    pattern = 50 + 10 * np.sin(np.linspace(0, 2 * np.pi, period))
    values = np.tile(pattern, n // period) + rng.normal(0, 0.3, n)
    ts = pd.date_range("2026-01-01", periods=n, freq="h")

    series = _make_series(ts, values)
    prepared = prepare_series(series, min_samples=20)
    split = chronological_train_validation_test_split(prepared)
    forecast = run_forecast_engine(prepared, split)
    result = run_anomaly_engine(prepared.series_id, forecast.selected_model_instance, split, forecast.forecast_result)

    assert result.applicable is True
    assert result.anomaly_rate < 10.0  # low anomaly activity expected


def test_anomaly_engine_detects_injected_spike():
    """TEST/SYNTHETIC: one obvious single-point spike must be detected."""
    rng = np.random.default_rng(2)
    n = 24 * 12
    period = 24
    pattern = 50 + 10 * np.sin(np.linspace(0, 2 * np.pi, period))
    values = np.tile(pattern, n // period) + rng.normal(0, 0.3, n)
    spike_idx = n - 10  # inject in the test region
    values[spike_idx] += 500  # enormous, obvious spike
    ts = pd.date_range("2026-01-01", periods=n, freq="h")

    series = _make_series(ts, values)
    prepared = prepare_series(series, min_samples=20)
    split = chronological_train_validation_test_split(prepared)
    forecast = run_forecast_engine(prepared, split)
    result = run_anomaly_engine(prepared.series_id, forecast.selected_model_instance, split, forecast.forecast_result)

    assert result.applicable is True
    assert result.highest_severity == "CRITICAL"
    assert any(e.persistence_class == "ISOLATED_SPIKE" for e in result.events)


def test_anomaly_engine_detects_sustained_event():
    """TEST/SYNTHETIC: a sustained high-load run should form ONE multi-point event."""
    rng = np.random.default_rng(3)
    n = 24 * 12
    period = 24
    pattern = 50 + 10 * np.sin(np.linspace(0, 2 * np.pi, period))
    values = np.tile(pattern, n // period) + rng.normal(0, 0.3, n)
    start = n - 15
    values[start:start + 6] += 300  # sustained elevated run of 6 points
    ts = pd.date_range("2026-01-01", periods=n, freq="h")

    series = _make_series(ts, values)
    prepared = prepare_series(series, min_samples=20)
    split = chronological_train_validation_test_split(prepared)
    forecast = run_forecast_engine(prepared, split)
    result = run_anomaly_engine(prepared.series_id, forecast.selected_model_instance, split, forecast.forecast_result)

    assert result.applicable is True
    sustained_events = [e for e in result.events if e.persistence_class == "SUSTAINED"]
    assert len(sustained_events) >= 1
    assert sustained_events[0].observation_count >= 4


def test_anomaly_engine_no_leakage_thresholds_fit_on_training_only():
    """
    Part D regression: for a FIXED model, the fitted robust stats must
    depend ONLY on training residuals — never on test data. (Model
    *selection* legitimately depends on test RMSE by design — Part F —
    so we hold the model fixed here rather than going through full
    engine model selection, which could legitimately pick a different
    winning model when test data changes and confound this check.)
    """
    from src.forecasting.naive import PersistenceModel

    rng = np.random.default_rng(4)
    train_values = pd.Series(rng.normal(50, 1, 80))

    test_values_a = pd.Series(rng.normal(50, 1, 20))
    test_values_b = pd.Series(rng.normal(500, 50, 20))  # wildly different test data

    model_a = PersistenceModel()
    model_a.fit(pd.Series(range(80)), train_values)
    stats_a = fit_robust_stats(model_a.in_sample_residuals(train_values))

    model_b = PersistenceModel()
    model_b.fit(pd.Series(range(80)), train_values)
    stats_b = fit_robust_stats(model_b.in_sample_residuals(train_values))

    # Same training data, same model -> identical fitted stats, regardless
    # of the (unused-for-fitting) test_values_a/test_values_b above.
    assert stats_a.median == stats_b.median
    assert stats_a.mad == stats_b.mad

    # And end-to-end through the anomaly engine with the model forced via
    # the same training split for both, confirming train_stats never
    # references split.test_values anywhere in the call chain.
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    for test_values in (test_values_a, test_values_b):
        values = pd.concat([train_values, test_values], ignore_index=True)
        series = _make_series(ts, values.tolist())
        prepared = prepare_series(series, min_samples=20)
        split = chronological_train_validation_test_split(prepared)
        model = PersistenceModel()
        model.fit(split.train_timestamps, split.train_values)
        forecast_result_stub = type("Stub", (), {"residuals": (np.asarray(split.test_values) - model.predict(split.test_timestamps)).tolist(), "test_timestamps": [str(t) for t in split.test_timestamps]})()
        result = run_anomaly_engine(prepared.series_id, model, split, forecast_result_stub)
        if test_values is test_values_a:
            engine_stats_a = result.train_stats
        else:
            engine_stats_b = result.train_stats
    assert engine_stats_a.median == engine_stats_b.median
    assert engine_stats_a.mad == engine_stats_b.mad


def test_anomaly_engine_unavailable_without_forecast():
    result = run_anomaly_engine("POWER:test", None, None, None)
    assert result.applicable is False
    assert "no forecast model" in result.reason
