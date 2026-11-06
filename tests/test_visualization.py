import os

import numpy as np
import pandas as pd

from src.visualization.research_plots import (
    plot_anomaly_distribution,
    plot_anomaly_score_timeline,
    plot_daily_load_profile,
    plot_dataset_profile,
    plot_energy_timeseries,
    plot_excess_energy_analysis,
    plot_forecast_residuals,
    plot_forecast_vs_actual,
)


def test_dataset_profile_generated(tmp_path):
    result = plot_dataset_profile(str(tmp_path), {"file": "x.csv", "rows": 100, "quality_score": 90.0})
    assert result.generated is True
    assert os.path.getsize(result.path) > 0


def test_energy_timeseries_generated(tmp_path):
    ts = pd.date_range("2026-01-01", periods=50, freq="h")
    values = pd.Series(range(50), dtype=float)
    result = plot_energy_timeseries(str(tmp_path), ts, values, "POWER", "kW")
    assert result.generated is True
    assert os.path.getsize(result.path) > 0


def test_energy_timeseries_skipped_when_all_null(tmp_path):
    ts = pd.date_range("2026-01-01", periods=5, freq="h")
    values = pd.Series([None] * 5, dtype=float)
    result = plot_energy_timeseries(str(tmp_path), ts, values, "POWER", "kW")
    assert result.generated is False
    assert result.reason is not None


def test_daily_load_profile_skipped_when_irregular(tmp_path):
    ts = pd.date_range("2026-01-01", periods=20, freq="h")
    values = pd.Series(range(20), dtype=float)
    result = plot_daily_load_profile(str(tmp_path), ts, values, "POWER", "kW", is_regular=False)
    assert result.generated is False
    assert "irregular" in result.reason


def test_daily_load_profile_generated_for_multi_day_regular_data(tmp_path):
    ts = pd.date_range("2026-01-01", periods=24 * 5, freq="h")
    values = pd.Series(range(24 * 5), dtype=float)
    result = plot_daily_load_profile(str(tmp_path), ts, values, "POWER", "kW", is_regular=True)
    assert result.generated is True


def test_forecast_vs_actual_generated(tmp_path):
    ts = pd.date_range("2026-01-01", periods=10, freq="h")
    result = plot_forecast_vs_actual(str(tmp_path), ts, list(range(10)), list(range(1, 11)), "Persistence", "POWER", "kW")
    assert result.generated is True


def test_forecast_vs_actual_skipped_when_no_forecast(tmp_path):
    result = plot_forecast_vs_actual(str(tmp_path), [], [], [], "Persistence", "POWER", "kW")
    assert result.generated is False


def test_forecast_residuals_generated(tmp_path):
    ts = pd.date_range("2026-01-01", periods=10, freq="h")
    result = plot_forecast_residuals(str(tmp_path), ts, list(range(-5, 5)), "kW")
    assert result.generated is True


def test_anomaly_score_timeline_skipped_when_no_observations(tmp_path):
    result = plot_anomaly_score_timeline(str(tmp_path), [], 3.0)
    assert result.generated is False


def test_excess_energy_analysis_skipped_when_unavailable(tmp_path):
    result = plot_excess_energy_analysis(str(tmp_path), [], [], [], None, "kW")
    assert result.generated is False
    assert "not available" in result.reason


def test_excess_energy_analysis_generated_when_available(tmp_path):
    ts = pd.date_range("2026-01-01", periods=5, freq="h")
    actual = [10.0, 12.0, 8.0, 15.0, 9.0]
    predicted = [8.0, 8.0, 8.0, 8.0, 8.0]
    result = plot_excess_energy_analysis(str(tmp_path), ts, actual, predicted, 12.5, "kW")
    assert result.generated is True


def test_anomaly_distribution_skipped_with_too_few_points(tmp_path):
    result = plot_anomaly_distribution(str(tmp_path), [])
    assert result.generated is False
