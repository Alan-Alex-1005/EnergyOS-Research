import numpy as np
import pandas as pd
import pytest

from src.analysis_series import AnalysisSeries
from src.forecasting.engine import SELECTION_CRITERION, run_forecast_engine
from src.forecasting.naive import PersistenceModel
from src.forecasting.seasonal_naive import SeasonalNaiveModel, infer_seasonal_period
from src.timeseries import chronological_train_validation_test_split, prepare_series


def _series(timestamps, values, series_id="POWER:test"):
    return AnalysisSeries(
        series_id=series_id,
        role="POWER",
        unit="kW",
        timestamps=pd.Series(pd.to_datetime(timestamps)),
        values=pd.Series(values, dtype=float),
        used_normalized=True,
        sample_count=len(values),
    )


def test_persistence_model_predicts_last_training_value():
    ts = pd.date_range("2026-01-01", periods=10, freq="h")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    model = PersistenceModel()
    model.fit(ts[:8], pd.Series(values[:8]))
    preds = model.predict(ts[8:])
    assert (preds == 8.0).all()


def test_persistence_requires_at_least_one_training_sample():
    model = PersistenceModel()
    with pytest.raises(ValueError):
        model.fit(pd.Series([], dtype="datetime64[ns]"), pd.Series([], dtype=float))


def test_infer_seasonal_period_known_intervals():
    assert infer_seasonal_period(900) == 96  # 15 min
    assert infer_seasonal_period(1800) == 48  # 30 min
    assert infer_seasonal_period(3600) == 24  # hourly
    assert infer_seasonal_period(123456) is None  # nothing close to a known cadence
    assert infer_seasonal_period(None) is None


def test_seasonal_naive_repeats_last_season():
    period = 4
    train_values = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]  # last season = [10,20,30,40]
    ts = pd.date_range("2026-01-01", periods=8, freq="h")
    model = SeasonalNaiveModel(seasonal_period=period)
    model.fit(ts, pd.Series(train_values))
    test_ts = pd.date_range("2026-01-01 08:00", periods=6, freq="h")
    preds = model.predict(test_ts)
    # first 4 predictions repeat the last season, then it cycles again
    assert list(preds) == [10.0, 20.0, 30.0, 40.0, 10.0, 20.0]


def test_seasonal_naive_rejects_insufficient_training_history():
    model = SeasonalNaiveModel(seasonal_period=96)
    ts = pd.date_range("2026-01-01", periods=10, freq="15min")
    applicable, reason = model.is_applicable(ts, pd.Series(range(10)))
    assert applicable is False
    assert "fewer than one full" in reason


def test_seasonal_naive_rejects_irregular_sampling():
    model = SeasonalNaiveModel(seasonal_period=None)
    ts = pd.date_range("2026-01-01", periods=200, freq="h")
    applicable, reason = model.is_applicable(
        ts, pd.Series(range(200)), is_regular=False, sampling_interval_seconds=None
    )
    assert applicable is False
    assert "no defensible seasonal period" in reason


def test_engine_selects_seasonal_naive_for_clean_seasonal_data():
    """
    Construct a perfectly seasonal series (period 24) with a tiny bit of
    noise — seasonal naive should win on RMSE over persistence.
    """
    rng = np.random.default_rng(42)
    period = 24
    n = 24 * 10  # 10 days hourly
    pattern = 50 + 20 * np.sin(np2 := np.linspace(0, 2 * np.pi, period))
    values = np.tile(pattern, n // period) + rng.normal(0, 0.5, n)
    ts = pd.date_range("2026-01-01", periods=n, freq="h")

    series = _series(ts, values)
    prepared = prepare_series(series, min_samples=20)
    split = chronological_train_validation_test_split(prepared)
    result = run_forecast_engine(prepared, split)

    assert result.selected_model == "Seasonal Naive"
    assert result.forecast_result is not None
    assert result.forecast_result.rmse < 5.0  # much better than the ~20-amplitude swings
    model_names = {e.model_name for e in result.comparison}
    assert "Persistence" in model_names
    assert "Seasonal Naive" in model_names


def test_engine_falls_back_to_persistence_when_seasonal_unavailable():
    ts = pd.date_range("2026-01-01", periods=15, freq="h")  # too short for any seasonal period
    values = list(range(15))
    series = _series(ts, values)
    prepared = prepare_series(series, min_samples=5)
    split = chronological_train_validation_test_split(prepared)
    result = run_forecast_engine(prepared, split)

    seasonal_eval = next(e for e in result.comparison if e.model_name == "Seasonal Naive")
    assert seasonal_eval.applicable is False
    assert seasonal_eval.attempted is False
    assert result.selected_model in ("Persistence", "Linear Lag (AR1)")


def test_engine_reports_comparison_for_every_candidate():
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    values = list(range(100))
    series = _series(ts, values)
    prepared = prepare_series(series)
    split = chronological_train_validation_test_split(prepared)
    result = run_forecast_engine(prepared, split)

    assert len(result.comparison) == 3  # Persistence, Seasonal Naive, Linear Lag
    assert result.selection_criterion == SELECTION_CRITERION
    assert "VALIDATION RMSE" in result.selection_reason


def test_engine_forecast_result_matches_selected_model_metrics():
    ts = pd.date_range("2026-01-01", periods=60, freq="h")
    values = list(range(60))
    series = _series(ts, values)
    prepared = prepare_series(series)
    split = chronological_train_validation_test_split(prepared)
    result = run_forecast_engine(prepared, split)

    selected_eval = next(e for e in result.comparison if e.model_name == result.selected_model)
    assert result.forecast_result.rmse == selected_eval.rmse
    assert result.forecast_result.mae == selected_eval.mae
    # residuals = actual - predicted, verified directly
    actual = np.array(result.forecast_result.actual)
    predicted = np.array(result.forecast_result.predicted)
    residuals = np.array(result.forecast_result.residuals)
    assert np.allclose(residuals, actual - predicted)


def test_engine_no_future_leakage_in_seasonal_naive_rollout():
    """
    Data-leakage regression test (Part D): seasonal-naive predictions for
    the test horizon must be derivable purely from TRAINING values — never
    from test-set actuals. We verify this by checking predictions are
    identical regardless of what the test values actually are.
    """
    period = 4
    train_values = [1.0, 2.0, 3.0, 4.0] * 5  # 20 training samples, clean period-4 pattern
    ts_train = pd.date_range("2026-01-01", periods=20, freq="h")
    ts_test = pd.date_range("2026-01-01 20:00", periods=8, freq="h")

    model_a = SeasonalNaiveModel(seasonal_period=period)
    model_a.fit(ts_train, pd.Series(train_values))
    preds_a = model_a.predict(ts_test)

    # Refit identically — predict() must not depend on what test values would be;
    # it only ever receives test TIMESTAMPS, never test VALUES, by construction.
    model_b = SeasonalNaiveModel(seasonal_period=period)
    model_b.fit(ts_train, pd.Series(train_values))
    preds_b = model_b.predict(ts_test)

    assert np.array_equal(preds_a, preds_b)
    # confirm predict()'s signature never accepts values at all
    import inspect

    sig = inspect.signature(model_a.predict)
    assert list(sig.parameters.keys()) == ["test_timestamps"]


def test_engine_uses_three_way_chronological_split_in_order():
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    series = _series(ts, list(range(100)))
    prepared = prepare_series(series)
    split = chronological_train_validation_test_split(prepared)
    assert split.train_samples == 70
    assert split.validation_samples == 15
    assert split.test_samples == 15
    assert split.train_timestamps.iloc[-1] < split.validation_timestamps.iloc[0]
    assert split.validation_timestamps.iloc[-1] < split.test_timestamps.iloc[0]


def test_engine_selection_uses_validation_rmse_not_test_rmse():
    """
    Regression: model SELECTION must be based on validation_rmse (Part F
    correction), and the winning model must be the one with the lowest
    validation_rmse among evaluated candidates — even if a different
    model happens to have a lower TEST rmse (selection never peeks at test).
    """
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    series = _series(ts, list(range(100)))
    prepared = prepare_series(series)
    split = chronological_train_validation_test_split(prepared)
    result = run_forecast_engine(prepared, split)

    scored = [e for e in result.comparison if e.attempted and e.validation_rmse is not None]
    best_by_validation = min(scored, key=lambda e: e.validation_rmse)
    assert result.selected_model == best_by_validation.model_name


def test_engine_test_set_never_consulted_during_selection():
    """
    Regression: corrupting/replacing the TEST portion's values must not
    change which model gets selected, since selection only ever looks at
    validation. (It CAN change the final reported test metrics, since
    those are computed on the — now different — test set, which is
    expected and correct.)
    """
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    base_values = list(range(100))
    series_a = _series(ts, base_values)
    prepared_a = prepare_series(series_a)
    split_a = chronological_train_validation_test_split(prepared_a)
    result_a = run_forecast_engine(prepared_a, split_a)

    corrupted_values = base_values[:85] + [9999.0] * 15  # blow up only the TEST segment
    series_b = _series(ts, corrupted_values)
    prepared_b = prepare_series(series_b)
    split_b = chronological_train_validation_test_split(prepared_b)
    result_b = run_forecast_engine(prepared_b, split_b)

    assert result_a.selected_model == result_b.selected_model  # selection unaffected by test corruption
    assert result_a.forecast_result.rmse != result_b.forecast_result.rmse  # but reported test metrics DO differ


def test_engine_validation_and_test_ratios_configurable():
    from src.timeseries import chronological_train_validation_test_split as split_fn

    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    series = _series(ts, list(range(100)))
    prepared = prepare_series(series)
    split = split_fn(prepared, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
    assert split.train_samples == 60
    assert split.validation_samples == 20
    assert split.test_samples == 20
    from src.timeseries import TimeSeriesPrepError

    ts = pd.date_range("2026-01-01", periods=1, freq="h")
    series = _series(ts, [1.0])
    prepared = prepare_series(series, min_samples=1)
    with pytest.raises(TimeSeriesPrepError):
        chronological_train_validation_test_split(prepared)
