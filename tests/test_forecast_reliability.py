
from types import SimpleNamespace

from src.forecasting.reliability import assess_forecast_reliability


def make_result(actual, predicted, r2, rmse):
    return SimpleNamespace(
        actual=actual,
        predicted=predicted,
        r2=r2,
        rmse=rmse,
    )


def test_constant_zero_period_requires_review():
    result = make_result(
        actual=[0, 0, 0, 0],
        predicted=[0, 0, 0, 0],
        r2=1.0,
        rmse=0.0,
    )

    assessment = assess_forecast_reliability(result)

    assert assessment.status == "REVIEW_REQUIRED"
    assert not assessment.suitable_for_anomaly_evidence


def test_negative_r2_requires_review():
    result = make_result(
        actual=[50, 80, 60, 90],
        predicted=[90, 40, 100, 30],
        r2=-0.5,
        rmse=45.0,
    )

    assessment = assess_forecast_reliability(result)

    assert assessment.status == "REVIEW_REQUIRED"
    assert not assessment.suitable_for_anomaly_evidence


def test_reasonable_forecast_passes_basic_checks():
    result = make_result(
        actual=[50, 60, 70, 80],
        predicted=[51, 59, 71, 79],
        r2=0.99,
        rmse=1.0,
    )

    assessment = assess_forecast_reliability(result)

    assert assessment.status == "PROVISIONALLY_ACCEPTABLE"
    assert assessment.suitable_for_anomaly_evidence
