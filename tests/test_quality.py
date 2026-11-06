from src.pipeline import run_pipeline
from tests.conftest import fixture_path


def test_quality_score_in_valid_range():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    assert 0 <= result.quality_result.overall <= 100
    for factor_value in result.quality_result.factors.values():
        assert 0 <= factor_value <= 100


def test_quality_factors_sum_to_overall_via_weights():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    q = result.quality_result
    recomputed = sum(q.factors[k] * q.weights[k] for k in q.weights)
    assert abs(round(recomputed, 1) - q.overall) < 0.5


def test_quality_status_thresholds():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    q = result.quality_result
    if q.overall >= 85:
        assert q.status == "GOOD"
    elif q.overall >= 60:
        assert q.status == "FAIR"
    else:
        assert q.status == "POOR"


def test_missing_values_reduce_completeness():
    clean_result = run_pipeline(fixture_path("simple_timestamp_energy.csv"))
    missing_result = run_pipeline(fixture_path("missing_values.csv"))
    assert missing_result.quality_result.factors["completeness"] < clean_result.quality_result.factors["completeness"]


def test_no_timestamp_zero_timestamp_validity():
    result = run_pipeline(fixture_path("no_timestamp.csv"))
    assert result.quality_result.factors["timestamp_validity"] == 0.0
