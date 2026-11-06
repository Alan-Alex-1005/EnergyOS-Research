import numpy as np

from src.forecasting.metrics import mae, mape, r_squared, rmse


def test_mae_known_example():
    actual = [10, 20, 30, 40]
    predicted = [12, 18, 33, 37]
    # errors: 2, 2, 3, 3 -> mean = 2.5
    result = mae(actual, predicted)
    assert result.available is True
    assert abs(result.value - 2.5) < 1e-9


def test_rmse_known_example():
    actual = [10, 20, 30, 40]
    predicted = [12, 18, 33, 37]
    # squared errors: 4, 4, 9, 9 -> mean=6.5 -> sqrt=2.5495...
    result = rmse(actual, predicted)
    assert result.available is True
    assert abs(result.value - np.sqrt(6.5)) < 1e-9


def test_mape_known_example():
    actual = [100, 200, 300]
    predicted = [110, 190, 300]
    # pct errors: 10/100=0.10, 10/200=0.05, 0/300=0 -> mean=0.05 -> 5%
    result = mape(actual, predicted)
    assert result.available is True
    assert abs(result.value - 5.0) < 1e-9
    assert result.note is None


def test_mape_zero_actual_handled_explicitly():
    actual = [0, 100, 200]
    predicted = [5, 110, 190]
    result = mape(actual, predicted)
    assert result.available is True
    assert result.note is not None
    assert "1 observation" in result.note
    # computed only over the 2 nonzero observations: 10/100=0.10, 10/200=0.05 -> mean 7.5%
    assert abs(result.value - 7.5) < 1e-9


def test_mape_all_zero_actuals_unavailable():
    actual = [0, 0, 0]
    predicted = [1, 2, 3]
    result = mape(actual, predicted)
    assert result.available is False
    assert result.value is None
    assert "unavailable" in result.note.lower()


def test_r2_perfect_fit():
    actual = [1, 2, 3, 4, 5]
    predicted = [1, 2, 3, 4, 5]
    result = r_squared(actual, predicted)
    assert result.available is True
    assert abs(result.value - 1.0) < 1e-9


def test_r2_known_example():
    actual = [1, 2, 3, 4]
    predicted = [1, 2, 4, 4]
    # mean=2.5, ss_tot = (1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2 = 2.25+0.25+0.25+2.25=5.0
    # ss_res = 0+0+1+0 = 1
    # r2 = 1 - 1/5 = 0.8
    result = r_squared(actual, predicted)
    assert abs(result.value - 0.8) < 1e-9


def test_r2_constant_actual_series_matched_exactly():
    actual = [5, 5, 5, 5]
    predicted = [5, 5, 5, 5]
    result = r_squared(actual, predicted)
    assert result.available is True
    assert result.value == 1.0
    assert "constant" in result.note.lower()


def test_r2_constant_actual_series_not_matched_is_undefined_not_infinite():
    actual = [5, 5, 5, 5]
    predicted = [4, 5, 6, 5]
    result = r_squared(actual, predicted)
    assert result.available is False
    assert result.value is None
    assert "undefined" in result.note.lower()


def test_metrics_handle_nan_pairs_by_dropping_them():
    actual = [10, np.nan, 30]
    predicted = [12, 18, 33]
    result = mae(actual, predicted)
    assert result.available is True
    # only pairs (10,12) and (30,33) remain: errors 2 and 3 -> mean 2.5
    assert abs(result.value - 2.5) < 1e-9


def test_metrics_no_paired_observations():
    result = mae([np.nan, np.nan], [1, 2])
    assert result.available is False
    assert result.value is None
