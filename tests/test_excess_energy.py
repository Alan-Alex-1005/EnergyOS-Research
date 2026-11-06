import numpy as np

from src.impact.excess_energy import compute_excess_energy


def test_excess_energy_known_example():
    # 1 kW excess for 2 hours -> 2 kWh; second point no excess
    actual = np.array([5.0, 3.0])
    predicted = np.array([4.0, 3.0])
    result = compute_excess_energy("POWER", "kW", 3600.0, True, actual, predicted)
    assert result.available is True
    assert abs(result.excess_energy_kwh - 1.0) < 1e-9  # (1*1h) + (0*1h)


def test_excess_energy_kw_times_hours_equals_kwh():
    actual = np.array([10.0, 10.0, 10.0, 10.0])
    predicted = np.array([4.0, 4.0, 4.0, 4.0])
    # 6 kW excess * 15-min intervals (0.25h) * 4 samples = 6*0.25*4 = 6.0 kWh
    result = compute_excess_energy("POWER", "kW", 900.0, True, actual, predicted)
    assert abs(result.excess_energy_kwh - 6.0) < 1e-9


def test_negative_residual_ignored():
    actual = np.array([2.0, 2.0])
    predicted = np.array([10.0, 10.0])  # actual below expected both times
    result = compute_excess_energy("POWER", "kW", 3600.0, True, actual, predicted)
    assert result.available is True
    assert result.excess_energy_kwh == 0.0


def test_energy_role_rejected():
    result = compute_excess_energy("ENERGY", "kWh", 3600.0, True, np.array([1.0]), np.array([1.0]))
    assert result.available is False
    assert "cumulative" in result.reason


def test_unknown_unit_rejected():
    result = compute_excess_energy("POWER", "W", 3600.0, True, np.array([1.0]), np.array([1.0]))
    assert result.available is False
    assert "kW" in result.reason


def test_irregular_sampling_rejected():
    result = compute_excess_energy("POWER", "kW", 3600.0, False, np.array([1.0]), np.array([1.0]))
    assert result.available is False
    assert "irregular" in result.reason


def test_unknown_sampling_interval_rejected():
    result = compute_excess_energy("POWER", "kW", None, True, np.array([1.0]), np.array([1.0]))
    assert result.available is False


def test_no_paired_observations_rejected():
    result = compute_excess_energy("POWER", "kW", 3600.0, True, np.array([np.nan]), np.array([np.nan]))
    assert result.available is False
    assert "no paired" in result.reason.lower()


def test_provenance_is_estimated():
    result = compute_excess_energy("POWER", "kW", 3600.0, True, np.array([5.0]), np.array([2.0]))
    assert result.provenance == "ESTIMATED"
