from src.units import is_known_unit, normalize_value


def test_power_watts_to_kw():
    result = normalize_value(1500, "W", "POWER")
    assert result.converted is True
    assert result.normalized_unit == "kW"
    assert abs(result.normalized_value - 1.5) < 1e-9


def test_power_mw_to_kw():
    result = normalize_value(2, "MW", "POWER")
    assert result.normalized_value == 2000.0


def test_energy_wh_to_kwh():
    result = normalize_value(2500, "Wh", "ENERGY")
    assert result.normalized_unit == "kWh"
    assert abs(result.normalized_value - 2.5) < 1e-9


def test_unknown_unit_is_not_guessed():
    result = normalize_value(100, "banana", "POWER")
    assert result.converted is False
    assert result.normalized_value is None
    assert result.note is not None


def test_missing_unit_not_converted():
    result = normalize_value(100, None, "POWER")
    assert result.converted is False
    assert result.normalized_value is None


def test_temperature_celsius_passthrough():
    result = normalize_value(25, "C", "TEMPERATURE")
    assert result.normalized_value == 25
    assert result.normalized_unit == "C"


def test_temperature_fahrenheit_explicit_conversion():
    result = normalize_value(32, "F", "TEMPERATURE")
    assert abs(result.normalized_value - 0.0) < 1e-9
    assert result.normalized_unit == "C"


def test_temperature_unrecognized_unit():
    result = normalize_value(25, "kelvin_typo", "TEMPERATURE")
    assert result.converted is False


def test_is_known_unit_true_and_false():
    assert is_known_unit("kW", "POWER") is True
    assert is_known_unit("banana", "POWER") is False
    assert is_known_unit(None, "POWER") is False
