from src.loader import load_file
from src.profiler import profile_dataframe
from src.role_detection import detect_role, detect_roles
from tests.conftest import fixture_path


def _profiles_for(fixture_name):
    result = load_file(fixture_path(fixture_name))
    profile = profile_dataframe(result.dataframe)
    return profile.columns


def test_power_column_detected_with_unit():
    cols = _profiles_for("wide_energy.csv")
    suggestion = detect_role("Main_Power_kW", cols["Main_Power_kW"])
    assert suggestion.role == "POWER"
    assert suggestion.inferred_unit == "kW"
    assert suggestion.confidence >= 0.7


def test_energy_column_detected_with_unit():
    cols = _profiles_for("wide_energy.csv")
    suggestion = detect_role("Energy_kWh", cols["Energy_kWh"])
    assert suggestion.role == "ENERGY"
    assert suggestion.inferred_unit == "kWh"


def test_temperature_column_detected():
    cols = _profiles_for("wide_energy.csv")
    suggestion = detect_role("OutdoorTemp_C", cols["OutdoorTemp_C"])
    assert suggestion.role == "TEMPERATURE"


def test_status_column_detected_from_name_and_shape():
    cols = _profiles_for("status_on_off.csv")
    suggestion = detect_role("chiller_status", cols["chiller_status"])
    assert suggestion.role == "STATUS"
    assert suggestion.confidence > 0.5


def test_strange_column_name_falls_back_to_unknown():
    cols = _profiles_for("strange_energy_name.csv")
    suggestion = detect_role("ChillerLoadReading", cols["ChillerLoadReading"])
    assert suggestion.role == "UNKNOWN"
    assert len(suggestion.evidence) >= 1


def test_device_id_detected_as_identifier():
    cols = _profiles_for("no_timestamp.csv")
    suggestion = detect_role("device_id", cols["device_id"])
    assert suggestion.role == "DEVICE_ID"


def test_low_confidence_columns_carry_warning():
    cols = _profiles_for("mixed_known_unknown.csv")
    suggestion = detect_role("misc_code", cols["misc_code"])
    if suggestion.confidence < 0.5:
        assert any("auto-map threshold" in w for w in suggestion.warnings) or suggestion.role == "UNKNOWN"


def test_detect_roles_returns_one_suggestion_per_column():
    cols = _profiles_for("mixed_known_unknown.csv")
    suggestions = detect_roles(cols)
    assert {s.column for s in suggestions} == set(cols.keys())
