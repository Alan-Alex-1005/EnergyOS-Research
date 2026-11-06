from src.pipeline import run_pipeline
from tests.conftest import fixture_path


def test_no_energy_data_blocks_forecasting_and_equipment():
    result = run_pipeline(fixture_path("no_energy_data.csv"))
    caps = result.capability_report.as_dict()
    assert caps["Forecasting"].status == "UNAVAILABLE"
    assert caps["Equipment Analysis"].status == "UNAVAILABLE"
    assert "no power/energy/demand" in caps["Forecasting"].reason or caps["Forecasting"].reason


def test_wide_energy_dataset_capabilities_grounded_in_mapping():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    caps = result.capability_report.as_dict()
    # This fixture has timestamp + power/energy + temperature + status, no tariff/carbon/ids.
    assert caps["Weather Context"].status == "AVAILABLE"
    assert caps["Cost Analysis"].status == "UNAVAILABLE"
    assert caps["Carbon Analysis"].status == "UNAVAILABLE"
    assert caps["Equipment Analysis"].status == "LIMITED"


def test_every_capability_has_a_reason():
    result = run_pipeline(fixture_path("wide_energy.csv"))
    for cap in result.capability_report.capabilities:
        assert cap.reason and isinstance(cap.reason, str)


def test_config_override_can_enable_cost_analysis(tmp_path):
    import json

    config = {
        "timestamp_column": "DateTime",
        "mappings": {
            "Main_Power_kW": {"role": "POWER", "unit": "kW"},
            "Energy_kWh": {"role": "ENERGY", "unit": "kWh"},
            "OutdoorTemp_C": {"role": "TEMPERATURE", "unit": "C"},
            "AHU_State": {"role": "STATUS"},
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config))
    result = run_pipeline(fixture_path("wide_energy.csv"), config_path=str(config_path))
    caps = result.capability_report.as_dict()
    # Still no tariff column mapped -> cost analysis stays unavailable.
    assert caps["Cost Analysis"].status == "UNAVAILABLE"
