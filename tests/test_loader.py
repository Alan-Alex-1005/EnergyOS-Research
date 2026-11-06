import pytest

from src.loader import LoaderError, load_file
from tests.conftest import fixture_path


def test_load_csv_success():
    result = load_file(fixture_path("simple_timestamp_energy.csv"))
    assert result.file_type == "CSV"
    assert result.dataframe.shape[0] == 48
    assert "timestamp" in result.dataframe.columns
    assert result.provenance["rows_loaded"] == 48


def test_load_missing_file():
    with pytest.raises(LoaderError, match="not found"):
        load_file(fixture_path("does_not_exist.csv"))


def test_load_empty_file():
    with pytest.raises(LoaderError, match="empty"):
        load_file(fixture_path("empty_file.csv"))


def test_load_malformed_csv():
    with pytest.raises(LoaderError):
        load_file(fixture_path("malformed.csv"))


def test_load_unsupported_extension(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("hello")
    with pytest.raises(LoaderError, match="Unsupported file extension"):
        load_file(str(bad_file))


def test_load_xlsx_single_sheet():
    result = load_file(fixture_path("sample_single_sheet.xlsx"))
    assert result.file_type == "XLSX"
    assert result.selected_sheet == "Data"
    assert result.dataframe.shape[0] == 30


def test_load_xlsx_multi_sheet_requires_selection():
    with pytest.raises(LoaderError, match="multiple sheets"):
        load_file(fixture_path("sample_multi_sheet.xlsx"))


def test_load_xlsx_multi_sheet_with_explicit_sheet():
    result = load_file(fixture_path("sample_multi_sheet.xlsx"), sheet="SiteB")
    assert result.selected_sheet == "SiteB"
    assert result.dataframe.shape[0] == 20


def test_load_xlsx_invalid_sheet_name():
    with pytest.raises(LoaderError, match="not found"):
        load_file(fixture_path("sample_multi_sheet.xlsx"), sheet="DoesNotExist")


def test_load_directory_fails():
    with pytest.raises(LoaderError):
        load_file(fixture_path(""))
