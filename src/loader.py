"""
loader.py — Day 1: File Loader

Loads CSV / XLSX files into a pandas DataFrame with metadata, warnings,
and provenance tracking. Never lets a raw pandas traceback escape to the
CLI — every failure mode is converted into a LoaderError with a clear
human-readable message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


class LoaderError(Exception):
    """Raised for any file-loading failure. Message is CLI-safe."""


@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    source_filename: str
    file_type: str  # "CSV" or "XLSX"
    selected_sheet: Optional[str] = None
    available_sheets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


def _fail(msg: str) -> None:
    raise LoaderError(msg)


def load_file(path: str, sheet: Optional[str] = None) -> LoadResult:
    """
    Load a CSV or XLSX file.

    Parameters
    ----------
    path : str
        Path to the input file.
    sheet : Optional[str]
        For XLSX inputs, the sheet name to load. If omitted and the
        workbook has exactly one sheet, that sheet is used automatically.
        If the workbook has multiple sheets and none is specified, this
        raises LoaderError (ambiguous multi-sheet input).

    Returns
    -------
    LoadResult
    """
    if not path:
        _fail("No file path was provided.")

    if not os.path.exists(path):
        _fail(f"File not found: {path}")

    if os.path.isdir(path):
        _fail(f"Path is a directory, not a file: {path}")

    if os.path.getsize(path) == 0:
        _fail(f"File is empty (0 bytes): {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        _fail(
            f"Unsupported file extension '{ext}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    warnings: list[str] = []
    filename = os.path.basename(path)

    if ext == ".csv":
        return _load_csv(path, filename, warnings)
    else:
        return _load_xlsx(path, filename, sheet, warnings)


def _load_csv(path: str, filename: str, warnings: list[str]) -> LoadResult:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        _fail(f"CSV file has no columns/data to parse: {filename}")
    except pd.errors.ParserError as e:
        _fail(f"Malformed CSV file '{filename}': {e}")
    except UnicodeDecodeError as e:
        _fail(f"Could not decode CSV file '{filename}' (bad encoding): {e}")
    except Exception as e:  # noqa: BLE001 — convert anything else to LoaderError
        _fail(f"Failed to read CSV file '{filename}': {e}")

    if df.shape[1] == 0:
        _fail(f"CSV file '{filename}' contains no columns.")
    if df.shape[0] == 0:
        warnings.append("File contains headers but zero data rows.")

    return LoadResult(
        dataframe=df,
        source_filename=filename,
        file_type="CSV",
        selected_sheet=None,
        available_sheets=[],
        warnings=warnings,
        provenance={
            "loader": "pandas.read_csv",
            "path": os.path.abspath(path),
            "rows_loaded": int(df.shape[0]),
            "columns_loaded": int(df.shape[1]),
        },
    )


def _load_xlsx(
    path: str, filename: str, sheet: Optional[str], warnings: list[str]
) -> LoadResult:
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:  # noqa: BLE001
        _fail(f"Failed to open XLSX file '{filename}': {e}")

    available_sheets = xl.sheet_names
    if not available_sheets:
        _fail(f"XLSX file '{filename}' contains no sheets.")

    selected_sheet = sheet
    if selected_sheet is None:
        if len(available_sheets) == 1:
            selected_sheet = available_sheets[0]
        else:
            _fail(
                f"XLSX file '{filename}' has multiple sheets "
                f"({', '.join(available_sheets)}) and no sheet was selected. "
                f"Pass --sheet <name> to disambiguate."
            )
    elif selected_sheet not in available_sheets:
        _fail(
            f"Sheet '{selected_sheet}' not found in '{filename}'. "
            f"Available sheets: {', '.join(available_sheets)}"
        )

    try:
        df = xl.parse(selected_sheet)
    except Exception as e:  # noqa: BLE001
        _fail(f"Failed to read sheet '{selected_sheet}' from '{filename}': {e}")

    if df.shape[1] == 0:
        _fail(f"Sheet '{selected_sheet}' in '{filename}' contains no columns.")
    if df.shape[0] == 0:
        warnings.append(
            f"Sheet '{selected_sheet}' contains headers but zero data rows."
        )
    if len(available_sheets) > 1:
        warnings.append(
            f"Workbook has {len(available_sheets)} sheets; using '{selected_sheet}'."
        )

    return LoadResult(
        dataframe=df,
        source_filename=filename,
        file_type="XLSX",
        selected_sheet=selected_sheet,
        available_sheets=available_sheets,
        warnings=warnings,
        provenance={
            "loader": "pandas.ExcelFile",
            "path": os.path.abspath(path),
            "sheet": selected_sheet,
            "rows_loaded": int(df.shape[0]),
            "columns_loaded": int(df.shape[1]),
        },
    )
