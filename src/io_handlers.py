"""
Lightweight IO helpers for exporting pandas DataFrames.

These functions keep file-format specifics out of the GUI layer so that
we can unit test export behavior and reuse it across views.
"""

from __future__ import annotations

from typing import Optional
import investa_core
import csv as _csv
import pandas as pd


def write_dataframe_to_csv(
    df: pd.DataFrame,
    file_path: str,
    *,
    encoding: str = "utf-8",
    quoting: int = _csv.QUOTE_MINIMAL,
) -> None:
    """Write a DataFrame to CSV with sensible defaults.

    Args:
        df: The DataFrame to write.
        file_path: Destination path ending in .csv.
        encoding: File encoding; default utf-8.
        quoting: CSV quoting mode; default QUOTE_MINIMAL.
    """
    # Ported to Rust: rust/investa_core/src/dl.rs
    investa_core.dl_write_csv(df, file_path, encoding, quoting)


def write_dataframe_to_excel(
    df: pd.DataFrame,
    file_path: str,
    *,
    engine: Optional[str] = "openpyxl",
) -> None:
    """Write a DataFrame to Excel (.xlsx).

    Args:
        df: The DataFrame to write.
        file_path: Destination path ending in .xlsx.
        engine: Excel writer engine; default 'openpyxl'.
    """
    # Ported to Rust: rust/investa_core/src/dl.rs
    investa_core.dl_write_excel(df, file_path, engine)
