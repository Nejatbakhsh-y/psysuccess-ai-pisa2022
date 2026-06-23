"""Data loading utilities for PISA-style public-use files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_table(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a data file using the extension.

    Supports common PISA public-use formats: SAV, SAS7BDAT, CSV, TXT, TSV,
    Parquet, and Feather. For fixed-width TXT files, users should first use
    the official SAS/SPSS control files or convert to a delimited format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".sav":
        return pd.read_spss(path)
    if suffix in {".sas7bdat", ".xpt"}:
        return pd.read_sas(path, format="sas7bdat" if suffix == ".sas7bdat" else "xport")
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if suffix == ".txt":
        # Assumes delimiter detection. Fixed-width files require conversion first.
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column found in a DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the candidate columns were found: {candidates}")
