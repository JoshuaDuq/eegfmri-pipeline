"""
TSV file I/O utilities.

This module provides functions for reading and writing TSV files
with sensible defaults for the EEG pipeline.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .paths import ensure_dir


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a TSV file with sensible defaults."""
    defaults = {"sep": "\t", "low_memory": False}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=index)


__all__ = [
    "read_tsv",
    "write_tsv",
]










