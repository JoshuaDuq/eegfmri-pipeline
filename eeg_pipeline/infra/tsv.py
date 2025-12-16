"""TSV/parquet file I/O utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import pyarrow as pa
import pyarrow.csv as pa_csv

from .paths import ensure_dir


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    defaults = {"sep": "\t", "low_memory": False}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    compression: str = "snappy",
) -> None:
    ensure_dir(path.parent)
    if index:
        df = df.reset_index(drop=False)
    df.to_parquet(path, index=False, compression=compression)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return read_parquet(path, **kwargs)
    return read_tsv(path, **kwargs)


def write_table(df: pd.DataFrame, path: Path, index: bool = False, **kwargs) -> None:
    if str(path).lower().endswith(".parquet"):
        return write_parquet(df, path, index=index, **kwargs)
    return write_tsv(df, path, index=index)


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    if index:
        df = df.reset_index(drop=False)

    table = pa.Table.from_pandas(df, preserve_index=False)
    write_options = pa_csv.WriteOptions(delimiter="\t")
    pa_csv.write_csv(table, path, write_options=write_options)


__all__ = [
    "read_tsv",
    "write_tsv",
    "read_parquet",
    "write_parquet",
    "read_table",
    "write_table",
]
