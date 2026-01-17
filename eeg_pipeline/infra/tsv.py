"""TSV/parquet file I/O utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from .paths import ensure_dir


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    """Read TSV file into DataFrame."""
    defaults = {"sep": "\t", "low_memory": False}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Read parquet file into DataFrame."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    compression: str = "snappy",
) -> None:
    """Write DataFrame to parquet file."""
    ensure_dir(path.parent)
    if index:
        df = df.reset_index(drop=False)
    df.to_parquet(path, index=False, compression=compression)


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write DataFrame to TSV file."""
    ensure_dir(path.parent)
    if index:
        df = df.reset_index(drop=False)
    table = pa.Table.from_pandas(df, preserve_index=False)
    write_options = pa_csv.WriteOptions(delimiter="\t")
    pa_csv.write_csv(table, path, write_options=write_options)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write DataFrame to CSV file."""
    ensure_dir(path.parent)
    if index:
        df = df.reset_index(drop=False)
    table = pa.Table.from_pandas(df, preserve_index=False)
    write_options = pa_csv.WriteOptions(delimiter=",")
    pa_csv.write_csv(table, path, write_options=write_options)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read table from TSV or parquet file based on extension."""
    if path.suffix.lower() == ".parquet":
        return read_parquet(path, **kwargs)
    return read_tsv(path, **kwargs)


def write_table(df: pd.DataFrame, path: Path, index: bool = False, **kwargs) -> None:
    """Write DataFrame to TSV, CSV, or parquet file based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        write_parquet(df, path, index=index, **kwargs)
    elif suffix == ".csv":
        write_csv(df, path, index=index)
    else:
        write_tsv(df, path, index=index)


_PARQUET_SIZE_THRESHOLD = 100


def write_stats_table(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    force_tsv: bool = False,
) -> Path:
    """Write stats DataFrame, using parquet for large tables.
    
    Automatically switches to parquet format for DataFrames with more than
    _PARQUET_SIZE_THRESHOLD rows, unless force_tsv is True.
    
    Args:
        df: DataFrame to write
        path: Output path (extension will be adjusted if needed)
        index: Whether to include index
        force_tsv: If True, always use TSV regardless of size
        
    Returns:
        Actual path written (may differ from input if extension changed)
    """
    use_parquet = not force_tsv and len(df) > _PARQUET_SIZE_THRESHOLD
    
    if use_parquet:
        actual_path = path.with_suffix(".parquet")
        write_parquet(df, actual_path, index=index)
    else:
        actual_path = path.with_suffix(".tsv") if path.suffix == ".parquet" else path
        write_tsv(df, actual_path, index=index)
    
    return actual_path


__all__ = [
    "read_tsv",
    "write_tsv",
    "read_parquet",
    "write_parquet",
    "read_table",
    "write_table",
    "write_csv",
    "write_stats_table",
]

