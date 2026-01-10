"""TSV/parquet file I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

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


def _prepare_dataframe_for_writing(
    df: pd.DataFrame, include_index: bool
) -> pd.DataFrame:
    """Prepare DataFrame for writing by handling index if needed."""
    if include_index:
        return df.reset_index(drop=False)
    return df


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    compression: str = "snappy",
) -> None:
    """Write DataFrame to parquet file."""
    ensure_dir(path.parent)
    prepared_df = _prepare_dataframe_for_writing(df, index)
    prepared_df.to_parquet(path, index=False, compression=compression)


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write DataFrame to TSV file."""
    ensure_dir(path.parent)
    prepared_df = _prepare_dataframe_for_writing(df, index)

    table = pa.Table.from_pandas(prepared_df, preserve_index=False)
    write_options = pa_csv.WriteOptions(delimiter="\t")
    pa_csv.write_csv(table, path, write_options=write_options)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write DataFrame to CSV file."""
    ensure_dir(path.parent)
    prepared_df = _prepare_dataframe_for_writing(df, index)

    table = pa.Table.from_pandas(prepared_df, preserve_index=False)
    write_options = pa_csv.WriteOptions(delimiter=",")
    pa_csv.write_csv(table, path, write_options=write_options)


def _is_parquet_file(path: Path) -> bool:
    """Check if path has parquet extension."""
    return path.suffix.lower() == ".parquet"


def _is_csv_file(path: Path) -> bool:
    """Check if path has CSV extension."""
    return path.suffix.lower() == ".csv"


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read table from TSV or parquet file based on extension."""
    if _is_parquet_file(path):
        return read_parquet(path, **kwargs)
    return read_tsv(path, **kwargs)


def write_table(df: pd.DataFrame, path: Path, index: bool = False, **kwargs) -> None:
    """Write DataFrame to TSV, CSV, or parquet file based on extension."""
    if _is_parquet_file(path):
        write_parquet(df, path, index=index, **kwargs)
    elif _is_csv_file(path):
        write_csv(df, path, index=index)
    else:
        write_tsv(df, path, index=index)


def write_table_with_formats(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    also_save_csv: bool = False,
) -> List[Path]:
    """Write DataFrame to TSV/parquet and optionally also as CSV.
    
    Args:
        df: DataFrame to write
        path: Primary output path (TSV or parquet)
        index: Whether to include index
        also_save_csv: If True, also write a CSV version
        
    Returns:
        List of paths written
    """
    paths_written = []
    
    write_table(df, path, index=index)
    paths_written.append(path)
    
    if also_save_csv and not _is_csv_file(path):
        csv_path = path.with_suffix(".csv")
        write_csv(df, csv_path, index=index)
        paths_written.append(csv_path)
    
    return paths_written


__all__ = [
    "read_tsv",
    "write_tsv",
    "read_parquet",
    "write_parquet",
    "read_table",
    "write_table",
    "write_csv",
    "write_table_with_formats",
]

