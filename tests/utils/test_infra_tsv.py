from __future__ import annotations

from pathlib import Path

import pandas as pd

from eeg_pipeline.infra.tsv import read_table, read_tsv, write_parquet, write_stats_table, write_tsv


def test_read_and_write_tsv_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["sub-0001", "sub-0002"],
            "score": [1.25, 2.5],
            "label": ["a", "b"],
        }
    )
    path = tmp_path / "table.tsv"

    write_tsv(df, path)

    loaded = read_tsv(path)
    pd.testing.assert_frame_equal(loaded, df, check_dtype=False)


def test_read_table_dispatches_by_suffix(tmp_path: Path) -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})

    tsv_path = tmp_path / "table.tsv"
    parquet_path = tmp_path / "table.parquet"
    write_tsv(df, tsv_path)
    write_parquet(df, parquet_path)

    loaded_tsv = read_table(tsv_path)
    loaded_parquet = read_table(parquet_path)

    pd.testing.assert_frame_equal(loaded_tsv, df, check_dtype=False)
    pd.testing.assert_frame_equal(loaded_parquet, df, check_dtype=False)


def test_write_stats_table_switches_format_by_size(tmp_path: Path) -> None:
    small_df = pd.DataFrame({"value": [1, 2]})
    large_df = pd.DataFrame({"value": list(range(101))})

    small_input = tmp_path / "small.parquet"
    large_input = tmp_path / "large.tsv"
    forced_input = tmp_path / "forced.parquet"

    small_output = write_stats_table(small_df, small_input)
    large_output = write_stats_table(large_df, large_input)
    forced_output = write_stats_table(large_df, forced_input, force_tsv=True)

    assert small_output == small_input.with_suffix(".tsv")
    assert small_output.exists()
    pd.testing.assert_frame_equal(read_table(small_output), small_df, check_dtype=False)

    assert large_output == large_input.with_suffix(".parquet")
    assert large_output.exists()
    pd.testing.assert_frame_equal(read_table(large_output), large_df, check_dtype=False)

    assert forced_output == forced_input.with_suffix(".tsv")
    assert forced_output.exists()
    pd.testing.assert_frame_equal(read_table(forced_output), large_df, check_dtype=False)
