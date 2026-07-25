from __future__ import annotations

import pandas as pd


def _write_desc(frame: pd.DataFrame, path):
    """Mirror the writer in eeg_pipeline.preprocessing.pipeline.stats."""
    frame.describe().rename_axis("statistic").to_csv(path, sep="\t")


def test_summary_statistics_keep_their_names(tmp_path) -> None:
    """describe() carries mean/std/min in its index; dropping it leaves anonymous rows."""
    frame = pd.DataFrame(
        {
            "n_bad_channels": [0, 3, 1],
            "total_clean_epochs": [58, 52, 60],
        }
    )
    path = tmp_path / "stats_desc.tsv"

    _write_desc(frame, path)
    written = pd.read_csv(path, sep="\t")

    assert written.columns[0] == "statistic"
    assert list(written["statistic"]) == [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    mean = written.set_index("statistic").loc["mean", "total_clean_epochs"]
    assert mean == pd.Series([58, 52, 60]).mean()


def test_the_writer_in_the_pipeline_matches_this_contract() -> None:
    """Guard the call site, so the index cannot be silently dropped again."""
    from tests import REPO_ROOT

    source = (REPO_ROOT / "eeg_pipeline" / "preprocessing" / "pipeline" / "stats.py").read_text(
        encoding="utf-8"
    )
    call_lines = [
        line for line in source.splitlines() if ".describe()" in line and "to_csv" in line
    ]
    assert len(call_lines) == 1, call_lines
    assert "index=False" not in call_lines[0]
    assert "rename_axis" in call_lines[0]
