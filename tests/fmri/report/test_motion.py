"""The motion numbers a methods section quotes, which the report did not carry.

The carpet drew framewise displacement as a trace. A trace answers "was there a spike";
it does not answer how much motion there was, how much survived into the model, or
whether one run is unlike the others -- and those are the figures every exclusion
criterion in the field is written against.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.figures import motion


def _confounds(
    tmp_path: Path,
    name: str,
    fd: np.ndarray | None,
    *,
    outliers: int = 0,
) -> Path:
    frame = pd.DataFrame({"trans_x": np.zeros(len(fd) if fd is not None else 10)})
    if fd is not None:
        # fMRIPrep writes the first frame as n/a: it has no defined displacement.
        frame["framewise_displacement"] = np.concatenate([[np.nan], fd[1:]])
    for index in range(outliers):
        column = np.zeros(len(frame))
        column[index] = 1.0
        frame[f"motion_outlier{index:02d}"] = column
    path = tmp_path / name
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")
    return path


def _rising(n: int = 100, peak: float = 1.2) -> np.ndarray:
    values = np.full(n, 0.05)
    values[n // 2] = peak
    return values


# --- reading the runs -----------------------------------------------------


def test_each_run_gets_its_own_row(tmp_path: Path) -> None:
    paths = [
        _confounds(tmp_path, "a.tsv", _rising()),
        _confounds(tmp_path, "b.tsv", _rising()),
    ]
    summaries = motion.summarise_run_motion(paths, run_labels=("run-01", "run-02"))
    assert [run.label for run in summaries] == ["run-01", "run-02"]


def test_a_run_without_a_label_is_numbered_by_position(tmp_path: Path) -> None:
    paths = [_confounds(tmp_path, "a.tsv", _rising())]
    assert motion.summarise_run_motion(paths, run_labels=())[0].label == "run-01"


def test_the_undefined_first_frame_is_not_counted_as_no_motion(tmp_path: Path) -> None:
    # The first frame of a run has no defined displacement. Reading it as zero pulls
    # every summary statistic down by one fabricated sample.
    fd = np.full(50, 0.4)
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", fd)], run_labels=("run-01",)
    )
    assert summaries[0].median_fd == pytest.approx(0.4)
    assert summaries[0].mean_fd == pytest.approx(0.4)


def test_the_worst_frame_is_reported_separately_from_the_typical_one(
    tmp_path: Path,
) -> None:
    # Motion is spiky: a run whose interquartile range spans 0.05 mm can still reach
    # 1.2 mm, and censoring decisions are made on the spike.
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising(peak=1.2))], run_labels=("run-01",)
    )
    run = summaries[0]
    assert run.median_fd == pytest.approx(0.05)
    assert run.max_fd == pytest.approx(1.2)


def test_the_fraction_above_each_reference_level_is_measured(tmp_path: Path) -> None:
    fd = np.concatenate([np.full(75, 0.1), np.full(25, 0.6)])
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", fd)], run_labels=("run-01",)
    )
    above_02, above_05 = summaries[0].fraction_above
    # The dropped first frame is one of the 75 low ones, so all 25 high frames remain
    # out of 99 defined.
    assert above_02 == pytest.approx(25 / 99, abs=0.005)
    assert above_05 == pytest.approx(25 / 99, abs=0.005)


def test_censoring_comes_from_the_masks_the_model_used(tmp_path: Path) -> None:
    # Recomputing censoring under a different rule would describe an analysis that did
    # not run.
    paths = [_confounds(tmp_path, "a.tsv", _rising(n=20))]
    keep = np.ones(20, dtype=bool)
    keep[:3] = False
    summaries = motion.summarise_run_motion(
        paths, run_labels=("run-01",), sample_masks=[keep]
    )
    assert summaries[0].n_censored == 3
    assert summaries[0].n_retained == 17


def test_a_mask_of_the_wrong_length_is_ignored_rather_than_misapplied(
    tmp_path: Path,
) -> None:
    paths = [_confounds(tmp_path, "a.tsv", _rising(n=20))]
    summaries = motion.summarise_run_motion(
        paths, run_labels=("run-01",), sample_masks=[np.ones(5, dtype=bool)]
    )
    assert summaries[0].n_censored == 0


def test_a_run_without_the_column_is_not_a_run_without_motion(tmp_path: Path) -> None:
    # An absent measurement and a measured zero are different facts, and a row that
    # showed 0.00 mm for the first would be a fabrication.
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", None)], run_labels=("run-01",)
    )
    assert summaries[0].median_fd is None
    assert summaries[0].n_frames == 10


def test_an_unreadable_run_costs_its_own_row_and_not_the_panel(tmp_path: Path) -> None:
    good = _confounds(tmp_path, "a.tsv", _rising())
    summaries = motion.summarise_run_motion(
        [tmp_path / "missing.tsv", good], run_labels=("run-01", "run-02")
    )
    assert len(summaries) == 1


# --- the figure -----------------------------------------------------------


def _figure(tmp_path: Path, runs: int = 3) -> plt.Figure:
    paths = [_confounds(tmp_path, f"{i}.tsv", _rising()) for i in range(runs)]
    summaries = motion.summarise_run_motion(
        paths, run_labels=[f"run-{i + 1:02d}" for i in range(runs)]
    )
    return motion.run_motion_figure(summaries)


def _figure_text(figure: plt.Figure) -> str:
    return " | ".join(artist.get_text() for artist in figure.texts)


def test_the_panel_draws_one_row_per_run(tmp_path: Path) -> None:
    figure = _figure(tmp_path, runs=4)
    assert len(figure.axes[0].get_yticks()) == 4
    plt.close(figure)


def test_the_motion_axis_starts_at_zero(tmp_path: Path) -> None:
    # Unlike the tSNR panel's. Displacement is a magnitude with a real origin: no
    # motion is a value it can take, and the distance from it is the reading.
    figure = _figure(tmp_path)
    assert figure.axes[0].get_xlim()[0] == pytest.approx(0.0)
    plt.close(figure)


def test_the_panel_names_the_reference_levels_it_draws(tmp_path: Path) -> None:
    # A bare dotted line is an unattributed threshold; the citation is what makes it a
    # comparison rather than a criterion this pipeline invented.
    figure = _figure(tmp_path)
    labels = " ".join(a.get_text() for a in figure.axes[0].texts)
    assert "Power" in labels
    plt.close(figure)


def test_the_panel_states_that_the_levels_are_not_criteria(tmp_path: Path) -> None:
    figure = _figure(tmp_path)
    assert "not criteria" in _figure_text(figure)
    plt.close(figure)


def test_the_panel_reports_the_frames_censored(tmp_path: Path) -> None:
    paths = [_confounds(tmp_path, "a.tsv", _rising(n=20))]
    keep = np.ones(20, dtype=bool)
    keep[:4] = False
    summaries = motion.summarise_run_motion(
        paths, run_labels=("run-01",), sample_masks=[keep]
    )
    figure = motion.run_motion_figure(summaries)
    assert "4 censored" in _figure_text(figure)
    plt.close(figure)


def test_a_run_missing_the_column_is_declared_rather_than_silently_dropped(
    tmp_path: Path,
) -> None:
    summaries = motion.summarise_run_motion(
        [
            _confounds(tmp_path, "a.tsv", _rising()),
            _confounds(tmp_path, "b.tsv", None),
        ],
        run_labels=("run-01", "run-02"),
    )
    figure = motion.run_motion_figure(summaries)
    assert "no framewise_displacement" in _figure_text(figure)
    plt.close(figure)


def test_the_panel_refuses_when_no_run_carries_motion(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", None)], run_labels=("run-01",)
    )
    with pytest.raises(ValueError, match="framewise displacement"):
        motion.run_motion_figure(summaries)


# --- the table ------------------------------------------------------------


def test_the_table_carries_every_run_including_unmeasured_ones(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [
            _confounds(tmp_path, "a.tsv", _rising()),
            _confounds(tmp_path, "b.tsv", None),
        ],
        run_labels=("run-01", "run-02"),
    )
    table, rows = motion.motion_table(summaries)
    assert "run-01" in table and "run-02" in table
    assert len(rows) == 3  # header plus two runs


def test_the_table_reports_an_absent_measurement_as_such(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", None)], run_labels=("run-01",)
    )
    table, _rows = motion.motion_table(summaries)
    assert "n/a" in table
    assert "0.000" not in table


def test_the_tsv_is_written_beside_the_report(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising())], run_labels=("run-01",)
    )
    path = motion.write_motion_tsv(summaries, path=tmp_path / "qc" / "motion.tsv")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].startswith("Run\t")
    assert lines[1].startswith("run-01\t")


def test_retained_frames_are_stated_not_left_to_subtraction(tmp_path: Path) -> None:
    # The number that decides the model's residual degrees of freedom.
    keep = np.ones(20, dtype=bool)
    keep[:6] = False
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising(n=20))],
        run_labels=("run-01",),
        sample_masks=[keep],
    )
    table, _rows = motion.motion_table(summaries)
    assert "Retained" in table
    assert summaries[0].n_retained == 14
