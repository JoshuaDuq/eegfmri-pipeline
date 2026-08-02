"""The motion numbers a methods section quotes, which the report did not carry.

The carpet drew framewise displacement as a trace. A trace answers "was there a spike";
it does not answer how much motion there was, how much survived into the model, or
whether one run is unlike the others -- and those are the figures every exclusion
criterion in the field is written against.
"""

from __future__ import annotations

import re
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


# --- motion against signal change -----------------------------------------
#
# Both traces already share the carpet's time axis, where they answer "was there a
# spike". Whether they move *together* is what decides whether motion contaminated
# the result, and no panel said so.


def _coupled_confounds(
    tmp_path: Path,
    name: str,
    *,
    n: int = 60,
    coupling: float = 1.0,
    seed: int = 0,
    dvars_column: str = "std_dvars",
) -> Path:
    rng = np.random.default_rng(seed)
    fd = np.abs(rng.normal(0.1, 0.04, size=n))
    dvars = 1.0 + coupling * (fd - fd.mean()) + rng.normal(0, 0.01, size=n)
    frame = pd.DataFrame(
        {
            # The first frame has no defined displacement; fMRIPrep writes n/a.
            "framewise_displacement": np.concatenate([[np.nan], fd[1:]]),
            dvars_column: dvars,
        }
    )
    path = tmp_path / name
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")
    return path


def test_the_coupling_panel_reports_the_correlation_it_measured(tmp_path: Path) -> None:
    figure = motion.motion_coupling_figure(
        [_coupled_confounds(tmp_path, "a.tsv", coupling=4.0)]
    )
    provenance = " ".join(artist.get_text() for artist in figure.texts)
    assert "within-run r = +" in provenance
    plt.close(figure)


def test_tightly_coupled_motion_reads_higher_than_independent_motion(
    tmp_path: Path,
) -> None:
    def correlation(coupling: float) -> float:
        figure = motion.motion_coupling_figure(
            [
                _coupled_confounds(
                    tmp_path, f"c{coupling}.tsv", coupling=coupling, n=400
                )
            ]
        )
        text = " ".join(artist.get_text() for artist in figure.texts)
        plt.close(figure)
        return float(re.search(r"within-run r = ([+-][\d.]+)", text).group(1))

    assert correlation(8.0) > correlation(0.0)


def test_the_undefined_first_frame_is_dropped_from_both_axes(tmp_path: Path) -> None:
    # Dropping one trace's gaps but not the other's pairs every subsequent frame
    # with its neighbour's value -- a scatter that looks entirely normal and is
    # scrambled. This is the bug the raw-column reader exists to avoid.
    figure = motion.motion_coupling_figure(
        [_coupled_confounds(tmp_path, "a.tsv", n=60)]
    )
    provenance = " ".join(artist.get_text() for artist in figure.texts)
    assert "59 paired frames" in provenance
    plt.close(figure)


def test_runs_are_pooled_and_counted(tmp_path: Path) -> None:
    figure = motion.motion_coupling_figure(
        [
            _coupled_confounds(tmp_path, "a.tsv", n=40, seed=1),
            _coupled_confounds(tmp_path, "b.tsv", n=40, seed=2),
        ]
    )
    provenance = " ".join(artist.get_text() for artist in figure.texts)
    assert "78 paired frames across 2 run(s)" in provenance
    plt.close(figure)


def test_plain_dvars_is_accepted_and_named_as_such(tmp_path: Path) -> None:
    figure = motion.motion_coupling_figure(
        [_coupled_confounds(tmp_path, "a.tsv", dvars_column="dvars")]
    )
    assert figure.axes[0].get_ylabel() == "DVARS"
    plt.close(figure)


def test_a_run_without_dvars_leaves_nothing_to_relate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to relate"):
        motion.motion_coupling_figure([_confounds(tmp_path, "a.tsv", _rising())])


def test_the_coupling_panel_scores_no_run(tmp_path: Path) -> None:
    # The reference levels are published conventions; which runs they make
    # acceptable is a study's decision, not this module's.
    figure = motion.motion_coupling_figure(
        [_coupled_confounds(tmp_path, "a.tsv", coupling=4.0)]
    )
    provenance = " ".join(artist.get_text() for artist in figure.texts)
    assert "not criteria applied here" in provenance
    plt.close(figure)


# --- tSNR joins the table rather than getting a panel ----------------------
#
# tSNR carries no published reference level to be located against -- unlike framewise
# displacement, which keeps its figure for exactly that reason -- so "is one run
# unlike the others" is a comparison between six numbers, and six numbers are a
# column. The dot plot it used to have put six runs within 2 tSNR of each other.


def test_the_table_carries_tsnr_when_it_is_supplied(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising()), _confounds(tmp_path, "b.tsv", _rising())],
        run_labels=("run-01", "run-02"),
    )
    table, rows = motion.motion_table(
        summaries, tsnr_median=[60.4, 58.9], tsnr_iqr=[(42.0, 77.0), (41.5, 76.2)]
    )
    assert "Median tSNR" in table and "tSNR IQR" in table
    assert "60.4" in table and "42.0–77.0" in table
    assert rows[0].endswith("Median tSNR\ttSNR IQR")


def test_the_table_omits_the_tsnr_columns_when_it_has_none(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising())], run_labels=("run-01",)
    )
    table, _rows = motion.motion_table(summaries)
    assert "Median tSNR" not in table


def test_mismatched_tsnr_is_ignored_rather_than_misaligned(tmp_path: Path) -> None:
    # A column shifted by one run would attribute every value to the wrong row.
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising()), _confounds(tmp_path, "b.tsv", _rising())],
        run_labels=("run-01", "run-02"),
    )
    table, _rows = motion.motion_table(summaries, tsnr_median=[60.4])
    assert "Median tSNR" not in table


def test_the_tsv_carries_the_same_columns_as_the_table(tmp_path: Path) -> None:
    summaries = motion.summarise_run_motion(
        [_confounds(tmp_path, "a.tsv", _rising())], run_labels=("run-01",)
    )
    path = motion.write_motion_tsv(
        summaries,
        path=tmp_path / "qc" / "run_qc.tsv",
        tsnr_median=[60.4],
        tsnr_iqr=[(42.0, 77.0)],
    )
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].endswith("Median tSNR\ttSNR IQR")
    assert lines[1].endswith("60.4\t42.0–77.0")


def test_the_coupling_panel_draws_the_relationship_it_reports(tmp_path: Path) -> None:
    # Reported only in the provenance strip, the correlation was a number beside a
    # cloud in which the reader could not see it -- and how tightly the two move
    # together is the entire reading.
    figure = motion.motion_coupling_figure(
        [_coupled_confounds(tmp_path, "a.tsv", coupling=6.0, n=600)]
    )
    axis = figure.axes[0]
    assert axis.lines, "no trend is drawn"
    trend = axis.lines[0].get_ydata()
    # Binned medians rather than a least-squares line: framewise displacement is
    # spike-dominated, so a line is levered by the few frames it least describes.
    assert trend[-1] > trend[0]
    on_plot = " ".join(artist.get_text() for artist in axis.texts)
    assert "within-run r" in on_plot
    plt.close(figure)


def _coupling_confounds(tmp_path, name, *, fd, dvars):
    import pandas as pd

    path = tmp_path / name
    pd.DataFrame(
        {"framewise_displacement": fd, "std_dvars": dvars}
    ).to_csv(path, sep="\t", index=False)
    return path


def test_coupling_is_measured_within_runs_not_across_them(tmp_path):
    """Pooling runs mixes within-run coupling with between-run baseline differences.

    Two runs each with zero internal coupling, offset from one another, produce a
    strong pooled correlation that describes the offset rather than any coupling.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    low = rng.uniform(0.02, 0.04, 400)
    high = rng.uniform(0.20, 0.22, 400)
    paths = [
        _coupling_confounds(tmp_path, "a.tsv", fd=low, dvars=rng.uniform(0.9, 1.0, 400)),
        _coupling_confounds(tmp_path, "b.tsv", fd=high, dvars=rng.uniform(1.4, 1.5, 400)),
    ]
    figure = motion.motion_coupling_figure(paths, run_labels=("run-01", "run-02"))
    text = " ".join(artist.get_text() for artist in figure.texts)

    assert "within-run" in text
    reported = float(
        re.search(r"within-run r = ([+-]?\d+\.\d+)", text).group(1)
    )
    assert abs(reported) < 0.2, f"the reported r is the between-run offset: {reported}"
    plt.close(figure)


def test_the_per_run_spread_is_reported(tmp_path):
    import numpy as np

    rng = np.random.default_rng(1)
    paths = [
        _coupling_confounds(
            tmp_path,
            f"r{i}.tsv",
            fd=rng.uniform(0.02, 0.15, 300),
            dvars=rng.uniform(0.9, 1.2, 300),
        )
        for i in range(3)
    ]
    figure = motion.motion_coupling_figure(
        paths, run_labels=("run-01", "run-02", "run-03")
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "per run" in text
    plt.close(figure)


def test_a_single_run_reports_its_own_correlation(tmp_path):
    import numpy as np

    rng = np.random.default_rng(2)
    fd = rng.uniform(0.02, 0.2, 400)
    path = _coupling_confounds(tmp_path, "solo.tsv", fd=fd, dvars=1.0 + 2.0 * fd)
    figure = motion.motion_coupling_figure([path], run_labels=("run-01",))
    text = " ".join(artist.get_text() for artist in figure.texts)
    reported = float(re.search(r"within-run r = ([+-]?\d+\.\d+)", text).group(1))
    assert reported > 0.9
    plt.close(figure)


def test_the_dvars_column_preference_is_standardised_first():
    """Raw DVARS is in image intensity units and is not comparable across runs.

    Both panels that show DVARS must resolve it the same way; picking raw in one and
    standardised in the other puts two different quantities under one name in a single
    report.
    """
    import pandas as pd

    both = pd.DataFrame({"dvars": [20.0, 30.0], "std_dvars": [0.9, 1.1]})
    values, label = motion._dvars_column(both)
    assert label == "std DVARS"
    assert values.tolist() == [0.9, 1.1]

    raw_only = pd.DataFrame({"dvars": [20.0, 30.0]})
    _values, label = motion._dvars_column(raw_only)
    assert label == "DVARS"
