from __future__ import annotations

import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.coverage import (  # noqa: E402
    coverage_html,
    load_channel_coverage,
    plot_coverage,
)


def _write(directory, *, rois, bad="", task="pain", subject="0015"):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "subject": subject,
                "task": task,
                "bad_channel_sync_policy": "per_run",
                "n_runs": 6,
                "n_channels": 63,
                "n_union_bad_channels": len([c for c in bad.split(",") if c.strip()]),
                "bad_channel_fraction": 0.0,
                "union_bad_channels": bad,
                "roi_coverage": json.dumps(rois),
            }
        ]
    ).to_csv(directory / f"bad_channel_union_qc_task-{task}.tsv", sep="\t", index=False)


def _roi(name, total, remaining):
    return {
        "roi": name,
        "n_channels": total,
        "n_remaining_channels": remaining,
        "n_bad_channels": total - remaining,
        "passes_min_two_channels": remaining >= 2,
    }


def test_intact_coverage_reports_no_failing_region(tmp_path) -> None:
    _write(tmp_path, rois=[_roi("Frontal", 12, 12), _roi("Midline", 3, 3)])

    coverage = load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    assert coverage.failed_rois == ()
    assert coverage.bad_channels == ()
    assert "fewer than" not in coverage_html(coverage)


def test_a_small_region_losing_channels_is_flagged(tmp_path) -> None:
    """A three-channel region losing two cannot support a regional average."""
    _write(
        tmp_path,
        rois=[_roi("Frontal", 12, 10), _roi("Midline", 3, 1)],
        bad="Pz, POz, Cz",
    )

    coverage = load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    assert coverage.failed_rois == ("Midline",)
    assert coverage.bad_channels == ("Pz", "POz", "Cz")
    document = coverage_html(coverage)
    assert "Midline" in document
    assert "min_roi_channels" in document
    for verdict in ("Exclude those regions", "&#9888;", "at best"):
        assert verdict not in document


def test_bad_fraction_is_relative_to_the_montage(tmp_path) -> None:
    _write(tmp_path, rois=[_roi("Frontal", 12, 12)], bad="Pz, POz, Cz")

    coverage = load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    assert coverage.bad_fraction == pytest.approx(3 / 63)


def test_absent_table_produces_no_section(tmp_path) -> None:
    assert load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015") is None


def test_absent_subject_produces_no_section(tmp_path) -> None:
    _write(tmp_path, rois=[_roi("Frontal", 12, 12)])

    assert load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0099") is None


def test_figure_marks_failing_regions_in_the_flag_colour(tmp_path) -> None:
    from eeg_pipeline.preprocessing.report.style import FLAG_COLOR

    _write(tmp_path, rois=[_roi("Frontal", 12, 10), _roi("Midline", 3, 1)])
    coverage = load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    figure = plot_coverage(coverage)
    axis = figure.axes[0]

    colours = [label.get_color() for label in axis.get_xticklabels()]
    assert FLAG_COLOR in colours


def test_no_exclusions_needs_no_figure_at_all() -> None:
    """With nothing excluded the two bar layers are identical, so the blue bars cover the
    grey ones completely and the legend advertises a series that cannot be seen. A clean
    subject gets the sentence instead: there is no loss to plot."""
    from eeg_pipeline.preprocessing.report.coverage import coverage_figure_is_informative

    intact = pd.DataFrame(
        [
            {"roi": "Frontal", "n_total": 12, "n_remaining": 12},
            {"roi": "Midline", "n_total": 3, "n_remaining": 3},
        ]
    )
    lost = pd.DataFrame(
        [
            {"roi": "Frontal", "n_total": 12, "n_remaining": 11},
            {"roi": "Midline", "n_total": 3, "n_remaining": 3},
        ]
    )

    assert not coverage_figure_is_informative(intact)
    assert coverage_figure_is_informative(lost)


def test_the_section_says_so_when_nothing_was_excluded(tmp_path) -> None:
    _write(tmp_path, rois=[_roi("Frontal", 12, 12), _roi("Midline", 3, 3)])
    coverage = load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    document = coverage_html(coverage)

    assert "every region keeps its full complement" in document.lower()


def test_channels_outside_every_region_are_accounted_for(tmp_path) -> None:
    """Eight regions summed to 52 of 63 channels and the other 11 went unmentioned, which
    reads as a discrepancy in the table rather than as channels with no ROI."""
    _write(tmp_path, rois=[_roi("Frontal", 12, 12), _roi("Midline", 3, 3)])
    coverage = load_channel_coverage(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    document = coverage_html(coverage)

    # 63 channels, 15 assigned to a region.
    assert "48" in document
    assert "no region" in document.lower()


def _write_run_bads(directory, *, subject="0015", task="pain", per_run):
    """Write the per-run ``_bads.tsv`` files the pipeline records beside each run.

    Written at the BIDS depth the pipeline actually uses — ``sub-X/eeg/`` under the
    derivatives root — so the discovery being tested is the recursive one.
    """
    directory = directory / f"sub-{subject}" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    for run, entries in per_run.items():
        frame = pd.DataFrame(
            [{"name": name, "reason": reason} for name, reason in entries],
            columns=["name", "reason"],
        )
        frame.to_csv(
            directory / f"sub-{subject}_task-{task}_run-{run}_bads.tsv",
            sep="\t",
            index=False,
        )


def test_run_bad_channels_are_read_per_run(tmp_path) -> None:
    """The per-run record is what says whether a channel failed once or throughout."""
    from eeg_pipeline.preprocessing.report.coverage import load_run_bad_channels

    _write_run_bads(
        tmp_path,
        per_run={1: [("TP9", "noisy")], 2: [], 3: [("TP9", "noisy"), ("Fp1", "flat")]},
    )

    runs = load_run_bad_channels(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    assert [run.run_label for run in runs] == ["run-1", "run-2", "run-3"]
    assert runs[0].bad_channels == ("TP9",)
    assert runs[1].bad_channels == ()
    assert runs[2].bad_channels == ("Fp1", "TP9")


def test_run_bad_channels_are_absent_when_nothing_was_recorded(tmp_path) -> None:
    """A dataset processed without the per-run record gets no table, not an empty one."""
    from eeg_pipeline.preprocessing.report.coverage import load_run_bad_channels

    assert load_run_bad_channels(deriv_eeg_root=tmp_path, task="pain", subject="0015") == []


def test_the_run_table_states_every_run_including_the_clean_ones(tmp_path) -> None:
    """A run with no bad channel is evidence too, so it gets a row rather than a gap."""
    from eeg_pipeline.preprocessing.report.coverage import (
        load_run_bad_channels,
        run_bad_channel_html,
    )

    _write_run_bads(tmp_path, per_run={1: [("TP9", "noisy")], 2: []})
    runs = load_run_bad_channels(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    document = run_bad_channel_html(runs)

    assert "run-1" in document and "run-2" in document
    assert "TP9" in document
    assert "—" in document


def test_the_run_matrix_is_drawn_only_when_runs_disagree(tmp_path) -> None:
    """A channel bad in every run is a montage fact; one bad in a single run is an event.

    The matrix exists to separate those two. When every run agrees there is nothing to
    separate and the table already says which channels they agree on.
    """
    from eeg_pipeline.preprocessing.report.coverage import (
        load_run_bad_channels,
        run_matrix_is_informative,
    )

    _write_run_bads(tmp_path, per_run={1: [("TP9", "noisy")], 2: [("TP9", "noisy")]})
    agreeing = load_run_bad_channels(deriv_eeg_root=tmp_path, task="pain", subject="0015")
    assert not run_matrix_is_informative(agreeing)

    _write_run_bads(tmp_path, per_run={1: [("TP9", "noisy")], 2: []})
    disagreeing = load_run_bad_channels(deriv_eeg_root=tmp_path, task="pain", subject="0015")
    assert run_matrix_is_informative(disagreeing)


def test_the_run_matrix_plots_every_channel_that_failed_anywhere(tmp_path) -> None:
    from eeg_pipeline.preprocessing.report.coverage import (
        load_run_bad_channels,
        plot_run_bad_channel_matrix,
    )

    _write_run_bads(
        tmp_path,
        per_run={1: [("TP9", "noisy")], 2: [], 3: [("Fp1", "flat"), ("TP9", "noisy")]},
    )
    runs = load_run_bad_channels(deriv_eeg_root=tmp_path, task="pain", subject="0015")

    figure = plot_run_bad_channel_matrix(runs)

    axis = figure.axes[0]
    assert [label.get_text() for label in axis.get_xticklabels()] == ["run-1", "run-2", "run-3"]
    assert set(label.get_text() for label in axis.get_yticklabels()) == {"Fp1", "TP9"}


def test_the_review_replaces_the_per_run_items_with_its_own_table(tmp_path) -> None:
    """The drop happens beside the replacement, so the report is never left with neither."""
    import mne

    from eeg_pipeline.preprocessing.report.coverage import add_coverage_review

    _write(tmp_path, rois=[_roi("Frontal", 12, 12)])
    _write_run_bads(tmp_path, per_run={1: [("TP9", "noisy")], 2: []})
    report = mne.Report(title="cov", verbose="ERROR")
    for run in (1, 2):
        report.add_html(
            html="<p/>",
            title=f"Bad channels: run-{run}",
            section="Data quality",
            tags=("raw", "data-quality"),
        )

    add_coverage_review(report=report, deriv_eeg_root=tmp_path, task="pain", subject="0015")

    titles = [element.name for element in report._content]
    assert not any(title.startswith("Bad channels: run-") for title in titles)
    assert "Bad channels by run" in titles
    assert "Which channels failed in which run" in titles


def test_the_per_run_items_survive_when_there_is_no_per_run_record(tmp_path) -> None:
    """Without a replacement to show, dropping MNE's items would lose the evidence."""
    import mne

    from eeg_pipeline.preprocessing.report.coverage import add_coverage_review

    _write(tmp_path, rois=[_roi("Frontal", 12, 12)])
    report = mne.Report(title="cov", verbose="ERROR")
    report.add_html(
        html="<p/>",
        title="Bad channels: run-1",
        section="Data quality",
        tags=("raw", "data-quality"),
    )

    add_coverage_review(report=report, deriv_eeg_root=tmp_path, task="pain", subject="0015")

    assert "Bad channels: run-1" in [element.name for element in report._content]
