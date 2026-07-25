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
