from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import distributions


def test_z_histogram_uses_a_log_y_axis() -> None:
    # A linear axis lets the null peak hide the tails, which are the signal.
    figure = distributions.z_histogram(np.random.default_rng(0).standard_normal(10_000))
    assert figure.axes[0].get_yscale() == "log"
    plt.close(figure)


def test_z_histogram_overlays_the_standard_normal_null() -> None:
    figure = distributions.z_histogram(np.random.default_rng(0).standard_normal(10_000))
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert any("N(0, 1)" in label for label in labels)
    plt.close(figure)


def test_z_histogram_draws_both_threshold_lines_when_given_a_threshold() -> None:
    figure = distributions.z_histogram(
        np.random.default_rng(0).standard_normal(1000), threshold=2.3
    )
    positions = sorted(
        line.get_xdata()[0]
        for line in figure.axes[0].lines
        if line.get_linestyle() == "--"
    )
    assert positions == pytest.approx([-2.3, 2.3])
    plt.close(figure)


def test_z_histogram_has_no_legend_entries_without_a_threshold() -> None:
    # An unconditional legend() call previously warned and drew an empty box.
    figure = distributions.z_histogram(np.random.default_rng(0).standard_normal(1000))
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 1  # the null curve only
    plt.close(figure)


def test_z_histogram_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        distributions.z_histogram(np.array([]))


def test_magnitude_histogram_marks_the_median() -> None:
    figure = distributions.magnitude_histogram(np.arange(100.0), xlabel="tSNR")
    positions = [
        line.get_xdata()[0]
        for line in figure.axes[0].lines
        if line.get_linestyle() == "--"
    ]
    assert positions == pytest.approx([49.5])
    plt.close(figure)


def test_magnitude_histogram_labels_the_axis_it_was_given() -> None:
    figure = distributions.magnitude_histogram(np.arange(100.0), xlabel="tSNR")
    assert figure.axes[0].get_xlabel() == "tSNR"
    plt.close(figure)
