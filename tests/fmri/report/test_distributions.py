from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import distributions

# The z histogram's tests moved to test_calibration.py with the panel that
# replaced it. It drew a theoretical null over an unmasked volume and offered no
# way to see the map's own.


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
