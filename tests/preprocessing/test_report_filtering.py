"""The report states the filter that was requested; this pins what was actually built.

``preprocessing.l_freq = 0.1`` is one line of configuration and a 33-second FIR filter at
500 Hz. Nothing in the report said so, and the length of the filter relative to the epoch
it is applied to is the fact that decides whether a slow drift in one trial can reach the
next one.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.filtering import (  # noqa: E402
    FilterDescription,
    describe_filter,
    filter_response_html,
    plot_filter_response,
)


def _description(**overrides) -> FilterDescription:
    defaults = {"sfreq": 500.0, "l_freq": 0.1, "h_freq": 100.0, "notch_freq": 60.0}
    defaults.update(overrides)
    return describe_filter(**defaults)


def test_the_filter_is_described_by_its_length_in_seconds() -> None:
    """A tap count means nothing beside an epoch measured in seconds."""
    description = _description()

    assert description.duration_s == pytest.approx(33.0, abs=0.5)
    assert "33" in filter_response_html(description)


def test_the_measured_corners_are_reported_rather_than_the_requested_ones() -> None:
    """MNE places the requested frequency in the passband and the -6 dB point below it,
    so the number in the configuration table is not where the filter actually turns over."""
    description = _description()

    assert description.low_cutoff_hz < 0.1
    assert description.high_cutoff_hz > 100.0


def test_a_recording_with_no_high_pass_has_no_lower_corner() -> None:
    """Low-passing alone is a legitimate configuration and must not invent a corner."""
    description = _description(l_freq=None)

    assert description.low_cutoff_hz is None
    assert description.duration_s > 0.0


def test_the_step_response_panel_marks_the_epoch_it_will_be_read_inside() -> None:
    """Filter ringing that settles before the epoch starts is harmless; ringing that does
    not is the reason a long high-pass filter distorts a slow evoked response."""
    figure = plot_filter_response(_description(), epoch_window_s=(-7.0, 15.0))

    step_axis = figure.axes[1]
    marked = [line.get_xdata()[0] for line in step_axis.lines if len(set(line.get_xdata())) == 1]
    assert pytest.approx(-7.0) in marked


def test_the_panel_renders_without_an_epoch_window() -> None:
    """Resting-state runs are analysed continuously and have no epoch to mark."""
    figure = plot_filter_response(_description(), epoch_window_s=None)

    assert len(figure.axes) >= 2


def test_the_magnitude_axis_is_logarithmic_in_frequency() -> None:
    """A 0.1 Hz corner and a 100 Hz corner on one linear axis puts the high-pass, which is
    the filter most able to distort an evoked response, inside the first pixel."""
    figure = plot_filter_response(_description(), epoch_window_s=None)

    assert figure.axes[0].get_xscale() == "log"


def test_the_two_requested_corners_are_told_apart_without_reading_the_legend_order() -> None:
    """Both were drawn in one colour with one dash pattern, so the high-pass and the
    low-pass markers were distinguishable only by which legend entry came first."""
    figure = plot_filter_response(_description(), epoch_window_s=None)

    styles = {
        (line.get_linestyle(), line.get_color())
        for line in figure.axes[0].lines
        if "requested" in str(line.get_label())
    }
    assert len(styles) == 2


class _Config:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_the_configured_filter_is_read_from_the_same_keys_the_provenance_table_records() -> None:
    """The panel and the configuration table must describe one filter, not two."""
    from eeg_pipeline.preprocessing.report.filtering import describe_configured_filter

    description = describe_configured_filter(
        _Config({"preprocessing.l_freq": 0.1, "preprocessing.h_freq": 100.0}),
        sfreq=500.0,
    )

    assert description is not None
    assert description.l_freq == 0.1 and description.h_freq == 100.0


def test_an_unfiltered_configuration_yields_no_panel_rather_than_an_empty_one() -> None:
    """A dataset filtered upstream has nothing for this section to describe."""
    from eeg_pipeline.preprocessing.report.filtering import describe_configured_filter

    assert describe_configured_filter(_Config({}), sfreq=500.0) is None
