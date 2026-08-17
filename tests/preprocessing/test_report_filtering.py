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


def test_the_filter_exposes_its_exact_half_support_for_edge_exclusion() -> None:
    description = _description()

    assert description.edge_support_s == pytest.approx(
        (description.n_taps - 1) / (2.0 * description.sfreq)
    )
    assert "Edge support" in filter_response_html(description)


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


def test_the_filter_section_reports_the_stopbands_applied_upstream() -> None:
    """The section described a passband the delivered data does not have.

    Configuration reported "Notch: not set" and the response showed a flat passband from
    0.061 to 112 Hz, while the recordings carried 36 to 82 stopbands per run from the
    upstream line-removal stage — 17 to 39 Hz taken out of the 15–90 Hz band alone on
    sub-0001. The intervals are already read by the comb code from the QC sidecar; what
    filtering was actually applied is the first thing a methods section needs.
    """
    description = describe_filter(sfreq=500.0, l_freq=0.1, h_freq=100.0)

    html = filter_response_html(
        description,
        unavailable_intervals_by_recording={
            "sub-0001_task-x_run-1": [(57.9, 61.5), (26.1, 26.5)],
            "sub-0001_task-x_run-2": [(57.9, 61.5)],
        },
    )

    assert "run-1" in html and "run-2" in html
    # Bandwidth removed per run: 3.6 + 0.4 Hz on run-1, 3.6 Hz on run-2.
    assert "4.0" in html and "3.6" in html
    # The span they fall between is a different number from their total width.
    assert "26.1–61.5" in html


def test_a_recording_with_no_upstream_removal_gets_no_stopband_table() -> None:
    """An unnotched dataset must not carry an empty table asserting something was done."""
    description = describe_filter(sfreq=500.0, l_freq=0.1, h_freq=100.0)

    assert "upstream" not in filter_response_html(description).lower()


def test_the_stopband_table_holds_only_the_reported_subject() -> None:
    """The settings carry every subject's intervals, and run labels drop the subject.

    Rendering the mapping whole produced one row per subject per run, all labelled
    run-1..run-6, so a six-run session showed ninety-six rows of six repeating labels.
    """
    description = describe_filter(sfreq=500.0, l_freq=0.1, h_freq=100.0)

    html = filter_response_html(
        description,
        unavailable_intervals_by_recording={
            "sub-0001_task-x_run-1": [(57.9, 61.5)],
            "sub-0002_task-x_run-1": [(10.0, 12.0), (20.0, 22.0)],
        },
        subject="0001",
    )

    body = html.split("Frequencies an upstream stage removed")[1]
    assert body.count("run-1") == 1
    # The other subject's two stopbands must not appear as this subject's.
    assert "4.0" not in body
