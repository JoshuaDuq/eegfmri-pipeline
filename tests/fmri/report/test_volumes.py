from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import volumes


def _bold(seed: int = 0, affine: np.ndarray | None = None) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.standard_normal((8, 8, 8, 20))).astype(np.float32)
    return nib.Nifti1Image(data, np.eye(4) if affine is None else affine)


def test_compute_tsnr_returns_mean_over_standard_deviation() -> None:
    """tSNR is the mean over the *detrended* temporal standard deviation."""
    rng = np.random.default_rng(7)
    n_frames = 40
    noise = rng.standard_normal((2, 2, 2, n_frames)) * 5.0
    data = (100.0 + noise).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))

    result = volumes.compute_tsnr([img])
    expected = np.mean(data, axis=3) / volumes.detrended_temporal_sd(
        data.astype(float), np.ones((2, 2, 2), dtype=bool)
    )
    assert np.allclose(np.asarray(result.mean_img.get_fdata()), expected, rtol=1e-4)


def test_a_pure_drift_voxel_reports_no_tsnr_rather_than_a_flattering_one() -> None:
    """A voxel whose only variation is drift has no measurable thermal noise.

    Before drift removal this fixture reported mean/std(ramp) -- a finite,
    respectable-looking tSNR computed entirely from the scanner's drift. Undefined
    is the honest answer.
    """
    data = np.zeros((2, 2, 2, 10), dtype=np.float32)
    data[...] = np.arange(10, dtype=np.float32)
    result = volumes.compute_tsnr([nib.Nifti1Image(data, np.eye(4))])
    assert np.allclose(np.asarray(result.mean_img.get_fdata()), 0.0)


def test_compute_tsnr_preserves_the_source_affine() -> None:
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    result = volumes.compute_tsnr([_bold(affine=affine)])
    assert np.allclose(result.mean_img.affine, affine)


def test_compute_tsnr_averages_across_runs() -> None:
    result = volumes.compute_tsnr([_bold(0), _bold(1)])
    assert result.mean_img.shape == (8, 8, 8)
    assert len(result.per_run_median) == 2


def test_non_steady_state_frames_are_excluded_from_tsnr() -> None:
    # The first frames of a run sit at much higher intensity before longitudinal
    # magnetisation saturates. Including them inflates the temporal standard
    # deviation, so every tSNR value comes out biased low.
    rng = np.random.default_rng(0)
    data = (100.0 + rng.standard_normal((6, 6, 6, 20))).astype(np.float32)
    data[..., :3] *= 3.0  # dummy volumes
    img = nib.Nifti1Image(data, np.eye(4))

    keep = np.ones(20, dtype=bool)
    keep[:3] = False
    with_dummies = volumes.compute_tsnr([img])
    without = volumes.compute_tsnr([img], sample_masks=[keep])

    assert without.per_run_median[0] > with_dummies.per_run_median[0] * 2
    assert without.frames_dropped == (3,)
    assert without.frames_used == (17,)


def test_per_run_medians_expose_a_single_bad_run() -> None:
    # Averaging maps across runs hides exactly what a QC panel exists to show.
    good = _bold(0)
    bad_data = np.asanyarray(_bold(1).dataobj).copy()
    bad_data += np.random.default_rng(2).standard_normal(bad_data.shape) * 50
    bad = nib.Nifti1Image(bad_data.astype(np.float32), np.eye(4))

    result = volumes.compute_tsnr([good, bad])
    assert result.per_run_median[0] > result.per_run_median[1] * 2


def test_per_run_tsnr_figure_labels_every_run() -> None:
    result = volumes.compute_tsnr([_bold(0), _bold(1)])
    figure = volumes.per_run_tsnr_figure(result, run_labels=["run-01", "run-02"])
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "run-01" in text and "run-02" in text
    plt.close(figure)


def test_sample_mask_length_must_match_the_run() -> None:
    with pytest.raises(ValueError, match="frames"):
        volumes.compute_tsnr([_bold()], sample_masks=[np.ones(5, dtype=bool)])


def test_compute_tsnr_rejects_a_three_dimensional_image() -> None:
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="4D"):
        volumes.compute_tsnr([img])


def test_compute_tsnr_rejects_an_empty_run_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        volumes.compute_tsnr([])


def test_tsnr_volume_renders_through_nilearn_rather_than_slicing_the_array() -> None:
    # Voxel-axis slicing labels panels by anatomy without consulting the affine,
    # which is wrong for any non-RAS-canonical image.
    result = volumes.compute_tsnr([_bold()])
    with patch("nilearn.plotting.plot_img") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            volumes.tsnr_volume(result)
        except Exception:
            pass
    assert mock_plot.called


def test_tsnr_volume_uses_the_single_hue_magnitude_colormap() -> None:
    result = volumes.compute_tsnr([_bold()])
    with patch("nilearn.plotting.plot_img") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            volumes.tsnr_volume(result)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["cmap"] == "cividis"


def test_tsnr_volume_returns_a_figure() -> None:
    figure = volumes.tsnr_volume(volumes.compute_tsnr([_bold()]))
    assert figure is not None
    plt.close(figure)


def test_tsnr_volume_states_how_many_frames_were_censored() -> None:
    keep = np.ones(20, dtype=bool)
    keep[:4] = False
    result = volumes.compute_tsnr([_bold()], sample_masks=[keep])
    figure = volumes.tsnr_volume(result)
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "4" in text and "censored" in text
    plt.close(figure)


def test_linear_drift_does_not_inflate_the_temporal_standard_deviation() -> None:
    """Drift is removed by the GLM's high-pass, so leaving it in under-reports tSNR."""
    from fmri_pipeline.analysis.report.figures import volumes

    rng = np.random.default_rng(0)
    n_frames = 60
    noise = rng.standard_normal((2, 2, 2, n_frames)) * 0.5
    drift = np.linspace(0.0, 20.0, n_frames)
    data = noise + drift
    mask = np.ones((2, 2, 2), dtype=bool)

    plain = np.std(data, axis=3)
    detrended = volumes.detrended_temporal_sd(data, mask)

    assert plain.mean() > 5.0, "the fixture must actually carry drift"
    assert detrended.mean() < 1.0
    assert np.allclose(detrended, 0.5, atol=0.2)


def test_detrending_degrades_gracefully_on_a_very_short_run() -> None:
    """Fewer frames than basis functions cannot be detrended; report the plain sd."""
    from fmri_pipeline.analysis.report.figures import volumes

    data = np.ones((2, 2, 2, 3), dtype=float)
    mask = np.ones((2, 2, 2), dtype=bool)
    result = volumes.detrended_temporal_sd(data, mask)
    assert result.shape == (2, 2, 2)
    assert np.all(np.isfinite(result))


def test_tsnr_is_not_biased_low_by_scanner_drift() -> None:
    """The end-to-end consequence: a drifting run must not report a depressed tSNR."""
    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import volumes

    rng = np.random.default_rng(1)
    n_frames = 60
    shape = (4, 4, 4)
    signal = 1000.0 + rng.standard_normal((*shape, n_frames)) * 10.0
    drifting = signal + np.linspace(0.0, 100.0, n_frames)

    steady = volumes.compute_tsnr([nib.Nifti1Image(signal, np.eye(4))])
    drifted = volumes.compute_tsnr([nib.Nifti1Image(drifting, np.eye(4))])

    # Same thermal noise in both, so the same tSNR -- within sampling error.
    assert drifted.per_run_median[0] == pytest.approx(
        steady.per_run_median[0], rel=0.15
    )


def test_the_tsnr_figure_declares_the_drift_correction() -> None:
    """A tSNR value cannot be compared against another unless its basis is stated."""
    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import volumes

    rng = np.random.default_rng(2)
    data = 1000.0 + rng.standard_normal((4, 4, 4, 30)) * 10.0
    result = volumes.compute_tsnr([nib.Nifti1Image(data, np.eye(4))])
    figure = volumes.tsnr_volume(result)
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "drift" in text.lower()
    finally:
        plt.close(figure)


def test_the_tsnr_volume_states_its_orientation_convention() -> None:
    """Every volume panel declares its convention; a L/R error is invisible otherwise."""
    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import volumes

    rng = np.random.default_rng(3)
    data = 1000.0 + rng.standard_normal((4, 4, 4, 30)) * 10.0
    result = volumes.compute_tsnr([nib.Nifti1Image(data, np.eye(4))])
    figure = volumes.tsnr_volume(result)
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "neurological" in text or "radiological" in text
    finally:
        plt.close(figure)


# --- the tSNR distribution follows the brain mask --------------------------


def test_the_tsnr_median_comes_from_the_mask_not_from_positive_voxels() -> None:
    """`tsnr > 0` drags partial-volume rim voxels into a distribution called 'masked'.

    Rim voxels sit at very low tSNR, so including them pulls the reported median
    down and widens the colour limit's range for no reason.
    """
    rng = np.random.default_rng(0)
    shape = (10, 10, 10)
    data = np.full((*shape, 40), 5.0)
    brain = np.zeros(shape, dtype=bool)
    brain[2:8, 2:8, 2:8] = True
    # In-brain voxels: high tSNR. Rim voxels: noisy, so very low tSNR.
    data[brain] = 1000.0 + rng.standard_normal((int(brain.sum()), 40)) * 10.0
    data[~brain] = 100.0 + rng.standard_normal((int((~brain).sum()), 40)) * 100.0

    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    mask_img = nib.Nifti1Image(brain.astype(np.uint8), np.eye(4))

    unmasked = volumes.compute_tsnr([img])
    masked = volumes.compute_tsnr([img], mask_img=mask_img)

    assert masked.per_run_median[0] > unmasked.per_run_median[0]
    assert masked.per_run_median[0] > 50.0


def test_the_tsnr_distribution_falls_back_to_positive_voxels_without_a_mask() -> None:
    """No mask is a state of the derivatives, not a fault."""
    data = np.zeros((6, 6, 6, 30), dtype=np.float32)
    rng = np.random.default_rng(1)
    data[1:4, 1:4, 1:4] = 500.0 + rng.standard_normal((3, 3, 3, 30)) * 5.0
    result = volumes.compute_tsnr([nib.Nifti1Image(data, np.eye(4))])
    assert result.per_run_median[0] > 0


# --- per-run tSNR: is any run unlike the others? ---------------------------


def _result(medians, iqr=None, dropped=None) -> volumes.TsnrResult:
    return volumes.TsnrResult(
        mean_img=nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)),
        per_run_median=tuple(medians),
        frames_used=tuple(100 for _ in medians),
        frames_dropped=tuple(dropped or [0] * len(medians)),
        per_run_iqr=tuple(iqr or [(m - 5, m + 5) for m in medians]),
    )


def test_the_panel_scales_to_the_data_rather_than_anchoring_at_zero() -> None:
    # Six runs between 58.6 and 60.5 drew six visually identical bars on a 0-60 axis,
    # so the between-run variation the panel exists to show was invisible.
    figure = volumes.per_run_tsnr_figure(
        _result([59.0, 58.6, 60.5, 60.3, 60.4, 60.2]),
        run_labels=[f"run-{i:02d}" for i in range(1, 7)],
    )
    low, _high = figure.axes[0].get_xlim()
    assert low > 10.0, "the axis still spends its range on the distance from zero"
    plt.close(figure)


def test_a_truncated_axis_says_that_it_is_truncated() -> None:
    # A truncated axis on a ratio quantity is only honest when the reader is told the
    # origin is off the figure.
    figure = volumes.per_run_tsnr_figure(
        _result([59.0, 60.5]), run_labels=["run-01", "run-02"]
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "does not start at zero" in text
    plt.close(figure)


def test_the_panel_draws_no_bars_from_a_baseline() -> None:
    # Dots carry no baseline claim, which is what makes scaling the axis honest.
    figure = volumes.per_run_tsnr_figure(
        _result([59.0, 60.5]), run_labels=["run-01", "run-02"]
    )
    assert not figure.axes[0].patches, "bars imply a zero baseline the axis does not show"
    plt.close(figure)


def test_the_panel_shows_each_run_s_spread_not_only_its_centre() -> None:
    # A median alone cannot separate a run that lost signal everywhere from one that
    # lost it in a region: both move the centre, only the second widens the spread.
    narrow = volumes.per_run_tsnr_figure(
        _result([60.0, 60.0], iqr=[(58, 62), (30, 90)]),
        run_labels=["run-01", "run-02"],
    )
    spans = [
        abs(line.get_xdata()[1] - line.get_xdata()[0])
        for line in narrow.axes[0].lines
        if len(line.get_xdata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
    ]
    assert max(spans) > min(spans), "both runs drew the same spread"
    plt.close(narrow)


def test_the_panel_marks_the_across_run_median_as_a_reference() -> None:
    figure = volumes.per_run_tsnr_figure(
        _result([50.0, 60.0, 70.0]), run_labels=["a", "b", "c"]
    )
    text = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "across-run median 60.0" in text
    plt.close(figure)


def test_the_panel_reports_the_range_across_runs() -> None:
    figure = volumes.per_run_tsnr_figure(
        _result([50.0, 60.0, 70.0]), run_labels=["a", "b", "c"]
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "20.0 tSNR" in text
    plt.close(figure)


def test_censored_frames_are_named_in_the_run_s_own_label() -> None:
    # Floated beside the dot the note landed between two rows and could be read as
    # belonging to either run.
    figure = volumes.per_run_tsnr_figure(
        _result([59.0, 60.0], dropped=[7, 0]), run_labels=["run-01", "run-02"]
    )
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert labels[0] == "run-01\n(7 censored)"
    assert labels[1] == "run-02"
    plt.close(figure)


def test_the_panel_refuses_an_empty_result() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        volumes.per_run_tsnr_figure(_result([]), run_labels=[])


def test_compute_tsnr_reports_each_run_s_quartiles() -> None:
    rng = np.random.default_rng(0)
    img = nib.Nifti1Image(
        (100.0 + rng.standard_normal((6, 6, 6, 30))).astype(np.float32), np.eye(4)
    )
    result = volumes.compute_tsnr([img])
    assert len(result.per_run_iqr) == 1
    low, high = result.per_run_iqr[0]
    assert low < result.per_run_median[0] < high
