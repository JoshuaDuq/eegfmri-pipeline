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
    data = np.zeros((2, 2, 2, 10), dtype=np.float32)
    data[...] = np.arange(10, dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    result = volumes.compute_tsnr([img])
    expected = float(np.mean(np.arange(10)) / np.std(np.arange(10)))
    assert np.allclose(np.asarray(result.mean_img.get_fdata()), expected)


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
