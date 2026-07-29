from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.assets import PlotAssets
from fmri_pipeline.analysis.report.figures import carpet as carpet_mod


def _carpet(n_voxels: int = 60, n_frames: int = 40) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n_voxels, n_frames))


def _probseg(tmp_path: Path, name: str, data: np.ndarray) -> Path:
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))
    return path


def test_resolve_tissue_codes_assigns_each_voxel_to_its_highest_probability_class(
    tmp_path: Path,
) -> None:
    gm = np.zeros((2, 2, 2), dtype=np.float32)
    wm = np.zeros((2, 2, 2), dtype=np.float32)
    csf = np.zeros((2, 2, 2), dtype=np.float32)
    gm[0, 0, 0] = 0.9
    wm[0, 0, 1] = 0.8
    csf[1, 1, 1] = 0.7
    assets = PlotAssets(
        probseg={
            "GM": _probseg(tmp_path, "gm.nii.gz", gm),
            "WM": _probseg(tmp_path, "wm.nii.gz", wm),
            "CSF": _probseg(tmp_path, "csf.nii.gz", csf),
        }
    )
    reference = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    codes, source = carpet_mod.resolve_tissue_codes(
        (2, 2, 2), assets=assets, reference_img=reference
    )
    assert source == "probseg"
    assert codes[0, 0, 0] == 1  # GM
    assert codes[0, 0, 1] == 2  # WM
    assert codes[1, 1, 1] == 3  # CSF


def test_resolve_tissue_codes_reports_none_when_no_segmentation_exists() -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    codes, source = carpet_mod.resolve_tissue_codes(
        (2, 2, 2), assets=PlotAssets(), reference_img=reference
    )
    assert codes is None
    assert source == "none"


def test_carpet_declares_when_voxels_are_not_tissue_ordered() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "unordered" in text.lower()
    plt.close(figure)


def test_carpet_time_axis_is_in_seconds_not_frames() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    # Located by label rather than position: the colorbar is also an axes.
    labels = [axis.get_xlabel().lower() for axis in figure.axes]
    assert any("second" in label for label in labels)
    plt.close(figure)


def test_carpet_preserves_nan_in_the_motion_trace() -> None:
    # The first frame of every run has undefined FD. Filling zero draws a dip to
    # "no motion" that is a fabricated value.
    fd = np.concatenate([[np.nan], np.full(39, 0.1)])
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
        fd=fd,
    )
    plotted = figure.axes[0].lines[0].get_ydata()
    assert np.isnan(plotted[0])
    plt.close(figure)


def test_carpet_labels_the_dvars_axis_with_the_series_it_was_given() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
        dvars=np.full(40, 1.0),
        dvars_label="std DVARS",
    )
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert "std DVARS" in labels
    plt.close(figure)


def test_carpet_marks_and_names_every_run_boundary() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[20],
        run_labels=["run-01", "run-02"],
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "run-01" in text and "run-02" in text
    plt.close(figure)


def test_carpet_rejects_a_motion_trace_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="frames"):
        carpet_mod.carpet_figure(
            _carpet(n_frames=40),
            tissue_codes=None,
            tissue_source="none",
            tr=2.0,
            run_boundaries=[],
            run_labels=["run-01"],
            fd=np.zeros(10),
        )


def test_carpet_carries_a_colorbar_naming_its_units() -> None:
    # An image panel without a scale is not a readable figure.
    figure = carpet_mod.carpet_figure(
        _carpet(),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert any("z" in label for label in labels)
    plt.close(figure)


def test_out_of_range_voxels_are_coloured_not_saturated_to_the_background() -> None:
    # Clipped to the top of a grey ramp, an extreme voxel renders white on a white
    # page and reads as missing data -- the opposite of the truth. Non-steady-state
    # volumes land out of range by design, so this is the common case.
    figure = carpet_mod.carpet_figure(
        _carpet(),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    images = [im for ax in figure.axes for im in ax.get_images()]
    assert images, "carpet drew no image"
    colormap = images[0].get_cmap()
    background = (1.0, 1.0, 1.0, 1.0)
    assert tuple(colormap.get_over()) != background
    assert tuple(colormap.get_under()) != tuple(colormap.get_over())
    plt.close(figure)


def test_carpet_colorbar_declares_that_extremes_are_coloured() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert any("coloured" in label for label in labels)
    plt.close(figure)


def test_subsampling_keeps_a_small_tissue_class_visible() -> None:
    # CSF is a few percent of voxels. Strictly proportional sampling can reduce it
    # to a handful of rows that vanish at figure resolution.
    codes = np.concatenate([np.full(9800, 1), np.full(150, 2), np.full(50, 3)])
    carpet = np.random.default_rng(0).standard_normal((10_000, 20))
    rows, sampled_codes = carpet_mod.subsample_rows(
        carpet, codes, max_rows=1000, min_rows_per_class=100
    )
    assert rows.shape[0] <= 1000 + 2 * 100
    assert np.count_nonzero(sampled_codes == 3) >= 50


def test_subsampling_is_a_no_op_below_the_row_budget() -> None:
    carpet = np.random.default_rng(0).standard_normal((100, 20))
    rows, codes = carpet_mod.subsample_rows(carpet, None, max_rows=1000)
    assert rows.shape[0] == 100
    assert codes is None


def test_subsampling_is_deterministic_for_the_same_input() -> None:
    codes = np.concatenate([np.full(5000, 1), np.full(5000, 2)])
    carpet = np.random.default_rng(0).standard_normal((10_000, 20))
    first, _ = carpet_mod.subsample_rows(carpet, codes, max_rows=500)
    second, _ = carpet_mod.subsample_rows(carpet, codes, max_rows=500)
    assert np.array_equal(first, second)


def test_standardisation_takes_its_scale_from_retained_frames_only() -> None:
    # Dummy volumes sit far above steady state. If they set the scale, every other
    # frame is compressed toward neutral and the carpet stops showing anything.
    series = np.tile(np.arange(20.0), (5, 1))
    series[:, :3] += 500.0
    keep = np.ones(20, dtype=bool)
    keep[:3] = False

    naive = carpet_mod.standardise_carpet(series)
    corrected = carpet_mod.standardise_carpet(series, sample_mask=keep)
    assert np.std(corrected[:, 3:]) > np.std(naive[:, 3:]) * 5


def test_standardisation_keeps_dummy_volumes_visible_as_outliers() -> None:
    series = np.tile(np.arange(20.0), (5, 1))
    series[:, :3] += 500.0
    keep = np.ones(20, dtype=bool)
    keep[:3] = False
    corrected = carpet_mod.standardise_carpet(series, sample_mask=keep)
    # Excluded from the scale, but still drawn -- and far off it.
    assert np.abs(corrected[:, :3]).min() > np.abs(corrected[:, 3:]).max()


def test_standardisation_handles_a_zero_variance_voxel() -> None:
    series = np.zeros((3, 10))
    series[1] = np.arange(10.0)
    result = carpet_mod.standardise_carpet(series)
    assert np.all(np.isfinite(result))


def test_carpet_draws_cited_fd_reference_lines() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
        fd=np.full(40, 0.4),
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "0.5" in text and "Power" in text
    plt.close(figure)


def test_carpet_draws_no_reference_lines_without_a_motion_trace() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "Power" not in text
    plt.close(figure)


def test_carpet_states_how_many_voxels_it_actually_drew() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_voxels=60),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "60" in text
    plt.close(figure)


# --- which voxels the carpet is built from --------------------------------


def test_the_carpet_names_the_voxels_it_was_built_from() -> None:
    # Counted against the field of view the sampling fraction reads far smaller than
    # what was applied, and says nothing about whether the voxels drawn are brain.
    figure = carpet_mod.carpet_figure(
        np.random.default_rng(0).standard_normal((50, 30)),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
        voxel_source="analysis mask",
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "analysis mask" in text
    plt.close(figure)


def test_the_carpet_makes_no_claim_about_its_voxels_when_told_nothing() -> None:
    figure = carpet_mod.carpet_figure(
        np.random.default_rng(0).standard_normal((50, 30)),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "voxels drawn" in text
    assert "analysis mask" not in text
    plt.close(figure)
