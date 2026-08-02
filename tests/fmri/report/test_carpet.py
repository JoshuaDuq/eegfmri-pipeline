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
    assert any("% signal change" in label for label in labels)
    plt.close(figure)


def test_the_colorbar_names_the_units_it_was_given() -> None:
    """The residual carpet is in per-voxel z, and mislabelling it would misstate it."""
    figure = carpet_mod.carpet_figure(
        _carpet(),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=["run-01"],
        colour_limit=carpet_mod.RESIDUAL_Z_CLIP,
        value_label="z (per voxel)",
    )
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert any("z (per voxel)" in label for label in labels)
    assert not any("% signal change" in label for label in labels)
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


def test_tissue_order_reports_unclassified_rows_as_their_own_block() -> None:
    carpet = np.arange(16, dtype=float).reshape(4, 4)
    codes = np.array([0, 1, 3, 2], dtype=np.int8)

    ordered, blocks = carpet_mod.order_by_tissue(carpet, codes)

    assert np.array_equal(ordered, carpet[[1, 3, 2, 0]])
    assert blocks == [
        ("GM", 0, 1),
        ("WM", 1, 2),
        ("CSF", 2, 3),
        ("Unclassified", 3, 4),
    ]


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


# --- censoring ------------------------------------------------------------
#
# The motion table already counts the censored frames. What the carpet adds is
# *which* frames, read against the motion that removed them -- a comparison between
# two things rather than a fact about either.


def test_censored_frames_are_shaded_on_the_motion_trace() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((40, 30))
    censored = np.zeros(30, dtype=bool)
    censored[5:8] = True
    censored[20] = True

    figure = carpet_mod.carpet_figure(
        data,
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=("run-01",),
        fd=np.abs(rng.normal(0.1, 0.05, size=30)),
        censored=censored,
    )
    # Two contiguous spans, not four separate frames: one patch per frame is
    # thousands of artists on a real concatenation.
    shaded = [p for p in figure.axes[0].patches if p.get_alpha() == 0.25]
    assert len(shaded) == 2
    plt.close(figure)


def test_the_carpet_keys_the_shading_it_draws() -> None:
    # An unexplained grey band is a mark a reader cannot read.
    rng = np.random.default_rng(0)
    censored = np.zeros(30, dtype=bool)
    censored[3:6] = True
    figure = carpet_mod.carpet_figure(
        rng.standard_normal((40, 30)),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=("run-01",),
        fd=np.abs(rng.normal(0.1, 0.05, size=30)),
        censored=censored,
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "3 censored frame(s)" in text
    plt.close(figure)


def test_a_carpet_without_censoring_draws_no_shading() -> None:
    rng = np.random.default_rng(0)
    figure = carpet_mod.carpet_figure(
        rng.standard_normal((40, 30)),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=("run-01",),
        fd=np.abs(rng.normal(0.1, 0.05, size=30)),
    )
    assert not [p for p in figure.axes[0].patches if p.get_alpha() == 0.25]
    plt.close(figure)


def test_a_censoring_mask_of_the_wrong_length_is_rejected() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        carpet_mod.carpet_figure(
            rng.standard_normal((40, 30)),
            tissue_codes=None,
            tissue_source="none",
            tr=2.0,
            run_boundaries=[],
            run_labels=("run-01",),
            censored=np.zeros(7, dtype=bool),
        )


def test_not_retained_frames_are_missing_columns_and_are_named() -> None:
    rng = np.random.default_rng(0)
    not_retained = np.zeros(30, dtype=bool)
    not_retained[[0, 12]] = True

    figure = carpet_mod.carpet_figure(
        rng.standard_normal((40, 30)),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=("run-01",),
        not_retained=not_retained,
    )

    image = next(image for axis in figure.axes for image in axis.get_images())
    drawn = np.ma.asarray(image.get_array())
    assert np.ma.getmaskarray(drawn)[:, [0, 12]].all()
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "2 acquired frame(s) not retained" in text
    plt.close(figure)


def test_carpet_can_state_the_full_voxel_denominator_after_presampling() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_voxels=40),
        tissue_codes=None,
        tissue_source="none",
        tr=2.0,
        run_boundaries=[],
        run_labels=("run-01",),
        voxel_count_total=100,
    )

    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "40 of 100 voxels drawn" in text
    plt.close(figure)


def _series(amplitudes, n_frames=200, baseline=100.0):
    """One voxel per amplitude, each a sinusoid of that amplitude about ``baseline``."""
    time = np.linspace(0.0, 6.0 * np.pi, n_frames)
    return np.vstack([baseline + a * np.sin(time) for a in amplitudes])


def test_percent_scaling_preserves_relative_amplitude_across_voxels():
    """Per-voxel z-scoring gives every row unit variance by construction.

    A quiet voxel and a badly corrupted one are then drawn identically, which
    destroys exactly the contrast a carpet exists to show.
    """
    scaled = carpet_mod.scale_carpet(_series([0.1, 3.0]))
    quiet, loud = np.ptp(scaled[0]), np.ptp(scaled[1])
    assert loud > 20 * quiet


def test_z_scoring_is_what_destroys_that_contrast():
    """Stated as a test so the reason for the default cannot be lost."""
    scaled = carpet_mod.standardise_carpet(_series([0.1, 3.0]))
    assert np.ptp(scaled[1]) == pytest.approx(np.ptp(scaled[0]), rel=0.01)


def test_percent_scaling_is_centred_on_each_voxels_own_mean():
    scaled = carpet_mod.scale_carpet(_series([0.5, 2.0]))
    assert abs(float(np.mean(scaled))) < 1e-6


def test_percent_scaling_is_in_percent_of_the_voxel_mean():
    """A 1% modulation reads as 1 whatever the voxel's absolute level.

    Scale invariance is the property: grey matter, white matter and CSF differ
    severalfold in intensity, and a common denominator would make the brightest
    tissue look the most active whatever it did.
    """
    dim = carpet_mod.scale_carpet(_series([1.0], baseline=100.0))
    bright = carpet_mod.scale_carpet(_series([20.0], baseline=2000.0))

    assert np.max(dim) == pytest.approx(np.max(bright), rel=1e-9)
    assert np.max(dim) == pytest.approx(1.0, rel=1e-3)


def test_scaling_takes_its_reference_from_retained_frames_only():
    """Non-steady-state volumes sit far above the rest and would inflate the mean."""
    series = _series([1.0])
    series[:, :3] += 500.0
    keep = np.ones(series.shape[1], dtype=bool)
    keep[:3] = False

    scaled = carpet_mod.scale_carpet(series, sample_mask=keep)
    assert abs(float(np.mean(scaled[:, keep]))) < 1e-6
    # The excluded frames are still drawn, and still extreme.
    assert np.max(scaled[:, :3]) > 100


def test_a_constant_voxel_is_flat_rather_than_extreme():
    series = np.full((1, 50), 100.0)
    assert np.allclose(carpet_mod.scale_carpet(series), 0.0)


def test_a_zero_mean_voxel_does_not_divide_by_zero():
    series = np.zeros((1, 40))
    scaled = carpet_mod.scale_carpet(series)
    assert np.all(np.isfinite(scaled))


def test_the_colour_limit_survives_one_corrupted_voxel():
    """A spike inflates its own voxel's mean, so every frame of that row goes extreme.

    Pooled over all values that is a whole row of outliers, which outruns any
    percentile once the voxel count is small -- and the voxel count is a display
    choice, not a property of the data.
    """
    series = _series([1.0] * 40)
    series[0, 0] = 100_000.0
    limit = carpet_mod.carpet_colour_limit(carpet_mod.scale_carpet(series))
    assert 0.5 < limit < 5.0


def test_a_frame_bad_across_every_voxel_needs_the_sample_mask():
    """Non-steady-state volumes are extreme in every row at once.

    No display limit can rescue this on its own: three frames at fifty times the
    signal shift each voxel's own mean by most of its value, so every frame in the
    run becomes a large percentage of the wrong reference. Excluding them from the
    reference is the fix, and it is why scale_carpet takes a sample mask at all.
    """
    series = _series([1.0] * 40)
    series[:, :3] += 5000.0

    unmasked = carpet_mod.carpet_colour_limit(carpet_mod.scale_carpet(series))
    assert unmasked > 100, "the fixture no longer reproduces an inflated reference"

    keep = np.ones(series.shape[1], dtype=bool)
    keep[:3] = False
    scaled = carpet_mod.scale_carpet(series, sample_mask=keep)
    assert np.max(np.abs(scaled[:, keep])) < 2.0, "the reference is still inflated"

    # The limit must exclude the same frames. Three bad frames in two hundred is 1.5%
    # of a row, so a 99th percentile over all of them lands on one of the bad frames
    # and stretches the scale to cover the very values it should place out of range.
    assert carpet_mod.carpet_colour_limit(scaled) > 100
    masked = carpet_mod.carpet_colour_limit(scaled, sample_mask=keep)
    assert 0.5 < masked < 5.0


def test_the_colour_limit_describes_a_typical_voxel_not_the_loudest():
    """Set by the loudest, ordinary tissue structure compresses toward mid-grey."""
    series = _series([1.0] * 39 + [50.0])
    limit = carpet_mod.carpet_colour_limit(carpet_mod.scale_carpet(series))
    assert limit < 5.0
