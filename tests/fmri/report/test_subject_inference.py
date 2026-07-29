"""Threshold resolution, masking, and the anatomical underlay in the subject report."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.figures import coverage
from fmri_pipeline.analysis.report.manifest import ContrastManifest


def _stat(tmp_path: Path, name: str = "z.nii.gz", seed: int = 0) -> Path:
    """A map with a real background: only a corner carries data."""
    rng = np.random.default_rng(seed)
    data = np.zeros((16, 16, 16), dtype=np.float32)
    brain = np.zeros((16, 16, 16), dtype=bool)
    brain[2:12, 2:12, 2:12] = True
    data[brain] = rng.standard_normal(int(brain.sum())).astype(np.float32)
    data[3:6, 3:6, 3:6] += 6.0
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _mask(tmp_path: Path, name: str = "mask.nii.gz") -> Path:
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[2:12, 2:12, 2:12] = 1
    path = tmp_path / name
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(path))
    return path


def _manifest(tmp_path: Path, **overrides) -> ContrastManifest:
    base = dict(
        subject="sub-01",
        task="heat",
        contrast_name="heat-warm",
        space="native",
        stat_map=_stat(tmp_path),
        effect_map=None,
        variance_map=None,
        mask=_mask(tmp_path),
        threshold_mode="z",
        z_threshold=2.3,
        fdr_q=0.05,
        cluster_min_voxels=0,
        two_sided=True,
        radiological=False,
        design_matrices=(),
        contrast_vector=None,
        contrast_columns=(),
        included_runs=("run-01",),
        excluded_runs=(),
        bold_paths=(),
        confounds_paths=(),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="auto",
        mask_is_analysis_mask=True,
    )
    base.update(overrides)
    return ContrastManifest(**base)


# --- masking --------------------------------------------------------------


def test_statistic_values_are_taken_inside_the_mask_not_over_the_volume(
    tmp_path: Path,
) -> None:
    # A whole volume is mostly exact background zero. Including it inflates the test
    # count behind every corrected threshold and drags a fitted null toward zero.
    manifest = _manifest(tmp_path)
    stat_img = nib.load(str(manifest.stat_map))
    values, source = subject.masked_stat_values(stat_img, nib.load(str(manifest.mask)))
    assert source == "analysis mask"
    assert values.size == 10**3
    assert values.size < np.asarray(stat_img.get_fdata()).size


def test_a_mask_that_does_not_fit_falls_back_and_says_so(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    wrong = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    _values, source = subject.masked_stat_values(
        nib.load(str(manifest.stat_map)), wrong
    )
    assert "no usable mask" in source


# --- threshold resolution -------------------------------------------------


def test_a_z_mode_contrast_resolves_its_configured_height(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, threshold_mode="z", z_threshold=2.3)
    threshold, label = subject.resolve_threshold(manifest, values=np.array([0.0, 1.0]))
    assert threshold == pytest.approx(2.3)
    assert "uncorrected" in label


def test_an_fdr_contrast_resolves_a_height_from_its_own_p_values(tmp_path: Path) -> None:
    # Previously this produced a contrast section with no panels at all: no map, no
    # glass brain, no cluster table, for a mode the config validates and accepts.
    manifest = _manifest(tmp_path, threshold_mode="fdr", fdr_q=0.05)
    stat_img = nib.load(str(manifest.stat_map))
    values, _ = subject.masked_stat_values(stat_img, nib.load(str(manifest.mask)))
    threshold, label = subject.resolve_threshold(manifest, values=values)
    assert threshold is not None and threshold > 0
    assert "FDR" in label


def test_an_fdr_contrast_that_rejects_nothing_says_so(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, threshold_mode="fdr", fdr_q=1e-12)
    values = np.random.default_rng(0).standard_normal(5_000)
    threshold, label = subject.resolve_threshold(manifest, values=values)
    assert threshold is None
    assert "no voxel survives" in label


def test_threshold_mode_none_resolves_no_height(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, threshold_mode="none")
    threshold, label = subject.resolve_threshold(manifest, values=np.array([0.0, 1.0]))
    assert threshold is None
    assert label == "none"


# --- what the section does with it ----------------------------------------


def _cfg(**kwargs) -> FmriReportConfig:
    base = dict(enabled=True, formats=("png",), include_design_qc=False)
    base.update(kwargs)
    return FmriReportConfig(**base)


def test_an_fdr_contrast_gets_a_thresholded_panel(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, threshold_mode="fdr", fdr_q=0.05)
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=_cfg()
    )
    assert "thresholded" in str(section).lower()


def test_a_contrast_with_no_resolved_threshold_still_gets_calibration(
    tmp_path: Path,
) -> None:
    # The panel that says where a threshold would fall is most valuable precisely
    # when none was applied.
    manifest = _manifest(tmp_path, threshold_mode="none")
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=_cfg()
    )
    assert "calibration" in str(section).lower()


def _titles(section) -> list[str]:
    # Block titles, not the stringified section: the latter carries file paths, and
    # under pytest those contain the test's own name.
    return [str(getattr(block, "title", "")).lower() for block in section.blocks]


def test_the_calibration_panel_leads_no_contrast_into_the_diagnostics_drawer(
    tmp_path: Path,
) -> None:
    # It sits in the contrast section, not the collapsed diagnostics: a reader who
    # never opens that block would read every result above at face value.
    manifest = _manifest(tmp_path)
    contrast = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=_cfg()
    )
    diagnostics = subject.build_diagnostics_section(
        manifest=manifest, out_dir=tmp_path, cfg=_cfg()
    )
    assert any("calibration" in title for title in _titles(contrast))
    assert not any("calibration" in title for title in _titles(diagnostics))
    assert diagnostics.collapsed and not contrast.collapsed


def test_the_cluster_caption_states_the_maps_smoothness(tmp_path: Path) -> None:
    # A cluster-extent count is not comparable to anything without it.
    manifest = _manifest(tmp_path, cluster_min_voxels=10)
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=_cfg()
    )
    text = str(section)
    assert "FWHM" in text
    assert "resels" in text


def test_smoothness_facts_survive_a_map_they_cannot_measure(tmp_path: Path) -> None:
    # An unresolved measurement is not a reason to lose the cluster table.
    tiny = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    assert subject.smoothness_facts(tiny, mask_img=None, cluster_min_voxels=10) == []


# --- smoothness is a property of the noise, not of the signal -------------


def _smoothed_map_with_a_blob(fwhm_mm: float = 6.0, voxel_mm: float = 3.0):
    """A z map of noise smoothed to a known FWHM, with a strong activation blob in it."""
    from scipy import ndimage

    rng = np.random.default_rng(0)
    field = rng.standard_normal((40, 40, 40))
    sigma_voxels = (fwhm_mm / voxel_mm) / np.sqrt(8.0 * np.log(2.0))
    smoothed = ndimage.gaussian_filter(field, sigma=sigma_voxels)
    smoothed /= smoothed.std()
    smoothed[14:26, 14:26, 14:26] += 12.0
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    mask = nib.Nifti1Image(np.ones((40, 40, 40), dtype=np.uint8), affine)
    return nib.Nifti1Image(smoothed.astype(np.float32), affine), mask


def test_the_smoothness_estimate_recovers_the_applied_kernel_from_the_noise() -> None:
    # Ground truth: a field smoothed to 6 mm on a 3 mm grid. Excluding the blob is what
    # makes the estimate a measurement of the smoothing rather than of the signal.
    stat_img, mask = _smoothed_map_with_a_blob()
    noise, source = subject.noise_mask(stat_img, mask_img=mask, threshold=2.3)
    fwhm = coverage.estimate_fwhm(stat_img, mask=noise)
    assert "sub-threshold" in source
    assert float(np.mean(fwhm)) == pytest.approx(6.0, rel=0.15)


def test_including_the_activation_inflates_the_smoothness_estimate() -> None:
    # The error being corrected: signal is spatially structured, and the estimator
    # cannot tell that structure from smoothing.
    stat_img, mask = _smoothed_map_with_a_blob()
    whole, _source = subject.noise_mask(stat_img, mask_img=mask, threshold=None)
    noise, _source = subject.noise_mask(stat_img, mask_img=mask, threshold=2.3)
    assert float(np.mean(coverage.estimate_fwhm(stat_img, mask=whole))) > float(
        np.mean(coverage.estimate_fwhm(stat_img, mask=noise))
    )


def test_an_unthresholded_map_names_its_estimate_an_upper_bound() -> None:
    # There is no height to exclude by, and a number that may be inflated must not be
    # reported as though it were not.
    stat_img, mask = _smoothed_map_with_a_blob()
    _noise, source = subject.noise_mask(stat_img, mask_img=mask, threshold=None)
    assert "upper bound" in source


def test_a_map_that_is_suprathreshold_everywhere_falls_back_to_the_whole_mask() -> None:
    # Excluding every voxel would leave nothing to estimate from at all.
    affine = np.eye(4)
    stat_img = nib.Nifti1Image(np.full((8, 8, 8), 9.0, dtype=np.float32), affine)
    mask = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), affine)
    voxels, source = subject.noise_mask(stat_img, mask_img=mask, threshold=2.3)
    assert voxels is not None and voxels.all()
    assert "upper bound" in source


def test_the_cluster_caption_states_the_search_volume_in_resels(tmp_path: Path) -> None:
    # Every corrected height in this report divides alpha across voxels; the resel
    # count is what says how far that overshoots the family of independent tests.
    manifest = _manifest(tmp_path, cluster_min_voxels=10)
    facts = subject.smoothness_facts(
        nib.load(str(manifest.stat_map)),
        mask_img=nib.load(str(manifest.mask)),
        cluster_min_voxels=10,
        threshold=2.3,
    )
    assert any("search volume" in fact and "resels" in fact for fact in facts)


# --- the anatomical underlay ----------------------------------------------


def test_volume_panels_are_drawn_over_the_discovered_anatomy(tmp_path: Path) -> None:
    # Every volume panel previously passed bg_img=None, so a cluster floated in empty
    # space and could not be judged against grey matter or a ventricle.
    manifest = _manifest(tmp_path)
    background = nib.Nifti1Image(np.ones((16, 16, 16), dtype=np.float32), np.eye(4))
    with patch(
        "fmri_pipeline.analysis.report.figures.stat_maps.stat_map_mosaic"
    ) as mosaic:
        mosaic.side_effect = RuntimeError("stop after the call is recorded")
        subject.build_contrast_section(
            manifest=manifest, out_dir=tmp_path, cfg=_cfg(), background=background
        )
    assert mosaic.call_args.kwargs["bg_img"] is background


def test_the_report_trims_the_underlay_to_what_was_modelled(tmp_path: Path) -> None:
    # Nilearn picks slice positions across the underlay's extent, so a whole-head T1w
    # put the vertex and the neck in every mosaic and left the brain occupying about
    # half of each tile.
    manifest = _manifest(tmp_path)
    background = nib.Nifti1Image(
        np.ones((60, 60, 60), dtype=np.float32), np.eye(4)
    )
    captured = {}

    def _record(figure, **kwargs):
        captured["bg"] = kwargs.get("bg_img")
        raise RuntimeError("stop after the call is recorded")

    with patch(
        "fmri_pipeline.analysis.report.subject.load_background",
        return_value=(background, "anat.nii.gz"),
    ), patch(
        "fmri_pipeline.analysis.report.figures.stat_maps.stat_map_mosaic",
        side_effect=_record,
    ):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "report.html",
            cfg=_cfg(),
        )

    drawn = captured["bg"]
    assert drawn is not None
    # The mask in this fixture is 10^3 inside a 60^3 underlay, so a trimmed underlay
    # has to be materially smaller than the one that was loaded.
    assert np.prod(drawn.shape[:3]) < np.prod(background.shape[:3])


def test_a_missing_background_is_reported_rather_than_left_to_assumption(
    tmp_path: Path,
) -> None:
    background, reason = subject.load_background(
        deriv_root=tmp_path, manifest=_manifest(tmp_path)
    )
    assert background is None
    assert "no anatomical image" in reason


def test_the_header_names_the_underlay_it_used(tmp_path: Path) -> None:
    section = subject.build_header_section(
        [_manifest(tmp_path)], background_source="/deriv/sub-01/anat/sub-01_T1w.nii.gz"
    )
    assert "sub-01_T1w.nii.gz" in str(section)


# --- the mask claim -------------------------------------------------------


def test_the_header_says_when_the_mask_is_the_one_the_glm_used(tmp_path: Path) -> None:
    section = subject.build_header_section([_manifest(tmp_path)])
    assert "fitted inside" in str(section)


def test_the_header_declines_the_claim_for_a_merely_discovered_mask(
    tmp_path: Path,
) -> None:
    # A discovered mask is a single run's; the fitted one is the intersection. Only
    # the second justifies "voxels outside this were not tested".
    section = subject.build_header_section(
        [_manifest(tmp_path, mask_is_analysis_mask=False)]
    )
    assert "not verified as the fitted mask" in str(section)


# --- the carpet's voxels --------------------------------------------------


def _bold(tmp_path: Path, name: str, n_frames: int = 12) -> Path:
    """A run whose background is noisy rather than exactly zero, as real data is."""
    rng = np.random.default_rng(1)
    data = (0.4 * rng.standard_normal((16, 16, 16, n_frames))).astype(np.float32)
    data[2:12, 2:12, 2:12] += 100.0
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def test_the_carpet_is_built_from_the_analysis_mask_not_the_field_of_view(
    tmp_path: Path,
) -> None:
    # "Mean signal is not exactly zero" admits the whole field of view on any
    # acquisition whose background carries noise, so the panel sampled air while its
    # caption said "as modelled".
    manifest = _manifest(
        tmp_path,
        bold_paths=(_bold(tmp_path, "r1.nii.gz"),),
        included_runs=("run-01",),
    )
    captured = {}

    def _record(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop once the call is recorded")

    with patch(
        "fmri_pipeline.analysis.report.figures.carpet.carpet_figure", _record
    ):
        subject.build_qc_sections(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_dir=tmp_path,
            cfg=_cfg(include_tsnr_qc=False),
        )
    assert captured.get("voxel_source") == "analysis mask"


def test_the_carpet_falls_back_when_no_fitted_mask_was_recorded(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        bold_paths=(_bold(tmp_path, "r2.nii.gz"),),
        included_runs=("run-01",),
        mask_is_analysis_mask=False,
    )
    captured = {}

    def _record(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop once the call is recorded")

    with patch(
        "fmri_pipeline.analysis.report.figures.carpet.carpet_figure", _record
    ):
        subject.build_qc_sections(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_dir=tmp_path,
            cfg=_cfg(include_tsnr_qc=False),
        )
    assert captured.get("voxel_source") == "nonzero mean signal"


# --- units ----------------------------------------------------------------


def test_the_standard_error_colourbar_is_not_labelled_effect(tmp_path: Path) -> None:
    # Same units as the effect, but a different quantity: reusing the effect label
    # named the wrong map on the colourbar.
    manifest = _manifest(tmp_path)
    assert subject._error_units(manifest).startswith("standard error")
    assert "arbitrary BOLD units" in subject._error_units(manifest)


def test_scaled_models_carry_percent_signal_change_into_the_error_units(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path, signal_scaling=True, signal_scaling_mode="voxel-mean"
    )
    assert subject._error_units(manifest) == "standard error (% signal change)"


def test_voxel_mean_scaling_puts_percent_signal_change_on_the_effect_colourbar(
    tmp_path: Path,
) -> None:
    # The units the maps have actually been in all along. Dividing each voxel by its
    # own temporal mean is what makes an effect a percentage of that voxel's baseline.
    manifest = _manifest(
        tmp_path, signal_scaling=True, signal_scaling_mode="voxel-mean"
    )
    assert subject._effect_units(manifest) == "% signal change"


def test_grand_mean_scaling_is_not_called_percent_signal_change(tmp_path: Path) -> None:
    # A different denominator is a different quantity. Both modes answer "scaled", and
    # collapsing them onto one label would put a number on the colourbar that the map
    # does not carry.
    manifest = _manifest(
        tmp_path, signal_scaling=True, signal_scaling_mode="grand-mean"
    )
    assert subject._effect_units(manifest) == "% of the grand mean signal"


def test_an_unrecognised_scaling_mode_declines_to_name_a_denominator(
    tmp_path: Path,
) -> None:
    # Scaled, but by what is not known. A guess here would be worse than the neutral
    # label, because a reader cannot check it against the map.
    manifest = _manifest(tmp_path, signal_scaling=True, signal_scaling_mode="unknown")
    assert subject._effect_units(manifest) == "effect (scaled BOLD units)"
