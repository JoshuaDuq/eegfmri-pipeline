"""Threshold resolution, masking, and the anatomical underlay in the subject report."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import subject
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
