from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.manifest import (
    REPORT_MANIFEST_SCHEMA_VERSION,
    ContrastManifest,
    validate_manifest_artifacts,
)


def _bold(tmp_path: Path, name: str, n_frames: int = 20) -> Path:
    rng = np.random.default_rng(0)
    data = (100.0 + rng.standard_normal((12, 12, 12, n_frames))).astype(np.float32)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _model_series(
    tmp_path: Path,
    name: str,
    *,
    shape: tuple[int, int, int] = (12, 12, 12),
    n_frames: int = 20,
) -> Path:
    time = np.arange(n_frames, dtype=np.float32)
    values = 0.1 * time if "predicted" in name else np.where(time.astype(int) % 2 == 0, 0.25, -0.25)
    data = np.broadcast_to(values, (*shape, n_frames)).copy()
    path = tmp_path / name
    nib.save(
        nib.Nifti1Image(data, np.eye(4)),
        str(path),
    )
    return path


def _stat(tmp_path: Path, name: str, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((12, 12, 12)).astype(np.float32)
    data[4:8, 4:8, 4:8] += 5.0
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _mask(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    nib.save(
        nib.Nifti1Image(np.ones((12, 12, 12), dtype=np.uint8), np.eye(4)),
        str(path),
    )
    return path


def _manifest(tmp_path: Path, name: str = "heat-warm", **overrides) -> ContrastManifest:
    base = dict(
        schema_version=REPORT_MANIFEST_SCHEMA_VERSION,
        subject="sub-01",
        task="heat",
        contrast_name=name,
        space="native",
        stat_map=_stat(tmp_path, f"{name}_z.nii.gz"),
        effect_map=_stat(tmp_path, f"{name}_eff.nii.gz", 1),
        variance_map=None,
        mask=_mask(tmp_path, f"{name}_mask.nii.gz"),
        threshold_mode="z",
        z_threshold=2.3,
        fdr_q=0.05,
        cluster_min_voxels=0,
        two_sided=True,
        radiological=False,
        design_matrices=(),
        contrast_vector=None,
        contrast_columns=(),
        included_runs=("run-01", "run-02"),
        excluded_runs=(("run-03", "fewer events than the contrast requires"),),
        bold_paths=(
            _bold(tmp_path, "run-01_bold.nii.gz"),
            _bold(tmp_path, "run-02_bold.nii.gz"),
        ),
        confounds_paths=(),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="motion+compcor",
        mask_is_analysis_mask=True,
        residual_paths=(
            _model_series(tmp_path, "run-01_residual.nii.gz"),
            _model_series(tmp_path, "run-02_residual.nii.gz"),
        ),
        predicted_paths=(
            _model_series(tmp_path, "run-01_predicted.nii.gz"),
            _model_series(tmp_path, "run-02_predicted.nii.gz"),
        ),
        retained_frame_indices=(tuple(range(20)), tuple(range(20))),
    )
    base.update(overrides)
    return ContrastManifest(**base)


def _cfg(**kwargs) -> FmriReportConfig:
    base = dict(enabled=True, formats=("png",), include_design_qc=False)
    base.update(kwargs)
    return FmriReportConfig(**base)


# --- header ---------------------------------------------------------------


def test_the_header_names_every_excluded_run_and_its_reason(tmp_path: Path) -> None:
    section = subject.build_header_section([_manifest(tmp_path)])
    text = str(section)
    assert "run-03" in text and "fewer events" in text


def test_the_header_states_the_acquisition_parameters(tmp_path: Path) -> None:
    text = str(subject.build_header_section([_manifest(tmp_path)]))
    assert "2 s" in text  # TR
    assert "6 mm FWHM" in text


def test_the_header_declines_to_claim_percent_signal_change(tmp_path: Path) -> None:
    text = str(subject.build_header_section([_manifest(tmp_path)]))
    assert "arbitrary BOLD units" in text


# --- QC -------------------------------------------------------------------


def test_qc_is_built_once_for_a_subject_with_several_contrasts(tmp_path: Path) -> None:
    manifests = [_manifest(tmp_path, "a"), _manifest(tmp_path, "b")]
    with patch("fmri_pipeline.analysis.report.figures.volumes.compute_tsnr") as mock_tsnr:
        mock_tsnr.side_effect = RuntimeError("stop here")
        subject.build_qc_sections(
            manifests=manifests, deriv_root=tmp_path, out_dir=tmp_path, cfg=_cfg()
        )
    # Two contrasts, one tSNR computation.
    assert mock_tsnr.call_count == 1


def test_qc_returns_a_section_even_when_every_panel_fails(tmp_path: Path) -> None:
    with (
        patch(
            "fmri_pipeline.analysis.report.figures.volumes.compute_tsnr",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "fmri_pipeline.analysis.report.figures.carpet.carpet_figure",
            side_effect=RuntimeError("boom"),
        ),
    ):
        sections = subject.build_qc_sections(
            manifests=[_manifest(tmp_path)],
            deriv_root=tmp_path,
            out_dir=tmp_path,
            cfg=_cfg(),
        )
    assert sections


def test_qc_is_labelled_as_modelled_not_as_preprocessed(tmp_path: Path) -> None:
    sections = subject.build_qc_sections(
        manifests=[_manifest(tmp_path)],
        deriv_root=tmp_path,
        out_dir=tmp_path,
        cfg=_cfg(),
    )
    assert "as modelled" in " ".join(str(s) for s in sections).lower()


def test_qc_uses_the_exact_retained_frame_indices(tmp_path: Path) -> None:
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        retained_frame_indices=(tuple(range(1, 20)), source.retained_frame_indices[1]),
    )
    captured = {}

    def _record(_series, *, sample_mask=None):
        captured.setdefault("sample_masks", []).append(sample_mask)
        raise RuntimeError("stop once the call is recorded")

    with patch("fmri_pipeline.analysis.report.figures.carpet.standardise_carpet", _record):
        subject.build_qc_sections(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_dir=tmp_path,
            cfg=_cfg(include_tsnr_qc=False),
        )

    first_mask = captured["sample_masks"][0]
    assert first_mask.tolist() == [False, *([True] * 19)]


# --- results --------------------------------------------------------------


def test_a_contrast_section_leads_with_the_dual_coded_panel(tmp_path: Path) -> None:
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path, cfg=_cfg()
    )
    titles = [b.title for b in section.blocks if hasattr(b, "title")]
    assert titles and "dual-coded" in titles[0].lower()


def test_diagnostics_are_collapsed_not_deleted(tmp_path: Path) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path),
        deriv_root=tmp_path,
        out_dir=tmp_path,
        cfg=_cfg(include_unthresholded=True),
    )
    assert section.collapsed is True
    titles = " ".join(b.title for b in section.blocks if hasattr(b, "title")).lower()
    assert "unthresholded" in titles


def test_diagnostics_include_the_exact_model_fit_measurement_table(
    tmp_path: Path,
) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path),
        deriv_root=tmp_path,
        out_dir=tmp_path,
        cfg=_cfg(include_unthresholded=False),
    )

    table = section.blocks[0]
    assert table.title == "Model-fit measurements by run"
    assert "Median residual ACF(1)" in table.html
    assert table.tsv_path == (
        tmp_path / "plots" / "contrast-heat-warm" / "model_fit_measurements.tsv"
    )
    assert table.tsv_path.is_file()
    assert "unwhitened model-response" in table.caption


def test_diagnostics_include_the_exact_model_response_residual_carpet(
    tmp_path: Path,
) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path),
        deriv_root=tmp_path,
        out_dir=tmp_path,
        cfg=_cfg(include_unthresholded=False),
    )

    carpet = section.blocks[1]
    assert carpet.title == "Model-response residual carpet"
    assert carpet.path == (tmp_path / "plots" / "contrast-heat-warm" / "residual_carpet.png")
    assert carpet.path.is_file()
    assert "Y − Xβ" in carpet.caption
    assert "unwhitened model-response" in carpet.caption


def test_diagnostics_include_the_pooled_residual_standard_deviation_map(
    tmp_path: Path,
) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path),
        deriv_root=tmp_path,
        out_dir=tmp_path,
        cfg=_cfg(include_unthresholded=False),
    )

    residual_sd = section.blocks[2]
    artifact_dir = tmp_path / "plots" / "contrast-heat-warm"
    assert residual_sd.title == "Pooled residual standard deviation"
    assert residual_sd.path == artifact_dir / "residual_standard_deviation.png"
    assert residual_sd.path.is_file()
    assert (artifact_dir / "residual_standard_deviation.nii.gz").is_file()
    assert "Σ(e − ē)² / N" in residual_sd.caption
    assert "unwhitened model-response" in residual_sd.caption
    assert "No criterion is applied" in residual_sd.caption


def test_diagnostics_include_residual_autocorrelation_at_acquired_lags(
    tmp_path: Path,
) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path),
        deriv_root=tmp_path,
        out_dir=tmp_path,
        cfg=_cfg(include_unthresholded=False),
    )

    residual_acf = section.blocks[3]
    artifact_dir = tmp_path / "plots" / "contrast-heat-warm"
    assert residual_acf.title == "Residual autocorrelation by run"
    assert residual_acf.path == artifact_dir / "residual_autocorrelation.svg"
    assert residual_acf.path.is_file()
    assert (artifact_dir / "residual_autocorrelation.tsv").is_file()
    assert "original acquired-frame indices differ by k" in residual_acf.caption
    assert "unwhitened model-response" in residual_acf.caption
    assert "No criterion is applied" in residual_acf.caption


def test_each_contrast_gets_its_own_anchor(tmp_path: Path) -> None:
    a = subject.build_contrast_section(
        manifest=_manifest(tmp_path, "heat-warm"), out_dir=tmp_path, cfg=_cfg()
    )
    b = subject.build_contrast_section(
        manifest=_manifest(tmp_path, "heat-rest"), out_dir=tmp_path, cfg=_cfg()
    )
    assert a.slug != b.slug


def test_the_document_covers_every_contrast(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path, "heat-warm"), _manifest(tmp_path, "heat-rest")],
        deriv_root=tmp_path,
        out_path=out,
        cfg=_cfg(),
    )
    text = out.read_text()
    assert "heat-warm" in text and "heat-rest" in text


def test_the_document_has_one_qc_section_regardless_of_contrast_count(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path, "a"), _manifest(tmp_path, "b")],
        deriv_root=tmp_path,
        out_path=out,
        cfg=_cfg(),
    )
    assert out.read_text().count('id="qc"') == 1


def test_building_a_report_with_no_contrasts_raises_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no contrasts"):
        subject.build_subject_report(
            manifests=[], deriv_root=tmp_path, out_path=tmp_path / "r.html", cfg=_cfg()
        )


def test_report_assembly_rejects_a_manifest_outside_the_contract(
    tmp_path: Path,
) -> None:
    manifest = replace(_manifest(tmp_path), mask_is_analysis_mask=False)

    with pytest.raises(ValueError, match="fitted analysis mask"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_a_missing_artifact(tmp_path: Path) -> None:
    manifest = replace(_manifest(tmp_path), stat_map=tmp_path / "missing_z.nii.gz")

    with pytest.raises(FileNotFoundError, match="stat_map"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_mixed_subject_task_inputs(tmp_path: Path) -> None:
    first = _manifest(tmp_path, "heat-warm")
    second = replace(_manifest(tmp_path, "heat-rest"), task="rest")

    with pytest.raises(ValueError, match="same subject and task"):
        subject.build_subject_report(
            manifests=[first, second],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_mixed_run_inputs(tmp_path: Path) -> None:
    first = _manifest(tmp_path, "heat-warm")
    second_source = _manifest(tmp_path, "heat-rest")
    second = replace(
        second_source,
        included_runs=("run-02", "run-01"),
        bold_paths=tuple(reversed(second_source.bold_paths)),
    )

    with pytest.raises(ValueError, match="same run inputs"):
        subject.build_subject_report(
            manifests=[first, second],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_misaligned_image_geometry(tmp_path: Path) -> None:
    wrong_mask = tmp_path / "wrong_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
        str(wrong_mask),
    )
    manifest = replace(_manifest(tmp_path), mask=wrong_mask)

    with pytest.raises(ValueError, match="mask geometry"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_a_missing_model_fit_series(tmp_path: Path) -> None:
    manifest = replace(
        _manifest(tmp_path),
        residual_paths=(tmp_path / "missing_residual.nii.gz",) * 2,
    )

    with pytest.raises(FileNotFoundError, match="residual_paths"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_model_fit_shape_disagreement(tmp_path: Path) -> None:
    wrong_prediction = _model_series(
        tmp_path,
        "wrong_predicted.nii.gz",
        n_frames=19,
    )
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        predicted_paths=(wrong_prediction, source.predicted_paths[1]),
    )

    with pytest.raises(ValueError, match="matching shapes"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_model_fit_spatial_misalignment(tmp_path: Path) -> None:
    wrong_residual = _model_series(
        tmp_path,
        "wrong_residual.nii.gz",
        shape=(4, 4, 4),
    )
    wrong_prediction = _model_series(
        tmp_path,
        "wrong_predicted.nii.gz",
        shape=(4, 4, 4),
    )
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        residual_paths=(wrong_residual, source.residual_paths[1]),
        predicted_paths=(wrong_prediction, source.predicted_paths[1]),
    )

    with pytest.raises(ValueError, match="BOLD geometry"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_wrong_model_fit_timepoints(tmp_path: Path) -> None:
    short_residual = _model_series(
        tmp_path,
        "short_residual.nii.gz",
        n_frames=19,
    )
    short_prediction = _model_series(
        tmp_path,
        "short_predicted.nii.gz",
        n_frames=19,
    )
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        residual_paths=(short_residual, source.residual_paths[1]),
        predicted_paths=(short_prediction, source.predicted_paths[1]),
    )

    with pytest.raises(ValueError, match="retained timepoints"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_retained_indices_that_disagree_with_fit_series(
    tmp_path: Path,
) -> None:
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        retained_frame_indices=(tuple(range(19)), source.retained_frame_indices[1]),
    )

    with pytest.raises(ValueError, match="retained frame indices"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_report_assembly_rejects_confounds_length_that_disagrees_with_bold(
    tmp_path: Path,
) -> None:
    design_path = tmp_path / "run-01_design.tsv"
    confounds_path = tmp_path / "run-01_confounds.tsv"
    pd.DataFrame({"constant": np.ones(20)}).to_csv(design_path, sep="\t", index=False)
    pd.DataFrame({"trans_x": np.zeros(19)}).to_csv(confounds_path, sep="\t", index=False)
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        included_runs=("run-01",),
        bold_paths=(source.bold_paths[0],),
        design_matrices=(design_path,),
        confounds_paths=(confounds_path,),
        residual_paths=(source.residual_paths[0],),
        predicted_paths=(source.predicted_paths[0],),
        retained_frame_indices=(source.retained_frame_indices[0],),
    )

    with pytest.raises(ValueError, match="confounds.*BOLD timepoints"):
        subject.build_subject_report(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_path=tmp_path / "r.html",
            cfg=_cfg(),
        )


def test_unused_confounds_do_not_define_censoring_for_a_none_strategy(
    tmp_path: Path,
) -> None:
    confounds_path = tmp_path / "unused_confounds.tsv"
    outliers = np.zeros(20)
    outliers[0] = 1
    pd.DataFrame({"motion_outlier00": outliers}).to_csv(
        confounds_path,
        sep="\t",
        index=False,
    )
    source = _manifest(tmp_path)
    manifest = replace(
        source,
        included_runs=("run-01",),
        bold_paths=(source.bold_paths[0],),
        confounds_paths=(confounds_path,),
        residual_paths=(source.residual_paths[0],),
        predicted_paths=(source.predicted_paths[0],),
        retained_frame_indices=(source.retained_frame_indices[0],),
        confound_strategy="none",
    )

    validate_manifest_artifacts(manifest)


# --- cluster peaks --------------------------------------------------------


def test_a_cluster_table_is_produced_with_peak_coordinates(tmp_path: Path) -> None:
    table, peaks = subject.build_cluster_table(manifest=_manifest(tmp_path), out_dir=tmp_path)
    assert table is not None
    assert len(peaks) >= 1
    label, coord = peaks[0]
    assert label.isdigit()
    assert len(coord) == 3


def test_the_table_caption_separates_threshold_from_extent(tmp_path: Path) -> None:
    table, _ = subject.build_cluster_table(manifest=_manifest(tmp_path), out_dir=tmp_path)
    caption = table.caption.lower()
    assert "height threshold" in caption
    assert "cluster-level significan" not in caption


def test_an_extent_filter_is_named_as_a_display_filter(tmp_path: Path) -> None:
    manifest = replace(_manifest(tmp_path), cluster_min_voxels=5)
    table, _ = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    assert "not familywise-error-corrected" in table.caption.lower()


def test_the_cluster_tsv_is_written_beside_the_report(tmp_path: Path) -> None:
    subject.build_cluster_table(manifest=_manifest(tmp_path), out_dir=tmp_path)
    assert (tmp_path / "plots" / "contrast-heat-warm" / "clusters.tsv").exists()


def test_an_empty_map_yields_no_peaks_rather_than_an_error(tmp_path: Path) -> None:
    flat = tmp_path / "flat.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.float32), np.eye(4)), str(flat))
    manifest = replace(_manifest(tmp_path), stat_map=flat)
    _table, peaks = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    assert peaks == ()


# --- methods --------------------------------------------------------------


def test_methods_records_the_confound_strategy(tmp_path: Path) -> None:
    assert "motion+compcor" in str(subject.build_methods_section([_manifest(tmp_path)]))


def test_methods_states_the_threshold_actually_applied(tmp_path: Path) -> None:
    assert "2.30" in str(subject.build_methods_section([_manifest(tmp_path)]))


def test_methods_states_the_orientation_convention(tmp_path: Path) -> None:
    assert "neurological" in str(subject.build_methods_section([_manifest(tmp_path)]))


def test_methods_names_an_extent_filter_as_a_display_filter(tmp_path: Path) -> None:
    manifest = replace(_manifest(tmp_path), cluster_min_voxels=20)
    text = str(subject.build_methods_section([manifest]))
    assert "not familywise-error-corrected" in text


# --- glass brain is only defined against the MNI schematic -----------------


def test_a_native_space_contrast_gets_no_glass_brain(tmp_path: Path) -> None:
    """The projection is drawn on a fixed MNI schematic.

    A native-space map projected onto it lands on anatomy it does not correspond
    to, which is an error rather than an approximation.
    """
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path, space="native"), out_dir=tmp_path, cfg=_cfg()
    )
    titles = [getattr(b, "title", "") for b in section.blocks]
    assert not any("Glass brain" in t for t in titles)


def test_the_missing_glass_brain_is_explained_rather_than_silent(tmp_path: Path) -> None:
    """A panel that vanishes without a word reads as a rendering failure."""
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path, space="native"), out_dir=tmp_path, cfg=_cfg()
    )
    text = " ".join(getattr(b, "text", "") for b in section.blocks)
    assert "glass brain" in text.lower()
    assert "mni" in text.lower()


def test_an_mni_contrast_still_gets_a_glass_brain(tmp_path: Path) -> None:
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path, space="mni"), out_dir=tmp_path, cfg=_cfg()
    )
    titles = [getattr(b, "title", "") for b in section.blocks]
    assert any("Glass brain" in t for t in titles)


def test_the_space_guard_is_case_and_whitespace_tolerant() -> None:
    assert subject.supports_glass_brain("MNI")
    assert subject.supports_glass_brain(" mni ")
    assert not subject.supports_glass_brain("T1w")
    assert not subject.supports_glass_brain("")


# --- cluster coordinates name their space ----------------------------------


def test_native_space_cluster_coordinates_are_not_left_to_read_as_mni(
    tmp_path: Path,
) -> None:
    """An unlabelled X/Y/Z column in an fMRI cluster table reads as MNI by convention."""
    table, _peaks = subject.build_cluster_table(
        manifest=_manifest(tmp_path, space="native"), out_dir=tmp_path
    )
    assert table is not None
    assert "not MNI" in table.caption
    assert "native" in table.caption.lower()


def test_mni_cluster_coordinates_say_so(tmp_path: Path) -> None:
    table, _peaks = subject.build_cluster_table(
        manifest=_manifest(tmp_path, space="mni"), out_dir=tmp_path
    )
    assert table is not None
    assert "MNI152" in table.caption


def test_the_coordinate_space_label_distinguishes_mni_from_everything_else() -> None:
    assert "MNI152" in subject.coordinate_space_label("mni")
    assert "not MNI" in subject.coordinate_space_label("native")
    assert "not MNI" in subject.coordinate_space_label("T1w")


def test_the_analysis_mask_reaches_the_colour_limit(tmp_path: Path) -> None:
    """Wiring test: the mask is only useful if the panels actually receive it.

    A stat map is mostly background, so a limit computed without the mask lands too
    low and the panel saturates. The figure-level fix is inert unless the mask is
    threaded through from the manifest.
    """
    mask_path = tmp_path / "mask.nii.gz"
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

    manifest = _manifest(tmp_path, mask=mask_path)
    with patch("fmri_pipeline.analysis.report.figures.stat_maps.stat_map_mosaic") as mosaic:
        mosaic.return_value = None
        subject.build_contrast_section(manifest=manifest, out_dir=tmp_path, cfg=_cfg())

    assert mosaic.called
    passed = [call.kwargs.get("mask_img") for call in mosaic.call_args_list]
    assert any(m is not None for m in passed), "no panel received the analysis mask"


# --- peaks key to clusters, not to table rows ------------------------------


def _clustered_stat(tmp_path: Path, name: str) -> Path:
    """Two clusters, the first with a secondary local peak."""
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[4:12, 4:12, 4:12] = 3.0
    data[5, 5, 5] = 9.0
    data[10, 10, 10] = 8.0
    data[16, 16, 16] = 6.0
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def test_subpeak_rows_do_not_become_their_own_markers(tmp_path: Path) -> None:
    """nilearn writes sub-peaks as rows '1a', '1b' beneath their cluster.

    Counting every row as a peak numbered the markers 1..N while the table's own
    Cluster ID column read 1, 1a, 2 -- so marker 2 pointed at a sub-peak of cluster
    1 while the reader looked up cluster 2. The caption claims the two key to each
    other, so they have to.
    """
    manifest = _manifest(
        tmp_path, "clustered", stat_map=_clustered_stat(tmp_path, "clusters_fixture.nii.gz")
    )
    table, peaks = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    assert table is not None
    # Two clusters, three rows: the sub-peak must not be among the markers.
    assert len(peaks) == 2


def test_peak_labels_are_the_tables_own_cluster_ids(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path, "labelled", stat_map=_clustered_stat(tmp_path, "labels_fixture.nii.gz")
    )
    _table, peaks = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    assert [label for label, _coord in peaks] == ["1", "2"]


def test_the_analysis_mask_reaches_the_tsnr_computation(tmp_path: Path) -> None:
    """Wiring test: without the mask the median is taken over rim voxels too.

    `tsnr > 0` picks up partial-volume voxels at the brain's edge, which sit at
    very low tSNR and pull the reported median down.
    """
    mask_path = tmp_path / "qc_mask.nii.gz"
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

    manifest = _manifest(tmp_path, "masked-qc", mask=mask_path)
    with patch("fmri_pipeline.analysis.report.figures.volumes.compute_tsnr") as compute:
        compute.side_effect = RuntimeError("stop after the call is inspected")
        subject.build_qc_sections(
            manifests=[manifest],
            deriv_root=tmp_path,
            out_dir=tmp_path,
            cfg=_cfg(include_carpet_qc=False),
        )

    assert compute.called
    assert compute.call_args.kwargs.get("mask_img") is not None


# --- signatures ------------------------------------------------------------


def _signature_tsv(directory: Path) -> Path:
    path = directory / "signature_expression.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "signature\tdot\tcosine\tpearson_r\tn_voxels\tweight_path\n"
        "NPS\t12.0\t0.31\t0.28\t9000\t/w/nps.nii.gz\n"
        "SIIPS\t-4.0\t-0.12\t-0.10\t9000\t/w/siips.nii.gz\n",
        encoding="utf-8",
    )
    return path


def test_a_signature_section_is_built_from_the_expression_table(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "sig")
    _signature_tsv(Path(manifest.stat_map).parent)
    section = subject.build_signature_section(manifest=manifest, out_dir=tmp_path, cfg=_cfg())
    assert section is not None
    titles = " ".join(getattr(b, "title", "") for b in section.blocks).lower()
    assert "signature" in titles


def test_no_configured_signatures_yields_no_section(tmp_path: Path) -> None:
    """The stock configuration has none; an empty section would be noise."""
    section = subject.build_signature_section(
        manifest=_manifest(tmp_path, "nosig"), out_dir=tmp_path, cfg=_cfg()
    )
    assert section is None


def test_the_signature_section_reaches_the_document(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "docsig")
    _signature_tsv(Path(manifest.stat_map).parent)
    out = tmp_path / "report.html"
    subject.build_subject_report(
        manifests=[manifest], deriv_root=tmp_path, out_path=out, cfg=_cfg()
    )
    assert "NPS" in out.read_text()


# --- cluster peaks: what the row actually says ----------------------------
#
# nilearn's table carries a peak z and a size: how strong the evidence is and how far
# it spreads, but not how large the effect *is*. Two peaks at z = 4 can differ tenfold
# in percent signal change, and only one of them is worth reporting.


def _peaks() -> tuple:
    return (("1", (6.0, 6.0, 6.0)), ("2", (2.0, 2.0, 2.0)))


def _frame(cluster_ids=("1", "2")):
    import pandas as pd

    return pd.DataFrame(
        {
            "Cluster ID": list(cluster_ids),
            "X": [6.0, 2.0],
            "Y": [6.0, 2.0],
            "Z": [6.0, 2.0],
            "Peak Stat": [5.1, 3.2],
            "Cluster Size (mm3)": [800, 300],
        }
    )


def test_the_cluster_table_carries_the_effect_at_each_peak(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, signal_scaling=True, signal_scaling_mode="voxel-mean")
    frame, notes = subject.enrich_cluster_frame(_frame(), manifest=manifest, peaks=_peaks())
    effect_columns = [c for c in frame.columns if c.startswith("Peak effect")]
    assert effect_columns, "the peak effect column is missing"
    assert "% signal change" in effect_columns[0]
    assert any("peak effect" in note for note in notes)


def test_the_peak_effect_is_the_value_at_the_peak_voxel(tmp_path: Path) -> None:
    # Nearest voxel, not interpolation: a peak is a voxel, and interpolating between
    # it and its neighbours reports a number no voxel in the map carries.
    manifest = _manifest(tmp_path)
    effect = np.asarray(nib.load(str(manifest.effect_map)).get_fdata())
    frame, _notes = subject.enrich_cluster_frame(_frame(), manifest=manifest, peaks=_peaks())
    column = next(c for c in frame.columns if c.startswith("Peak effect"))
    assert frame[column].iloc[0] == pytest.approx(effect[6, 6, 6])


def test_the_cluster_table_carries_the_standard_error_at_each_peak(
    tmp_path: Path,
) -> None:
    # What distinguishes a large effect from an imprecise one.
    variance = tmp_path / "var.nii.gz"
    nib.save(
        nib.Nifti1Image(np.full((12, 12, 12), 4.0, dtype=np.float32), np.eye(4)),
        str(variance),
    )
    manifest = _manifest(tmp_path, variance_map=variance)
    frame, _notes = subject.enrich_cluster_frame(_frame(), manifest=manifest, peaks=_peaks())
    assert "Peak SE" in frame.columns
    # The standard error is the root of the variance.
    assert frame["Peak SE"].iloc[0] == pytest.approx(2.0)


def test_a_subpeak_row_gets_no_peak_columns(tmp_path: Path) -> None:
    # nilearn writes secondary local maxima as 1a, 1b. They are not clusters, and
    # `peaks` holds one entry per cluster.
    manifest = _manifest(tmp_path)
    frame, _notes = subject.enrich_cluster_frame(
        _frame(cluster_ids=("1", "1a")), manifest=manifest, peaks=(("1", (6.0, 6.0, 6.0)),)
    )
    column = next(c for c in frame.columns if c.startswith("Peak effect"))
    assert frame[column].iloc[0] != ""
    assert frame[column].iloc[1] == ""


def test_a_native_space_table_says_why_it_has_no_region_names(tmp_path: Path) -> None:
    # An MNI atlas read at a native coordinate names whatever structure sits at those
    # millimetres in a different brain, and the output looks entirely correct.
    manifest = _manifest(tmp_path, space="native")
    frame, notes = subject.enrich_cluster_frame(_frame(), manifest=manifest, peaks=_peaks())
    assert "Region" not in frame.columns
    assert any("no anatomical labels" in note for note in notes)


def test_an_mni_table_carries_region_names_when_an_atlas_is_configured(
    tmp_path: Path,
) -> None:
    labels = np.zeros((12, 12, 12), dtype=np.int16)
    labels[4:9, 4:9, 4:9] = 1
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(atlas_path))

    from fmri_pipeline.analysis.report import atlas as atlas_module

    labeller = atlas_module.load_atlas(labels_img=atlas_path)
    manifest = _manifest(tmp_path, space="mni")
    frame, notes = subject.enrich_cluster_frame(
        _frame(), manifest=manifest, peaks=_peaks(), labeller=labeller
    )
    assert list(frame["Region"]) == ["1", "unlabelled"]
    assert any("atlas.nii.gz" in note for note in notes)


def test_an_atlas_is_not_read_for_a_native_space_contrast(tmp_path: Path) -> None:
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((12, 12, 12), dtype=np.int16), np.eye(4)), str(atlas_path))
    cfg = _cfg(atlas_labels_img=str(atlas_path))
    assert subject.resolve_labeller(_manifest(tmp_path, space="native"), cfg) is None
    assert subject.resolve_labeller(_manifest(tmp_path, space="mni"), cfg) is not None


def test_a_table_without_peaks_is_returned_unchanged(tmp_path: Path) -> None:
    frame = _frame()
    returned, notes = subject.enrich_cluster_frame(frame, manifest=_manifest(tmp_path), peaks=())
    assert returned is frame
    assert notes == []


def test_the_displayed_table_rounds_to_the_precision_the_grid_supports() -> None:
    # nilearn writes a peak coordinate as 54.884781, claiming about a thousandth of a
    # millimetre on a 3 mm grid.
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Cluster ID": ["1"],
            "X": [54.884781],
            "Y": [-6.542206],
            "Z": [30.777603],
            "Peak Stat": [6.074831216163631],
            "Peak effect (% signal change)": [0.41983116688701305],
            "Peak SE": [0.06890086135368591],
        }
    )
    shown = subject.for_display(frame)
    assert shown["X"].iloc[0] == pytest.approx(55.0)
    assert shown["Peak Stat"].iloc[0] == pytest.approx(6.07)
    assert shown["Peak effect (% signal change)"].iloc[0] == pytest.approx(0.42)
    # Significant figures, not decimal places: an effect of 0.42 and an error of
    # 0.0689 need different decimal counts to carry the same information.
    assert shown["Peak SE"].iloc[0] == pytest.approx(0.0689)


def test_rounding_for_display_leaves_the_source_frame_untouched() -> None:
    # The TSV is written from the unrounded frame: it is read by machines, and
    # rounding it would lose real information.
    import pandas as pd

    frame = pd.DataFrame({"Cluster ID": ["1"], "X": [54.884781]})
    subject.for_display(frame)
    assert frame["X"].iloc[0] == pytest.approx(54.884781)


def test_rounding_leaves_text_columns_alone() -> None:
    import pandas as pd

    frame = pd.DataFrame({"Cluster ID": ["1a"], "Region": ["Left insula"], "X": [1.234]})
    shown = subject.for_display(frame)
    assert shown["Region"].iloc[0] == "Left insula"
    assert shown["Cluster ID"].iloc[0] == "1a"


def test_rounding_never_touches_an_identifier_or_a_count() -> None:
    # An integer in this table is a cluster's number or its size in cubic
    # millimetres. Three significant figures turned cluster 2 into "2.0" and a
    # 5,211 mm3 cluster into 5,210 -- a fabricated measurement.
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Cluster ID": [2],
            "Cluster Size (mm3)": [5211],
            "Peak Stat": [6.074831],
        }
    )
    shown = subject.for_display(frame)
    assert shown["Cluster ID"].iloc[0] == 2
    assert shown["Cluster Size (mm3)"].iloc[0] == 5211
    assert shown["Peak Stat"].iloc[0] == pytest.approx(6.07)


# --- run consistency in the document --------------------------------------


def _run_level_maps(tmp_path: Path, n_runs: int = 4):
    """Per-run effect and variance volumes matching the _stat fixture's geometry."""
    rng = np.random.default_rng(3)
    effect = rng.normal(0.0, 0.05, (12, 12, 12, n_runs)).astype(np.float32)
    effect[4:8, 4:8, 4:8, :] = 0.4
    variance = np.full((12, 12, 12, n_runs), 0.01, dtype=np.float32)
    effect_path = tmp_path / "perrun_effect.nii.gz"
    variance_path = tmp_path / "perrun_variance.nii.gz"
    nib.save(nib.Nifti1Image(effect, np.eye(4)), str(effect_path))
    nib.save(nib.Nifti1Image(variance, np.eye(4)), str(variance_path))
    return effect_path, variance_path


def test_the_contrast_section_carries_a_run_consistency_panel(tmp_path: Path) -> None:
    effect_path, variance_path = _run_level_maps(tmp_path)
    manifest = _manifest(
        tmp_path,
        run_effect_map=effect_path,
        run_variance_map=variance_path,
        included_runs=("run-01", "run-02", "run-03", "run-04"),
    )
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path / "out", cfg=_cfg()
    )
    titles = [getattr(block, "title", "") for block in section.blocks]
    assert "Run consistency at each peak" in titles


def test_a_contrast_without_run_level_maps_simply_has_no_such_panel(
    tmp_path: Path,
) -> None:
    # A single-run contrast has nothing to compare, and a manifest written before
    # run-level maps existed carries none. Neither is a fault, so neither gets a note.
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path / "out", cfg=_cfg()
    )
    titles = [getattr(block, "title", "") for block in section.blocks]
    assert "Run consistency at each peak" not in titles
    assert not any(
        "run" in getattr(block, "text", "").lower()
        and "consist" in getattr(block, "text", "").lower()
        for block in section.blocks
    )


def test_missing_run_level_files_cost_the_panel_and_not_the_section(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        run_effect_map=tmp_path / "absent_effect.nii.gz",
        run_variance_map=tmp_path / "absent_variance.nii.gz",
    )
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path / "out", cfg=_cfg()
    )
    assert section.blocks


# --- the summary ----------------------------------------------------------
#
# Deciding whether a subject's result is usable meant scrolling eight sections: the
# censoring is in the motion table, the tSNR in a QC panel, the survivor count in a
# calibration figure. Across ninety subjects that is the difference between triage
# and reading ninety documents.


def test_facts_are_collected_in_the_order_they_were_measured() -> None:
    facts = subject.SummaryFacts()
    facts.add("Runs modelled", 6)
    facts.add("Median tSNR", "60.0")
    assert facts.items == (("Runs modelled", "6"), ("Median tSNR", "60.0"))


def test_a_repeated_label_replaces_rather_than_duplicates() -> None:
    # Several contrasts write the same subject-level fact.
    facts = subject.SummaryFacts()
    facts.add("Runs modelled", 6)
    facts.add("Runs modelled", 5)
    assert facts.items == (("Runs modelled", "5"),)


def test_an_empty_summary_produces_no_section() -> None:
    # An empty block would claim the document had been summarised.
    assert subject.build_summary_section(subject.SummaryFacts()) is None


def test_the_summary_says_it_scores_nothing() -> None:
    # Which censoring fraction makes a subject usable is a study's decision.
    facts = subject.SummaryFacts()
    facts.add("Median tSNR", "60.0")
    section = subject.build_summary_section(facts)
    notes = " ".join(getattr(b, "text", "") for b in section.blocks)
    assert "scored against" in notes


def test_the_contrast_section_reports_its_threshold_and_survivors(
    tmp_path: Path,
) -> None:
    facts = subject.SummaryFacts()
    subject.build_contrast_section(
        manifest=_manifest(tmp_path),
        out_dir=tmp_path / "out",
        cfg=_cfg(),
        facts=facts,
    )
    labels = dict(facts.items)
    assert any("height threshold" in key for key in labels)
    survivors = next(k for k in labels if "voxels above the threshold" in k)
    assert " of " in labels[survivors]


def test_the_summary_appears_near_the_top_of_the_document(tmp_path: Path) -> None:
    out = tmp_path / "report" / "r.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path)],
        deriv_root=tmp_path,
        out_path=out,
        cfg=_cfg(),
    )
    html = out.read_text(encoding="utf-8")
    assert html.index("At a glance") < html.index("Contrast:")


def test_a_contrast_section_without_a_summary_still_builds(tmp_path: Path) -> None:
    # facts is optional: existing callers pass nothing.
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path / "out", cfg=_cfg()
    )
    assert section.blocks


def test_a_summary_fact_recorded_late_in_a_contrast_section_survives(
    tmp_path: Path,
) -> None:
    # A local named `facts` once shadowed the SummaryFacts parameter with a list of
    # smoothness strings, so every panel after the cluster table raised
    # AttributeError -- swallowed by the panel guard, which removed the panel from
    # the document and left only a log line.
    facts = subject.SummaryFacts()
    subject.build_contrast_section(
        manifest=_manifest(tmp_path),
        out_dir=tmp_path / "out",
        cfg=_cfg(),
        facts=facts,
        deriv_root=tmp_path,
    )
    assert isinstance(facts, subject.SummaryFacts)
    assert facts.items


# --- vector where vector belongs -------------------------------------------
#
# style.figure_format existed, was documented, and was tested, and nothing called it:
# every panel was rasterised at 150 dpi regardless of the `dense` flag each caller
# was carefully setting on its html.Figure.


def test_a_line_figure_is_embedded_as_vector(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    path = subject._save(figure, out_dir=tmp_path, stem="line", formats=("png",), dense=False)
    assert path.suffix == ".svg"
    # The configured format is still written beside it.
    assert (tmp_path / "line.png").exists()


def test_a_dense_figure_stays_raster(tmp_path: Path) -> None:
    # Wrapping the same pixels in base64 inside an SVG costs more bytes while only
    # sharpening the axis text.
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure.add_subplot(111).imshow(np.random.default_rng(0).random((40, 40)))
    path = subject._save(figure, out_dir=tmp_path, stem="dense", formats=("png",), dense=True)
    assert path.suffix == ".png"
    assert not (tmp_path / "dense.svg").exists()


def test_the_vector_figures_of_a_real_document_are_the_line_figures(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report" / "r.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path)],
        deriv_root=tmp_path,
        out_path=out,
        cfg=_cfg(),
    )
    written = {p.name for p in (out.parent).rglob("*.svg")}
    # The calibration panel is a histogram with curves: vector.
    assert any("threshold_calibration" in name for name in written)
    # The mosaics are dense image layers: they must not have produced one.
    assert not any("stat_thresholded" in name for name in written)


def test_calibration_caption_does_not_blame_autocorrelation():
    """This study's median residual ACF(1) is 0.05-0.07; it cannot widen a null to 1.51."""
    from fmri_pipeline.analysis.report.subject import CALIBRATION_CAPTION

    assert "autocorrelation" in CALIBRATION_CAPTION.lower()
    assert "unmodelled autocorrelation is routinely" not in CALIBRATION_CAPTION.lower()
    assert "measurement, not an assumption" in CALIBRATION_CAPTION


def test_threshold_caption_names_the_p_floor_when_it_binds():
    from fmri_pipeline.analysis.report import inference
    from fmri_pipeline.analysis.report.subject import _threshold_table_caption

    summary = inference.SignFlipSummary(
        height=7.02, survivors=38, global_p=0.0606, p_floor=0.0606,
        n_runs=6, n_patterns=32, observed_max=8.87,
    )
    caption = _threshold_table_caption(
        inference.threshold_context(
            np.random.default_rng(0).standard_normal(5000),
            applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=True,
            sign_flip=summary,
        )
    )
    assert "0.061" in caption
    assert "smallest value this test can return" in caption
    assert "no map-level p below" in caption


def test_threshold_caption_drops_the_stale_no_correction_claim():
    """A sign-flip height IS familywise-corrected over voxels."""
    from fmri_pipeline.analysis.report import inference
    from fmri_pipeline.analysis.report.subject import _threshold_table_caption

    caption = _threshold_table_caption(
        inference.threshold_context(
            np.random.default_rng(0).standard_normal(5000),
            applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=True,
            sign_flip=inference.SignFlipSummary(
                height=7.02, survivors=38, global_p=0.0606, p_floor=0.0606,
                n_runs=6, n_patterns=32, observed_max=8.87,
            ),
        )
    )
    assert "none of these heights is familywise-corrected" not in caption
    assert "familywise-corrected over voxels, not over extent" in caption


def test_sign_flip_summary_is_none_without_manifest_scalars():
    from fmri_pipeline.analysis.report.subject import _sign_flip_summary

    class _M:
        sign_flip_fwe_height = None
        sign_flip_n_runs = None

    assert _sign_flip_summary(_M()) is None


def test_sign_flip_summary_recomputes_a_missing_floor():
    """An older manifest may carry the height without the floor; derive it."""
    from fmri_pipeline.analysis.report.subject import _sign_flip_summary

    class _M:
        sign_flip_fwe_height = 7.02
        sign_flip_fwe_survivors = 38
        sign_flip_global_p = 0.0606
        sign_flip_p_floor = None
        sign_flip_n_patterns = 32
        sign_flip_n_runs = 6
        sign_flip_observed_max = 8.87

    assert _sign_flip_summary(_M()).p_floor == pytest.approx(2 / 33)


def _cluster_frame(rows):
    """A minimal clusters table in nilearn's own column shape."""
    import pandas as pd

    return pd.DataFrame(
        [
            {"Cluster ID": cid, "X": x, "Y": y, "Z": z, "Peak Stat": stat}
            for cid, x, y, z, stat in rows
        ]
    )


def test_peaks_are_ordered_by_absolute_stat():
    """nilearn orders clusters by signed stat, so a map's largest effect can be last.

    On sub-0001 the strongest cluster peaks at z = -8.81 over 138,213 mm3 and sorts
    below every positive cluster, which kept it out of every panel that caps.
    """
    from fmri_pipeline.analysis.report.subject import _cluster_peaks

    frame = _cluster_frame(
        [(1, -50, 23, -2, 6.26), (2, 55, -7, 31, 6.07), (177, 13, -28, 58, -8.81)]
    )
    assert [label for label, _ in _cluster_peaks(frame)] == ["177", "1", "2"]


def test_subpeak_rows_are_still_skipped_when_ordering():
    from fmri_pipeline.analysis.report.subject import _cluster_peaks

    frame = _cluster_frame(
        [(1, -50, 23, -2, 6.26), ("1a", -44, 20, -2, 5.10), (2, 55, -7, 31, -9.0)]
    )
    assert [label for label, _ in _cluster_peaks(frame)] == ["2", "1"]


def test_peak_order_is_stable_without_a_stat_column():
    """A frame lacking Peak Stat must keep the table's own order, not an arbitrary one."""
    import pandas as pd

    from fmri_pipeline.analysis.report.subject import _cluster_peaks

    frame = pd.DataFrame(
        [
            {"Cluster ID": 1, "X": 0, "Y": 0, "Z": 0},
            {"Cluster ID": 2, "X": 1, "Y": 1, "Z": 1},
        ]
    )
    assert [label for label, _ in _cluster_peaks(frame)] == ["1", "2"]


def test_coordinates_travel_with_their_own_peak():
    """Reordering must not shear labels away from coordinates."""
    from fmri_pipeline.analysis.report.subject import _cluster_peaks

    frame = _cluster_frame(
        [(1, -50, 23, -2, 3.0), (2, 55, -7, 31, -9.0), (3, 10, 10, 10, 5.0)]
    )
    assert _cluster_peaks(frame) == (
        ("2", (55.0, -7.0, 31.0)),
        ("3", (10.0, 10.0, 10.0)),
        ("1", (-50.0, 23.0, -2.0)),
    )


def test_mni_companion_is_discovered_beside_the_native_map(tmp_path):
    """Deterministic naming: _space-MNI152NLin2009cAsym goes in front of _stat-."""
    from fmri_pipeline.analysis.report.subject import _mni_companion

    stem = "sub-01_task-heat_contrast-c"
    native = tmp_path / f"{stem}_stat-z_score_abc.nii.gz"
    native.touch()
    for quantity in ("z_score", "effect_size", "effect_variance"):
        (tmp_path / f"{stem}_space-MNI152NLin2009cAsym_stat-{quantity}_abc.nii.gz").touch()

    found = _mni_companion(native)
    assert found is not None
    assert found.stat_map.name.endswith("space-MNI152NLin2009cAsym_stat-z_score_abc.nii.gz")
    assert found.effect_map is not None and found.variance_map is not None
    assert found.space == "mni"


def test_no_companion_when_the_mni_z_map_is_absent(tmp_path):
    from fmri_pipeline.analysis.report.subject import _mni_companion

    native = tmp_path / "sub-01_task-heat_contrast-c_stat-z_score_abc.nii.gz"
    native.touch()
    assert _mni_companion(native) is None


def test_companion_tolerates_missing_effect_maps(tmp_path):
    """A z map alone still buys MNI coordinates and atlas labels."""
    from fmri_pipeline.analysis.report.subject import _mni_companion

    stem = "sub-01_task-heat_contrast-c"
    native = tmp_path / f"{stem}_stat-z_score_abc.nii.gz"
    native.touch()
    (tmp_path / f"{stem}_space-MNI152NLin2009cAsym_stat-z_score_abc.nii.gz").touch()

    found = _mni_companion(native)
    assert found is not None
    assert found.effect_map is None and found.variance_map is None


def test_a_native_stat_map_is_not_its_own_companion(tmp_path):
    """Guards against matching an already-MNI stat map and recursing on itself."""
    from fmri_pipeline.analysis.report.subject import _mni_companion

    stem = "sub-01_task-heat_contrast-c"
    already_mni = tmp_path / f"{stem}_space-MNI152NLin2009cAsym_stat-z_score_abc.nii.gz"
    already_mni.touch()
    assert _mni_companion(already_mni) is None


def _manifest_with_companion(tmp_path):
    """A fitted manifest whose contrast also has a standard-space companion on disk."""
    import dataclasses

    import nibabel as nib
    from fmri_pipeline.analysis.report.manifest import read_manifest, write_report_manifest

    stem = "sub-01_task-heat_contrast-c"
    rng = np.random.default_rng(0)
    for name in (
        f"{stem}_stat-z_score_abc.nii.gz",
        f"{stem}_space-MNI152NLin2009cAsym_stat-z_score_abc.nii.gz",
        f"{stem}_space-MNI152NLin2009cAsym_stat-effect_size_abc.nii.gz",
        f"{stem}_space-MNI152NLin2009cAsym_stat-effect_variance_abc.nii.gz",
    ):
        nib.save(
            nib.Nifti1Image(rng.standard_normal((6, 6, 6)).astype(np.float32), np.eye(4)),
            str(tmp_path / name),
        )
    native = tmp_path / f"{stem}_stat-z_score_abc.nii.gz"
    written = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=native,
        mask=native,
        mask_is_analysis_mask=True,
        run_meta={
            "analysis_space": "T1w",
            "tr": 2.0,
            "included_bold_paths": [str(tmp_path / "sub-01_run-01_bold.nii.gz")],
            "included_confounds_paths": [],
            "retained_frame_indices": [[0]],
        },
        contrast_cfg=SimpleNamespace(hrf_model="spm"),
        residual_paths=(native,),
        predicted_paths=(native,),
        run_effect_map=None,
        sign_flip_null_tsv=None,
    )
    return read_manifest(written)


def test_companion_manifest_points_at_the_standard_space_maps(tmp_path):
    from fmri_pipeline.analysis.report.subject import companion_manifest

    companion = companion_manifest(_manifest_with_companion(tmp_path))
    assert companion is not None
    assert companion.space == "mni"
    assert "space-MNI152NLin2009cAsym" in Path(companion.stat_map).name
    assert "space-MNI152NLin2009cAsym" in Path(companion.effect_map).name


def test_companion_gets_its_own_plot_directory(tmp_path):
    """Sharing a slug would have the companion's plots overwrite the fitted ones."""
    from fmri_pipeline.analysis.report.subject import _slug, companion_manifest

    manifest = _manifest_with_companion(tmp_path)
    assert _slug(companion_manifest(manifest)) != _slug(manifest)


def test_companion_drops_every_fitted_space_artifact(tmp_path):
    """Inherited, each of these would be read at coordinates from a different fit."""
    from fmri_pipeline.analysis.report.subject import companion_manifest

    companion = companion_manifest(_manifest_with_companion(tmp_path))
    assert companion.run_effect_map is None
    assert companion.run_variance_map is None
    assert companion.sign_flip_null_tsv is None
    assert companion.sign_flip_fwe_height is None
    assert companion.run_influence_tsv is None
    assert companion.bold_paths == ()
    assert companion.residual_paths == ()
    assert companion.predicted_paths == ()
    assert companion.mask is None
    assert companion.mask_is_analysis_mask is False


def test_companion_is_atlas_referable_where_the_fit_is_not(tmp_path):
    """The whole point: the fitted section cannot carry labels and this one can."""
    from fmri_pipeline.analysis.report import atlas
    from fmri_pipeline.analysis.report.subject import (
        cluster_table_source,
        companion_manifest,
    )

    manifest = _manifest_with_companion(tmp_path)
    companion = companion_manifest(manifest)
    assert not atlas.atlas_applies_to(cluster_table_source(manifest).space)
    assert atlas.atlas_applies_to(cluster_table_source(companion).space)


def test_no_companion_manifest_without_companion_maps(tmp_path):
    import nibabel as nib
    from fmri_pipeline.analysis.report.manifest import read_manifest, write_report_manifest
    from fmri_pipeline.analysis.report.subject import companion_manifest

    native = tmp_path / "sub-01_task-heat_contrast-c_stat-z_score_abc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(native))
    manifest = read_manifest(
        write_report_manifest(
            contrast_dir=tmp_path,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=native,
            mask=native,
            mask_is_analysis_mask=True,
            run_meta={
                "analysis_space": "T1w",
                "tr": 2.0,
                "included_bold_paths": [str(tmp_path / "b.nii.gz")],
                "included_confounds_paths": [],
                "retained_frame_indices": [[0]],
            },
            contrast_cfg=SimpleNamespace(hrf_model="spm"),
            residual_paths=(native,),
            predicted_paths=(native,),
        )
    )
    assert companion_manifest(manifest) is None


def test_survivor_line_states_the_expected_count_beside_the_observed():
    """The observed count alone reads as a result; the pair is the measurement.

    On sub-0001 the applied height yields 8,463 voxels where the map's own fitted
    null predicts 8,001 -- a 5.8% excess, which the bare count does not convey.
    """
    from fmri_pipeline.analysis.report.subject import survivor_summary

    line = survivor_summary(
        surviving=8463, n_voxels=50626, expected_under_fitted_null=8001.4
    )
    assert "8,463" in line
    assert "50,626" in line
    assert "16.72%" in line
    assert "8,001" in line
    assert "fitted null" in line


def test_survivor_line_without_a_fitted_null_states_only_what_it_has():
    from fmri_pipeline.analysis.report.subject import survivor_summary

    line = survivor_summary(
        surviving=8463, n_voxels=50626, expected_under_fitted_null=None
    )
    assert "8,463 of 50,626" in line
    assert "fitted null" not in line


def test_survivor_line_survives_an_empty_mask():
    from fmri_pipeline.analysis.report.subject import survivor_summary

    assert "0" in survivor_summary(
        surviving=0, n_voxels=0, expected_under_fitted_null=None
    )


def test_familywise_line_carries_the_floor_when_it_binds():
    from fmri_pipeline.analysis.report import inference
    from fmri_pipeline.analysis.report.subject import familywise_summary

    line = familywise_summary(
        inference.SignFlipSummary(
            height=7.02, survivors=38, global_p=0.0606, p_floor=0.0606,
            n_runs=6, n_patterns=32, observed_max=8.87,
        )
    )
    assert "7.02" in line and "38" in line
    assert "0.061" in line
    assert "floor" in line.lower()


def test_familywise_line_omits_the_floor_note_when_it_does_not_bind():
    from fmri_pipeline.analysis.report import inference
    from fmri_pipeline.analysis.report.subject import familywise_summary

    line = familywise_summary(
        inference.SignFlipSummary(
            height=6.0, survivors=120, global_p=0.008, p_floor=0.0155,
            n_runs=8, n_patterns=128, observed_max=9.1,
        )
    )
    assert "6.00" in line and "120" in line
    assert "at its floor" not in line.lower()


def test_the_glass_brain_caps_its_peak_markers():
    """557 numbered markers cover the projection they are drawn on.

    Markers exist so a reader can key the table's strongest rows to the picture.
    Past a handful they stop doing that and start hiding the map, which the native
    sections never revealed because a native fit gets no glass brain at all.
    """
    from fmri_pipeline.analysis.report.subject import GLASS_BRAIN_MAX_MARKERS, _marker_peaks

    peaks = tuple((str(i), (float(i), 0.0, 0.0)) for i in range(557))
    shown = _marker_peaks(peaks)
    assert len(shown) == GLASS_BRAIN_MAX_MARKERS
    assert shown[0][0] == "0"


def test_marker_cap_leaves_a_short_table_alone():
    from fmri_pipeline.analysis.report.subject import _marker_peaks

    peaks = tuple((str(i), (float(i), 0.0, 0.0)) for i in range(4))
    assert len(_marker_peaks(peaks)) == 4


def test_marker_caption_says_how_many_are_drawn():
    from fmri_pipeline.analysis.report.subject import _marker_caption

    assert _marker_caption(4, 4) == "Markers number the peaks in the cluster table below."
    text = _marker_caption(10, 557)
    assert "10" in text and "557" in text
    assert "strongest" in text


def test_marker_caption_is_empty_without_peaks():
    from fmri_pipeline.analysis.report.subject import _marker_caption

    assert _marker_caption(0, 0) == ""


def _run_effect_map(tmp_path, name, per_run_means):
    data = np.stack(
        [np.full((12, 12, 12), v, dtype=np.float32) for v in per_run_means], axis=-1
    )
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def test_run_contributions_render_as_a_table_not_a_figure(tmp_path: Path) -> None:
    """Six runs and four numbers each is a table; a chart would lose precision."""
    manifest = _manifest(
        tmp_path,
        run_effect_map=_run_effect_map(tmp_path, "perrun.nii.gz", [-0.08, 0.06]),
    )
    block = subject.build_run_contribution_block(manifest=manifest, out_dir=tmp_path)
    assert isinstance(block, subject.html.Table)


def test_run_contributions_state_the_offset_per_run(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        run_effect_map=_run_effect_map(tmp_path, "perrun2.nii.gz", [-0.08, 0.06]),
    )
    block = subject.build_run_contribution_block(manifest=manifest, out_dir=tmp_path)
    assert "run-01" in block.html and "run-02" in block.html
    assert "-0.08" in block.html and "0.06" in block.html


def test_run_contributions_merge_the_influence_table(tmp_path: Path) -> None:
    influence = tmp_path / "influence.tsv"
    pd.DataFrame(
        {
            "dropped_run": ["run-01", "run-02"],
            "survivors": [7108, 8003],
            "delta": [-1367, -472],
            "max_abs_z": [8.18, 7.68],
            "correlation": [0.941, 0.913],
        }
    ).to_csv(influence, sep="\t", index=False)

    manifest = _manifest(
        tmp_path,
        run_effect_map=_run_effect_map(tmp_path, "perrun3.nii.gz", [-0.08, 0.06]),
        run_influence_tsv=influence,
    )
    block = subject.build_run_contribution_block(manifest=manifest, out_dir=tmp_path)
    assert "-1367" in block.html and "0.941" in block.html
    assert "Mean effect over mask" in block.html


def test_run_contributions_absent_without_any_measurement(tmp_path: Path) -> None:
    assert (
        subject.build_run_contribution_block(
            manifest=_manifest(tmp_path), out_dir=tmp_path
        )
        is None
    )


def test_run_contributions_score_no_run(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        run_effect_map=_run_effect_map(tmp_path, "perrun4.nii.gz", [-0.08, 0.06]),
    )
    caption = subject.build_run_contribution_block(
        manifest=manifest, out_dir=tmp_path
    ).caption.lower()
    for word in ("outlier", "exclude", "fail", "reject", "bad run"):
        assert word not in caption
    assert "measurement, not a fault" in caption
