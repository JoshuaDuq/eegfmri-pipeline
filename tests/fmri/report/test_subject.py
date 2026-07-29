from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.manifest import ContrastManifest


def _bold(tmp_path: Path, name: str, n_frames: int = 20) -> Path:
    rng = np.random.default_rng(0)
    data = (100.0 + rng.standard_normal((6, 6, 6, n_frames))).astype(np.float32)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _stat(tmp_path: Path, name: str, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((12, 12, 12)).astype(np.float32)
    data[4:8, 4:8, 4:8] += 5.0
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _manifest(tmp_path: Path, name: str = "heat-warm", **overrides) -> ContrastManifest:
    base = dict(
        subject="sub-01",
        task="heat",
        contrast_name=name,
        space="native",
        stat_map=_stat(tmp_path, f"{name}_z.nii.gz"),
        effect_map=_stat(tmp_path, f"{name}_eff.nii.gz", 1),
        variance_map=None,
        mask=None,
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
        bold_paths=(_bold(tmp_path, "r1.nii.gz"), _bold(tmp_path, "r2.nii.gz")),
        confounds_paths=(),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="motion+compcor",
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
    with patch(
        "fmri_pipeline.analysis.report.figures.volumes.compute_tsnr"
    ) as mock_tsnr:
        mock_tsnr.side_effect = RuntimeError("stop here")
        subject.build_qc_sections(
            manifests=manifests, deriv_root=tmp_path, out_dir=tmp_path, cfg=_cfg()
        )
    # Two contrasts, one tSNR computation.
    assert mock_tsnr.call_count == 1


def test_qc_returns_a_section_even_when_every_panel_fails(tmp_path: Path) -> None:
    with patch(
        "fmri_pipeline.analysis.report.figures.volumes.compute_tsnr",
        side_effect=RuntimeError("boom"),
    ), patch(
        "fmri_pipeline.analysis.report.figures.carpet.carpet_figure",
        side_effect=RuntimeError("boom"),
    ):
        sections = subject.build_qc_sections(
            manifests=[_manifest(tmp_path)], deriv_root=tmp_path, out_dir=tmp_path,
            cfg=_cfg(),
        )
    assert sections


def test_qc_is_labelled_as_modelled_not_as_preprocessed(tmp_path: Path) -> None:
    sections = subject.build_qc_sections(
        manifests=[_manifest(tmp_path)], deriv_root=tmp_path, out_dir=tmp_path,
        cfg=_cfg(),
    )
    assert "as modelled" in " ".join(str(s) for s in sections).lower()


# --- results --------------------------------------------------------------


def test_a_contrast_section_leads_with_the_dual_coded_panel(tmp_path: Path) -> None:
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path, cfg=_cfg()
    )
    titles = [b.title for b in section.blocks if hasattr(b, "title")]
    assert titles and "dual-coded" in titles[0].lower()


def test_diagnostics_are_collapsed_not_deleted(tmp_path: Path) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path,
        cfg=_cfg(include_unthresholded=True),
    )
    assert section.collapsed is True
    titles = " ".join(b.title for b in section.blocks if hasattr(b, "title")).lower()
    assert "unthresholded" in titles


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
        deriv_root=tmp_path, out_path=out, cfg=_cfg(),
    )
    text = out.read_text()
    assert "heat-warm" in text and "heat-rest" in text


def test_the_document_has_one_qc_section_regardless_of_contrast_count(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path, "a"), _manifest(tmp_path, "b")],
        deriv_root=tmp_path, out_path=out, cfg=_cfg(),
    )
    assert out.read_text().count('id="qc"') == 1


def test_building_a_report_with_no_contrasts_raises_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no contrasts"):
        subject.build_subject_report(
            manifests=[], deriv_root=tmp_path, out_path=tmp_path / "r.html", cfg=_cfg()
        )


# --- cluster peaks --------------------------------------------------------


def test_a_cluster_table_is_produced_with_peak_coordinates(tmp_path: Path) -> None:
    table, peaks = subject.build_cluster_table(
        manifest=_manifest(tmp_path), out_dir=tmp_path
    )
    assert table is not None
    assert len(peaks) >= 1 and len(peaks[0]) == 3


def test_the_table_caption_separates_threshold_from_extent(tmp_path: Path) -> None:
    table, _ = subject.build_cluster_table(
        manifest=_manifest(tmp_path), out_dir=tmp_path
    )
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
    nib.save(
        nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.float32), np.eye(4)), str(flat)
    )
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
