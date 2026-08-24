from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.cohort import (
    CohortReportConfig,
    CohortReportInputs,
    CohortThresholdConfig,
    build_cohort_report,
    cohort_assumption_note,
    effective_two_sided,
    resolve_group_threshold,
    cohort_report_config_from_mapping,
    cohort_threshold_config_from_mapping,
    validate_cohort_map,
)


def _image(data: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))


def _write_image(path: Path, data: np.ndarray, dtype: Any = np.float32) -> Path:
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), np.eye(4)), path)
    return path


def _write_image64(path: Path, data: np.ndarray) -> Path:
    return _write_image(path, data, dtype=np.float64)


def test_cohort_report_config_rejects_unknown_or_invalid_values() -> None:
    config = cohort_report_config_from_mapping(
        {
            "include_design_correlation": False,
            "include_input_map_concordance": False,
            "cluster_table_max_rows": 8,
        }
    )
    assert config.include_design_correlation is False
    assert config.include_input_map_concordance is False
    assert config.cluster_table_max_rows == 8
    with pytest.raises(ValueError, match="formats"):
        CohortReportConfig(formats=("pdf",)).validate()
    with pytest.raises(TypeError, match="enabled"):
        CohortReportConfig(enabled=1).validate()
    with pytest.raises(TypeError, match="cluster_table_max_rows"):
        CohortReportConfig(cluster_table_max_rows=10.0).validate()
    with pytest.raises(ValueError, match="cluster_table_max_rows"):
        CohortReportConfig(cluster_table_max_rows=0).validate()
    with pytest.raises(ValueError, match="atlas_labels_img is unset"):
        CohortReportConfig(atlas_labels_tsv="labels.tsv").validate()
    with pytest.raises(ValueError, match="Unknown.*report"):
        cohort_report_config_from_mapping({"enabeld": True})
    with pytest.raises(ValueError, match="Unknown.*threshold"):
        cohort_threshold_config_from_mapping({"fdr_q": 0.05})


def test_cohort_threshold_config_validates_inferential_choices() -> None:
    CohortThresholdConfig(
        height_control="fdr",
        alpha=0.05,
        cluster_min_voxels=0,
        two_sided=True,
        min_distance_mm=8.0,
    ).validate()

    with pytest.raises(ValueError, match="height_control"):
        CohortThresholdConfig(height_control="cluster-fwe").validate()
    with pytest.raises(ValueError, match="alpha"):
        CohortThresholdConfig(alpha=0.0).validate()
    with pytest.raises(TypeError, match="alpha"):
        CohortThresholdConfig(alpha="0.05").validate()
    with pytest.raises(ValueError, match="cluster_min_voxels"):
        CohortThresholdConfig(cluster_min_voxels=-1).validate()
    with pytest.raises(TypeError, match="min_distance_mm"):
        CohortThresholdConfig(min_distance_mm="8.0").validate()


def test_f_contrast_is_one_sided_even_when_t_tests_are_two_sided() -> None:
    assert effective_two_sided(stat_type="t", configured=True) is True
    assert effective_two_sided(stat_type="t", configured=False) is False
    assert effective_two_sided(stat_type="F", configured=True) is False


def test_permutation_assumption_follows_the_tested_variate_not_model_name(
    tmp_path: Path,
) -> None:
    design = pd.DataFrame({"intercept": np.ones(6), "cov_age": np.linspace(-1.0, 1.0, 6)})
    inputs = CohortReportInputs(
        task="pain",
        model="one-sample",
        output_name="age_association",
        stat_type="t",
        output_dir=tmp_path,
        design_matrix=design,
        contrast_spec="cov_age",
        input_manifest=pd.DataFrame({"subject": [f"sub-{index:04d}" for index in range(6)]}),
        metadata={},
        saved_maps={},
        design_outputs={},
        n_permutations=5000,
        permutation_two_sided=True,
        permutation_random_state=42,
    )

    note = cohort_assumption_note(inputs)

    assert "permutes design rows" in note
    assert "sign-flips participant effects" not in note


def test_cohort_maps_require_one_exact_finite_binary_mask_grid() -> None:
    image = _image(np.ones((5, 5, 5)))
    invalid_mask = _image(np.full((5, 5, 5), 0.5))
    with pytest.raises(ValueError, match="finite binary"):
        validate_cohort_map(image=image, mask_image=invalid_mask, label="z score")

    shifted_mask = nib.Nifti1Image(np.ones((5, 5, 5)), np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="same affine"):
        validate_cohort_map(image=image, mask_image=shifted_mask, label="z score")

    nonfinite = np.ones((5, 5, 5))
    nonfinite[2, 2, 2] = np.nan
    with pytest.raises(ValueError, match="finite inside"):
        validate_cohort_map(
            image=_image(nonfinite),
            mask_image=image,
            label="z score",
        )


def test_resolve_group_threshold_uses_nilearn_fdr_and_fixed_uncorrected_height() -> None:
    values = np.zeros((9, 9, 9), dtype=float)
    values[3:6, 3:6, 3:6] = 8.0
    stat_img = _image(values)
    mask_img = _image(np.ones_like(values))

    fdr = resolve_group_threshold(
        stat_img=stat_img,
        mask_img=mask_img,
        config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
        stat_type="t",
    )
    assert np.isfinite(fdr.threshold)
    assert fdr.threshold > 0
    assert "FDR q = 0.05" in fdr.label

    fixed = resolve_group_threshold(
        stat_img=stat_img,
        mask_img=mask_img,
        config=CohortThresholdConfig(height_control="none", uncorrected_z_threshold=3.09),
        stat_type="t",
    )
    assert fixed.threshold == pytest.approx(3.09)
    assert "uncorrected" in fixed.label

    extent_filtered = resolve_group_threshold(
        stat_img=stat_img,
        mask_img=mask_img,
        config=CohortThresholdConfig(
            height_control="none",
            uncorrected_z_threshold=3.09,
            cluster_min_voxels=28,
        ),
        stat_type="t",
    )
    assert np.count_nonzero(extent_filtered.image.get_fdata()) == 0


def test_build_cohort_report_writes_a_self_contained_scientific_report(
    tmp_path: Path,
) -> None:
    shape = (11, 11, 11)
    mask = np.zeros(shape)
    mask[1:-1, 1:-1, 1:-1] = 1
    z_values = np.zeros(shape)
    z_values[3:7, 3:7, 3:7] = 5.5
    z_values[7:9, 7:9, 7:9] = -4.5
    effect = z_values / 10.0
    variance = np.full(shape, 0.04)
    logp = np.zeros(shape)
    logp[3:7, 3:7, 3:7] = 2.0

    maps = {
        "z_score": _write_image64(tmp_path / "z.nii.gz", z_values),
        "effect_size": _write_image64(tmp_path / "effect.nii.gz", effect),
        "effect_variance": _write_image64(tmp_path / "variance.nii.gz", variance),
        "analysis_mask": _write_image64(tmp_path / "mask.nii.gz", mask),
        "permutation_logp_max_t": _write_image64(tmp_path / "logp.nii.gz", logp),
    }
    design = pd.DataFrame(
        {
            "intercept": np.ones(6),
            "cov_age": np.linspace(-1.0, 1.0, 6),
            "cov_age_squared": np.linspace(-1.0, 1.0, 6) ** 2,
            "cov_group": np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        }
    )
    input_paths = []
    for index in range(1, 7):
        subject_values = effect + (index - 3.5) * np.indices(shape)[0] / 100.0
        input_paths.append(
            str(_write_image(tmp_path / f"sub-{index:04d}_pain.nii.gz", subject_values))
        )
    manifest = pd.DataFrame(
        {
            "subject": [f"sub-{index:04d}" for index in range(1, 7)],
            "contrast_name": ["pain"] * 6,
            "condition_label": ["pain"] * 6,
            "map_path": input_paths,
        }
    )
    inputs = CohortReportInputs(
        task="pain",
        model="one-sample",
        output_name="pain_group_mean",
        stat_type="t",
        output_dir=tmp_path,
        design_matrix=design,
        contrast_spec="intercept",
        input_manifest=manifest,
        metadata={"model": "one-sample", "input_contrast_names": ["pain"]},
        saved_maps=maps,
        design_outputs={
            "design_condition_number": "1.2",
            "design_max_vif": "1.0",
            "design_max_vif_regressor": "cov_age",
            "design_contrast_efficiency": "0.166667",
        },
        n_permutations=5000,
        permutation_two_sided=True,
        permutation_random_state=42,
    )

    report_path = build_cohort_report(
        inputs=inputs,
        report_config=CohortReportConfig(
            enabled=True,
            html_report=True,
            formats=("png", "svg"),
            embed_images=True,
        ),
        threshold_config=CohortThresholdConfig(
            height_control="none",
            uncorrected_z_threshold=3.09,
            alpha=0.05,
            two_sided=True,
        ),
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert (
        report_path.name == "group_task-pain_model-one-sample_contrast-pain-group-mean_report.html"
    )
    assert "Cohort fMRI report" in report_text
    assert "Second-level design matrix" in report_text
    assert "Regressor correlation" in report_text
    # The contrast rides in the design panel, on the axis of the columns it acts on,
    # rather than standing alone as a second block.
    assert "compute_contrast" in report_text
    assert "Analysis mask" in report_text
    assert "Group effect" in report_text
    assert "Standard error" in report_text
    assert "Thresholded z statistic" in report_text
    assert "Leading parametric clusters" in report_text
    assert "Input-map spatial concordance" in report_text
    assert "Max-T permutation evidence" in report_text
    assert "Max-T corrected peaks" in report_text
    assert "signed z" in report_text
    assert "5000" in report_text
    assert "two-sided" in report_text
    assert "smallest attainable p = 0.0002" in report_text
    assert "random seed 42" in report_text
    assert "symmetric about zero" in report_text
    assert "data:image/" in report_text
    assert 'href="plots/input_map_concordance.tsv"' in report_text

    assert "True-discovery proportion" in report_text
    assert "Leave-one-participant-out influence" in report_text
    assert "cluster_level_inference" in report_text

    plots_dir = tmp_path / "report" / "plots"
    assert (plots_dir / "true_discovery_proportion.svg").exists()
    assert (plots_dir / "leave_one_out_influence.tsv").exists()
    assert (plots_dir / "second_level_design_matrix.svg").exists()
    assert (plots_dir / "second_level_design_correlation.svg").exists()
    assert not (plots_dir / "second_level_contrast.svg").exists()
    assert (plots_dir / "group_standard_error.nii.gz").exists()
    assert (plots_dir / "clusters.tsv").exists()
    assert (plots_dir / "input_map_concordance.tsv").exists()
    assert (plots_dir / "max_t_peaks.tsv").exists()
    assert (tmp_path / "input_manifest.tsv").exists()


def test_disabled_cohort_report_writes_nothing(tmp_path: Path) -> None:
    inputs = CohortReportInputs(
        task="pain",
        model="one-sample",
        output_name="pain_group_mean",
        stat_type="t",
        output_dir=tmp_path,
        design_matrix=pd.DataFrame({"intercept": [1.0, 1.0]}),
        contrast_spec="intercept",
        input_manifest=pd.DataFrame({"subject": ["sub-0001", "sub-0002"], "map_path": ["a", "b"]}),
        metadata={},
        saved_maps={},
        design_outputs={},
    )

    with pytest.raises(ValueError, match="enabled"):
        build_cohort_report(
            inputs=inputs,
            report_config=CohortReportConfig(enabled=False),
            threshold_config=CohortThresholdConfig(),
        )


def test_t_cohort_report_requires_effect_and_variance_artifacts(tmp_path: Path) -> None:
    values = np.ones((5, 5, 5))
    inputs = CohortReportInputs(
        task="pain",
        model="one-sample",
        output_name="pain_group_mean",
        stat_type="t",
        output_dir=tmp_path,
        design_matrix=pd.DataFrame({"intercept": [1.0, 1.0]}),
        contrast_spec="intercept",
        input_manifest=pd.DataFrame({"subject": ["sub-0001", "sub-0002"], "map_path": ["a", "b"]}),
        metadata={},
        saved_maps={
            "z_score": _write_image64(tmp_path / "z.nii.gz", values),
            "analysis_mask": _write_image64(tmp_path / "mask.nii.gz", values),
        },
        design_outputs={},
    )

    with pytest.raises(ValueError, match="effect_size"):
        build_cohort_report(
            inputs=inputs,
            report_config=CohortReportConfig(),
            threshold_config=CohortThresholdConfig(),
        )


def test_f_cohort_report_omits_signed_estimate_panels(tmp_path: Path) -> None:
    values = np.zeros((9, 9, 9), dtype=float)
    values[3:6, 3:6, 3:6] = 5.0
    maps = {
        "z_score": _write_image64(tmp_path / "z.nii.gz", values),
        "analysis_mask": _write_image64(tmp_path / "mask.nii.gz", np.ones_like(values)),
    }
    design = pd.DataFrame(
        {
            "condition_a": [1.0, 0.0, 1.0, 0.0],
            "condition_b": [0.0, 1.0, 0.0, 1.0],
        }
    )
    inputs = CohortReportInputs(
        task="pain",
        model="repeated-measures",
        output_name="condition_omnibus",
        stat_type="F",
        output_dir=tmp_path,
        design_matrix=design,
        contrast_spec=np.eye(2),
        input_manifest=pd.DataFrame(
            {
                "subject": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
                "map_path": ["a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz"],
            }
        ),
        metadata={},
        saved_maps=maps,
        design_outputs={},
    )

    report_path = build_cohort_report(
        inputs=inputs,
        report_config=CohortReportConfig(
            formats=("png",),
            include_input_map_concordance=False,
        include_leave_one_out_influence=False,
        include_residual_diagnostics=False,
        ),
        threshold_config=CohortThresholdConfig(
            height_control="none",
            uncorrected_z_threshold=3.09,
            two_sided=True,
        ),
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "omnibus F contrast has no single signed effect" in report_text
    assert "Group effect" not in report_text
    assert "Standard error" not in report_text


def test_cohort_report_omits_empty_thresholded_brain_panels(tmp_path: Path) -> None:
    values = np.zeros((9, 9, 9), dtype=float)
    values[3:6, 3:6, 3:6] = 5.0
    inputs = CohortReportInputs(
        task="pain",
        model="repeated-measures",
        output_name="condition_omnibus",
        stat_type="F",
        output_dir=tmp_path,
        design_matrix=pd.DataFrame(
            {
                "condition_a": [1.0, 0.0, 1.0, 0.0],
                "condition_b": [0.0, 1.0, 0.0, 1.0],
            }
        ),
        contrast_spec=np.eye(2),
        input_manifest=pd.DataFrame(
            {
                "subject": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
                "map_path": ["a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz"],
            }
        ),
        metadata={},
        saved_maps={
            "z_score": _write_image64(tmp_path / "z.nii.gz", values),
            "analysis_mask": _write_image64(tmp_path / "mask.nii.gz", np.ones_like(values)),
        },
        design_outputs={},
    )

    report_path = build_cohort_report(
        inputs=inputs,
        report_config=CohortReportConfig(
            formats=("png",),
            include_input_map_concordance=False,
        include_leave_one_out_influence=False,
        include_residual_diagnostics=False,
            include_unthresholded=False,
        ),
        threshold_config=CohortThresholdConfig(
            height_control="none",
            uncorrected_z_threshold=3.09,
            cluster_min_voxels=28,
        ),
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "No voxel survived" in report_text
    assert "Glass-brain overview" not in report_text
    assert "Thresholded z statistic" not in report_text


def test_cohort_design_panel_carries_its_contrast_and_colour_scale() -> None:
    """A one-column design is a flat block; the panel has to say what the block is worth.

    Drawn through the report's figure layer, so the weight sits on the column it lands
    on and the colour has a stated scale. The bare Nilearn plotter gave neither, and a
    constant design collapsed matplotlib's autoscale onto the colormap floor -- the
    value 1.0 rendered in the colour a reader takes for zero.
    """
    import matplotlib.pyplot as plt

    from fmri_pipeline.analysis.report import cohort

    figure = cohort._design_matrix_figure(
        pd.DataFrame({"intercept": np.ones(13)}),
        contrast_spec="intercept",
    )
    try:
        annotations = [text.get_text() for axes in figure.axes for text in axes.texts]
        assert "+1" in annotations
        assert any(axes.get_ylabel() == "column scaled to its own peak" for axes in figure.axes)
    finally:
        plt.close(figure)


def _t_model_inputs(tmp_path: Path) -> CohortReportInputs:
    """A minimal one-sample t cohort with every map the estimate panels need."""
    shape = (11, 11, 11)
    mask = np.zeros(shape)
    mask[1:-1, 1:-1, 1:-1] = 1
    # Graded rather than flat: a block of one repeated value puts Nilearn's FDR height
    # exactly on the peak, leaving nothing above it to draw or tabulate.
    z_values = np.zeros(shape)
    z_values[3:7, 3:7, 3:7] = np.linspace(3.5, 6.0, 64).reshape(4, 4, 4)
    # The threshold is two-sided by default, so give the negative tail something to find.
    z_values[7:9, 7:9, 7:9] = np.linspace(-5.0, -3.6, 8).reshape(2, 2, 2)
    variance = np.full(shape, 0.04)
    logp = np.zeros(shape)
    logp[3:7, 3:7, 3:7] = 2.0
    logp[8, 8, 8] = 1.0
    # float64 throughout, because that is what Nilearn's second level actually writes.
    # A float32 fixture makes the source header carry the output's dtype by accident,
    # which is exactly the coincidence that hid a float64 standard-error map in
    # production while this test read float32 and passed.
    return CohortReportInputs(
        task="pain",
        model="one-sample",
        output_name="pain_group_mean",
        stat_type="t",
        output_dir=tmp_path,
        design_matrix=pd.DataFrame({"intercept": np.ones(6)}),
        contrast_spec="intercept",
        input_manifest=pd.DataFrame(
            {
                "subject": [f"sub-{index:04d}" for index in range(1, 7)],
                "map_path": [f"{index}.nii.gz" for index in range(1, 7)],
            }
        ),
        metadata={},
        saved_maps={
            "z_score": _write_image64(tmp_path / "z.nii.gz", z_values),
            "effect_size": _write_image64(tmp_path / "effect.nii.gz", z_values / 10.0),
            "effect_variance": _write_image64(tmp_path / "variance.nii.gz", variance),
            "analysis_mask": _write_image64(tmp_path / "mask.nii.gz", mask),
            "permutation_logp_max_t": _write_image64(tmp_path / "logp.nii.gz", logp),
        },
        design_outputs={},
        n_permutations=5000,
        permutation_two_sided=True,
        permutation_random_state=42,
    )


def _build(tmp_path: Path, **report_kwargs: object) -> str:
    report_path = build_cohort_report(
        inputs=_t_model_inputs(tmp_path),
        report_config=CohortReportConfig(
            formats=("png",),
            include_input_map_concordance=False,
        include_leave_one_out_influence=False,
        include_residual_diagnostics=False,
            include_unthresholded=False,
            **report_kwargs,
        ),
        threshold_config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
    )
    return report_path.read_text(encoding="utf-8")


def test_true_discovery_proportion_panel_is_omitted_when_not_requested(tmp_path: Path) -> None:
    assert "True-discovery proportion" not in _build(
        tmp_path, include_true_discovery_proportion=False
    )


def test_derived_niftis_match_the_nilearn_expressions_that_define_them(tmp_path: Path) -> None:
    """Pin the two derived maps by value, so routing them through Nilearn cannot drift.

    Both were hand-rolled in NumPy. The standard error is the square root of Nilearn's
    effect variance inside the mask and exactly zero outside it; the permutation map
    keeps every voxel at or above the α cutoff and zeroes the rest.
    """
    _build(tmp_path)
    plots_dir = tmp_path / "report" / "plots"

    variance = nib.load(str(tmp_path / "variance.nii.gz")).get_fdata()
    mask = nib.load(str(tmp_path / "mask.nii.gz")).get_fdata() > 0
    standard_error = nib.load(str(plots_dir / "group_standard_error.nii.gz"))
    values = standard_error.get_fdata()
    assert values[mask] == pytest.approx(np.sqrt(variance[mask]))
    assert not values[~mask].any()
    assert standard_error.get_data_dtype() == np.float32

    cutoff = -np.log10(0.05)
    logp = nib.load(str(tmp_path / "logp.nii.gz")).get_fdata()
    survivors = nib.load(str(plots_dir / "permutation_logp_max_t_thresholded.nii.gz")).get_fdata()
    assert survivors == pytest.approx(np.where(logp >= cutoff, logp, 0.0))


def _two_tailed_stat_image() -> nib.Nifti1Image:
    """Several moderate positive clusters and one much stronger negative cluster."""
    values = np.zeros((20, 20, 20), dtype=float)
    for index, origin in enumerate([(2, 2, 2), (2, 2, 10), (2, 10, 2), (10, 2, 2)]):
        x, y, z = origin
        values[x : x + 3, y : y + 3, z : z + 3] = 3.5 + 0.2 * index
    values[14:17, 14:17, 14:17] = -8.0
    return _image(values)


def test_leading_clusters_rank_both_tails_when_the_test_is_two_sided(tmp_path: Path) -> None:
    """Nilearn appends the negative tail after the positive one, each peak-sorted.

    Taking the head of that concatenation shows only positive clusters whenever the
    positive tail is long enough to fill the table, so the strongest finding in a
    two-sided contrast can be one the table structurally cannot reach.
    """
    from fmri_pipeline.analysis.report import cohort

    stat_img = _two_tailed_stat_image()
    result = cohort._cluster_table(
        stat_img=stat_img,
        threshold=cohort.ResolvedGroupThreshold(
            image=stat_img, threshold=3.0, label="|z| > 3", two_sided=True
        ),
        config=CohortThresholdConfig(height_control="none", uncorrected_z_threshold=3.0),
        output_path=tmp_path / "clusters.tsv",
        max_rows=2,
        labeller=None,
    )

    assert "-8" in result.block.html
    assert "4 positive, 1 negative" in result.block.caption


def test_cluster_peaks_carry_anatomical_names_when_an_atlas_is_configured(
    tmp_path: Path,
) -> None:
    from fmri_pipeline.analysis.report import cohort
    from fmri_pipeline.analysis.report.atlas import AtlasLabeller

    stat_img = _two_tailed_stat_image()
    labels = np.zeros((20, 20, 20), dtype=np.int16)
    labels[14:17, 14:17, 14:17] = 1
    result = cohort._cluster_table(
        stat_img=stat_img,
        threshold=cohort.ResolvedGroupThreshold(
            image=stat_img, threshold=3.0, label="|z| > 3", two_sided=True
        ),
        config=CohortThresholdConfig(height_control="none", uncorrected_z_threshold=3.0),
        output_path=tmp_path / "clusters.tsv",
        max_rows=2,
        labeller=AtlasLabeller(
            label_img=nib.Nifti1Image(labels, np.eye(4)),
            names={1: "Test Region"},
            source="test_atlas.nii.gz",
        ),
    )

    assert "Region" in result.block.html
    assert "Test Region" in result.block.html
    assert "test_atlas.nii.gz" in result.block.caption


def test_a_negative_cluster_gets_no_true_discovery_bound_rather_than_a_zero() -> None:
    """Nilearn's bound covers the positive tail only.

    A negative-tail peak lies outside every bounded cluster and samples 0.0 there.
    Written into the table that zero reads as "none of this cluster is real", which is
    the opposite of "this cluster was never covered".
    """
    from fmri_pipeline.analysis.report import cohort

    proportion = np.zeros((6, 6, 6))
    proportion[1, 1, 1] = 0.25
    frame = pd.DataFrame({"X": [1.0, 4.0], "Y": [1.0, 4.0], "Z": [1.0, 4.0], "Peak z": [5.0, -5.0]})

    bounded = cohort._with_true_discovery_bound(frame, _image(proportion))

    assert list(bounded["true-discovery bound"]) == ["0.250", ""]


def test_cluster_rows_carry_the_true_discovery_bound_for_their_own_cluster(
    tmp_path: Path,
) -> None:
    """The bound is only readable against the cluster it belongs to.

    Formed at the display height, so the clusters Nilearn bounds are exactly the ones
    the table lists, and each row's number describes that row.
    """
    from fmri_pipeline.analysis.report import cohort

    values = np.zeros((14, 14, 14), dtype=float)
    values[3:8, 3:8, 3:8] = 6.0
    values[10:12, 10:12, 10:12] = -4.0
    stat_img = _image(values)
    mask_img = _image(np.ones_like(values))
    threshold = cohort.ResolvedGroupThreshold(
        image=stat_img, threshold=2.8, label="|z| > 2.8", two_sided=True
    )

    result = cohort._cluster_table(
        stat_img=stat_img,
        threshold=threshold,
        config=CohortThresholdConfig(height_control="none", uncorrected_z_threshold=2.8),
        output_path=tmp_path / "clusters.tsv",
        max_rows=10,
        labeller=None,
        true_discovery_img=cohort._true_discovery_image(
            z_img=stat_img, mask_img=mask_img, threshold=threshold.threshold, alpha=0.05
        ),
    )

    assert "true-discovery bound" in result.block.html
    assert "1.000" in result.block.html
    assert "positive tail only" in result.block.caption


def test_an_unusable_surface_mesh_costs_the_panel_not_the_report(tmp_path: Path) -> None:
    """The mesh is a fetched dataset, so a report machine may simply not have it.

    Nilearn would go to the network for it. A report that reads a derivatives tree
    offline must not acquire a download as a side effect of a figure, and must not die
    because a figure could not be drawn.
    """
    report_text = _build(tmp_path, surface_mesh="not-a-real-mesh")

    assert "Cohort fMRI report" in report_text
    assert "Cortical surface" not in report_text


@pytest.mark.skipif(
    not (Path.home() / "nilearn_data" / "fsaverage" / "infl_left.gii.gz").is_file(),
    reason="fsaverage mesh is not cached",
)
def test_a_configured_surface_mesh_adds_the_cortical_panel(tmp_path: Path) -> None:
    report_text = _build(tmp_path, surface_mesh="fsaverage")

    assert "Cortical surface" in report_text
    # The figure states its own cortex-only limit in pixels; the caption has to repeat it
    # in text, because that is the copy a reader can search and a screen reader can read.
    assert "absent by construction" in report_text


def test_every_permutation_correction_nilearn_wrote_is_reported(tmp_path: Path) -> None:
    """One permutation run yields four corrected maps; the report showed only the first.

    Voxel max-T is the most conservative of them, so a report that shows it alone is the
    one most likely to read as a null result while the cluster-extent, cluster-mass and
    TFCE maps from the same run sit unopened on disk.
    """
    inputs = _t_model_inputs(tmp_path)
    logp = tmp_path / "logp.nii.gz"
    saved = dict(inputs.saved_maps)
    for key in ("logp_max_tfce", "logp_max_size", "logp_max_mass"):
        saved[f"permutation_{key}"] = logp
    # The cluster-shaped corrections are defined by their forming threshold, so the
    # report refuses to present them without it.
    inputs = replace(inputs, saved_maps=saved, permutation_cluster_forming_p=0.001)

    report_path = build_cohort_report(
        inputs=inputs,
        report_config=CohortReportConfig(
            formats=("png",),
            include_input_map_concordance=False,
        include_leave_one_out_influence=False,
        include_residual_diagnostics=False,
            include_unthresholded=False,
        ),
        threshold_config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
    )
    report_text = report_path.read_text(encoding="utf-8")

    for label in ("Max-T", "TFCE", "Cluster extent", "Cluster mass"):
        assert f"{label} permutation evidence" in report_text, label
        assert f"{label} corrected peaks" in report_text, label

    plots_dir = tmp_path / "report" / "plots"
    for stem in ("max_t", "tfce", "cluster_size", "cluster_mass"):
        assert (plots_dir / f"{stem}_peaks.tsv").exists(), stem


def test_a_constant_cluster_is_reported_as_a_centre_of_mass_not_a_peak(tmp_path: Path) -> None:
    """A cluster-level p-value is a property of the cluster, not of a voxel.

    Every voxel of a cluster-extent or cluster-mass cluster therefore carries the same
    corrected value, and Nilearn answers a request for its peak with the cluster's centre
    of mass. Calling that a peak in the caption claims a maximum that does not exist, and
    hides that the signed z beside it was sampled at a centroid.
    """
    from fmri_pipeline.analysis.report import cohort

    evidence = np.zeros((14, 14, 14), dtype=float)
    evidence[4:9, 4:9, 4:9] = 2.0  # one cluster, one value throughout
    z = np.zeros((14, 14, 14), dtype=float)
    z[4:9, 4:9, 4:9] = np.linspace(3.0, 6.0, 125).reshape(5, 5, 5)

    result = cohort._max_t_peak_table(
        evidence_img=_image(evidence),
        z_img=_image(z),
        effect_img=_image(z / 10.0),
        cutoff=-np.log10(0.05),
        min_distance_mm=8.0,
        max_rows=10,
        output_path=tmp_path / "peaks.tsv",
        label="Cluster extent",
    )

    assert "centre of mass" in result.block.caption
    assert "1 of 1" in result.block.caption


def test_leave_one_out_measures_influence_that_concordance_cannot_see(tmp_path: Path) -> None:
    """Pairwise correlation is scale-invariant, so it cannot see amplitude.

    The participant here has the cohort's exact spatial pattern at twelve times its
    amplitude, so they correlate with everyone and the concordance panel ranks them
    unremarkable. What they actually do is inflate the between-subject variance enough to
    suppress the whole result: the full cohort finds nothing, and dropping them alone
    uncovers the signal. Only a refit can show that.
    """
    from fmri_pipeline.analysis.report import cohort

    shape = (10, 10, 10)
    signal = np.zeros(shape)
    signal[3:7, 3:7, 3:7] = 1.0
    rng = np.random.default_rng(0)
    paths, subjects = [], []
    for index in range(6):
        amplitude = 12.0 if index == 0 else 1.0
        values = signal * amplitude + rng.normal(scale=0.05, size=shape)
        subjects.append(f"sub-{index:04d}")
        paths.append(str(_write_image64(tmp_path / f"s{index}.nii.gz", values)))
    mask_img = _write_image64(tmp_path / "mask.nii.gz", np.ones(shape))

    frame = cohort._leave_one_out_influence(
        manifest=pd.DataFrame({"subject": subjects, "map_path": paths}),
        design_matrix=pd.DataFrame({"intercept": np.ones(6)}),
        contrast_spec="intercept",
        mask_img=nib.load(str(mask_img)),
        threshold_config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
        stat_type="t",
    )

    assert list(frame["subject"]) == subjects
    assert frame["estimable"].all()
    without = dict(zip(frame["subject"], frame["surviving voxels"]))
    assert without["sub-0000"] > 0
    assert all(count == 0 for subject, count in without.items() if subject != "sub-0000")


def test_residual_diagnostics_measure_the_gaussian_assumption_the_report_states(
    tmp_path: Path,
) -> None:
    """The assumption note claims Gaussian errors; nothing in the report bore on it.

    Nilearn's OLSModel gives the residuals of the very design being reported, so the
    claim can be shown rather than asserted. Two participants here carry a systematic
    offset, which is exactly the kind of departure a pooled residual distribution shows
    and a per-participant spread attributes.
    """
    from fmri_pipeline.analysis.report import cohort

    shape = (8, 8, 8)
    rng = np.random.default_rng(3)
    paths, subjects = [], []
    for index in range(8):
        offset = 4.0 if index < 2 else 0.0
        values = rng.normal(size=shape) + offset
        subjects.append(f"sub-{index:04d}")
        paths.append(str(_write_image64(tmp_path / f"s{index}.nii.gz", values)))
    mask_img = nib.load(str(_write_image64(tmp_path / "mask.nii.gz", np.ones(shape))))

    result = cohort._residual_diagnostics(
        manifest=pd.DataFrame({"subject": subjects, "map_path": paths}),
        design_matrix=pd.DataFrame({"intercept": np.ones(8)}),
        mask_img=mask_img,
    )

    assert list(result.per_subject["subject"]) == subjects
    assert result.normalized.shape == (8, 512)
    # The offset participants sit apart from the rest on residual spread.
    rms = result.per_subject["residual RMS"].to_numpy(dtype=float)
    assert rms[:2].min() > rms[2:].max()


def test_the_report_embeds_an_interactive_viewer_and_a_parametric_cross_check(
    tmp_path: Path,
) -> None:
    """Two things the permutation run already paid for but the report never used.

    Nilearn's view_img inlines its own data and script, so the reader can reach any
    cluster rather than the few a fixed mosaic cuts through. And the permutation path
    computes its own t map, which the parametric path computes too: they are the same
    contrast on the same data, so a disagreement means the two saw different inputs.
    """
    inputs = _t_model_inputs(tmp_path)
    saved = dict(inputs.saved_maps)
    # The same file for both paths, so agreement must be exact.
    saved["permutation_t"] = saved["z_score"]
    saved["stat"] = saved["z_score"]
    inputs = replace(inputs, saved_maps=saved)

    report_path = build_cohort_report(
        inputs=inputs,
        report_config=CohortReportConfig(
            formats=("png",),
            include_input_map_concordance=False,
            include_leave_one_out_influence=False,
            include_residual_diagnostics=False,
            include_unthresholded=False,
        ),
        threshold_config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
    )
    report_text = report_path.read_text(encoding="utf-8")

    assert "Interactive volume viewer" in report_text
    assert "Permutation cross-check" in report_text
    # The two t maps are the same file here, so they must agree exactly.
    assert "1.000000" in report_text


def test_the_summary_states_the_choices_the_cluster_corrections_depend_on(
    tmp_path: Path,
) -> None:
    """Cluster-extent and cluster-mass FWE are answers to a threshold, and to smoothness.

    Their clusters are formed at a p the reader never chose and could not see, and all
    three cluster-shaped corrections are functions of the smoothing applied. Reporting
    the results without either leaves them unreproducible.
    """
    inputs = _t_model_inputs(tmp_path)
    saved = dict(inputs.saved_maps)
    saved["permutation_logp_max_size"] = saved["permutation_logp_max_t"]
    inputs = replace(
        inputs,
        saved_maps=saved,
        permutation_cluster_forming_p=0.001,
        smoothing_fwhm=None,
    )

    report_text = build_cohort_report(
        inputs=inputs,
        report_config=CohortReportConfig(
            formats=("png",),
            include_input_map_concordance=False,
            include_leave_one_out_influence=False,
            include_residual_diagnostics=False,
            include_unthresholded=False,
            include_interactive_viewer=False,
        ),
        threshold_config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
    ).read_text(encoding="utf-8")

    assert "Cluster-forming threshold" in report_text
    assert "p &lt; 0.001" in report_text or "p < 0.001" in report_text
    assert "Second-level smoothing" in report_text
    assert "none" in report_text


def test_cluster_level_evidence_cannot_be_reported_without_its_forming_threshold(
    tmp_path: Path,
) -> None:
    """The parameter that defines the clusters is not optional provenance.

    Voxel max-T needs no forming threshold, so its absence is only a fault once a
    cluster-shaped correction is present -- which is exactly the case that shipped
    without it.
    """
    inputs = _t_model_inputs(tmp_path)
    saved = dict(inputs.saved_maps)
    saved["permutation_logp_max_mass"] = saved["permutation_logp_max_t"]
    inputs = replace(inputs, saved_maps=saved, permutation_cluster_forming_p=None)

    with pytest.raises(ValueError, match="cluster-forming"):
        build_cohort_report(
            inputs=inputs,
            report_config=CohortReportConfig(
                formats=("png",),
                include_input_map_concordance=False,
                include_leave_one_out_influence=False,
                include_residual_diagnostics=False,
                include_unthresholded=False,
                include_interactive_viewer=False,
            ),
            threshold_config=CohortThresholdConfig(height_control="fdr", alpha=0.05),
        )
