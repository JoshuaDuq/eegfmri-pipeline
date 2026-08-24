from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.second_level import (
    FirstLevelMapRecord,
    PreparedSecondLevelInput,
    SecondLevelConfig,
    SecondLevelPermutationConfig,
    _discover_first_level_effect_size_map,
    _write_design_matrix_files,
    load_second_level_config_section,
    prepare_second_level_input,
    run_second_level_analysis,
    second_level_permutation_config_from_mapping,
)
from fmri_pipeline.analysis.report.cohort import (
    CohortReportConfig,
    CohortThresholdConfig,
)


def _make_record(
    tmp_path: Path,
    subject: str,
    contrast_name: str,
    *,
    contrast_cfg: dict[str, object] | None = None,
    run_meta: dict[str, object] | None = None,
) -> FirstLevelMapRecord:
    map_path = tmp_path / f"sub-{subject}_{contrast_name}.nii.gz"
    sidecar_path = tmp_path / f"sub-{subject}_{contrast_name}.json"
    map_path.write_text("nii", encoding="utf-8")
    sidecar_path.write_text("{}", encoding="utf-8")
    default_run_meta = {
        "discovered_bold_paths": [
            str(tmp_path / f"sub-{subject}_task-pain_run-01_desc-preproc_bold.nii.gz"),
            str(tmp_path / f"sub-{subject}_task-pain_run-02_desc-preproc_bold.nii.gz"),
        ],
        "included_bold_paths": [
            str(tmp_path / f"sub-{subject}_task-pain_run-01_desc-preproc_bold.nii.gz"),
            str(tmp_path / f"sub-{subject}_task-pain_run-02_desc-preproc_bold.nii.gz"),
        ],
    }
    return FirstLevelMapRecord(
        subject=subject,
        subject_label=f"sub-{subject}",
        contrast_name=contrast_name,
        map_path=map_path,
        sidecar_path=sidecar_path,
        contrast_cfg=contrast_cfg or {"fmriprep_space": "MNI152NLin2009cAsym"},
        run_meta=run_meta or default_run_meta,
    )


def test_first_level_discovery_ignores_appledouble_sidecars(tmp_path: Path) -> None:
    contrast_dir = (
        tmp_path
        / "sub-0001"
        / "fmri"
        / "first_level"
        / "task-pain"
        / "contrast-pain"
    )
    contrast_dir.mkdir(parents=True)
    stem = "sub-0001_task-pain_contrast-pain_stat-effect_size_deadbeef"
    map_path = contrast_dir / f"{stem}.nii.gz"
    sidecar_path = contrast_dir / f"{stem}.json"
    map_path.write_bytes(b"nii")
    sidecar_path.write_text(
        """{
  "subject": "sub-0001",
  "task": "pain",
  "contrast_name": "pain",
  "output_type_actual": "effect_size",
  "contrast_cfg": {"fmriprep_space": "MNI152NLin2009cAsym"},
  "run_meta": {
    "included_bold_paths": ["sub-0001_task-pain_run-01_bold.nii.gz"],
    "discovered_bold_paths": ["sub-0001_task-pain_run-01_bold.nii.gz"]
  }
} """,
        encoding="utf-8",
    )
    (contrast_dir / f"._{stem}.json").write_bytes(b"\x00\x05\x16\x07\xb0AppleDouble")

    record = _discover_first_level_effect_size_map(
        input_root=tmp_path,
        subject="0001",
        task="pain",
        contrast_name="pain",
    )

    assert record.map_path == map_path


def test_write_design_matrix_files_surfaces_plot_failures(tmp_path: Path) -> None:
    design_matrix = pd.DataFrame({"intercept": [1.0, 1.0]})

    with patch(
        "fmri_pipeline.analysis.report.figures.design.design_matrix_figure",
        side_effect=RuntimeError("plot failed"),
    ):
        with pytest.raises(RuntimeError, match="plot failed"):
            _write_design_matrix_files(
                output_dir=tmp_path,
                design_matrix=design_matrix,
            )


def test_the_second_level_design_reports_its_conditioning(tmp_path: Path) -> None:
    # A second-level design whose covariate is collinear with its group regressor is
    # the classic group-analysis confound, and nilearn's bare design plotter shows
    # nothing about it.
    design_matrix = pd.DataFrame(
        {
            "intercept": np.ones(12),
            "group": np.r_[np.ones(6), np.zeros(6)],
            "covariate": np.r_[np.ones(6), np.zeros(6)] * 2.0 + 1e-9 * np.arange(12),
        }
    )
    out = _write_design_matrix_files(
        output_dir=tmp_path, design_matrix=design_matrix, contrast_spec="group"
    )
    assert Path(out["design_matrix_png"]).exists()
    assert Path(out["design_correlation_png"]).exists()
    assert Path(out["design_vif_png"]).exists()
    assert float(out["design_condition_number"]) > 1.0
    assert "design_max_vif" in out


def test_the_second_level_design_carries_the_contrast_it_tests(tmp_path: Path) -> None:
    design_matrix = pd.DataFrame({"intercept": np.ones(8), "group": np.r_[np.ones(4), -np.ones(4)]})
    out = _write_design_matrix_files(
        output_dir=tmp_path, design_matrix=design_matrix, contrast_spec="group"
    )
    # Efficiency is only defined once the contrast has landed on the columns.
    assert "design_contrast_efficiency" in out


def test_an_f_contrast_leaves_the_design_figure_without_a_strip(tmp_path: Path) -> None:
    # Several rows cannot be drawn as one strip; the design is still worth drawing.
    design_matrix = pd.DataFrame(
        {"a": np.r_[np.ones(4), np.zeros(4)], "b": np.r_[np.zeros(4), np.ones(4)]}
    )
    out = _write_design_matrix_files(
        output_dir=tmp_path,
        design_matrix=design_matrix,
        contrast_spec=np.array([[1.0, -1.0], [1.0, 1.0]]),
    )
    assert Path(out["design_matrix_png"]).exists()
    assert "design_contrast_efficiency" not in out


def test_prepare_second_level_one_sample_builds_intercept_design(tmp_path: Path) -> None:
    records = {
        ("0001", "pain"): _make_record(tmp_path, "0001", "pain"),
        ("0002", "pain"): _make_record(tmp_path, "0002", "pain"),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(model="one-sample", contrast_names=("pain",)).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
            side_effect=_discover,
        ),
        patch("fmri_pipeline.analysis.second_level._validate_same_grid"),
    ):
        prepared = prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert list(prepared.design_matrix.columns) == ["intercept"]
    assert prepared.contrast_spec == "intercept"
    assert prepared.output_name == "pain_group_mean"
    assert len(prepared.image_paths) == 2


def test_prepare_second_level_two_sample_uses_group_columns_and_covariates(
    tmp_path: Path,
) -> None:
    records = {
        ("0001", "pain"): _make_record(tmp_path, "0001", "pain"),
        ("0002", "pain"): _make_record(tmp_path, "0002", "pain"),
        ("0003", "pain"): _make_record(tmp_path, "0003", "pain"),
        ("0004", "pain"): _make_record(tmp_path, "0004", "pain"),
    }
    covariates_path = tmp_path / "group.tsv"
    pd.DataFrame(
        {
            "subject": ["sub-0001", "sub-0002", "sub-0003", "sub-0004"],
            "group": ["control", "patient", "control", "patient"],
            "age": [20, 24, 26, 30],
        }
    ).to_csv(covariates_path, sep="\t", index=False)

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="two-sample",
        contrast_names=("pain",),
        covariates_file=str(covariates_path),
        subject_column="subject",
        covariate_columns=("age",),
        group_column="group",
        group_a_value="control",
        group_b_value="patient",
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
            side_effect=_discover,
        ),
        patch("fmri_pipeline.analysis.second_level._validate_same_grid"),
    ):
        prepared = prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0002", "0003", "0004"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert list(prepared.design_matrix.columns) == [
        "group_control",
        "group_patient",
        "cov_age",
    ]
    assert prepared.contrast_spec == "group_patient - group_control"
    assert prepared.metadata["group_design_columns"]["group_a"] == "group_control"
    assert prepared.metadata["group_design_columns"]["group_b"] == "group_patient"


def test_prepare_second_level_repeated_measures_defaults_to_omnibus_f(
    tmp_path: Path,
) -> None:
    records = {
        ("0001", "low"): _make_record(tmp_path, "0001", "low"),
        ("0002", "low"): _make_record(tmp_path, "0002", "low"),
        ("0001", "med"): _make_record(tmp_path, "0001", "med"),
        ("0002", "med"): _make_record(tmp_path, "0002", "med"),
        ("0001", "high"): _make_record(tmp_path, "0001", "high"),
        ("0002", "high"): _make_record(tmp_path, "0002", "high"),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="repeated-measures",
        contrast_names=("low", "med", "high"),
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
            side_effect=_discover,
        ),
        patch("fmri_pipeline.analysis.second_level._validate_same_grid"),
    ):
        prepared = prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert prepared.stat_type == "F"
    assert prepared.contrast_spec.shape == (2, len(prepared.design_matrix.columns))
    assert list(prepared.design_matrix.columns) == [
        "condition_low",
        "condition_med",
        "condition_high",
        "subject_0002",
    ]
    np.testing.assert_allclose(
        prepared.contrast_spec,
        np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
            ]
        ),
    )


def test_second_level_config_rejects_repeated_measures_permutation_inference() -> None:
    cfg = SecondLevelConfig(
        model="repeated-measures",
        contrast_names=("low", "high"),
        permutation=SecondLevelPermutationConfig(enabled=True),
    )

    with pytest.raises(ValueError, match="Repeated-measures permutation inference is unsupported"):
        cfg.normalized()


def test_second_level_permutation_config_requires_a_reproducible_integer_seed() -> None:
    assert (
        SecondLevelPermutationConfig(enabled=True, random_state=17).normalized().random_state == 17
    )

    with pytest.raises(TypeError, match="random_state"):
        SecondLevelPermutationConfig(enabled=True, random_state="17").normalized()
    with pytest.raises(TypeError, match="enabled"):
        SecondLevelPermutationConfig(enabled="false").normalized()
    with pytest.raises(TypeError, match="n_permutations"):
        SecondLevelPermutationConfig(n_permutations="5000").normalized()
    with pytest.raises(TypeError, match="two_sided"):
        SecondLevelPermutationConfig(two_sided="false").normalized()
    with pytest.raises(ValueError, match="Unknown.*permutation"):
        second_level_permutation_config_from_mapping({"random_seed": 42})


def test_second_level_yaml_section_rejects_unknown_keys_and_boolean_strings() -> None:
    with pytest.raises(ValueError, match="Unknown fmri_group_level key"):
        load_second_level_config_section({"fmri_group_level": {"random_sate": 42}})
    with pytest.raises(TypeError, match="write_design_matrix"):
        load_second_level_config_section({"fmri_group_level": {"write_design_matrix": "false"}})


def test_second_level_config_carries_explicit_cohort_report_settings() -> None:
    cfg = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        report=CohortReportConfig(enabled=True, html_report=True),
        threshold=CohortThresholdConfig(
            height_control="fdr",
            alpha=0.05,
            two_sided=True,
        ),
    ).normalized()

    assert cfg.report.enabled is True
    assert cfg.report.html_report is True
    assert cfg.threshold.height_control == "fdr"
    assert cfg.threshold.alpha == pytest.approx(0.05)


def test_prepare_second_level_rejects_cross_contrast_model_mismatch(
    tmp_path: Path,
) -> None:
    low_cfg = {
        "fmriprep_space": "MNI152NLin2009cAsym",
        "hrf_model": "spm",
        "high_pass_hz": 0.008,
    }
    high_cfg = {
        "fmriprep_space": "MNI152NLin2009cAsym",
        "hrf_model": "spm",
        "high_pass_hz": 0.01,
    }
    records = {
        ("0001", "low"): _make_record(tmp_path, "0001", "low", contrast_cfg=low_cfg),
        ("0002", "low"): _make_record(tmp_path, "0002", "low", contrast_cfg=low_cfg),
        ("0001", "high"): _make_record(tmp_path, "0001", "high", contrast_cfg=high_cfg),
        ("0002", "high"): _make_record(tmp_path, "0002", "high", contrast_cfg=high_cfg),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="paired",
        contrast_names=("low", "high"),
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
            side_effect=_discover,
        ),
        patch("fmri_pipeline.analysis.second_level._validate_same_grid"),
    ):
        with pytest.raises(ValueError, match="same first-level model settings"):
            prepare_second_level_input(
                config=cfg,
                subjects=["0001", "0002"],
                task="pain",
                deriv_root=tmp_path,
            )


def test_prepare_second_level_rejects_rank_deficient_design(
    tmp_path: Path,
) -> None:
    records = {
        ("0001", "pain"): _make_record(tmp_path, "0001", "pain"),
        ("0002", "pain"): _make_record(tmp_path, "0002", "pain"),
        ("0003", "pain"): _make_record(tmp_path, "0003", "pain"),
    }
    covariates_path = tmp_path / "group.tsv"
    pd.DataFrame(
        {
            "subject": ["sub-0001", "sub-0002", "sub-0003"],
            "age": [20, 24, 28],
            "age_copy": [30, 34, 38],
        }
    ).to_csv(covariates_path, sep="\t", index=False)

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        covariates_file=str(covariates_path),
        subject_column="subject",
        covariate_columns=("age", "age_copy"),
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
            side_effect=_discover,
        ),
        patch("fmri_pipeline.analysis.second_level._validate_same_grid"),
    ):
        with pytest.raises(ValueError, match="rank-deficient"):
            prepare_second_level_input(
                config=cfg,
                subjects=["0001", "0002", "0003"],
                task="pain",
                deriv_root=tmp_path,
            )


def test_prepare_second_level_rejects_duplicate_subjects(tmp_path: Path) -> None:
    cfg = SecondLevelConfig(model="one-sample", contrast_names=("pain",)).normalized()

    with pytest.raises(ValueError, match="requires unique subjects"):
        prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0001"],
            task="pain",
            deriv_root=tmp_path,
        )


def test_prepare_second_level_rejects_inconsistent_run_inclusion(tmp_path: Path) -> None:
    records = {
        ("0001", "pain"): _make_record(
            tmp_path,
            "0001",
            "pain",
            run_meta={
                "discovered_bold_paths": [
                    str(tmp_path / "sub-0001_task-pain_run-01_desc-preproc_bold.nii.gz"),
                    str(tmp_path / "sub-0001_task-pain_run-02_desc-preproc_bold.nii.gz"),
                ],
                "included_bold_paths": [
                    str(tmp_path / "sub-0001_task-pain_run-01_desc-preproc_bold.nii.gz"),
                    str(tmp_path / "sub-0001_task-pain_run-02_desc-preproc_bold.nii.gz"),
                ],
            },
        ),
        ("0002", "pain"): _make_record(
            tmp_path,
            "0002",
            "pain",
            run_meta={
                "discovered_bold_paths": [
                    str(tmp_path / "sub-0002_task-pain_run-01_desc-preproc_bold.nii.gz"),
                    str(tmp_path / "sub-0002_task-pain_run-02_desc-preproc_bold.nii.gz"),
                ],
                "included_bold_paths": [
                    str(tmp_path / "sub-0002_task-pain_run-01_desc-preproc_bold.nii.gz"),
                ],
            },
        ),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(model="one-sample", contrast_names=("pain",)).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
            side_effect=_discover,
        ),
        patch("fmri_pipeline.analysis.second_level._validate_same_grid"),
    ):
        with pytest.raises(ValueError, match="same discovered and included runs"):
            prepare_second_level_input(
                config=cfg,
                subjects=["0001", "0002"],
                task="pain",
                deriv_root=tmp_path,
            )


def test_second_level_analysis_persists_mask_and_builds_enabled_cohort_report(
    tmp_path: Path,
) -> None:
    import nibabel as nib

    image = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    prepared = PreparedSecondLevelInput(
        image_paths=(tmp_path / "sub-0001.nii.gz", tmp_path / "sub-0002.nii.gz"),
        design_matrix=pd.DataFrame({"intercept": [1.0, 1.0]}),
        manifest=pd.DataFrame(
            {
                "subject": ["sub-0001", "sub-0002"],
                "contrast_name": ["pain", "pain"],
                "condition_label": ["pain", "pain"],
                "map_path": ["sub-0001.nii.gz", "sub-0002.nii.gz"],
            }
        ),
        contrast_spec="intercept",
        stat_type="t",
        output_name="pain_group_mean",
        output_dir=tmp_path / "group",
        metadata={"model": "one-sample"},
    )

    class _Model:
        class _Masker:
            mask_img_ = image

        # The real SecondLevelModel is constructed with smoothing_fwhm, which the
        # pipeline now passes explicitly instead of leaning on Nilearn's default.
        def __init__(self, **_kwargs):
            pass

        masker_ = _Masker()

        def fit(self, **_kwargs):
            return self

        def compute_contrast(self, **_kwargs):
            return {
                "effect_size": image,
                "effect_variance": image,
                "p_value": image,
                "stat": image,
                "z_score": image,
            }

    report_path = prepared.output_dir / "report" / "cohort.html"

    def _build_report(**kwargs):
        assert "analysis_mask" in kwargs["inputs"].saved_maps
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("report", encoding="utf-8")
        return report_path

    config = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        write_design_matrix=False,
        report=CohortReportConfig(enabled=True, html_report=True),
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level.prepare_second_level_input",
            return_value=prepared,
        ),
        patch("nilearn.glm.second_level.SecondLevelModel", _Model),
        patch(
            "fmri_pipeline.analysis.report.cohort.build_cohort_report",
            side_effect=_build_report,
        ) as build_report,
    ):
        outputs = run_second_level_analysis(
            config=config,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert Path(outputs["saved_maps"]["analysis_mask"]).exists()
    assert outputs["report_path"] == str(report_path)
    build_report.assert_called_once()


def test_second_level_analysis_skips_disabled_cohort_report(tmp_path: Path) -> None:
    import nibabel as nib

    image = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    prepared = PreparedSecondLevelInput(
        image_paths=(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"),
        design_matrix=pd.DataFrame({"intercept": [1.0, 1.0]}),
        manifest=pd.DataFrame({"subject": ["sub-1", "sub-2"], "map_path": ["a", "b"]}),
        contrast_spec="intercept",
        stat_type="t",
        output_name="pain_group_mean",
        output_dir=tmp_path / "group",
        metadata={"model": "one-sample"},
    )

    class _Model:
        class _Masker:
            mask_img_ = image

        # The real SecondLevelModel is constructed with smoothing_fwhm, which the
        # pipeline now passes explicitly instead of leaning on Nilearn's default.
        def __init__(self, **_kwargs):
            pass

        masker_ = _Masker()

        def fit(self, **_kwargs):
            return self

        def compute_contrast(self, **_kwargs):
            return {"z_score": image}

    config = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        write_design_matrix=False,
        report=CohortReportConfig(enabled=False, html_report=True),
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level.prepare_second_level_input",
            return_value=prepared,
        ),
        patch("nilearn.glm.second_level.SecondLevelModel", _Model),
        patch("fmri_pipeline.analysis.report.cohort.build_cohort_report") as build_report,
    ):
        outputs = run_second_level_analysis(
            config=config,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert "report_path" not in outputs
    build_report.assert_not_called()


def test_second_level_analysis_uses_the_fitted_nilearn_masker(tmp_path: Path) -> None:
    import nibabel as nib

    shape = (9, 9, 9)
    support = np.zeros(shape, dtype=bool)
    support[1:-1, 1:-1, 1:-1] = True
    rng = np.random.default_rng(17)
    image_paths = []
    for index in range(6):
        data = np.zeros(shape, dtype=np.float32)
        data[support] = rng.normal(loc=0.2, scale=1.0, size=support.sum())
        path = tmp_path / f"sub-{index + 1:04d}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0])), path)
        image_paths.append(path)

    prepared = PreparedSecondLevelInput(
        image_paths=tuple(image_paths),
        design_matrix=pd.DataFrame({"intercept": np.ones(len(image_paths))}),
        manifest=pd.DataFrame(
            {
                "subject": [f"sub-{index + 1:04d}" for index in range(len(image_paths))],
                "map_path": [str(path) for path in image_paths],
            }
        ),
        contrast_spec="intercept",
        stat_type="t",
        output_name="pain_group_mean",
        output_dir=tmp_path / "group",
        metadata={"model": "one-sample"},
    )
    config = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        write_design_matrix=False,
        report=CohortReportConfig(enabled=False),
    ).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level.prepare_second_level_input",
        return_value=prepared,
    ):
        outputs = run_second_level_analysis(
            config=config,
            subjects=[path.stem for path in image_paths],
            task="pain",
            deriv_root=tmp_path,
        )

    mask_img = nib.load(outputs["saved_maps"]["analysis_mask"])
    assert mask_img.shape == shape
    assert np.array_equal(mask_img.affine, np.diag([2.0, 2.0, 2.0, 1.0]))
    assert np.count_nonzero(mask_img.get_fdata()) == support.sum()


def test_second_level_permutation_passes_the_configured_seed_to_nilearn(
    tmp_path: Path,
) -> None:
    import nibabel as nib

    image = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    prepared = PreparedSecondLevelInput(
        image_paths=(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"),
        design_matrix=pd.DataFrame({"intercept": [1.0, 1.0]}),
        manifest=pd.DataFrame({"subject": ["sub-1", "sub-2"], "map_path": ["a", "b"]}),
        contrast_spec="intercept",
        stat_type="t",
        output_name="pain_group_mean",
        output_dir=tmp_path / "group",
        metadata={"model": "one-sample"},
    )

    class _Model:
        class _Masker:
            mask_img_ = image

        # The real SecondLevelModel is constructed with smoothing_fwhm, which the
        # pipeline now passes explicitly instead of leaning on Nilearn's default.
        def __init__(self, **_kwargs):
            pass

        masker_ = _Masker()

        def fit(self, **_kwargs):
            return self

        def compute_contrast(self, **_kwargs):
            return {"z_score": image}

    config = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        write_design_matrix=False,
        permutation=SecondLevelPermutationConfig(
            enabled=True,
            n_permutations=100,
            two_sided=True,
            random_state=19,
        ),
        report=CohortReportConfig(enabled=False),
    ).normalized()

    with (
        patch(
            "fmri_pipeline.analysis.second_level.prepare_second_level_input",
            return_value=prepared,
        ),
        patch("nilearn.glm.second_level.SecondLevelModel", _Model),
        patch(
            "nilearn.glm.second_level.non_parametric_inference",
            return_value=image,
        ) as inference,
    ):
        run_second_level_analysis(
            config=config,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert inference.call_args.kwargs["random_state"] == 19
    assert inference.call_args.kwargs["mask"] is image


def test_second_level_permutation_saves_every_correction_nilearn_returns(
    tmp_path: Path,
) -> None:
    """Nilearn returns one image for voxel max-T, and a dict once TFCE or a
    cluster-forming threshold is asked for.

    Only ``logp_max_t`` was ever kept, which is the most conservative correction on
    offer; the cluster-extent, cluster-mass and TFCE maps come out of the very same
    permutation run and were being discarded.
    """
    import nibabel as nib

    image = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    prepared = PreparedSecondLevelInput(
        image_paths=(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"),
        design_matrix=pd.DataFrame({"intercept": [1.0, 1.0]}),
        manifest=pd.DataFrame({"subject": ["sub-1", "sub-2"], "map_path": ["a", "b"]}),
        contrast_spec="intercept",
        stat_type="t",
        output_name="pain_group_mean",
        output_dir=tmp_path / "group",
        metadata={"model": "one-sample"},
    )

    class _Model:
        class _Masker:
            mask_img_ = image

        # The real SecondLevelModel is constructed with smoothing_fwhm, which the
        # pipeline now passes explicitly instead of leaning on Nilearn's default.
        def __init__(self, **_kwargs):
            pass

        masker_ = _Masker()

        def fit(self, **_kwargs):
            return self

        def compute_contrast(self, **_kwargs):
            return {"z_score": image}

    config = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        write_design_matrix=False,
        permutation=SecondLevelPermutationConfig(
            enabled=True,
            n_permutations=100,
            two_sided=True,
            random_state=19,
            tfce=True,
            cluster_forming_p=0.001,
        ),
        report=CohortReportConfig(enabled=False),
    ).normalized()

    dict_output = {
        key: image
        for key in (
            "t",
            "logp_max_t",
            "tfce",
            "logp_max_tfce",
            "size",
            "logp_max_size",
            "mass",
            "logp_max_mass",
        )
    }

    with (
        patch(
            "fmri_pipeline.analysis.second_level.prepare_second_level_input",
            return_value=prepared,
        ),
        patch("nilearn.glm.second_level.SecondLevelModel", _Model),
        patch(
            "nilearn.glm.second_level.non_parametric_inference",
            return_value=dict_output,
        ) as inference,
    ):
        result = run_second_level_analysis(
            config=config,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert inference.call_args.kwargs["tfce"] is True
    # Nilearn takes the cluster-forming threshold in p-scale, converting it to t itself.
    assert inference.call_args.kwargs["threshold"] == pytest.approx(0.001)

    saved = result["saved_maps"]
    for key in (
        "permutation_logp_max_t",
        "permutation_logp_max_tfce",
        "permutation_logp_max_size",
        "permutation_logp_max_mass",
        # The statistics the corrections were computed from, not only the corrected
        # p-maps. `t` is the permutation path's own t map, which the parametric path
        # also produces -- the two must agree, and nothing checks that unless it is kept.
        "permutation_t",
        "permutation_size",
        "permutation_mass",
        "permutation_tfce",
    ):
        assert key in saved, key
        assert Path(saved[key]).is_file()
