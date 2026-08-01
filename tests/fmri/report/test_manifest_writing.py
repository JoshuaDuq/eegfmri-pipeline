"""The analysis run must leave behind everything the report needs.

This is the seam. Until the fitting path writes a manifest, a report can only be
produced from inside the GLM, which is what made subject QC recompute per contrast
and what made iterating on a figure mean re-running the model.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fmri_pipeline.analysis.report.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    write_report_manifest as _write_report_manifest,
)


def _run_meta(**overrides) -> dict:
    base = dict(
        analysis_space="T1w",
        tr=2.0,
        confounds_strategy="24HMP+aCompCor",
        included_bold_paths=["/d/sub-01_run-01_bold.nii.gz"],
        included_confounds_paths=["/d/sub-01_run-01_confounds.tsv"],
        retained_frame_indices=[[0]],
        n_runs_included=1,
        skipped_runs=[{"run_label": "run-02", "reason": "mean FD 0.9 mm"}],
    )
    base.update(overrides)
    return base


def _contrast_cfg() -> SimpleNamespace:
    return SimpleNamespace(hrf_model="spm")


def write_report_manifest(*, stat_map: Path, **kwargs) -> Path:
    """Supply valid model-fit artifacts unless a test provides explicit ones."""
    run_meta = kwargs["run_meta"]
    run_count = len(run_meta["included_bold_paths"]) if isinstance(run_meta, dict) else 1
    kwargs.setdefault("residual_paths", (stat_map,) * run_count)
    kwargs.setdefault("predicted_paths", (stat_map,) * run_count)
    return _write_report_manifest(stat_map=stat_map, **kwargs)


def test_a_fitted_contrast_leaves_a_manifest_beside_its_stat_maps(
    tmp_path: Path,
) -> None:
    contrast_dir = tmp_path / "contrast-heatgtwarm"
    contrast_dir.mkdir(parents=True)
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="heat>warm",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        run_meta=_run_meta(),
        contrast_cfg=_contrast_cfg(),
        smoothing_fwhm=6.0,
        signal_scaling_mode="voxel-mean",
    )

    assert written.name == MANIFEST_FILENAME
    assert written.parent == contrast_dir

    restored = read_manifest(written)
    assert restored.subject == "sub-01"
    assert restored.task == "heat"
    assert restored.contrast_name == "heat>warm"
    assert restored.t_r == 2.0
    assert restored.smoothing_fwhm == 6.0
    assert restored.signal_scaling is True
    assert restored.signal_scaling_mode == "voxel-mean"
    assert restored.confound_strategy == "24HMP+aCompCor"


def test_the_manifest_records_exact_model_fit_series(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    residual_path = contrast_dir / "run-01_residual.nii.gz"
    predicted_path = contrast_dir / "run-01_predicted.nii.gz"
    for path in (stat_map, residual_path, predicted_path):
        path.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        residual_paths=(residual_path,),
        predicted_paths=(predicted_path,),
        run_meta=_run_meta(),
        contrast_cfg=_contrast_cfg(),
    )

    restored = read_manifest(written)
    assert restored.residual_paths == (residual_path,)
    assert restored.predicted_paths == (predicted_path,)
    assert restored.retained_frame_indices == ((0,),)
    assert restored.model_fit_series_space == "unwhitened-model-response"


def test_the_manifest_derives_that_no_scaling_happened_from_an_absent_mode(
    tmp_path: Path,
) -> None:
    # The flag and the mode cannot disagree, because there is only one of them on the
    # wire. Recording "scaled" beside "no mode" would leave the units unresolvable.
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    restored = read_manifest(
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="heat",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            run_meta=_run_meta(),
            contrast_cfg=_contrast_cfg(),
            signal_scaling_mode=None,
        )
    )
    assert restored.signal_scaling is False
    assert restored.signal_scaling_mode is None


def test_the_manifest_records_why_a_run_was_excluded(tmp_path: Path) -> None:
    """A report that says a run was dropped without saying why gives nothing to act on."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        contrast_cfg=_contrast_cfg(),
        run_meta=_run_meta(
            skipped_runs=[
                {"run_label": "run-03", "reason": "no events"},
                {"run_label": "run-04", "reason": "confounds file missing"},
            ]
        ),
    )
    excluded = read_manifest(written).excluded_runs
    assert len(excluded) == 2
    assert excluded[0][1] == "no events"
    assert "run" in excluded[0][0]


def test_the_space_comes_from_what_was_actually_fit(tmp_path: Path) -> None:
    """A manifest that misreports its space sends the glass brain the wrong way."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    mni = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        run_meta=_run_meta(analysis_space="MNI152NLin2009cAsym"),
        contrast_cfg=_contrast_cfg(),
    )
    assert read_manifest(mni).space == "mni"

    native = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        run_meta=_run_meta(analysis_space="T1w"),
        contrast_cfg=_contrast_cfg(),
    )
    assert read_manifest(native).space == "native"


def test_a_missing_analysis_space_is_rejected(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(ValueError, match="analysis_space"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            run_meta=_run_meta(analysis_space=None),
            contrast_cfg=_contrast_cfg(),
        )


def test_a_manifest_without_an_included_run_is_rejected(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(ValueError, match="at least one included run"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            contrast_cfg=_contrast_cfg(),
            run_meta=_run_meta(
                included_bold_paths=[],
                included_confounds_paths=[],
            ),
        )


def test_a_manifest_without_the_fitted_analysis_mask_is_rejected(
    tmp_path: Path,
) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(ValueError, match="fitted analysis mask"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            run_meta=_run_meta(),
            contrast_cfg=_contrast_cfg(),
        )


def test_included_run_labels_are_derived_from_the_bold_filenames(
    tmp_path: Path,
) -> None:
    """The header names runs, so they need names a reader recognises."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        contrast_cfg=_contrast_cfg(),
        run_meta=_run_meta(
            included_bold_paths=[
                "/d/sub-01_task-heat_run-01_bold.nii.gz",
                "/d/sub-01_task-heat_run-02_bold.nii.gz",
            ],
            included_confounds_paths=[None, None],
            retained_frame_indices=[[0], [0]],
            skipped_runs=[],
        ),
    )
    assert read_manifest(written).included_runs == ("run-01", "run-02")


def test_one_run_without_a_run_entity_is_recorded_as_runless(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        contrast_cfg=_contrast_cfg(),
        run_meta=_run_meta(
            included_bold_paths=["/d/sub-01_task-heat_bold.nii.gz"],
        ),
    )

    assert read_manifest(written).included_runs == ("runless",)


def test_multiple_runs_without_run_entities_are_rejected(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(ValueError, match="run entity"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            contrast_cfg=_contrast_cfg(),
            run_meta=_run_meta(
                included_bold_paths=[
                    "/d/sub-01_task-heat_acq-a_bold.nii.gz",
                    "/d/sub-01_task-heat_acq-b_bold.nii.gz",
                ],
                included_confounds_paths=[None, None],
            ),
        )


def test_duplicate_run_entities_are_rejected(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(ValueError, match="unique"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            contrast_cfg=_contrast_cfg(),
            run_meta=_run_meta(
                included_bold_paths=[
                    "/d/sub-01_task-heat_run-01_acq-a_bold.nii.gz",
                    "/d/sub-01_task-heat_run-01_acq-b_bold.nii.gz",
                ],
                included_confounds_paths=[None, None],
            ),
        )


@pytest.mark.parametrize(
    ("skipped_runs", "error", "message"),
    [
        (["run-02"], TypeError, "mapping"),
        ([{"reason": "motion"}], ValueError, "run_label"),
        ([{"run_index": 2, "reason": "motion"}], ValueError, "run_label"),
        ([{"run_label": "", "reason": "motion"}], ValueError, "BIDS run label"),
        ([{"run_label": "run-02"}], ValueError, "reason"),
        ([{"run_label": "run-02", "reason": "  "}], ValueError, "reason"),
    ],
)
def test_malformed_excluded_run_provenance_is_rejected(
    tmp_path: Path,
    skipped_runs: list[object],
    error: type[Exception],
    message: str,
) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(error, match=message):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            contrast_cfg=_contrast_cfg(),
            run_meta=_run_meta(skipped_runs=skipped_runs),
        )


def test_a_missing_confounds_entry_does_not_become_a_path(tmp_path: Path) -> None:
    """nilearn allows a run without confounds; None must not serialise as 'None'."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        run_meta=_run_meta(included_confounds_paths=[None]),
        contrast_cfg=_contrast_cfg(),
    )
    assert read_manifest(written).confounds_paths == ()


def test_the_manifest_rejects_a_missing_tr(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(ValueError, match="TR"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            run_meta=_run_meta(tr=None),
            contrast_cfg=_contrast_cfg(),
        )


def test_writing_a_manifest_surfaces_an_invalid_run_record(tmp_path: Path) -> None:
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    with pytest.raises(TypeError, match="run_meta must be a mapping"):
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            run_meta="not a mapping at all",  # type: ignore[arg-type]
            contrast_cfg=_contrast_cfg(),
        )


def test_manifest_records_run_inference_paths_and_scalars(tmp_path: Path) -> None:
    """Both artefacts and the scalars a panel reads without reopening a TSV."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()
    sign_flip = contrast_dir / "signflip.tsv"
    sign_flip.write_text("pattern\tsigns\tmax_abs_z\n1\t+++\t4.0\n")
    influence = contrast_dir / "influence.tsv"
    influence.write_text("dropped_run\tsurvivors\tdelta\tmax_abs_z\tcorrelation\n")

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        mask=stat_map,
        mask_is_analysis_mask=True,
        run_meta=_run_meta(),
        contrast_cfg=_contrast_cfg(),
        sign_flip_null_tsv=sign_flip,
        run_influence_tsv=influence,
        sign_flip_fwe_height=7.02,
        sign_flip_fwe_survivors=38,
        sign_flip_global_p=0.0606,
        sign_flip_p_floor=0.0606,
        sign_flip_n_patterns=32,
        sign_flip_n_runs=6,
        sign_flip_observed_max=8.87,
    )

    manifest = read_manifest(written)
    assert manifest.sign_flip_null_tsv == sign_flip
    assert manifest.run_influence_tsv == influence
    assert manifest.sign_flip_fwe_height == 7.02
    assert manifest.sign_flip_fwe_survivors == 38
    assert manifest.sign_flip_p_floor == 0.0606
    assert manifest.sign_flip_n_runs == 6
    assert manifest.sign_flip_observed_max == 8.87


def test_sign_flip_scalars_absent_when_no_null_was_computed(tmp_path: Path) -> None:
    """A single-run contrast records nothing rather than zeros."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    manifest = read_manifest(
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            run_meta=_run_meta(),
            contrast_cfg=_contrast_cfg(),
        )
    )
    assert manifest.sign_flip_null_tsv is None
    assert manifest.sign_flip_fwe_height is None
    assert manifest.sign_flip_p_floor is None
    assert manifest.run_influence_tsv is None


def test_a_height_without_its_null_is_rejected(tmp_path: Path) -> None:
    """A height with no enumerated null cannot be checked against anything."""
    import dataclasses

    from fmri_pipeline.analysis.report.manifest import validate_manifest

    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()
    sign_flip = contrast_dir / "signflip.tsv"
    sign_flip.write_text("pattern\tsigns\tmax_abs_z\n1\t+++\t4.0\n")

    manifest = read_manifest(
        write_report_manifest(
            contrast_dir=contrast_dir,
            subject="sub-01",
            task="heat",
            contrast_name="c",
            stat_map=stat_map,
            mask=stat_map,
            mask_is_analysis_mask=True,
            run_meta=_run_meta(),
            contrast_cfg=_contrast_cfg(),
            sign_flip_null_tsv=sign_flip,
            sign_flip_fwe_height=7.02,
        )
    )
    validate_manifest(manifest)  # the pair is intact

    with pytest.raises(ValueError, match="sign_flip_null_tsv and sign_flip_fwe_height"):
        validate_manifest(dataclasses.replace(manifest, sign_flip_null_tsv=None))

    with pytest.raises(ValueError, match="sign_flip_null_tsv and sign_flip_fwe_height"):
        validate_manifest(dataclasses.replace(manifest, sign_flip_fwe_height=None))
