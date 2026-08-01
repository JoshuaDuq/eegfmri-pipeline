from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmri_pipeline.analysis.report.manifest import (
    REPORT_MANIFEST_SCHEMA_VERSION,
    ContrastManifest,
    discover_manifests,
    read_manifest,
    sample_masks_from_confounds,
    write_manifest,
)


def _manifest(**overrides) -> ContrastManifest:
    base = dict(
        schema_version=REPORT_MANIFEST_SCHEMA_VERSION,
        subject="sub-01",
        task="heat",
        contrast_name="heat-warm",
        space="native",
        stat_map=Path("/d/z.nii.gz"),
        effect_map=Path("/d/eff.nii.gz"),
        variance_map=None,
        mask=Path("/d/mask.nii.gz"),
        threshold_mode="z",
        z_threshold=2.3,
        fdr_q=0.05,
        cluster_min_voxels=10,
        two_sided=True,
        radiological=False,
        design_matrices=(Path("/d/run-01_dm.tsv"),),
        contrast_vector=(1.0, -1.0),
        contrast_columns=("heat", "warm"),
        included_runs=("run-01",),
        excluded_runs=(("run-03", "no events file"),),
        bold_paths=(Path("/d/run-01_bold.nii.gz"),),
        confounds_paths=(Path("/d/run-01_conf.tsv"),),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="motion+compcor",
        mask_is_analysis_mask=True,
        residual_paths=(Path("/d/run-01_residual.nii.gz"),),
        predicted_paths=(Path("/d/run-01_predicted.nii.gz"),),
        retained_frame_indices=((0,),),
    )
    base.update(overrides)
    return ContrastManifest(**base)


def test_a_manifest_round_trips_through_json(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "report_manifest.json")
    assert read_manifest(path) == _manifest()


def test_paths_survive_the_round_trip_as_paths(tmp_path: Path) -> None:
    restored = read_manifest(write_manifest(_manifest(), tmp_path / "m.json"))
    assert isinstance(restored.stat_map, Path)
    assert restored.variance_map is None


def test_the_manifest_carries_exact_model_fit_series() -> None:
    manifest = _manifest()

    assert hasattr(manifest, "residual_paths")
    assert hasattr(manifest, "predicted_paths")
    assert manifest.model_fit_series_space == "unwhitened-model-response"
    assert manifest.retained_frame_indices == ((0,),)


def test_exclusions_keep_their_reasons(tmp_path: Path) -> None:
    restored = read_manifest(write_manifest(_manifest(), tmp_path / "m.json"))
    assert restored.excluded_runs == (("run-03", "no events file"),)


def test_the_manifest_is_human_readable_json(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["contrast_name"] == "heat-warm"
    assert payload["z_threshold"] == 2.3


def test_a_manifest_without_a_schema_version_is_rejected(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    payload.pop("schema_version", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_manifest(path)


def test_an_unsupported_manifest_schema_is_rejected(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported report manifest schema"):
        read_manifest(path)


def test_unknown_manifest_fields_are_rejected(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    payload["unrecognized_artifact"] = "silent schema drift"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown report manifest field"):
        read_manifest(path)


def test_a_missing_optional_field_reads_as_its_default(tmp_path: Path) -> None:
    """Additive fields must not invalidate manifests written before they existed.

    This relaxes an earlier rule that rejected *any* absent field. That rule made the
    schema un-extendable: adding one diagnostic made every manifest already on disk
    unreadable, and the only recovery was refitting the model -- while the report's
    whole purpose is rendering from a tree an earlier run produced.

    The version gate still covers incompatible change, and unknown fields are still
    rejected, so schema drift in the other direction is still caught. What is given up
    is detecting a writer that quietly stops emitting an optional field; for a field
    whose default *is* absence, that is not a detectable difference anyway.
    """
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    payload.pop("run_effect_map")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_manifest(path).run_effect_map is None


def test_a_missing_required_field_is_still_rejected(tmp_path: Path) -> None:
    """A field with no default carries meaning that absence cannot stand in for."""
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    payload.pop("t_r")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing report manifest field"):
        read_manifest(path)


def test_a_null_stat_map_is_rejected(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    payload["stat_map"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stat_map"):
        read_manifest(path)


def test_run_labels_must_align_with_bold_paths(tmp_path: Path) -> None:
    manifest = _manifest(included_runs=("run-01", "run-02"))

    with pytest.raises(ValueError, match="included_runs"):
        write_manifest(manifest, tmp_path / "m.json")


def test_run_labels_must_match_their_bold_entities(tmp_path: Path) -> None:
    manifest = _manifest(included_runs=("run-02",))

    with pytest.raises(ValueError, match="BOLD run entities"):
        write_manifest(manifest, tmp_path / "m.json")


def test_a_run_cannot_be_both_included_and_excluded(tmp_path: Path) -> None:
    manifest = _manifest(excluded_runs=(("run-01", "motion"),))

    with pytest.raises(ValueError, match="both included and excluded"):
        write_manifest(manifest, tmp_path / "m.json")


def test_contrast_weights_must_align_with_their_columns(tmp_path: Path) -> None:
    manifest = _manifest(
        contrast_vector=(1.0,),
        contrast_columns=("heat", "warm"),
    )

    with pytest.raises(ValueError, match="contrast_vector"):
        write_manifest(manifest, tmp_path / "m.json")


def test_run_level_effect_and_variance_are_an_indivisible_pair(tmp_path: Path) -> None:
    manifest = _manifest(run_effect_map=tmp_path / "effects.nii.gz")

    with pytest.raises(ValueError, match="run_effect_map and run_variance_map"):
        write_manifest(manifest, tmp_path / "m.json")


def test_model_fit_series_are_required_for_every_included_run(tmp_path: Path) -> None:
    manifest = _manifest(residual_paths=())

    with pytest.raises(ValueError, match="residual_paths and predicted_paths"):
        write_manifest(manifest, tmp_path / "m.json")


def test_model_fit_series_must_align_with_included_runs(tmp_path: Path) -> None:
    manifest = _manifest(
        predicted_paths=(
            Path("/d/run-01_predicted.nii.gz"),
            Path("/d/run-02_predicted.nii.gz"),
        )
    )

    with pytest.raises(ValueError, match="model-fit paths must align"):
        write_manifest(manifest, tmp_path / "m.json")


def test_a_manifest_rejects_a_nonpositive_tr(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="t_r"):
        write_manifest(_manifest(t_r=0.0), tmp_path / "m.json")


def test_discovery_finds_every_contrast_of_one_subject_and_task(tmp_path: Path) -> None:
    root = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat"
    for name in ("heat-warm", "heat-rest"):
        directory = root / f"contrast-{name}"
        directory.mkdir(parents=True)
        write_manifest(_manifest(contrast_name=name), directory / "report_manifest.json")

    found = discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")
    assert sorted(m.contrast_name for m in found) == ["heat-rest", "heat-warm"]


def test_discovery_rejects_a_manifest_in_the_wrong_task_directory(tmp_path: Path) -> None:
    directory = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat" / "contrast-a"
    directory.mkdir(parents=True)
    write_manifest(_manifest(task="rest"), directory / "report_manifest.json")

    with pytest.raises(ValueError, match="declares task 'rest'"):
        discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")


def test_discovery_returns_empty_when_nothing_has_been_fit(tmp_path: Path) -> None:
    assert discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat") == []


def test_an_unreadable_manifest_stops_discovery(tmp_path: Path) -> None:
    directory = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat" / "contrast-a"
    directory.mkdir(parents=True)
    (directory / "report_manifest.json").write_text("{ not json")

    with pytest.raises(json.JSONDecodeError):
        discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")


def test_discovery_is_ordered_so_a_regenerated_report_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat"
    for name in ("zeta", "alpha", "mid"):
        directory = root / f"contrast-{name}"
        directory.mkdir(parents=True)
        write_manifest(_manifest(contrast_name=name), directory / "report_manifest.json")

    found = discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")
    assert [m.contrast_name for m in found] == ["alpha", "mid", "zeta"]


def test_sample_masks_follow_the_censoring_the_model_applied(tmp_path: Path) -> None:
    # The report must censor exactly what the GLM censored, or it describes a
    # different analysis than the one that ran.
    import pandas as pd

    from fmri_pipeline.analysis.report.manifest import sample_masks_from_confounds

    frame = pd.DataFrame(
        {
            "trans_x": [0.0] * 6,
            "non_steady_state_outlier00": [1, 0, 0, 0, 0, 0],
            "motion_outlier00": [0, 0, 0, 1, 0, 0],
        }
    )
    path = tmp_path / "conf.tsv"
    frame.to_csv(path, sep="\t", index=False)

    (mask,) = sample_masks_from_confounds([path])
    assert mask.tolist() == [False, True, True, False, True, True]


def test_a_confounds_file_with_no_censor_columns_keeps_every_frame(
    tmp_path: Path,
) -> None:
    import pandas as pd

    from fmri_pipeline.analysis.report.manifest import sample_masks_from_confounds

    path = tmp_path / "conf.tsv"
    pd.DataFrame({"trans_x": [0.0] * 4}).to_csv(path, sep="\t", index=False)

    (mask,) = sample_masks_from_confounds([path])
    assert mask.all() and mask.size == 4


def test_censoring_every_frame_is_rejected(
    tmp_path: Path,
) -> None:
    import pandas as pd

    path = tmp_path / "confounds.tsv"
    pd.DataFrame({"non_steady_state_outlier00": [1, 1, 1]}).to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="flags every frame"):
        sample_masks_from_confounds([path])


def test_partial_censoring_is_still_applied(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "confounds.tsv"
    pd.DataFrame({"non_steady_state_outlier00": [1, 0, 0, 1]}).to_csv(path, sep="\t", index=False)
    mask = sample_masks_from_confounds([path])[0]
    assert mask.tolist() == [False, True, True, False]


# --- run-level maps -------------------------------------------------------


def test_run_level_maps_survive_a_write_and_read(tmp_path) -> None:
    from fmri_pipeline.analysis.report.manifest import (
        ContrastManifest,
        read_manifest,
        write_manifest,
    )

    manifest = ContrastManifest(
        schema_version=REPORT_MANIFEST_SCHEMA_VERSION,
        subject="sub-01",
        task="heat",
        contrast_name="a-b",
        space="mni",
        stat_map=tmp_path / "z.nii.gz",
        effect_map=None,
        variance_map=None,
        mask=tmp_path / "mask.nii.gz",
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
        excluded_runs=(),
        bold_paths=(tmp_path / "run-01_bold.nii.gz", tmp_path / "run-02_bold.nii.gz"),
        confounds_paths=(),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=True,
        signal_scaling_mode="voxel-mean",
        confound_strategy="motion24",
        mask_is_analysis_mask=True,
        run_effect_map=tmp_path / "perrun_effect.nii.gz",
        run_variance_map=tmp_path / "perrun_variance.nii.gz",
        residual_paths=(
            tmp_path / "run-01_residual.nii.gz",
            tmp_path / "run-02_residual.nii.gz",
        ),
        predicted_paths=(
            tmp_path / "run-01_predicted.nii.gz",
            tmp_path / "run-02_predicted.nii.gz",
        ),
        retained_frame_indices=((0,), (0,)),
    )
    path = write_manifest(manifest, tmp_path / "report_manifest.json")
    restored = read_manifest(path)
    assert restored.run_effect_map == tmp_path / "perrun_effect.nii.gz"
    assert restored.run_variance_map == tmp_path / "perrun_variance.nii.gz"
