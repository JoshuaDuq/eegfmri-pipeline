from __future__ import annotations

import json
from pathlib import Path

from fmri_pipeline.analysis.report.manifest import (
    sample_masks_from_confounds,
    ContrastManifest,
    discover_manifests,
    read_manifest,
    write_manifest,
)


def _manifest(**overrides) -> ContrastManifest:
    base = dict(
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
        included_runs=("run-01", "run-02"),
        excluded_runs=(("run-03", "no events file"),),
        bold_paths=(Path("/d/run-01_bold.nii.gz"),),
        confounds_paths=(Path("/d/run-01_conf.tsv"),),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="motion+compcor",
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


def test_exclusions_keep_their_reasons(tmp_path: Path) -> None:
    restored = read_manifest(write_manifest(_manifest(), tmp_path / "m.json"))
    assert restored.excluded_runs == (("run-03", "no events file"),)


def test_the_manifest_is_human_readable_json(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    assert payload["contrast_name"] == "heat-warm"
    assert payload["z_threshold"] == 2.3


def test_discovery_finds_every_contrast_of_one_subject_and_task(tmp_path: Path) -> None:
    root = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat"
    for name in ("heat-warm", "heat-rest"):
        directory = root / f"contrast-{name}"
        directory.mkdir(parents=True)
        write_manifest(_manifest(contrast_name=name), directory / "report_manifest.json")

    found = discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")
    assert sorted(m.contrast_name for m in found) == ["heat-rest", "heat-warm"]


def test_discovery_ignores_another_task(tmp_path: Path) -> None:
    for task in ("heat", "rest"):
        directory = (
            tmp_path / "sub-01" / "fmri" / "first_level" / f"task-{task}" / "contrast-a"
        )
        directory.mkdir(parents=True)
        write_manifest(_manifest(task=task), directory / "report_manifest.json")

    found = discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")
    assert [m.task for m in found] == ["heat"]


def test_discovery_returns_empty_when_nothing_has_been_fit(tmp_path: Path) -> None:
    assert discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat") == []


def test_an_unreadable_manifest_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    directory = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat" / "contrast-a"
    directory.mkdir(parents=True)
    (directory / "report_manifest.json").write_text("{ not json")

    assert discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat") == []


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


def test_censoring_every_frame_is_refused_rather_than_leaving_nothing(
    tmp_path: Path,
) -> None:
    """A confounds file that flags every frame would otherwise erase the run.

    An empty keep-mask makes the carpet unstandardisable and the tSNR
    uncomputable, so the QC panels that exist to show what censoring did are the
    ones censoring destroys. Keeping the frames and saying so preserves the
    measurement.
    """
    import pandas as pd

    path = tmp_path / "confounds.tsv"
    pd.DataFrame({"non_steady_state_outlier00": [1, 1, 1]}).to_csv(
        path, sep="\t", index=False
    )
    mask = sample_masks_from_confounds([path])[0]
    assert mask.tolist() == [True, True, True]


def test_partial_censoring_is_still_applied(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "confounds.tsv"
    pd.DataFrame({"non_steady_state_outlier00": [1, 0, 0, 1]}).to_csv(
        path, sep="\t", index=False
    )
    mask = sample_masks_from_confounds([path])[0]
    assert mask.tolist() == [False, True, True, False]
