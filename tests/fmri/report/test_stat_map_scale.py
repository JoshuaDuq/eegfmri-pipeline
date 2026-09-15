"""Inference must run on a z-scaled statistic, per nilearn's documented contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from fmri_pipeline.analysis.report import manifest as manifest_mod


def test_a_manifest_naming_a_z_statistic_is_thresholdable() -> None:
    assert manifest_mod.stat_map_is_z_scaled(
        stat_map_output_type="z_score", stat_map=Path("z.nii.gz"), effect_map=Path("e.nii.gz")
    )


def test_a_manifest_naming_effect_size_is_not_thresholdable() -> None:
    # nilearn's threshold_stats_img: a non-z input makes the computed threshold
    # "not rigorous and likely meaningless".
    assert not manifest_mod.stat_map_is_z_scaled(
        stat_map_output_type="effect_size", stat_map=Path("e.nii.gz"), effect_map=Path("e.nii.gz")
    )


def test_an_older_manifest_whose_stat_map_is_its_effect_map_is_not_thresholdable() -> None:
    # Schema 2 recorded no output type. The one case it can still be caught in is the
    # one that shipped: stat_map and effect_map pointing at the same file.
    assert not manifest_mod.stat_map_is_z_scaled(
        stat_map_output_type=None,
        stat_map=Path("/x/sub-01_stat-effect_size_h.nii.gz"),
        effect_map=Path("/x/sub-01_stat-effect_size_h.nii.gz"),
    )


def test_an_older_manifest_with_distinct_maps_is_taken_as_z() -> None:
    # The historical default was output_type z-score, which is correct; refusing it
    # would drop working panels from every report already on disk.
    assert manifest_mod.stat_map_is_z_scaled(
        stat_map_output_type=None,
        stat_map=Path("/x/sub-01_stat-z_score_h.nii.gz"),
        effect_map=Path("/x/sub-01_stat-effect_size_h.nii.gz"),
    )


def test_a_manifest_with_no_effect_map_is_taken_as_z() -> None:
    assert manifest_mod.stat_map_is_z_scaled(
        stat_map_output_type=None, stat_map=Path("s.nii.gz"), effect_map=None
    )


@pytest.mark.parametrize("kind", ["effect_size", "effect_variance", "stat", "p_value"])
def test_no_non_z_output_type_is_thresholdable(kind: str) -> None:
    assert not manifest_mod.stat_map_is_z_scaled(
        stat_map_output_type=kind, stat_map=Path("a.nii.gz"), effect_map=Path("b.nii.gz")
    )


def test_the_output_type_round_trips_through_the_manifest_file(tmp_path: Path) -> None:
    # The field is only load-bearing if it survives the trip to disk and back.
    from tests.fmri.report.test_manifest import _manifest

    written = manifest_mod.write_manifest(
        _manifest(stat_map_output_type="z_score"), tmp_path / "report_manifest.json"
    )
    assert manifest_mod.read_manifest(written).stat_map_output_type == "z_score"


def test_a_manifest_written_without_the_field_reads_back_as_none(tmp_path: Path) -> None:
    # Schema 2 manifests already on disk have no such key; they must still load.
    from tests.fmri.report.test_manifest import _manifest

    written = manifest_mod.write_manifest(_manifest(), tmp_path / "report_manifest.json")
    assert manifest_mod.read_manifest(written).stat_map_output_type is None


def _non_z_manifest(tmp_path: Path):
    from tests.fmri.report.test_manifest import _manifest

    shared = tmp_path / "sub-01_stat-effect_size_h.nii.gz"
    return _manifest(stat_map=shared, effect_map=shared, threshold_mode="z", z_threshold=2.3)


def test_a_non_z_stat_map_is_not_thresholded(tmp_path: Path) -> None:
    # Refusing beats relabelling: the shipped behaviour thresholded a percent signal
    # change map at |z| > 2.30 and reported "0 voxels" as this subject's result.
    import numpy as np

    from fmri_pipeline.analysis.report import subject as subject_mod

    threshold, label = subject_mod.resolve_threshold(
        _non_z_manifest(tmp_path), values=np.linspace(-1.0, 1.0, 100)
    )
    assert threshold is None
    assert "z" in label.lower() and "not" in label.lower()


def test_a_z_stat_map_still_thresholds_normally(tmp_path: Path) -> None:
    import numpy as np

    from fmri_pipeline.analysis.report import subject as subject_mod

    from tests.fmri.report.test_manifest import _manifest

    threshold, label = subject_mod.resolve_threshold(
        _manifest(
            stat_map=tmp_path / "z.nii.gz",
            effect_map=tmp_path / "e.nii.gz",
            stat_map_output_type="z_score",
            threshold_mode="z",
            z_threshold=2.3,
        ),
        values=np.linspace(-4.0, 4.0, 100),
    )
    assert threshold == pytest.approx(2.3)
    assert "2.30" in label


def test_the_inference_panels_are_skipped_for_a_non_z_map(tmp_path: Path) -> None:
    # The calibration panel fits a null "in z" and the table quotes z heights, so both
    # are as meaningless on an effect map as the threshold itself.
    from fmri_pipeline.analysis.report import subject as subject_mod

    assert not subject_mod.contrast_is_thresholdable(_non_z_manifest(tmp_path))


def test_the_inference_panels_run_for_a_z_map(tmp_path: Path) -> None:
    from fmri_pipeline.analysis.report import subject as subject_mod

    from tests.fmri.report.test_manifest import _manifest

    assert subject_mod.contrast_is_thresholdable(
        _manifest(
            stat_map=tmp_path / "z.nii.gz",
            effect_map=tmp_path / "e.nii.gz",
            stat_map_output_type="z_score",
        )
    )


def test_the_pipeline_saves_the_z_map_with_an_artifact_name() -> None:
    # _save_required takes a keyword-only artifact_name. Omitting it raised only at
    # run time, after a five-minute GLM fit had already completed.
    import inspect

    from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

    source = inspect.getsource(FmriAnalysisPipeline)
    call = source[source.index("_stat-z_score_") - 400 : source.index("_stat-z_score_") + 200]
    assert "artifact_name=" in call, "the z-map save omits the required artifact_name"
