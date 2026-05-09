"""Tests for studies.pain_study.study1.output_cleanup direct behavior."""

from __future__ import annotations

from pathlib import Path

from studies.pain_study.study1.output_cleanup import (
    prune_windowed_feature_artifacts,
    remove_appledouble_sidecars,
)


###################################################################
# remove_appledouble_sidecars
###################################################################


def test_remove_appledouble_sidecars_cleans_nested_tree(tmp_path) -> None:
    sidecar_a = tmp_path / "._sidecar"
    sidecar_a.write_text("", encoding="utf-8")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    sidecar_b = nested / "._other"
    sidecar_b.write_text("", encoding="utf-8")
    real_file = nested / "data.parquet"
    real_file.write_text("content", encoding="utf-8")

    removed = remove_appledouble_sidecars(tmp_path)

    assert removed == 2
    assert not sidecar_a.exists()
    assert not sidecar_b.exists()
    assert real_file.exists()


def test_remove_appledouble_sidecars_returns_zero_for_missing_root(tmp_path) -> None:
    assert remove_appledouble_sidecars(tmp_path / "nonexistent") == 0


def test_remove_appledouble_sidecars_returns_zero_when_no_sidecars(tmp_path) -> None:
    (tmp_path / "real_file.txt").write_text("ok", encoding="utf-8")
    assert remove_appledouble_sidecars(tmp_path) == 0


###################################################################
# prune_windowed_feature_artifacts
###################################################################


def _create_windowed_artifacts(feature_root: Path, subject: str, family: str) -> None:
    family_dir = feature_root / subject / "eeg" / "features" / family
    metadata_dir = family_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    (family_dir / f"features_{family}.parquet").write_text("main", encoding="utf-8")
    (family_dir / f"features_{family}_active.parquet").write_text("dup", encoding="utf-8")
    (metadata_dir / f"features_{family}_active.json").write_text("{}", encoding="utf-8")
    (metadata_dir / "extraction_config_active.json").write_text("{}", encoding="utf-8")
    (metadata_dir / "extraction_config_baseline.json").write_text("{}", encoding="utf-8")


def test_prune_windowed_removes_active_and_baseline_artifacts(tmp_path) -> None:
    _create_windowed_artifacts(tmp_path, "sub-0001", "erds")

    removed = prune_windowed_feature_artifacts(
        feature_root=tmp_path,
        subjects=["sub-0001"],
        feature_families=["erds"],
    )

    assert removed == 4
    erds_dir = tmp_path / "sub-0001" / "eeg" / "features" / "erds"
    assert (erds_dir / "features_erds.parquet").exists()
    assert not (erds_dir / "features_erds_active.parquet").exists()
    assert not (erds_dir / "metadata" / "extraction_config_active.json").exists()
    assert not (erds_dir / "metadata" / "extraction_config_baseline.json").exists()


def test_prune_windowed_skips_missing_families(tmp_path) -> None:
    removed = prune_windowed_feature_artifacts(
        feature_root=tmp_path,
        subjects=["sub-0001"],
        feature_families=["erds"],
    )

    assert removed == 0


def test_prune_windowed_handles_multiple_subjects_and_families(tmp_path) -> None:
    _create_windowed_artifacts(tmp_path, "sub-0001", "erds")
    _create_windowed_artifacts(tmp_path, "sub-0002", "bursts")

    removed = prune_windowed_feature_artifacts(
        feature_root=tmp_path,
        subjects=["sub-0001", "sub-0002"],
        feature_families=["erds", "bursts"],
    )

    assert removed == 8
