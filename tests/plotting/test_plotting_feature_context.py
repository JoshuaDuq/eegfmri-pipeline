from __future__ import annotations

import logging
from pathlib import Path

from eeg_pipeline.plotting.features.context import FeaturePlotContext


def _build_context(features_dir: Path, plots_dir: Path) -> FeaturePlotContext:
    return FeaturePlotContext(
        subject="0000",
        plots_dir=plots_dir,
        features_dir=features_dir,
        logger=logging.getLogger("test.plotting.feature_context"),
    )


def test_collect_feature_paths_uses_category_directory_only_when_present(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    plots_dir = tmp_path / "plots"
    power_dir = features_dir / "power"
    power_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    (power_dir / "features_power_baseline.parquet").touch()
    (power_dir / "features_power_plateau.parquet").touch()
    (features_dir / "features_power_baseline.parquet").touch()
    (features_dir / "features_power_plateau.parquet").touch()

    context = _build_context(features_dir, plots_dir)
    context.time_range_suffixes = ["baseline", "plateau"]

    paths = context._collect_feature_paths("features_power", [".parquet", ".tsv"])

    assert paths == [
        power_dir / "features_power_baseline.parquet",
        power_dir / "features_power_plateau.parquet",
    ]


def test_collect_feature_paths_falls_back_to_root_when_category_has_no_matching_files(
    tmp_path: Path,
) -> None:
    features_dir = tmp_path / "features"
    plots_dir = tmp_path / "plots"
    (features_dir / "power").mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    root_file = features_dir / "features_power.parquet"
    root_file.touch()

    context = _build_context(features_dir, plots_dir)
    paths = context._collect_feature_paths("features_power", [".parquet", ".tsv"])

    assert paths == [root_file]


def test_collect_feature_paths_prefers_single_extension_family(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    plots_dir = tmp_path / "plots"
    power_dir = features_dir / "power"
    power_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    parquet_file = power_dir / "features_power_baseline.parquet"
    tsv_file = power_dir / "features_power_baseline.tsv"
    parquet_file.touch()
    tsv_file.touch()

    context = _build_context(features_dir, plots_dir)
    context.time_range_suffixes = ["baseline"]

    paths = context._collect_feature_paths("features_power", [".parquet", ".tsv"])

    assert paths == [parquet_file]


def test_load_feature_tables_respects_category_filter_sourcelocalization(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    plots_dir = tmp_path / "plots"
    features_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    context = _build_context(features_dir, plots_dir)
    context.feature_categories_to_load = {"sourcelocalization"}

    requested_stems: list[str] = []

    def fake_collect(stem: str, exts: list[str]) -> list[Path]:
        requested_stems.append(stem)
        return []

    context._collect_feature_paths = fake_collect  # type: ignore[method-assign]
    context._load_feature_set = lambda paths, mode, stem: None  # type: ignore[method-assign]

    context._load_feature_tables()

    assert requested_stems == ["features_sourcelocalization"]


def test_load_feature_tables_respects_category_filter_pac_aliases(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    plots_dir = tmp_path / "plots"
    features_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    context = _build_context(features_dir, plots_dir)
    context.feature_categories_to_load = {"pac"}

    requested_stems: list[str] = []

    def fake_collect(stem: str, exts: list[str]) -> list[Path]:
        requested_stems.append(stem)
        return []

    context._collect_feature_paths = fake_collect  # type: ignore[method-assign]
    context._load_feature_set = lambda paths, mode, stem: None  # type: ignore[method-assign]

    context._load_feature_tables()

    assert requested_stems == ["features_pac", "features_pac_trials", "features_pac_time"]
