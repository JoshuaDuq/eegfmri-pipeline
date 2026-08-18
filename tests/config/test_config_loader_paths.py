from __future__ import annotations

import os
from pathlib import Path

import pytest

from eeg_pipeline.utils.config import loader
from eeg_pipeline.utils.config.overrides import apply_runtime_overrides


def _reset_loader_cache() -> None:
    loader._CONFIG = None
    loader._CONFIG_PATH = None
    loader._CONFIG_MTIME = None
    loader._CONFIG_OVERRIDES_PATH = None
    loader._CONFIG_OVERRIDES_MTIME = None


def test_default_config_paths_resolve_to_repo_data(monkeypatch) -> None:
    _reset_loader_cache()
    missing_overrides = Path("/tmp/__no_such_tui_overrides__.json")
    monkeypatch.setenv("EEG_PIPELINE_TUI_OVERRIDES", str(missing_overrides))

    cfg = loader.load_config(apply_thread_limits=False)
    project_root = loader.get_project_root()
    expected_deriv = (project_root / "data" / "derivatives").resolve().as_posix()
    expected_bids = (project_root / "data" / "bids_output" / "eeg_linecleaned").resolve().as_posix()

    assert str(cfg.get("paths.deriv_root")) == expected_deriv
    assert "/eeg_pipeline/data/derivatives" not in str(cfg.get("paths.deriv_root"))
    assert str(cfg.get("paths.bids_root")) == expected_bids
    assert "/eeg_pipeline/data/bids_output/eeg" not in str(cfg.get("paths.bids_root"))


def test_overrides_path_defaults_to_repo_data_derivatives(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EEG_PIPELINE_TUI_OVERRIDES", raising=False)
    monkeypatch.setattr(loader, "get_project_root", lambda: tmp_path)

    config_path = tmp_path / "eeg_pipeline" / "utils" / "config" / "eeg_config.yaml"
    overrides_path = loader._get_overrides_path(config_path)
    assert overrides_path == tmp_path / "data" / "derivatives" / ".tui_overrides.json"


def test_overrides_path_ignores_legacy_location(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EEG_PIPELINE_TUI_OVERRIDES", raising=False)
    monkeypatch.setattr(loader, "get_project_root", lambda: tmp_path)
    legacy = tmp_path / "eeg_pipeline" / "data" / "derivatives" / ".tui_overrides.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("{}", encoding="utf-8")

    config_path = tmp_path / "eeg_pipeline" / "utils" / "config" / "eeg_config.yaml"
    overrides_path = loader._get_overrides_path(config_path)
    assert overrides_path == tmp_path / "data" / "derivatives" / ".tui_overrides.json"


def test_apply_config_overrides_raises_for_invalid_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        loader, "_get_overrides_path", lambda _config_path: tmp_path / ".tui_overrides.json"
    )
    overrides_path = tmp_path / ".tui_overrides.json"
    overrides_path.write_text("{bad", encoding="utf-8")

    with pytest.raises(loader.ConfigError, match="Failed to parse TUI overrides"):
        loader._apply_config_overrides({"project": {"task": "x"}}, tmp_path / "config.yaml")


def test_apply_config_overrides_raises_for_non_mapping_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        loader, "_get_overrides_path", lambda _config_path: tmp_path / ".tui_overrides.json"
    )
    overrides_path = tmp_path / ".tui_overrides.json"
    overrides_path.write_text('["not", "a", "mapping"]', encoding="utf-8")

    with pytest.raises(loader.ConfigError, match="must contain a JSON object"):
        loader._apply_config_overrides({"project": {"task": "x"}}, tmp_path / "config.yaml")


def test_load_config_returns_isolated_nested_data(monkeypatch) -> None:
    _reset_loader_cache()
    missing_overrides = Path("/tmp/__no_such_tui_overrides__.json")
    monkeypatch.setenv("EEG_PIPELINE_TUI_OVERRIDES", str(missing_overrides))

    cfg1 = loader.load_config(apply_thread_limits=False)
    original_deriv_root = cfg1.get("paths.deriv_root")

    cfg1["paths"]["deriv_root"] = "/tmp/should_not_leak"
    cfg2 = loader.load_config(apply_thread_limits=False)

    assert cfg2.get("paths.deriv_root") == original_deriv_root


def test_runtime_overrides_do_not_leak_into_cached_config(monkeypatch) -> None:
    _reset_loader_cache()
    missing_overrides = Path("/tmp/__no_such_tui_overrides__.json")
    monkeypatch.setenv("EEG_PIPELINE_TUI_OVERRIDES", str(missing_overrides))

    cfg = loader.load_config(apply_thread_limits=False)
    original_task = cfg.get("project.task")

    apply_runtime_overrides(cfg, task="__temp_runtime_task__")
    fresh_cfg = loader.load_config(apply_thread_limits=False)

    assert fresh_cfg.get("project.task") == original_task


def test_load_config_reloads_when_tui_overrides_change(tmp_path, monkeypatch) -> None:
    _reset_loader_cache()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  task: base\n", encoding="utf-8")
    overrides_path = tmp_path / ".tui_overrides.json"
    overrides_path.write_text('{"project": {"task": "first"}}', encoding="utf-8")
    monkeypatch.setenv("EEG_PIPELINE_TUI_OVERRIDES", str(overrides_path))

    first_cfg = loader.load_config(config_path=config_path, apply_thread_limits=False)
    assert first_cfg.get("project.task") == "first"

    overrides_path.write_text('{"project": {"task": "second"}}', encoding="utf-8")
    first_mtime = overrides_path.stat().st_mtime
    os.utime(overrides_path, (first_mtime + 1, first_mtime + 1))

    reloaded_cfg = loader.load_config(config_path=config_path, apply_thread_limits=False)
    assert reloaded_cfg.get("project.task") == "second"


def test_load_config_surfaces_invalid_tui_overrides_after_cache_warm(tmp_path, monkeypatch) -> None:
    _reset_loader_cache()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  task: base\n", encoding="utf-8")
    overrides_path = tmp_path / ".tui_overrides.json"
    overrides_path.write_text('{"project": {"task": "first"}}', encoding="utf-8")
    monkeypatch.setenv("EEG_PIPELINE_TUI_OVERRIDES", str(overrides_path))

    loader.load_config(config_path=config_path, apply_thread_limits=False)

    overrides_path.write_text("{bad", encoding="utf-8")
    first_mtime = overrides_path.stat().st_mtime
    os.utime(overrides_path, (first_mtime + 1, first_mtime + 1))

    with pytest.raises(loader.ConfigError, match="Failed to parse TUI overrides"):
        loader.load_config(config_path=config_path, apply_thread_limits=False)


def test_resolve_single_path_keeps_docker_image_like_values() -> None:
    config_dir = Path("/tmp/config")
    project_root = Path("/tmp/project")
    resolved = loader._resolve_single_path("nipreps/fmriprep:25.2.4", config_dir, project_root)
    assert resolved == "nipreps/fmriprep:25.2.4"


def test_resolve_single_path_keeps_windows_drive_absolute_paths(tmp_path) -> None:
    config_dir = tmp_path / "config"
    project_root = tmp_path / "project"
    config_dir.mkdir()
    project_root.mkdir()

    resolved = loader._resolve_single_path(
        "D:/EEG_fMRI_data/derivatives",
        config_dir,
        project_root,
    )

    assert resolved == "D:/EEG_fMRI_data/derivatives"


def test_resolve_paths_recursive_skips_non_path_scalar_keys(tmp_path) -> None:
    config = {
        "project": {
            "task": "task",
            "random_state": "42",
            "picks": "eeg",
            "project_root": "workspace-root",
        }
    }
    loader._resolve_paths_recursive(config, tmp_path / "config", tmp_path / "project")
    assert config["project"]["task"] == "task"
    assert config["project"]["random_state"] == "42"
    assert config["project"]["picks"] == "eeg"
    assert config["project"]["project_root"] == "workspace-root"


def test_resolve_paths_recursive_preserves_annotation_descriptions(tmp_path) -> None:
    config = {"ica": {"cardiac_review": {"marker_description": "Pulse Artifact/R"}}}

    loader._resolve_paths_recursive(config, tmp_path / "config", tmp_path / "project")

    assert config["ica"]["cardiac_review"]["marker_description"] == "Pulse Artifact/R"


def test_resolve_single_path_uses_project_root_for_known_prefixes(tmp_path) -> None:
    config_dir = tmp_path / "cfg"
    project_root = tmp_path / "repo"
    config_dir.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    resolved_data = loader._resolve_single_path("data/derivatives", config_dir, project_root)
    resolved_pkg = loader._resolve_single_path("eeg_pipeline/data", config_dir, project_root)

    assert resolved_data == (project_root / "data/derivatives").resolve().as_posix()
    assert resolved_pkg == (project_root / "eeg_pipeline/data").resolve().as_posix()


def test_get_condition_column_candidates_uses_config_only() -> None:
    config = loader.ConfigDict({"event_columns": {"condition": []}})

    assert loader.get_condition_column_candidates(config) == []


def test_require_config_value_resolves_nested_keys_from_dict_and_configdict() -> None:
    plain = {"behavior_analysis": {"statistics": {"correlation_method": "spearman"}}}
    wrapped = loader.ConfigDict(plain)

    assert (
        loader.require_config_value(plain, "behavior_analysis.statistics.correlation_method")
        == "spearman"
    )
    assert (
        loader.require_config_value(wrapped, "behavior_analysis.statistics.correlation_method")
        == "spearman"
    )


def test_require_config_value_raises_for_missing_nested_key() -> None:
    with pytest.raises(loader.ConfigError):
        loader.require_config_value({}, "behavior_analysis.statistics.correlation_method")
