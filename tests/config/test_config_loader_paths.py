from __future__ import annotations

from pathlib import Path

from eeg_pipeline.utils.config import loader
from eeg_pipeline.utils.config.overrides import apply_runtime_overrides


def _reset_loader_cache() -> None:
    loader._CONFIG = None
    loader._CONFIG_PATH = None
    loader._CONFIG_MTIME = None


def test_default_config_paths_resolve_to_repo_data(monkeypatch) -> None:
    _reset_loader_cache()
    missing_overrides = Path("/tmp/__no_such_tui_overrides__.json")
    monkeypatch.setenv("EEG_PIPELINE_TUI_OVERRIDES", str(missing_overrides))

    cfg = loader.load_config(apply_thread_limits=False)

    assert "/data/derivatives" in str(cfg.get("paths.deriv_root"))
    assert "/eeg_pipeline/data/derivatives" not in str(cfg.get("paths.deriv_root"))
    assert "/data/bids_output/eeg" in str(cfg.get("paths.bids_root"))
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


def test_resolve_single_path_keeps_docker_image_like_values() -> None:
    config_dir = Path("/tmp/config")
    project_root = Path("/tmp/project")
    resolved = loader._resolve_single_path("nipreps/fmriprep:25.2.4", config_dir, project_root)
    assert resolved == "nipreps/fmriprep:25.2.4"


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


def test_resolve_single_path_uses_project_root_for_known_prefixes(tmp_path) -> None:
    config_dir = tmp_path / "cfg"
    project_root = tmp_path / "repo"
    config_dir.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    resolved_data = loader._resolve_single_path("data/derivatives", config_dir, project_root)
    resolved_pkg = loader._resolve_single_path("eeg_pipeline/data", config_dir, project_root)

    assert resolved_data == str((project_root / "data/derivatives").resolve())
    assert resolved_pkg == str((project_root / "eeg_pipeline/data").resolve())


def test_get_condition_column_candidates_uses_config_only() -> None:
    config = loader.ConfigDict({"event_columns": {"condition": []}})

    assert loader.get_condition_column_candidates(config) == []
