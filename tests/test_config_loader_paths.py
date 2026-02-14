from __future__ import annotations

from pathlib import Path

from eeg_pipeline.utils.config import loader


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


def test_overrides_path_falls_back_to_legacy_location(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EEG_PIPELINE_TUI_OVERRIDES", raising=False)
    monkeypatch.setattr(loader, "get_project_root", lambda: tmp_path)
    legacy = tmp_path / "eeg_pipeline" / "data" / "derivatives" / ".tui_overrides.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("{}", encoding="utf-8")

    config_path = tmp_path / "eeg_pipeline" / "utils" / "config" / "eeg_config.yaml"
    overrides_path = loader._get_overrides_path(config_path)
    assert overrides_path == legacy
