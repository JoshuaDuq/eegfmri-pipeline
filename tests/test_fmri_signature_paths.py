from __future__ import annotations

from pathlib import Path

from fmri_pipeline.utils.signature_paths import discover_signature_root


class _BadConfig:
    def get(self, *_args, **_kwargs):
        raise RuntimeError("bad config")


def test_discover_signature_root_prefers_existing_config_path(tmp_path: Path) -> None:
    configured = tmp_path / "configured_signatures"
    configured.mkdir(parents=True, exist_ok=True)
    config = {"paths": {"signature_dir": str(configured)}}

    discovered = discover_signature_root(config, tmp_path / "derivatives")
    assert discovered == configured


def test_discover_signature_root_falls_back_to_external_sibling(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)

    discovered = discover_signature_root({}, deriv_root)
    assert discovered == external


def test_discover_signature_root_returns_none_on_invalid_inputs() -> None:
    assert discover_signature_root(_BadConfig(), object()) is None
