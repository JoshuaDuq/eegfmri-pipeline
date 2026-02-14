from __future__ import annotations

from eeg_pipeline.utils.config.overrides import apply_runtime_overrides


def test_apply_runtime_overrides_updates_only_provided_values() -> None:
    cfg: dict[str, object] = {"project": {"task": "old"}, "paths": {"bids_root": "/old"}}
    apply_runtime_overrides(
        cfg,
        task="new_task",
        source_root="/src",
        bids_root="/bids",
        bids_fmri_root="/fmri",
        deriv_root="/deriv",
    )

    assert cfg["project"]["task"] == "new_task"
    assert cfg["paths"]["source_data"] == "/src"
    assert cfg["paths"]["bids_root"] == "/bids"
    assert cfg["paths"]["bids_fmri_root"] == "/fmri"
    assert cfg["paths"]["deriv_root"] == "/deriv"


def test_apply_runtime_overrides_ignores_none_values() -> None:
    cfg: dict[str, object] = {"project": {"task": "keep"}, "paths": {"bids_root": "/keep"}}
    apply_runtime_overrides(cfg)

    assert cfg["project"]["task"] == "keep"
    assert cfg["paths"]["bids_root"] == "/keep"
