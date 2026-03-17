from __future__ import annotations

from eeg_pipeline.utils.config.overrides import apply_runtime_overrides, apply_set_overrides


def test_apply_runtime_overrides_updates_only_provided_values() -> None:
    cfg: dict[str, object] = {"project": {"task": "old"}, "paths": {"bids_root": "/old"}}
    apply_runtime_overrides(
        cfg,
        task="new_task",
        source_root="/src",
        bids_root="/bids",
        bids_rest_root="/bids-rest",
        bids_fmri_root="/fmri",
        deriv_root="/deriv",
        deriv_rest_root="/deriv-rest",
    )

    assert cfg["project"]["task"] == "new_task"
    assert cfg["paths"]["source_data"] == "/src"
    assert cfg["paths"]["bids_root"] == "/bids"
    assert cfg["paths"]["bids_rest_root"] == "/bids-rest"
    assert cfg["paths"]["bids_fmri_root"] == "/fmri"
    assert cfg["paths"]["deriv_root"] == "/deriv"
    assert cfg["paths"]["deriv_rest_root"] == "/deriv-rest"


def test_apply_runtime_overrides_ignores_none_values() -> None:
    cfg: dict[str, object] = {"project": {"task": "keep"}, "paths": {"bids_root": "/keep"}}
    apply_runtime_overrides(cfg)

    assert cfg["project"]["task"] == "keep"
    assert cfg["paths"]["bids_root"] == "/keep"


def test_apply_runtime_overrides_applies_set_overrides() -> None:
    cfg: dict[str, object] = {"project": {"task": "old"}, "analysis": {"alpha": 0.05}}
    apply_runtime_overrides(
        cfg,
        set_overrides=[
            "project.task=new_task",
            "analysis.min_subjects_for_group=7",
            "analysis.alpha=0.01",
            "analysis.enabled=true",
            "analysis.metadata={\"mode\":\"strict\"}",
            "analysis.labels=[\"a\",\"b\"]",
            "analysis.optional=null",
        ],
    )

    assert cfg["project"]["task"] == "new_task"
    assert cfg["analysis"]["min_subjects_for_group"] == 7
    assert cfg["analysis"]["alpha"] == 0.01
    assert cfg["analysis"]["enabled"] is True
    assert cfg["analysis"]["metadata"] == {"mode": "strict"}
    assert cfg["analysis"]["labels"] == ["a", "b"]
    assert cfg["analysis"]["optional"] is None


def test_apply_runtime_overrides_ignores_invalid_set_overrides() -> None:
    cfg: dict[str, object] = {"project": {"task": "keep"}}
    apply_runtime_overrides(cfg, set_overrides=["", "invalid", ".=x", "project.task=updated"])
    assert cfg["project"]["task"] == "updated"


def test_apply_set_overrides_can_take_precedence_after_other_mutations() -> None:
    cfg: dict[str, object] = {"behavior_analysis": {"statistics": {"predictor_control": "linear"}}}
    cfg["behavior_analysis"]["statistics"]["predictor_control"] = "spline"
    apply_set_overrides(cfg, ["behavior_analysis.statistics.predictor_control=none"])
    assert cfg["behavior_analysis"]["statistics"]["predictor_control"] == "none"
