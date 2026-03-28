from __future__ import annotations

from pathlib import Path

from eeg_pipeline.cli.commands import info_helpers
from eeg_pipeline.cli.commands.base_feature_availability import _empty_feature_availability


def test_build_subject_status_json_uses_window_summary_helper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    deriv_root = tmp_path / "derivatives"
    features_dir = deriv_root / "sub-0001" / "eeg" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        info_helpers,
        "_collect_available_time_windows",
        lambda features_dir, config, feature_groups: (
            ["baseline", "stim"],
            {"power": ["baseline", "stim"]},
        ),
        raising=False,
    )

    def fail_if_legacy_window_scan_runs(*args, **kwargs):
        raise AssertionError("_get_available_time_windows should not be called")

    monkeypatch.setattr(
        info_helpers,
        "_get_available_time_windows",
        fail_if_legacy_window_scan_runs,
    )

    payload = info_helpers._build_subject_status_json(
        discovered_subjects=["0001"],
        epochs_subjects=set(),
        features_subjects={"0001"},
        deriv_root=deriv_root,
        task="pain",
        config={},
        bids_root_override=None,
    )

    assert payload["available_windows"] == ["baseline", "stim"]
    assert payload["available_windows_by_feature"] == {"power": ["baseline", "stim"]}


def test_process_single_subject_uses_feature_inventory_helper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    deriv_root = tmp_path / "derivatives"
    features_dir = deriv_root / "sub-0001" / "eeg" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    feature_availability = _empty_feature_availability()
    feature_availability["features"]["power"] = {
        "available": True,
        "last_modified": "2026-03-28T00:00:00Z",
    }
    feature_availability["bands"]["alpha"] = {
        "available": True,
        "last_modified": "2026-03-28T00:00:00Z",
    }

    monkeypatch.setattr(
        info_helpers,
        "detect_feature_inventory",
        lambda path: {
            "feature_availability": feature_availability,
            "available_bands": ["alpha"],
        },
        raising=False,
    )

    def fail_if_legacy_feature_scan_runs(*args, **kwargs):
        raise AssertionError("legacy feature scan should not be called")

    monkeypatch.setattr(
        info_helpers,
        "detect_feature_availability",
        fail_if_legacy_feature_scan_runs,
        raising=False,
    )
    monkeypatch.setattr(
        info_helpers,
        "detect_available_bands",
        fail_if_legacy_feature_scan_runs,
        raising=False,
    )

    payload = info_helpers._process_single_subject(
        subj_id="0001",
        has_epochs=False,
        has_features=True,
        deriv_root=deriv_root,
        task="pain",
        config={},
        global_epoch_metadata={},
        bids_root=None,
        source_root=None,
    )

    assert payload["available_bands"] == ["alpha"]
    assert payload["feature_availability"] == feature_availability
