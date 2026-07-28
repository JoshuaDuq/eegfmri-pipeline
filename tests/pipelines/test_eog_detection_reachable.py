"""Requesting ocular detection that upstream would skip must fail loudly.

MNE-BIDS-Pipeline recovers blinks by correlating components against EOG channels. Given
neither a configured name nor a channel typed EOG, ``_06a2_find_ica_artifacts`` breaks out
of its detection loop, records no ocular components, and raises nothing. The resulting
report is indistinguishable from a participant who never blinked. This montage has no
dedicated EOG electrode, so that silent path is the default one unless surrogates are
named.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest

_REAL_IMPORTS = (
    "eeg_pipeline.pipelines.preprocessing",
    "eeg_pipeline.pipelines.base",
    "eeg_pipeline.pipelines.progress",
    "eeg_pipeline.utils.config.roots",
    "eeg_pipeline.utils.config.loader",
)


@pytest.fixture
def pipeline(tmp_path):
    preexisting = {name: sys.modules.get(name) for name in _REAL_IMPORTS}

    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    from eeg_pipeline.utils.config.loader import load_config

    instance = object.__new__(PreprocessingPipeline)
    instance.config = load_config()
    instance.logger = Mock()
    instance.bids_root = tmp_path / "bids"
    instance.deriv_root = tmp_path / "deriv"
    instance.bids_root.mkdir(parents=True, exist_ok=True)
    instance.config["epochs.conditions"] = ["painful", "neutral"]

    yield instance

    for name, module in preexisting.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _write_channels(bids_root, rows: list[tuple[str, str]]) -> None:
    directory = bids_root / "sub-0001" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["name\ttype\tunits\tstatus"]
    lines += [f"{name}\t{ch_type}\tµV\tgood" for name, ch_type in rows]
    (directory / "sub-0001_task-thermalactive_run-1_channels.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_eog_detection_without_any_eog_channel_is_rejected(pipeline) -> None:
    pipeline.config["eeg.eog_channels"] = None
    _write_channels(pipeline.bids_root, [("Fp1", "EEG"), ("Cz", "EEG"), ("ECG", "ECG")])

    with pytest.raises(ValueError, match="no EOG channel is available"):
        pipeline._generate_mne_bids_config(
            "preprocessing/_06a2_find_ica_artifacts",
            subjects=["0001"],
            task="thermalactive",
            task_is_rest=False,
        )


def test_configured_surrogates_satisfy_the_guard_and_reach_upstream(pipeline) -> None:
    pipeline.config["eeg.eog_channels"] = ["Fp1", "Fp2"]
    _write_channels(pipeline.bids_root, [("Fp1", "EEG"), ("Fp2", "EEG")])

    source = pipeline._generate_mne_bids_config(
        "preprocessing/_06a2_find_ica_artifacts",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
    )

    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    assert namespace["eog_channels"] == ["Fp1", "Fp2"]
    assert namespace["ica_use_eog_detection"] is True


def test_a_real_eog_electrode_satisfies_the_guard_without_configuration(pipeline) -> None:
    """Upstream finds an EOG-typed channel on its own, so naming it again is redundant."""
    pipeline.config["eeg.eog_channels"] = None
    _write_channels(pipeline.bids_root, [("Cz", "EEG"), ("VEOG", "EOG")])

    source = pipeline._generate_mne_bids_config(
        "preprocessing/_06a2_find_ica_artifacts",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
    )

    assert "eog_channels" not in source


def test_disabling_eog_detection_makes_the_omission_explicit_and_allowed(pipeline) -> None:
    pipeline.config["eeg.eog_channels"] = None
    pipeline.config["ica.use_eog_detection"] = False
    _write_channels(pipeline.bids_root, [("Cz", "EEG")])

    source = pipeline._generate_mne_bids_config(
        "preprocessing/_06a2_find_ica_artifacts",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
    )

    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    assert namespace["ica_use_eog_detection"] is False


def test_the_shipped_config_names_surrogates_so_detection_actually_runs(pipeline) -> None:
    """The default config must not select the silent path."""
    _write_channels(pipeline.bids_root, [("Fp1", "EEG"), ("Fp2", "EEG")])

    assert pipeline.config.get("ica.use_eog_detection") is True
    assert pipeline._resolve_eog_detection_channels() == ["Fp1", "Fp2"]


def test_a_signed_off_review_disables_cardiac_promotion(pipeline, monkeypatch) -> None:
    """Promotion is the automated baseline the manual review adjusts. Re-applying it over
    a completed review would silently re-exclude components the reviewer had cleared,
    because a ``good`` row records no distinction between "cleared" and "not yet looked
    at"."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.ica_cardiac_report.generate_ica_cardiac_review",
        _capture,
    )
    monkeypatch.setattr(
        pipeline, "_resolve_bad_harmonization_subjects", lambda subjects: ["0001"]
    )
    monkeypatch.setattr(pipeline, "_find_filtered_raw_run_files", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "_find_band_ica_report_inputs", lambda subject: [])

    pipeline.config["ica.cardiac_review"] = {
        "enabled": True,
        "promote_exclusions": True,
    }
    pipeline.config["ica.manual_review_complete"] = True

    pipeline._run_ica_cardiac_review(subjects=["0001"], task="thermalactive")

    pipeline.logger.info.assert_called()
    logged = " ".join(str(call) for call in pipeline.logger.info.call_args_list)
    assert "manual_review_complete" in logged


def test_promotion_is_active_before_the_review_is_signed_off(pipeline, monkeypatch) -> None:
    from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings

    monkeypatch.setattr(
        pipeline, "_resolve_bad_harmonization_subjects", lambda subjects: ["0001"]
    )
    monkeypatch.setattr(pipeline, "_find_filtered_raw_run_files", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "_find_band_ica_report_inputs", lambda subject: [])

    pipeline.config["ica.cardiac_review"] = {
        "enabled": True,
        "promote_exclusions": True,
    }
    pipeline.config["ica.manual_review_complete"] = False

    pipeline._run_ica_cardiac_review(subjects=["0001"], task="thermalactive")

    logged = " ".join(str(call) for call in pipeline.logger.info.call_args_list)
    assert "manual_review_complete" not in logged
    assert CardiacReviewSettings.from_mapping(
        pipeline.config.get("ica.cardiac_review")
    ).promote_exclusions is True
