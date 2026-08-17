"""The scanner stages are gated on an explicit declaration, and EEG-only still runs.

``preprocessing.eeg_fmri`` says the recordings were made inside a scanner;
``preprocessing.brainvision_analyzer.enabled`` says an Analyzer correction ran upstream.
One switch used to decide both, so a plain EEG dataset either ran scanner QC against
inputs it does not have, or lost the scanner-harmonic measurement — which needs no
Analyzer output — merely by not having used Analyzer.
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
    yield instance
    for name, module in preexisting.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _write_channels(bids_root, *, with_ecg: bool) -> None:
    directory = bids_root / "sub-0001" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [("Fp1", "EEG"), ("Fp2", "EEG"), ("Cz", "EEG")]
    if with_ecg:
        rows.append(("ECG", "ECG"))
    lines = ["name\ttype\tunits\tstatus"]
    lines += [f"{n}\t{t}\tµV\tgood" for n, t in rows]
    (directory / "sub-0001_task-thermalactive_run-1_channels.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _steps(pipeline, mode="epochs"):
    from eeg_pipeline.pipelines import preprocessing as module

    # The manual-review gate is a separate concern from dataset type.
    pipeline.config["ica.require_manual_review"] = False
    return module.PreprocessingPipeline._get_steps_for_run(
        pipeline, mode, task_is_rest=False, subjects=["0001"], task="thermalactive"
    )


SCANNER_STEPS = {
    "pulse-marker-qc",
    "ica-cardiac-qc",
    "cardiac-attenuation-qc",
    "scanner-harmonic-qc",
}


def test_eeg_only_runs_without_any_scanner_stage(pipeline) -> None:
    pipeline.config["preprocessing.eeg_fmri"] = False
    _write_channels(pipeline.bids_root, with_ecg=False)

    steps = _steps(pipeline)

    assert not SCANNER_STEPS & set(steps)
    # The ordinary path is untouched.
    assert "epochs" in steps


def test_analyzer_flag_alone_does_not_enable_scanner_stages(pipeline) -> None:
    """A dataset is not EEG-fMRI because a config left the Analyzer switch on."""
    pipeline.config["preprocessing.eeg_fmri"] = False
    pipeline.config["preprocessing.brainvision_analyzer.enabled"] = True
    _write_channels(pipeline.bids_root, with_ecg=False)

    assert not SCANNER_STEPS & set(_steps(pipeline))


def test_scanner_harmonics_do_not_require_analyzer(pipeline) -> None:
    """It is measured from the EEG spectrum against the sequence timing, so it needs the
    scanner but not the correction that may or may not have preceded this pipeline."""
    pipeline.config["preprocessing.eeg_fmri"] = True
    pipeline.config["preprocessing.brainvision_analyzer.enabled"] = False
    _write_channels(pipeline.bids_root, with_ecg=True)

    steps = _steps(pipeline)

    assert "scanner-harmonic-qc" in steps
    assert "pulse-marker-qc" not in steps
    assert "cardiac-attenuation-qc" not in steps


def test_eeg_fmri_with_analyzer_keeps_every_scanner_stage(pipeline) -> None:
    """Each stage attaches to the mode that produces the derivative it measures:
    the cardiac component QC to ICA fitting, the attenuation and harmonic QC to
    epoching, and the pulse-marker QC to both."""
    pipeline.config["preprocessing.eeg_fmri"] = True
    pipeline.config["preprocessing.brainvision_analyzer.enabled"] = True
    _write_channels(pipeline.bids_root, with_ecg=True)

    epoch_steps = set(_steps(pipeline, mode="epochs"))
    ica_steps = set(_steps(pipeline, mode="ica"))

    assert {"pulse-marker-qc", "cardiac-attenuation-qc", "scanner-harmonic-qc"} <= epoch_steps
    assert {"pulse-marker-qc", "ica-cardiac-qc"} <= ica_steps
    assert SCANNER_STEPS <= epoch_steps | ica_steps


def test_declaring_eeg_fmri_without_an_ecg_channel_is_rejected(pipeline) -> None:
    """Otherwise this fails much later, inside a review, with an error about a missing
    channel rather than about the declaration that asked for it."""
    pipeline.config["preprocessing.eeg_fmri"] = True
    pipeline.config["preprocessing.brainvision_analyzer.enabled"] = True
    _write_channels(pipeline.bids_root, with_ecg=False)

    with pytest.raises(ValueError, match="no channel is typed ECG"):
        _steps(pipeline)


def test_the_cardiac_review_is_skipped_out_of_scanner_and_says_so(pipeline) -> None:
    from eeg_pipeline.pipelines import preprocessing as module

    pipeline.config["preprocessing.eeg_fmri"] = False
    pipeline.config["ica.cardiac_review"] = {"enabled": True}
    pipeline.config["ica.ocular_review"] = {"enabled": False}
    pipeline.config["ica.band_specific_report"] = {"enabled": False}
    called = []
    pipeline._run_mne_bids_pipeline = lambda *a, **k: None
    pipeline._harmonize_filtered_raw_bads_for_mne_concat = lambda *a, **k: None
    pipeline._run_ica_cardiac_review = lambda **k: called.append("cardiac")

    module.PreprocessingPipeline._run_ica_fitting(
        pipeline, ["0001"], "thermalactive", task_is_rest=False, n_jobs=1
    )

    assert called == [], "the cardiac review has no ballistocardiogram to measure here"
    logged = " ".join(str(c) for c in pipeline.logger.info.call_args_list)
    assert "eeg_fmri is false" in logged


def test_the_shipped_config_declares_this_study_as_eeg_fmri(pipeline) -> None:
    assert pipeline.config.get("preprocessing.eeg_fmri") is True


def test_a_config_predating_the_key_keeps_its_previous_behaviour(pipeline) -> None:
    """Defaulting the new key to false would strip scanner QC from every existing config
    without saying so — the same silent omission this separation exists to prevent."""
    from eeg_pipeline.pipelines import preprocessing as module

    preprocessing_section = pipeline.config.get("preprocessing")
    preprocessing_section.pop("eeg_fmri", None)
    pipeline.config["preprocessing.brainvision_analyzer.enabled"] = True
    _write_channels(pipeline.bids_root, with_ecg=True)

    assert module.PreprocessingPipeline._is_eeg_fmri(pipeline) is True
    assert "scanner-harmonic-qc" in _steps(pipeline)


def test_an_explicit_declaration_wins_over_the_fallback(pipeline) -> None:
    from eeg_pipeline.pipelines import preprocessing as module

    pipeline.config["preprocessing.eeg_fmri"] = False
    pipeline.config["preprocessing.brainvision_analyzer.enabled"] = True

    assert module.PreprocessingPipeline._is_eeg_fmri(pipeline) is False


def test_the_ecg_coupling_metric_is_gated_on_the_lead_not_the_declaration() -> None:
    # Issue #14: an out-of-scanner dataset with a recorded ECG lead used to have this
    # metric switched off underneath it. It now runs wherever the lead is named.
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.eeg_fmri"] = False
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = True
    config["eeg.ecg_channels"] = ["ECG"]

    assert CleanEventsQCConfig.from_config(config).ecg_coupling.enabled is True


def test_the_ecg_coupling_metric_survives_for_an_eeg_fmri_dataset() -> None:
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.eeg_fmri"] = True
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = True

    assert CleanEventsQCConfig.from_config(config).ecg_coupling.enabled is True


def test_cardiac_only_qc_switches_itself_off_rather_than_raising() -> None:
    # Asking for QC and naming no metric is a config mistake worth raising on. Asking
    # for the cardiac metric alone with no ECG lead named is not: nothing is left to
    # compute and the user got nothing wrong. True inside a scanner too -- the room is
    # not the condition.
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.eeg_fmri"] = True
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = True
    config["preprocessing.clean_events_qc.peripheral_low_gamma.enabled"] = False
    config["eeg.ecg_channels"] = []

    assert CleanEventsQCConfig.from_config(config).enabled is False


def test_naming_no_metric_at_all_still_raises() -> None:
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.eeg_fmri"] = False
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = False
    config["preprocessing.clean_events_qc.peripheral_low_gamma.enabled"] = False

    with pytest.raises(ValueError, match="at least one QC metric"):
        CleanEventsQCConfig.from_config(config)
