# The CTPS threshold the ECG detector flags on has to be settable per study.
#
# The generated config named ica_use_ecg_detection but never the threshold, so every
# study ran at MNE-BIDS-Pipeline's 0.1 whatever its recordings looked like. That value
# suits data carrying a ballistocardiogram; on EEG recorded outside a scanner it sits
# inside the null distribution of CTPS scores, and a study had no way to say so short of
# switching the detector off altogether.

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from tests.utils.pipelines_test_utils import DotConfig


def _pipeline(ecg_threshold):
    # Imported inside the test, as the neighbouring preprocessing tests do: importing
    # the pipeline module at collection time initialises configuration state that the
    # run_batch tests in test_pipeline_preprocessing.py expect to set up themselves.
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

    pipeline = object.__new__(PreprocessingPipeline)
    pipeline.bids_root = Path("/tmp/bids")
    pipeline.deriv_root = Path("/tmp/deriv")
    pipeline.logger = Mock()
    ica = {
        "algorithm": "extended_infomax",
        "n_components": None,
        "l_freq": 1.0,
        "h_freq": 100.0,
        "reject": "autoreject_local",
        "use_icalabel": True,
        "use_ecg_detection": True,
        "use_eog_detection": False,
        "labels_to_keep": ["brain", "other"],
        "probability_threshold": 0.8,
        "process_raw_clean": True,
    }
    if ecg_threshold is not None:
        ica["ecg_threshold"] = ecg_threshold
    pipeline.config = DotConfig(
        {
            "eeg": {"ch_types": "eeg", "reference": "average", "eog_channels": "EOG001"},
            "preprocessing": {
                "task_is_rest": False,
                "l_freq": 0.1,
                "h_freq": 40.0,
                "find_breaks": False,
            },
            "ica": ica,
            "epochs": {
                "baseline": [None, 0],
                "reject_method": "none",
                "tmin": -0.2,
                "tmax": 0.8,
            },
        }
    )
    return pipeline


def _generate(pipeline):
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

    with patch.object(
        PreprocessingPipeline, "_detect_conditions_from_bids", return_value=["stim"]
    ):
        return pipeline._generate_mne_bids_config(
            "preprocessing/_06a2_find_ica_artifacts", subjects=["0001"]
        )


def test_configured_ecg_threshold_reaches_the_generated_config():
    config = _generate(_pipeline(0.25))

    assert "ica_ecg_threshold = 0.25" in config


def test_absent_ecg_threshold_is_left_to_mne_bids_pipeline():
    """An unset threshold must not be written, so the upstream default still applies."""
    config = _generate(_pipeline(None))

    assert "ica_ecg_threshold" not in config
