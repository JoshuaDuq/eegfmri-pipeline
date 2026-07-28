"""AutoReject's channel-by-epoch verdict must survive as a derivative.

MNE-BIDS-Pipeline computes the reject log in ``_09_ptp_reject``, prints one line about it
and draws one figure from it, then discards it. Nothing downstream can tell which
channel-in-trial samples are measured and which are spline estimates from neighbours, so
these tests pin the serialization, the per-trial counts, and the check that the
reconstructed log actually describes the epochs on disk.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.autoreject_log import (
    AutorejectLog,
    AutorejectLogSettings,
    autoreject_log_path_for_epochs,
    compute_autoreject_log,
    kept_epoch_counts,
    pre_rejection_epochs_path,
    read_autoreject_log,
    verify_log_describes_clean_epochs,
    write_autoreject_log,
)


def _log() -> AutorejectLog:
    """Three epochs over four channels; epoch 1 was dropped."""
    labels = np.array(
        [
            [0, 2, 2, 1],
            [1, 1, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )
    return AutorejectLog(
        ch_names=("C3", "Cz", "C4", "Pz"),
        labels=labels,
        bad_epochs=np.array([False, True, False]),
        n_interpolate=2,
        consensus=0.8,
    )


def _epochs(n_epochs: int, ch_names: list[str]) -> mne.EpochsArray:
    sfreq = 100.0
    info = mne.create_info(ch_names, sfreq, "eeg")
    info.set_montage("standard_1020")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 2e-5, (n_epochs, len(ch_names), 20))
    return mne.EpochsArray(data, info, tmin=-0.05, verbose="ERROR")


def test_written_log_reads_back_with_the_same_verdicts(tmp_path) -> None:
    original = _log()

    path = write_autoreject_log(original, tmp_path / "sub-01_desc-autoreject_log.tsv")
    restored = read_autoreject_log(path)

    assert restored.ch_names == original.ch_names
    np.testing.assert_array_equal(restored.labels, original.labels)
    np.testing.assert_array_equal(restored.bad_epochs, original.bad_epochs)
    assert restored.n_interpolate == original.n_interpolate
    assert restored.consensus == pytest.approx(original.consensus)


def test_counts_cover_only_the_epochs_that_survived_rejection() -> None:
    counts = kept_epoch_counts(_log())

    # Epoch 1 was dropped, so it has no row in the clean derivative to annotate.
    assert len(counts) == 2
    assert counts["n_channels_interpolated"].tolist() == [2, 0]
    assert counts["n_channels_bad_not_interpolated"].tolist() == [1, 0]


def test_a_log_that_does_not_describe_the_clean_epochs_is_an_error() -> None:
    log = _log()
    # The log says two epochs survived; this derivative holds three.
    clean = _epochs(3, ["C3", "Cz", "C4", "Pz"])

    with pytest.raises(ValueError, match="2 epochs.*3"):
        verify_log_describes_clean_epochs(log, clean)


def test_a_log_matching_the_clean_epochs_verifies() -> None:
    log = _log()
    clean = _epochs(2, ["C3", "Cz", "C4", "Pz"])

    verify_log_describes_clean_epochs(log, clean)


def test_a_log_whose_channels_differ_from_the_clean_epochs_is_an_error() -> None:
    log = _log()
    clean = _epochs(2, ["C3", "Cz", "C4", "Oz"])

    with pytest.raises(ValueError, match="channel"):
        verify_log_describes_clean_epochs(log, clean)


def test_settings_come_from_the_same_config_keys_the_pipeline_passes_autoreject() -> None:
    settings = AutorejectLogSettings.from_config(_Config(_settings_config()))

    assert settings.n_interpolate == (4, 8, 16)
    assert settings.random_state == 42
    assert settings.n_jobs == 6


def test_missing_interpolation_grid_fails_rather_than_inventing_one() -> None:
    config = _settings_config()
    del config["epochs.autoreject_n_interpolate"]

    with pytest.raises(ValueError, match="epochs.autoreject_n_interpolate"):
        AutorejectLogSettings.from_config(_Config(config))


def test_logging_autoreject_when_the_pipeline_does_not_run_it_is_an_error() -> None:
    config = _settings_config()
    config["epochs.reject"] = None

    with pytest.raises(ValueError, match="autoreject_local"):
        AutorejectLogSettings.from_config(_Config(config))


def test_pre_rejection_epochs_are_the_input_autoreject_was_fitted_on() -> None:
    clean = Path("/deriv/sub-0001_task-thermalactive_proc-clean_epo.fif")

    assert pre_rejection_epochs_path(clean).name == "sub-0001_task-thermalactive_epo.fif"


def _settings_config() -> dict:
    return {
        "epochs.autoreject_n_interpolate": [4, 8, 16],
        "epochs.reject": "autoreject_local",
        "preprocessing.random_state": 42,
        "preprocessing.n_jobs": 6,
    }


def test_computed_log_has_one_verdict_per_channel_per_epoch() -> None:
    epochs = _epochs(12, ["C3", "Cz", "C4", "Pz"])
    settings = AutorejectLogSettings(n_interpolate=(1,), random_state=42, n_jobs=1)

    log = compute_autoreject_log(epochs, settings)

    assert log.labels.shape == (12, 4)
    assert log.bad_epochs.shape == (12,)
    assert log.ch_names == ("C3", "Cz", "C4", "Pz")
    assert set(np.unique(log.labels)) <= {0, 1, 2}


def test_log_path_sits_beside_the_clean_epochs_it_annotates() -> None:
    epochs = Path("/deriv/sub-0001_task-thermalactive_proc-clean_epo.fif")

    path = autoreject_log_path_for_epochs(epochs)

    assert path.parent == epochs.parent
    assert path.name == "sub-0001_task-thermalactive_desc-autoreject_log.tsv"


class _Config:
    """The dotted-key accessor the pipeline config exposes."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def get(self, key: str, default=None):
        return self._values.get(key, default)
