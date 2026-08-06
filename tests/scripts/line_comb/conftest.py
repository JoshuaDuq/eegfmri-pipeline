import numpy as np
import pytest


@pytest.fixture
def brainvision_run(tmp_path):
    """A small BrainVision file written the way the BIDS dataset writes them."""
    import mne

    mne.set_log_level("ERROR")
    sfreq, n_times = 1000.0, 4000
    rng = np.random.default_rng(0)
    data = rng.normal(scale=2e-5, size=(4, n_times))
    info = mne.create_info(["Fp1", "Cz", "Oz", "ECG"], sfreq, ["eeg", "eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(data, info)
    path = tmp_path / "sub-0001_task-thermalactive_run-1_eeg.vhdr"
    mne.export.export_raw(path, raw, fmt="brainvision", overwrite=True)
    return path, raw
