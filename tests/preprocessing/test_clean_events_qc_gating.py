# ECG coupling needs a recorded lead, not a scanner.

from __future__ import annotations

from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig


def _config(*, eeg_fmri: bool, ecg_channels: list[str]) -> dict:
    return {
        "preprocessing": {
            "eeg_fmri": eeg_fmri,
            "clean_events_qc": {
                "enabled": True,
                "ecg_coupling": {"enabled": True, "channels": ["ECG"]},
                "peripheral_low_gamma": {"enabled": False},
            },
        },
        "eeg": {"ecg_channels": ecg_channels},
    }


def test_ecg_coupling_runs_outside_a_scanner_when_a_lead_is_named():
    config = _config(eeg_fmri=False, ecg_channels=["ECG"])
    parsed = CleanEventsQCConfig.from_config(config)
    assert parsed.ecg_coupling.enabled is True


def test_ecg_coupling_is_disabled_when_no_lead_is_named():
    config = _config(eeg_fmri=True, ecg_channels=[])
    parsed = CleanEventsQCConfig.from_config(config)
    assert parsed.ecg_coupling.enabled is False
