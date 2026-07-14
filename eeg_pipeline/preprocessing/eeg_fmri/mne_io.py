"""Strict MNE boundary validation for native EEG-fMRI correction."""

from __future__ import annotations

import mne
import numpy as np


def validate_acquisition(
    raw: mne.io.BaseRaw,
    *,
    expected_sampling_frequency: float,
    expected_channel_count: int,
    ecg_channel: str,
) -> None:
    """Validate the original acquisition and assign its recorded ECG channel."""
    sampling_frequency = float(raw.info["sfreq"])
    if not np.isclose(
        sampling_frequency,
        expected_sampling_frequency,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Unexpected acquisition sampling frequency: "
            f"{sampling_frequency} != {expected_sampling_frequency} Hz"
        )
    if len(raw.ch_names) != expected_channel_count:
        raise ValueError(
            f"Expected {expected_channel_count} acquisition channels, found {len(raw.ch_names)}"
        )
    if raw.ch_names.count(ecg_channel) != 1:
        raise ValueError(f"Expected exactly one channel named {ecg_channel!r}")

    raw.set_channel_types({ecg_channel: "ecg"}, on_unit_change="ignore", verbose=False)
    channel_types = raw.get_channel_types()
    if channel_types.count("ecg") != 1:
        raise ValueError("Expected exactly one ECG channel after channel-type assignment")
    non_ecg_types = {
        channel_type
        for channel_name, channel_type in zip(raw.ch_names, channel_types, strict=True)
        if channel_name != ecg_channel
    }
    if non_ecg_types != {"eeg"}:
        raise ValueError(f"All non-ECG acquisition channels must be EEG, found {non_ecg_types}")


def extract_volume_samples(
    raw: mne.io.BaseRaw,
    *,
    annotation_description: str,
) -> np.ndarray:
    """Return zero-based samples for exact scanner-volume annotations."""
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    remaining_collisions = sorted(
        {
            description
            for description in descriptions
            if description != annotation_description and description.endswith("/V  1")
        }
    )
    if remaining_collisions:
        raise ValueError(
            "Ambiguous scanner marker descriptions remain: " + ", ".join(remaining_collisions)
        )
    expected_count = int(np.count_nonzero(descriptions == annotation_description))
    if expected_count == 0:
        raise ValueError(f"No exact {annotation_description!r} annotations were found")

    events, _ = mne.events_from_annotations(
        raw,
        event_id={annotation_description: 1},
        use_rounding=True,
        verbose=False,
    )
    if events.shape[0] != expected_count:
        raise ValueError("Volume annotations did not map one-to-one onto acquisition samples")
    samples = events[:, 0].astype(int) - int(raw.first_samp)
    if np.unique(samples).size != samples.size:
        raise ValueError("Multiple volume annotations map to the same acquisition sample")
    return samples
