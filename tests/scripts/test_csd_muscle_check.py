"""The CSD arm must interpolate; the voltage arm must not.

CSD fits a spherical spline over the montage, so a gap distorts its neighbourhood. That is
the one place the pipeline's no-interpolation rule is set aside, and only inside this arm.
"""

from __future__ import annotations

import numpy as np

from studies.pain_study.scripts import csd_muscle_check as cmc


def _epochs(n_epochs=4, n_channels=32, sfreq=500.0, n_times=1000, n_bads=1):
    """Epochs on a real montage, with `n_bads` of the channels marked bad.

    The bad channel is taken from the montage that was actually built rather than named
    literally, so the fixture cannot quietly mark a channel that is not present.
    """
    import mne

    mne.set_log_level("ERROR")
    montage = mne.channels.make_standard_montage("standard_1020")
    names = [name for name in montage.ch_names if name not in {"T3", "T4", "T5", "T6"}]
    names = names[:n_channels]
    info = mne.create_info(names, sfreq, "eeg")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-6, (n_epochs, len(names), n_times))
    epochs = mne.EpochsArray(data, info, verbose="ERROR")
    epochs.set_montage(montage, on_missing="ignore")
    # Middle of the list: an interior channel, so interpolation has neighbours on all sides.
    epochs.info["bads"] = names[len(names) // 2 : len(names) // 2 + n_bads]
    return epochs


def test_the_voltage_arm_excludes_bad_channels():
    epochs = _epochs()
    (bad,) = epochs.info["bads"]
    kept = cmc.voltage_arm(epochs).ch_names
    assert bad not in kept, "voltage-space power keeps the pipeline's no-interpolation rule"


def test_the_csd_arm_interpolates_so_the_spline_has_no_gap():
    epochs = _epochs()
    (bad,) = epochs.info["bads"]
    kept = cmc.csd_arm(epochs).ch_names
    assert bad in kept, (
        "CSD fits a spherical spline over the montage; a missing channel distorts its "
        "neighbourhood, so bad channels are interpolated inside this arm only"
    )


def test_the_two_arms_disagree_on_channel_count_when_there_are_bads():
    epochs = _epochs()
    assert len(cmc.csd_arm(epochs).ch_names) > len(cmc.voltage_arm(epochs).ch_names)


def test_no_bads_means_the_arms_agree_on_channels():
    epochs = _epochs(n_bads=0)
    assert set(cmc.csd_arm(epochs).ch_names) == set(cmc.voltage_arm(epochs).ch_names)


def test_the_line_mask_tracks_what_the_removal_recorded():
    """The manifest is the source, with the configured list as the fallback.

    A hardcoded copy drifted once already: it masked 61.0353 Hz, dropped from the removal
    for sitting 0.128 Hz from comb harmonic 51, while masking nothing near 94 Hz -- where
    the strongest residual in the cohort sits, at 10.1% of sub-0008's 62-95 Hz power and
    10.9% of sub-0001's. The muscle index here is a 62-95 Hz ratio, so that line sat in the
    numerator of the measurement the mask exists to protect.

    Reading the config fixed that and is no longer sufficient: lines are detected per
    session now, so only the manifest records what was actually removed.
    """
    from studies.pain_study.analysis.line_comb import removal as lr
    from studies.pain_study.scripts.workflow_config import load_workflow_config

    configured = tuple(
        float(f) for f in load_workflow_config("line_comb").get("line_comb_removal.isolated_hz")
    )
    assert tuple(cmc.LINES) == lr.removed_isolated_lines(cmc.MANIFEST, fallback=configured)


def test_the_mask_does_not_carry_a_frequency_the_removal_dropped():
    """61.0353 Hz is not a removal target, so masking it discards real spectrum."""
    covered = lambda f: any(lo <= f <= hi for lo, hi in cmc.MASK)
    assert not covered(61.0353), (
        "61.0353 Hz was dropped from isolated_hz for sitting 0.128 Hz from comb harmonic "
        "51; masking it removes band power that was never contaminated"
    )
