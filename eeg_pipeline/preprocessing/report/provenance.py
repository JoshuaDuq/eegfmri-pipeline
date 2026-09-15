"""Provenance record for the subject HTML report.

Filter, reference, ICA and baseline settings all change the numbers a report shows, so
two reports from different pipeline generations are not comparable unless the settings
that produced them are written down. This records those settings. It draws no conclusion
from them: judging the data is the reviewer's job, and a section that graded a subject
against thresholds would lend invented numbers an authority they do not have.
"""

from __future__ import annotations

from typing import Any

import mne

from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Config keys worth recording, because changing any of them makes two reports
#: incomparable. Kept explicit rather than dumping the whole config, which would bury
#: the handful of settings that actually alter the numbers.
PROVENANCE_KEYS = (
    ("project.random_state", "Random seed"),
    ("paths.decomb_manifest", "Decomb correction manifest"),
    ("preprocessing.task_is_rest", "Resting-state mode"),
    ("eeg.reference", "EEG reference"),
    ("preprocessing.line_freq", "Power-line frequency (Hz)"),
    ("preprocessing.l_freq", "High-pass (Hz)"),
    ("preprocessing.h_freq", "Low-pass (Hz)"),
    ("preprocessing.notch_freq", "Notch (Hz)"),
    ("preprocessing.resample_freq", "Resampled to (Hz)"),
    ("pyprep.detection_low_pass", "Bad-channel detection low-pass (Hz)"),
    ("pyprep.ransac", "PyPREP RANSAC used"),
    ("pyprep.repeats", "PyPREP detection repeats"),
    ("pyprep.consider_previous_bads", "Previous bad channels retained"),
    ("ica.algorithm", "ICA algorithm"),
    ("ica.n_components", "ICA components requested"),
    ("ica.l_freq", "ICA high-pass (Hz)"),
    ("ica.h_freq", "ICA low-pass (Hz)"),
    ("ica.reject", "ICA fitting rejection threshold"),
    ("ica.use_icalabel", "ICLabel used"),
    ("ica.probability_threshold", "ICLabel exclusion threshold"),
    ("ica.labels_to_keep", "ICLabel classes kept"),
    ("ica.use_ecg_detection", "ECG component detection used"),
    ("ica.ecg_threshold", "ECG component detection threshold"),
    ("ica.use_eog_detection", "EOG component detection used"),
    ("ica.require_manual_review", "Manual ICA review required"),
    ("ica.manual_review_complete", "Manual ICA review marked complete"),
    ("ica.cardiac_review.beat_source", "ECG beat source"),
    ("ica.cardiac_review.marker_description", "ECG beat marker"),
    ("ica.cardiac_review.promote_exclusions", "All-run cardiac exclusions promoted"),
    (
        "ica.cardiac_review.promotion_minimum_run_fraction",
        "Minimum run fraction for cardiac promotion",
    ),
    ("ica.ocular_review.eog_channels", "EOG channels used for component detection"),
    ("epochs.tmin", "Epoch start (s)"),
    ("epochs.tmax", "Epoch end (s)"),
    ("epochs.baseline", "Epoch baseline (s)"),
    # These two decide the dropped-epoch count reported in the rejection section, and
    # the bad-channel union reported in the coverage section. Without them neither of
    # those headline numbers can be reproduced.
    ("epochs.reject", "Epoch rejection method"),
    ("epochs.autoreject_n_interpolate", "Autoreject interpolation candidates"),
    ("pyprep.bad_channel_sync_policy", "Bad-channel sync policy"),
    ("time_windows.baseline_tfr_morlet", "TFR baseline, Morlet (s)"),
    ("time_windows.baseline_tfr_multitaper", "TFR baseline, multitaper (s)"),
    ("report.analysis.aperiodic_fit_range_hz", "Aperiodic fit range (Hz)"),
    ("report.analysis.aperiodic_exclude_hz", "Aperiodic exclusion windows (Hz)"),
    ("report.analysis.response_window_s", "Split-half response window (s)"),
    ("report.analysis.alpha_band_hz", "Posterior rhythm band (Hz)"),
    ("report.analysis.alpha_reference_band_hz", "Posterior rhythm reference band (Hz)"),
    ("report.analysis.muscle_filter_freq_hz", "Muscle screening band (Hz)"),
    ("report.analysis.muscle_zscore_threshold", "Muscle screening threshold (z)"),
    (
        "report.analysis.muscle_min_length_good_s",
        "Minimum good gap in muscle screening (s)",
    ),
    (
        "report.analysis.bridge_diagnostic_duration_s",
        "Bridge diagnostic final segment (s)",
    ),
    ("report.acquisition.posterior_channel_pattern", "Posterior channel pattern"),
    ("report.acquisition.non_event_prefixes", "Annotation prefixes that are not events"),
    ("report.acquisition.component_label_patterns", "Component label patterns"),
    ("report.thresholds.notch_exclusion_half_width_hz", "Notch exclusion half-width (Hz)"),
    ("report.thresholds.plausible_heart_rate_bpm", "Possible heart rate (bpm)"),
    ("report.thresholds.marker_agreement_tolerance_s", "Beat-marker match tolerance (s)"),
    ("report.thresholds.channel_position_tolerance_m", "Shared electrode-position tolerance (m)"),
    ("report.thresholds.min_subjects_for_median", "Participants required for a cohort median"),
    (
        "report.thresholds.min_subjects_for_outer_band",
        "Participants required for the outer cohort band",
    ),
    ("report.display.continuity_window_seconds", "Continuity window (s)"),
    ("report.display.figure_dpi", "Report figure resolution (dpi)"),
    ("report.display.figure_max_width_px", "Report figure maximum width (px)"),
    ("report.display.evoked_topomap_count", "Evoked topographies per condition"),
)


#: Sentinel distinguishing "the config has no such key" from "the key is set to None".
_ABSENT = object()


def _format_value(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        # An empty sequence is a real, common value here -- aperiodic_exclude_hz ships
        # empty by default -- and joining zero items silently gives back "", which reads
        # as a blank cell rather than as the recorded choice it is.
        if not value:
            return "none"
        return ", ".join(str(item) for item in value)
    return str(value)


def provenance_html(
    config: Any | None, *, keys: tuple[tuple[str, str], ...] = PROVENANCE_KEYS
) -> str:
    """Render the settings that determine whether two reports are comparable.

    A key the configuration does not contain is omitted rather than listed as "not set".
    The list below spans every paradigm this pipeline supports, so an EEG-only dataset
    would otherwise carry a row about a stage it never ran and a resting-state dataset rows
    about epoch baselines — settings that influenced nothing, occupying the table that
    exists to record what did. A key that is present and false is a recorded decision and
    stays, because reproducing the report needs it.
    """
    if config is None:
        return (
            "<p>No configuration was recorded for this report, so the settings that "
            "produced it cannot be reconstructed.</p>"
        )
    rows = []
    for key, label in keys:
        value = config.get(key, _ABSENT)
        if value is _ABSENT:
            continue
        rows.append([label, _format_value(value), key])
    if not rows:
        return (
            "<p>The configuration carried no recorded settings from the provenance list, "
            "so the choices that produced this report cannot be reconstructed from it.</p>"
        )
    return (
        "<p>The settings below determine whether this report can be compared with "
        "another. Filter, reference, ICA and baseline choices all change the numbers "
        "reported above, so a report without them is not reproducible. Settings this "
        "dataset did not configure are omitted rather than listed as unset.</p>"
        + _manual_review_status(config)
        + grid_table(
            (
                Column("Setting", align=Align.TEXT),
                Column("Value", align=Align.TEXT),
                Column("Config key", align=Align.TEXT, code=True),
            ),
            rows,
        )
    )


def _manual_review_status(config: Any) -> str:
    """State what the manual-review booleans do—and do not—prove."""
    required = config.get("ica.require_manual_review", _ABSENT)
    completed = config.get("ica.manual_review_complete", _ABSENT)
    if required is _ABSENT and completed is _ABSENT:
        return ""
    if required is True and completed is True:
        return (
            "<p><strong>Manual ICA review status: marked complete in configuration.</strong> "
            "This is a configuration assertion, not independent proof that every component "
            "was reviewed; verify the component decision record before accepting the data.</p>"
        )
    if required is True:
        return (
            "<p><strong>Manual ICA review status: pending.</strong> Manual review is pending; "
            "epoch creation is blocked "
            "until component decisions have been reviewed and the configuration is explicitly "
            "updated. Any existing cleaned derivative is not thereby manually approved.</p>"
        )
    return (
        "<p><strong>Manual ICA review status: not required by configuration.</strong> "
        "Component exclusions therefore reflect the configured automated procedure.</p>"
    )


def add_provenance_review(
    *,
    report: mne.Report,
    config: Any | None = None,
    section: str = "Configuration",
) -> None:
    """Place the provenance record at the top of the report."""
    from eeg_pipeline.preprocessing.report.organize import (
        remove_tagged_content,
    )

    remove_tagged_content(report, tag="provenance")
    report.add_html(
        html=provenance_html(config),
        title="Configuration that produced this report",
        section=section,
        tags=("summary", "provenance"),
        replace=True,
    )


__all__ = [
    "PROVENANCE_KEYS",
    "add_provenance_review",
    "provenance_html",
]
