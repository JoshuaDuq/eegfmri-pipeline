"""Provenance record for the subject HTML report.

Filter, reference, ICA and baseline settings all change the numbers a report shows, so
two reports from different pipeline generations are not comparable unless the settings
that produced them are written down. This records those settings. It draws no conclusion
from them: judging the data is the reviewer's job, and a section that graded a subject
against thresholds would lend invented numbers an authority they do not have.
"""

from __future__ import annotations

import html
from typing import Any

import mne

#: Config keys worth recording, because changing any of them makes two reports
#: incomparable. Kept explicit rather than dumping the whole config, which would bury
#: the handful of settings that actually alter the numbers.
PROVENANCE_KEYS = (
    ("preprocessing.task_is_rest", "Resting-state mode"),
    ("eeg.reference", "EEG reference"),
    ("preprocessing.l_freq", "High-pass (Hz)"),
    ("preprocessing.h_freq", "Low-pass (Hz)"),
    ("preprocessing.notch_freq", "Notch (Hz)"),
    ("preprocessing.resample_freq", "Resampled to (Hz)"),
    ("ica.method", "ICA method"),
    ("ica.n_components", "ICA components requested"),
    ("ica.l_freq", "ICA high-pass (Hz)"),
    ("ica.use_icalabel", "ICLabel used"),
    ("ica.probability_threshold", "ICLabel exclusion threshold"),
    ("ica.labels_to_keep", "ICLabel classes kept"),
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
    ("preprocessing.brainvision_analyzer.enabled", "Analyzer correction upstream"),
)


def _format_value(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def provenance_html(
    config: Any | None, *, keys: tuple[tuple[str, str], ...] = PROVENANCE_KEYS
) -> str:
    """Render the settings that determine whether two reports are comparable."""
    if config is None:
        return (
            "<p>No configuration was recorded for this report, so the settings that "
            "produced it cannot be reconstructed.</p>"
        )
    rows = []
    for key, label in keys:
        value = config.get(key)
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{html.escape(_format_value(value))}</td>"
            f"<td><code>{html.escape(key)}</code></td></tr>"
        )
    return (
        "<p>The settings below determine whether this report can be compared with "
        "another. Filter, reference, ICA and baseline choices all change the numbers "
        "reported above, so a report without them is not reproducible.</p>"
        "<table><thead><tr><th>Setting</th><th>Value</th><th>Config key</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def add_provenance_review(
    *,
    report: mne.Report,
    config: Any | None = None,
    section: str = "Configuration",
) -> None:
    """Place the provenance record at the top of the report."""
    from eeg_pipeline.preprocessing.report.organize import (
        move_tagged_content_first,
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
    move_tagged_content_first(report, tag="provenance")


__all__ = [
    "PROVENANCE_KEYS",
    "add_provenance_review",
    "provenance_html",
]
