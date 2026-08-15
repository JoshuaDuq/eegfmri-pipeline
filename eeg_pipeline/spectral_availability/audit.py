from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from eeg_pipeline.spectral_availability.model import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingKey,
)

AUDIT_SUFFIX = "desc-spectralavailability.tsv"

ESTIMATOR_MULTITAPER = "multitaper"
ESTIMATOR_WELCH = "welch"
ESTIMATOR_MORLET = "morlet"
ESTIMATOR_HILBERT = "hilbert"


@dataclass(frozen=True)
class AvailabilityAuditRow:
    subject: str
    session: str
    task: str
    run: str
    analysis_target: str
    estimator: str
    support_rule: str
    unavailable_intervals_hz: str
    nominal_bandwidth_hz: float
    retained_bandwidth_hz: float
    retained_share: float
    psd_eligible: bool
    contiguous_band_eligible: bool
    n_epochs_aligned: int
    n_epochs_eligible: int
    n_epochs_ineligible: int
    manifest_sha256: str

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (self.subject, self.session, self.task, self.run, self.analysis_target)


def _format_intervals(intervals: Iterable[FrequencyInterval]) -> str:
    return ";".join(f"{interval.low_hz:g}-{interval.high_hz:g}" for interval in intervals)


def _overlap_width(interval: FrequencyInterval, fmin: float, fmax: float) -> float:
    return max(0.0, min(interval.high_hz, fmax) - max(interval.low_hz, fmin))


class SpectralAvailabilityAudit:
    """Collects one row per recording and analysis target, then writes it atomically.

    Registration is keyed by recording and target, so the same target may be
    registered again with identical content without duplicating a row. Registering
    conflicting content for the same key is an error rather than a silent overwrite.
    """

    def __init__(
        self,
        availability: EpochSpectralAvailability,
        manifest_sha256: str,
        *,
        subject: str,
        task: str,
    ) -> None:
        self._availability = availability
        self._manifest_sha256 = str(manifest_sha256)
        self._subject = str(subject)
        self._task = str(task)
        self._rows: dict[tuple[str, str, str, str, str], AvailabilityAuditRow] = {}

    def _epoch_groups(self) -> dict[RecordingKey, np.ndarray]:
        groups: dict[RecordingKey, list[int]] = {}
        for index, key in enumerate(self._availability.recording_keys):
            groups.setdefault(key, []).append(index)
        return {key: np.asarray(indices, dtype=int) for key, indices in groups.items()}

    def _add(self, row: AvailabilityAuditRow) -> None:
        existing = self._rows.get(row.identity)
        if existing is None:
            self._rows[row.identity] = row
            return
        if existing != row:
            raise ValueError(
                f"Conflicting spectral availability audit rows for {row.identity}; "
                "the same analysis target was registered with different content."
            )

    def register_grid(
        self,
        analysis_target: str,
        frequencies: ArrayLike,
        valid_frequency_mask: ArrayLike,
        *,
        estimator: str,
        support_rule: str,
        weights: ArrayLike | None = None,
    ) -> None:
        """Register a frequency-grid target from the mask the estimator actually produced."""
        from eeg_pipeline.utils.analysis.spectral import compute_frequency_weights

        freqs = np.asarray(frequencies, dtype=float)
        valid = np.asarray(valid_frequency_mask, dtype=bool)
        if valid.shape != (len(self._availability.recording_keys), freqs.size):
            raise ValueError(
                f"valid_frequency_mask shape {valid.shape} does not match the "
                f"(epochs, freqs) axes "
                f"({len(self._availability.recording_keys)}, {freqs.size})."
            )

        bin_weights = (
            compute_frequency_weights(freqs) if weights is None else np.asarray(weights, dtype=float)
        )
        nominal = float(np.sum(bin_weights))
        retained_per_epoch = valid @ bin_weights

        for key, indices in self._epoch_groups().items():
            retained = float(retained_per_epoch[indices[0]])
            eligible = retained_per_epoch[indices] > 0
            self._add(
                self._build_row(
                    key,
                    indices,
                    analysis_target,
                    estimator,
                    support_rule,
                    fmin=float(freqs[0]),
                    fmax=float(freqs[-1]),
                    nominal=nominal,
                    retained=retained,
                    eligible=eligible,
                    psd_eligible=retained > 0,
                )
            )

    def register_band(
        self,
        analysis_target: str,
        fmin: float,
        fmax: float,
        *,
        estimator: str,
        support_rule: str,
    ) -> None:
        """Register a contiguous-band target, which a single overlap makes ineligible."""
        eligible_by_epoch = self._availability.contiguous_band_eligible(fmin, fmax)
        intersections = self._availability.intersections(fmin, fmax)
        nominal = float(fmax) - float(fmin)

        for key, indices in self._epoch_groups().items():
            first = int(indices[0])
            removed = sum(
                _overlap_width(interval, float(fmin), float(fmax))
                for interval in intersections[first]
            )
            retained = max(0.0, nominal - removed)
            self._add(
                self._build_row(
                    key,
                    indices,
                    analysis_target,
                    estimator,
                    support_rule,
                    fmin=float(fmin),
                    fmax=float(fmax),
                    nominal=nominal,
                    retained=retained,
                    eligible=eligible_by_epoch[indices],
                    psd_eligible=retained > 0,
                )
            )

    def _build_row(
        self,
        key: RecordingKey,
        indices: np.ndarray,
        analysis_target: str,
        estimator: str,
        support_rule: str,
        *,
        fmin: float,
        fmax: float,
        nominal: float,
        retained: float,
        eligible: np.ndarray,
        psd_eligible: bool,
    ) -> AvailabilityAuditRow:
        first = int(indices[0])
        intersecting = self._availability.intersections(fmin, fmax)[first]
        n_eligible = int(np.sum(eligible))
        return AvailabilityAuditRow(
            subject=key.subject,
            session=key.session or "n/a",
            task=key.task,
            run=key.run,
            analysis_target=str(analysis_target),
            estimator=str(estimator),
            support_rule=str(support_rule),
            unavailable_intervals_hz=_format_intervals(intersecting) or "none",
            nominal_bandwidth_hz=float(nominal),
            retained_bandwidth_hz=float(retained),
            retained_share=float(retained / nominal) if nominal > 0 else float("nan"),
            psd_eligible=bool(psd_eligible),
            contiguous_band_eligible=not intersecting,
            n_epochs_aligned=int(indices.size),
            n_epochs_eligible=n_eligible,
            n_epochs_ineligible=int(indices.size) - n_eligible,
            manifest_sha256=self._manifest_sha256,
        )

    def rows(self) -> tuple[AvailabilityAuditRow, ...]:
        return tuple(sorted(self._rows.values(), key=lambda row: row.identity))

    def filename(self) -> str:
        return f"sub-{self._subject}_task-{self._task}_{AUDIT_SUFFIX}"

    def write(self, directory: str | Path) -> Path:
        """Write the audit beside the feature outputs, replacing any previous version."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / self.filename()

        column_names = [field.name for field in fields(AvailabilityAuditRow)]
        lines = ["\t".join(column_names)]
        for row in self.rows():
            lines.append("\t".join(_format_value(getattr(row, name)) for name in column_names))
        payload = "\n".join(lines) + "\n"

        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination,
            prefix=".spectralavailability-",
            suffix=".tsv",
            delete=False,
        )
        try:
            with handle:
                handle.write(payload)
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        return path


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
