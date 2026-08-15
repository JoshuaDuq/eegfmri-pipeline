"""Remove the bands the line removal cannot: a wide FIR notch over measured clusters.

    eeg-pipeline line-comb notch

``apply`` subtracts a sinusoid wherever a line is resolvable. Some contamination is not
resolvable: 56.8-57.7 Hz in this room carries about a hundred distinct peaks above 3 dB at
2 mHz resolution, non-stationary between 54 s windows, which is the same structure already
established for mains at 59-61 Hz. Subtracting sinusoids one at a time cannot win against
that -- remove the tallest and the next one surfaces -- and no detection threshold changes
it. A notch wide enough to span the cluster is the only thing that does.

That is a different trade from the rest of this workflow, so it is a different stage. The
line removal buys its suppression at a few hundredths of a hertz per line; this pays the
full width of a band whether or not signal was in it. It runs last, reads what ``apply``
wrote, and writes its own BIDS root, so the two are separable and either can be inspected
alone.

Like ``apply``, every sidecar is copied byte-for-byte and only the ``.eeg`` binaries are
rewritten, so sampling rate, channel set, length and annotations cannot drift. Only EEG
channels are filtered; ECG and EOG stay byte-identical.

Nothing here is gated. A fixed-width FIR notch has no estimator that can be wrong, so
there is no fit to certify -- what there is instead is a manifest recording, per recording
and per band, how much the notch actually took and what it cost in each analysed band.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from studies.pain_study.scripts.line_comb import remove as rm

WORKFLOW = rm.WORKFLOW

DEFAULT_BANDS_HZ = ((56.8, 57.7),)
"""Bands notched unless the configuration names others.

56.8-57.7 Hz is measured, not assumed. Across sub-0008, sub-0011 and sub-0000 it carries a
median 10.9% of all 28-95 Hz power -- up to 24.7% -- in 1.34% of the band's width, and
92.6% of that stands above the shoulders on either side, so it is artifact rather than
spectrum the notch would take either way. The burden is participant-specific over a
ten-fold range, which is the same systematic between-participant offset the line comb is
removed for.
"""

CANONICAL_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
"""Bands the cost is reported against, when the configuration defines them."""


@dataclass(frozen=True)
class NotchBand:
    """One contiguous span to remove, named by its edges rather than a centre and width.

    Edges are what a diagnosis measures and what a reader can check against a spectrum; a
    centre and a width are what the filter wants. Converting here keeps the configuration
    in the units the evidence is in.
    """

    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.low_hz, self.high_hz))):
            raise ValueError("Notch band edges must be finite.")
        if self.low_hz <= 0.0:
            raise ValueError(f"Notch band edges must be positive, got {self.low_hz}.")
        if self.high_hz <= self.low_hz:
            raise ValueError(
                f"Notch band must have increasing edges, got [{self.low_hz}, {self.high_hz}]."
            )

    @property
    def centre_hz(self) -> float:
        return (self.low_hz + self.high_hz) / 2.0

    @property
    def width_hz(self) -> float:
        return self.high_hz - self.low_hz

    def overlap_hz(self, low_hz: float, high_hz: float) -> float:
        """Hertz shared with another span, zero when they are disjoint."""
        return max(0.0, min(self.high_hz, high_hz) - max(self.low_hz, low_hz))

    def __str__(self) -> str:
        return f"{self.low_hz:g}-{self.high_hz:g} Hz"


@dataclass(frozen=True)
class NotchSettings:
    """Everything the notch stage needs, resolved from configuration."""

    bands: tuple[NotchBand, ...]
    analysed_bands: tuple[tuple[str, float, float], ...]
    filter_jobs: int = 4
    trans_bandwidth_hz: float = 1.0
    """Width of each transition, in Hz, on both sides of a band.

    MNE's own default for ``notch_filter``. It is why a notch always costs more than the
    span it is asked for: the 60 Hz notch this pipeline already applies is configured at
    ``freq/200`` = 0.3 Hz and measures 0.97 Hz wide on the delivered epochs. The manifest
    reports what was actually attenuated rather than what was requested, for that reason.
    """

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("At least one notch band is required.")
        if self.filter_jobs < 1:
            raise ValueError("filter_jobs must be positive.")
        if not np.isfinite(self.trans_bandwidth_hz) or self.trans_bandwidth_hz <= 0.0:
            raise ValueError("trans_bandwidth_hz must be finite and positive.")
        edges = sorted((band.low_hz, band.high_hz) for band in self.bands)
        for (_, earlier_high), (later_low, _) in zip(edges, edges[1:]):
            if later_low < earlier_high:
                raise ValueError(
                    "Notch bands must not overlap; merge them into one span instead."
                )

    @classmethod
    def from_config(cls, config) -> "NotchSettings":
        """Read ``notch_bands`` and the analysed bands from the workflow configuration."""
        configured = config.get("notch_bands")
        if configured is None:
            bands = tuple(NotchBand(low, high) for low, high in DEFAULT_BANDS_HZ)
        else:
            if not isinstance(configured, (list, tuple)) or not configured:
                raise ValueError("notch_bands must be a non-empty list of [low, high] pairs.")
            bands = tuple(_band_from_entry(entry) for entry in configured)
        defaults = cls(bands=bands, analysed_bands=())
        return cls(
            bands=bands,
            analysed_bands=analysed_bands_from_config(config),
            filter_jobs=int(config.get("line_comb_removal.filter_jobs", defaults.filter_jobs)),
            trans_bandwidth_hz=float(
                config.get("notch_trans_bandwidth_hz", defaults.trans_bandwidth_hz)
            ),
        )


def _band_from_entry(entry) -> NotchBand:
    """Build one band from a configuration entry, refusing anything not a pair."""
    if isinstance(entry, dict):
        try:
            return NotchBand(float(entry["low"]), float(entry["high"]))
        except KeyError as error:
            raise ValueError(f"A notch band mapping needs 'low' and 'high': {entry!r}") from error
    values = tuple(entry) if isinstance(entry, (list, tuple)) else ()
    if len(values) != 2:
        raise ValueError(f"Each notch band must be a [low, high] pair, got {entry!r}.")
    return NotchBand(float(values[0]), float(values[1]))


def analysed_bands_from_config(config) -> tuple[tuple[str, float, float], ...]:
    """The bands the notch's cost is reported against.

    Read from the study's own ``frequency_bands`` so the manifest speaks in the units the
    analyses do. Only the canonical five are taken: the ``*_clean`` variants were carved
    out when the harmonics were thought unremovable and no longer describe anything.
    """
    defined = config.get("frequency_bands") or {}
    if not isinstance(defined, dict):
        raise ValueError("frequency_bands must be a mapping of name to [low, high].")
    bands = []
    for name in CANONICAL_BANDS:
        edges = defined.get(name)
        if edges is None:
            continue
        low, high = (float(value) for value in edges)
        if high <= low:
            raise ValueError(f"frequency_bands.{name} must have increasing edges.")
        bands.append((name, low, high))
    return tuple(bands)


def notch_eeg(raw, settings: NotchSettings):
    """Return a copy with every configured band filtered out of the EEG channels."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("The notch requires at least one EEG channel.")
    nyquist_hz = float(raw.info["sfreq"]) / 2.0
    for band in settings.bands:
        if band.high_hz >= nyquist_hz:
            raise ValueError(f"Notch band {band} reaches the {nyquist_hz:g} Hz Nyquist limit.")

    filtered = raw.copy()
    filtered.notch_filter(
        # Arrays, not lists: notch_filter tests ``notch_widths < 0`` before coercing, and a
        # list raises a TypeError there rather than a message about widths.
        freqs=np.array([band.centre_hz for band in settings.bands], dtype=float),
        picks=picks,
        notch_widths=np.array([band.width_hz for band in settings.bands], dtype=float),
        trans_bandwidth=settings.trans_bandwidth_hz,
        method="fir",
        n_jobs=settings.filter_jobs,
        verbose="ERROR",
    )
    return filtered


def band_power(freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Total power across the channel-mean spectrum between two edges."""
    frequency_array = np.asarray(freqs, dtype=float)
    inside = (frequency_array >= low_hz) & (frequency_array <= high_hz)
    if not np.any(inside):
        raise ValueError(f"No frequency bin lies in {low_hz:g}-{high_hz:g} Hz.")
    return float(np.mean(np.asarray(psd, dtype=float)[..., inside].sum(axis=-1)))


def _change_db(before: float, after: float) -> float:
    """Decibel change, negative for a loss. Zero power before means nothing to report."""
    if before <= 0.0:
        return 0.0
    return 10.0 * np.log10(max(after, np.finfo(float).tiny) / before)


def notch_metrics(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    settings: NotchSettings,
) -> list[dict[str, float | str]]:
    """One row per notched band: what it took, and what each analysed band lost."""
    rows: list[dict[str, float | str]] = []
    for band in settings.bands:
        row: dict[str, float | str] = {
            "band_hz": str(band),
            "band_low_hz": band.low_hz,
            "band_high_hz": band.high_hz,
            "in_band_change_db": _change_db(
                band_power(freqs, psd_before, band.low_hz, band.high_hz),
                band_power(freqs, psd_after, band.low_hz, band.high_hz),
            ),
        }
        for name, low_hz, high_hz in settings.analysed_bands:
            before = band_power(freqs, psd_before, low_hz, high_hz)
            after = band_power(freqs, psd_after, low_hz, high_hz)
            row[f"{name}_change_db"] = _change_db(before, after)
            # What the notch costs this band before any question of content: the share of
            # its width the requested span covers. The transitions make the real footprint
            # larger, which is what the change in dB above measures.
            row[f"{name}_width_share"] = band.overlap_hz(low_hz, high_hz) / (high_hz - low_hz)
        rows.append(row)
    return rows


def notch_run(
    vhdr: Path,
    output_root: Path,
    source_root: Path,
    settings: NotchSettings,
    removal_settings: "rm.RemovalSettings | None" = None,
) -> list[dict[str, float | str]]:
    """Notch one recording, write its binary, and measure what changed."""
    import mne

    mne.set_log_level("ERROR")
    raw = rm.read_bids_raw(vhdr)
    filtered = notch_eeg(raw, settings)

    destination = output_root / vhdr.relative_to(source_root).with_suffix(".eeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rm.write_eeg_binary(vhdr, destination, filtered.get_data())

    written = rm.read_bids_raw(output_root / vhdr.relative_to(source_root))
    expected = filtered.get_data()
    deviation = float(np.max(np.abs(written.get_data() - expected)))
    scale = float(np.max(np.abs(expected)))
    tolerance = rm.ROUNDTRIP_RELATIVE_TOLERANCE * scale
    if deviation > tolerance:
        raise RuntimeError(
            f"{vhdr.name}: written data differs by {deviation:.3e} V, "
            f"above the {tolerance:.3e} V float32 round-trip tolerance."
        )

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    geometry = removal_settings or rm.RemovalSettings()
    freqs, psd_before = rm._psd(raw, picks, geometry)
    _, psd_after = rm._psd(filtered, picks, geometry)
    rows = notch_metrics(freqs, psd_before, psd_after, settings)
    for row in rows:
        row["recording"] = vhdr.stem
        row["roundtrip_deviation_v"] = deviation
    return rows


def write_derivative_description(
    output_root: Path,
    source_root: Path,
    settings: NotchSettings,
) -> Path:
    """Declare the notched root a derivative and record the bands that made it."""
    import json

    path = Path(output_root) / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset description was not mirrored to {path}.")
    described = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(described, dict):
        raise ValueError("BIDS dataset_description.json must contain a JSON object.")

    described["DatasetType"] = "derivative"
    described.setdefault("Name", "band-notched EEG")
    described.setdefault("BIDSVersion", "1.8.0")
    existing = described.get("GeneratedBy", [])
    if not isinstance(existing, list) or not all(isinstance(entry, dict) for entry in existing):
        raise ValueError("BIDS GeneratedBy must be a list of objects.")
    generated = [entry for entry in existing if "line-comb notch" not in str(entry.get("Name", ""))]
    generated.append(
        {
            "Name": "line-comb notch",
            "Version": rm._code_revision(),
            "Description": (
                "Wide FIR notch over measured contamination bands that are clusters rather "
                "than resolvable lines. Sidecars are byte-identical to the source; only the "
                ".eeg binaries differ, and only EEG channels are filtered."
            ),
            "Parameters": {
                "bands_hz": [[band.low_hz, band.high_hz] for band in settings.bands],
                "trans_bandwidth_hz": settings.trans_bandwidth_hz,
            },
        }
    )
    described["GeneratedBy"] = generated
    described["SourceDatasets"] = [{"URL": f"../{Path(source_root).name}"}]

    path.write_text(json.dumps(described, indent=2) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    """Execute the notch stage."""
    import time

    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    source_root = config.path("output_root", override=getattr(args, "bids_root", None))
    notched_root = config.path("notched_root", override=getattr(args, "output_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))
    settings = NotchSettings.from_config(config)
    # The task label lives with the removal settings; this stage reads the same dataset.
    removal_settings = rm.RemovalSettings.from_config(config)

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"No line-cleaned dataset at {source_root}. Run `line-comb apply` first; the "
            "notch is the last stage, not a replacement for the removal."
        )
    runs = rm.discover_runs(source_root, subjects=None, task=removal_settings.task)
    print(f"Notching {len(runs)} recordings from {source_root}")
    for band in settings.bands:
        shares = ", ".join(
            f"{name} {band.overlap_hz(low, high) / (high - low):.1%}"
            for name, low, high in settings.analysed_bands
            if band.overlap_hz(low, high) > 0.0
        )
        print(f"  {band} (centre {band.centre_hz:g} Hz, width {band.width_hz:g} Hz){f' -> {shares}' if shares else ''}")

    if notched_root.exists():
        raise FileExistsError(
            f"Refusing to mix a new derivative with existing output: {notched_root}"
        )
    staging = notched_root.with_name(f".{notched_root.name}.staging")
    if staging.exists():
        raise FileExistsError(
            f"Incomplete staging output exists at {staging}; inspect it before retrying."
        )
    staging.mkdir(parents=True)
    print(f"Staging a complete derivative in {staging}")
    print(f"  copied {rm.mirror_sidecars(source_root, staging)} sidecars")

    rows: list[dict[str, float | str]] = []
    for index, vhdr in enumerate(runs, start=1):
        started = time.time()
        measured = notch_run(vhdr, staging, source_root, settings, removal_settings)
        rows.extend(measured)
        summary = "  ".join(f"{row['band_hz']} {row['in_band_change_db']:+.1f} dB" for row in measured)
        print(f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} {summary} ({time.time()-started:.0f}s)")

    frame = pd.DataFrame(rows)
    rm._write_tsv_atomic(frame, staging / "line_comb_notch_manifest.tsv")
    described = write_derivative_description(staging, source_root, settings)

    import os

    os.replace(staging, notched_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    rm._write_tsv_atomic(frame, report_dir / "notch_manifest.tsv")

    print("\nmedian change per band:")
    for band_label, block in frame.groupby("band_hz"):
        costs = "  ".join(
            f"{name} {block[f'{name}_change_db'].median():+.2f} dB"
            for name, _, _ in settings.analysed_bands
            if f"{name}_change_db" in block
        )
        print(f"  {band_label}  in band {block.in_band_change_db.median():+.2f} dB   {costs}")
    print(f"  declared {(notched_root / described.name)} a derivative of {source_root}")
    print(f"  wrote {report_dir / 'notch_manifest.tsv'}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=None, help="Line-cleaned source root")
    parser.add_argument("--output-root", type=Path, default=None, help="Where the notched copy goes")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
