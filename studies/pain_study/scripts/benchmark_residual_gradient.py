"""Benchmark residual scanner-gradient OBS after BrainVision preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd
import yaml

from eeg_pipeline.analysis.qc.residual_gradient import (
    ComponentDecision,
    InjectionSettings,
    SelectionThresholds,
    ValidationInjection,
    build_validation_injection,
    compute_outside_harmonic_psd_change,
    evaluate_injection_recovery,
    select_component_count,
    summarize_candidate,
)
from eeg_pipeline.preprocessing.residual_gradient import (
    ResidualObsSettings,
    VolumeLayout,
    apply_residual_obs,
    brainvision_source_files,
    build_volume_layout,
)


@dataclass(frozen=True)
class WelchSettings:
    nperseg: int
    overlap_fraction: float

    def __post_init__(self) -> None:
        if self.nperseg <= 0:
            raise ValueError("welch.nperseg must be positive.")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("welch.overlap_fraction must be in [0, 1).")


@dataclass(frozen=True)
class BenchmarkConfig:
    pilot_subject: str
    expected_runs: int
    component_counts: tuple[int, ...]
    obs: ResidualObsSettings
    harmonic_windows_hz: tuple[tuple[float, float], ...]
    welch: WelchSettings
    injection: InjectionSettings
    acceptance: SelectionThresholds

    def __post_init__(self) -> None:
        if self.pilot_subject != "0006":
            raise ValueError("pilot_subject must be '0006' for the prespecified excluded pilot.")
        if self.expected_runs != 6:
            raise ValueError("expected_runs must be exactly 6 for the prespecified pilot.")
        if self.component_counts != (0, 1, 2, 3, 4):
            raise ValueError("component_counts must be exactly [0, 1, 2, 3, 4].")
        if not self.harmonic_windows_hz:
            raise ValueError("harmonic_windows_hz must be non-empty.")
        previous_high = 0.0
        for low_hz, high_hz in self.harmonic_windows_hz:
            if not 0 < low_hz < high_hz:
                raise ValueError("Each harmonic window must have 0 < low_hz < high_hz.")
            if low_hz <= previous_high:
                raise ValueError("harmonic_windows_hz must be sorted and non-overlapping.")
            previous_high = high_hz


class OutputPolicy(str, Enum):
    ERROR = "error"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class ReportPaths:
    run_audit: Path
    component_audit: Path
    preservation_audit: Path
    provenance: Path
    decision: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.run_audit,
            self.component_audit,
            self.preservation_audit,
            self.provenance,
            self.decision,
        )


@dataclass(frozen=True)
class RunBenchmark:
    run_rows: tuple[dict[str, Any], ...]
    component_rows: tuple[dict[str, Any], ...]
    preservation_rows: tuple[dict[str, Any], ...]
    candidate_paths: tuple[Path, ...]
    provenance: dict[str, Any]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load the benchmark's exact, fail-fast YAML schema."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark configuration does not exist: {config_path}")
    payload = _require_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "benchmark configuration root",
    )
    _require_exact_keys(
        payload,
        {
            "pilot_subject",
            "expected_runs",
            "component_counts",
            "obs",
            "harmonic_windows_hz",
            "welch",
            "injection",
            "acceptance",
        },
        "benchmark",
    )

    obs_values = _require_mapping(payload["obs"], "obs")
    _require_exact_keys(
        obs_values,
        {
            "expected_sfreq_hz",
            "volume_marker",
            "tr_s",
            "min_complete_epochs",
            "n_folds",
        },
        "obs",
    )
    welch_values = _require_mapping(payload["welch"], "welch")
    _require_exact_keys(welch_values, {"nperseg", "overlap_fraction"}, "welch")
    injection_values = _require_mapping(payload["injection"], "injection")
    _require_exact_keys(
        injection_values,
        {
            "frequencies_hz",
            "sinusoid_amplitude_uv",
            "transient_amplitude_uv",
            "transient_spacing_s",
            "transient_width_s",
        },
        "injection",
    )
    acceptance_values = _require_mapping(payload["acceptance"], "acceptance")
    _require_exact_keys(
        acceptance_values,
        {
            "minimum_sinusoid_amplitude_ratio",
            "maximum_phase_error_deg",
            "minimum_transient_peak_ratio",
            "maximum_outside_harmonic_psd_change_db",
            "maximum_run_prominence_increase_db",
        },
        "acceptance",
    )

    return BenchmarkConfig(
        pilot_subject=_require_string(payload["pilot_subject"], "pilot_subject"),
        expected_runs=_require_integer(payload["expected_runs"], "expected_runs"),
        component_counts=_integer_tuple(payload["component_counts"], "component_counts"),
        obs=ResidualObsSettings(
            expected_sfreq_hz=_require_number(
                obs_values["expected_sfreq_hz"], "obs.expected_sfreq_hz"
            ),
            volume_marker=_require_string(obs_values["volume_marker"], "obs.volume_marker"),
            tr_s=_require_number(obs_values["tr_s"], "obs.tr_s"),
            min_complete_epochs=_require_integer(
                obs_values["min_complete_epochs"], "obs.min_complete_epochs"
            ),
            n_folds=_require_integer(obs_values["n_folds"], "obs.n_folds"),
        ),
        harmonic_windows_hz=_frequency_windows(payload["harmonic_windows_hz"]),
        welch=WelchSettings(
            nperseg=_require_integer(welch_values["nperseg"], "welch.nperseg"),
            overlap_fraction=_require_number(
                welch_values["overlap_fraction"], "welch.overlap_fraction"
            ),
        ),
        injection=InjectionSettings(
            frequencies_hz=_number_tuple(
                injection_values["frequencies_hz"], "injection.frequencies_hz"
            ),
            sinusoid_amplitude_uv=_require_number(
                injection_values["sinusoid_amplitude_uv"],
                "injection.sinusoid_amplitude_uv",
            ),
            transient_amplitude_uv=_require_number(
                injection_values["transient_amplitude_uv"],
                "injection.transient_amplitude_uv",
            ),
            transient_spacing_s=_require_number(
                injection_values["transient_spacing_s"],
                "injection.transient_spacing_s",
            ),
            transient_width_s=_require_number(
                injection_values["transient_width_s"],
                "injection.transient_width_s",
            ),
        ),
        acceptance=SelectionThresholds(
            minimum_sinusoid_amplitude_ratio=_require_number(
                acceptance_values["minimum_sinusoid_amplitude_ratio"],
                "acceptance.minimum_sinusoid_amplitude_ratio",
            ),
            maximum_phase_error_deg=_require_number(
                acceptance_values["maximum_phase_error_deg"],
                "acceptance.maximum_phase_error_deg",
            ),
            minimum_transient_peak_ratio=_require_number(
                acceptance_values["minimum_transient_peak_ratio"],
                "acceptance.minimum_transient_peak_ratio",
            ),
            maximum_outside_harmonic_psd_change_db=_require_number(
                acceptance_values["maximum_outside_harmonic_psd_change_db"],
                "acceptance.maximum_outside_harmonic_psd_change_db",
            ),
            maximum_run_prominence_increase_db=_require_number(
                acceptance_values["maximum_run_prominence_increase_db"],
                "acceptance.maximum_run_prominence_increase_db",
            ),
        ),
    )


def discover_pilot_files(source_root: str | Path, subject: str) -> list[Path]:
    """Discover only Analyzer-corrected headers for the prespecified pilot."""
    root = Path(source_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {root}")
    eeg_dir = root / f"sub-{subject}" / "eeg"
    paths = sorted(
        path
        for path in eeg_dir.glob("*_scannerpulse_corrected.vhdr")
        if path.is_file() and not path.name.startswith("._")
    )
    if not paths:
        raise FileNotFoundError(
            f"No _scannerpulse_corrected.vhdr files found for sub-{subject}: {eeg_dir}"
        )
    return paths


def write_benchmark_reports(
    *,
    output_root: str | Path,
    run_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    preservation_rows: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
    decision: dict[str, Any],
    policy: OutputPolicy,
) -> ReportPaths:
    """Write all benchmark audit artifacts with atomic per-file replacement."""
    if not isinstance(policy, OutputPolicy):
        raise TypeError("policy must be an OutputPolicy.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = _report_paths(root)
    for path in paths.all_paths():
        if path.exists() and policy is OutputPolicy.ERROR:
            raise FileExistsError(f"Refusing to overwrite existing benchmark output: {path}")
    for path, rows in (
        (paths.run_audit, run_rows),
        (paths.component_audit, component_rows),
        (paths.preservation_audit, preservation_rows),
    ):
        if not rows:
            raise ValueError(f"Cannot write an empty benchmark table: {path}")

    _atomic_tsv(paths.run_audit, run_rows)
    _atomic_tsv(paths.component_audit, component_rows)
    _atomic_tsv(paths.preservation_audit, preservation_rows)
    _atomic_json(paths.provenance, provenance)
    _atomic_json(paths.decision, decision)
    return paths


def benchmark_run(
    vhdr_path: Path,
    *,
    output_root: Path,
    config: BenchmarkConfig,
    output_policy: OutputPolicy,
) -> RunBenchmark:
    """Evaluate every fixed OBS order for one pilot recording."""
    source_files = brainvision_source_files(vhdr_path)
    source_path = source_files[0]
    raw = mne.io.read_raw_brainvision(source_path, preload=True, verbose="ERROR")
    layout = build_volume_layout(raw, config.obs)
    run = _extract_run(source_path)
    injection = build_validation_injection(
        raw.n_times,
        float(raw.info["sfreq"]),
        config.injection,
    )
    injected_raw = _add_injection(raw, injection)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    run_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    for count in config.component_counts:
        baseline = apply_residual_obs(
            raw,
            layout,
            n_components=count,
            n_folds=config.obs.n_folds,
        )
        injected = apply_residual_obs(
            injected_raw,
            layout,
            n_components=count,
            n_folds=config.obs.n_folds,
        )
        _assert_candidate_contract(raw, baseline.raw, layout)
        recovered = np.median(
            injected.raw.get_data(eeg_picks) - baseline.raw.get_data(eeg_picks),
            axis=0,
        )
        recovery = evaluate_injection_recovery(
            expected=injection,
            recovered_v=recovered,
            sfreq=float(raw.info["sfreq"]),
        )
        outside_change = compute_outside_harmonic_psd_change(
            raw.get_data(eeg_picks),
            baseline.raw.get_data(eeg_picks),
            sfreq=float(raw.info["sfreq"]),
            nperseg=config.welch.nperseg,
            overlap_fraction=config.welch.overlap_fraction,
            harmonic_windows_hz=config.harmonic_windows_hz,
        )
        run_rows.append(
            summarize_candidate(
                baseline.raw,
                layout=layout,
                source_file=source_path,
                run=run,
                n_components=count,
                nperseg=config.welch.nperseg,
                overlap_fraction=config.welch.overlap_fraction,
                harmonic_windows_hz=config.harmonic_windows_hz,
            )
        )
        component_rows.extend({"run": run, **row} for row in baseline.component_rows)
        preservation_rows.append(
            {
                "run": run,
                "n_components": count,
                **asdict(recovery),
                "outside_harmonic_psd_change_db": outside_change.median_change_db,
                "maximum_channel_outside_harmonic_psd_change_db": (
                    outside_change.maximum_channel_absolute_change_db
                ),
            }
        )
        if count > 0:
            candidate_path = _candidate_path(output_root, config.pilot_subject, run, count)
            _save_candidate(baseline.raw, candidate_path, output_policy)
            candidate_paths.append(candidate_path)

    provenance = {
        "input_files": [{"path": str(path), "sha256": _sha256(path)} for path in source_files],
        "run": run,
        "eligible_epochs": layout.n_epochs,
        "blocks": int(layout.block_ids.max()) + 1,
        "excluded_samples": raw.n_times - layout.n_epochs * layout.epoch_samples,
        "candidate_paths": [str(path) for path in candidate_paths],
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
    }
    return RunBenchmark(
        run_rows=tuple(run_rows),
        component_rows=tuple(component_rows),
        preservation_rows=tuple(preservation_rows),
        candidate_paths=tuple(candidate_paths),
        provenance=provenance,
    )


def run_pilot_benchmark(
    *,
    source_root: Path,
    output_root: Path,
    config: BenchmarkConfig,
    output_policy: OutputPolicy,
) -> ComponentDecision:
    """Run and report the prespecified six-run excluded-pilot benchmark."""
    paths = discover_pilot_files(source_root, config.pilot_subject)
    if len(paths) != config.expected_runs:
        raise ValueError(
            f"Expected {config.expected_runs} pilot runs for sub-{config.pilot_subject}, "
            f"found {len(paths)}."
        )
    runs = tuple(_extract_run(path) for path in paths)
    expected_runs = tuple(range(1, config.expected_runs + 1))
    if tuple(sorted(runs)) != expected_runs:
        raise ValueError(
            f"Expected pilot run indices {expected_runs}, found {tuple(sorted(runs))}."
        )
    _preflight_outputs(paths, output_root, config, output_policy)

    results = [
        benchmark_run(
            path,
            output_root=output_root,
            config=config,
            output_policy=output_policy,
        )
        for path in paths
    ]
    run_rows = [row for result in results for row in result.run_rows]
    component_rows = [row for result in results for row in result.component_rows]
    preservation_rows = [row for result in results for row in result.preservation_rows]
    decision = select_component_count(
        run_rows,
        preservation_rows,
        config.component_counts,
        config.acceptance,
    )
    provenance = {
        "subject": config.pilot_subject,
        "config": asdict(config),
        "config_sha256": _payload_sha256(asdict(config)),
        "runs": [result.provenance for result in results],
    }
    write_benchmark_reports(
        output_root=output_root,
        run_rows=run_rows,
        component_rows=component_rows,
        preservation_rows=preservation_rows,
        provenance=provenance,
        decision=decision.to_dict(),
        policy=output_policy,
    )
    return decision


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately narrow benchmark command-line interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark residual OBS on BrainVision scanner- and pulse-corrected EEG."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-policy",
        choices=[policy.value for policy in OutputPolicy],
        default=OutputPolicy.ERROR.value,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_benchmark_config(args.config)
    decision = run_pilot_benchmark(
        source_root=args.source_root,
        output_root=args.output_root,
        config=config,
        output_policy=OutputPolicy(args.output_policy),
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


def _temporary_path(destination: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _atomic_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = _temporary_path(path)
    try:
        pd.DataFrame(rows).to_csv(temporary, sep="\t", index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _report_paths(root: Path) -> ReportPaths:
    return ReportPaths(
        run_audit=root / "residual_obs_run_audit.tsv",
        component_audit=root / "residual_obs_component_variance.tsv",
        preservation_audit=root / "residual_obs_signal_preservation.tsv",
        provenance=root / "residual_obs_provenance.json",
        decision=root / "residual_obs_decision.json",
    )


def _candidate_path(root: Path, subject: str, run: int, count: int) -> Path:
    return (
        root
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_run-{run:02d}_desc-resobs{count:02d}_raw.fif"
    )


def _preflight_outputs(
    source_paths: Sequence[Path],
    output_root: Path,
    config: BenchmarkConfig,
    policy: OutputPolicy,
) -> None:
    if not isinstance(policy, OutputPolicy):
        raise TypeError("output_policy must be an OutputPolicy.")
    if policy is OutputPolicy.OVERWRITE:
        return

    destinations = list(_report_paths(output_root).all_paths())
    for source_path in source_paths:
        run = _extract_run(source_path)
        destinations.extend(
            _candidate_path(output_root, config.pilot_subject, run, count)
            for count in config.component_counts
            if count > 0
        )
    existing = [path for path in destinations if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing benchmark outputs:\n{formatted}")


def _add_injection(raw: mne.io.BaseRaw, injection: ValidationInjection) -> mne.io.BaseRaw:
    injected = raw.copy().load_data()
    picks = mne.pick_types(injected.info, eeg=True, exclude=[])
    injected._data[picks] += injection.signal_v[np.newaxis, :]
    return injected


def _assert_candidate_contract(
    source: mne.io.BaseRaw,
    candidate: mne.io.BaseRaw,
    layout: VolumeLayout,
) -> None:
    if candidate.n_times != source.n_times:
        raise RuntimeError("Residual OBS changed the sample count.")
    if candidate.ch_names != source.ch_names:
        raise RuntimeError("Residual OBS changed channel order or names.")
    if candidate.get_channel_types() != source.get_channel_types():
        raise RuntimeError("Residual OBS changed channel types.")
    if candidate.info["meas_date"] != source.info["meas_date"]:
        raise RuntimeError("Residual OBS changed the measurement date.")
    annotations_match = (
        np.array_equal(candidate.annotations.onset, source.annotations.onset)
        and np.array_equal(candidate.annotations.duration, source.annotations.duration)
        and np.array_equal(candidate.annotations.description, source.annotations.description)
        and candidate.annotations.orig_time == source.annotations.orig_time
    )
    if not annotations_match:
        raise RuntimeError("Residual OBS changed annotations.")
    candidate_data = candidate.get_data()
    source_data = source.get_data()
    if not np.isfinite(candidate_data).all():
        raise RuntimeError("Residual OBS produced non-finite samples.")

    eeg_picks = mne.pick_types(source.info, eeg=True, exclude=[])
    non_eeg_picks = np.setdiff1d(np.arange(len(source.ch_names)), eeg_picks)
    if not np.array_equal(candidate_data[non_eeg_picks], source_data[non_eeg_picks]):
        raise RuntimeError("Residual OBS changed a non-EEG channel.")
    eligible = np.zeros(source.n_times, dtype=bool)
    for start in layout.starts:
        eligible[start : start + layout.epoch_samples] = True
    if not np.array_equal(candidate_data[:, ~eligible], source_data[:, ~eligible]):
        raise RuntimeError("Residual OBS changed samples outside complete volume epochs.")


def _save_candidate(raw: mne.io.BaseRaw, path: Path, policy: OutputPolicy) -> None:
    if path.exists() and policy is OutputPolicy.ERROR:
        raise FileExistsError(f"Refusing to overwrite candidate derivative: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name.removesuffix('_raw.fif')}.",
        suffix="_raw.fif",
        dir=path.parent,
        delete=False,
    )
    handle.close()
    temporary = Path(handle.name)
    temporary.unlink()
    try:
        raw.save(temporary, overwrite=False, verbose="ERROR")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_run(path: Path) -> int:
    match = re.search(r"(?:^|_)run-?(\d+)(?:_|$)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"BrainVision filename has no explicit run index: {path.name}")
    return int(match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    extra = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if extra:
        raise ValueError(f"Unknown {label} configuration keys: {extra}")
    if missing:
        raise ValueError(f"Missing {label} configuration keys: {missing}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys.")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def _require_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer.")
    return value


def _require_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError(f"{label} must be a finite number.")
    return float(value)


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list.")
    return value


def _integer_tuple(value: Any, label: str) -> tuple[int, ...]:
    values = _require_list(value, label)
    return tuple(_require_integer(item, f"{label} item") for item in values)


def _number_tuple(value: Any, label: str) -> tuple[float, ...]:
    values = _require_list(value, label)
    return tuple(_require_number(item, f"{label} item") for item in values)


def _frequency_windows(value: Any) -> tuple[tuple[float, float], ...]:
    windows = _require_list(value, "harmonic_windows_hz")
    parsed: list[tuple[float, float]] = []
    for index, window in enumerate(windows):
        bounds = _number_tuple(window, f"harmonic_windows_hz[{index}]")
        if len(bounds) != 2:
            raise ValueError(f"harmonic_windows_hz[{index}] must contain exactly two bounds.")
        parsed.append(bounds)
    return tuple(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
