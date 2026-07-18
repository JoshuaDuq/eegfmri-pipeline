"""End-to-end pipeline for condition-separated ICA-component TFRs."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.analysis.component_tfr import (
    ComponentTFRParameters,
    ConditionTFR,
    apply_unmixing_to_broadband_epochs,
    compute_condition_tfrs,
    fit_band_limited_ica,
    prepare_component_epochs_input,
)
from eeg_pipeline.infra.paths import find_clean_epochs_path
from eeg_pipeline.plotting.component_tfr import (
    save_component_tfr_overview,
    save_ica_topography_overview,
)


class ComponentTFRPipeline:
    """Run analysis ICA and component TFRs for one or more subjects."""

    def __init__(self, config: Any, deriv_root: Path, logger: logging.Logger | None = None):
        self.config = config
        self.deriv_root = Path(deriv_root)
        self.logger = logger or logging.getLogger(__name__)
        self.parameters = ComponentTFRParameters.from_config(config)

    def run_batch(
        self,
        subjects: Sequence[str],
        task: str,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> list[Path]:
        """Run the complete analysis for every selected subject."""
        if not subjects:
            raise ValueError("Component TFR analysis requires at least one subject.")

        outputs = []
        for subject in subjects:
            inputs = self._discover_inputs(subject, task)
            output_dir = self._output_dir(subject, task)
            if dry_run:
                self.logger.info(
                    "Would compute component TFRs for sub-%s, task-%s from %s into %s",
                    _subject_id(subject),
                    task,
                    [str(path) for path in inputs],
                    output_dir,
                )
                continue
            outputs.append(
                self._run_subject(
                    subject=subject,
                    task=task,
                    input_paths=inputs,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )
            )
        return outputs

    def _run_subject(
        self,
        *,
        subject: str,
        task: str,
        input_paths: list[Path],
        output_dir: Path,
        overwrite: bool,
    ) -> Path:
        if output_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Component TFR output already exists: {output_dir}. Pass --overwrite "
                "to replace this subject/task output."
            )

        self.logger.info(
            "Computing component TFRs for sub-%s, task-%s",
            _subject_id(subject),
            task,
        )
        broadband_epochs, event_paths = self._load_inputs(input_paths)
        ica = fit_band_limited_ica(broadband_epochs, self.parameters)
        component_epochs = apply_unmixing_to_broadband_epochs(ica, broadband_epochs)
        condition_tfrs = compute_condition_tfrs(component_epochs, self.parameters)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".component-tfr-", dir=output_dir.parent))
        try:
            self._write_outputs(
                output_dir=staging_dir,
                subject=subject,
                task=task,
                input_paths=input_paths,
                event_paths=event_paths,
                broadband_epochs=broadband_epochs,
                ica=ica,
                component_epochs=component_epochs,
                condition_tfrs=condition_tfrs,
            )
            if output_dir.exists():
                shutil.rmtree(output_dir)
            staging_dir.replace(output_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        self.logger.info("Saved component TFR outputs to %s", output_dir)
        return output_dir

    def _load_inputs(self, input_paths: list[Path]) -> tuple[mne.Epochs, list[Path]]:
        prepared_epochs = []
        event_paths = []
        for epochs_path in input_paths:
            events_path = _events_path_for_epochs(epochs_path)
            if not events_path.exists():
                raise FileNotFoundError(
                    f"Clean events file corresponding to {epochs_path} does not exist: "
                    f"{events_path}"
                )
            epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
            events = pd.read_csv(events_path, sep="\t")
            prepared_epochs.append(prepare_component_epochs_input(epochs, events, self.parameters))
            event_paths.append(events_path)

        if len(prepared_epochs) == 1:
            return prepared_epochs[0], event_paths
        combined = mne.concatenate_epochs(prepared_epochs, add_offset=True, verbose=False)
        return combined, event_paths

    def _write_outputs(
        self,
        *,
        output_dir: Path,
        subject: str,
        task: str,
        input_paths: list[Path],
        event_paths: list[Path],
        broadband_epochs: mne.Epochs,
        ica: mne.preprocessing.ICA,
        component_epochs: mne.Epochs,
        condition_tfrs: list[ConditionTFR],
    ) -> None:
        file_prefix = f"sub-{_subject_id(subject)}_task-{task}"
        ica_path = output_dir / f"{file_prefix}_desc-bandpass6to14-ica.fif"
        component_epochs_path = output_dir / (f"{file_prefix}_desc-broadband-components-epo.fif")
        ica.save(ica_path, overwrite=True, verbose=False)
        component_epochs.save(
            component_epochs_path,
            overwrite=True,
            split_naming="bids",
            verbose=False,
        )

        condition_rows = []
        for condition in condition_tfrs:
            raw_path = output_dir / (
                f"{file_prefix}_cond-{condition.file_label}_desc-rawpower-tfr.h5"
            )
            baseline_path = output_dir / (
                f"{file_prefix}_cond-{condition.file_label}_desc-logratio-tfr.h5"
            )
            condition.raw_power.save(raw_path, overwrite=True, verbose=False)
            condition.baseline_power.save(baseline_path, overwrite=True, verbose=False)
            condition_rows.append(
                {
                    "condition_column": self.parameters.condition_column,
                    "condition_value": condition.value,
                    "condition_label": condition.label,
                    "epoch_count": condition.epoch_count,
                    "raw_power_file": raw_path.name,
                    "baseline_power_file": baseline_path.name,
                }
            )

        pd.DataFrame(condition_rows).to_csv(
            output_dir / f"{file_prefix}_desc-conditions.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame(
            {
                "component": np.arange(ica.n_components_, dtype=int),
                "component_name": component_epochs.ch_names,
            }
        ).to_csv(
            output_dir / f"{file_prefix}_desc-components.tsv",
            sep="\t",
            index=False,
        )

        save_ica_topography_overview(
            ica=ica,
            epochs=broadband_epochs,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        save_component_tfr_overview(
            condition_tfrs=condition_tfrs,
            condition_column=self.parameters.condition_column,
            output_dir=output_dir,
            file_prefix=file_prefix,
            components_per_page=self.parameters.components_per_page,
        )
        _write_provenance(
            path=output_dir / f"{file_prefix}_desc-provenance.json",
            subject=_subject_id(subject),
            task=task,
            input_paths=input_paths,
            event_paths=event_paths,
            parameters=self.parameters,
            epoch_count=int(len(component_epochs)),
            component_count=int(ica.n_components_),
            condition_tfrs=condition_tfrs,
        )

    def _discover_inputs(self, subject: str, task: str) -> list[Path]:
        subject_label = f"sub-{_subject_id(subject)}"
        epochs_path = find_clean_epochs_path(
            subject=_subject_id(subject),
            task=task,
            deriv_root=self.deriv_root,
            config=self.config,
        )
        if epochs_path is None or not epochs_path.is_file():
            raise FileNotFoundError(
                f"No clean epochs found for {subject_label}, task-{task} under "
                f"{self.deriv_root}."
            )
        return [epochs_path.resolve()]

    def _output_dir(self, subject: str, task: str) -> Path:
        return (
            self.deriv_root
            / "component_time_frequency"
            / f"sub-{_subject_id(subject)}"
            / "eeg"
            / f"task-{task}"
        )


def _events_path_for_epochs(epochs_path: Path) -> Path:
    replacements = (
        ("_proc-clean_epo.fif", "_proc-clean_events.tsv"),
        ("_proc-cleaned_epo.fif", "_proc-cleaned_events.tsv"),
        ("_clean_epo.fif", "_clean_events.tsv"),
    )
    for epochs_suffix, events_suffix in replacements:
        if epochs_path.name.endswith(epochs_suffix):
            return epochs_path.with_name(epochs_path.name.replace(epochs_suffix, events_suffix))
    raise ValueError(f"Unsupported clean-epochs filename: {epochs_path}")


def _write_provenance(
    *,
    path: Path,
    subject: str,
    task: str,
    input_paths: list[Path],
    event_paths: list[Path],
    parameters: ComponentTFRParameters,
    epoch_count: int,
    component_count: int,
    condition_tfrs: list[ConditionTFR],
) -> None:
    provenance = {
        "analysis": "component_time_frequency",
        "subject": subject,
        "task": task,
        "input_epochs": [str(input_path) for input_path in input_paths],
        "input_events": [str(event_path) for event_path in event_paths],
        "parameters": asdict(parameters),
        "epoch_count": epoch_count,
        "component_count": component_count,
        "conditions": [
            {
                "value": _json_scalar(condition.value),
                "label": condition.label,
                "epoch_count": condition.epoch_count,
            }
            for condition in condition_tfrs
        ],
        "software": {
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _subject_id(subject: str) -> str:
    normalized = str(subject).strip()
    if normalized.startswith("sub-"):
        normalized = normalized[4:]
    if not normalized:
        raise ValueError("Subject identifier must not be empty.")
    return normalized
