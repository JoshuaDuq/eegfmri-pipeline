"""
Preprocessing Pipeline
======================

Pipeline class for EEG preprocessing orchestration:
- Bad channel detection (PyPREP)
- ICA fitting and labeling (MNE-BIDS pipeline + mne-icalabel)
- Epoch creation and cleaning

Usage:
    pipeline = PreprocessingPipeline(config=config)
    pipeline.run_batch(subjects, task="task", mode="ica")

Modes:
- bad-channels: Only bad channel detection
- ica: ICA fitting and native MNE-BIDS artifact classification
- epochs: Only epoch creation and cleaning
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.pipelines.progress import ensure_progress_reporter
from eeg_pipeline.utils.config.loader import get_condition_column_candidates
from eeg_pipeline.utils.config.roots import resolve_eeg_bids_root, resolve_eeg_deriv_root

STEP_BAD_CHANNELS = "bad-channels"
STEP_ICA_FIT = "ica-fit"
STEP_EPOCHS = "epochs"
STEP_STATS = "stats"
STEP_SCANNER_HARMONIC_QC = "scanner-harmonic-qc"
STEP_PULSE_MARKER_QC = "pulse-marker-qc"
STEP_ICA_CARDIAC_QC = "ica-cardiac-qc"
STEP_CARDIAC_ATTENUATION_QC = "cardiac-attenuation-qc"


def _is_events_tsv(path: Path) -> bool:
    return path.is_file() and path.name.endswith("_events.tsv") and not path.name.startswith("._")


class PreprocessingPipeline(PipelineBase):
    """Pipeline for EEG preprocessing.

    This pipeline wraps the preprocessing workflow from the
    eeg_pipeline/preprocessing/ module, providing:

    1. Bad channel detection using PyPREP
    2. Bad channel synchronization across runs
    3. ICA fitting via MNE-BIDS pipeline
    4. ICA component labeling via mne-icalabel
    5. Epoch creation with ICA applied
    6. Preprocessing statistics collection

    Attributes:
        bids_root: Path to BIDS dataset
        deriv_root: Path to derivatives output
    """

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="preprocessing", config=config)
        self._apply_processing_roots(self._resolve_task_is_rest())

    def _resolve_pipeline_deriv_root(self) -> Path:
        """Resolve EEG preprocessing derivatives root."""
        return resolve_eeg_deriv_root(
            self.config,
            task_is_rest=self._resolve_task_is_rest(),
        )

    def _apply_processing_roots(self, task_is_rest: bool) -> None:
        """Resolve and assign BIDS and derivatives roots for the active processing mode."""
        resolved_task_is_rest = bool(task_is_rest)
        self.bids_root = resolve_eeg_bids_root(
            self.config,
            task_is_rest=resolved_task_is_rest,
        )
        self.deriv_root = resolve_eeg_deriv_root(
            self.config,
            task_is_rest=resolved_task_is_rest,
        )

    def _refresh_processing_roots_if_initialized(self, task_is_rest: bool) -> None:
        """Refresh processing roots only when the pipeline was fully initialized."""
        if hasattr(self, "bids_root") and hasattr(self, "deriv_root"):
            self._apply_processing_roots(task_is_rest)

    def _extract_preprocessing_params(
        self,
        task: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Optional[str], str, bool, bool, int, Any]:
        """Extract and normalize preprocessing parameters from kwargs.

        Returns:
            Tuple of (
                resolved_task,
                mode,
                use_pyprep,
                task_is_rest,
                n_jobs,
                progress,
            )
        """
        task_is_rest = self._resolve_task_is_rest(kwargs.get("task_is_rest"))
        resolved_task = self._resolve_requested_task(task, task_is_rest)
        mode = kwargs.get("mode", "ica")
        use_pyprep = kwargs.get("use_pyprep", True)
        n_jobs = kwargs.get("n_jobs", 1)
        progress = ensure_progress_reporter(kwargs.get("progress"))

        return (
            resolved_task,
            mode,
            use_pyprep,
            task_is_rest,
            n_jobs,
            progress,
        )

    def _resolve_requested_task(
        self,
        task: Optional[str],
        task_is_rest: bool,
    ) -> Optional[str]:
        """Resolve the task selector for the active preprocessing mode."""
        if task is not None:
            return task
        if task_is_rest:
            return None

        resolved_task = self.config.get("project.task")
        if resolved_task is None:
            raise ValueError("Missing required config value: project.task")
        return resolved_task

    def _resolve_task_is_rest(self, override: Optional[bool] = None) -> bool:
        """Resolve resting-state mode from explicit override or config."""
        if override is not None:
            return bool(override)
        return bool(self.config.get("preprocessing.task_is_rest", False))

    def process_subject(
        self,
        subject: str,
        task: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Process a single subject through preprocessing steps.

        Args:
            subject: Subject ID without 'sub-' prefix
            task: Task name (defaults to config value)
            **kwargs: Additional options:
                - mode: 'bad-channels', 'ica', or 'epochs'
                - task_is_rest: Override config to enable/disable resting-state preprocessing
                - n_jobs: Number of parallel jobs
                - progress: ProgressReporter for TUI feedback
        """
        resolved_task, mode, use_pyprep, task_is_rest, n_jobs, progress = (
            self._extract_preprocessing_params(task, kwargs)
        )
        self._refresh_processing_roots_if_initialized(task_is_rest)

        progress.subject_start(f"sub-{subject}")

        try:
            steps = self._get_steps_for_run(mode, task_is_rest)

            self._execute_steps(
                steps=steps,
                subjects=[subject],
                task=resolved_task,
                use_pyprep=use_pyprep,
                task_is_rest=task_is_rest,
                n_jobs=n_jobs,
                progress=progress,
            )
        except Exception:
            progress.subject_done(f"sub-{subject}", success=False)
            raise

        progress.subject_done(f"sub-{subject}", success=True)

    def run_batch(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run preprocessing for multiple subjects.

        Args:
            subjects: List of subject IDs
            task: Task name
            **kwargs: Preprocessing options:
                - mode: 'bad-channels', 'ica', or 'epochs'
                - task_is_rest: Override config to enable/disable resting-state preprocessing
                - n_jobs: Number of parallel jobs
                - progress: ProgressReporter for TUI feedback

        Returns:
            List of per-subject status dictionaries
        """
        resolved_task, mode, use_pyprep, task_is_rest, n_jobs, progress = (
            self._extract_preprocessing_params(task, kwargs)
        )
        self._refresh_processing_roots_if_initialized(task_is_rest)
        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs=kwargs,
        )
        run_status = "failed"
        run_error: Optional[str] = None
        caught_error: Optional[Exception] = None
        result: Optional[List[Dict[str, Any]]] = None
        run_outputs: Dict[str, str] = {}

        try:
            progress.start("preprocessing", subjects)

            steps = self._get_steps_for_run(mode, task_is_rest)

            run_outputs = self._execute_steps(
                steps=steps,
                subjects=subjects,
                task=resolved_task,
                use_pyprep=use_pyprep,
                task_is_rest=task_is_rest,
                n_jobs=n_jobs,
                progress=progress,
            )

            progress.complete(success=True)
            run_status = "success"
            result = [
                {
                    "subject": subject,
                    "mode": mode,
                    "status": "success",
                }
                for subject in subjects
            ]
        except Exception as exc:
            progress.complete(success=False)
            run_error = str(exc)
            caught_error = exc
        finally:
            metadata_error: Optional[Exception] = None
            try:
                self._write_run_metadata(
                    run_context,
                    status=run_status,
                    error=run_error,
                    outputs=run_outputs,
                    summary={
                        "n_subjects": len(subjects),
                        "mode": mode,
                    },
                )
            except Exception as exc:
                metadata_error = exc

            if caught_error is not None:
                if metadata_error is not None:
                    caught_error.add_note(f"Run metadata writing also failed: {metadata_error}")
                raise caught_error
            if metadata_error is not None:
                raise metadata_error

        if result is None:
            raise RuntimeError("Preprocessing batch completed without producing a result.")
        return result

    def _normalize_subjects(self, subjects: List[str]) -> Union[str, List[str]]:
        """Normalize subjects list to 'all' string if needed."""
        if subjects == ["all"]:
            return "all"
        return subjects

    def _get_steps_for_mode(self, mode: str) -> List[str]:
        """Get preprocessing steps for the given mode."""
        mode_steps = {
            "full": [STEP_BAD_CHANNELS, STEP_ICA_FIT, STEP_EPOCHS, STEP_STATS],
            "bad-channels": [STEP_BAD_CHANNELS],
            "ica": [STEP_ICA_FIT],
            "epochs": [STEP_EPOCHS, STEP_STATS],
        }

        manual_review_required = bool(self.config.get("ica.require_manual_review", False))
        manual_review_complete = bool(self.config.get("ica.manual_review_complete", False))
        if mode in {"full", "epochs"} and manual_review_required and not manual_review_complete:
            raise ValueError(
                "Epoch creation is blocked because manual ICA review is required. Run "
                "mode='ica', review *_proc-ica_components.tsv, then set "
                "ica.manual_review_complete=true before running mode='epochs'."
            )

        if mode not in mode_steps:
            raise ValueError(f"Unknown preprocessing mode: {mode}")

        return mode_steps[mode]

    def _get_steps_for_run(self, mode: str, task_is_rest: bool) -> List[str]:
        """Append cohort QC only when task epoch outputs are produced."""
        steps = self._get_steps_for_mode(mode)
        analyzer_enabled = bool(
            self.config.get("preprocessing.brainvision_analyzer.enabled", False)
        )
        if analyzer_enabled:
            steps.insert(0, STEP_PULSE_MARKER_QC)
            if mode == "ica":
                steps.append(STEP_ICA_CARDIAC_QC)
            if mode == "epochs":
                steps.append(STEP_CARDIAC_ATTENUATION_QC)
                if not task_is_rest:
                    steps.append(STEP_SCANNER_HARMONIC_QC)
        return steps

    def _execute_steps(
        self,
        steps: List[str],
        subjects: List[str],
        task: Optional[str],
        use_pyprep: bool,
        task_is_rest: bool,
        n_jobs: int,
        progress: Any,
    ) -> Dict[str, str]:
        """Execute preprocessing steps in sequence."""
        total_steps = len(steps)
        outputs: Dict[str, str] = {}

        for i, step in enumerate(steps, 1):
            progress.step(step, current=i, total=total_steps)
            self.logger.info("Running step: %s", step)

            if step == STEP_PULSE_MARKER_QC:
                output_path = self._run_pulse_marker_qc(
                    subjects=subjects,
                    task=task,
                )
                outputs["pulse_marker_qc_tsv"] = str(output_path)
            elif step == STEP_ICA_CARDIAC_QC:
                output_path = self._run_marker_ctps_qc(
                    subjects=subjects,
                    task=task,
                )
                outputs["marker_ctps_qc_tsv"] = str(output_path)
            elif step == STEP_CARDIAC_ATTENUATION_QC:
                output_path = self._run_cardiac_attenuation_qc(
                    subjects=subjects,
                    task=task,
                )
                outputs["cardiac_attenuation_qc_tsv"] = str(output_path)
                outputs["cardiac_attenuation_qc_png"] = str(output_path.with_suffix(".png"))
            elif step == STEP_BAD_CHANNELS:
                if not use_pyprep:
                    self.logger.info("Skipping bad channel detection (PyPREP disabled)")
                    continue
                self._run_bad_channel_detection(
                    subjects=subjects,
                    task=task,
                    n_jobs=n_jobs,
                )
            elif step == STEP_ICA_FIT:
                self._run_ica_fitting(
                    subjects=subjects,
                    task=task,
                    task_is_rest=task_is_rest,
                )
            elif step == STEP_EPOCHS:
                self._run_epoch_creation(
                    subjects=subjects,
                    task=task,
                    task_is_rest=task_is_rest,
                )
            elif step == STEP_STATS:
                self._collect_stats(task=task)
            elif step == STEP_SCANNER_HARMONIC_QC:
                if task is None:
                    raise ValueError("Scanner harmonic QC requires a task name.")
                outputs.update(
                    self._run_scanner_harmonic_qc(
                        subjects=subjects,
                        task=task,
                    )
                )
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")

        return outputs

    def _run_pulse_marker_qc(
        self,
        subjects: List[str],
        task: Optional[str],
    ) -> Path:
        """Validate preserved Analyzer R markers before preprocessing."""
        from mne_bids import BIDSPath, read_raw_bids

        from eeg_pipeline.preprocessing.pulse_artifact_qc import (
            PulseMarkerCriteria,
            validate_pulse_marker_recordings,
        )

        qc_config = self.config.get("preprocessing.brainvision_analyzer.pulse_artifact_qc")
        if not qc_config:
            raise ValueError(
                "Missing required config mapping: "
                "preprocessing.brainvision_analyzer.pulse_artifact_qc"
            )

        selected_subjects = [None] if subjects == ["all"] else subjects
        bids_paths = []
        for subject in selected_subjects:
            bids_paths.extend(
                BIDSPath(
                    root=self.bids_root,
                    subject=subject,
                    task=task,
                    datatype="eeg",
                    suffix="eeg",
                    extension=self.config.get("pyprep.file_extension"),
                    check=True,
                ).match()
            )
        visible_paths = {
            str(path.fpath): path
            for path in bids_paths
            if path.fpath.is_file() and not path.fpath.name.startswith("._")
        }
        bids_paths = [visible_paths[key] for key in sorted(visible_paths)]
        if not bids_paths:
            raise FileNotFoundError(
                f"No BIDS EEG recordings found for pulse-marker QC under {self.bids_root}."
            )

        recordings = [
            (
                path.fpath.stem,
                read_raw_bids(path, extra_params={"preload": False}, verbose=False),
            )
            for path in bids_paths
        ]
        criteria = PulseMarkerCriteria(
            minimum_bpm=float(qc_config["minimum_bpm"]),
            maximum_bpm=float(qc_config["maximum_bpm"]),
            minimum_marker_fraction=float(qc_config["minimum_marker_fraction"]),
            minimum_recording_coverage=float(qc_config["minimum_recording_coverage"]),
        )
        task_entity = f"task-{task}_" if task is not None else ""
        output_path = (
            self.deriv_root
            / "preprocessed"
            / "eeg"
            / "qc"
            / f"{task_entity}desc-pulsemarkers_qc.tsv"
        )
        return validate_pulse_marker_recordings(
            recordings,
            criteria,
            output_path=output_path,
        )

    def _get_analyzer_cardiac_qc_config(self) -> Any:
        config = self.config.get("preprocessing.brainvision_analyzer.cardiac_artifact_qc")
        if not config:
            raise ValueError(
                "Missing required config mapping: "
                "preprocessing.brainvision_analyzer.cardiac_artifact_qc"
            )
        return config

    def _run_marker_ctps_qc(
        self,
        subjects: List[str],
        task: Optional[str],
    ) -> Path:
        """Add marker-based CTPS evidence to native ICA component metadata."""
        from eeg_pipeline.preprocessing.cardiac_artifact_qc import run_marker_ctps_qc

        config = self._get_analyzer_cardiac_qc_config()
        return run_marker_ctps_qc(
            pipeline_root=self.deriv_root / "preprocessed" / "eeg",
            subjects=subjects,
            task=task,
            threshold=float(config["ctps_threshold"]),
            epoch_window=tuple(config["ctps_epoch_window"]),
        )

    def _run_cardiac_attenuation_qc(
        self,
        subjects: List[str],
        task: Optional[str],
    ) -> Path:
        """Measure marker-locked EEG attenuation after ICA application."""
        from eeg_pipeline.preprocessing.cardiac_artifact_qc import (
            run_cardiac_attenuation_qc,
        )

        config = self._get_analyzer_cardiac_qc_config()
        return run_cardiac_attenuation_qc(
            pipeline_root=self.deriv_root / "preprocessed" / "eeg",
            subjects=subjects,
            task=task,
            baseline=tuple(config["baseline"]),
            measurement_window=tuple(config["measurement_window"]),
        )

    def _run_bad_channel_detection(
        self,
        subjects: List[str],
        task: Optional[str],
        n_jobs: int = 1,
    ) -> None:
        """Detect bad channels using PyPREP."""
        from eeg_pipeline.preprocessing.pipeline.preprocess import (
            run_bads_detection,
            synchronize_bad_channels_across_runs,
        )

        normalized_subjects = self._normalize_subjects(subjects)
        subject_count = len(subjects) if isinstance(normalized_subjects, list) else "all"
        self.logger.info("Running PyPREP bad channel detection for %s subject(s)", subject_count)

        pyprep_cfg = self.config.get("pyprep", {})
        bad_channel_sync_policy = self._resolve_bad_channel_sync_policy()
        random_state = pyprep_cfg.get("random_state")
        if random_state is None:
            random_state = self.config.get("project.random_state", 42)
        run_bads_detection(
            bids_path=str(self.bids_root),
            pipeline_path=str(self.deriv_root / "preprocessed" / "eeg"),
            task=task,
            subjects=normalized_subjects,
            n_jobs=n_jobs,
            montage=self.config.get("eeg.montage", "easycap-M1"),
            l_pass=self.config.get("preprocessing.h_freq", 100),
            notch=self.config.get("preprocessing.notch_freq"),
            ransac=pyprep_cfg.get("ransac", False),
            repeats=pyprep_cfg.get("repeats", 3),
            average_reref=pyprep_cfg.get("average_reref", False),
            file_extension=pyprep_cfg.get("file_extension", ".vhdr"),
            consider_previous_bads=pyprep_cfg.get("consider_previous_bads", False),
            overwrite_chans_tsv=pyprep_cfg.get("overwrite_chans_tsv", True),
            delete_breaks=pyprep_cfg.get("delete_breaks", False),
            breaks_min_length=pyprep_cfg.get("breaks_min_length", 20),
            t_start_after_previous=pyprep_cfg.get("t_start_after_previous", 2),
            t_stop_before_next=pyprep_cfg.get("t_stop_before_next", 2),
            rename_anot_dict=pyprep_cfg.get("rename_anot_dict"),
            custom_bad_dict=pyprep_cfg.get("custom_bad_dict"),
            random_state=random_state,
        )

        if bad_channel_sync_policy == "subject_union":
            synchronize_bad_channels_across_runs(
                bids_path=str(self.bids_root),
                task=task,
                subjects=normalized_subjects,
            )
        else:
            self.logger.info(
                "Keeping PyPREP bad-channel markings per run "
                "(pyprep.bad_channel_sync_policy='per_run')"
            )

        self.logger.info("Bad channel detection complete")

    def _resolve_bad_channel_sync_policy(self) -> str:
        """Return the explicitly configured PyPREP bad-channel sync policy."""
        bad_channel_sync_policy = self.config.get("pyprep.bad_channel_sync_policy")
        if bad_channel_sync_policy not in {"per_run", "subject_union"}:
            raise ValueError(
                "pyprep.bad_channel_sync_policy must be explicitly set to "
                "'per_run' or 'subject_union'."
            )
        return str(bad_channel_sync_policy)

    def _get_ica_fitting_steps(self) -> str:
        """Get MNE-BIDS pipeline steps for ICA fitting."""
        return ",".join(
            [
                "init",
                "preprocessing/_01_data_quality",
                "preprocessing/_04_frequency_filter",
                "preprocessing/_05_regress_artifact",
                "preprocessing/_06a1_fit_ica",
                "preprocessing/_06a2_find_ica_artifacts",
            ]
        )

    def _get_ica_preparation_steps(self) -> str:
        """Get MNE-BIDS steps that produce filtered raw files for ICA."""
        return ",".join(
            [
                "init",
                "preprocessing/_01_data_quality",
                "preprocessing/_04_frequency_filter",
                "preprocessing/_05_regress_artifact",
            ]
        )

    def _get_ica_decomposition_steps(self) -> str:
        """Get MNE-BIDS steps that concatenate epochs and fit ICA."""
        return "preprocessing/_06a1_fit_ica,preprocessing/_06a2_find_ica_artifacts"

    def _run_ica_fitting(
        self,
        subjects: List[str],
        task: Optional[str],
        task_is_rest: Optional[bool] = None,
    ) -> None:
        """Run ICA fitting via MNE-BIDS pipeline."""
        self._run_mne_bids_pipeline(
            self._get_ica_preparation_steps(),
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
        )
        self._harmonize_filtered_raw_bads_for_mne_concat(subjects, task)
        self._run_mne_bids_pipeline(
            self._get_ica_decomposition_steps(),
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
        )

        if bool(self.config.get("ica.band_specific_report.enabled", False)):
            self._run_band_specific_ica_report(subjects=subjects, task=task)

        self.logger.info("ICA fitting complete")

    def _run_band_specific_ica_report(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Append exploratory frequency-specific ICA diagnostics to subject reports."""
        from eeg_pipeline.preprocessing.band_ica_report import (
            BandIcaReportSettings,
            generate_band_ica_report,
        )

        settings = BandIcaReportSettings.from_mapping(
            self.config.get("ica.band_specific_report", {})
        )
        random_state = int(self.config.get("project.random_state", 42))
        for subject in self._resolve_bad_harmonization_subjects(subjects):
            for epochs_path, report_path, output_prefix in self._find_band_ica_report_inputs(
                subject
            ):
                generate_band_ica_report(
                    epochs_path=epochs_path,
                    report_path=report_path,
                    output_dir=epochs_path.parent / "band-specific-ica",
                    output_prefix=output_prefix,
                    random_state=random_state,
                    settings=settings,
                )
            self.logger.info(
                "Added exploratory band-specific ICA diagnostics for sub-%s, task=%s",
                subject,
                task,
            )

    def _find_band_ica_report_inputs(
        self,
        subject: str,
    ) -> List[tuple[Path, Path, str]]:
        """Find session-aware ICA epochs and their corresponding MNE reports."""
        subject_dir = self.deriv_root / "preprocessed" / "eeg" / f"sub-{subject}"
        epochs_paths = sorted(
            path
            for path in subject_dir.rglob(f"sub-{subject}*_proc-icafit_epo.fif")
            if path.is_file() and not path.name.startswith("._")
        )
        if not epochs_paths:
            raise FileNotFoundError(
                f"No ICA-fitting epochs found for band-specific report: sub-{subject}"
            )

        inputs = []
        suffix = "_proc-icafit_epo.fif"
        for epochs_path in epochs_paths:
            output_prefix = epochs_path.name.removesuffix(suffix)
            report_path = epochs_path.with_name(f"{output_prefix}_report.h5")
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Band-specific ICA report input does not exist: {report_path}"
                )
            inputs.append((epochs_path, report_path, output_prefix))
        return inputs

    def _harmonize_filtered_raw_bads_for_mne_concat(
        self,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Set filtered run bads to the subject union before MNE epoch concatenation."""
        import mne

        qc_records = []
        for subject in self._resolve_bad_harmonization_subjects(subjects):
            filtered_paths = self._find_filtered_raw_run_files(subject, task)
            if len(filtered_paths) < 2:
                continue

            raw_by_path = {}
            bads_by_path = {}
            ch_names_by_path = {}
            eeg_ch_names_by_path = {}
            for path in filtered_paths:
                raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
                raw_by_path[path] = raw
                bads_by_path[path] = sorted(set(raw.info.get("bads", [])))
                ch_names_by_path[path] = set(raw.ch_names)
                eeg_ch_names_by_path[path] = set(self._get_eeg_channel_names(raw))

            subject_bad_union = sorted(
                {channel for bads in bads_by_path.values() for channel in bads}
            )
            all_channel_names = sorted(
                {channel for ch_names in eeg_ch_names_by_path.values() for channel in ch_names}
            )
            qc_records.append(
                self._build_bad_channel_union_qc_record(
                    subject=subject,
                    task=task,
                    paths=filtered_paths,
                    channel_names=all_channel_names,
                    subject_bad_union=subject_bad_union,
                )
            )
            if all(bads == subject_bad_union for bads in bads_by_path.values()):
                continue

            bad_channel_sync_policy = self._resolve_bad_channel_sync_policy()
            if bad_channel_sync_policy != "subject_union":
                raise ValueError(
                    "MNE-BIDS shared ICA/epoch concatenation requires matching "
                    "filtered raw bad-channel metadata across runs. Found mismatched "
                    f"bad-channel sets for sub-{subject}: {bads_by_path}. Set "
                    "pyprep.bad_channel_sync_policy='subject_union' for this workflow "
                    "or fit/process runs separately."
                )

            for path, ch_names in ch_names_by_path.items():
                missing_channels = sorted(set(subject_bad_union) - ch_names)
                if missing_channels:
                    raise ValueError(
                        f"Cannot harmonize bad channels for {path}: "
                        f"channels missing from this run: {missing_channels}"
                    )

            for path, raw in raw_by_path.items():
                raw.info["bads"] = subject_bad_union
                self._save_raw_with_updated_bads(raw, path)

            self.logger.info(
                "Harmonized filtered raw bad channels for sub-%s before "
                "MNE-BIDS cross-run concatenation: %s",
                subject,
                subject_bad_union,
            )

        self._write_bad_channel_union_qc(task=task, records=qc_records)

    def _build_bad_channel_union_qc_record(
        self,
        subject: str,
        task: Optional[str],
        paths: List[Path],
        channel_names: List[str],
        subject_bad_union: List[str],
    ) -> Dict[str, Any]:
        """Build a subject-level bad-channel union QC row."""
        bad_channel_count = len(subject_bad_union)
        channel_count = len(channel_names)
        bad_channel_fraction = bad_channel_count / channel_count if channel_count else 0.0
        return {
            "subject": subject,
            "task": task or "",
            "bad_channel_sync_policy": self._resolve_bad_channel_sync_policy(),
            "n_runs": len(paths),
            "n_channels": channel_count,
            "n_union_bad_channels": bad_channel_count,
            "bad_channel_fraction": f"{bad_channel_fraction:.6f}",
            "union_bad_channels": ",".join(subject_bad_union),
            "roi_coverage": json.dumps(
                self._summarize_bad_channel_roi_coverage(
                    channel_names=channel_names,
                    subject_bad_union=subject_bad_union,
                ),
                sort_keys=True,
            ),
        }

    def _get_eeg_channel_names(self, raw: Any) -> List[str]:
        """Return EEG channel names from an MNE raw object."""
        channel_types = raw.get_channel_types(picks=raw.ch_names)
        if len(channel_types) != len(raw.ch_names):
            raise ValueError(
                "raw.get_channel_types(picks=raw.ch_names) returned "
                f"{len(channel_types)} entries for {len(raw.ch_names)} channels."
            )
        return [
            channel_name
            for channel_name, channel_type in zip(raw.ch_names, channel_types)
            if channel_type == "eeg"
        ]

    def _summarize_bad_channel_roi_coverage(
        self,
        channel_names: List[str],
        subject_bad_union: List[str],
    ) -> List[Dict[str, Any]]:
        """Return per-ROI remaining-channel counts after subject-union bad exclusion."""
        roi_definitions = self.config.get("rois", {})
        if not isinstance(roi_definitions, dict) or not roi_definitions:
            return []

        from eeg_pipeline.utils.analysis.channels import build_roi_map

        bad_channels = set(subject_bad_union)
        roi_map = build_roi_map(channel_names, roi_definitions)
        roi_records = []
        for roi_name in sorted(roi_definitions):
            roi_channels = [channel_names[idx] for idx in roi_map.get(roi_name, [])]
            bad_roi_channels = sorted(set(roi_channels) & bad_channels)
            remaining_count = len(roi_channels) - len(bad_roi_channels)
            roi_records.append(
                {
                    "roi": roi_name,
                    "n_channels": len(roi_channels),
                    "n_bad_channels": len(bad_roi_channels),
                    "n_remaining_channels": remaining_count,
                    "passes_min_two_channels": remaining_count >= 2,
                    "bad_channels": bad_roi_channels,
                }
            )
        return roi_records

    def _write_bad_channel_union_qc(
        self,
        task: Optional[str],
        records: List[Dict[str, Any]],
    ) -> None:
        """Write subject-union bad-channel QC rows for shared MNE-BIDS stages."""
        if not records:
            return

        task_label = str(task or "all").replace(os.sep, "_")
        qc_path = (
            self.deriv_root / "preprocessed" / "eeg" / f"bad_channel_union_qc_task-{task_label}.tsv"
        )
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "subject",
            "task",
            "bad_channel_sync_policy",
            "n_runs",
            "n_channels",
            "n_union_bad_channels",
            "bad_channel_fraction",
            "union_bad_channels",
            "roi_coverage",
        ]
        with qc_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)

    def _resolve_bad_harmonization_subjects(self, subjects: List[str]) -> List[str]:
        """Resolve explicit or discovered subjects for bad-channel harmonization."""
        if subjects == ["all"]:
            root = self.deriv_root / "preprocessed" / "eeg"
            return sorted(
                path.name.removeprefix("sub-") for path in root.glob("sub-*") if path.is_dir()
            )
        return list(subjects)

    def _find_filtered_raw_run_files(
        self,
        subject: str,
        task: Optional[str],
    ) -> List[Path]:
        """Find non-split filtered raw run files for one subject."""
        subject_root = self.deriv_root / "preprocessed" / "eeg" / f"sub-{subject}"
        subject_prefix = f"sub-{subject}_"
        task_selector = f"_task-{task}_" if task is not None else "_task-"
        all_paths = sorted(
            path
            for path in subject_root.rglob("*_run-*_proc-filt_raw.fif")
            if path.name.startswith(subject_prefix) and task_selector in path.name
        )
        split_paths = [path for path in all_paths if "_split-" in path.name]
        if split_paths:
            raise RuntimeError(
                "Cannot harmonize split filtered raw files for MNE-BIDS "
                f"concatenation: {split_paths}"
            )
        return all_paths

    def _save_raw_with_updated_bads(self, raw: Any, path: Path) -> None:
        """Rewrite a raw FIF after changing only its bad-channel metadata."""
        if not path.name.endswith("_raw.fif"):
            raise ValueError(f"Expected raw FIF path ending in '_raw.fif', got {path}")
        tmp_path = path.with_name(path.name.replace("_raw.fif", "_badsync_raw.fif"))
        tmp_path.unlink(missing_ok=True)
        try:
            raw.save(tmp_path, overwrite=True, split_naming="bids")
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_epoch_creation(
        self,
        subjects: List[str],
        task: Optional[str],
        task_is_rest: bool,
    ) -> None:
        """Create epochs and apply ICA via MNE-BIDS pipeline."""
        steps = "preprocessing/_07_make_epochs,preprocessing/_08a_apply_ica,preprocessing/_09_ptp_reject"
        self._harmonize_filtered_raw_bads_for_mne_concat(subjects, task)

        self._run_mne_bids_pipeline(
            steps,
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
        )

        if task_is_rest:
            self.logger.info("Skipping clean events export for resting-state preprocessing")
        elif bool(self.config.get("preprocessing.write_clean_events", True)):
            self._write_clean_events_tsv(subjects=subjects, task=task)

        band_report_config = self.config.get("ica.band_specific_report", {})
        if bool(band_report_config.get("enabled", False)) and band_report_config.get("comparisons"):
            self._append_band_ica_condition_tfrs(subjects=subjects, task=task)

        self.logger.info("Epoch creation complete")

    def _append_band_ica_condition_tfrs(
        self,
        *,
        subjects: List[str],
        task: str,
    ) -> None:
        """Append clean-trial metadata comparisons to band-specific ICA reports."""
        from eeg_pipeline.preprocessing.band_ica_report import (
            BandIcaReportSettings,
            append_condition_tfr_report,
        )

        settings = BandIcaReportSettings.from_mapping(
            self.config.get("ica.band_specific_report", {})
        )
        for subject in self._resolve_bad_harmonization_subjects(subjects):
            subject_dir = self.deriv_root / "preprocessed" / "eeg" / f"sub-{subject}"
            clean_epochs_paths = sorted(
                path
                for path in subject_dir.rglob(f"sub-{subject}*_task-{task}_proc-clean_epo.fif")
                if not path.name.startswith("._")
            )
            if not clean_epochs_paths:
                raise FileNotFoundError(
                    f"No clean epochs found for band-specific comparisons: sub-{subject}"
                )
            for clean_epochs_path in clean_epochs_paths:
                entity_prefix = clean_epochs_path.name.removesuffix(
                    f"_task-{task}_proc-clean_epo.fif"
                )
                pre_ica_epochs_path = clean_epochs_path.with_name(
                    f"{entity_prefix}_task-{task}_epo.fif"
                )
                clean_events_path = clean_epochs_path.with_name(
                    f"{entity_prefix}_task-{task}_proc-clean_events.tsv"
                )
                report_path = clean_epochs_path.with_name(f"{entity_prefix}_report.h5")
                for required_path in (
                    pre_ica_epochs_path,
                    clean_events_path,
                    report_path,
                ):
                    if not required_path.is_file():
                        raise FileNotFoundError(
                            f"Band-specific comparison input does not exist: {required_path}"
                        )
                append_condition_tfr_report(
                    pre_ica_epochs_path=pre_ica_epochs_path,
                    clean_epochs_path=clean_epochs_path,
                    clean_events_path=clean_events_path,
                    report_path=report_path,
                    output_dir=clean_epochs_path.parent / "band-specific-ica",
                    output_prefix=entity_prefix,
                    settings=settings,
                )

    def _resolve_epoch_conditions(self, task: Optional[str] = None) -> list[str] | None:
        conditions = self.config.get("epochs.conditions")
        if conditions:
            return list(conditions)
        detected = self._detect_conditions_from_bids(task)
        return list(detected) if detected else None

    def _write_clean_events_tsv(self, *, subjects: List[str], task: str) -> None:
        from eeg_pipeline.infra.paths import find_clean_epochs_path
        from eeg_pipeline.utils.data.preprocessing import write_clean_events_tsv_for_epochs

        conditions = self._resolve_epoch_conditions(task)
        overwrite = bool(self.config.get("preprocessing.clean_events_overwrite", True))
        strict = bool(self.config.get("preprocessing.clean_events_strict", True))

        for subj in subjects:
            epochs_path = find_clean_epochs_path(
                subj,
                task,
                deriv_root=self.deriv_root,
                config=self.config,
            )
            if epochs_path is None or not epochs_path.exists():
                msg = (
                    f"Clean epochs not found; cannot write clean events for sub-{subj}, task-{task}"
                )
                if strict:
                    raise FileNotFoundError(msg)
                self.logger.warning(msg)
                continue

            try:
                write_clean_events_tsv_for_epochs(
                    subject=subj,
                    task=task,
                    bids_root=self.bids_root,
                    epochs_path=epochs_path,
                    config=self.config,
                    conditions=conditions,
                    overwrite=overwrite,
                    _logger=self.logger,
                )
            except Exception as exc:
                msg = f"Failed writing clean events for sub-{subj}, task-{task}: {exc}"
                if strict:
                    raise RuntimeError(msg) from exc
                self.logger.warning(msg)

    def _collect_stats(self, task: Optional[str]) -> None:
        """Collect preprocessing statistics."""
        from eeg_pipeline.preprocessing.pipeline.stats import collect_preprocessing_stats

        self.logger.info("Collecting preprocessing statistics")

        collect_preprocessing_stats(
            bids_path=str(self.bids_root),
            pipeline_path=str(self.deriv_root / "preprocessed" / "eeg"),
            task=task,
        )

        self.logger.info("Statistics collection complete")

    def _run_scanner_harmonic_qc(
        self,
        *,
        subjects: List[str],
        task: str,
    ) -> Dict[str, str]:
        """Write the input-versus-final cohort scanner-harmonic comb."""
        from eeg_pipeline.preprocessing.pipeline.scanner_harmonic_qc import (
            run_scanner_harmonic_qc,
            scanner_comb_parameters_from_config,
        )

        parameters = scanner_comb_parameters_from_config(self.config)
        outputs = run_scanner_harmonic_qc(
            subjects=subjects,
            task=task,
            bids_root=self.bids_root,
            deriv_root=self.deriv_root,
            input_extension=self.config.get("pyprep.file_extension"),
            parameters=parameters,
        )
        return {
            "scanner_harmonic_comb_png": str(outputs.png_path),
            "scanner_harmonic_comb_tsv": str(outputs.tsv_path),
        }

    def _run_mne_bids_pipeline(
        self,
        steps: str,
        subjects: List[str] = None,
        task: Optional[str] = None,
        task_is_rest: Optional[bool] = None,
    ) -> None:
        """Run MNE-BIDS pipeline with a generated config file.

        mne_bids_pipeline requires settings in a Python config file,
        not CLI arguments. This generates a temporary config and passes it
        via --config.

        Args:
            steps: MNE-BIDS pipeline steps to run
            subjects: List of subject IDs to process (without 'sub-' prefix)
            task: Task name to constrain MNE-BIDS-Pipeline subject selection
            task_is_rest: Override config to enable/disable resting-state preprocessing
        """
        import tempfile

        config_content = self._generate_mne_bids_config(
            steps,
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_mne_bids_config.py",
            delete=False,
        ) as f:
            f.write(config_content)
            config_path = f.name

        try:
            args = [
                f"--config={config_path}",
                f"--steps={steps}",
            ]

            self.logger.info("Running MNE-BIDS pipeline: %s", steps)

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            invoke_script = (
                "import sys; "
                "sys.argv = ['mne_bids_pipeline'] + sys.argv[1:]; "
                "from mne_bids_pipeline._main import main; "
                "main()"
            )
            result = subprocess.run(
                [sys.executable, "-c", invoke_script] + args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            if result.stdout:
                self.logger.debug("MNE-BIDS stdout: %s", result.stdout)
            if result.stderr:
                self.logger.warning("MNE-BIDS stderr: %s", result.stderr)

            if result.returncode != 0:
                output_sections = []
                if result.stdout:
                    output_sections.append(f"stdout:\n{result.stdout}")
                if result.stderr:
                    output_sections.append(f"stderr:\n{result.stderr}")
                error_msg = "\n\n".join(output_sections) or "Unknown error"
                raise RuntimeError(f"MNE-BIDS pipeline failed: {error_msg}")
        finally:
            Path(config_path).unlink(missing_ok=True)

    def _get_rest_epoch_parameters(self) -> tuple[float, float]:
        """Return validated fixed-length epoch settings for resting-state preprocessing."""
        rest_epochs_duration = self.config.get("preprocessing.rest_epochs_duration")
        rest_epochs_overlap = self.config.get("preprocessing.rest_epochs_overlap", 0.0)

        if rest_epochs_duration is None:
            raise ValueError(
                "Resting-state preprocessing requires preprocessing.rest_epochs_duration."
            )

        duration = float(rest_epochs_duration)
        overlap = float(rest_epochs_overlap)
        if duration <= 0:
            raise ValueError("preprocessing.rest_epochs_duration must be greater than 0.")
        if overlap < 0:
            raise ValueError(
                "preprocessing.rest_epochs_overlap must be greater than or equal to 0."
            )
        if overlap > 0:
            raise ValueError(
                "Resting-state preprocessing does not support preprocessing.rest_epochs_overlap > 0, "
                "because overlapping epochs would be treated as independent trial rows. "
                "Set preprocessing.rest_epochs_overlap = 0."
            )
        if overlap >= duration:
            raise ValueError(
                "preprocessing.rest_epochs_overlap must be smaller than preprocessing.rest_epochs_duration."
            )

        return duration, overlap

    def _append_rest_epoch_config(self, lines: list[str]) -> None:
        """Append resting-state epoch configuration for MNE-BIDS-Pipeline."""
        duration, overlap = self._get_rest_epoch_parameters()
        lines.append("conditions = None")
        lines.append("epochs_tmin = 0.0")
        lines.append("baseline = None")
        lines.append(f"rest_epochs_duration = {duration}")
        lines.append(f"rest_epochs_overlap = {overlap}")

    def _append_task_epoch_config(
        self,
        lines: list[str],
        task: Optional[str],
    ) -> None:
        """Append event-locked epoch configuration for MNE-BIDS-Pipeline."""
        conditions = self.config.get("epochs.conditions")
        if not conditions:
            conditions = self._detect_conditions_from_bids(task)
        if not conditions:
            raise ValueError(
                "Non-resting-state preprocessing requires epochs.conditions or "
                "detectable BIDS event conditions."
            )
        lines.append(f"conditions = {list(conditions)}")

        epochs_tmin = self.config.get("epochs.tmin", -3.0)
        if epochs_tmin is not None:
            lines.append(f"epochs_tmin = {epochs_tmin}")

        epochs_tmax = self.config.get("epochs.tmax", 12.0)
        if epochs_tmax is not None:
            lines.append(f"epochs_tmax = {epochs_tmax}")

        baseline = self.config.get("epochs.baseline")
        if baseline is not None:
            if isinstance(baseline, list):
                baseline = tuple(baseline)
            lines.append(f"baseline = {baseline}")
        else:
            lines.append("baseline = None")

    def _append_rejection_config(self, lines: list[str]) -> None:
        """Append epoch rejection configuration for MNE-BIDS-Pipeline."""
        reject = self.config.get("epochs.reject")
        reject_method = self.config.get("epochs.reject_method")
        if reject is None and reject_method:
            rm = str(reject_method).strip().lower()
            if rm == "none":
                reject = None
            elif rm in {"autoreject_local", "autoreject_global"}:
                reject = rm

        if reject is not None:
            if isinstance(reject, str):
                lines.append(f'reject = "{reject}"')
            else:
                lines.append(f"reject = {reject}")

        reject_tmin = self.config.get("epochs.reject_tmin")
        if reject_tmin is not None:
            lines.append(f"reject_tmin = {float(reject_tmin)}")

        reject_tmax = self.config.get("epochs.reject_tmax")
        if reject_tmax is not None:
            lines.append(f"reject_tmax = {float(reject_tmax)}")

        ar_n_interp = self.config.get("epochs.autoreject_n_interpolate")
        if ar_n_interp is not None:
            lines.append(f"autoreject_n_interpolate = {ar_n_interp}")

    def _generate_mne_bids_config(
        self,
        steps: str,
        subjects: List[str] = None,
        task: Optional[str] = None,
        task_is_rest: Optional[bool] = None,
    ) -> str:
        """Generate Python config file content for mne_bids_pipeline.

        Args:
            steps: MNE-BIDS pipeline steps to run
            subjects: List of subject IDs to process (without 'sub-' prefix)
        """
        resolved_task_is_rest = self._resolve_task_is_rest(task_is_rest)
        resolved_task = task
        if resolved_task is None and not resolved_task_is_rest:
            resolved_task = self.config.get("project.task")
        lines = [
            '"""Auto-generated MNE-BIDS pipeline config."""',
            "",
            f'bids_root = "{self.bids_root}"',
            f'deriv_root = "{self.deriv_root / "preprocessed" / "eeg"}"',
            "",
        ]

        # Subject filter (critical to avoid processing all subjects)
        if subjects:
            lines.append(f"subjects = {subjects}")
            lines.append("")

        if resolved_task:
            lines.append(f'task = "{resolved_task}"')
            lines.append("")

        # Channel types
        ch_types = self.config.get("eeg.ch_types", "eeg")
        if ch_types:
            if isinstance(ch_types, (list, tuple)):
                lines.append(f"ch_types = {list(ch_types)}")
            else:
                lines.append(f'ch_types = ["{ch_types}"]')

        # EEG reference
        eeg_reference = self.config.get("eeg.reference", "average")
        if eeg_reference:
            lines.append(f'eeg_reference = "{eeg_reference}"')

        # EOG channels
        eog_channels = self.config.get("eeg.eog_channels")
        if eog_channels:
            if isinstance(eog_channels, list):
                lines.append(f"eog_channels = {eog_channels}")
            elif isinstance(eog_channels, str):
                # Handle comma-separated string
                eog_list = [ch.strip() for ch in eog_channels.split(",") if ch.strip()]
                if eog_list:
                    lines.append(f"eog_channels = {eog_list}")
            else:
                lines.append(f'eog_channels = ["{eog_channels}"]')

        # NOTE: mne-bids-pipeline does not accept an `ecg_channels` config variable.
        # ECG channel typing is handled via BIDS channels.tsv (type=ECG) and MNE.

        # Random state
        random_state = self.config.get("preprocessing.random_state")
        if random_state is not None:
            lines.append(f"random_state = {random_state}")

        # Task is rest
        lines.append(f"task_is_rest = {resolved_task_is_rest}")

        lines.append("")
        lines.append("# Filtering")

        # Filtering configs
        l_freq = self.config.get("preprocessing.l_freq", 0.1)
        if l_freq is not None:
            lines.append(f"l_freq = {l_freq}")

        h_freq = self.config.get("preprocessing.h_freq", 100)
        if h_freq is not None:
            lines.append(f"h_freq = {h_freq}")

        notch_freq = self.config.get("preprocessing.notch_freq")
        if notch_freq is not None:
            lines.append(f"notch_freq = {notch_freq}")

        # Resampling
        resample_sfreq = self.config.get("preprocessing.resample_freq")
        if resample_sfreq is not None:
            lines.append(f"raw_resample_sfreq = {resample_sfreq}")

        # Find breaks
        find_breaks = self.config.get("preprocessing.find_breaks", False)
        lines.append(f"find_breaks = {find_breaks}")

        lines.append("")
        lines.append("# ICA")

        # Spatial filter
        spatial_filter = self.config.get("ica.spatial_filter", "ica")
        if spatial_filter:
            lines.append(f'spatial_filter = "{spatial_filter}"')

        # ICA algorithm
        ica_algorithm = self.config.get("ica.method") or self.config.get(
            "ica.algorithm", "extended_infomax"
        )
        if ica_algorithm:
            lines.append(f'ica_algorithm = "{ica_algorithm}"')

        # ICA n_components
        ica_n_components = self.config.get("ica.n_components")
        lines.append(f"ica_n_components = {ica_n_components!r}")

        # ICA l_freq
        ica_l_freq = self.config.get("ica.l_freq", 1.0)
        if ica_l_freq is not None:
            lines.append(f"ica_l_freq = {ica_l_freq}")

        ica_h_freq = self.config.get("ica.h_freq")
        if ica_h_freq is not None:
            lines.append(f"ica_h_freq = {ica_h_freq}")

        # ICA reject
        ica_reject = self.config.get("ica.reject")
        if ica_reject is not None:
            lines.append(f"ica_reject = {ica_reject!r}")

        use_icalabel = bool(self.config.get("ica.use_icalabel"))
        use_ecg_detection = bool(self.config.get("ica.use_ecg_detection"))
        use_eog_detection = bool(self.config.get("ica.use_eog_detection"))
        lines.append(f"ica_use_icalabel = {use_icalabel}")
        lines.append(f"ica_use_ecg_detection = {use_ecg_detection}")
        lines.append(f"ica_use_eog_detection = {use_eog_detection}")

        if use_icalabel:
            labels_to_keep = tuple(self.config.get("ica.labels_to_keep", []))
            if not labels_to_keep:
                raise ValueError("ica.labels_to_keep must contain at least one ICLabel class.")
            lines.append(f"ica_icalabel_include = {labels_to_keep!r}")

            probability_threshold = float(self.config.get("ica.probability_threshold"))
            artifact_labels = (
                "muscle artifact",
                "eye blink",
                "heart beat",
                "line noise",
                "channel noise",
            )
            exclusion_thresholds = {
                label: probability_threshold
                for label in artifact_labels
                if label not in labels_to_keep
            }
            lines.append(f"ica_exclusion_thresholds = {exclusion_thresholds!r}")
        process_raw_clean = bool(self.config.get("ica.process_raw_clean"))
        lines.append(f"process_raw_clean = {process_raw_clean}")

        lines.append("")
        lines.append("# Epochs")

        if resolved_task_is_rest:
            self._append_rest_epoch_config(lines)
        else:
            self._append_task_epoch_config(lines, resolved_task)

        self._append_rejection_config(lines)

        lines.append("")

        return "\n".join(lines)

    def _detect_conditions_from_bids(self, task: Optional[str] = None) -> list | None:
        """Detect unique condition names from BIDS events files.

        Reads a configured condition column from first available events TSV and returns
        unique values as a list suitable for mne_bids_pipeline conditions.

        Returns:
            List of unique condition values, or None if detection fails.
        """
        if task is None:
            events_files = sorted(
                path
                for path in self.bids_root.rglob("*_events.tsv")
                if _is_events_tsv(path) and "eeg" in path.parts
            )
        else:
            candidate_paths = sorted(
                path
                for path in self.bids_root.rglob(f"*_task-{task}*_events.tsv")
                if _is_events_tsv(path) and "eeg" in path.parts
            )
            events_files = [
                path
                for path in candidate_paths
                if f"_task-{task}_" in path.name or path.name.endswith(f"_task-{task}_events.tsv")
            ]
        config_obj = getattr(self, "config", None)

        if not events_files:
            if task is None:
                self.logger.debug("No EEG events files found in %s", self.bids_root)
            else:
                self.logger.debug(
                    "No EEG events files found in %s for task '%s'",
                    self.bids_root,
                    task,
                )
            return None

        candidates = list(get_condition_column_candidates(config_obj))
        if not candidates:
            candidates = ["condition", "trial_type"]

        conditions = set()
        detected_columns = set()
        for events_file in events_files:
            with open(events_file, "r", encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
                header_lookup = {str(name).strip().lower(): idx for idx, name in enumerate(header)}
                condition_column = None
                condition_idx = None
                for candidate in candidates:
                    idx = header_lookup.get(candidate.lower())
                    if idx is not None:
                        condition_column = candidate
                        condition_idx = idx
                        break

                if condition_idx is None:
                    continue

                detected_columns.add(str(condition_column))
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) <= condition_idx:
                        continue
                    condition_value = parts[condition_idx].strip()
                    if condition_value and condition_value != "n/a":
                        conditions.add(condition_value)

        if not detected_columns:
            self.logger.debug(
                "No configured condition column found in any events file header (candidates=%s)",
                candidates,
            )
            return None

        if not conditions:
            return None

        configured_prefixes = None
        if config_obj is not None and hasattr(config_obj, "get"):
            configured_prefixes = config_obj.get("preprocessing.condition_preferred_prefixes")
        if isinstance(configured_prefixes, (list, tuple)):
            preferred_prefixes = tuple(
                str(p).strip() for p in configured_prefixes if str(p).strip()
            )
        elif isinstance(configured_prefixes, str) and configured_prefixes.strip():
            preferred_prefixes = tuple(
                part.strip() for part in configured_prefixes.split(",") if part.strip()
            )
        else:
            preferred_prefixes = ()
        excluded_prefixes = (
            "Volume",
            "Pulse",
            "SyncStatus",
            "New Segment",
            "Bad",
            "EDGE",
            "Response",
        )

        preferred = sorted(
            t for t in conditions if any(t.startswith(p) for p in preferred_prefixes)
        )
        if preferred:
            self.logger.info(
                "Auto-detected task conditions from BIDS columns %s: %s",
                sorted(detected_columns),
                preferred,
            )
            return preferred

        filtered = sorted(
            t for t in conditions if not any(t.startswith(p) for p in excluded_prefixes)
        )
        if not filtered:
            return None

        if len(filtered) > 50:
            self.logger.warning(
                "Auto-detected %d conditions from BIDS (too many). "
                "Set epochs.conditions explicitly in config to avoid ambiguity.",
                len(filtered),
            )
            return None

        self.logger.info("Auto-detected filtered conditions from BIDS: %s", filtered)
        return filtered


__all__ = ["PreprocessingPipeline"]
