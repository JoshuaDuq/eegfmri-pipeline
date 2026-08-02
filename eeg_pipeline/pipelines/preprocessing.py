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
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _mne_annotation_event_pattern() -> "re.Pattern[str]":
    """Return the pattern MNE uses to decide which annotations become events.

    Read from ``mne.events_from_annotations`` rather than copied, so that the conditions
    this pipeline epochs on and the events MNE actually creates cannot drift apart. The
    default is a negative lookahead on ``BAD``/``EDGE``, applied to the whole description
    and anchored at its start.

    Raises rather than guessing. A silent fallback here would restore exactly the
    duplicate-rule problem this exists to remove, and it would do so invisibly.

    Imported inside the function to match this module's convention: binding MNE at module
    level would defeat the dependency stubbing in ``tests/pipelines``.
    """
    import mne

    parameter = inspect.signature(mne.events_from_annotations).parameters.get("regexp")
    if parameter is None or not isinstance(parameter.default, str) or not parameter.default:
        raise RuntimeError(
            "Cannot read the default 'regexp' of mne.events_from_annotations, so the "
            "conditions this pipeline selects cannot be checked against the events MNE "
            "will create. MNE's signature has changed; update "
            "_mne_annotation_event_pattern to match it."
        )
    return re.compile(parameter.default)


def _is_events_tsv(path: Path) -> bool:
    return path.is_file() and path.name.endswith("_events.tsv") and not path.name.startswith("._")


def _preservation_measurements(*, reliability, alpha) -> dict:
    """Headline numbers from the preservation section, for the report's landing panel.

    Which of the two exists depends on the paradigm rather than on whether anything went
    wrong: rest has no evoked response to split in half, and a montage with no posterior
    sensors has no alpha to measure. Each is recorded only when it was measured, so the
    panel carries the evidence this recording could actually supply.
    """
    measurements: dict = {}
    if reliability is not None:
        measurements["split_half_r"] = float(reliability.corrected_correlation)
        # Reliability grows with test length, so the correlation above is only comparable
        # with another participant's once both are stepped to a common trial count. The
        # count is recorded here so a cohort can do that without reopening the epochs.
        measurements["split_half_n_trials"] = int(reliability.n_trials)
        window_start, window_end = reliability.response_window_s
        measurements["split_half_window_start_s"] = float(window_start)
        measurements["split_half_window_end_s"] = float(window_end)
    if alpha is not None:
        measurements["alpha_prominence_db"] = float(alpha.prominence_db)
    return measurements


def _recorded_versions(record: dict) -> dict:
    """The package versions that built this report, latest stage wins.

    Collapsed the same way the measurements are, and for the same reason: the sidecar
    describes the document as it stands, and a stage rerun under a newer MNE built the
    section that is in the file now.
    """
    versions: dict = {}
    stages = sorted(
        (entry for entry in record.get("stages", ()) if isinstance(entry, dict)),
        key=lambda entry: str(entry.get("written_at", "")),
    )
    for entry in stages:
        recorded = entry.get("versions") or {}
        if isinstance(recorded, dict):
            versions.update({str(k): str(v) for k, v in recorded.items()})
    return versions


def _subject_of_report(report_path: Path) -> Optional[str]:
    """Recover the participant label a BIDS derivative filename encodes."""
    for part in report_path.name.split("_"):
        if part.startswith("sub-"):
            return part[len("sub-") :]
    return None


def _review_stage_measurements(*, coverage, evidence) -> dict:
    """Headline numbers the review stage measures, for the report's landing panel.

    Only what this stage owns. The panel is assembled from the build record so that every
    number on it is the one its own section measured, which means each stage writes down
    its own and none of them recompute anyone else's.

    Absent inputs produce absent keys rather than zeros. A dataset recorded outside a
    scanner has no marker agreement to report, and a row reading "0%" for it would state
    a total disagreement between two detectors where there was only ever one.
    """
    measurements: dict = {}
    if coverage is not None:
        measurements["n_channels"] = int(coverage.n_channels)
        measurements["n_bad_channels"] = len(coverage.bad_channels)
        measurements["n_runs"] = int(coverage.n_runs)
    if evidence is not None:
        if getattr(evidence, "spectra", None):
            measurements["n_runs"] = len(evidence.spectra)
        # The lowest agreement across runs, because the panel exists to surface the run
        # that stands apart rather than an average that hides it. Runs whose fraction is
        # undefined carry no number to be lowest.
        fractions = [
            agreement.matched_fraction
            for agreement in getattr(evidence, "marker_agreements", ())
            if agreement.matched_fraction is not None
        ]
        if fractions:
            measurements["worst_marker_agreement"] = float(min(fractions))
    return measurements


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
        n_jobs = self._validate_n_jobs(kwargs.get("n_jobs", 1))
        progress = ensure_progress_reporter(kwargs.get("progress"))

        return (
            resolved_task,
            mode,
            use_pyprep,
            task_is_rest,
            n_jobs,
            progress,
        )

    @staticmethod
    def _validate_n_jobs(n_jobs: Any) -> int:
        """Return the worker count, rejecting the one value neither backend accepts.

        Negatives are the "leave this many cores free" selector both joblib and
        MNE-BIDS-Pipeline understand, so they pass through. Zero is not a smaller version
        of that: joblib raises on it deep inside a worker pool, and MNE-BIDS-Pipeline
        resolves it to zero workers. Either way the failure surfaces from a subprocess,
        after the run has already opened recordings.
        """
        try:
            resolved = int(n_jobs)
        except (TypeError, ValueError) as error:
            raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}.") from error
        if resolved == 0:
            raise ValueError(
                "n_jobs must be a non-zero integer: a positive worker count, or a "
                "negative value to leave that many cores free (-1 uses all cores)."
            )
        return resolved

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
        self._check_config_coherence()

        progress.subject_start(f"sub-{subject}")

        try:
            steps = self._get_steps_for_run(mode, task_is_rest, [subject], resolved_task)

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
            One status dictionary per subject, all carrying the same status.

            The outcome is genuinely batch-level rather than per-subject: MNE-BIDS-Pipeline
            processes the whole subject list in one invocation and a failure anywhere
            raises for the run, so this either returns success for every subject or raises.
            Do not read an individual entry as evidence that that subject specifically
            succeeded; the per-subject record of what happened is the run metadata and the
            subject reports.
        """
        resolved_task, mode, use_pyprep, task_is_rest, n_jobs, progress = (
            self._extract_preprocessing_params(task, kwargs)
        )
        self._refresh_processing_roots_if_initialized(task_is_rest)
        self._check_config_coherence()
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

            steps = self._get_steps_for_run(mode, task_is_rest, subjects, resolved_task)

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

    def _get_steps_for_run(
        self,
        mode: str,
        task_is_rest: bool,
        subjects: Optional[List[str]] = None,
        task: Optional[str] = None,
    ) -> List[str]:
        """Append cohort QC only when the outputs it measures are produced.

        Keyed off the steps themselves rather than the mode name, so that ``full`` — which
        runs both ICA fitting and epoching — gets the same QC as running ``ica`` and
        ``epochs`` separately, instead of silently producing derivatives with no cardiac
        QC attached.
        """
        steps = self._get_steps_for_mode(mode)

        # Two independent facts about a dataset, previously decided by one switch.
        #
        # ``eeg_fmri`` says the recordings were made inside a scanner, which is what makes
        # the gradient, ballistocardiogram and scanner-harmonic stages meaningful at all.
        # ``brainvision_analyzer`` says an Analyzer correction ran upstream and left pulse
        # markers behind for those stages to read. EEG recorded outside a scanner has
        # neither; EEG recorded inside one and corrected elsewhere has the first only.
        #
        # Conflating them meant a plain EEG dataset either ran scanner QC against inputs
        # it does not have, or lost the scanner-harmonic measurement — which needs no
        # Analyzer output — merely by not having used Analyzer.
        eeg_fmri = self._is_eeg_fmri()
        analyzer_enabled = eeg_fmri and bool(
            self.config.get("preprocessing.brainvision_analyzer.enabled", False)
        )

        if analyzer_enabled:
            steps.insert(0, STEP_PULSE_MARKER_QC)
            if STEP_ICA_FIT in steps:
                steps.append(STEP_ICA_CARDIAC_QC)
            if STEP_EPOCHS in steps:
                steps.append(STEP_CARDIAC_ATTENUATION_QC)
        if eeg_fmri and STEP_EPOCHS in steps and not task_is_rest:
            # Measured from the EEG spectrum against the sequence timing, so it needs the
            # scanner but not the Analyzer correction.
            steps.append(STEP_SCANNER_HARMONIC_QC)

        if subjects is not None:
            self._validate_eeg_fmri_declaration()
            self._validate_bad_channel_sync_policy_for_steps(steps, subjects, task)
        return steps

    def _check_config_coherence(self) -> None:
        """Report every config-only contradiction before the first step runs.

        These used to surface one at a time from the stage that tripped over them, two of
        them only after ICA had been fitted. Nothing here reads a recording, so there is
        no reason for any of it to wait that long.

        Warnings are logged rather than raised: a config adapted from another study
        asking for stages this dataset cannot supply is expected, and the listing is how
        the reader learns which ones are being skipped.
        """
        from eeg_pipeline.utils.config.coherence import check_config_coherence

        report = check_config_coherence(self.config)
        report.log_warnings(self.logger)
        report.raise_if_errors()

    def _is_eeg_fmri(self) -> bool:
        """Whether these recordings were acquired inside an MR scanner.

        Delegates to :func:`eeg_pipeline.utils.config.acquisition.is_eeg_fmri` so that the
        stages selected here and the QC metrics resolved in
        ``eeg_pipeline.utils.data.preprocessing`` cannot answer this differently.
        """
        from eeg_pipeline.utils.config.acquisition import is_eeg_fmri

        return is_eeg_fmri(self.config)

    def _validate_bad_channel_sync_policy_for_steps(
        self,
        steps: List[str],
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Reject a sync policy the requested steps cannot satisfy, before any work runs.

        MNE-BIDS-Pipeline concatenates runs for the shared ICA and for epoching, which
        requires one bad-channel set per subject. Under ``per_run`` that constraint is
        violated by any subject whose runs disagree — the normal case once PyPREP has run.

        The BIDS ``channels.tsv`` files already record the per-run decision and cost
        nothing to read, so the same failure that used to surface after PyPREP and
        filtering surfaces here instead, before the first recording is opened. Subjects
        whose channel files do not exist yet are left to the later check.
        """
        if not ({STEP_ICA_FIT, STEP_EPOCHS} & set(steps)):
            return
        if self._resolve_bad_channel_sync_policy() == "subject_union":
            return

        for subject in self._resolve_bids_subjects(subjects):
            bads_by_run = self._read_bids_bad_channels(subject, task)
            if len(bads_by_run) < 2:
                continue
            distinct = {tuple(bads) for bads in bads_by_run.values()}
            if len(distinct) > 1:
                raise ValueError(
                    f"sub-{subject} has different bad channels in different runs "
                    f"({bads_by_run}), but pyprep.bad_channel_sync_policy='per_run'. "
                    "MNE-BIDS-Pipeline concatenates runs for the shared ICA and for "
                    "epoching, which requires one bad-channel set per subject. Set "
                    "pyprep.bad_channel_sync_policy='subject_union', or process each run "
                    "as its own task."
                )

    def _resolve_bids_subjects(self, subjects: List[str]) -> List[str]:
        """Resolve explicit or discovered subjects against the BIDS root."""
        if subjects == ["all"]:
            return sorted(
                path.name.removeprefix("sub-")
                for path in self.bids_root.glob("sub-*")
                if path.is_dir()
            )
        return [subject.removeprefix("sub-") for subject in subjects]

    def _read_bids_bad_channels(
        self,
        subject: str,
        task: Optional[str],
    ) -> Dict[str, tuple[str, ...]]:
        """Read the bad EEG channels each run's channels.tsv records."""
        selector = f"_task-{task}_" if task is not None else "_task-"
        bads_by_run: Dict[str, tuple[str, ...]] = {}
        subject_dir = self.bids_root / f"sub-{subject}"
        for path in sorted(subject_dir.rglob(f"sub-{subject}_*_channels.tsv")):
            if path.name.startswith("._") or selector not in path.name:
                continue
            # utf-8-sig: mne-bids writes these with a BOM.
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            bads_by_run[path.name] = tuple(
                sorted(
                    row["name"]
                    for row in rows
                    if str(row.get("type", "")).lower() == "eeg"
                    and str(row.get("status", "")).lower() == "bad"
                )
            )
        return bads_by_run

    def _validate_eeg_fmri_declaration(self) -> None:
        """Reject a scanner declaration the recordings cannot support.

        ``preprocessing.eeg_fmri`` turns on stages that read an ECG channel and scanner
        volume markers. Declaring it for a dataset that has neither would fail later, deep
        inside a review, with an error about a missing channel rather than about the
        declaration that asked for it.

        Only the ECG channel is checked here. Volume markers are named differently across
        conversion paths, so their absence is not reliable evidence, whereas an ECG
        channel typed in ``channels.tsv`` is unambiguous and is what every cardiac stage
        actually opens.
        """
        if not self._is_eeg_fmri():
            return
        if not bool(self.config.get("ica.cardiac_review.enabled", False)) and not bool(
            self.config.get("preprocessing.brainvision_analyzer.enabled", False)
        ):
            return

        for path in sorted(self.bids_root.rglob("*_channels.tsv")):
            if path.name.startswith("._") or "eeg" not in path.parts:
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if str(row.get("type", "")).lower() == "ecg":
                        return
            # One recording is enough to settle it; reading the rest costs time and
            # changes nothing.
            break
        raise ValueError(
            "preprocessing.eeg_fmri is true, but no channel is typed ECG in the "
            f"channels.tsv files under {self.bids_root}. The cardiac stages this enables "
            "measure the ballistocardiogram against a recorded ECG and cannot run without "
            "one. Set preprocessing.eeg_fmri=false for EEG recorded outside a scanner, or "
            "correct the channel types if an ECG was recorded."
        )

    def _bids_eog_channel_names(self) -> set[str]:
        """Return every channel any run types as EOG in its channels.tsv."""
        names: set[str] = set()
        for path in sorted(self.bids_root.rglob("*_channels.tsv")):
            if path.name.startswith("._") or "eeg" not in path.parts:
                continue
            # utf-8-sig: mne-bids writes these with a BOM.
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if str(row.get("type", "")).lower() == "eog":
                        names.add(str(row["name"]))
        return names

    def _resolve_eog_detection_channels(self) -> List[str]:
        """Return the channels MNE-BIDS-Pipeline will use to recover blinks.

        MNE-BIDS-Pipeline detects ocular components by correlating them against EOG
        channels. When neither ``eeg.eog_channels`` nor a channel typed EOG in the BIDS
        ``channels.tsv`` supplies one, upstream breaks out of its detection loop and
        leaves the component list empty — no warning, no error, and a report that reads
        exactly like a recording with no blinks in it. This montage has no dedicated EOG
        electrode, so that is the default outcome unless surrogates are named.
        """
        configured = self.config.get("eeg.eog_channels")
        if isinstance(configured, str):
            channels = [name.strip() for name in configured.split(",") if name.strip()]
        elif isinstance(configured, (list, tuple)):
            channels = [str(name).strip() for name in configured if str(name).strip()]
        else:
            channels = []

        if channels:
            return channels
        return sorted(self._bids_eog_channel_names())

    def _validate_eog_detection_is_reachable(self) -> None:
        """Reject an EOG detection request that upstream would silently skip."""
        if not bool(self.config.get("ica.use_eog_detection")):
            return
        if self._resolve_eog_detection_channels():
            return
        raise ValueError(
            "ica.use_eog_detection is true, but no EOG channel is available: "
            "eeg.eog_channels is unset and no channels.tsv under "
            f"{self.bids_root} types a channel as EOG. MNE-BIDS-Pipeline skips ocular "
            "component detection entirely in that case without raising, so blink removal "
            "would rest on ICLabel alone. Name frontopolar surrogates in eeg.eog_channels "
            "(upstream treats them as virtual EOG without removing them from the EEG "
            "analysis), or set ica.use_eog_detection=false to make the omission explicit."
        )

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
                    n_jobs=n_jobs,
                )
            elif step == STEP_EPOCHS:
                self._run_epoch_creation(
                    subjects=subjects,
                    task=task,
                    task_is_rest=task_is_rest,
                    n_jobs=n_jobs,
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

        # The review sections describe what preprocessing produced, not what any scanner
        # correction did, so they belong to every run that fitted an ICA — an EEG-only
        # dataset gets the same evidence minus the panels whose inputs it lacks. They run
        # after the step loop because the Analyzer panel reads QC tables that a later
        # step writes, and each section already omits itself when its inputs are absent.
        if {STEP_ICA_FIT, STEP_EPOCHS} & set(steps):
            self._append_report_review_sections(subjects=subjects, task=task)

        return outputs

    def _run_pulse_marker_qc(
        self,
        subjects: List[str],
        task: Optional[str],
    ) -> Path:
        """Measure the preserved Analyzer R markers and record them per run.

        Descriptive by default: the configured bounds are written beside the measurements
        rather than used to drop runs. ``strict_pulse_qc`` turns them into a gate for a
        caller that wants one.
        """
        from mne_bids import BIDSPath, read_raw_bids

        from eeg_pipeline.preprocessing.pulse_artifact_qc import (
            PulseMarkerCriteria,
            summarize_pulse_marker_recordings,
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
        qc_root = self.deriv_root / "preprocessed" / "eeg" / "qc"
        output_path = qc_root / f"{task_entity}desc-pulsemarkers_qc.tsv"
        strict_validation = self.config.get(
            "preprocessing.brainvision_analyzer.strict_pulse_qc", False
        )

        return summarize_pulse_marker_recordings(
            recordings,
            criteria,
            output_path=output_path,
            strict=strict_validation,
            # The stage that measures how much of each run the pulse correction covered
            # also records the intervals it did not, so nothing downstream has to rederive
            # them from the marker train.
            annotations_dir=qc_root / f"{task_entity}bcg_uncorrected",
        )

    def _get_analyzer_cardiac_qc_config(self) -> Any:
        config = self.config.get("preprocessing.brainvision_analyzer.cardiac_artifact_qc")
        if not config:
            raise ValueError(
                "Missing required config mapping: "
                "preprocessing.brainvision_analyzer.cardiac_artifact_qc"
            )
        return config

    def _resolve_ecg_channel(self) -> str:
        """Resolve the single ECG channel the cardiac QC steps read."""
        channels = self.config.get("eeg.ecg_channels")
        if not channels:
            raise ValueError(
                "Cardiac QC requires eeg.ecg_channels to name the recorded ECG channel."
            )
        if isinstance(channels, str):
            return channels.strip()
        if len(channels) != 1:
            raise ValueError(
                "Cardiac QC reads exactly one ECG channel; eeg.ecg_channels has "
                f"{len(channels)}: {list(channels)}."
            )
        return str(channels[0]).strip()

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
            ecg_channel=self._resolve_ecg_channel(),
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
            ecg_channel=self._resolve_ecg_channel(),
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
            # Detection filtering is its own decision, not the analysis band. Passing
            # ``preprocessing.h_freq`` here removed the >100 Hz content that PyPREP's
            # high-frequency-noise criterion is defined on, so the detector measured a
            # band that had already been filtered away. ``pyprep.detection_low_pass``
            # defaults to no low-pass; set it only if a recording needs one.
            l_pass=pyprep_cfg.get("detection_low_pass"),
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
        *,
        n_jobs: int,
    ) -> None:
        """Run ICA fitting via MNE-BIDS pipeline."""
        self._run_mne_bids_pipeline(
            self._get_ica_preparation_steps(),
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
            n_jobs=n_jobs,
        )
        self._harmonize_filtered_raw_bads_for_mne_concat(subjects, task)
        self._run_mne_bids_pipeline(
            self._get_ica_decomposition_steps(),
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
            n_jobs=n_jobs,
        )

        # The cardiac review measures the ballistocardiogram against a recorded ECG. Both
        # the artifact and the channel are properties of scanner acquisition, so outside
        # one there is nothing for it to measure. Skipping is what the dataset declaration
        # asks for, and is logged rather than silent — the opposite case, a stage quietly
        # doing nothing while its config says it is on, is the failure mode that hid the
        # missing ocular detection for so long.
        if bool(self.config.get("ica.cardiac_review.enabled", False)):
            if self._is_eeg_fmri():
                self._run_ica_cardiac_review(subjects=subjects, task=task)
            else:
                self.logger.info(
                    "Skipping the ICA cardiac review: preprocessing.eeg_fmri is false, so "
                    "these recordings carry no ballistocardiogram to measure. Set "
                    "ica.cardiac_review.enabled=false to stop requesting it."
                )

        if bool(self.config.get("ica.ocular_review.enabled", False)):
            self._run_ica_ocular_review(subjects=subjects, task=task)

        if bool(self.config.get("ica.band_specific_report.enabled", False)):
            if task_is_rest and bool(self.config.get("ica.band_specific_report.tfr.enabled", True)):
                raise ValueError(
                    "Resting-state epochs are fixed-length segments with no event and no "
                    "pre-stimulus interval, so the baseline-relative component TFR has no "
                    "baseline. Set ica.band_specific_report.tfr.enabled=false to review "
                    "resting-state components by topography and spectrum."
                )
            self._run_band_specific_ica_report(subjects=subjects, task=task)
            if self.config.get("ica.band_specific_report.comparisons"):
                if task_is_rest:
                    raise ValueError(
                        "Band-specific condition comparisons require event-related task epochs."
                    )
                self._run_mne_bids_pipeline(
                    "preprocessing/_07_make_epochs",
                    subjects=subjects,
                    task=task,
                    task_is_rest=False,
                    n_jobs=n_jobs,
                )
                self._append_provisional_band_ica_condition_tfrs(
                    subjects=subjects,
                    task=task,
                )

        self.logger.info("ICA fitting complete")

    def _run_ica_cardiac_review(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Append direct ECG detection and component evidence to MNE reports."""
        from eeg_pipeline.preprocessing.ica_cardiac_report import (
            generate_ica_cardiac_review,
        )
        from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings

        settings = CardiacReviewSettings.from_mapping(self.config.get("ica.cardiac_review", {}))

        # Promotion writes into the same table the manual review edits, and it cannot tell
        # a component nobody has looked at from one a reviewer deliberately cleared. Once
        # the review is signed off, the reviewer's table is the answer and this step must
        # not re-open it. Before sign-off it is the automated baseline the review adjusts,
        # which is the whole point of running it.
        if bool(self.config.get("ica.manual_review_complete")) and getattr(
            settings, "promote_exclusions", False
        ):
            settings = replace(settings, promote_exclusions=False)
            self.logger.info(
                "ica.manual_review_complete is set; leaving the reviewed component table "
                "alone instead of re-applying CTPS promotions."
            )

        for subject in self._resolve_bad_harmonization_subjects(subjects):
            filtered_paths = self._find_filtered_raw_run_files(subject, task)
            for epochs_path, report_path, output_prefix in self._find_band_ica_report_inputs(
                subject
            ):
                session_filtered_paths = [
                    path
                    for path in filtered_paths
                    if path.name.startswith(f"{output_prefix}_task-")
                ]
                if not session_filtered_paths:
                    raise FileNotFoundError(
                        f"No filtered raw runs match ECG review prefix {output_prefix!r}."
                    )
                generate_ica_cardiac_review(
                    filtered_raw_paths=session_filtered_paths,
                    ica_path=epochs_path.with_name(f"{output_prefix}_proc-ica_ica.fif"),
                    report_path=report_path,
                    output_path=epochs_path.with_name(
                        f"{output_prefix}_desc-icaecg_components.tsv"
                    ),
                    settings=settings,
                )

    def _run_ica_ocular_review(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        from eeg_pipeline.preprocessing.ica_ocular_report import (
            generate_ica_ocular_review,
            OcularReviewSettings,
        )

        settings = OcularReviewSettings.from_mapping(self.config.get("ica.ocular_review", {}))
        for subject in self._resolve_bad_harmonization_subjects(subjects):
            filtered_paths = self._find_filtered_raw_run_files(subject, task)
            for epochs_path, report_path, output_prefix in self._find_band_ica_report_inputs(
                subject
            ):
                session_filtered_paths = [
                    path
                    for path in filtered_paths
                    if path.name.startswith(f"{output_prefix}_task-")
                ]
                if not session_filtered_paths:
                    raise FileNotFoundError(
                        f"No filtered raw runs match EOG review prefix {output_prefix!r}."
                    )
                generate_ica_ocular_review(
                    filtered_raw_paths=session_filtered_paths,
                    ica_path=epochs_path.with_name(f"{output_prefix}_proc-ica_ica.fif"),
                    report_path=report_path,
                    output_path=epochs_path.with_name(
                        f"{output_prefix}_desc-icaeog_components.tsv"
                    ),
                    settings=settings,
                )

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
            filtered_paths = self._find_filtered_raw_run_files(subject, task)
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
                    filtered_raw_paths=[
                        path
                        for path in filtered_paths
                        if path.name.startswith(f"{output_prefix}_task-")
                    ],
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

    def _append_provisional_band_ica_condition_tfrs(
        self,
        *,
        subjects: List[str],
        task: str,
    ) -> None:
        """Append pre-review comparisons from all pre-ICA task epochs."""
        from eeg_pipeline.preprocessing.band_ica_report import (
            BandIcaReportSettings,
            append_condition_tfr_report,
        )
        from eeg_pipeline.utils.data.preprocessing import write_clean_events_tsv_for_epochs

        settings = BandIcaReportSettings.from_mapping(
            self.config.get("ica.band_specific_report", {})
        )
        conditions = self._resolve_epoch_conditions(task)
        for subject in self._resolve_bad_harmonization_subjects(subjects):
            subject_dir = self.deriv_root / "preprocessed" / "eeg" / f"sub-{subject}"
            task_epochs_paths = sorted(
                path
                for path in subject_dir.rglob(f"sub-{subject}*_task-{task}_epo.fif")
                if "_proc-" not in path.name and not path.name.startswith("._")
            )
            if not task_epochs_paths:
                raise FileNotFoundError(
                    f"No pre-ICA task epochs found for provisional comparisons: sub-{subject}"
                )
            for task_epochs_path in task_epochs_paths:
                entity_prefix = task_epochs_path.name.removesuffix(f"_task-{task}_epo.fif")
                aligned_events_path = write_clean_events_tsv_for_epochs(
                    subject=subject,
                    task=task,
                    bids_root=self.bids_root,
                    epochs_path=task_epochs_path,
                    config=self.config,
                    conditions=conditions,
                    overwrite=True,
                    # These are the pre-ICA task epochs, before any rejection: that is what
                    # makes the comparison below provisional. AutoReject is fitted in the
                    # rejection step, so no per-trial repair record exists yet to attach.
                    after_rejection=False,
                    _logger=self.logger,
                )
                report_path = task_epochs_path.with_name(f"{entity_prefix}_report.h5")
                ica_fit_epochs_path = task_epochs_path.with_name(
                    f"{entity_prefix}_proc-icafit_epo.fif"
                )
                standard_ica_path = task_epochs_path.with_name(f"{entity_prefix}_proc-ica_ica.fif")
                append_condition_tfr_report(
                    ica_fit_epochs_path=ica_fit_epochs_path,
                    standard_ica_path=standard_ica_path,
                    pre_ica_epochs_path=task_epochs_path,
                    clean_epochs_path=task_epochs_path,
                    clean_events_path=aligned_events_path,
                    report_path=report_path,
                    settings=settings,
                    analysis_status="Provisional — all task epochs",
                )

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

            # Only metadata is needed to decide whether anything must change, and a
            # rewrite is the exception rather than the rule — the second call of this
            # method in a run always finds the runs already harmonized. Reading headers
            # here keeps that pass off the data entirely.
            bads_by_path = {}
            ch_names_by_path = {}
            eeg_ch_names_by_path = {}
            for path in filtered_paths:
                raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
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

            for path in filtered_paths:
                raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
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

    def _filtered_sampling_rate(self, filtered_path: Path) -> float:
        """Return the sampling rate of a filtered run, without loading its data."""
        import mne

        return float(mne.io.read_raw_fif(filtered_path, verbose="ERROR").info["sfreq"])

    def _configured_epoch_window(self) -> Optional[tuple]:
        """Return the epoch bounds the filter response is drawn against.

        ``None`` for a resting-state run, which is analysed continuously and has no epoch
        for filter ringing to reach into.
        """
        if self._resolve_task_is_rest():
            return None
        tmin = self.config.get("epochs.tmin", None)
        tmax = self.config.get("epochs.tmax", None)
        if tmin is None or tmax is None:
            return None
        return (float(tmin), float(tmax))

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
        from eeg_pipeline.preprocessing.derivatives import find_filtered_raw_runs

        return find_filtered_raw_runs(
            self.deriv_root / "preprocessed" / "eeg",
            subject=subject,
            task=task,
        )

    def _save_raw_with_updated_bads(self, raw: Any, path: Path) -> None:
        """Rewrite a raw FIF after changing only its bad-channel metadata."""
        if not path.name.endswith("_raw.fif"):
            raise ValueError(f"Expected raw FIF path ending in '_raw.fif', got {path}")
        # The marker is matched anywhere in the name so that split parts are found under
        # either naming scheme: 'bids' inserts a `split-NN` entity mid-name rather than
        # appending to the stem.
        marker = "_badsync_"
        tmp_path = path.with_name(path.name.replace("_raw.fif", f"{marker}raw.fif"))

        def written_parts() -> List[Path]:
            return sorted(p for p in tmp_path.parent.glob("*.fif") if marker in p.name)

        for stale in written_parts():
            stale.unlink(missing_ok=True)
        try:
            raw.save(tmp_path, overwrite=True, split_naming="bids")
            # A split save would leave the later parts behind while only the first was
            # moved into place, silently truncating the run.
            parts = written_parts()
            if parts != [tmp_path]:
                raise RuntimeError(
                    f"Rewriting bad-channel metadata split {path} across {len(parts)} "
                    f"files: {parts}. Split recordings are not supported here."
                )
            tmp_path.replace(path)
        finally:
            for leftover in written_parts():
                leftover.unlink(missing_ok=True)

    def _run_epoch_creation(
        self,
        subjects: List[str],
        task: Optional[str],
        task_is_rest: bool,
        *,
        n_jobs: int,
    ) -> None:
        """Create epochs and apply ICA via MNE-BIDS pipeline."""
        steps = "preprocessing/_07_make_epochs,preprocessing/_08a_apply_ica,preprocessing/_09_ptp_reject"
        self._harmonize_filtered_raw_bads_for_mne_concat(subjects, task)

        self._run_mne_bids_pipeline(
            steps,
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
            n_jobs=n_jobs,
        )

        # Before clean events, which carry the per-trial repair counts this produces.
        if bool(self.config.get("preprocessing.autoreject_log", False)):
            self._write_autoreject_logs(subjects=subjects, task=task)

        if task_is_rest:
            self.logger.info("Skipping clean events export for resting-state preprocessing")
        elif bool(self.config.get("preprocessing.write_clean_events", True)):
            self._write_clean_events_tsv(subjects=subjects, task=task)

        self._append_epoch_rejection_review(subjects=subjects, task=task)

        band_report_config = self.config.get("ica.band_specific_report", {})
        if bool(band_report_config.get("enabled", False)) and band_report_config.get("comparisons"):
            self._append_band_ica_condition_tfrs(subjects=subjects, task=task)

        self.logger.info("Epoch creation complete")

    def _write_autoreject_logs(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Persist AutoReject's per-channel verdict beside each subject's epochs.

        MNE-BIDS-Pipeline discards the reject log after using it for one report figure,
        so it is refitted here with the same settings on the same input and checked
        against the cleaned epochs it claims to describe.
        """
        import mne

        from eeg_pipeline.infra.paths import find_clean_epochs_path
        from eeg_pipeline.preprocessing.autoreject_log import (
            AutorejectLogSettings,
            autoreject_log_path_for_epochs,
            compute_autoreject_log,
            pre_rejection_epochs_path,
            verify_log_describes_clean_epochs,
            write_autoreject_log,
        )

        settings = AutorejectLogSettings.from_config(self.config)

        for subject in self._resolve_bad_harmonization_subjects(subjects):
            clean_path = find_clean_epochs_path(
                subject,
                task,
                deriv_root=self.deriv_root,
                config=self.config,
            )
            if clean_path is None or not clean_path.is_file():
                raise FileNotFoundError(
                    f"Clean epochs not found; cannot log AutoReject for sub-{subject}, "
                    f"task-{task}"
                )
            fit_path = pre_rejection_epochs_path(clean_path)
            if not fit_path.is_file():
                raise FileNotFoundError(
                    f"Pre-rejection epochs not found at {fit_path}; AutoReject cannot be "
                    f"refitted for sub-{subject}, task-{task}"
                )

            log = compute_autoreject_log(
                mne.read_epochs(fit_path, preload=True, verbose="ERROR"),
                settings,
            )
            verify_log_describes_clean_epochs(
                log,
                mne.read_epochs(clean_path, preload=False, verbose="ERROR"),
            )
            written = write_autoreject_log(log, autoreject_log_path_for_epochs(clean_path))
            self.logger.info(
                "sub-%s: AutoReject dropped %d of %d epochs, interpolated %d channel-trials "
                "(n_interpolate=%d, consensus=%.2f) -> %s",
                subject,
                int(log.bad_epochs.sum()),
                len(log.bad_epochs),
                int((log.labels[~log.bad_epochs] == 2).sum()),
                log.n_interpolate,
                log.consensus,
                written,
            )

    def _append_epoch_rejection_review(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Append trial-retention evidence to each subject report."""
        import mne
        import pandas as pd

        from eeg_pipeline.infra.paths import find_clean_epochs_path
        from eeg_pipeline.preprocessing.report.build_record import save_subject_report
        from eeg_pipeline.preprocessing.report.organize import open_subject_report
        from eeg_pipeline.preprocessing.report.rejection import add_rejection_review

        for subject in self._resolve_bad_harmonization_subjects(subjects):
            epochs_path = find_clean_epochs_path(
                subject,
                task,
                deriv_root=self.deriv_root,
                config=self.config,
            )
            if epochs_path is None or not epochs_path.exists():
                self.logger.warning(
                    "Clean epochs not found; skipping epoch-rejection review for sub-%s",
                    subject,
                )
                continue

            report_path = self._find_subject_report_path(epochs_path)
            if report_path is None:
                self.logger.warning(
                    "Subject report not found; skipping epoch-rejection review for sub-%s",
                    subject,
                )
                continue

            events_path = epochs_path.with_name(epochs_path.name.replace("_epo.fif", "_events.tsv"))
            clean_events = pd.read_csv(events_path, sep="\t") if events_path.is_file() else None
            report = open_subject_report(report_path)
            summary = add_rejection_review(
                report=report,
                clean_epochs=mne.read_epochs(epochs_path, preload=False, verbose="ERROR"),
                clean_events=clean_events,
                config=self.config,
            )
            preservation = self._append_signal_preservation(
                report=report,
                epochs_path=epochs_path,
                subject=subject,
            )
            save_subject_report(
                report,
                report_path,
                stage="epochs",
                measurements={
                    "epochs_kept": int(summary.kept),
                    "epochs_total": int(summary.total),
                    "epochs_dropped_fraction": float(summary.dropped_fraction),
                    **preservation,
                },
            )
            self.logger.info(
                "sub-%s retained %d of %d epochs (%.1f%% dropped)",
                subject,
                summary.kept,
                summary.total,
                100.0 * summary.dropped_fraction,
            )

    def _append_signal_preservation(
        self,
        *,
        report,
        epochs_path: Path,
        subject: str,
    ) -> dict:
        """Add evidence that brain signal survived, beside the evidence of removal.

        Returns the measurements it made so the caller can record them beside the report.
        """
        import mne

        from eeg_pipeline.preprocessing.report.preservation import (
            add_rest_preservation_review,
            add_task_preservation_review,
        )
        from eeg_pipeline.preprocessing.report.settings import ReportSettings

        if not ReportSettings.from_config(self.config).enabled:
            return {}
        epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
        if bool(self.config.get("preprocessing.task_is_rest", False)):
            reliability = None
            alpha = add_rest_preservation_review(report=report, epochs=epochs)
        else:
            reliability, alpha = add_task_preservation_review(report=report, epochs=epochs)
        self.logger.info(
            "sub-%s preservation: split-half r=%s posterior alpha=%s",
            subject,
            "n/a" if reliability is None else f"{reliability.corrected_correlation:.3f}",
            "n/a" if alpha is None else f"{alpha.prominence_db:.1f} dB",
        )
        return _preservation_measurements(reliability=reliability, alpha=alpha)

    def _append_provisional_signal_preservation(
        self,
        *,
        report,
        report_path: Path,
        task: str,
        subject: str,
    ):
        """Add preservation evidence before the exclusions are approved.

        The same section is rendered again after rejection, on the epochs that survived
        it. This earlier pass exists because by the time the final one runs the decision
        it informs has already been taken: at ICA review the reviewer is looking at a
        page on which every panel measures removal, and nothing yet says whether anything
        was left. The status string is what keeps the two readings apart, and the
        tag-scoped removal inside the preservation section means the later pass replaces
        this one rather than sitting beside it.

        Absent inputs are not an error. A report built from the bad-channel stage alone
        has no epochs to measure, and a resting-state recording is measured on its
        posterior rhythm rather than an evoked response.
        """
        import mne

        from eeg_pipeline.preprocessing.report.preservation import (
            add_rest_preservation_review,
            add_task_preservation_review,
        )

        prefix = report_path.name.removesuffix("_report.h5")
        epochs_path = report_path.with_name(f"{prefix}_task-{task}_epo.fif")
        if not epochs_path.exists():
            self.logger.info(
                "No pre-rejection epochs for sub-%s; skipping provisional preservation",
                subject,
            )
            return None

        status = "Provisional — all task epochs, before rejection"
        epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
        if bool(self.config.get("preprocessing.task_is_rest", False)):
            return add_rest_preservation_review(
                report=report,
                epochs=epochs,
                analysis_status=status,
            )
        _, alpha = add_task_preservation_review(
            report=report,
            epochs=epochs,
            analysis_status=status,
        )
        # Returned rather than discarded: this is the one place the posterior rhythm is
        # measured on the cleaned data, and the cohort sidecar needs the measurement
        # object rather than the headline scalar the build record keeps.
        return alpha

    def _append_report_review_sections(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
    ) -> None:
        """Append the configurable review sections to each subject report.

        Every section is optional and absent when its inputs are: a dataset recorded
        outside a scanner simply has no Analyzer section, rather than an empty one.
        """
        from eeg_pipeline.preprocessing.report.analyzer_qc import (
            add_analyzer_correction_review,
        )
        from eeg_pipeline.preprocessing.report.coverage import add_coverage_review
        from eeg_pipeline.preprocessing.report.filtering import (
            add_filter_review,
            describe_configured_filter,
        )
        from eeg_pipeline.preprocessing.report.build_record import save_subject_report
        from eeg_pipeline.preprocessing.report.organize import open_subject_report
        from eeg_pipeline.preprocessing.report.provenance import add_provenance_review
        from eeg_pipeline.preprocessing.report.settings import ReportSettings

        report_settings = ReportSettings.from_config(self.config)
        if not report_settings.enabled:
            self.logger.info("Skipping report review sections (report.enabled=false)")
            return
        if task is None:
            self.logger.info("Skipping report review sections without a task label")
            return

        deriv_eeg_root = self.deriv_root / "preprocessed" / "eeg"

        for subject in self._resolve_bad_harmonization_subjects(subjects):
            reports = sorted(
                path
                for path in (deriv_eeg_root / f"sub-{subject}").rglob("*_report.h5")
                if not path.name.startswith("._")
            )
            if not reports:
                self.logger.warning("No report found; skipping review sections for sub-%s", subject)
                continue
            filtered_paths = self._find_filtered_raw_run_files(subject, task)
            for report_path in reports:
                report = open_subject_report(report_path)
                add_provenance_review(report=report, config=self.config)
                # Skipped without a filtered run to read the sampling rate from, which is
                # the case for a dataset filtered upstream: there is no pipeline filter to
                # describe, and inventing a rate would describe one that was never applied.
                if filtered_paths:
                    description = describe_configured_filter(
                        self.config,
                        sfreq=self._filtered_sampling_rate(filtered_paths[0]),
                    )
                    if description is not None:
                        add_filter_review(
                            report=report,
                            description=description,
                            epoch_window_s=self._configured_epoch_window(),
                        )
                analyzer = add_analyzer_correction_review(
                    report=report,
                    qc_dir=deriv_eeg_root / "qc",
                    task=task,
                    subject=subject,
                )
                coverage = add_coverage_review(
                    report=report,
                    deriv_eeg_root=deriv_eeg_root,
                    task=task,
                    subject=subject,
                    settings=report_settings,
                )
                evidence = self._append_run_evidence(
                    report=report,
                    report_path=report_path,
                    filtered_paths=filtered_paths,
                    settings=report_settings,
                )
                self._append_provisional_signal_preservation(
                    report=report,
                    report_path=report_path,
                    task=task,
                    subject=subject,
                )
                record = save_subject_report(
                    report,
                    report_path,
                    stage="report-review",
                    measurements=_review_stage_measurements(
                        coverage=coverage,
                        evidence=evidence,
                    ),
                )
                # Written after the report is saved, so the sidecar carries the record of
                # the document that exists rather than of the one being built.
                self._write_qc_sidecar(
                    report_path=report_path,
                    record=record,
                    subject=subject,
                    task=task,
                    evidence=evidence,
                    settings=report_settings,
                )
                self.logger.info(
                    "sub-%s report sections: analyzer=%s coverage=%s gradient=%s runs=%d",
                    subject,
                    "present" if analyzer is not None else "absent",
                    "present" if coverage is not None else "absent",
                    (
                        "present"
                        if evidence is not None and evidence.has_scanner_evidence
                        else "absent"
                    ),
                    0 if evidence is None else len(evidence.spectra),
                )

    def _write_qc_sidecar(
        self,
        *,
        report_path: Path,
        record: dict,
        subject: str,
        task: str,
        evidence,
        settings,
    ) -> None:
        """Write the QC sidecar a cohort report reads, beside the subject report.

        This is the seam the whole cohort feature rests on. ``measure_runs`` has just made
        the one expensive pass over every run -- read, apply the exclusions, measure -- and
        those results are about to go out of scope. Writing them down here means a cohort
        document reads tables rather than buying that pass again from a gigabyte of
        filtered raw per participant, and it means the cohort figure and the subject figure
        beneath it are provably the same measurement rather than two computations that have
        to be kept agreeing.

        Failure here is logged and swallowed. The subject report is already written and is
        the deliverable; losing a participant from a future cohort run is a smaller harm
        than failing the stage that produced the document, and the cohort command lists a
        participant with no sidecar rather than silently dropping it.
        """
        from dataclasses import asdict

        from eeg_pipeline.preprocessing.report.at_a_glance import latest_measurements
        from eeg_pipeline.preprocessing.report.cohort.record import (
            AFTER,
            BEFORE,
            build_subject_sidecar,
            pool_alpha_runs,
        )
        from eeg_pipeline.preprocessing.report.cohort.sidecar import write_sidecar

        if evidence is None or not evidence.spectra:
            self.logger.info("No per-run evidence for sub-%s; no QC sidecar written", subject)
            return

        try:
            presented, retained = self._condition_counts(report_path=report_path, task=task)
            sidecar = build_subject_sidecar(
                subject=subject,
                task=task,
                spectra=evidence.spectra,
                continuity=evidence.continuity,
                timings=evidence.timings,
                locked_averages=evidence.locked_averages,
                combs=evidence.combs,
                rr_intervals=evidence.rr_intervals,
                marker_agreements=evidence.marker_agreements,
                cardiac_residuals=evidence.cardiac_residuals,
                # Both sides come from the measuring pass, which is the only place the
                # same data exists before and after the exclusions. The epochs-based
                # measurement in ``alpha`` stays on the subject panel under its own
                # unsuffixed key: it is a better measurement of the final rhythm, but it
                # has no "before" to be paired against, and pairing two estimators would
                # measure the difference between them rather than the effect of cleaning.
                alpha=self._paired_alpha(evidence, pool=pool_alpha_runs, stages=(BEFORE, AFTER)),
                components=self._ica_component_table(report_path),
                channel_positions=evidence.channel_positions,
                bad_channels_by_run=evidence.bad_channels_by_run,
                trials_by_condition=presented,
                retained_by_condition=retained,
                measurements=latest_measurements(record),
                settings=asdict(settings),
                versions=_recorded_versions(record),
                acquisition_date=evidence.acquisition_date,
            )
            paths = write_sidecar(report_path, sidecar)
        except (ValueError, OSError, KeyError) as error:
            self.logger.warning("Could not write the QC sidecar for sub-%s: %s", subject, error)
            return
        self.logger.info(
            "sub-%s QC sidecar: %d run(s), %d condition(s) -> %s",
            subject,
            sidecar.n_runs,
            len(sidecar.conditions),
            paths.subject_json.name,
        )

    @staticmethod
    def _paired_alpha(evidence, *, pool, stages: tuple[str, str]) -> dict:
        """The rhythm either side of the exclusions, or nothing when it is not paired.

        Both sides or neither. A participant carrying only one side would sit in the paired
        panel as half a comparison, and the panel would either drop them silently or draw a
        difference against a value it does not have.
        """
        before, after = stages
        pooled = {
            before: pool(getattr(evidence, "posterior_alpha_before", ())),
            after: pool(getattr(evidence, "posterior_alpha_after", ())),
        }
        if any(value is None for value in pooled.values()):
            return {}
        return pooled

    def _ica_component_table(self, report_path: Path):
        """The reviewed component table beside a report, or ``None`` when there is none."""
        import pandas as pd

        from eeg_pipeline.preprocessing.ica_exclusions import components_path_for_ica

        prefix = report_path.name.removesuffix("_report.h5")
        path = components_path_for_ica(report_path.with_name(f"{prefix}_proc-ica_ica.fif"))
        if not path.is_file():
            return None
        try:
            return pd.read_csv(path, sep="\t")
        except (OSError, ValueError):
            return None

    def _condition_counts(self, *, report_path: Path, task: str) -> tuple[dict, dict]:
        """Trials presented and trials retained, per condition.

        Presented comes from the pre-rejection epochs and retained from the clean ones, so
        the two counts are read from the two files that define them rather than one being
        inferred from the other. Either may be absent -- a resting-state recording has
        neither, and a run that stopped after ICA has only the first -- and an absent one
        contributes no counts rather than zeros.
        """
        from eeg_pipeline.infra.paths import find_clean_epochs_path

        prefix = report_path.name.removesuffix("_report.h5")
        presented = self._epoch_condition_counts(
            report_path.with_name(f"{prefix}_task-{task}_epo.fif")
        )
        subject = _subject_of_report(report_path)
        retained = (
            {}
            if subject is None
            else self._epoch_condition_counts(
                find_clean_epochs_path(
                    subject, task, deriv_root=self.deriv_root, config=self.config
                )
            )
        )
        return presented, retained

    def _epoch_condition_counts(self, epochs_path: Optional[Path]) -> dict:
        """Count trials per condition in an epochs file, from its event mapping.

        ``event_id`` is used rather than the metadata table because it is what defines a
        condition to MNE: it is present whether or not the pipeline wrote metadata, and it
        names conditions exactly as every other section of the report does.

        Counted by matching the event codes directly rather than by selecting with
        ``epochs[name]``. Selection resolves a hierarchical id such as ``painful/left``
        against every level of the hierarchy, so ``painful`` would count the trials of
        every condition beneath it and the counts would sum to more than the trials that
        were presented.
        """
        import mne

        if epochs_path is None or not Path(epochs_path).exists():
            return {}
        try:
            epochs = mne.read_epochs(epochs_path, preload=False, verbose="ERROR")
        except (OSError, ValueError) as error:
            self.logger.info("Could not read %s for condition counts: %s", epochs_path, error)
            return {}
        codes = epochs.events[:, 2]
        # A condition present in the mapping and absent from the data counts zero rather
        # than dropping out: "this participant saw none of that condition" is exactly the
        # cell the cohort events panel exists to make visible.
        return {str(name): int((codes == code).sum()) for name, code in epochs.event_id.items()}

    def _append_run_evidence(
        self,
        *,
        report,
        report_path: Path,
        filtered_paths: List[Path],
        settings,
    ):
        """Add every per-run section for the runs belonging to one report.

        The sensor spectra, gradient residual, time-resolved quality, and beat detection
        all need each run before and after the ICA exclusions, so they are measured in a
        single pass rather than one read and one ``ICA.apply`` per section.
        """
        from eeg_pipeline.preprocessing.ica_exclusions import (
            read_ica_with_reviewed_exclusions,
        )
        from eeg_pipeline.preprocessing.report.run_evidence import add_run_evidence_review

        prefix = report_path.name.removesuffix("_report.h5")
        run_paths = [path for path in filtered_paths if path.name.startswith(f"{prefix}_task-")]
        ica_path = report_path.with_name(f"{prefix}_proc-ica_ica.fif")
        if not run_paths or not ica_path.is_file():
            self.logger.info("Skipping per-run report evidence for %s (missing inputs)", prefix)
            return None
        return add_run_evidence_review(
            report=report,
            filtered_raw_paths=run_paths,
            # The exclusions live in the component table, not in the ICA file: reading
            # them off the file would compare each run against an uncleaned copy of
            # itself and render that as a reassuringly small correction.
            ica=read_ica_with_reviewed_exclusions(ica_path),
            settings=settings,
        )

    def _find_subject_report_path(self, epochs_path: Path) -> Optional[Path]:
        """Find the MNE report that sits beside a subject's epochs."""
        candidates = sorted(
            path
            for path in epochs_path.parent.glob("*_report.h5")
            if not path.name.startswith("._")
        )
        return candidates[0] if candidates else None

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
                ica_fit_epochs_path = clean_epochs_path.with_name(
                    f"{entity_prefix}_proc-icafit_epo.fif"
                )
                standard_ica_path = clean_epochs_path.with_name(f"{entity_prefix}_proc-ica_ica.fif")
                for required_path in (
                    ica_fit_epochs_path,
                    standard_ica_path,
                    pre_ica_epochs_path,
                    clean_events_path,
                    report_path,
                ):
                    if not required_path.is_file():
                        raise FileNotFoundError(
                            f"ICA component review input does not exist: {required_path}"
                        )
                append_condition_tfr_report(
                    ica_fit_epochs_path=ica_fit_epochs_path,
                    standard_ica_path=standard_ica_path,
                    pre_ica_epochs_path=pre_ica_epochs_path,
                    clean_epochs_path=clean_epochs_path,
                    clean_events_path=clean_events_path,
                    report_path=report_path,
                    settings=settings,
                    analysis_status="Finalized — retained epochs",
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

        for subj in self._resolve_bad_harmonization_subjects(subjects):
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
        *,
        n_jobs: int,
    ) -> None:
        """Run MNE-BIDS pipeline with a generated config file.

        mne_bids_pipeline requires settings in a Python config file,
        not CLI arguments. This generates a temporary config and passes it
        via --config.

        ``n_jobs`` is keyword-only and has no default on purpose. MNE-BIDS-Pipeline
        defaults it to 1, so a call site that forgets it produces a run that succeeds
        serially rather than one that fails — the failure mode this argument exists to
        prevent is silent.

        Args:
            steps: MNE-BIDS pipeline steps to run
            subjects: List of subject IDs to process (without 'sub-' prefix)
            task: Task name to constrain MNE-BIDS-Pipeline subject selection
            task_is_rest: Override config to enable/disable resting-state preprocessing
            n_jobs: Worker count for MNE-BIDS-Pipeline's own parallelism
        """
        import tempfile

        config_content = self._generate_mne_bids_config(
            steps,
            subjects=subjects,
            task=task,
            task_is_rest=task_is_rest,
            n_jobs=n_jobs,
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
                self._log_mne_bids_warnings(result.stderr)

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

    #: Substrings marking a stderr line worth raising above the rest of the subprocess
    #: output. A successful MNE-BIDS run still reports rank deficiency, ICA
    #: non-convergence, and dropped epochs this way, and logging the whole stream as one
    #: blob is how those go unread.
    _MNE_BIDS_NOTABLE_WARNINGS = (
        "rank",
        "did not converge",
        "not converge",
        "dropped",
        "RuntimeWarning",
        "UserWarning",
        "DeprecationWarning",
    )

    def _log_mne_bids_warnings(self, stderr: str) -> None:
        """Log the MNE-BIDS subprocess stderr, promoting the lines that matter."""
        notable = [
            line
            for line in stderr.splitlines()
            if any(marker.lower() in line.lower() for marker in self._MNE_BIDS_NOTABLE_WARNINGS)
        ]
        for line in notable:
            self.logger.warning("MNE-BIDS: %s", line.strip())
        self.logger.info("MNE-BIDS stderr: %s", stderr)

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
        n_jobs: int = 1,
    ) -> str:
        """Generate Python config file content for mne_bids_pipeline.

        Args:
            steps: MNE-BIDS pipeline steps to run
            subjects: List of subject IDs to process (without 'sub-' prefix)
            task: Task name to constrain MNE-BIDS-Pipeline subject selection
            task_is_rest: Override config to enable/disable resting-state preprocessing
            n_jobs: Worker count. The default mirrors upstream's own default rather than
                inventing one, so the generated config says what MNE-BIDS-Pipeline would
                have assumed anyway. Production always passes the resolved value.
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

        # Execution. MNE-BIDS-Pipeline sorts its settings into ones that shape the
        # outputs and ones that only shape how the work is executed, and n_jobs is in the
        # second group: it fans the loop over subjects and runs out across workers, and
        # hands autoreject its worker count, while pinning each worker's BLAS to one
        # thread (inner_max_num_threads=1) and hardcoding n_jobs=1 inside filtering, ICA
        # artifact detection and ICA application so nothing nests. Every step still reads
        # the same inputs and computes the same function of them.
        lines.append(f"n_jobs = {int(n_jobs)}")
        lines.append("")

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

        # EOG channels. Emitted from the configured names only: a channel already typed
        # EOG in channels.tsv is found by upstream on its own, and restating it here
        # would be redundant. The guard below is what makes the "neither" case loud.
        self._validate_eog_detection_is_reachable()
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
        # Markers this project records alongside the task, which are never conditions.
        # Unlike the bad/edge rule below, these are ours to decide.
        excluded_prefixes = (
            "Volume",
            "Pulse",
            "SyncStatus",
            "New Segment",
            "Response",
        )
        becomes_an_event = _mne_annotation_event_pattern()

        def _is_excluded(trial_type: str) -> bool:
            """Whether a trial type is something MNE will not epoch, or is not a condition.

            The bad/edge half of this is taken from MNE rather than restated, because the
            two have to agree exactly. ``events_from_annotations`` drops those annotations
            before an event exists, so a condition list that admits one selects rows in the
            events table that have no epoch behind them, and the trial table silently stops
            being 1:1 with the epochs.

            That is what happened with ``BAD_restart/Trig_therm/T  1``: a hand-maintained
            list tested ``startswith("Bad")`` against a string spelled ``BAD_``, admitted it
            as its own condition, and the events table came out two rows longer than the
            epochs. Restating MNE's rule is how the two lists drift; reading it is how they
            cannot.

            Note the rule is deliberately leading-anchored, matching MNE: a bad tag that is
            not first — ``Trig_therm/BAD_restart/T  1`` — *does* become an event upstream,
            so excluding it here would recreate the same mismatch in the other direction.
            """
            if not becomes_an_event.match(trial_type):
                return True
            return any(
                trial_type.lower().startswith(prefix.lower()) for prefix in excluded_prefixes
            )

        preferred = sorted(
            t
            for t in conditions
            if any(t.startswith(p) for p in preferred_prefixes) and not _is_excluded(t)
        )
        if preferred:
            self.logger.info(
                "Auto-detected task conditions from BIDS columns %s: %s",
                sorted(detected_columns),
                preferred,
            )
            return preferred

        filtered = sorted(t for t in conditions if not _is_excluded(t))
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
