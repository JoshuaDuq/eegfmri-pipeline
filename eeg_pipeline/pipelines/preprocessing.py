"""
Preprocessing Pipeline
======================

Pipeline class for EEG preprocessing orchestration:
- Bad channel detection (PyPREP)
- ICA fitting and labeling (MNE-BIDS pipeline + mne-icalabel)
- Epoch creation and cleaning

Usage:
    pipeline = PreprocessingPipeline(config=config)
    pipeline.run_batch(subjects, task="thermalactive", mode="full")

Modes:
- full: Complete preprocessing (bad channels → ICA → epochs)
- bad-channels: Only bad channel detection
- ica: Only ICA fitting and labeling
- epochs: Only epoch creation and cleaning
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eeg_pipeline.pipelines.base import PipelineBase


STEP_BAD_CHANNELS = "bad-channels"
STEP_ICA_FIT = "ica-fit"
STEP_ICA_LABEL = "ica-label"
STEP_EPOCHS = "epochs"
STEP_STATS = "stats"


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
        self.bids_root = Path(self.config.bids_root)
    
    def _extract_preprocessing_params(
        self,
        task: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[str, str, bool, int, Any]:
        """Extract and normalize preprocessing parameters from kwargs.
        
        Returns:
            Tuple of (resolved_task, mode, use_icalabel, n_jobs, progress)
        """
        from eeg_pipeline.cli.common import ProgressReporter
        
        resolved_task = task or self.config.get("project.task", "thermalactive")
        mode = kwargs.get("mode", "full")
        use_icalabel = kwargs.get("use_icalabel", True)
        n_jobs = kwargs.get("n_jobs", 1)
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        
        return resolved_task, mode, use_icalabel, n_jobs, progress
    
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
                - mode: 'full', 'bad-channels', 'ica', or 'epochs'
                - use_icalabel: Whether to use mne-icalabel
                - n_jobs: Number of parallel jobs
                - progress: ProgressReporter for TUI feedback
        """
        resolved_task, mode, use_icalabel, n_jobs, progress = self._extract_preprocessing_params(task, kwargs)
        
        progress.subject_start(f"sub-{subject}")
        
        steps = self._get_steps_for_mode(mode)
        
        self._execute_steps(
            steps=steps,
            subjects=[subject],
            task=resolved_task,
            use_icalabel=use_icalabel,
            n_jobs=n_jobs,
            progress=progress,
        )
        
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
                - mode: 'full', 'bad-channels', 'ica', or 'epochs'
                - use_icalabel: Whether to use mne-icalabel
                - n_jobs: Number of parallel jobs
                - progress: ProgressReporter for TUI feedback
            
        Returns:
            List of per-subject status dictionaries
        """
        resolved_task, mode, use_icalabel, n_jobs, progress = self._extract_preprocessing_params(task, kwargs)
        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs=kwargs,
        )
        run_status = "failed"
        run_error: Optional[str] = None

        try:
            progress.start("preprocessing", subjects)

            steps = self._get_steps_for_mode(mode)

            self._execute_steps(
                steps=steps,
                subjects=subjects,
                task=resolved_task,
                use_icalabel=use_icalabel,
                n_jobs=n_jobs,
                progress=progress,
            )

            progress.complete(success=True)
            run_status = "success"

            return [{
                "subjects": subjects,
                "mode": mode,
                "status": "success",
            }]
        except Exception as exc:
            run_error = str(exc)
            raise
        finally:
            self._write_run_metadata(
                run_context,
                status=run_status,
                error=run_error,
                outputs={},
                summary={
                    "n_subjects": len(subjects),
                    "mode": mode,
                },
            )
    
    def _normalize_subjects(self, subjects: List[str]) -> Union[str, List[str]]:
        """Normalize subjects list to 'all' string if needed."""
        if subjects == ["all"]:
            return "all"
        return subjects
    
    def _get_steps_for_mode(self, mode: str) -> List[str]:
        """Get preprocessing steps for the given mode."""
        mode_steps = {
            "full": [STEP_BAD_CHANNELS, STEP_ICA_FIT, STEP_ICA_LABEL, STEP_EPOCHS, STEP_STATS],
            "bad-channels": [STEP_BAD_CHANNELS],
            "ica": [STEP_ICA_FIT, STEP_ICA_LABEL],
            "epochs": [STEP_EPOCHS, STEP_STATS],
        }
        
        if mode not in mode_steps:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        
        return mode_steps[mode]
    
    def _execute_steps(
        self,
        steps: List[str],
        subjects: List[str],
        task: str,
        use_icalabel: bool,
        n_jobs: int,
        progress: Any,
    ) -> None:
        """Execute preprocessing steps in sequence."""
        total_steps = len(steps)
        
        for i, step in enumerate(steps, 1):
            progress.step(step, current=i, total=total_steps)
            self.logger.info("Running step: %s", step)
            
            if step == STEP_BAD_CHANNELS:
                self._run_bad_channel_detection(
                    subjects=subjects,
                    task=task,
                    n_jobs=n_jobs,
                )
            elif step == STEP_ICA_FIT:
                self._run_ica_fitting(
                    subjects=subjects,
                    task=task,
                    use_icalabel=use_icalabel,
                )
            elif step == STEP_ICA_LABEL:
                if use_icalabel:
                    self._run_ica_labeling(
                        subjects=subjects,
                        task=task,
                    )
            elif step == STEP_EPOCHS:
                self._run_epoch_creation(
                    subjects=subjects,
                    task=task,
                )
            elif step == STEP_STATS:
                self._collect_stats(task=task)
    
    def _run_bad_channel_detection(
        self,
        subjects: List[str],
        task: str,
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
        )
        
        synchronize_bad_channels_across_runs(
            bids_path=str(self.bids_root),
            task=task,
            subjects=normalized_subjects,
        )
        
        self.logger.info("Bad channel detection complete")
    
    def _get_ica_fitting_steps(self, use_icalabel: bool) -> str:
        """Get MNE-BIDS pipeline steps for ICA fitting."""
        base_steps = [
            "init",
            "preprocessing/_01_data_quality",
            "preprocessing/_04_frequency_filter",
            "preprocessing/_05_regress_artifact",
            "preprocessing/_06a1_fit_ica",
        ]
        
        if not use_icalabel:
            base_steps.append("preprocessing/_06a2_find_ica_artifacts")
        
        return ",".join(base_steps)
    
    def _run_ica_fitting(
        self,
        subjects: List[str],
        task: str,
        use_icalabel: bool = True,
    ) -> None:
        """Run ICA fitting via MNE-BIDS pipeline."""
        steps = self._get_ica_fitting_steps(use_icalabel)
        
        self._run_mne_bids_pipeline(steps, subjects=subjects)
        
        self.logger.info("ICA fitting complete")
    
    def _run_ica_labeling(
        self,
        subjects: List[str],
        task: str,
    ) -> None:
        """Run ICA component labeling using mne-icalabel."""
        from eeg_pipeline.preprocessing.pipeline.ica import run_ica_label
        
        normalized_subjects = self._normalize_subjects(subjects)
        subject_count = len(subjects) if isinstance(normalized_subjects, list) else "all"
        self.logger.info("Running ICA labeling for %s subject(s)", subject_count)
        
        icalabel_cfg = self.config.get("icalabel", {})
        run_ica_label(
            pipeline_path=str(self.deriv_root / "preprocessed" / "eeg"),
            task=task,
            subjects=normalized_subjects,
            prob_threshold=icalabel_cfg.get("prob_threshold", self.config.get("ica.probability_threshold", 0.8)),
            labels_to_keep=icalabel_cfg.get("labels_to_keep", self.config.get("ica.labels_to_keep", ["brain", "other"])),
            keep_mnebids_bads=icalabel_cfg.get("keep_mnebids_bads", False),
        )
        
        self.logger.info("ICA labeling complete")
    
    def _run_epoch_creation(
        self,
        subjects: List[str],
        task: str,
    ) -> None:
        """Create epochs and apply ICA via MNE-BIDS pipeline."""
        steps = "preprocessing/_07_make_epochs,preprocessing/_08a_apply_ica,preprocessing/_09_ptp_reject"
        
        self._run_mne_bids_pipeline(steps, subjects=subjects)

        if bool(self.config.get("preprocessing.write_clean_events", True)):
            self._write_clean_events_tsv(subjects=subjects, task=task)
        
        self.logger.info("Epoch creation complete")

    def _resolve_epoch_conditions(self) -> list[str] | None:
        conditions = self.config.get("epochs.conditions")
        if conditions:
            return list(conditions)
        detected = self._detect_conditions_from_bids()
        return list(detected) if detected else None

    def _write_clean_events_tsv(self, *, subjects: List[str], task: str) -> None:
        from eeg_pipeline.infra.paths import find_clean_epochs_path
        from eeg_pipeline.utils.data.preprocessing import write_clean_events_tsv_for_epochs

        conditions = self._resolve_epoch_conditions()
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
                msg = f"Clean epochs not found; cannot write clean events for sub-{subj}, task-{task}"
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
                    conditions=conditions,
                    overwrite=overwrite,
                    _logger=self.logger,
                )
            except Exception as exc:
                msg = f"Failed writing clean events for sub-{subj}, task-{task}: {exc}"
                if strict:
                    raise RuntimeError(msg) from exc
                self.logger.warning(msg)
    
    def _collect_stats(self, task: str) -> None:
        """Collect preprocessing statistics."""
        from eeg_pipeline.preprocessing.pipeline.stats import collect_preprocessing_stats
        
        self.logger.info("Collecting preprocessing statistics")
        
        collect_preprocessing_stats(
            bids_path=str(self.bids_root),
            pipeline_path=str(self.deriv_root / "preprocessed" / "eeg"),
            task=task,
        )
        
        self.logger.info("Statistics collection complete")
    
    def _run_mne_bids_pipeline(self, steps: str, subjects: List[str] = None) -> None:
        """Run MNE-BIDS pipeline with a generated config file.
        
        mne_bids_pipeline requires settings in a Python config file,
        not CLI arguments. This generates a temporary config and passes it
        via --config.
        
        Args:
            steps: MNE-BIDS pipeline steps to run
            subjects: List of subject IDs to process (without 'sub-' prefix)
        """
        import tempfile
        
        config_content = self._generate_mne_bids_config(steps, subjects=subjects)
        
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
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(f"MNE-BIDS pipeline failed: {error_msg}")
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def _generate_mne_bids_config(self, steps: str, subjects: List[str] = None) -> str:
        """Generate Python config file content for mne_bids_pipeline.
        
        Args:
            steps: MNE-BIDS pipeline steps to run
            subjects: List of subject IDs to process (without 'sub-' prefix)
        """
        lines = [
            '"""Auto-generated MNE-BIDS pipeline config."""',
            "",
            f'bids_root = "{self.bids_root}"',
            f'deriv_root = "{self.deriv_root / "preprocessed" / "eeg"}"',
            "",
        ]
        
        # Subject filter (critical to avoid processing all subjects)
        if subjects:
            lines.append(f'subjects = {subjects}')
            lines.append("")
        
        # Channel types
        ch_types = self.config.get("eeg.ch_types", "eeg")
        if ch_types:
            lines.append(f'ch_types = ["{ch_types}"]')
        
        # EEG reference
        eeg_reference = self.config.get("eeg.reference", "average")
        if eeg_reference:
            lines.append(f'eeg_reference = "{eeg_reference}"')
        
        # EOG channels
        eog_channels = self.config.get("eeg.eog_channels")
        if eog_channels:
            if isinstance(eog_channels, list):
                lines.append(f'eog_channels = {eog_channels}')
            elif isinstance(eog_channels, str):
                # Handle comma-separated string
                eog_list = [ch.strip() for ch in eog_channels.split(",") if ch.strip()]
                if eog_list:
                    lines.append(f'eog_channels = {eog_list}')
            else:
                lines.append(f'eog_channels = ["{eog_channels}"]')

        # NOTE: mne-bids-pipeline does not accept an `ecg_channels` config variable.
        # ECG channel typing is handled via BIDS channels.tsv (type=ECG) and MNE.
        
        # Random state
        random_state = self.config.get("preprocessing.random_state", 42)
        if random_state is not None:
            lines.append(f'random_state = {random_state}')
        
        # Task is rest
        task_is_rest = self.config.get("preprocessing.task_is_rest", False)
        lines.append(f'task_is_rest = {task_is_rest}')
        
        lines.append("")
        lines.append("# Filtering")
        
        # Filtering configs
        l_freq = self.config.get("preprocessing.l_freq", 0.1)
        if l_freq is not None:
            lines.append(f'l_freq = {l_freq}')
        
        h_freq = self.config.get("preprocessing.h_freq", 100)
        if h_freq is not None:
            lines.append(f'h_freq = {h_freq}')
        
        notch_freq = self.config.get("preprocessing.notch_freq")
        if notch_freq is not None:
            lines.append(f'notch_freq = {notch_freq}')
        
        # Resampling
        resample_sfreq = self.config.get("preprocessing.resample_freq")
        if resample_sfreq is not None:
            lines.append(f'raw_resample_sfreq = {resample_sfreq}')
        
        # Find breaks
        find_breaks = self.config.get("preprocessing.find_breaks", False)
        lines.append(f'find_breaks = {find_breaks}')
        
        lines.append("")
        lines.append("# ICA")
        
        # Spatial filter
        spatial_filter = self.config.get("ica.spatial_filter", "ica")
        if spatial_filter:
            lines.append(f'spatial_filter = "{spatial_filter}"')
        
        # ICA algorithm
        ica_algorithm = self.config.get("ica.method") or self.config.get("ica.algorithm", "extended_infomax")
        if ica_algorithm:
            lines.append(f'ica_algorithm = "{ica_algorithm}"')
        
        # ICA n_components
        ica_n_components = self.config.get("ica.n_components", 0.99)
        if ica_n_components is not None:
            lines.append(f'ica_n_components = {ica_n_components}')
        
        # ICA l_freq
        ica_l_freq = self.config.get("ica.l_freq", 1.0)
        if ica_l_freq is not None:
            lines.append(f'ica_l_freq = {ica_l_freq}')
        
        # ICA reject
        ica_reject = self.config.get("ica.reject")
        if ica_reject is not None:
            lines.append(f'ica_reject = {ica_reject}')
        
        lines.append("")
        lines.append("# Epochs")
        
        # Conditions (required for non-resting-state)
        conditions = self.config.get("epochs.conditions")
        if conditions:
            lines.append(f'conditions = {conditions}')
        else:
            # Auto-detect conditions from BIDS events files
            detected = self._detect_conditions_from_bids()
            if detected:
                lines.append(f'conditions = {detected}')
            else:
                # Fallback: use None but mne_bids_pipeline may reject this
                self.logger.warning(
                    "No conditions found in config or BIDS events. "
                    "mne_bids_pipeline may fail for non-resting-state data."
                )
        
        # Epoch time window
        epochs_tmin = self.config.get("epochs.tmin", -3.0)
        if epochs_tmin is not None:
            lines.append(f'epochs_tmin = {epochs_tmin}')
        
        epochs_tmax = self.config.get("epochs.tmax", 12.0)
        if epochs_tmax is not None:
            lines.append(f'epochs_tmax = {epochs_tmax}')
        
        # Baseline
        baseline = self.config.get("epochs.baseline")
        if baseline is not None:
            # Convert list to tuple if needed (MNE-BIDS expects tuple)
            if isinstance(baseline, list):
                baseline = tuple(baseline)
            lines.append(f'baseline = {baseline}')
        else:
            lines.append('baseline = None')
        
        # Reject
        reject = self.config.get("epochs.reject")
        reject_method = self.config.get("epochs.reject_method")
        if reject is None and reject_method:
            # CLI sets epochs.reject_method ("none"/"autoreject_local"/"autoreject_global")
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

        # Optional PTP reject time window (ignored by autoreject_local)
        reject_tmin = self.config.get("epochs.reject_tmin")
        if reject_tmin is not None:
            lines.append(f"reject_tmin = {float(reject_tmin)}")
        reject_tmax = self.config.get("epochs.reject_tmax")
        if reject_tmax is not None:
            lines.append(f"reject_tmax = {float(reject_tmax)}")

        # Autoreject local configuration
        ar_n_interp = self.config.get("epochs.autoreject_n_interpolate")
        if ar_n_interp is not None:
            lines.append(f"autoreject_n_interpolate = {ar_n_interp}")
        
        # Source estimation
        run_source_estimation = self.config.get("preprocessing.run_source_estimation", False)
        lines.append(f'run_source_estimation = {run_source_estimation}')
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _detect_conditions_from_bids(self) -> list | None:
        """Detect unique condition names from BIDS events files.
        
        Reads trial_type column from first available events TSV and returns
        unique values as a list suitable for mne_bids_pipeline conditions.
        
        Returns:
            List of unique trial_type values, or None if detection fails.
        """
        events_files = sorted(self.bids_root.glob("sub-*/eeg/*_events.tsv"))
        
        if not events_files:
            self.logger.debug("No events files found in %s", self.bids_root)
            return None
        
        try:
            with open(events_files[0], "r", encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
                if "trial_type" not in header:
                    self.logger.debug("No 'trial_type' column in events file")
                    return None
                
                trial_type_idx = header.index("trial_type")
                conditions = set()
                
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) > trial_type_idx:
                        trial_type = parts[trial_type_idx].strip()
                        if trial_type and trial_type != "n/a":
                            conditions.add(trial_type)

                if not conditions:
                    return None

                # Heuristic filtering:
                # - Prefer task triggers (Trig_therm*) for thermal pain EEG-fMRI runs.
                # - Avoid scanner/housekeeping markers (Volume, Pulse Artifact, SyncStatus, etc.)
                preferred_prefixes = ("Trig_therm", "Trig_")
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
                    self.logger.info("Auto-detected task conditions from BIDS: %s", preferred)
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
                    
        except Exception as e:
            self.logger.debug("Failed to detect conditions: %s", e)
        
        return None


__all__ = ["PreprocessingPipeline"]
