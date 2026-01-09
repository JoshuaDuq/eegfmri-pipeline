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

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.pipelines.base import PipelineBase

logger = logging.getLogger(__name__)


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
        from eeg_pipeline.cli.common import ProgressReporter
        
        resolved_task = task or self.config.get("project.task", "thermalactive")
        mode = kwargs.get("mode", "full")
        use_icalabel = kwargs.get("use_icalabel", True)
        n_jobs = kwargs.get("n_jobs", 1)
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        
        progress.subject_start(f"sub-{subject}")
        
        steps = self._get_steps_for_mode(mode)
        subjects = [subject]
        
        self._execute_steps(
            steps=steps,
            subjects=subjects,
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
            **kwargs: Preprocessing options (see process_subject)
            
        Returns:
            List of per-subject status dictionaries
        """
        from eeg_pipeline.cli.common import ProgressReporter
        
        resolved_task = task or self.config.get("project.task", "thermalactive")
        mode = kwargs.get("mode", "full")
        use_icalabel = kwargs.get("use_icalabel", True)
        n_jobs = kwargs.get("n_jobs", 1)
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        
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
        
        return [{
            "subjects": subjects,
            "mode": mode,
            "status": "success",
        }]
    
    def _normalize_subjects(self, subjects: List[str]) -> str | List[str]:
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
            self.logger.info(f"Running step: {step}")
            
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
        self.logger.info(f"Running PyPREP bad channel detection for {subject_count} subjects")
        
        run_bads_detection(
            bids_path=str(self.bids_root),
            pipeline_path=str(self.deriv_root),
            task=task,
            subjects=normalized_subjects,
            n_jobs=n_jobs,
            montage=self.config.get("eeg.montage", "easycap-M1"),
            l_pass=self.config.get("preprocessing.h_freq", 100),
            notch=self.config.get("preprocessing.notch_freq"),
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
        config_file = self._get_or_create_config(subjects, task)
        steps = self._get_ica_fitting_steps(use_icalabel)
        
        self._run_mne_bids_pipeline(config_file, steps)
        
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
        self.logger.info(f"Running ICA labeling for {subject_count} subjects")
        
        run_ica_label(
            pipeline_path=str(self.deriv_root),
            task=task,
            subjects=normalized_subjects,
            prob_threshold=self.config.get("ica.probability_threshold", 0.8),
            labels_to_keep=self.config.get("ica.labels_to_keep", ["brain", "other"]),
        )
        
        self.logger.info("ICA labeling complete")
    
    def _run_epoch_creation(
        self,
        subjects: List[str],
        task: str,
    ) -> None:
        """Create epochs and apply ICA via MNE-BIDS pipeline."""
        config_file = self._get_or_create_config(subjects, task)
        
        steps = "preprocessing/_07_make_epochs,preprocessing/_08a_apply_ica,preprocessing/_09_ptp_reject"
        
        self._run_mne_bids_pipeline(config_file, steps)
        
        self.logger.info("Epoch creation complete")
    
    def _collect_stats(self, task: str) -> None:
        """Collect preprocessing statistics."""
        from eeg_pipeline.preprocessing.pipeline.stats import collect_preprocessing_stats
        
        self.logger.info("Collecting preprocessing statistics")
        
        collect_preprocessing_stats(
            bids_path=str(self.bids_root),
            pipeline_path=str(self.deriv_root),
            task=task,
        )
        
        self.logger.info("Statistics collection complete")
    
    def _run_mne_bids_pipeline(self, config_file: str, steps: str) -> None:
        """Run MNE-BIDS pipeline with specified steps."""
        python_code = (
            "from mne_bids_pipeline._main import main; "
            "import sys; "
            f"sys.argv = ['mne_bids_pipeline', '--config={config_file}', '--steps={steps}']; "
            "main()"
        )
        
        cmd = [sys.executable, "-c", python_code]
        
        self.logger.info(f"Running MNE-BIDS pipeline: {steps}")
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        
        if result.stdout:
            self.logger.debug(f"MNE-BIDS stdout: {result.stdout}")
        if result.stderr:
            self.logger.warning(f"MNE-BIDS stderr: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"MNE-BIDS pipeline failed: {error_msg}")
    
    def _get_or_create_config(self, subjects: List[str], task: str) -> str:
        """Get or create MNE-BIDS pipeline config file."""
        config_dir = self.deriv_root / "logs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"preprocessing_config_{task}.py"
        
        if not config_file.exists():
            config_content = self._generate_config(subjects, task)
            config_file.write_text(config_content)
            
        return str(config_file)
    
    def _generate_config(self, subjects: List[str], task: str) -> str:
        """Generate MNE-BIDS pipeline configuration."""
        normalized_subjects = self._normalize_subjects(subjects)
        subjects_repr = repr(normalized_subjects) if isinstance(normalized_subjects, list) else '"all"'
        
        low_freq = self.config.get("preprocessing.l_freq", 0.1)
        high_freq = self.config.get("preprocessing.h_freq", 100)
        notch_freq = self.config.get("preprocessing.notch_freq", 60)
        ica_method = self.config.get("ica.method", "fastica")
        ica_n_components = self.config.get("ica.n_components", 0.99)
        epochs_tmin = self.config.get("epochs.tmin", -3.0)
        epochs_tmax = self.config.get("epochs.tmax", 12.0)
        baseline = self.config.get("epochs.baseline", None)
        reject = self.config.get("epochs.reject", None)
        
        return f'''"""
MNE-BIDS Pipeline Configuration
Auto-generated by PreprocessingPipeline
"""

bids_root = r"{self.bids_root}"
deriv_root = r"{self.deriv_root}"

task = "{task}"
subjects = {subjects_repr}

ch_types = ["eeg"]
eeg_reference = "average"

l_freq = {low_freq}
h_freq = {high_freq}
notch_freq = {notch_freq}

ica_method = "{ica_method}"
ica_n_components = {ica_n_components}

epochs_tmin = {epochs_tmin}
epochs_tmax = {epochs_tmax}
baseline = {baseline}

reject = {reject}
'''

__all__ = ["PreprocessingPipeline"]
