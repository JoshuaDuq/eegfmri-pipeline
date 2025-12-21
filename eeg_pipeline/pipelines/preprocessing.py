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


###################################################################
# Preprocessing Pipeline Class
###################################################################


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
                - use_pyprep: Whether to use PyPREP for bad channels
                - use_icalabel: Whether to use mne-icalabel
                - n_jobs: Number of parallel jobs
                - progress: ProgressReporter for TUI feedback
        """
        from eeg_pipeline.cli.common import ProgressReporter
        
        task = task or self.config.get("project.task", "thermalactive")
        mode = kwargs.get("mode", "full")
        use_pyprep = kwargs.get("use_pyprep", True)
        use_icalabel = kwargs.get("use_icalabel", True)
        n_jobs = kwargs.get("n_jobs", 1)
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        
        progress.subject_start(f"sub-{subject}")
        
        steps = self._get_steps_for_mode(mode)
        total_steps = len(steps)
        
        for i, step in enumerate(steps, 1):
            progress.step(step, current=i, total=total_steps)
            
            if step == "bad-channels":
                self._run_bad_channel_detection(
                    subjects=[subject],
                    task=task,
                    n_jobs=n_jobs,
                )
            elif step == "ica-fit":
                self._run_ica_fitting(
                    subjects=[subject],
                    task=task,
                    use_icalabel=use_icalabel,
                )
            elif step == "ica-label":
                if use_icalabel:
                    self._run_ica_labeling(
                        subjects=[subject],
                        task=task,
                    )
            elif step == "epochs":
                self._run_epoch_creation(
                    subjects=[subject],
                    task=task,
                )
            elif step == "stats":
                self._collect_stats(task=task)
        
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
        
        task = task or self.config.get("project.task", "thermalactive")
        mode = kwargs.get("mode", "full")
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        
        progress.start("preprocessing", subjects)
        
        steps = self._get_steps_for_mode(mode)
        total_steps = len(steps)
        
        for i, step in enumerate(steps, 1):
            progress.step(step, current=i, total=total_steps)
            self.logger.info(f"Running step: {step}")
            
            if step == "bad-channels":
                self._run_bad_channel_detection(
                    subjects=subjects,
                    task=task,
                    n_jobs=kwargs.get("n_jobs", 1),
                )
            elif step == "ica-fit":
                self._run_ica_fitting(
                    subjects=subjects,
                    task=task,
                    use_icalabel=kwargs.get("use_icalabel", True),
                )
            elif step == "ica-label":
                if kwargs.get("use_icalabel", True):
                    self._run_ica_labeling(
                        subjects=subjects,
                        task=task,
                    )
            elif step == "epochs":
                self._run_epoch_creation(
                    subjects=subjects,
                    task=task,
                )
            elif step == "stats":
                self._collect_stats(task=task)
        
        progress.complete(success=True)
        
        return [{
            "subjects": subjects,
            "mode": mode,
            "status": "success",
        }]
    
    def _get_steps_for_mode(self, mode: str) -> List[str]:
        """Get preprocessing steps for the given mode."""
        if mode == "full":
            return ["bad-channels", "ica-fit", "ica-label", "epochs", "stats"]
        elif mode == "bad-channels":
            return ["bad-channels"]
        elif mode == "ica":
            return ["ica-fit", "ica-label"]
        elif mode == "epochs":
            return ["epochs", "stats"]
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
    
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
        
        self.logger.info(f"Running PyPREP bad channel detection for {len(subjects)} subjects")
        
        run_bads_detection(
            bids_path=str(self.bids_root),
            pipeline_path=str(self.deriv_root),
            task=task,
            subjects=subjects if subjects != ["all"] else "all",
            n_jobs=n_jobs,
            montage=self.config.get("eeg.montage", "easycap-M1"),
            l_pass=self.config.get("preprocessing.h_freq", 100),
            notch=self.config.get("preprocessing.notch_freq"),
        )
        
        synchronize_bad_channels_across_runs(
            bids_path=str(self.bids_root),
            task=task,
            subjects=subjects if subjects != ["all"] else "all",
        )
        
        self.logger.info("Bad channel detection complete")
    
    def _run_ica_fitting(
        self,
        subjects: List[str],
        task: str,
        use_icalabel: bool = True,
    ) -> None:
        """Run ICA fitting via MNE-BIDS pipeline."""
        config_file = self._get_or_create_config(subjects, task)
        
        if use_icalabel:
            steps = "init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06a1_fit_ica"
        else:
            steps = "init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06a1_fit_ica,preprocessing/_06a2_find_ica_artifacts"
        
        self._run_mne_bids_pipeline(config_file, steps)
        
        self.logger.info("ICA fitting complete")
    
    def _run_ica_labeling(
        self,
        subjects: List[str],
        task: str,
    ) -> None:
        """Run ICA component labeling using mne-icalabel."""
        from eeg_pipeline.preprocessing.pipeline.ica import run_ica_label
        
        self.logger.info(f"Running ICA labeling for {len(subjects)} subjects")
        
        run_ica_label(
            pipeline_path=str(self.deriv_root),
            task=task,
            subjects=subjects if subjects != ["all"] else "all",
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
        cmd = [
            sys.executable,
            "-c",
            f"from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config={config_file}', '--steps={steps}']; main()",
        ]
        
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
        subjects_str = repr(subjects) if subjects != ["all"] else '"all"'
        
        l_freq = self.config.get("preprocessing.l_freq", 0.1)
        h_freq = self.config.get("preprocessing.h_freq", 100)
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
subjects = {subjects_str}

ch_types = ["eeg"]
eeg_reference = "average"

l_freq = {l_freq}
h_freq = {h_freq}
notch_freq = {notch_freq}

ica_method = "{ica_method}"
ica_n_components = {ica_n_components}

epochs_tmin = {epochs_tmin}
epochs_tmax = {epochs_tmax}
baseline = {baseline}

reject = {reject}
'''


###################################################################
# Exports
###################################################################

__all__ = ["PreprocessingPipeline"]
