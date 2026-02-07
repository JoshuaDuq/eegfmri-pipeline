"""
Progress Tracking Module
========================

Unified progress tracking for long-running pipeline operations.
Provides consistent logging and ETA estimation.

Usage
-----
```python
from eeg_pipeline.utils.progress import PipelineProgress

# Context manager usage
with PipelineProgress(total=10, logger=logger, desc="Processing") as progress:
    for i in range(10):
        # Do work...
        progress.step(f"Completed step {i}")

# Manual usage
progress = PipelineProgress(total=5, logger=logger, desc="Features")
progress.start()
for item in items:
    process(item)
    progress.step(f"Processed {item}")
progress.finish()
```
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List


###################################################################
# Progress Tracker
###################################################################


@dataclass
class PipelineProgress:
    """
    Unified progress tracker for pipeline operations.
    
    Provides:
    - Step counting and logging
    - ETA estimation
    - Duration tracking
    - Consistent formatting
    
    Parameters
    ----------
    total : int
        Total number of steps
    logger : logging.Logger
        Logger instance
    desc : str
        Description of the operation
    log_every : int
        Log progress every N steps (default: 1)
    """
    
    total: int
    logger: logging.Logger
    desc: str = "Progress"
    log_every: int = 1
    
    # State
    current: int = 0
    start_time: float = field(default_factory=time.time)
    
    # Tracking
    _started: bool = False
    _finished: bool = False
    
    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        if self.total <= 0:
            raise ValueError(f"total must be positive, got {self.total}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}")
    
    def start(self) -> "PipelineProgress":
        """Start the progress tracker."""
        self.start_time = time.time()
        self.current = 0
        self._started = True
        self._finished = False
        
        self.logger.info(f"[{self.desc}] Starting ({self.total} steps)")
        return self
    
    def step(self, message: Optional[str] = None) -> None:
        """
        Record completion of a step.
        
        Parameters
        ----------
        message : str, optional
            Description of what was completed
        """
        if not self._started:
            self.start()
        
        self.current += 1
        
        should_log = (
            self.current % self.log_every == 0 or self.current == self.total
        )
        if should_log:
            log_message = self._build_progress_message(message)
            self.logger.info(log_message)
    
    def finish(self) -> float:
        """
        Mark progress as finished and log summary.
        
        Returns
        -------
        float
            Total duration in seconds
        """
        if self._finished:
            return self.elapsed
        
        self._finished = True
        duration = self.elapsed
        
        self.logger.info(
            f"[{self.desc}] Completed {self.current}/{self.total} steps "
            f"in {self._format_duration(duration)}"
        )
        
        return duration
    
    def __enter__(self) -> "PipelineProgress":
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Context manager exit."""
        self.finish()
    
    def _calculate_eta(self) -> float:
        """Calculate estimated time remaining."""
        if self.current == 0:
            return 0.0
        
        elapsed = self.elapsed
        avg_time_per_step = elapsed / self.current
        remaining_steps = self.total - self.current
        return avg_time_per_step * remaining_steps
    
    def _build_progress_message(self, message: Optional[str] = None) -> str:
        """Build formatted progress log message."""
        percentage = self.percent_complete
        log_message = (
            f"[{self.desc}] {self.current}/{self.total} ({percentage:.0f}%)"
        )
        
        eta = self._calculate_eta()
        if eta > 0:
            log_message += f" | ETA: {self._format_duration(eta)}"
        
        if message:
            log_message += f" | {message}"
        
        return log_message
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds_remaining = seconds % 60
            return f"{minutes}m {seconds_remaining:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def percent_complete(self) -> float:
        """Completion percentage (0-100)."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100


###################################################################
# Batch Progress
###################################################################


@dataclass
class BatchProgress:
    """
    Progress tracker for batch operations (multiple subjects).
    
    Provides subject-level progress with per-subject timing.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to process
    logger : logging.Logger
        Logger instance
    desc : str
        Description of the operation
    """
    
    subjects: List[str] = field(default_factory=list)
    logger: Optional[logging.Logger] = None
    desc: str = "Batch"
    
    # State
    current_idx: int = 0
    start_time: float = field(default_factory=time.time)
    subject_times: dict[str, float] = field(default_factory=dict)
    
    _started: bool = False
    
    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        if self.logger is None:
            raise ValueError("logger is required")
    
    def start(self) -> "BatchProgress":
        """Start batch processing."""
        self.start_time = time.time()
        self.current_idx = 0
        self.subject_times = {}
        self._started = True
        
        self.logger.info(f"[{self.desc}] Processing {len(self.subjects)} subjects")
        return self
    
    def start_subject(self, subject: str) -> float:
        """
        Mark start of subject processing.
        
        Returns
        -------
        float
            Start time for this subject
        """
        if not self._started:
            self.start()
        
        self.current_idx += 1
        start_time = time.time()
        
        percentage = (self.current_idx / len(self.subjects)) * 100
        self.logger.info(
            f"[{self.desc}] Processing sub-{subject} "
            f"({self.current_idx}/{len(self.subjects)}, {percentage:.0f}%)"
        )
        
        return start_time
    
    def finish_subject(self, subject: str, start_time: float) -> None:
        """Mark completion of subject processing."""
        duration = time.time() - start_time
        self.subject_times[subject] = duration
        
        self.logger.info(f"[{self.desc}] sub-{subject} completed in {duration:.1f}s")
    
    def finish(self) -> dict[str, float | int]:
        """
        Mark batch as finished.
        
        Returns
        -------
        dict
            Summary statistics with keys: total_duration_s, n_subjects,
            n_completed, mean_duration_s, min_duration_s, max_duration_s
        """
        total_duration = time.time() - self.start_time
        
        times = list(self.subject_times.values())
        n_completed = len(self.subject_times)
        n_subjects = len(self.subjects)
        
        summary = {
            "total_duration_s": total_duration,
            "n_subjects": n_subjects,
            "n_completed": n_completed,
            "mean_duration_s": sum(times) / n_completed if n_completed > 0 else 0.0,
            "min_duration_s": min(times) if times else 0.0,
            "max_duration_s": max(times) if times else 0.0,
        }
        
        mean_duration = summary["mean_duration_s"]
        self.logger.info(
            f"[{self.desc}] Batch completed: {n_completed}/{n_subjects} subjects "
            f"in {PipelineProgress._format_duration(total_duration)} "
            f"(avg {mean_duration:.1f}s/subject)"
        )
        
        return summary
    
    def __enter__(self) -> "BatchProgress":
        return self.start()
    
    def __exit__(self, *args) -> None:
        self.finish()
