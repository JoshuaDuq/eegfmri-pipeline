"""
Progress Reporting Module
=========================

Centralized progress event types and reporting for CLI and TUI communication.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional


_BYTES_PER_GB = 1024 ** 3
_KB_PER_GB = 1024 ** 2


class ProgressEvent(str, Enum):
    """Progress event types for CLI-TUI communication."""
    
    START = "start"
    SUBJECT_START = "subject_start"
    PROGRESS = "progress"
    SUBJECT_DONE = "subject_done"
    COMPLETE = "complete"
    LOG = "log"
    ERROR = "error"


class ProgressReporter:
    """Report progress events as JSON lines for TUI consumption."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.current_subject = None
        self._start_wall_time = None
        self._start_cpu_time = None

    def start(self, operation: str, subjects: List[str]) -> None:
        if not self.enabled:
            return
        self._emit({
            "event": ProgressEvent.START.value,
            "operation": operation,
            "subjects": subjects,
            "total_subjects": len(subjects),
        })

    def subject_start(self, subject: str) -> None:
        if not self.enabled:
            return
        self.current_subject = subject
        self._emit({
            "event": ProgressEvent.SUBJECT_START.value,
            "subject": subject,
        })

    def step(self, step_name: str, current: int = None, total: int = None) -> None:
        if not self.enabled:
            return
        msg = {
            "event": ProgressEvent.PROGRESS.value,
            "step": step_name,
            "subject": self.current_subject,
        }
        if current is not None and total is not None:
            msg["current"] = current
            msg["total"] = total
            msg["pct"] = round(100.0 * current / total) if total > 0 else 0
        self._emit(msg)

    def log(self, level: str, message: str) -> None:
        if not self.enabled:
            return
        self._emit({
            "event": ProgressEvent.LOG.value,
            "level": level,
            "message": message,
            "subject": self.current_subject,
        })

    def subject_done(self, subject: str, success: bool = True) -> None:
        if not self.enabled:
            return
        self._emit({
            "event": ProgressEvent.SUBJECT_DONE.value,
            "subject": subject,
            "success": success,
        })

    def complete(self, success: bool, duration: float = None, outputs: List[str] = None) -> None:
        if not self.enabled:
            return
        self._emit({
            "event": ProgressEvent.COMPLETE.value,
            "success": success,
            "duration": duration,
            "outputs": outputs or [],
        })

    def error(self, code: str, message: str, suggestion: str = None) -> None:
        if not self.enabled:
            return
        msg = {
            "event": ProgressEvent.ERROR.value,
            "code": code,
            "message": message,
        }
        if suggestion:
            msg["suggestion"] = suggestion
        self._emit(msg)

    def _emit(self, data: Dict[str, Any]) -> None:
        usage = self._get_resource_usage()
        data.update(usage)
        print(json.dumps(data), flush=True)

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current CPU and memory usage for the process."""
        try:
            memory_gb = self._get_memory_usage_gb()
            cpu_percent = self._get_cpu_usage_percent()
            return {
                "cpu": round(cpu_percent, 1),
                "memory": round(memory_gb, 2)
            }
        except (OSError, ValueError):
            return {"cpu": 0.0, "memory": 0.0}

    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = usage.ru_maxrss
        
        is_darwin = sys.platform == 'darwin'
        if is_darwin:
            memory_gb = maxrss / _BYTES_PER_GB
        else:
            memory_gb = maxrss / _KB_PER_GB
        
        return memory_gb

    def _get_cpu_usage_percent(self) -> float:
        """Get CPU usage as percentage since reporter initialization."""
        if self._start_wall_time is None:
            self._start_wall_time = time.time()
            self._start_cpu_time = self._get_current_cpu_time()
            return 0.0
        
        elapsed_wall_time = time.time() - self._start_wall_time
        if elapsed_wall_time <= 0:
            return 0.0
        
        current_cpu_time = self._get_current_cpu_time()
        cpu_time_delta = current_cpu_time - self._start_cpu_time
        cpu_percent = (cpu_time_delta / elapsed_wall_time) * 100.0
        
        cpu_count = os.cpu_count() or 1
        max_cpu_percent = 100.0 * cpu_count
        cpu_percent = min(max(cpu_percent, 0.0), max_cpu_percent)
        return cpu_percent

    def _get_current_cpu_time(self) -> float:
        """Get cumulative CPU time (user + system) in seconds."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime


def create_progress_reporter(args) -> ProgressReporter:
    """Create a progress reporter from parsed args."""
    enabled = getattr(args, "progress_json", False)
    return ProgressReporter(enabled=enabled)
