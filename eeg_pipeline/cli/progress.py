"""
Progress Reporting Module
=========================

Centralized progress event types and reporting for CLI and TUI communication.
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Any, Dict, List, Optional


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

    def __init__(self, enabled: bool = False, total_steps: int = 0):
        self.enabled = enabled
        self.total_steps = total_steps
        self.current_step = 0
        self.current_subject = None

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
        self.current_step += 1
        msg = {
            "event": ProgressEvent.PROGRESS.value,
            "step": step_name,
            "subject": self.current_subject,
        }
        if current is not None and total is not None:
            msg["current"] = current
            msg["total"] = total
            msg["pct"] = round(100 * current / total) if total > 0 else 0
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
        # Inject resource usage if not already present
        if "cpu" not in data or "memory" not in data:
            usage = self._get_resource_usage()
            data.update(usage)
        print(json.dumps(data), flush=True)

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current CPU and Memory usage for the process."""
        try:
            import os
            import resource
            import time
            
            # Memory usage in GB
            # ru_maxrss is in bytes on macOS, KB on Linux
            mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == 'darwin':
                mem_gb = mem_bytes / (1024**3)
            else:
                mem_gb = mem_bytes / (1024**2) # KB to GB
            
            # CPU usage estimate (Total CPU time / Wall time since reporter start)
            # This is a cumulative average, but better than nothing
            if not hasattr(self, '_start_time_res'):
                self._start_time_res = time.time()
                self._start_cpu_res = sum(os.times()[:2])
                return {"cpu": 0.0, "memory": mem_gb}
            
            elapsed = time.time() - self._start_time_res
            if elapsed > 0:
                cpu_delta = sum(os.times()[:2]) - self._start_cpu_res
                cpu_pct = (cpu_delta / elapsed) * 100
                # Clamp to reasonable range
                cpu_pct = min(max(cpu_pct, 0.0), 100.0 * os.cpu_count())
            else:
                cpu_pct = 0.0
                
            return {
                "cpu": round(cpu_pct, 1),
                "memory": round(mem_gb, 2)
            }
        except Exception:
            return {"cpu": 0.0, "memory": 0.0}


def create_progress_reporter(args) -> ProgressReporter:
    """Create a progress reporter from parsed args."""
    enabled = getattr(args, "progress_json", False)
    return ProgressReporter(enabled=enabled)
