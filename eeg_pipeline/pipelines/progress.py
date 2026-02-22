"""Pipeline-local progress adapter to avoid CLI-layer dependencies."""

from __future__ import annotations

from typing import Any


class NoOpProgressReporter:
    """No-op reporter used when no progress sink is provided."""

    def start(self, operation: str, subjects: list[str]) -> None:
        return

    def subject_start(self, subject: str) -> None:
        return

    def step(self, step_name: str, current: int | None = None, total: int | None = None) -> None:
        return

    def log(self, level: str, message: str) -> None:
        return

    def subject_done(self, subject: str, success: bool = True) -> None:
        return

    def complete(self, success: bool, duration: float | None = None, outputs: list[str] | None = None) -> None:
        return

    def error(self, code: str, message: str, suggestion: str | None = None) -> None:
        return


def ensure_progress_reporter(progress: Any) -> Any:
    """Return caller-provided reporter or a no-op implementation."""
    if progress is None:
        return NoOpProgressReporter()
    return progress
