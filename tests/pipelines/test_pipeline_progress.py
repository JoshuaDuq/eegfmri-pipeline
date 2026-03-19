from __future__ import annotations

import unittest

from eeg_pipeline.pipelines.progress import (
    NoOpProgressReporter,
    ensure_progress_reporter,
)


class TestPipelineProgress(unittest.TestCase):
    def test_ensure_progress_reporter_returns_noop_for_none(self) -> None:
        reporter = ensure_progress_reporter(None)

        self.assertIsInstance(reporter, NoOpProgressReporter)

    def test_ensure_progress_reporter_preserves_existing_reporter(self) -> None:
        sentinel = object()

        self.assertIs(ensure_progress_reporter(sentinel), sentinel)

    def test_noop_progress_reporter_accepts_all_calls(self) -> None:
        reporter = NoOpProgressReporter()

        self.assertIsNone(reporter.start("run", ["0001"]))
        self.assertIsNone(reporter.subject_start("0001"))
        self.assertIsNone(reporter.step("step", current=1, total=2))
        self.assertIsNone(reporter.log("info", "message"))
        self.assertIsNone(reporter.subject_done("0001", success=True))
        self.assertIsNone(reporter.complete(success=True, duration=1.2, outputs=["a"]))
        self.assertIsNone(reporter.error("E001", "message", suggestion="fix"))
