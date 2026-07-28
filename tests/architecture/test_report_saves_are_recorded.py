"""Every write of a subject report has to record which stage wrote it.

The build record is only as complete as the stages that report themselves, and a panel
listing four of five stages is worse than no panel: it implies the fifth did not run.
The recording therefore lives inside ``save_subject_report``, and this test is what stops
a new stage from saving around it.
"""

from __future__ import annotations

import re

import pytest

from tests import REPO_ROOT

#: Modules that append sections to a subject report and save it again.
REPORT_WRITING_MODULES = (
    "eeg_pipeline/pipelines/preprocessing.py",
    "eeg_pipeline/preprocessing/band_ica_report.py",
    "eeg_pipeline/preprocessing/ica_cardiac_report.py",
    "eeg_pipeline/preprocessing/ica_ocular_report.py",
)

#: A direct save of the subject report, which bypasses the record.
_DIRECT_SAVE = re.compile(r"\breport\.save\s*\(")


@pytest.mark.parametrize("module", REPORT_WRITING_MODULES)
def test_a_subject_report_is_never_saved_around_the_build_record(module: str) -> None:
    source = (REPO_ROOT / module).read_text(encoding="utf-8")

    offenders = [line.strip() for line in source.splitlines() if _DIRECT_SAVE.search(line)]

    assert offenders == [], (
        f"{module} saves a subject report directly, so the stage that wrote those "
        "sections will be missing from the build record. Use "
        "report.build_record.save_subject_report instead."
    )
