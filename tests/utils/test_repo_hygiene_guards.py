from __future__ import annotations

import subprocess

from tests import REPO_ROOT


FORBIDDEN_TRACKED_PREFIXES = (
    "eeg_pipeline/cli/tui/.cache/",
    "eeg_pipeline/cli/tui/.gocache/",
    "eeg_pipeline/cli/tui/.gomodcache/",
)

REMOVED_LEGACY_ENTRYPOINTS = (
    "scripts/eeg_raw_to_bids.py",
    "scripts/merge_psychopy.py",
    "eeg_pipeline/pipelines/eeg_raw_to_bids.py",
    "eeg_pipeline/pipelines/merge_psychopy.py",
    "eeg_pipeline/pipelines/utilities.py",
    "eeg_pipeline/analysis/utilities/eeg_raw_to_bids.py",
    "eeg_pipeline/analysis/utilities/merge_psychopy.py",
    "eeg_pipeline/cli/commands/utilities.py",
    "fmri_pipeline/analysis/raw_to_bids.py",
)


def _tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def test_no_tracked_tui_cache_artifacts() -> None:
    tracked = _tracked_files()
    violations = [
        path
        for path in tracked
        if any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES)
    ]
    assert not violations, (
        "Tracked TUI cache artifacts are not allowed. "
        "Keep build caches out of source tree.\n"
        f"Found: {violations}"
    )


def test_no_tracked_root_organize_ml_script() -> None:
    tracked = _tracked_files()
    tracked_and_present = "organize_ml.py" in tracked and (REPO_ROOT / "organize_ml.py").exists()
    assert not tracked_and_present, (
        "Root-level organize_ml.py should not be tracked. "
        "Use scripts/devtools/organize_ml.py instead."
    )


def test_removed_legacy_scripts_do_not_reappear() -> None:
    tracked = _tracked_files()
    reintroduced = [
        path
        for path in REMOVED_LEGACY_ENTRYPOINTS
        if path in tracked and (REPO_ROOT / path).exists()
    ]
    assert not reintroduced, (
        "Deprecated standalone utility scripts must not be reintroduced. "
        "Use `python paradigm-specific-scripts/run_paradigm_specific.py ...` instead.\n"
        f"Found: {reintroduced}"
    )
