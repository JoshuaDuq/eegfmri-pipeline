from __future__ import annotations

import subprocess

from tests import REPO_ROOT


FORBIDDEN_TRACKED_PREFIXES = (
    "eeg_pipeline/cli/tui/.cache/",
    "eeg_pipeline/cli/tui/.gocache/",
    "eeg_pipeline/cli/tui/.gomodcache/",
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
