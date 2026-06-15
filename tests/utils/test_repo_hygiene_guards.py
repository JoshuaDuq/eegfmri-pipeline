from __future__ import annotations

import subprocess
import tomllib

import yaml

from tests import REPO_ROOT


FORBIDDEN_TRACKED_PREFIXES = (
    "eeg_pipeline/cli/tui/.cache/",
    "eeg_pipeline/cli/tui/.gocache/",
    "eeg_pipeline/cli/tui/.gomodcache/",
)

REMOVED_LEGACY_ENTRYPOINTS = (
    "paradigm-specific-scripts/run_paradigm_specific.py",
    "paradigm-specific-scripts/eeg_raw_to_bids.py",
    "paradigm-specific-scripts/fmri_raw_to_bids.py",
    "paradigm-specific-scripts/merge_psychopy.py",
    "paradigm-specific-scripts/fix_fmri_bids_outputs.py",
    "eeg_pipeline/cli/commands/coupling.py",
    "eeg_pipeline/analysis/paradigms/pain/eeg_bold_coupling.py",
    "eeg_pipeline/pipelines/paradigms/pain/eeg_bold_coupling.py",
    "eeg_pipeline/utils/config/paradigms/pain/eeg_bold_coupling_loader.py",
    "scripts/eeg_raw_to_bids.py",
    "scripts/merge_psychopy.py",
    "eeg_pipeline/pipelines/eeg_raw_to_bids.py",
    "eeg_pipeline/pipelines/merge_psychopy.py",
    "eeg_pipeline/pipelines/utilities.py",
    "eeg_pipeline/analysis/utilities/eeg_raw_to_bids.py",
    "eeg_pipeline/analysis/utilities/merge_psychopy.py",
    "eeg_pipeline/cli/commands/utilities.py",
    "fmri_pipeline/analysis/raw_to_bids.py",
    "eeg_pipeline/preprocessing/pipeline/config.py",
    "eeg_pipeline/preprocessing/scripts/create_config.py",
    "eeg_pipeline/preprocessing/scripts/run_pipeline.py",
    "eeg_pipeline/preprocessing/scripts/__init__.py",
)

REMOVED_WORKTREE_PATHS = (
    "paradigm-specific-scripts",
)

REQUIRED_SOURCE_CONTROL_PATHS = (
    "studies/pain_study/study2/alliance/lib/study2_alliance_common.sh",
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
        "Keep study-specific entrypoints in the private studies package instead.\n"
        f"Found: {reintroduced}"
    )


def test_removed_legacy_worktree_paths_do_not_exist() -> None:
    present = [path for path in REMOVED_WORKTREE_PATHS if (REPO_ROOT / path).exists()]
    assert not present, (
        "Legacy pain-study worktree paths must not exist. "
        "Do not keep study-specific worktrees in the public repository.\n"
        f"Found: {present}"
    )


def test_required_source_control_paths_are_not_ignored() -> None:
    ignored_paths = []
    for path in REQUIRED_SOURCE_CONTROL_PATHS:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=str(REPO_ROOT),
            check=False,
        )
        if proc.returncode == 0:
            ignored_paths.append(path)

    assert not ignored_paths, (
        "Required source-control paths must not be ignored. "
        "CI checks out only tracked, non-ignored project files.\n"
        f"Found: {ignored_paths}"
    )


def test_ruff_configuration_is_not_f821_only() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    select = list(config["tool"]["ruff"]["lint"]["select"])

    assert select != ["F821"], (
        "Ruff cannot be configured as an F821-only gate. "
        "Enable a broader static quality baseline."
    )
    assert "F" in select, "Ruff should enable the broader Pyflakes rule family."


def test_requirements_installs_full_test_stack() -> None:
    requirements_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert ".[dev,ml]" in requirements_text or ".[ml,dev]" in requirements_text, (
        "requirements.txt must install both dev and ml extras so a fresh environment "
        "can run the full pytest suite."
    )


def test_targeted_pytest_invocations_do_not_inherit_coverage_gate() -> None:
    pytest_config = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8")

    assert "--cov-fail-under" not in pytest_config

    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    windows_steps = workflow["jobs"]["windows-compat"]["steps"]
    targeted_step = next(
        step for step in windows_steps if step.get("name") == "Run targeted Windows compatibility tests"
    )

    assert "--no-cov" in targeted_step["run"]


def test_full_ci_pytest_invocation_enforces_pipeline_coverage() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    test_steps = workflow["jobs"]["test"]["steps"]
    run_tests_step = next(step for step in test_steps if step.get("name") == "Run tests")

    assert "--cov=eeg_pipeline/pipelines" in run_tests_step["run"]
    assert "--cov=fmri_pipeline/pipelines" in run_tests_step["run"]
    assert "--cov-fail-under=75" in run_tests_step["run"]
