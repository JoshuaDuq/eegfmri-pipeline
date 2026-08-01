"""Checking a YAML must not require the packages that analyze data.

Reported in issue #14: ``validate --config-only`` entered the general CLI import graph,
so in a minimal environment it failed on Seaborn before it ever looked at the config. The
command that exists to tell a new user their setup is wrong was unrunnable for exactly
the user who needed it, and the failure named a plotting package rather than anything
about their study.

Nothing here measures speed. The assertion is about what a config check is *allowed to
depend on*, which is a property of the import graph and does not drift with hardware.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests import REPO_ROOT

#: Import roots that mean data analysis rather than configuration. MNE reads recordings,
#: scikit-learn and Torch fit models, Nilearn handles volumes, and Matplotlib and Seaborn
#: draw. A config check needs none of them, and a user validating a study config should
#: not have to install them first.
_ANALYSIS_ONLY_PACKAGES = frozenset(
    {"matplotlib", "seaborn", "mne", "sklearn", "nilearn", "torch", "numba"}
)

_PROBE = """
import runpy
import sys

sys.argv = ["eeg-pipeline", "--config", sys.argv[1], "validate", "--config-only"]
try:
    runpy.run_module("eeg_pipeline", run_name="__main__")
except SystemExit:
    pass

roots = sorted({name.partition(".")[0] for name in sys.modules})
sys.stderr.write("IMPORTED:" + ",".join(roots) + "\\n")
"""


def _imported_roots(config_path) -> frozenset:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    marker = [line for line in result.stderr.splitlines() if line.startswith("IMPORTED:")]
    if not marker:
        pytest.fail(
            f"probe did not report its imports.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return frozenset(marker[-1][len("IMPORTED:") :].split(","))


@pytest.fixture
def study_config(tmp_path):
    config = tmp_path / "study.yaml"
    config.write_text(
        'extends: "eeg_only"\n'
        "project:\n"
        '  task: "oddball"\n'
        "paths:\n"
        f'  bids_root: "{tmp_path}/bids"\n'
        f'  deriv_root: "{tmp_path}/derivatives"\n',
        encoding="utf-8",
    )
    return config


def test_config_only_validation_does_not_import_the_analysis_stack(study_config) -> None:
    imported = _imported_roots(study_config)

    assert not (imported & _ANALYSIS_ONLY_PACKAGES), (
        "validate --config-only imported "
        f"{', '.join(sorted(imported & _ANALYSIS_ONLY_PACKAGES))}. A config check must "
        "run on the configuration dependencies alone."
    )


def test_config_only_validation_still_reaches_the_config(study_config) -> None:
    """The guard above is satisfied trivially by a command that does nothing, so this
    pins that the run still loaded the config layer it is supposed to check."""
    imported = _imported_roots(study_config)

    assert "eeg_pipeline" in imported
    assert "yaml" in imported


def test_every_command_is_still_registered() -> None:
    """Building one subparser instead of twelve must not change what the CLI offers."""
    from eeg_pipeline.cli.commands import get_command, get_command_names

    names = get_command_names()

    assert {"validate", "preprocessing", "features", "info", "ml"} <= set(names)
    assert len(names) == len(set(names)), f"duplicate command names: {names}"
    assert all(get_command(name) is not None for name in names)


def test_an_uninstalled_analysis_package_is_reported_as_an_environment_problem() -> None:
    """A missing package and a wrong config both used to arrive as 'Error running X',
    with a traceback through the import machinery. They need different fixes, so the
    message has to say which one this is."""
    from eeg_pipeline.cli.commands import MissingCommandDependency, _deferred

    call = _deferred("a_third_party_package_that_is_not_installed", "run_thing", "features")

    with pytest.raises(MissingCommandDependency) as excinfo:
        call()

    message = str(excinfo.value)
    assert "'features' command" in message
    assert "not installed" in message
    assert "validate --config-only" in message


def test_a_broken_first_party_import_keeps_its_traceback() -> None:
    """Only a missing *dependency* is an environment problem. A first-party module that
    fails to import is a bug, and dressing it up as 'please pip install' would send the
    reader to the wrong place entirely."""
    from eeg_pipeline.cli.commands import MissingCommandDependency, _deferred

    call = _deferred("eeg_pipeline.cli.commands.validate_parser", "setup_validate", "validate")

    # Sanity: this one imports fine, so the guard below is about the failure mode only.
    assert callable(call)

    broken = _deferred("eeg_pipeline.does_not_exist", "anything", "validate")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        broken()

    assert not isinstance(excinfo.value, MissingCommandDependency)
