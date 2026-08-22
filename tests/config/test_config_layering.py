"""A study configures the pipeline with its own file, not by editing the package.

Before this, ``load_config()`` read one path — inside the installed package — with no CLI
flag and no environment variable pointing anywhere else. Configuring a second study meant
editing the first study's settings in place.
"""

from __future__ import annotations

import textwrap

import pytest

from eeg_pipeline.utils.config.loader import (
    ConfigError,
    get_presets_dir,
    load_config,
    set_default_config_path,
)


@pytest.fixture(autouse=True)
def _restore_default_config_path():
    yield
    set_default_config_path(None)


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_a_study_config_inherits_everything_it_does_not_name(tmp_path) -> None:
    """The point of ``extends``: a study differing in three keys states three keys.

    Copying the packaged config instead would freeze 1200 lines of scientific defaults at
    the version that was current on the day it was copied.
    """
    study = _write(
        tmp_path,
        "study.yaml",
        """
        extends: "../eeg_config.yaml"
        project:
          task: "oddball"
        """,
    )
    # The relative extends is resolved against the extending file, so point it at the
    # packaged config from wherever tmp_path happens to be.
    study.write_text(
        f'extends: "{get_presets_dir().parent / "eeg_config.yaml"}"\n'
        'project:\n  task: "oddball"\n',
        encoding="utf-8",
    )

    config = load_config(study)

    assert config.get("project.task") == "oddball"
    # Untouched keys still come from the base.
    assert config.get("preprocessing.resample_freq") == 500
    assert config.get("ica.use_icalabel") is True


def test_a_named_preset_resolves_without_a_path(tmp_path) -> None:
    study = _write(
        tmp_path,
        "study.yaml",
        """
        extends: "rest"
        project:
          task: "oddball"
        """,
    )

    config = load_config(study)

    # A value only the preset supplies: the packaged base says "task".
    assert config.get("project.paradigm") == "rest"
    assert config.get("project.task") == "oddball"


def test_an_unknown_preset_names_the_ones_that_exist(tmp_path) -> None:
    study = _write(tmp_path, "study.yaml", 'extends: "eeg_onlyy"\n')

    with pytest.raises(ConfigError, match="Available presets"):
        load_config(study)


def test_a_circular_extends_is_reported_rather_than_recursing(tmp_path) -> None:
    _write(tmp_path, "a.yaml", 'extends: "b.yaml"\n')
    _write(tmp_path, "b.yaml", 'extends: "a.yaml"\n')

    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_the_environment_variable_selects_the_config(tmp_path, monkeypatch) -> None:
    study = _write(
        tmp_path,
        "study.yaml",
        """
        extends: "rest"
        project:
          task: "envtask"
        """,
    )
    monkeypatch.setenv("EEG_PIPELINE_CONFIG", str(study))

    assert load_config().get("project.task") == "envtask"


def test_a_missing_config_path_is_rejected_rather_than_silently_ignored(tmp_path) -> None:
    """Falling back to the packaged default here would run someone else's study."""
    with pytest.raises(ConfigError, match="not found"):
        set_default_config_path(tmp_path / "absent.yaml")


def test_the_process_default_redirects_argument_less_loads(tmp_path) -> None:
    """Most of the pipeline reaches configuration through ``load_config()`` with no
    argument, so this is what makes ``--config`` reach those call sites."""
    study = _write(
        tmp_path,
        "study.yaml",
        """
        extends: "rest"
        project:
          task: "redirected"
        """,
    )

    set_default_config_path(study)

    assert load_config().get("project.task") == "redirected"
