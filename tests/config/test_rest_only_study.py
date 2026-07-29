"""A study that acquires only rest has one dataset, one root, and one switch.

The rest-specific roots exist for a study that acquires *both* and keeps them apart. A
rest-only study was made to write the same path twice under a second name, and to set the
same boolean in four places, two pairs of which raise if they disagree.
"""

from __future__ import annotations

import pytest

from eeg_pipeline.utils.config.loader import ConfigDict, load_config
from eeg_pipeline.utils.config.roots import (
    resolve_eeg_bids_root,
    resolve_eeg_deriv_root,
    resolve_resting_state_eeg_mode,
)

_REST_FLAGS = (
    "preprocessing.task_is_rest",
    "feature_engineering.task_is_rest",
    "fmri_preprocessing.task_is_rest",
    "fmri_resting_state.task_is_rest",
)


def _config(**overrides) -> ConfigDict:
    config = load_config()
    for key, value in overrides.items():
        config[key.replace("__", ".")] = value
    return config


def test_rest_mode_falls_back_to_the_primary_roots(tmp_path) -> None:
    config = _config(
        paths__bids_root=str(tmp_path / "bids"),
        paths__deriv_root=str(tmp_path / "deriv"),
        paths__bids_rest_root=None,
        paths__deriv_rest_root=None,
    )

    assert resolve_eeg_bids_root(config, task_is_rest=True) == tmp_path / "bids"
    assert resolve_eeg_deriv_root(config, task_is_rest=True) == tmp_path / "deriv"


def test_a_configured_rest_root_still_wins(tmp_path) -> None:
    """A study acquiring both keeps its two datasets apart, which is what these keys are
    for; the fallback must not take that away."""
    config = _config(
        paths__bids_root=str(tmp_path / "bids"),
        paths__deriv_root=str(tmp_path / "deriv"),
        paths__bids_rest_root=str(tmp_path / "bids_rest"),
        paths__deriv_rest_root=str(tmp_path / "deriv_rest"),
    )

    assert resolve_eeg_bids_root(config, task_is_rest=True) == tmp_path / "bids_rest"
    assert resolve_eeg_deriv_root(config, task_is_rest=True) == tmp_path / "deriv_rest"
    # Task mode is unaffected either way.
    assert resolve_eeg_bids_root(config, task_is_rest=False) == tmp_path / "bids"


def test_an_unset_root_is_reported_rather_than_becoming_the_path_None(tmp_path) -> None:
    """``null`` reached ``str()`` and became the literal path "None", so a missing root
    produced a nonexistent directory instead of the error naming the key."""
    config = _config(paths__bids_root=None, paths__bids_rest_root=None)

    with pytest.raises(ValueError, match="paths.bids_root"):
        resolve_eeg_bids_root(config, task_is_rest=True)


@pytest.mark.parametrize("paradigm,expected", [("rest", True), ("task", False)])
def test_the_paradigm_sets_every_rest_flag_together(tmp_path, paradigm, expected) -> None:
    study = tmp_path / "study.yaml"
    study.write_text(f'extends: "eeg_only"\nproject:\n  paradigm: "{paradigm}"\n', encoding="utf-8")

    config = load_config(study)

    assert [config.get(flag) for flag in _REST_FLAGS] == [expected] * len(_REST_FLAGS)
    assert resolve_resting_state_eeg_mode(config) is expected


def test_the_paradigm_overrides_the_flags_it_inherits(tmp_path) -> None:
    """Without this a preset saying ``paradigm: rest`` would have to restate four
    booleans that the base config had already set to false."""
    study = tmp_path / "study.yaml"
    study.write_text(
        'extends: "eeg_only"\nproject:\n  paradigm: "rest"\n'
        "preprocessing:\n  task_is_rest: false\n",
        encoding="utf-8",
    )

    assert load_config(study).get("preprocessing.task_is_rest") is True


def test_an_unknown_paradigm_names_the_two_that_exist(tmp_path) -> None:
    from eeg_pipeline.utils.config.loader import ConfigError

    study = tmp_path / "study.yaml"
    study.write_text('extends: "eeg_only"\nproject:\n  paradigm: "baseline"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="'task' or 'rest'"):
        load_config(study)


def test_the_rest_preset_is_runnable_as_written(tmp_path) -> None:
    """A preset that still contradicts itself is not a starting point."""
    from eeg_pipeline.utils.config.coherence import check_config_coherence

    study = tmp_path / "study.yaml"
    study.write_text(
        f'extends: "rest"\npaths:\n  bids_root: "{tmp_path}/bids"\n'
        f'  deriv_root: "{tmp_path}/deriv"\n',
        encoding="utf-8",
    )

    config = load_config(study)
    report = check_config_coherence(config)

    assert report.errors == (), [str(issue) for issue in report.errors]
    assert resolve_eeg_bids_root(config) == tmp_path / "bids"


def test_the_eeg_only_preset_asks_for_the_task_label_and_nothing_else(tmp_path) -> None:
    from eeg_pipeline.utils.config.coherence import check_config_coherence

    study = tmp_path / "study.yaml"
    study.write_text('extends: "eeg_only"\n', encoding="utf-8")

    report = check_config_coherence(load_config(study))

    assert {issue.key for issue in report.errors} == {"project.task"}
    # And nothing scanner-only is left switched on to warn about.
    assert report.warnings == (), [str(issue) for issue in report.warnings]
