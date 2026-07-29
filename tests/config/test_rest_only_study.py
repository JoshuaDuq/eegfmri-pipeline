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
        f'extends: "rest"\nproject:\n  task: "rest"\npaths:\n  bids_root: "{tmp_path}/bids"\n'
        f'  deriv_root: "{tmp_path}/deriv"\n',
        encoding="utf-8",
    )

    config = load_config(study)
    report = check_config_coherence(config)

    assert report.errors == (), [str(issue) for issue in report.errors]
    assert resolve_eeg_bids_root(config) == tmp_path / "bids"


def test_the_rest_preset_does_not_carry_another_studys_task_label(tmp_path) -> None:
    """It inherited ``thermalactive``. A study extending it selected recordings that its
    own BIDS tree does not contain, and the run found nothing rather than saying so."""
    study = tmp_path / "study.yaml"
    study.write_text('extends: "rest"\n', encoding="utf-8")

    assert load_config(study).get("project.task") is None


def test_the_eeg_only_preset_asks_for_the_task_label_and_nothing_else(tmp_path) -> None:
    from eeg_pipeline.utils.config.coherence import check_config_coherence

    study = tmp_path / "study.yaml"
    study.write_text('extends: "eeg_only"\n', encoding="utf-8")

    report = check_config_coherence(load_config(study))

    assert {issue.key for issue in report.errors} == {"project.task"}
    # And nothing scanner-only is left switched on to warn about.
    assert report.warnings == (), [str(issue) for issue in report.warnings]


###################################################################
# What ``paradigm: rest`` neutralizes in a base written for task epochs
###################################################################


def _rest_study(tmp_path, body: str = "") -> ConfigDict:
    study = tmp_path / "study.yaml"
    study.write_text(
        f'extends: "eeg_only"\nproject:\n  paradigm: "rest"\n  task: "rest"\n{body}',
        encoding="utf-8",
    )
    return load_config(study)


def test_the_two_knob_route_runs_as_written(tmp_path) -> None:
    """``eeg_only`` + ``paradigm: rest`` is the documented way to say "resting-state EEG
    outside a scanner". It reported two errors the user had to clear by hand first, both
    of them values the base config chose for task epochs."""
    from eeg_pipeline.utils.config.coherence import check_config_coherence

    report = check_config_coherence(_rest_study(tmp_path))

    assert report.errors == (), [str(issue) for issue in report.errors]


def test_inherited_event_locked_families_are_dropped(tmp_path) -> None:
    """The base config's list is written for a task acquisition. Left as inherited it
    reached ``features compute`` and raised there — after preprocessing a whole cohort."""
    from eeg_pipeline.analysis.features.rest import REST_INCOMPATIBLE_FEATURE_CATEGORIES

    categories = _rest_study(tmp_path).get("feature_engineering.feature_categories")

    assert set(categories).isdisjoint(REST_INCOMPATIBLE_FEATURE_CATEGORIES)
    # The families that do survive a fixed-length segment are all still there.
    assert {"power", "connectivity", "aperiodic", "microstates"} <= set(categories)


def test_a_family_the_study_asks_for_is_reported_rather_than_removed(tmp_path) -> None:
    """Dropping a key the study's own file names would answer a question with silence.
    Only what arrives through ``extends`` is neutralized."""
    from eeg_pipeline.analysis.features.rest import validate_rest_feature_categories

    config = _rest_study(
        tmp_path,
        "feature_engineering:\n  feature_categories: ['power', 'erp']\n",
    )
    categories = config.get("feature_engineering.feature_categories")

    assert categories == ["power", "erp"]
    with pytest.raises(ValueError, match="erp"):
        validate_rest_feature_categories(categories, config)


def test_a_rest_study_of_only_event_locked_families_names_the_keys(tmp_path) -> None:
    """Trimming the inherited list to nothing would surface downstream as "No feature
    categories specified", which names neither the paradigm nor the families."""
    from eeg_pipeline.utils.config.loader import ConfigError

    base = tmp_path / "base.yaml"
    base.write_text(
        'extends: "eeg_only"\nfeature_engineering:\n  feature_categories: ["erp", "itpc"]\n',
        encoding="utf-8",
    )
    study = tmp_path / "study.yaml"
    study.write_text('extends: "base.yaml"\nproject:\n  paradigm: "rest"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="feature_engineering.feature_categories"):
        load_config(study)


def test_the_band_report_stops_asking_for_a_baseline_it_has_no_event_for(tmp_path) -> None:
    config = _rest_study(tmp_path)

    assert config.get("ica.band_specific_report.tfr.enabled") is False
    assert config.get("ica.band_specific_report.comparisons") == []


def test_task_mode_keeps_everything_the_base_config_asked_for(tmp_path) -> None:
    """The neutralizing runs on the rest branch only. This is the guard that says so."""
    study = tmp_path / "study.yaml"
    study.write_text(
        'extends: "eeg_only"\nproject:\n  paradigm: "task"\n  task: "oddball"\n',
        encoding="utf-8",
    )

    config = load_config(study)
    base = load_config()

    assert config.get("feature_engineering.feature_categories") == base.get(
        "feature_engineering.feature_categories"
    )
    assert config.get("ica.band_specific_report.tfr.enabled") is True


def test_rest_mode_still_needs_the_bids_task_label(tmp_path) -> None:
    """Rest preprocessing tolerates an unset task; feature extraction does not, because
    the cleaned-epochs file is found by its ``task-`` entity. The coherence check said
    the opposite — "or set project.paradigm to 'rest'" — so a rest study that took that
    advice preprocessed fine and then died on a missing key."""
    from eeg_pipeline.utils.config.coherence import check_config_coherence

    study = tmp_path / "study.yaml"
    study.write_text('extends: "eeg_only"\nproject:\n  paradigm: "rest"\n', encoding="utf-8")

    report = check_config_coherence(load_config(study))

    assert {issue.key for issue in report.errors} == {"project.task"}
