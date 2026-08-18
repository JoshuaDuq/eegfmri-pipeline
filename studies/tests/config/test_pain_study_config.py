# The study's own config must load, inherit from core, and win where it disagrees.

from __future__ import annotations

from pathlib import Path

import yaml

from eeg_pipeline.utils.config.loader import load_config

STUDY_CONFIG = Path("studies/pain_study/config/pain_study.yaml")
CORE_CONFIG = Path("eeg_pipeline/utils/config/eeg_config.yaml")


def test_the_study_config_loads():
    assert STUDY_CONFIG.exists()
    assert load_config(STUDY_CONFIG) is not None


def test_it_inherits_keys_it_does_not_set():
    config = load_config(STUDY_CONFIG)
    # Set in core, not overridden here, so inheritance is what supplies it.
    assert config.get("report.analysis.aperiodic_fit_range_hz", None) == [2.0, 45.0]


def test_it_names_the_study_data_roots():
    config = load_config(STUDY_CONFIG)
    for key in ("paths.bids_root", "paths.deriv_root", "paths.decomb_manifest"):
        assert config.get(key, None), f"{key} must be set by the study config"


def test_it_inherits_paths_it_does_not_override():
    # paths: is a block the study only PARTIALLY overrides (4 of core's keys). A merge
    # that replaced paths: wholesale instead of merging into it -- a shallow dict.update,
    # or a restructured pain_study.yaml that repeats every core key -- would pass both
    # tests above and still silently drop every sibling key (freesurfer_dir, source_data,
    # ...) that a study run still needs. The sibling set is read from the files rather
    # than hardcoded, so this keeps checking something if either file's keys change.
    core_paths = yaml.safe_load(CORE_CONFIG.read_text(encoding="utf-8"))["paths"]
    study_paths = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))["paths"]
    untouched_keys = sorted(set(core_paths) - set(study_paths))
    assert untouched_keys, "expected at least one core paths key the study does not override"

    core = load_config()
    study = load_config(STUDY_CONFIG)
    unset = object()
    for key in untouched_keys:
        dotted = f"paths.{key}"
        assert study.get(dotted, unset) == core.get(dotted, unset), (
            f"{dotted} should be inherited unchanged from core "
            f"(core={core.get(dotted, unset)!r}, study={study.get(dotted, unset)!r})"
        )
