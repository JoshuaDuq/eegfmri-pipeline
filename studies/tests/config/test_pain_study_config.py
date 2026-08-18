# The study's own config must load, inherit from core, and win where it disagrees.

from __future__ import annotations

from pathlib import Path

from eeg_pipeline.utils.config.loader import load_config

STUDY_CONFIG = Path("studies/pain_study/config/pain_study.yaml")


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
