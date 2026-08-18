# The study's own config must load, inherit from core, and win where it disagrees.

from __future__ import annotations

from pathlib import Path

import yaml

from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.utils.config.loader import load_config

STUDY_CONFIG = Path("studies/pain_study/config/pain_study.yaml")
CORE_CONFIG = Path("eeg_pipeline/utils/config/eeg_config.yaml")

#: Harmonics of the scanner's volume rate withheld from the aperiodic fit, one window per
#: tooth across the 1-100 Hz spectrum grid. Pinned because the count is a property of the
#: acquisition, not a style choice: changing it moves every published aperiodic exponent,
#: so it should take a deliberate edit here rather than pass unnoticed.
EXPECTED_COMB_WINDOWS = 89


def test_the_study_config_loads():
    assert STUDY_CONFIG.exists()
    assert load_config(STUDY_CONFIG) is not None


def test_it_inherits_keys_it_does_not_set():
    config = load_config(STUDY_CONFIG)
    # Set in core, not overridden here, so inheritance is what supplies it.
    assert config.get("report.analysis.aperiodic_fit_range_hz", None) == [2.0, 45.0]


def test_the_comb_exclusion_reaches_the_report_through_the_loader():
    # Resolved through load_config and ReportSettings -- the path that produces the
    # numbers -- rather than read off the YAML. A file-reading check passes on a config
    # that is loaded by nothing, which is exactly how these windows spent a commit sitting
    # in scripts/config/thermal_pain_eeg_overrides.yaml: it parsed, it was in the shipped
    # configs test, and every aperiodic fit still ran with an empty exclusion.
    study = ReportSettings.from_config(load_config(STUDY_CONFIG))

    assert len(study.aperiodic_exclude_hz) == EXPECTED_COMB_WINDOWS
    assert all(low < high for low, high in study.aperiodic_exclude_hz)
    # Non-empty is not enough: a list that missed the fitted band entirely would withhold
    # nothing, and the fit would silently run across the comb as if unconfigured.
    fit_low, fit_high = study.aperiodic_fit_range_hz
    assert any(
        low <= fit_high and high >= fit_low for low, high in study.aperiodic_exclude_hz
    ), "no configured window overlaps the aperiodic fit range"


def test_core_withholds_nothing_without_a_study_saying_so():
    # The other half of the test above. Both configs resolving the same thing would mean
    # the study's own value is not what is being read.
    assert not ReportSettings.from_config(load_config()).aperiodic_exclude_hz


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
