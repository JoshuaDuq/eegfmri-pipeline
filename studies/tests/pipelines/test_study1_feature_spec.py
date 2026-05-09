"""Tests for studies.pain_study.study1.feature_spec family resolution."""

from __future__ import annotations

import pytest

from studies.tests.test_support import DotConfig


def _config(**overrides) -> DotConfig:
    base = {
        "study1": {
            "features": {
                "exploratory_feature_families": [
                    "spectral",
                    "aperiodic",
                    "erds",
                    "ratios",
                    "asymmetry",
                    "complexity",
                    "bursts",
                ],
            }
        }
    }
    base.update(overrides)
    return DotConfig(base)


###################################################################
# resolve_exploratory_feature_families
###################################################################


def test_default_exploratory_families_matches_constant() -> None:
    from studies.pain_study.study1.feature_spec import (
        DEFAULT_EXPLORATORY_FEATURE_FAMILIES,
        resolve_exploratory_feature_families,
    )

    cfg = DotConfig({"study1": {}})
    resolved = resolve_exploratory_feature_families(cfg)

    assert resolved == list(DEFAULT_EXPLORATORY_FEATURE_FAMILIES)


def test_configured_exploratory_families_overrides_defaults() -> None:
    from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = ["spectral", "erds"]
    resolved = resolve_exploratory_feature_families(cfg)

    assert resolved == ["spectral", "erds"]


def test_exploratory_families_rejects_unsupported_family() -> None:
    from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = ["spectral", "made_up"]

    with pytest.raises(ValueError, match="Unsupported"):
        resolve_exploratory_feature_families(cfg)


def test_exploratory_families_deduplicates_entries() -> None:
    from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = [
        "spectral",
        "spectral",
        "erds",
    ]
    resolved = resolve_exploratory_feature_families(cfg)

    assert resolved == ["spectral", "erds"]


def test_exploratory_families_rejects_empty_name() -> None:
    from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = ["spectral", ""]

    with pytest.raises(ValueError, match="empty family name"):
        resolve_exploratory_feature_families(cfg)


def test_exploratory_families_normalizes_case() -> None:
    from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = ["Spectral", "ERDS"]
    resolved = resolve_exploratory_feature_families(cfg)

    assert resolved == ["spectral", "erds"]


def test_exploratory_families_rejects_non_list_value() -> None:
    from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = "spectral"

    with pytest.raises(ValueError, match="list"):
        resolve_exploratory_feature_families(cfg)


###################################################################
# resolve_study1_feature_families
###################################################################


def test_study1_feature_families_always_starts_with_power() -> None:
    from studies.pain_study.study1.feature_spec import (
        PRIMARY_FEATURE_FAMILY,
        resolve_study1_feature_families,
    )

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = ["spectral"]
    resolved = resolve_study1_feature_families(cfg)

    assert resolved[0] == PRIMARY_FEATURE_FAMILY
    assert resolved == ["power", "spectral"]


def test_empty_exploratory_families_returns_only_power() -> None:
    from studies.pain_study.study1.feature_spec import resolve_study1_feature_families

    cfg = _config()
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    resolved = resolve_study1_feature_families(cfg)

    assert resolved == ["power"]
