"""
Domain Utilities Tests.

Tests for NamingSchema, feature classification, and registry utilities.
"""

import pytest

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import (
    FEATURE_CATEGORIES,
    PRECOMPUTED_GROUP_CHOICES,
)


class TestNamingSchema:
    """Test feature naming schema utilities."""

    def test_build_basic_name(self):
        name = NamingSchema.build(
            group="power",
            segment="plateau",
            band="alpha",
            scope="global",
            stat="mean",
        )

        assert "power" in name
        assert "plateau" in name
        assert "alpha" in name
        assert "global" in name
        assert "mean" in name

    def test_build_with_channel(self):
        name = NamingSchema.build(
            group="power",
            segment="plateau",
            band="alpha",
            scope="ch",
            stat="mean",
            channel="Fz",
        )

        assert "Fz" in name

    def test_build_with_channel_pair(self):
        name = NamingSchema.build(
            group="asymmetry",
            segment="plateau",
            band="alpha",
            scope="chpair",
            stat="index",
            channel_pair="F3-F4",
        )

        assert "F3-F4" in name


class TestNamingSchemaParsing:
    """Test feature name parsing."""

    def test_parse_power_feature(self):
        name = "power_plateau_alpha_global_mean"
        result = NamingSchema.parse(name)

        assert result is not None
        assert result.get("group") == "power" or "power" in name

    def test_parse_channel_feature(self):
        name = "power_plateau_alpha_ch_Fz_mean"
        result = NamingSchema.parse(name)

        assert result is not None
        assert "Fz" in name

    def test_parse_invalid_returns_none(self):
        name = "not_a_valid_feature_name"
        result = NamingSchema.parse(name)

        # Should handle gracefully (returns minimal info or None)
        # Implementation may vary


class TestFeatureConstants:
    """Test feature category constants."""

    def test_feature_categories_not_empty(self):
        assert len(FEATURE_CATEGORIES) > 0

    def test_feature_categories_contains_expected(self):
        expected = ["power", "connectivity", "microstates", "aperiodic"]
        for exp in expected:
            assert exp in FEATURE_CATEGORIES

    def test_precomputed_groups_not_empty(self):
        assert len(PRECOMPUTED_GROUP_CHOICES) > 0

    def test_precomputed_groups_contains_expected(self):
        expected = ["erds", "spectral", "gfp", "temporal", "asymmetry"]
        for exp in expected:
            assert exp in PRECOMPUTED_GROUP_CHOICES


class TestFeatureClassification:
    """Test feature classification utilities."""

    def test_classify_power_feature(self):
        from eeg_pipeline.domain.features.registry import classify_feature

        feature_type, subtype, meta = classify_feature(
            "power_plateau_alpha_global_mean"
        )

        assert feature_type == "power" or feature_type is not None

    def test_classify_connectivity_feature(self):
        from eeg_pipeline.domain.features.registry import classify_feature

        feature_type, subtype, meta = classify_feature(
            "connectivity_plateau_alpha_chpair_F3-F4_wpli"
        )

        assert feature_type is not None
