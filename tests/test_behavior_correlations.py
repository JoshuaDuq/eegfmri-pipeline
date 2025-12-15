import logging

import numpy as np
import pandas as pd

import eeg_pipeline.analysis.behavior.feature_correlator as feature_correlator
from eeg_pipeline.analysis.behavior.feature_correlator import CorrelationConfig, FeatureBehaviorCorrelator
from eeg_pipeline.utils.analysis.stats.correlation import correlate_features_loop, run_pain_sensitivity_correlations
from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr


def test_run_pain_sensitivity_aligns_and_masks_inputs():
    features_df = pd.DataFrame({"feat": [0, 1, 2, 3, 4, 5]}, index=list("abcdef"))
    ratings = pd.Series([10, 20, 30, 40, 50], index=list("bcdef"))
    temperatures = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0], index=list("bcdef"))

    result = run_pain_sensitivity_correlations(
        features_df, ratings, temperatures, method="spearman", min_samples=3
    )

    assert not result.empty
    assert result.loc[0, "feature"] == "feat"
    # Only the overlapping b-f rows should contribute (5 samples, not 6)
    assert result.loc[0, "n"] == 5


def test_correlate_df_uses_bootstrap_and_permutation(monkeypatch):
    correlator = FeatureBehaviorCorrelator.__new__(FeatureBehaviorCorrelator)
    correlator.logger = logging.getLogger("test")
    correlator.config = {}
    correlator.registry = None

    df = pd.DataFrame({"feat1": np.arange(6)}, index=np.arange(6))
    targets = pd.Series(np.arange(0, 12, 2), index=np.arange(6))

    config = CorrelationConfig(
        method="spearman",
        min_samples=3,
        n_bootstrap=30,
        n_permutations=20,
        rng=np.random.default_rng(0),
    )

    def fake_classify_feature(column, source_file_type=None, include_subtype=True, registry=None):
        return "power", "sub", {"identifier": column, "band": "alpha"}

    monkeypatch.setattr(feature_correlator, "classify_feature", fake_classify_feature)

    result = correlator._correlate_df(df, targets, config, "power")
    records_df = result.to_dataframe()

    assert not records_df.empty
    assert np.isfinite(records_df.loc[0, "ci_low"])
    assert np.isfinite(records_df.loc[0, "ci_high"])
    assert np.isfinite(records_df.loc[0, "p_perm"])


def test_apply_global_fdr_respects_include_glob(tmp_path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()

    corr_df = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "p_primary": [0.01, 0.2],
            "target": ["rating", "rating"],
            "feature_type": ["power", "power"],
        }
    )
    other_df = pd.DataFrame({"hedges_g": [0.1], "p_value": [0.5]})

    corr_path = stats_dir / "corr_stats_power_vs_rating.tsv"
    other_path = stats_dir / "condition_effects.tsv"

    corr_df.to_csv(corr_path, sep="\t", index=False)
    other_df.to_csv(other_path, sep="\t", index=False)

    apply_global_fdr(stats_dir, alpha=0.05, include_glob="corr_stats_*.tsv")

    corr_updated = pd.read_csv(corr_path, sep="\t")
    other_updated = pd.read_csv(other_path, sep="\t")

    assert "q_global" in corr_updated.columns
    assert "q_global" not in other_updated.columns


def test_correlate_features_loop_condition_mask_aligns_targets():
    df = pd.DataFrame(
        {
            "feat1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "feat2": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=list("abcde"),
    )
    y = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=list("abcde"))

    # Use a boolean mask that selects a strict subset.
    bool_mask = np.array([True, False, True, True, False])
    recs_bool, _ = correlate_features_loop(
        df,
        y,
        method="spearman",
        min_samples=3,
        condition_mask=bool_mask,
        identifier_type="feature",
        analysis_type="test",
    )

    # Use explicit indices for the same subset.
    idx_mask = np.where(bool_mask)[0]
    recs_idx, _ = correlate_features_loop(
        df,
        y,
        method="spearman",
        min_samples=3,
        condition_mask=idx_mask,
        identifier_type="feature",
        analysis_type="test",
    )

    assert len(recs_bool) == len(recs_idx)
    assert recs_bool[0].n_valid == recs_idx[0].n_valid == 3
