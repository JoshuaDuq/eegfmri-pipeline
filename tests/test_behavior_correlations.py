import logging

import numpy as np
import pandas as pd

import eeg_pipeline.analysis.behavior.feature_correlator as feature_correlator
from eeg_pipeline.analysis.behavior.feature_correlator import CorrelationConfig, FeatureBehaviorCorrelator
from eeg_pipeline.utils.analysis.stats.correlation import run_pain_sensitivity_correlations


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
