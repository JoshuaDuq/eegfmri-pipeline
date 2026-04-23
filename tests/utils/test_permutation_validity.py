from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.analysis.stats.permutation import (
    compute_permutation_pvalues_with_cov_predictor,
    perm_pval_partial_freedman_lane,
)


def test_freedman_lane_rejects_invalid_permutation_denominator() -> None:
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    y = pd.Series([0.2, 1.1, 1.9, 3.1, 3.8])
    z = pd.DataFrame({"cov": [0.0, 0.0, 1.0, 1.0, 2.0]})
    real_lstsq = np.linalg.lstsq
    call_count = {"n": 0}

    def fail_permutation_lstsq(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] > 2:
            raise np.linalg.LinAlgError("invalid permutation")
        return real_lstsq(*args, **kwargs)

    with patch(
        "eeg_pipeline.utils.analysis.stats.permutation.np.linalg.lstsq",
        side_effect=fail_permutation_lstsq,
    ):
        with pytest.raises(ValueError, match="produced invalid permutations"):
            perm_pval_partial_freedman_lane(
                x=x,
                y=y,
                Z=z,
                method="pearson",
                n_perm=4,
                rng=np.random.default_rng(0),
            )


def test_combined_covariate_predictor_permutation_surfaces_invalid_partial_test() -> None:
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    y = pd.Series([0.2, 1.1, 1.9, 3.1, 3.8])
    covariates = pd.DataFrame({"cov": [0.0, 0.0, 1.0, 1.0, 2.0]})
    predictor = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])

    with patch(
        "eeg_pipeline.utils.analysis.stats.permutation.perm_pval_partial_freedman_lane",
        side_effect=[0.5, 0.5, ValueError("invalid denominator")],
    ):
        with pytest.raises(ValueError, match="invalid denominator"):
            compute_permutation_pvalues_with_cov_predictor(
                x_aligned=x,
                y_aligned=y,
                covariates_df=covariates,
                predictor_series=predictor,
                method="pearson",
                n_perm=4,
                n_eff=5,
                rng=np.random.default_rng(0),
                min_samples=3,
                config={"behavior_analysis": {"statistics": {"predictor_control": "linear"}}},
            )
