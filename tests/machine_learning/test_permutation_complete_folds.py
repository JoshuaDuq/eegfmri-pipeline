"""Null draws must retain the full observed participant composition."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.machine_learning import orchestration
from tests.utils.pipelines_test_utils import DotConfig


@pytest.mark.parametrize("staged", [False, True])
def test_permutation_discards_entire_draw_with_nonfinite_fold(monkeypatch, staged):
    y = np.arange(22, dtype=float)
    groups = np.repeat(["s1", "s2"], 11)
    folds = [(np.arange(11), np.arange(11, 22)), (np.arange(11, 22), np.arange(11))]
    meta = pd.DataFrame({"run": np.ones(22), "trial_index": np.tile(np.arange(1, 12), 2)})
    config = DotConfig({"machine_learning": {
        "cv": {"permutation_scheme": "circular_shift_within_run"},
        "target_residualization": {"strategy": "staged_residual_learning"},
    }})
    scores = iter([0.2, np.nan, 0.2, 0.3])

    def predictions(**kwargs):
        return SimpleNamespace(records=[{"r2": next(scores)} for _ in kwargs["outer_folds"]])

    monkeypatch.setattr(orchestration, "model_comparison_cv_predictions", predictions)
    monkeypatch.setattr(
        orchestration, "reconstruct_staged_permutation_target_for_fold", lambda **kwargs: y.copy()
    )
    result = orchestration._model_comparison_permutation_p_value(
        observed_mean_r2=0.4, X=y[:, None], y=y, groups=groups, meta=meta,
        outer_folds=folds, model_name="ridge", pipe=object(), param_grid={},
        inner_splits=2, outer_jobs=1, config=config, harmonization_mode="none",
        covariates=None, target_residualization_columns=("nuisance",) if staged else (),
        rng=np.random.default_rng(2), n_perm=1,
    )
    assert result.n_perm_completed == 1
    assert result.n_perm_attempted == 2
    assert result.n_invalid_permutations == 1
    assert result.p_value == 0.5
