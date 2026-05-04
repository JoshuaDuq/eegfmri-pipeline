from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_residualize_targets_for_fold_fits_only_training_rows() -> None:
    from eeg_pipeline.analysis.machine_learning.target_residualization import (
        residualize_targets_for_fold,
    )

    y = np.asarray([100.0, 110.0, 0.0, 10.0, 0.0, 10.0], dtype=float)
    meta = pd.DataFrame(
        {
            "pain_binary_coded": [0, 1, 0, 1, 0, 1],
            "subject_id": ["sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03"],
        }
    )
    train_idx = np.asarray([2, 3, 4, 5], dtype=int)
    test_idx = np.asarray([0, 1], dtype=int)

    y_train, y_test, details = residualize_targets_for_fold(
        y=y,
        meta=meta,
        train_idx=train_idx,
        test_idx=test_idx,
        columns=("pain_binary_coded",),
    )

    assert np.allclose(y_train, [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(y_test, [100.0, 100.0])
    assert details["columns"] == ["pain_binary_coded"]


def test_residualize_targets_for_fold_rejects_missing_or_degenerate_design() -> None:
    from eeg_pipeline.analysis.machine_learning.target_residualization import (
        residualize_targets_for_fold,
    )

    y = np.asarray([1.0, 2.0, 3.0], dtype=float)
    meta = pd.DataFrame({"pain_binary_coded": [1, 1, 1]})

    with pytest.raises(ValueError, match="missing"):
        residualize_targets_for_fold(
            y=y,
            meta=meta,
            train_idx=np.asarray([0, 1], dtype=int),
            test_idx=np.asarray([2], dtype=int),
            columns=("missing",),
        )

    with pytest.raises(ValueError, match="rank deficient"):
        residualize_targets_for_fold(
            y=y,
            meta=meta,
            train_idx=np.asarray([0, 1], dtype=int),
            test_idx=np.asarray([2], dtype=int),
            columns=("pain_binary_coded",),
        )
