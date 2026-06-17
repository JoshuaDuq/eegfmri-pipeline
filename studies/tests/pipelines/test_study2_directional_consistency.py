from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.study2.config import load_study2_config


def test_directional_consistency_accepts_matching_prediction_and_target_maps() -> None:
    from studies.pain_study.study2.directional_consistency import (
        evaluate_directional_consistency,
    )

    qc = evaluate_directional_consistency(
        prediction_map=np.asarray([0.5, 0.4, -0.3, -0.2], dtype=float),
        target_map=np.asarray([0.6, 0.2, -0.4, 0.1], dtype=float),
        cluster_mask=np.asarray([True, True, True, False]),
        config=load_study2_config(),
    )

    assert qc.directional_criteria_met is True
    assert qc.spatial_r > 0.20
    assert qc.same_sign_fraction == pytest.approx(1.0)
    assert qc.unmet_criteria == ()


def test_directional_consistency_rejects_empty_cluster_mask() -> None:
    from studies.pain_study.study2.directional_consistency import (
        evaluate_directional_consistency,
    )

    with pytest.raises(ValueError, match="cluster_mask"):
        evaluate_directional_consistency(
            prediction_map=np.asarray([0.5, 0.4], dtype=float),
            target_map=np.asarray([0.6, 0.2], dtype=float),
            cluster_mask=np.asarray([False, False]),
            config=load_study2_config(),
        )
