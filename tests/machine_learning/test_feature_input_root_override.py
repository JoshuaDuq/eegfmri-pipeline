from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests.pipelines_test_utils import DotConfig


def _config(deriv_root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(deriv_root)},
            "feature_engineering": {"analysis_mode": "trial_ml_safe"},
            "machine_learning": {
                "data": {
                    "feature_set": "combined",
                    "require_trial_ml_safe": True,
                }
            },
        }
    )


def _write_feature_table(feature_root: Path) -> None:
    feature_dir = feature_root / "sub-0001" / "eeg" / "features" / "power"
    metadata_dir = feature_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"power_alpha_global_mean": [1.0]}).to_parquet(
        feature_dir / "features_power.parquet",
        index=False,
    )
    (metadata_dir / "extraction_config.json").write_text(
        '{"analysis_mode": "trial_ml_safe"}\n',
        encoding="utf-8",
    )


def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_index": [1],
            "onset": [0.0],
            "duration": [1.0],
        }
    )


def _target_payload() -> tuple[pd.Series, str, pd.DataFrame]:
    return pd.Series([2.5], dtype=float), "NPS", pd.DataFrame(index=[0])


def test_load_active_matrix_reads_features_from_override_root(tmp_path) -> None:
    from eeg_pipeline.utils.data.machine_learning import load_active_matrix

    deriv_root = tmp_path / "derivatives"
    feature_root = tmp_path / "study1_features"
    _write_feature_table(feature_root)

    with patch(
        "eeg_pipeline.utils.data.machine_learning.load_events_df",
        return_value=_events_frame(),
    ), patch(
        "eeg_pipeline.utils.data.machine_learning._load_fmri_signature_target_for_subject",
        return_value=_target_payload(),
    ):
        X, y, groups, feature_names, meta = load_active_matrix(
            subjects=["0001"],
            task="thermalactive",
            deriv_root=deriv_root,
            config=_config(deriv_root),
            log=logging.getLogger(__name__),
            feature_families=["power"],
            feature_input_root=feature_root,
            target="fmri_signature",
            target_kind="continuous",
        )

    assert X.shape == (1, 1)
    assert y.tolist() == [2.5]
    assert groups.tolist() == ["sub-0001"]
    assert feature_names == ["power_alpha_global_mean"]
    assert meta["subject_id"].tolist() == ["sub-0001"]


def test_load_active_matrix_defaults_to_deriv_root_when_override_absent(tmp_path) -> None:
    from eeg_pipeline.utils.data.machine_learning import load_active_matrix

    deriv_root = tmp_path / "derivatives"
    _write_feature_table(deriv_root)

    with patch(
        "eeg_pipeline.utils.data.machine_learning.load_events_df",
        return_value=_events_frame(),
    ), patch(
        "eeg_pipeline.utils.data.machine_learning._load_fmri_signature_target_for_subject",
        return_value=_target_payload(),
    ):
        X, y, groups, feature_names, _meta = load_active_matrix(
            subjects=["0001"],
            task="thermalactive",
            deriv_root=deriv_root,
            config=_config(deriv_root),
            log=logging.getLogger(__name__),
            feature_families=["power"],
            target="fmri_signature",
            target_kind="continuous",
        )

    assert X.shape == (1, 1)
    assert y.tolist() == [2.5]
    assert groups.tolist() == ["sub-0001"]
    assert feature_names == ["power_alpha_global_mean"]
