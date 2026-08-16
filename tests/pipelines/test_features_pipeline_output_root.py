from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.utils.pipelines_test_utils import DotConfig


class _FakeEpochs:
    def __init__(self) -> None:
        self.info = {"sfreq": 100.0}
        self.times = np.array([0.0, 0.1], dtype=float)
        self.ch_names = ["Cz"]

    def __len__(self) -> int:
        return 1


def _config(root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {
                "bids_root": str(root / "bids"),
                "deriv_root": str(root / "derivatives"),
            },
            "project": {"task": "thermalactive"},
            "feature_engineering": {"analysis_mode": "group_stats"},
        }
    )


def test_feature_pipeline_writes_features_to_override_root(tmp_path) -> None:
    from eeg_pipeline.pipelines.features import FeaturePipeline

    cfg = _config(tmp_path)
    pipeline = FeaturePipeline(config=cfg)
    output_root = tmp_path / "study1_features"
    captured: dict[str, Path] = {}
    dummy_progress = SimpleNamespace(
        subject_start=lambda *_args, **_kwargs: None,
        step=lambda *_args, **_kwargs: None,
        subject_done=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )

    def _capture_save_all_features(**kwargs):
        features_dir = Path(kwargs["features_dir"])
        captured["features_dir"] = features_dir
        (features_dir / "power").mkdir(parents=True, exist_ok=True)
        (features_dir / "power" / "features_power.parquet").write_text(
            "placeholder",
            encoding="utf-8",
        )
        return pd.DataFrame({"power_alpha_global_mean": [1.0]})

    with (
        patch("eeg_pipeline.pipelines.features.validate_rest_configuration"),
        patch("eeg_pipeline.pipelines.features.setup_matplotlib"),
        patch(
            "eeg_pipeline.pipelines.features.resolve_feature_categories",
            return_value=["power"],
        ),
        patch(
            "eeg_pipeline.pipelines.features.ensure_progress_reporter",
            return_value=dummy_progress,
        ),
        patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis",
            return_value=(_FakeEpochs(), pd.DataFrame({"trial_index": [1]})),
        ) as load_epochs,
        patch(
            "eeg_pipeline.pipelines.features._load_events_df",
            return_value=None,
        ),
        patch(
            "eeg_pipeline.pipelines.features.extract_all_features",
            return_value=SimpleNamespace(
                aper_qc=None, ratios_df=None, asymmetry_df=None, quality_df=None
            ),
        ),
        patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results",
            return_value={
                "pow_df": pd.DataFrame({"power_alpha_global_mean": [1.0]}),
                "pow_cols": ["power_alpha_global_mean"],
                "baseline_df": pd.DataFrame(),
                "baseline_cols": [],
                "conn_df": pd.DataFrame(),
                "conn_cols": [],
                "dconn_df": pd.DataFrame(),
                "dconn_cols": [],
                "source_df": pd.DataFrame(),
                "source_cols": [],
                "source_contrast_df": pd.DataFrame(),
                "source_contrast_cols": [],
                "aper_df": pd.DataFrame(),
                "aper_cols": [],
                "erp_df": pd.DataFrame(),
                "erp_cols": [],
                "itpc_df": pd.DataFrame(),
                "itpc_cols": [],
                "itpc_trial_df": pd.DataFrame(),
                "itpc_trial_cols": [],
                "pac_df": pd.DataFrame(),
                "pac_trials_df": pd.DataFrame(),
                "pac_time_df": pd.DataFrame(),
                "comp_df": pd.DataFrame(),
                "comp_cols": [],
                "bursts_df": pd.DataFrame(),
                "bursts_cols": [],
                "spectral_df": pd.DataFrame(),
                "spectral_cols": [],
                "erds_df": pd.DataFrame(),
                "erds_cols": [],
                "ratios_df": pd.DataFrame(),
                "asymmetry_df": pd.DataFrame(),
                "quality_df": pd.DataFrame(),
                "quality_cols": [],
                "microstates_df": pd.DataFrame(),
                "microstates_cols": [],
                "aper_qc": None,
            },
        ),
        patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"power_alpha_global_mean": [1.0]}),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                {"n_retained": 1, "n_original": 1, "extra_blocks": {}},
            ),
        ),
        patch(
            "eeg_pipeline.pipelines.features._build_feature_qc",
            return_value={},
        ),
        patch(
            "eeg_pipeline.pipelines.features.save_all_features",
            side_effect=_capture_save_all_features,
        ),
        patch(
            "eeg_pipeline.pipelines.features._collect_trial_table_feature_tables",
            return_value=[],
        ),
        patch("eeg_pipeline.pipelines.features._save_canonical_trial_table_artifact"),
        patch("eeg_pipeline.pipelines.features._save_extraction_config"),
    ):
        pipeline.process_subject(
            "0001",
            task="thermalactive",
            progress=dummy_progress,
            feature_categories=["power"],
            feature_output_root=output_root,
        )

    assert load_epochs.call_args.kwargs["deriv_root"] == pipeline.deriv_root
    assert captured["features_dir"] == output_root / "sub-0001" / "eeg" / "features"
    assert (
        output_root / "sub-0001" / "eeg" / "features" / "power" / "features_power.parquet"
    ).exists()


def test_feature_pipeline_can_disable_canonical_trial_table_export(tmp_path) -> None:
    from eeg_pipeline.pipelines.features import FeaturePipeline

    cfg = _config(tmp_path)
    pipeline = FeaturePipeline(config=cfg)
    dummy_progress = SimpleNamespace(
        subject_start=lambda *_args, **_kwargs: None,
        step=lambda *_args, **_kwargs: None,
        subject_done=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )

    with (
        patch("eeg_pipeline.pipelines.features.validate_rest_configuration"),
        patch("eeg_pipeline.pipelines.features.setup_matplotlib"),
        patch(
            "eeg_pipeline.pipelines.features.resolve_feature_categories",
            return_value=["power"],
        ),
        patch(
            "eeg_pipeline.pipelines.features.ensure_progress_reporter",
            return_value=dummy_progress,
        ),
        patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis",
            return_value=(_FakeEpochs(), pd.DataFrame({"trial_index": [1]})),
        ),
        patch(
            "eeg_pipeline.pipelines.features._load_events_df",
            return_value=None,
        ),
        patch(
            "eeg_pipeline.pipelines.features.extract_all_features",
            return_value=SimpleNamespace(
                aper_qc=None, ratios_df=None, asymmetry_df=None, quality_df=None
            ),
        ),
        patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results",
            return_value={
                "pow_df": pd.DataFrame({"power_alpha_global_mean": [1.0]}),
                "pow_cols": ["power_alpha_global_mean"],
                "baseline_df": pd.DataFrame(),
                "baseline_cols": [],
                "conn_df": pd.DataFrame(),
                "conn_cols": [],
                "dconn_df": pd.DataFrame(),
                "dconn_cols": [],
                "source_df": pd.DataFrame(),
                "source_cols": [],
                "source_contrast_df": pd.DataFrame(),
                "source_contrast_cols": [],
                "aper_df": pd.DataFrame(),
                "aper_cols": [],
                "erp_df": pd.DataFrame(),
                "erp_cols": [],
                "itpc_df": pd.DataFrame(),
                "itpc_cols": [],
                "itpc_trial_df": pd.DataFrame(),
                "itpc_trial_cols": [],
                "pac_df": pd.DataFrame(),
                "pac_trials_df": pd.DataFrame(),
                "pac_time_df": pd.DataFrame(),
                "comp_df": pd.DataFrame(),
                "comp_cols": [],
                "bursts_df": pd.DataFrame(),
                "bursts_cols": [],
                "spectral_df": pd.DataFrame(),
                "spectral_cols": [],
                "erds_df": pd.DataFrame(),
                "erds_cols": [],
                "ratios_df": pd.DataFrame(),
                "asymmetry_df": pd.DataFrame(),
                "quality_df": pd.DataFrame(),
                "quality_cols": [],
                "microstates_df": pd.DataFrame(),
                "microstates_cols": [],
                "aper_qc": None,
            },
        ),
        patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"power_alpha_global_mean": [1.0]}),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                {"n_retained": 1, "n_original": 1, "extra_blocks": {}},
            ),
        ),
        patch(
            "eeg_pipeline.pipelines.features._build_feature_qc",
            return_value={},
        ),
        patch(
            "eeg_pipeline.pipelines.features.save_all_features",
            return_value=pd.DataFrame({"power_alpha_global_mean": [1.0]}),
        ),
        patch(
            "eeg_pipeline.pipelines.features._collect_trial_table_feature_tables",
            return_value=[],
        ),
        patch(
            "eeg_pipeline.pipelines.features._save_canonical_trial_table_artifact"
        ) as save_trial_table,
        patch("eeg_pipeline.pipelines.features._save_extraction_config"),
    ):
        pipeline.process_subject(
            "0001",
            task="thermalactive",
            progress=dummy_progress,
            feature_categories=["power"],
            save_canonical_trial_table=False,
        )

    save_trial_table.assert_not_called()
