from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _feature_io_import_stubs() -> dict[str, types.ModuleType]:
    return {
        "mne": _make_module("mne"),
        "eeg_pipeline.analysis.features.rest": _make_module(
            "eeg_pipeline.analysis.features.rest",
            is_resting_state_feature_mode=lambda config: False,
        ),
        "eeg_pipeline.utils.data.columns": _make_module(
            "eeg_pipeline.utils.data.columns",
            pick_target_column=lambda *_args, **_kwargs: None,
        ),
        "eeg_pipeline.utils.data.feature_alignment": _make_module(
            "eeg_pipeline.utils.data.feature_alignment",
            attach_feature_alignment_columns=lambda df, *_args, **_kwargs: df,
            filter_feature_payload_columns=lambda df, *_args, **_kwargs: df,
        ),
        "eeg_pipeline.utils.config.loader": _make_module(
            "eeg_pipeline.utils.config.loader",
            get_config_value=lambda config, key, default=None: default,
        ),
        "eeg_pipeline.utils.data.epochs": _make_module(
            "eeg_pipeline.utils.data.epochs",
            load_epochs_for_analysis=lambda *_args, **_kwargs: (None, None),
        ),
        "eeg_pipeline.infra.paths": _make_module(
            "eeg_pipeline.infra.paths",
            deriv_features_path=lambda deriv_root, subject: Path(deriv_root) / f"sub-{subject}" / "eeg" / "features",
            find_connectivity_features_path=lambda deriv_root, subject: Path(deriv_root) / f"sub-{subject}" / "eeg" / "features" / "connectivity" / "features_connectivity.parquet",
        ),
        "eeg_pipeline.infra.tsv": _make_module(
            "eeg_pipeline.infra.tsv",
            read_table=lambda path: pd.DataFrame(),
            write_parquet=lambda *_args, **_kwargs: None,
            write_tsv=lambda *_args, **_kwargs: None,
        ),
        "eeg_pipeline.utils.data.source_localization_paths": _make_module(
            "eeg_pipeline.utils.data.source_localization_paths",
            resolve_source_localization_method=lambda *_args, **_kwargs: None,
            resolve_source_localization_method_from_attrs=lambda *_args, **_kwargs: None,
            source_localization_candidate_paths=lambda *_args, **_kwargs: [],
            source_localization_folder=lambda *_args, **_kwargs: "sourcelocalization",
        ),
    }


class TestFeatureIoFailFast(unittest.TestCase):
    def setUp(self):
        self._patcher = patch.dict(sys.modules, _feature_io_import_stubs())
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        sys.modules.pop("eeg_pipeline.utils.data.feature_io", None)
        self.feature_io = importlib.import_module("eeg_pipeline.utils.data.feature_io")
        self.addCleanup(sys.modules.pop, "eeg_pipeline.utils.data.feature_io", None)

    def test_safe_read_table_returns_none_for_missing_path(self):
        logger = Mock()
        missing_path = Path(tempfile.mkdtemp()) / "missing.parquet"

        self.assertIsNone(self.feature_io._safe_read_table(missing_path, logger))

    def test_safe_read_table_raises_for_invalid_existing_table(self):
        logger = Mock()
        existing_path = Path(tempfile.mkdtemp()) / "broken.parquet"
        existing_path.write_text("broken", encoding="utf-8")

        with patch.object(
            self.feature_io,
            "read_table",
            side_effect=pd.errors.ParserError("bad table"),
        ):
            with self.assertRaisesRegex(pd.errors.ParserError, "bad table"):
                self.feature_io._safe_read_table(existing_path, logger)

    def test_safe_read_feature_table_with_path_raises_for_invalid_existing_table(self):
        logger = Mock()
        features_dir = Path(tempfile.mkdtemp())
        feature_path = features_dir / "power" / "features_power.parquet"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text("broken", encoding="utf-8")

        with patch.object(
            self.feature_io,
            "read_table",
            side_effect=pd.errors.ParserError("bad feature table"),
        ):
            with self.assertRaisesRegex(pd.errors.ParserError, "bad feature table"):
                self.feature_io._safe_read_feature_table_with_path(
                    features_dir,
                    "features_power",
                    logger,
                )

    def test_save_feature_metadata_uses_runtime_task(self):
        logger = Mock()
        features_dir = Path(tempfile.mkdtemp()) / "sub-0001" / "eeg" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"power_alpha": [1.0]})
        manifest_calls: list[dict[str, object]] = []

        def _generate_manifest(**kwargs):
            manifest_calls.append(kwargs)
            return {"task": kwargs["task"]}

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.domain.features.naming": _make_module(
                    "eeg_pipeline.domain.features.naming",
                    generate_manifest=_generate_manifest,
                )
            },
        ):
            self.feature_io._save_feature_metadata(
                df=df,
                base_filename="features_power",
                features_dir=features_dir,
                config={"project": {"task": "config-task"}},
                logger=logger,
                task="runtime-task",
                qc={"range_qc": {"status": "ok"}},
            )

        self.assertEqual(manifest_calls[-1]["task"], "runtime-task")
        self.assertEqual(manifest_calls[-1]["qc"], {"range_qc": {"status": "ok"}})

    def test_save_all_features_propagates_feature_qc_to_metadata(self):
        features_dir = Path(tempfile.mkdtemp())
        pow_df = pd.DataFrame({"power_alpha": [1.0]})
        qc_payload = {"range_qc": {"status": "ok"}}

        with patch.object(self.feature_io, "_save_feature_metadata") as mock_save_metadata:
            self.feature_io.save_all_features(
                pow_df=pow_df,
                pow_cols=list(pow_df.columns),
                baseline_df=pd.DataFrame(),
                baseline_cols=[],
                conn_df=None,
                conn_cols=[],
                aper_df=None,
                aper_cols=[],
                features_dir=features_dir,
                config={},
                feature_qc=qc_payload,
            )

        self.assertTrue(mock_save_metadata.called)
        self.assertEqual(mock_save_metadata.call_args.kwargs["qc"], qc_payload)

    def test_save_aperiodic_qc_writes_tsv_not_parquet(self):
        features_dir = Path(tempfile.mkdtemp())
        logger = Mock()
        qc_payload = {
            "slopes": np.array([[1.0]]),
            "offsets": np.array([[2.0]]),
            "r2": np.array([[0.95]]),
            "rms": np.array([[0.1]]),
            "fit_ok": np.array([[True]]),
            "valid_bins": np.array([[10]]),
            "kept_bins": np.array([[8]]),
            "peak_rejected": np.array([[False]]),
            "channel_names": ["Cz"],
        }

        with patch.object(self.feature_io, "write_tsv") as mock_write_tsv, patch.object(
            self.feature_io, "write_parquet"
        ) as mock_write_parquet:
            self.feature_io._save_aperiodic_qc(qc_payload, features_dir, logger)

        mock_write_tsv.assert_called_once()
        mock_write_parquet.assert_not_called()

    def test_load_features_and_targets_passes_deriv_root_to_aligned_events_lookup(self):
        deriv_root = Path(tempfile.mkdtemp())
        features_dir = deriv_root / "sub-0001" / "eeg" / "features" / "power"
        features_dir.mkdir(parents=True, exist_ok=True)
        (features_dir / "features_power.parquet").write_text("x", encoding="utf-8")
        captured: dict[str, object] = {}

        def _get_aligned_events(*_args, **kwargs):
            captured["deriv_root"] = kwargs["deriv_root"]
            return pd.DataFrame({"rating": [1.0]})

        with patch.object(
            self.feature_io,
            "read_table",
            side_effect=[
                pd.DataFrame({"power_alpha": [1.0]}),
                pd.DataFrame({"power_alpha": [1.0]}),
            ],
        ), patch.object(
            self.feature_io,
            "pick_target_column",
            return_value="rating",
        ), patch.dict(
            sys.modules,
            {
                "eeg_pipeline.utils.data.alignment": _make_module(
                    "eeg_pipeline.utils.data.alignment",
                    get_aligned_events=_get_aligned_events,
                )
            },
        ):
            self.feature_io._load_features_and_targets(
                subject="0001",
                task="task",
                deriv_root=deriv_root,
                config={},
                epochs=object(),
            )

        self.assertEqual(captured["deriv_root"], deriv_root)
