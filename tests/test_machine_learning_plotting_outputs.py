from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd


class TestMachineLearningPlottingOutputs(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")

    def _write_regression_inputs(self, out_dir: Path) -> None:
        df = pd.DataFrame(
            {
                "subject_id": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
                "y_true": [10.0, 20.0, 30.0, 40.0],
                "y_pred": [11.0, 18.0, 28.0, 41.0],
            }
        )
        df.to_csv(out_dir / "loso_predictions.tsv", sep="\t", index=False)

    def _write_classification_inputs(self, out_dir: Path) -> None:
        df = pd.DataFrame(
            {
                "subject_id": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
                "y_true": [0, 1, 0, 1],
                "y_pred": [0, 1, 1, 1],
                "y_prob": [0.1, 0.8, 0.7, 0.9],
            }
        )
        df.to_csv(out_dir / "loso_predictions.tsv", sep="\t", index=False)

    def _write_timegen_inputs(self, out_dir: Path) -> None:
        n = 4
        r = np.eye(n, dtype=float) * 0.35
        r2 = np.eye(n, dtype=float) * 0.12
        centers = np.array([0.1, 0.3, 0.5, 0.7], dtype=float)
        tested = np.isfinite(r)
        sig = np.eye(n, dtype=bool)
        np.savez_compressed(
            out_dir / "time_generalization_regression.npz",
            r_matrix=r,
            r2_matrix=r2,
            window_centers=centers,
            tested_mask=tested,
            sig_fdr=sig,
            sig_maxstat=np.zeros((n, n), dtype=bool),
            sig_cluster=np.zeros((n, n), dtype=bool),
        )

    def _write_model_comparison_inputs(self, out_dir: Path) -> None:
        rows = []
        for fold, subj in enumerate(["sub-0001", "sub-0002", "sub-0003"]):
            rows.extend(
                [
                    {"model": "elasticnet", "fold": fold, "test_subject": subj, "r2": 0.22 + fold * 0.01, "mae": 4.0},
                    {"model": "ridge", "fold": fold, "test_subject": subj, "r2": 0.19 + fold * 0.01, "mae": 4.3},
                    {"model": "rf", "fold": fold, "test_subject": subj, "r2": 0.25 + fold * 0.01, "mae": 3.8},
                ]
            )
        pd.DataFrame(rows).to_csv(out_dir / "model_comparison.tsv", sep="\t", index=False)

    def _write_incremental_validity_inputs(self, out_dir: Path) -> None:
        pd.DataFrame(
            {
                "fold": [0, 1, 2],
                "test_subject": ["sub-0001", "sub-0002", "sub-0003"],
                "r2_baseline": [0.05, 0.11, 0.02],
                "r2_full": [0.15, 0.23, 0.07],
                "delta_r2": [0.10, 0.12, 0.05],
                "mae_baseline": [6.0, 5.5, 6.2],
                "mae_full": [5.1, 4.8, 5.7],
            }
        ).to_csv(out_dir / "incremental_validity.tsv", sep="\t", index=False)

    def _write_uncertainty_inputs(self, out_dir: Path) -> None:
        pd.DataFrame(
            {
                "y_pred": [10.0, 20.0, 30.0, 40.0],
                "lower": [8.0, 17.5, 27.0, 37.0],
                "upper": [12.0, 22.5, 33.0, 43.0],
                "y_true": [10.2, 21.0, 28.0, 39.2],
                "in_interval": [True, True, True, True],
            }
        ).to_csv(out_dir / "prediction_intervals.tsv", sep="\t", index=False)
        with open(out_dir / "uncertainty_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"target_coverage": 0.9, "empirical_coverage": 1.0}, f)

    def _write_shap_inputs(self, out_dir: Path) -> None:
        pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "shap_importance": [0.35, 0.2, 0.1],
                "shap_std_across_folds": [0.05, 0.04, 0.02],
            }
        ).to_csv(out_dir / "shap_importance.tsv", sep="\t", index=False)

    def _write_permutation_inputs(self, out_dir: Path) -> None:
        pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "importance_mean": [0.12, 0.08, 0.03],
                "importance_std": [0.02, 0.01, 0.01],
                "n_folds": [4, 4, 4],
            }
        ).to_csv(out_dir / "permutation_importance.tsv", sep="\t", index=False)

    def test_all_ml_modes_generate_at_least_one_plot(self) -> None:
        from eeg_pipeline.analysis.machine_learning.plotting import generate_ml_mode_plots

        mode_setups = {
            "regression": self._write_regression_inputs,
            "classify": self._write_classification_inputs,
            "timegen": self._write_timegen_inputs,
            "model_comparison": self._write_model_comparison_inputs,
            "incremental_validity": self._write_incremental_validity_inputs,
            "uncertainty": self._write_uncertainty_inputs,
            "shap": self._write_shap_inputs,
            "permutation": self._write_permutation_inputs,
        }

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for mode, writer in mode_setups.items():
                out_dir = base / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                writer(out_dir)
                generated = generate_ml_mode_plots(mode=mode, results_dir=out_dir, logger=Mock())
                self.assertGreaterEqual(len(generated), 1, f"No plots produced for mode={mode}")
                for plot_path in generated:
                    path = Path(plot_path)
                    self.assertTrue(path.exists(), f"Missing plot for mode={mode}: {path}")
                    self.assertGreater(path.stat().st_size, 0, f"Empty plot file for mode={mode}: {path}")

    def test_plotting_respects_formats_and_top_n(self) -> None:
        from eeg_pipeline.analysis.machine_learning.plotting import generate_ml_mode_plots

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "shap"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._write_shap_inputs(out_dir)
            generated = generate_ml_mode_plots(
                mode="shap",
                results_dir=out_dir,
                logger=Mock(),
                formats=("png", "pdf"),
                top_n_features=2,
            )

            png_path = out_dir / "plots" / "shap_importance_top_features.png"
            pdf_path = out_dir / "plots" / "shap_importance_top_features.pdf"
            self.assertTrue(png_path.exists(), f"Expected PNG output: {png_path}")
            self.assertTrue(pdf_path.exists(), f"Expected PDF output: {pdf_path}")
            self.assertIn(str(png_path), generated)
            self.assertIn(str(pdf_path), generated)
