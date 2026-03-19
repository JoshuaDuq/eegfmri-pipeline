from __future__ import annotations

from pathlib import Path

import pandas as pd

import eeg_pipeline.infra.machine_learning as machine_learning_infra

from eeg_pipeline.infra.machine_learning import (
    export_indices,
    export_predictions,
    prepare_best_params_path,
)


def test_prepare_best_params_path_scoped_and_truncate_modes(tmp_path: Path) -> None:
    base_path = tmp_path / "best_params" / "params.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text("stale", encoding="utf-8")

    scoped_path = prepare_best_params_path(base_path, "run_scoped", run_id="abc123")
    assert scoped_path == base_path.with_name("params_abc123.json")
    assert scoped_path.parent.exists()

    truncated_path = prepare_best_params_path(base_path, "truncate")
    assert truncated_path == base_path
    assert not base_path.exists()


def test_export_predictions_writes_expected_columns(tmp_path: Path, monkeypatch) -> None:
    meta = pd.DataFrame(
        {
            "trial_id": [10, 11, 12],
            "run_id": ["r1", "r2", "r3"],
            "block": [1, 2, 3],
        }
    )
    save_path = tmp_path / "predictions.parquet"
    y_true = [0.0, 1.0]
    y_pred = [0.1, 0.9]
    groups_ordered = ["sub-0001", "sub-0002"]
    test_indices = [0, 2]
    fold_ids = [1, 1]

    captured = {}

    def _capture_write_tsv(df, path):
        captured["tsv_df"] = df.copy()
        captured["tsv_path"] = path

    def _capture_write_parquet(df, path):
        captured["parquet_df"] = df.copy()
        captured["parquet_path"] = path

    monkeypatch.setattr(machine_learning_infra, "write_tsv", _capture_write_tsv)
    monkeypatch.setattr(machine_learning_infra, "write_parquet", _capture_write_parquet)

    pred_df = export_predictions(
        y_true=y_true,
        y_pred=y_pred,
        groups_ordered=groups_ordered,
        test_indices=test_indices,
        fold_ids=fold_ids,
        model_name="elasticnet",
        meta=meta,
        save_path=save_path,
    )

    assert pred_df["subject_id"].tolist() == groups_ordered
    assert pred_df["trial_id"].tolist() == [10, 12]
    assert pred_df["run_id"].tolist() == ["r1", "r3"]
    assert pred_df["block"].tolist() == [1, 3]
    assert captured["tsv_path"] == save_path.with_suffix(".tsv")
    assert captured["parquet_path"] == save_path.with_suffix(".parquet")
    pd.testing.assert_frame_equal(captured["tsv_df"], pred_df)
    pd.testing.assert_frame_equal(captured["parquet_df"], pred_df)


def test_export_indices_writes_heldout_subject_id(tmp_path: Path, monkeypatch) -> None:
    meta = pd.DataFrame(
        {
            "trial_id": [10, 11],
            "run_id": ["r1", "r2"],
            "block": [1, 2],
        }
    )
    save_path = tmp_path / "indices.parquet"
    groups_ordered = ["sub-0001", "sub-0002"]
    test_indices = [0, 1]
    fold_ids = [1, 2]

    captured = {}

    def _capture_write_tsv(df, path):
        captured["tsv_df"] = df.copy()
        captured["tsv_path"] = path

    def _capture_write_parquet(df, path):
        captured["parquet_df"] = df.copy()
        captured["parquet_path"] = path

    monkeypatch.setattr(machine_learning_infra, "write_tsv", _capture_write_tsv)
    monkeypatch.setattr(machine_learning_infra, "write_parquet", _capture_write_parquet)

    export_indices(
        groups_ordered=groups_ordered,
        test_indices=test_indices,
        fold_ids=fold_ids,
        meta=meta,
        save_path=save_path,
        blocks_source="behavioral",
        add_heldout_subject_id=True,
    )

    expected_columns = [
        "subject_id",
        "trial_id",
        "fold",
        "blocks_source",
        "run_id",
        "block",
        "heldout_subject_id",
    ]
    assert list(captured["tsv_df"].columns) == expected_columns
    assert captured["tsv_df"]["heldout_subject_id"].tolist() == groups_ordered
    assert captured["parquet_path"] == save_path.with_suffix(".parquet")
    assert captured["tsv_path"] == save_path.with_suffix(".tsv")
