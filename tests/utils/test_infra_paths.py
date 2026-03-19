from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eeg_pipeline.infra.paths import (
    ensure_derivatives_dataset_description,
    find_clean_epochs_path,
    find_clean_events_path,
    load_events_df,
    resolve_deriv_root,
)
from tests.pipelines_test_utils import DotConfig


def test_resolve_deriv_root_accepts_direct_config_and_constants(tmp_path: Path) -> None:
    direct_root = tmp_path / "direct"
    config_root = tmp_path / "config"
    constants_root = tmp_path / "constants"

    config = DotConfig({"deriv_root": str(config_root)})
    constants = {"DERIV_ROOT": str(constants_root)}

    assert resolve_deriv_root(direct_root) == direct_root
    assert resolve_deriv_root(None, config=config) == config_root
    assert resolve_deriv_root(None, constants=constants) == constants_root

    with pytest.raises(ValueError, match="Either deriv_root, config, or constants"):
        resolve_deriv_root()


def test_find_clean_epochs_and_events_paths(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    epochs_path = (
        deriv_root
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-rest_proc-clean_epo.fif"
    )
    events_path = epochs_path.with_name("sub-0001_task-rest_proc-clean_events.tsv")
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_path.write_text("epochs", encoding="utf-8")
    events_path.write_text("onset\tduration\n1\t2\n", encoding="utf-8")

    found_epochs = find_clean_epochs_path("0001", "rest", deriv_root=deriv_root)
    found_events = find_clean_events_path("0001", "rest", deriv_root=deriv_root)

    assert found_epochs == epochs_path
    assert found_events == events_path


def test_load_events_df_prefers_clean_events_when_available(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    bids_root = tmp_path / "bids"

    clean_epochs_path = (
        deriv_root
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-rest_proc-clean_epo.fif"
    )
    clean_events_path = clean_epochs_path.with_name(
        "sub-0001_task-rest_proc-clean_events.tsv"
    )
    bids_events_path = bids_root / "sub-0001" / "eeg" / "sub-0001_task-rest_events.tsv"

    clean_events_path.parent.mkdir(parents=True, exist_ok=True)
    bids_events_path.parent.mkdir(parents=True, exist_ok=True)
    clean_epochs_path.write_text("epochs", encoding="utf-8")
    pd.DataFrame({"marker": ["clean"]}).to_csv(clean_events_path, sep="\t", index=False)
    pd.DataFrame({"marker": ["bids"]}).to_csv(bids_events_path, sep="\t", index=False)

    loaded = load_events_df(
        "0001",
        "rest",
        constants={"DERIV_ROOT": str(deriv_root), "BIDS_ROOT": str(bids_root)},
    )

    assert loaded is not None
    assert loaded["marker"].tolist() == ["clean"]


def test_load_events_df_falls_back_to_bids_events_when_requested(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    bids_events_path = bids_root / "sub-0001" / "eeg" / "sub-0001_task-rest_events.tsv"
    bids_events_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"marker": ["bids"]}).to_csv(bids_events_path, sep="\t", index=False)

    loaded = load_events_df("0001", "rest", bids_root=bids_root, prefer_clean=False)

    assert loaded is not None
    assert loaded["marker"].tolist() == ["bids"]


def test_ensure_derivatives_dataset_description_creates_metadata_once(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"

    ensure_derivatives_dataset_description(deriv_root=deriv_root)

    desc_path = deriv_root / "dataset_description.json"
    assert desc_path.exists()

    metadata = json.loads(desc_path.read_text(encoding="utf-8"))
    assert metadata["DatasetType"] == "derivative"
    assert metadata["GeneratedBy"][0]["Name"] == "PAIN_EEG_fMRI"

    sentinel = {"preserved": True}
    desc_path.write_text(json.dumps(sentinel), encoding="utf-8")

    ensure_derivatives_dataset_description(deriv_root=deriv_root)

    assert json.loads(desc_path.read_text(encoding="utf-8")) == sentinel
