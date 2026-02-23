from __future__ import annotations

import io
import json
import logging
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from eeg_pipeline.cli.commands.info_helpers import SOURCE_BIDS_FMRI, _handle_subjects_mode
from eeg_pipeline.utils.data.subjects import get_available_subjects
from tests.pipelines_test_utils import DotConfig


def test_get_available_subjects_matches_numeric_config_to_zero_padded_bids(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)

    bids_root = tmp_path / "bids"
    (bids_root / "sub-0002").mkdir(parents=True, exist_ok=True)

    config = DotConfig(
        {
            "project": {"subject_list": [2]},
            "paths": {"bids_root": str(bids_root)},
        }
    )

    subjects = get_available_subjects(
        config=config,
        deriv_root=deriv_root,
        bids_root=bids_root,
        task="task",
        discovery_sources=["bids"],
        subject_discovery_policy="intersection",
    )

    assert subjects == ["0002"]


def test_info_subjects_fmri_status_json_includes_new_bids_subjects(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)

    bids_fmri_root = tmp_path / "bids_fmri"
    (bids_fmri_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    (bids_fmri_root / "sub-0002").mkdir(parents=True, exist_ok=True)

    config = DotConfig(
        {
            "project": {"subject_list": ["0001"]},
            "paths": {
                "bids_root": str(tmp_path / "bids_eeg"),
                "bids_fmri_root": str(bids_fmri_root),
            },
        }
    )

    args = Namespace(
        source=SOURCE_BIDS_FMRI,
        status=True,
        output_json=True,
        subjects_cache=False,
        subjects_refresh=False,
    )

    out = io.StringIO()
    with redirect_stdout(out):
        _handle_subjects_mode(args, deriv_root, "task", config, logging.getLogger("test"))

    payload = json.loads(out.getvalue())
    ids = [subj["id"] for subj in payload["subjects"]]
    assert ids == ["0001", "0002"]
