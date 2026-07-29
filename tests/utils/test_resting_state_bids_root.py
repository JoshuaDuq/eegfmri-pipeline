from __future__ import annotations

import io
import json
import logging
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from eeg_pipeline.cli.commands.info_helpers import SOURCE_BIDS, _handle_subjects_mode
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from eeg_pipeline.utils.data.subjects import get_available_subjects, resolve_eeg_bids_root
from tests.pipelines_test_utils import DotConfig


def test_resolve_eeg_bids_root_uses_rest_root_when_rest_mode_enabled() -> None:
    config = DotConfig(
        {
            "paths": {
                "bids_root": "/tmp/task",
                "bids_rest_root": "/tmp/rest",
            }
        }
    )

    resolved = resolve_eeg_bids_root(config, task_is_rest=True)

    assert resolved == Path("/tmp/rest")


def test_resolve_eeg_bids_root_falls_back_to_the_primary_root() -> None:
    """The rest-specific root is for a study that acquires *both* and keeps them apart.
    Requiring it made a rest-only study write the same path twice under a second name."""
    config = DotConfig({"paths": {"bids_root": "/tmp/task"}})

    assert resolve_eeg_bids_root(config, task_is_rest=True) == Path("/tmp/task")


def test_resolve_eeg_deriv_root_uses_rest_root_when_rest_mode_enabled() -> None:
    config = DotConfig(
        {
            "paths": {
                "deriv_root": "/tmp/derivatives-task",
                "deriv_rest_root": "/tmp/derivatives-rest",
            }
        }
    )

    resolved = resolve_eeg_deriv_root(config, task_is_rest=True)

    assert resolved == Path("/tmp/derivatives-rest")


def test_resolve_eeg_deriv_root_falls_back_to_the_primary_root() -> None:
    config = DotConfig({"paths": {"deriv_root": "/tmp/derivatives-task"}})

    assert resolve_eeg_deriv_root(config, task_is_rest=True) == Path("/tmp/derivatives-task")


@pytest.mark.parametrize(
    "resolve,key",
    [
        (resolve_eeg_bids_root, "paths.bids_root"),
        (resolve_eeg_deriv_root, "paths.deriv_root"),
    ],
)
def test_rest_mode_with_no_root_at_all_names_the_primary_key(resolve, key) -> None:
    """The fallback removed the demand for a second root, not the error for having none.
    What it must name is the key a rest-only study actually sets."""
    config = DotConfig({"paths": {}})

    with pytest.raises(ValueError, match=key):
        resolve(config, task_is_rest=True)


def test_get_available_subjects_uses_rest_root_in_rest_mode(tmp_path: Path) -> None:
    task_root = tmp_path / "bids_task"
    rest_root = tmp_path / "bids_rest"
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)
    (task_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    (rest_root / "sub-0002").mkdir(parents=True, exist_ok=True)

    config = DotConfig(
        {
            "project": {"subject_list": None},
            "paths": {
                "bids_root": str(task_root),
                "bids_rest_root": str(rest_root),
            },
            "preprocessing": {"task_is_rest": True},
        }
    )

    subjects = get_available_subjects(
        config=config,
        deriv_root=deriv_root,
        task="rest",
        task_is_rest=True,
        discovery_sources=["bids"],
        subject_discovery_policy="union",
    )

    assert subjects == ["0002"]


def test_get_available_subjects_uses_rest_deriv_root_by_default_in_rest_mode(
    tmp_path: Path,
) -> None:
    deriv_task_root = tmp_path / "derivatives-task"
    deriv_rest_root = tmp_path / "derivatives-rest"
    features_dir = deriv_rest_root / "sub-0002" / "eeg" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "features_power.tsv").write_text("power\n1.0\n", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"subject_list": None},
            "paths": {
                "deriv_root": str(deriv_task_root),
                "deriv_rest_root": str(deriv_rest_root),
            },
            "feature_engineering": {"task_is_rest": True},
        }
    )

    subjects = get_available_subjects(
        config=config,
        task="rest",
        task_is_rest=True,
        discovery_sources=["features"],
        subject_discovery_policy="union",
    )

    assert subjects == ["0002"]


def test_info_subjects_mode_uses_rest_root_when_rest_mode_enabled(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)

    task_root = tmp_path / "bids_task"
    rest_root = tmp_path / "bids_rest"
    (task_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    (rest_root / "sub-0002").mkdir(parents=True, exist_ok=True)

    config = DotConfig(
        {
            "project": {"subject_list": None},
            "paths": {
                "bids_root": str(task_root),
                "bids_rest_root": str(rest_root),
            },
            "preprocessing": {"task_is_rest": True},
        }
    )

    args = Namespace(
        source=SOURCE_BIDS,
        status=True,
        output_json=True,
        subjects_cache=False,
        subjects_refresh=False,
    )

    out = io.StringIO()
    with redirect_stdout(out):
        _handle_subjects_mode(args, deriv_root, "rest", config, logging.getLogger("test"))

    payload = json.loads(out.getvalue())
    ids = [subj["id"] for subj in payload["subjects"]]
    assert ids == ["0002"]
