from __future__ import annotations

from argparse import Namespace

from eeg_pipeline.cli.common import get_deriv_root
from eeg_pipeline.cli.main import get_subjects_for_command, update_config_from_args
from tests.pipelines_test_utils import DotConfig


def test_update_config_from_args_applies_rest_mode_before_subject_discovery(tmp_path) -> None:
    task_root = tmp_path / "bids-task"
    rest_root = tmp_path / "bids-rest"
    deriv_task_root = tmp_path / "derivatives-task"
    deriv_rest_root = tmp_path / "derivatives-rest"

    (task_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    (rest_root / "sub-0002").mkdir(parents=True, exist_ok=True)

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {
                "bids_root": str(task_root),
                "bids_rest_root": str(rest_root),
                "deriv_root": str(deriv_task_root),
                "deriv_rest_root": str(deriv_rest_root),
            },
            "preprocessing": {"task_is_rest": False},
            "feature_engineering": {"task_is_rest": False},
        }
    )
    args = Namespace(
        command="preprocessing",
        mode="full",
        task="rest",
        task_is_rest=True,
        source_root=None,
        bids_root=None,
        bids_rest_root=None,
        bids_fmri_root=None,
        deriv_root=None,
        deriv_rest_root=None,
        set_overrides=None,
        group=None,
        all_subjects=True,
        subject=None,
        subjects=None,
        source=None,
    )

    update_config_from_args(config, args)
    deriv_root = get_deriv_root(config, command=args.command)
    subjects = get_subjects_for_command(args, config, deriv_root)

    assert deriv_root == deriv_rest_root
    assert subjects == ["0002"]
