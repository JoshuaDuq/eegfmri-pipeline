from __future__ import annotations

from argparse import Namespace

from eeg_pipeline.cli.common import get_deriv_root
from eeg_pipeline.cli.main import get_subjects_for_command, update_config_from_args
from tests.utils.pipelines_test_utils import DotConfig


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


def test_ml_all_subjects_discovers_feature_subjects_by_default(tmp_path) -> None:
    deriv_root = tmp_path / "derivatives"
    features_dir = deriv_root / "sub-0002" / "eeg" / "features" / "power"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "features_power.tsv").write_text("power\n1.0\n", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {"deriv_root": str(deriv_root)},
        }
    )
    args = Namespace(
        command="ml",
        mode="regression",
        task="task",
        task_is_rest=None,
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

    subjects = get_subjects_for_command(args, config, deriv_root)

    assert subjects == ["0002"]


def test_ml_all_subjects_discovers_epoch_subjects_for_channels_mean_feature_set(tmp_path) -> None:
    deriv_root = tmp_path / "derivatives"
    epochs_path = deriv_root / "sub-0005" / "eeg" / "sub-0005_task-task_proc-clean_epo.fif"
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_path.write_text("epochs", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {"deriv_root": str(deriv_root)},
            "machine_learning": {"data": {"feature_set": "channels_mean"}},
        }
    )
    args = Namespace(
        command="ml",
        mode="regression",
        task="task",
        task_is_rest=None,
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
        classification_model=None,
    )

    subjects = get_subjects_for_command(args, config, deriv_root)

    assert subjects == ["0005"]


def test_ml_all_subjects_rejects_feature_families_with_channels_mean_feature_set(tmp_path) -> None:
    deriv_root = tmp_path / "derivatives"
    epochs_path = deriv_root / "sub-0005" / "eeg" / "sub-0005_task-task_proc-clean_epo.fif"
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_path.write_text("epochs", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {"deriv_root": str(deriv_root)},
            "machine_learning": {"data": {"feature_set": "channels_mean"}},
        }
    )
    args = Namespace(
        command="ml",
        mode="regression",
        task="task",
        task_is_rest=None,
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
        feature_families=["power"],
        classification_model=None,
    )

    try:
        get_subjects_for_command(args, config, deriv_root)
    except ValueError as exc:
        assert "channels_mean" in str(exc)
    else:
        raise AssertionError("Expected channels_mean/feature_families conflict to raise.")


def test_ml_explicit_feature_source_rejected_for_channels_mean_feature_set(tmp_path) -> None:
    deriv_root = tmp_path / "derivatives"
    features_dir = deriv_root / "sub-0005" / "eeg" / "features" / "power"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "features_power.tsv").write_text("power\n1.0\n", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {"deriv_root": str(deriv_root)},
            "machine_learning": {"data": {"feature_set": "channels_mean"}},
        }
    )
    args = Namespace(
        command="ml",
        mode="regression",
        task="task",
        task_is_rest=None,
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
        source="features",
        feature_families=None,
        classification_model=None,
    )

    try:
        get_subjects_for_command(args, config, deriv_root)
    except ValueError as exc:
        assert "channels_mean" in str(exc)
    else:
        raise AssertionError("Expected channels_mean/features source conflict to raise.")


def test_ml_timegen_all_subjects_discovers_epoch_subjects(tmp_path) -> None:
    deriv_root = tmp_path / "derivatives"
    epochs_path = deriv_root / "sub-0003" / "eeg" / "sub-0003_task-task_proc-clean_epo.fif"
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_path.write_text("epochs", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {"deriv_root": str(deriv_root)},
        }
    )
    args = Namespace(
        command="ml",
        mode="timegen",
        classification_model=None,
        task="task",
        task_is_rest=None,
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

    subjects = get_subjects_for_command(args, config, deriv_root)

    assert subjects == ["0003"]


def test_ml_classify_cnn_all_subjects_discovers_epoch_subjects(tmp_path) -> None:
    deriv_root = tmp_path / "derivatives"
    epochs_path = deriv_root / "sub-0004" / "eeg" / "sub-0004_task-task_proc-clean_epo.fif"
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_path.write_text("epochs", encoding="utf-8")

    config = DotConfig(
        {
            "project": {"task": "task", "subject_list": None},
            "paths": {"deriv_root": str(deriv_root)},
        }
    )
    args = Namespace(
        command="ml",
        mode="classify",
        classification_model="cnn",
        task="task",
        task_is_rest=None,
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

    subjects = get_subjects_for_command(args, config, deriv_root)

    assert subjects == ["0004"]
