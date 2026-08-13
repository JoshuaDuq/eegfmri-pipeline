"""The command that turns sidecars on disk into a document.

Its own decisions are few and each one can quietly produce a wrong cohort: which
participants to aggregate, where to look for them, and what to name a document assembled
from more than one task. Those are what this file pins; the aggregation itself is tested
against the package that does it.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.cli.commands import get_command  # noqa: E402
from eeg_pipeline.cli.commands.cohort_report_orchestrator import (  # noqa: E402
    run_cohort_report,
)
from eeg_pipeline.preprocessing.report.cohort.record import channel_table  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
    write_sidecar,
)

FREQUENCIES = [1.0, 2.0, 4.0, 8.0, 16.0]


def _sidecar(subject: str, *, task: str = "thermalactive") -> SubjectSidecar:
    runs = pd.DataFrame(
        {
            "run": [f"sub-{subject}_task-{task}_run-1"],
            "n_channels": [63],
            "duration_s": [600.0],
            "flagged_fraction": [0.02],
            "continuity_median_db": [0.2],
            "continuity_max_db": [5.0],
        }
    )
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "run": f"sub-{subject}_task-{task}_run-1",
                    "stage": stage,
                    "freq_hz": FREQUENCIES,
                    "median_db": [20.0 - 10.0 * index for index in range(len(FREQUENCIES))],
                    "max_db": [26.0 - 10.0 * index for index in range(len(FREQUENCIES))],
                }
            )
            for stage in ("before", "after")
        ],
        ignore_index=True,
    )
    return SubjectSidecar(
        subject=subject,
        task=task,
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=Paradigm.REST,
        measurements={"variance_removed": 0.84, "n_channels": 63},
        settings={"spectra_line_frequency": 60.0},
        versions={"mne": "1.12.1"},
        runs=runs,
        spectrum_curves=curves,
        channels=channel_table(
            positions={"Cz": (0.0, 0.0, 0.1), "O1": (-0.03, -0.08, 0.02)},
            bad_by_run={f"sub-{subject}_task-{task}_run-1": []},
        ),
    )


def _write_participant(eeg_root, subject: str, task: str) -> None:
    """One participant on disk: the report discovery looks for, and its sidecar."""
    directory = eeg_root / f"sub-{subject}" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / f"sub-{subject}_report.h5"
    report.write_text("", encoding="utf-8")
    write_sidecar(report, _sidecar(subject, task=task))


@pytest.fixture
def deriv_root(tmp_path):
    """A derivatives tree with two participants that have sidecars and one that does not."""
    eeg_root = tmp_path / "preprocessed" / "eeg"
    for subject, task in (("0014", "thermalactive"), ("0015", "thermalactive")):
        _write_participant(eeg_root, subject, task)
    bare = eeg_root / "sub-0016" / "eeg"
    bare.mkdir(parents=True, exist_ok=True)
    (bare / "sub-0016_report.h5").write_text("", encoding="utf-8")
    return tmp_path


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "subjects": None,
        "task": None,
        "deriv_root": None,
        "output_dir": None,
        # None as the parser now leaves them: unset means "take the config's value",
        # which is what distinguishes not passing a flag from passing its default.
        "min_subjects_for_median": None,
        "min_subjects_for_outer_band": None,
        "title": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_the_command_is_registered_and_needs_no_subject_list() -> None:
    """The default really is the whole cohort, so a missing list is not a usage error."""
    command = get_command("cohort-report")

    assert command is not None
    assert command.requires_subjects is False


def test_a_run_writes_the_document_its_log_and_its_audit_tables(deriv_root, tmp_path) -> None:
    output = tmp_path / "group"

    run_cohort_report(_args(deriv_root=str(deriv_root), output_dir=str(output)), [], None)

    assert (output / "task-thermalactive_desc-cohort_report.html").is_file()
    log = json.loads(
        (output / "task-thermalactive_desc-cohort_log.json").read_text(encoding="utf-8")
    )
    assert log["participants"] == ["0014", "0015"]
    assert list(output.glob("*_qc.tsv"))


def test_a_participant_without_a_sidecar_is_named_rather_than_dropped(
    deriv_root, tmp_path, capsys
) -> None:
    """A silently smaller cohort is indistinguishable from a correct one."""
    run_cohort_report(
        _args(deriv_root=str(deriv_root), output_dir=str(tmp_path / "group")), [], None
    )

    printed = capsys.readouterr().out

    assert "sub-0016" in printed
    assert "not aggregated" in printed


def test_an_explicit_subject_list_narrows_the_cohort(deriv_root, tmp_path) -> None:
    output = tmp_path / "group"

    run_cohort_report(
        _args(deriv_root=str(deriv_root), output_dir=str(output), subjects=["0014"]), [], None
    )

    log = json.loads(
        (output / "task-thermalactive_desc-cohort_log.json").read_text(encoding="utf-8")
    )
    assert log["participants"] == ["0014"]


def test_a_sub_prefix_on_a_requested_participant_is_accepted(deriv_root, tmp_path) -> None:
    output = tmp_path / "group"

    run_cohort_report(
        _args(deriv_root=str(deriv_root), output_dir=str(output), subjects=["sub-0014"]),
        [],
        None,
    )

    log = json.loads(
        (output / "task-thermalactive_desc-cohort_log.json").read_text(encoding="utf-8")
    )
    assert log["participants"] == ["0014"]


def test_a_requested_participant_that_does_not_exist_is_an_error(deriv_root, tmp_path) -> None:
    """A typo must not quietly produce a smaller cohort that looks correct."""
    with pytest.raises(ValueError, match="0099"):
        run_cohort_report(
            _args(
                deriv_root=str(deriv_root),
                output_dir=str(tmp_path / "group"),
                subjects=["0099"],
            ),
            [],
            None,
        )


def test_the_eeg_directory_may_be_given_directly(deriv_root, tmp_path) -> None:
    """Both the derivatives root and the EEG directory are things a user has in hand."""
    output = tmp_path / "group"

    run_cohort_report(
        _args(deriv_root=str(deriv_root / "preprocessed" / "eeg"), output_dir=str(output)),
        [],
        None,
    )

    assert (output / "task-thermalactive_desc-cohort_report.html").is_file()


def test_a_missing_derivatives_tree_says_what_to_do(tmp_path) -> None:
    with pytest.raises(NotADirectoryError, match="preprocessing report stage"):
        run_cohort_report(
            _args(deriv_root=str(tmp_path / "nothing"), output_dir=str(tmp_path)), [], None
        )


def test_a_two_task_cohort_is_not_named_after_one_of_them(tmp_path) -> None:
    """A filename naming one task would misdescribe a document covering two."""
    eeg_root = tmp_path / "preprocessed" / "eeg"
    for subject, task in (("0014", "thermalactive"), ("0015", "rest")):
        _write_participant(eeg_root, subject, task)
    output = tmp_path / "group"

    run_cohort_report(_args(deriv_root=str(tmp_path), output_dir=str(output)), [], None)

    assert (output / "cohort_desc-cohort_report.html").is_file()
    log = json.loads((output / "cohort_desc-cohort_log.json").read_text(encoding="utf-8"))
    assert sorted(log["tasks_found"]) == ["rest", "thermalactive"]


def test_the_default_output_lands_beside_the_participants(deriv_root) -> None:
    run_cohort_report(_args(deriv_root=str(deriv_root)), [], None)

    expected = deriv_root / "preprocessed" / "eeg" / "group"
    assert (expected / "task-thermalactive_desc-cohort_report.html").is_file()


def test_the_gates_reach_the_document(deriv_root, tmp_path) -> None:
    output = tmp_path / "group"

    run_cohort_report(
        _args(
            deriv_root=str(deriv_root),
            output_dir=str(output),
            min_subjects_for_median=6,
            min_subjects_for_outer_band=12,
        ),
        [],
        None,
    )

    log = json.loads(
        (output / "task-thermalactive_desc-cohort_log.json").read_text(encoding="utf-8")
    )
    assert log["band_gates"] == {
        "min_subjects_for_median": 6,
        "min_subjects_for_outer_band": 12,
    }


class _GateConfig:
    """A config carrying only the report block, read through dotted-key lookup."""

    def __init__(self, **thresholds) -> None:
        self._report = {"thresholds": dict(thresholds)}

    def get(self, key: str, default=None):
        return self._report if key == "report" else default


def test_the_gates_come_from_the_config_when_no_flag_was_given(deriv_root, tmp_path) -> None:
    """The gates a cohort was drawn under travel with the study, not with the command."""
    output = tmp_path / "group"

    run_cohort_report(
        _args(deriv_root=str(deriv_root), output_dir=str(output)),
        [],
        _GateConfig(min_subjects_for_median=7, min_subjects_for_outer_band=14),
    )

    log = json.loads(
        (output / "task-thermalactive_desc-cohort_log.json").read_text(encoding="utf-8")
    )
    assert log["band_gates"] == {
        "min_subjects_for_median": 7,
        "min_subjects_for_outer_band": 14,
    }


def test_a_flag_overrides_the_configured_gate(deriv_root, tmp_path) -> None:
    """The flags remain, for a one-off run against a different threshold."""
    output = tmp_path / "group"

    run_cohort_report(
        _args(
            deriv_root=str(deriv_root),
            output_dir=str(output),
            min_subjects_for_median=6,
        ),
        [],
        _GateConfig(min_subjects_for_median=7, min_subjects_for_outer_band=14),
    )

    log = json.loads(
        (output / "task-thermalactive_desc-cohort_log.json").read_text(encoding="utf-8")
    )
    # The flag moved its own gate; the one not passed still came from the config.
    assert log["band_gates"] == {
        "min_subjects_for_median": 6,
        "min_subjects_for_outer_band": 14,
    }


def test_a_configured_gate_that_would_extrapolate_is_refused_like_a_flag(
    deriv_root, tmp_path
) -> None:
    """Config is not a way around the arithmetic the flags are checked against."""
    with pytest.raises(ValueError, match="extrapolate"):
        run_cohort_report(
            _args(deriv_root=str(deriv_root), output_dir=str(tmp_path / "group")),
            [],
            _GateConfig(min_subjects_for_median=2),
        )


def test_a_gate_that_would_extrapolate_a_quantile_is_refused(deriv_root, tmp_path) -> None:
    """No setting may produce a band the sample cannot support."""
    with pytest.raises(ValueError, match="extrapolate"):
        run_cohort_report(
            _args(
                deriv_root=str(deriv_root),
                output_dir=str(tmp_path / "group"),
                min_subjects_for_median=2,
            ),
            [],
            None,
        )
