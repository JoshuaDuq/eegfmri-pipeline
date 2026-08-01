"""What a study preflight reports about a dataset it has not started processing.

Issue #14: setting a study up meant discovering its shape by hand — how many subjects and
runs, whether every recording has events, whether the sampling rate is the same
throughout, whether the ECG channel the config names is really there. Each of those was
answerable only by opening files or by running preprocessing until something failed.

Two things this deliberately does not do. It does not open recordings: everything below
comes from BIDS metadata — the JSON sidecars, channels.tsv, events.tsv — which is where
the answers already are, and which keeps the check fast and free of MNE. And it does not
grade: a rate that differs across runs is reported as the two rates, not as a failure,
because whether that is wrong depends on the study and the pipeline does not know.
"""

from __future__ import annotations

import json

from eeg_pipeline.utils.data.preflight import Observation, run_preflight


class _Config:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def _config(tmp_path, **overrides):
    values = {
        "paths.bids_root": str(tmp_path / "bids"),
        "paths.deriv_root": str(tmp_path / "derivatives"),
        "project.task": "oddball",
        "project.paradigm": "task",
        "eeg.ecg_channels": [],
        "eeg.eog_channels": [],
        "preprocessing.notch_freq": 60,
        "event_columns.required": [],
    }
    values.update(overrides)
    return _Config(values)


def _run(
    tmp_path,
    subject="0001",
    task="oddball",
    run="01",
    *,
    sfreq=500.0,
    line=60.0,
    channels=(("Cz", "EEG"),),
    events=True,
    powerline=True,
):
    eeg_dir = tmp_path / "bids" / f"sub-{subject}" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject}_task-{task}_run-{run}"

    (eeg_dir / f"{stem}_eeg.vhdr").write_text("header\n", encoding="utf-8")

    sidecar = {"SamplingFrequency": sfreq}
    if powerline:
        sidecar["PowerLineFrequency"] = line
    (eeg_dir / f"{stem}_eeg.json").write_text(json.dumps(sidecar), encoding="utf-8")

    rows = ["name\ttype"] + [f"{n}\t{k}" for n, k in channels]
    (eeg_dir / f"{stem}_channels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if events:
        (eeg_dir / f"{stem}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0.5\t0.1\tgo\n", encoding="utf-8"
        )
    return eeg_dir


def _by_key(report) -> dict[str, Observation]:
    return {observation.key: observation for observation in report.observations}


###################################################################
# Roots
###################################################################


def test_a_missing_bids_root_is_reported_and_stops_the_inventory(tmp_path) -> None:
    """Nothing downstream can be measured, and reporting twenty absences from one
    missing directory buries the one fact that matters."""
    report = run_preflight(_config(tmp_path))

    assert _by_key(report)["paths.bids_root"].status == "absent"
    assert "inventory" not in _by_key(report)


def test_a_derivatives_directory_that_does_not_exist_yet_is_reported_as_creatable(
    tmp_path,
) -> None:
    """Its absence is the normal state before a first run. What matters is whether it
    can be made, which is a fact about the parent directory."""
    _run(tmp_path)

    observation = _by_key(run_preflight(_config(tmp_path)))["paths.deriv_root"]

    assert observation.status == "ok"
    assert "created" in observation.message


###################################################################
# Inventory
###################################################################


def test_subjects_tasks_and_runs_are_counted(tmp_path) -> None:
    _run(tmp_path, subject="0001", run="01")
    _run(tmp_path, subject="0001", run="02")
    _run(tmp_path, subject="0002", run="01")

    observation = _by_key(run_preflight(_config(tmp_path)))["inventory"]

    assert observation.status == "ok"
    assert "2 subjects" in observation.message
    assert "3 runs" in observation.message


def test_a_task_label_no_recording_carries_is_reported_against_what_is_there(
    tmp_path,
) -> None:
    """The commonest setup mistake, and the one that otherwise processes nothing without
    saying why."""
    _run(tmp_path, task="rest")

    observation = _by_key(run_preflight(_config(tmp_path, **{"project.task": "oddball"})))[
        "project.task"
    ]

    assert observation.status == "absent"
    assert "rest" in observation.message


###################################################################
# Pairing and consistency
###################################################################


def test_recordings_without_an_events_file_are_counted(tmp_path) -> None:
    _run(tmp_path, run="01", events=True)
    _run(tmp_path, run="02", events=False)

    observation = _by_key(run_preflight(_config(tmp_path)))["events_pairing"]

    assert observation.status == "differs"
    assert "1/2" in observation.message


def test_events_pairing_is_not_reported_for_a_resting_state_study(tmp_path) -> None:
    """Fixed-length segments are cut without events, so their absence is not a finding."""
    _run(tmp_path, task="rest", events=False)

    report = run_preflight(
        _config(tmp_path, **{"project.task": "rest", "project.paradigm": "rest"})
    )

    assert "events_pairing" not in _by_key(report)


def test_one_sampling_rate_across_runs_is_reported_as_that_rate(tmp_path) -> None:
    _run(tmp_path, run="01", sfreq=500.0)
    _run(tmp_path, run="02", sfreq=500.0)

    observation = _by_key(run_preflight(_config(tmp_path)))["sampling_frequency"]

    assert observation.status == "ok"
    assert "500" in observation.message


def test_sampling_rates_that_differ_are_reported_as_both_rates(tmp_path) -> None:
    """Stated, not judged: resampling may well be intended. The pipeline does not know,
    so it says what it found and leaves the decision where it belongs."""
    _run(tmp_path, run="01", sfreq=500.0)
    _run(tmp_path, run="02", sfreq=1000.0)

    observation = _by_key(run_preflight(_config(tmp_path)))["sampling_frequency"]

    assert observation.status == "differs"
    assert "500" in observation.message and "1000" in observation.message


def test_channel_layouts_that_differ_name_the_channels_that_differ(tmp_path) -> None:
    _run(tmp_path, run="01", channels=(("Cz", "EEG"), ("Pz", "EEG")))
    _run(tmp_path, run="02", channels=(("Cz", "EEG"),))

    observation = _by_key(run_preflight(_config(tmp_path)))["channel_layout"]

    assert observation.status == "differs"
    assert "Pz" in observation.message


###################################################################
# Declared channels and line frequency
###################################################################


def test_a_declared_ecg_channel_is_reported_per_run(tmp_path) -> None:
    _run(tmp_path, run="01", channels=(("Cz", "EEG"), ("ECG", "ECG")))
    _run(tmp_path, run="02", channels=(("Cz", "EEG"),))

    observation = _by_key(run_preflight(_config(tmp_path, **{"eeg.ecg_channels": ["ECG"]})))[
        "eeg.ecg_channels"
    ]

    assert observation.status == "differs"
    assert "1/2" in observation.message


def test_an_undeclared_ecg_channel_is_not_looked_for(tmp_path) -> None:
    _run(tmp_path, channels=(("Cz", "EEG"),))

    assert "eeg.ecg_channels" not in _by_key(run_preflight(_config(tmp_path)))


def test_a_line_frequency_disagreeing_with_the_notch_is_reported(tmp_path) -> None:
    """50 vs 60 Hz is the mistake a study inherits by copying a config across continents,
    and the recording states which one it was made under."""
    _run(tmp_path, line=50.0)

    observation = _by_key(run_preflight(_config(tmp_path, **{"preprocessing.notch_freq": 60})))[
        "preprocessing.notch_freq"
    ]

    assert observation.status == "differs"
    assert "50" in observation.message and "60" in observation.message


def test_a_line_frequency_matching_the_notch_is_confirmed(tmp_path) -> None:
    _run(tmp_path, line=60.0)

    observation = _by_key(run_preflight(_config(tmp_path, **{"preprocessing.notch_freq": 60})))[
        "preprocessing.notch_freq"
    ]

    assert observation.status == "ok"


###################################################################
# Event columns
###################################################################


def test_a_required_event_column_that_no_alias_matches_is_reported(tmp_path) -> None:
    _run(tmp_path)

    observation = _by_key(
        run_preflight(
            _config(
                tmp_path,
                **{
                    "event_columns.required": ["outcome"],
                    "event_columns.outcome": ["rating", "response"],
                },
            )
        )
    )["event_columns.outcome"]

    assert observation.status == "absent"
    assert "rating" in observation.message


def test_missing_values_in_a_required_event_column_are_counted(tmp_path) -> None:
    eeg_dir = _run(tmp_path)
    (eeg_dir / "sub-0001_task-oddball_run-01_events.tsv").write_text(
        "onset\tduration\trating\n0.5\t0.1\t3\n1.5\t0.1\tn/a\n", encoding="utf-8"
    )

    observation = _by_key(
        run_preflight(
            _config(
                tmp_path,
                **{
                    "event_columns.required": ["outcome"],
                    "event_columns.outcome": ["rating"],
                },
            )
        )
    )["event_columns.outcome"]

    assert observation.status == "differs"
    assert "1" in observation.message and "2" in observation.message


def test_a_required_column_that_is_complete_is_confirmed(tmp_path) -> None:
    eeg_dir = _run(tmp_path)
    (eeg_dir / "sub-0001_task-oddball_run-01_events.tsv").write_text(
        "onset\tduration\trating\n0.5\t0.1\t3\n1.5\t0.1\t4\n", encoding="utf-8"
    )

    observation = _by_key(
        run_preflight(
            _config(
                tmp_path,
                **{
                    "event_columns.required": ["outcome"],
                    "event_columns.outcome": ["rating"],
                },
            )
        )
    )["event_columns.outcome"]

    assert observation.status == "ok"


###################################################################
# The command reads nothing it was not asked to
###################################################################


def test_preflight_writes_nothing(tmp_path) -> None:
    """It runs before a study is set up, on data the user has not committed to
    processing. Creating the derivatives tree here would be the pipeline making the
    decision it exists to inform."""
    _run(tmp_path)
    before = {path for path in (tmp_path).rglob("*")}

    run_preflight(_config(tmp_path))

    assert {path for path in (tmp_path).rglob("*")} == before
