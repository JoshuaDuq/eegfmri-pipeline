"""The sidecar is the contract between the subject stage and the cohort report.

Every way it can be wrong -- a moved file, a stale layout, an absent column, an
acquisition that produced no gradient evidence -- reaches a cohort figure as a plausible
number rather than as an error, unless it is caught here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    COMB_COLUMNS,
    RUN_COLUMNS,
    SCHEMA_VERSION,
    SPECTRUM_COLUMNS,
    Paradigm,
    SubjectSidecar,
    has_sidecar,
    read_sidecar,
    sidecar_paths,
    write_sidecar,
)


def _runs(n_runs: int = 2, *, in_scanner: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [720.0] * n_runs,
            "flagged_fraction": [0.04] * n_runs,
            "continuity_median_db": [0.5] * n_runs,
            "continuity_max_db": [6.0] * n_runs,
        }
    )
    if in_scanner:
        frame["n_volumes"] = [360] * n_runs
        frame["repetition_time_s"] = [2.0] * n_runs
        frame["volume_jitter_s"] = [0.003] * n_runs
        frame["volume_locked_rms_before_uv"] = [2.0] * n_runs
        frame["volume_locked_floor_before_uv"] = [0.5] * n_runs
        frame["volume_locked_excess_power_before_uv2"] = [3.75] * n_runs
        frame["volume_locked_resolved_before"] = [True] * n_runs
        frame["volume_locked_rms_after_uv"] = [0.35] * n_runs
        frame["volume_locked_floor_after_uv"] = [0.4] * n_runs
        frame["volume_locked_excess_power_after_uv2"] = [-0.0375] * n_runs
        frame["volume_locked_resolved_after"] = [False] * n_runs
        frame["median_bpm"] = [62.0] * n_runs
        frame["n_beats"] = [700.0] * n_runs
        frame["beat_dropouts"] = [2.0] * n_runs
        frame["marker_matched_fraction"] = [0.96] * n_runs
        frame["marker_median_lag_s"] = [0.004] * n_runs
        frame["marker_lag_iqr_s"] = [0.010] * n_runs
        frame["n_markers"] = [704.0] * n_runs
        frame["n_detected_beats"] = [700.0] * n_runs
        frame["n_matched_beats"] = [672.0] * n_runs
        frame["pulse_marker_count"] = [704.0] * n_runs
        frame["beat_source"] = ["ECG"] * n_runs
        frame["bcg_residual_uv"] = [0.8] * n_runs
        frame["bcg_beat_train_coverage"] = [0.98] * n_runs
        frame["bcg_noise_floor_uv"] = [0.5] * n_runs
        frame["bcg_excess_power_uv2"] = [0.39] * n_runs
        frame["bcg_resolved"] = [True] * n_runs
        frame["bcg_n_beats"] = [700.0] * n_runs
    return frame


def _spectra() -> pd.DataFrame:
    rows = []
    for stage in ("before", "after"):
        for frequency in (1.0, 2.0, 3.0):
            rows.append(
                {
                    "run": "run-1",
                    "stage": stage,
                    "freq_hz": frequency,
                    "median_db": -10.0,
                    "max_db": -5.0,
                }
            )
    return pd.DataFrame(rows)


def _comb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": ["run-1", "run-1"],
            "harmonic_index": [1, 2],
            "harmonic_hz": [21.5, 43.0],
            "notched": False,
            "before_excess_db_median": [12.0, 8.0],
            "before_excess_db_max": [20.0, 14.0],
            "after_excess_db_median": [2.0, 1.0],
            "after_excess_db_max": [4.0, 3.0],
        }
    )


def _sidecar(**overrides) -> SubjectSidecar:
    defaults = dict(
        subject="0014",
        task="thermalactive",
        paradigm=Paradigm.TASK,
        measurements={"variance_removed": 0.856, "n_excluded": 29},
        settings={"comb_welch_seconds": 8.0},
        versions={"mne": "1.12.1"},
        acquisition_date="2026-03-04",
        written_at="2026-07-26T16:46:38+00:00",
        runs=_runs(),
        spectrum_curves=_spectra(),
        comb_curves=_comb(),
    )
    defaults.update(overrides)
    return SubjectSidecar(**defaults)


def _report(tmp_path, subject: str = "0014"):
    return tmp_path / f"sub-{subject}_report.h5"


def test_a_sidecar_survives_a_round_trip(tmp_path) -> None:
    report = _report(tmp_path)

    write_sidecar(report, _sidecar())
    restored = read_sidecar(report)

    assert restored.subject == "0014"
    assert restored.task == "thermalactive"
    assert restored.paradigm is Paradigm.TASK
    assert restored.measurements["variance_removed"] == pytest.approx(0.856)
    assert restored.versions == {"mne": "1.12.1"}
    assert restored.acquisition_date == "2026-03-04"
    assert restored.n_runs == 2
    assert_frame_equal(restored.runs, _runs())
    assert_frame_equal(restored.comb_curves, _comb())


def test_paths_are_derived_from_the_report_they_describe(tmp_path) -> None:
    """A sidecar cannot be separated from its report by a configuration mistake."""
    paths = sidecar_paths(tmp_path / "sub-0014_report.h5")

    assert paths.subject_json.name == "sub-0014_desc-qcsubject.json"
    assert paths.runs.name == "sub-0014_desc-qcruns.tsv"
    assert paths.comb_curves.name == "sub-0014_desc-qccomb_curves.tsv"


def test_an_eeg_only_participant_writes_no_gradient_tables(tmp_path) -> None:
    """An absent measurement and a failed one must not look the same on disk."""
    report = _report(tmp_path)

    paths = write_sidecar(
        report,
        _sidecar(
            runs=_runs(in_scanner=False),
            comb_curves=_empty_comb(),
        ),
    )

    assert not paths.comb_curves.exists()
    assert paths.runs.exists()

    restored = read_sidecar(report)
    assert restored.comb_curves.empty
    assert tuple(restored.comb_curves.columns) == COMB_COLUMNS
    assert not restored.has_comb_evidence


def _empty_comb() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in COMB_COLUMNS})


def test_a_scanner_participant_reports_gradient_evidence(tmp_path) -> None:
    report = _report(tmp_path)

    write_sidecar(report, _sidecar())

    assert read_sidecar(report).has_comb_evidence


def test_a_participant_without_a_sidecar_is_a_question_not_an_exception(tmp_path) -> None:
    """A report predating this feature is listed, not raised."""
    assert not has_sidecar(_report(tmp_path))


def test_a_written_sidecar_is_reported_as_present(tmp_path) -> None:
    report = _report(tmp_path)

    write_sidecar(report, _sidecar())

    assert has_sidecar(report)


def test_a_stale_schema_is_refused_rather_than_pooled(tmp_path) -> None:
    """Columns whose meaning changed would produce a wrong figure that looks right."""
    report = _report(tmp_path)
    paths = write_sidecar(report, _sidecar())
    document = json.loads(paths.subject_json.read_text(encoding="utf-8"))
    document["schema_version"] = SCHEMA_VERSION + 1
    paths.subject_json.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_sidecar(report)


def test_a_sidecar_describing_another_participant_is_refused(tmp_path) -> None:
    """The one failure that would silently double-count somebody."""
    report = _report(tmp_path)
    paths = write_sidecar(report, _sidecar())
    document = json.loads(paths.subject_json.read_text(encoding="utf-8"))
    document["subject"] = "0015"
    paths.subject_json.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="moved or renamed"):
        read_sidecar(report)


def test_a_missing_column_is_named(tmp_path) -> None:
    report = _report(tmp_path)
    paths = write_sidecar(report, _sidecar())
    _runs().drop(columns=["duration_s"]).to_csv(paths.runs, sep="\t", index=False)

    with pytest.raises(ValueError, match="duration_s"):
        read_sidecar(report)


def test_a_missing_required_table_is_raised(tmp_path) -> None:
    report = _report(tmp_path)
    paths = write_sidecar(report, _sidecar())
    paths.spectrum_curves.unlink()

    with pytest.raises(FileNotFoundError, match="qcspectrum"):
        read_sidecar(report)


def test_unparseable_json_is_raised_rather_than_replaced(tmp_path) -> None:
    report = _report(tmp_path)
    paths = write_sidecar(report, _sidecar())
    paths.subject_json.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        read_sidecar(report)


def test_required_columns_are_the_documented_ones() -> None:
    """The schema is a contract, so its shape is asserted rather than assumed."""
    assert RUN_COLUMNS[:3] == ("run", "n_channels", "duration_s")
    assert "flagged_fraction" in RUN_COLUMNS
    assert SPECTRUM_COLUMNS[:3] == ("run", "stage", "freq_hz")


def test_a_required_continuity_column_must_survive_the_round_trip(tmp_path) -> None:
    """The silent failure this guard exists for.

    A writer that omitted a required reduction would produce a sidecar that reads back
    without complaint and quietly removes its participant from the panels built on it,
    which they report only as a smaller denominator nobody has reason to question.
    """
    report = _report(tmp_path)
    paths = write_sidecar(report, _sidecar())
    _runs().drop(columns=["continuity_median_db"]).to_csv(paths.runs, sep="\t", index=False)

    with pytest.raises(ValueError, match="continuity_median_db"):
        read_sidecar(report)


def test_an_infinite_table_value_is_rejected_before_writing(tmp_path) -> None:
    report = _report(tmp_path)
    runs = _runs()
    runs.loc[0, "continuity_max_db"] = float("inf")

    with pytest.raises(ValueError, match="non-finite.*continuity_max_db"):
        write_sidecar(report, _sidecar(runs=runs))

    assert not sidecar_paths(report).subject_json.exists()


def test_nonstandard_json_numbers_are_rejected_before_writing(tmp_path) -> None:
    report = _report(tmp_path)

    with pytest.raises(ValueError, match="JSON"):
        write_sidecar(
            report,
            _sidecar(measurements={"variance_removed": float("inf")}),
        )

    assert not sidecar_paths(report).subject_json.exists()


def test_a_sidecar_carrying_only_the_required_columns_is_accepted(tmp_path) -> None:
    """The contract is the required columns; extra acquisition scalars are optional."""
    report = _report(tmp_path)

    write_sidecar(report, _sidecar(runs=_runs(in_scanner=False)))

    assert read_sidecar(report).n_runs == 2


def test_a_writer_omitting_a_required_column_fails_before_touching_disk(tmp_path) -> None:
    """Named at the stage that is wrong, rather than weeks later at a cohort run."""
    report = _report(tmp_path)

    with pytest.raises(ValueError, match="continuity_max_db"):
        write_sidecar(report, _sidecar(runs=_runs().drop(columns=["continuity_max_db"])))

    assert not sidecar_paths(report).runs.exists()


def test_schema_version_is_four_and_scanner_columns_are_gone():
    from eeg_pipeline.preprocessing.report.cohort import sidecar

    assert sidecar.SCHEMA_VERSION == 4
    assert not hasattr(sidecar, "SCANNER_RUN_COLUMNS")
    assert not hasattr(sidecar, "AcquisitionContext")
