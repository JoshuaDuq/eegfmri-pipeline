"""The seam: the measuring pass has to write down what a cohort will read.

``measure_runs`` reads every run once, applies the ICA exclusions and measures everything,
which is by a wide margin the most expensive thing the pipeline does. If those results are
not written at the moment they are in memory, a cohort document can only be built by paying
that cost again per participant from a gigabyte of filtered raw each.

So this file pins that the review stage writes the sidecar, that it writes it from the
evidence rather than recomputing anything, and that a failure to write one fails the stage.
Otherwise a successful-looking subject report silently becomes unusable by the cohort.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    AcquisitionContext,
    Paradigm,
    has_sidecar,
    read_sidecar,
)
from eeg_pipeline.preprocessing.report.continuity import RunContinuity
from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.preprocessing.report.spectra import RunSpectra, StageSpectrum
from tests.utils.pipelines_test_utils import DotConfig

RUN = "sub-0014_task-thermalactive_run-1"
FREQUENCIES = np.asarray([1.0, 2.0, 3.0])


def _stage() -> StageSpectrum:
    return StageSpectrum(
        median_db=np.full(3, -10.0),
        spread_low_db=np.full(3, -12.0),
        spread_high_db=np.full(3, -8.0),
        max_db=np.full(3, -5.0),
        aperiodic=None,
    )


def _evidence(*, events=(1.0, 2.0), positions=None, dates=None) -> SimpleNamespace:
    return SimpleNamespace(
        spectra=[
            RunSpectra(
                recording_id=RUN,
                frequencies=FREQUENCIES,
                before=_stage(),
                after=_stage(),
                n_channels=63,
                fmax_reason="low-pass",
            )
        ],
        continuity=[
            RunContinuity(
                recording_id=RUN,
                window_seconds=1.0,
                times_s=np.arange(10.0),
                channel_names=("Cz",),
                relative_db=np.zeros((1, 10)) + 3.0,
                bad_spans=((0.0, 30.0),),
                duration_s=600.0,
                event_onsets=tuple(events),
            )
        ],
        timings={},
        locked_averages=[],
        combs=[],
        rr_intervals=[],
        marker_agreements=[],
        cardiac_residuals=[],
        channel_positions=positions or {"Cz": (0.0, 0.0, 0.1)},
        bad_channels_by_run={RUN: ("TP9",)},
        measurement_dates=list(dates or ["2026-03-04"]),
        acquisition_date=(dates or ["2026-03-04"])[0] if dates is not False else None,
    )


def _record() -> dict:
    return {
        "stages": [
            {
                "stage": "epochs",
                "written_at": "2026-03-05T10:00:00+00:00",
                "versions": {"mne": "1.12.1", "eeg_pipeline": "1.0.0"},
                "measurements": {"epochs_total": 120, "epochs_kept": 100},
            },
            {
                "stage": "report-review",
                "written_at": "2026-03-05T11:00:00+00:00",
                "versions": {"mne": "1.12.1", "eeg_pipeline": "1.0.0"},
                "measurements": {"n_runs": 1, "variance_removed": 0.86},
            },
        ]
    }


@pytest.fixture
def pipeline(tmp_path):
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

    cfg = DotConfig(
        {
            "paths": {"bids_root": str(tmp_path / "bids"), "deriv_root": str(tmp_path / "deriv")},
            "preprocessing": {"task_is_rest": False},
        }
    )

    def _init(self, name, config=None):
        self.name = name
        self.config = config or cfg
        self.logger = Mock()

    with patch("eeg_pipeline.pipelines.preprocessing.PipelineBase.__init__", _init):
        built = PreprocessingPipeline(config=cfg)
    built.deriv_root = tmp_path / "deriv"
    return built


@pytest.fixture
def report_path(tmp_path):
    path = tmp_path / "deriv" / "preprocessed" / "eeg" / "sub-0014" / "eeg"
    path.mkdir(parents=True, exist_ok=True)
    return path / "sub-0014_report.h5"


def test_the_review_stage_writes_a_sidecar_a_cohort_can_read(pipeline, report_path) -> None:
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(),
        settings=ReportSettings(),
    )

    assert has_sidecar(report_path)
    sidecar = read_sidecar(report_path)
    assert sidecar.subject == "0014"
    assert sidecar.task == "thermalactive"
    assert sidecar.n_runs == 1


def test_the_sidecar_carries_the_measurements_the_stages_recorded(pipeline, report_path) -> None:
    """Copied from the build record, so the cohort and the landing panel agree."""
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(),
        settings=ReportSettings(),
    )

    measurements = read_sidecar(report_path).measurements

    assert measurements["variance_removed"] == 0.86
    assert measurements["epochs_kept"] == 100


def test_the_sidecar_carries_the_settings_that_shape_the_numbers(pipeline, report_path) -> None:
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(),
        settings=ReportSettings(continuity_window_seconds=4.0),
    )

    assert read_sidecar(report_path).settings["continuity_window_seconds"] == 4.0


def test_the_sidecar_carries_the_versions_that_measured(pipeline, report_path) -> None:
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(),
        settings=ReportSettings(),
    )

    assert read_sidecar(report_path).versions["mne"] == "1.12.1"


def test_the_sensor_positions_come_from_the_recording(pipeline, report_path) -> None:
    """A montage name would place them plausibly and, where it was wrong, silently wrongly."""
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(positions={"Cz": (0.0, 0.0, 0.1), "O1": (-0.03, -0.08, 0.02)}),
        settings=ReportSettings(),
    )

    channels = read_sidecar(report_path).channels

    assert set(channels["channel"]) == {"Cz", "O1"}
    assert channels.set_index("channel").loc["Cz", "z"] == pytest.approx(0.1)
    # The channel bad in a run is recorded as bad in one run, not as bad everywhere.
    assert channels.set_index("channel")["n_runs_bad"].sum() == 0


def test_an_acquisition_date_is_recorded_separately_from_the_processing_date(
    pipeline, report_path
) -> None:
    """One indexes cap ageing, the other pipeline change; conflating them ruins a drift panel."""
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(dates=["2026-03-04"]),
        settings=ReportSettings(),
    )

    assert read_sidecar(report_path).acquisition_date == "2026-03-04"


def test_an_eeg_only_recording_is_classified_from_its_own_evidence(
    pipeline, report_path
) -> None:
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=_evidence(),
        settings=ReportSettings(),
    )

    sidecar = read_sidecar(report_path)

    assert sidecar.context is AcquisitionContext.OUT_OF_SCANNER
    assert sidecar.paradigm is Paradigm.TASK


def test_a_resting_state_recording_classifies_itself_as_one(pipeline, report_path) -> None:
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="rest",
        evidence=_evidence(events=()),
        settings=ReportSettings(),
    )

    assert read_sidecar(report_path).paradigm is Paradigm.REST


def test_a_stage_with_no_run_evidence_writes_no_sidecar(pipeline, report_path) -> None:
    """A participant with nothing measured is listed as not aggregated, not faked."""
    pipeline._write_qc_sidecar(
        report_path=report_path,
        record=_record(),
        subject="0014",
        task="thermalactive",
        evidence=None,
        settings=ReportSettings(),
    )

    assert not has_sidecar(report_path)


@pytest.mark.parametrize("failure", [OSError("disk full"), ValueError("missing column")])
def test_a_sidecar_failure_surfaces_at_the_stage_that_created_it(
    pipeline, report_path, failure
) -> None:
    with patch(
        "eeg_pipeline.preprocessing.report.cohort.sidecar.write_sidecar", side_effect=failure
    ):
        with pytest.raises(type(failure), match=str(failure)):
            pipeline._write_qc_sidecar(
                report_path=report_path,
                record=_record(),
                subject="0014",
                task="thermalactive",
                evidence=_evidence(),
                settings=ReportSettings(),
            )

    assert not has_sidecar(report_path)


def test_the_condition_counts_come_from_the_two_files_that_define_them(
    pipeline, report_path
) -> None:
    """Presented from the pre-rejection epochs, retained from the clean ones."""
    with patch.object(
        type(pipeline),
        "_epoch_condition_counts",
        side_effect=[{"painful": 60, "neutral": 60}, {"painful": 42, "neutral": 58}],
    ):
        presented, retained = pipeline._condition_counts(
            report_path=report_path, task="thermalactive"
        )

    assert presented == {"painful": 60, "neutral": 60}
    assert retained == {"painful": 42, "neutral": 58}


def test_an_absent_epochs_file_contributes_no_counts_rather_than_zeros(pipeline) -> None:
    assert pipeline._epoch_condition_counts(None) == {}
    assert pipeline._epoch_condition_counts(Path("/nonexistent/sub-0014_epo.fif")) == {}


def test_a_hierarchical_condition_is_not_counted_at_every_level(pipeline, tmp_path) -> None:
    """``epochs['painful']`` resolves against every level, so the counts would over-sum."""
    import mne

    info = mne.create_info(["Cz", "Pz"], 100.0, "eeg")
    events = np.column_stack(
        [np.arange(6) * 100 + 50, np.zeros(6, int), np.asarray([1, 1, 2, 2, 2, 3])]
    )
    epochs = mne.EpochsArray(
        np.zeros((6, 2, 20)),
        info,
        events=events,
        event_id={"painful/left": 1, "painful/right": 2, "neutral/left": 3},
        tmin=0.0,
        verbose="ERROR",
    )
    path = tmp_path / "sub-0014_task-t_epo.fif"
    epochs.save(path, overwrite=True, verbose="ERROR")

    counts = pipeline._epoch_condition_counts(path)

    assert counts == {"painful/left": 2, "painful/right": 3, "neutral/left": 1}
    assert sum(counts.values()) == len(events)
