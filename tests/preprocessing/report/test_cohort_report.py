"""The document, its audit tables and its log, built from one cohort.

The assembler's own failures are structural rather than numerical: a section that renders
for a cohort it does not apply to, an audit file written empty, or a log that does not
record why a panel drew no median.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.collect import (  # noqa: E402
    Cohort,
    NotAggregated,
)
from eeg_pipeline.preprocessing.report.cohort.homogeneity import (  # noqa: E402
    homogeneity_html,
    setting_disagreements,
    version_disagreements,
)
from eeg_pipeline.preprocessing.report.cohort.record import channel_table  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.report import (  # noqa: E402
    build_cohort_report,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)

FREQUENCIES = np.round(np.logspace(np.log10(1.0), np.log10(60.0), 40), 4)
HARMONICS = np.asarray([30, 31, 32, 33])
POSITIONS = {
    "Fp1": (-0.03, 0.08, 0.02),
    "C3": (-0.05, 0.0, 0.08),
    "Cz": (0.0, 0.0, 0.1),
    "O1": (-0.03, -0.08, 0.02),
}


def _runs(n_runs: int, *, in_scanner: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [600.0] * n_runs,
            # Varying across runs, because a grid whose every cell agrees is not a grid:
            # the continuity section withholds it rather than drawing one flat colour.
            "flagged_fraction": [0.02 + 0.01 * index for index in range(n_runs)],
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [5.0 + index for index in range(n_runs)],
            "aperiodic_exponent_before": [1.4] * n_runs,
            "aperiodic_exponent_after": [1.35] * n_runs,
            "aperiodic_offset_db_before": [22.0] * n_runs,
            "aperiodic_offset_db_after": [21.0] * n_runs,
            "aperiodic_r_squared_before": [0.98] * n_runs,
            "aperiodic_r_squared_after": [0.97] * n_runs,
        }
    )
    if in_scanner:
        frame["n_volumes"] = [300] * n_runs
        frame["repetition_time_s"] = [2.0] * n_runs
        frame["volume_jitter_s"] = [0.003] * n_runs
        frame["volume_locked_rms_before_uv"] = [1.2] * n_runs
        frame["volume_locked_floor_before_uv"] = [0.4] * n_runs
        frame["volume_locked_excess_power_before_uv2"] = [1.28] * n_runs
        frame["volume_locked_resolved_before"] = [True] * n_runs
        frame["volume_locked_rms_after_uv"] = [0.76] * n_runs
        frame["volume_locked_floor_after_uv"] = [0.3] * n_runs
        frame["volume_locked_excess_power_after_uv2"] = [0.49] * n_runs
        frame["volume_locked_resolved_after"] = [True] * n_runs
        # The beat detectors only exist where there was a pulse correction to check.
        frame["marker_matched_fraction"] = [0.97] * n_runs
        frame["marker_median_lag_s"] = [0.004] * n_runs
        frame["marker_lag_iqr_s"] = [0.010] * n_runs
        frame["n_markers"] = [600.0] * n_runs
        frame["n_detected_beats"] = [608.0] * n_runs
        frame["median_bpm"] = [62.0] * n_runs
        frame["n_beats"] = [600.0] * n_runs
        frame["beat_dropouts"] = [3.0] * n_runs
    return frame


def _spectra(n_runs: int) -> pd.DataFrame:
    frames = []
    for run in range(n_runs):
        for stage, offset in (("before", 4.0), ("after", 0.0)):
            median = 20.0 + offset - 10.0 * np.log10(FREQUENCIES)
            frames.append(
                pd.DataFrame(
                    {
                        "run": f"run-{run + 1}",
                        "stage": stage,
                        "freq_hz": FREQUENCIES,
                        "median_db": median,
                        "max_db": median + 6.0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _comb(n_runs: int) -> pd.DataFrame:
    frames = []
    for run in range(n_runs):
        frames.append(
            pd.DataFrame(
                {
                    "run": f"run-{run + 1}",
                    "harmonic_index": HARMONICS,
                    "harmonic_hz": HARMONICS / 2.0,
                    "notched": False,
                    "before_excess_db_median": np.full(HARMONICS.size, 14.0),
                    "before_excess_db_max": np.full(HARMONICS.size, 20.0),
                    "after_excess_db_median": np.full(HARMONICS.size, 2.0),
                    "after_excess_db_max": np.full(HARMONICS.size, 4.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _participant(
    subject: str,
    *,
    in_scanner: bool = True,
    paradigm: Paradigm = Paradigm.TASK,
    n_runs: int = 2,
    versions: dict | None = None,
    settings: dict | None = None,
    bad: dict | None = None,
) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=(
            AcquisitionContext.IN_SCANNER if in_scanner else AcquisitionContext.OUT_OF_SCANNER
        ),
        paradigm=paradigm,
        measurements={
            "n_channels": 63,
            "n_components": 62,
            "data_rank": 62,
            "n_excluded": 29,
            "retained_dimensions": 33,
            "variance_removed": 0.86,
            "samples_per_squared_component": 188.0,
            "condition_number": 245.0,
            "n_excluded_eye": 2,
            "n_excluded_heart": 19,
            "n_excluded_muscle": 8,
            "alpha_prominence_db_before": 8.0,
            "alpha_prominence_db_after": 6.0,
            "alpha_peak_resolvable_after": True,
            "alpha_peak_frequency_hz_after": 10.2,
            # Trial-level measurements only where there were trials. The epochs and
            # reliability stages do not run on a resting-state recording, so a rest sidecar
            # does not carry their keys and a fixture that gave it them would be testing
            # section-dropping against a participant that cannot occur.
            **(
                {
                    "split_half_r": 0.74,
                    "split_half_n_trials": 80,
                    "epochs_total": 120,
                    "epochs_kept": 100,
                    "epochs_dropped_AUTOREJECT": 20,
                }
                if paradigm is Paradigm.TASK
                else {}
            ),
        },
        settings=settings or {"spectra_line_frequency": 60.0},
        versions=versions or {"eeg_pipeline": "1.0.0", "mne": "1.12.1"},
        acquisition_date="2026-03-04",
        runs=_runs(n_runs, in_scanner=in_scanner),
        spectrum_curves=_spectra(n_runs),
        comb_curves=_comb(n_runs) if in_scanner else pd.DataFrame(),
        channels=channel_table(
            positions=POSITIONS, bad_by_run={"run-1": list((bad or {}).keys())}
        ),
        conditions=(
            pd.DataFrame(
                {
                    "condition": ["painful", "neutral"],
                    "n_total": [60, 60],
                    "n_kept": [48, 52],
                }
            )
            if paradigm is Paradigm.TASK
            else pd.DataFrame()
        ),
    )


def _cohort(*participants, not_aggregated=()) -> Cohort:
    return Cohort(participants=tuple(participants), not_aggregated=tuple(not_aggregated))


# --------------------------------------------------------------------------------------
# Homogeneity
# --------------------------------------------------------------------------------------


def test_a_uniformly_processed_cohort_says_so_in_a_line() -> None:
    html = homogeneity_html(_cohort(_participant("0014"), _participant("0015")))

    assert "same package versions" in html
    assert "same settings" in html


def test_a_version_disagreement_names_who_differs() -> None:
    cohort = _cohort(
        _participant("0014", versions={"eeg_pipeline": "1.0.0"}),
        _participant("0015", versions={"eeg_pipeline": "1.1.0"}),
    )

    found = version_disagreements(cohort)

    assert set(found) == {"eeg_pipeline"}
    assert homogeneity_html(cohort).count("0015") >= 1
    assert "not strictly comparable" in homogeneity_html(cohort)


def test_a_settings_disagreement_is_reported_as_one_about_the_figures() -> None:
    cohort = _cohort(
        _participant("0014", settings={"comb_welch_seconds": 8.0}),
        _participant("0015", settings={"comb_welch_seconds": 4.0}),
    )

    assert set(setting_disagreements(cohort)) == {"comb_welch_seconds"}
    assert "disagreement about the figures" in homogeneity_html(cohort)


def test_a_display_only_difference_is_not_reported_as_a_disagreement() -> None:
    """A site that changed a colour limit did not change a measurement."""
    cohort = _cohort(
        _participant("0014", settings={"color_limit_percentile": 98.0}),
        _participant("0015", settings={"color_limit_percentile": 95.0}),
    )

    assert setting_disagreements(cohort) == {}


# --------------------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------------------


def test_a_cohort_report_is_written_with_its_log(tmp_path) -> None:
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
        task="thermalactive",
    )

    assert paths.html.is_file()
    assert paths.html.name == "task-thermalactive_desc-cohort_report.html"
    assert paths.log.is_file()
    assert paths.html.stat().st_size > 0


def test_every_plotted_value_is_written_beside_the_document(tmp_path) -> None:
    """A figure a reader does not believe has to be checkable without re-deriving it."""
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
        task="thermalactive",
    )

    names = {path.name for path in paths.audit}
    assert any("comb" in name for name in names)
    assert any("spectra" in name for name in names)
    assert any("cleaning" in name for name in names)
    for path in paths.audit:
        assert not pd.read_csv(path, sep="\t").empty


def test_a_section_with_nothing_to_show_writes_no_audit_file(tmp_path) -> None:
    """An empty file and an absent measurement must not look the same on disk."""
    paths = build_cohort_report(
        _cohort(
            _participant("0014", in_scanner=False),
            _participant("0015", in_scanner=False),
        ),
        output_dir=tmp_path,
        task="rest",
    )

    assert not any("comb" in path.name for path in paths.audit)


def test_the_log_records_who_contributed_and_who_did_not(tmp_path) -> None:
    paths = build_cohort_report(
        _cohort(
            _participant("0014"),
            _participant("0015"),
            not_aggregated=[NotAggregated("0016", "No QC sidecar.")],
        ),
        output_dir=tmp_path,
        task="thermalactive",
    )

    log = json.loads(paths.log.read_text(encoding="utf-8"))
    assert log["participants"] == ["0014", "0015"]
    assert log["n_runs"] == 4
    assert log["not_aggregated"][0]["subject"] == "0016"


def test_the_log_records_the_gates_a_missing_median_is_explained_by(tmp_path) -> None:
    """So a panel with no median reads as a participant count, not a failed measurement."""
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
        task="thermalactive",
        gates=BandGates(min_subjects_for_median=6, min_subjects_for_outer_band=12),
    )

    log = json.loads(paths.log.read_text(encoding="utf-8"))
    assert log["band_gates"]["min_subjects_for_median"] == 6
    assert log["band_gates"]["min_subjects_for_outer_band"] == 12


def test_the_log_records_the_versions_that_produced_the_document(tmp_path) -> None:
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
        task="thermalactive",
    )

    log = json.loads(paths.log.read_text(encoding="utf-8"))
    assert "mne" in log["versions"]
    assert "eeg_pipeline" in log["versions"]
    assert log["sidecar_schema_version"] >= 1


def test_an_unnamed_task_still_produces_a_document(tmp_path) -> None:
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
    )

    assert paths.html.name.startswith("cohort")
    assert json.loads(paths.log.read_text(encoding="utf-8"))["task"] is None


def test_the_document_carries_the_sections_the_cohort_supports(tmp_path) -> None:
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
        task="thermalactive",
    )

    html = paths.html.read_text(encoding="utf-8")
    for section in (
        "Cohort composition",
        "At a glance",
        "Preprocessing homogeneity",
        "Channel and region coverage",
        "Epoch rejection",
        "ICA decomposition quality",
        "Scanner artifact correction (Analyzer)",
        "Residual scanner gradient",
        "Sensor spectra",
        "Signal preservation",
        "Data quality over time",
        "How this report was built",
    ):
        assert section in html


def test_the_sections_run_from_what_the_cohort_is_to_what_survived(tmp_path) -> None:
    """The later sections are only interpretable against the earlier ones."""
    paths = build_cohort_report(
        _cohort(_participant("0014"), _participant("0015")),
        output_dir=tmp_path,
        task="thermalactive",
    )

    html = paths.html.read_text(encoding="utf-8")
    ordered = [
        "Cohort composition",
        "At a glance",
        "Preprocessing homogeneity",
        "Sensor spectra",
        "Signal preservation",
        "How this report was built",
    ]
    positions = [html.index(section) for section in ordered]

    assert positions == sorted(positions)


def test_a_resting_state_cohort_drops_the_trial_sections(tmp_path) -> None:
    """A recording with no trials has no retention to report, and no events."""
    paths = build_cohort_report(
        _cohort(
            _participant("0014", in_scanner=False, paradigm=Paradigm.REST),
            _participant("0015", in_scanner=False, paradigm=Paradigm.REST),
        ),
        output_dir=tmp_path,
        task="rest",
    )

    html = paths.html.read_text(encoding="utf-8")

    assert "Epoch rejection" not in html
    assert "What was presented" not in html
    # The sections a resting-state cohort does support are still there.
    assert "Sensor spectra" in html
    assert "Data quality over time" in html


def test_an_eeg_only_cohort_has_no_gradient_section(tmp_path) -> None:
    """A section is absent, not empty, when the acquisition never supported it."""
    paths = build_cohort_report(
        _cohort(
            _participant("0014", in_scanner=False),
            _participant("0015", in_scanner=False),
        ),
        output_dir=tmp_path,
        task="rest",
    )

    html = paths.html.read_text(encoding="utf-8")
    assert "Residual scanner gradient" not in html
    assert "Sensor spectra" in html


def test_a_rebuild_replaces_rather_than_appends(tmp_path) -> None:
    cohort = _cohort(_participant("0014"), _participant("0015"))

    first = build_cohort_report(cohort, output_dir=tmp_path, task="thermalactive")
    size = first.html.stat().st_size
    second = build_cohort_report(cohort, output_dir=tmp_path, task="thermalactive")

    assert second.html == first.html
    assert second.html.stat().st_size == pytest.approx(size, rel=0.05)
