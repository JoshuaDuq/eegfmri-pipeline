import numpy as np
import pytest

correct_cardiac_gaps = pytest.importorskip("studies.pain_study.scripts.cardiac_gaps.correct")


def test_benchmark_row_carries_both_arms():
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    beats = 5.0 + np.arange(120) * 0.9
    data = rng.normal(0, 10.0, (4, int(130 * sfreq)))

    rows = correct_cardiac_gaps.benchmark_arrays(
        data, beats, sfreq, methods=("aas",), n_components=(4,), n_surrogate=5
    )

    assert len(rows) == 1
    row = rows[0]
    for field in (
        "method",
        "removal_max",
        "removal_null_max",
        "removal_channels_above_null",
        "sham_alpha_retained",
        "real_alpha_retained",
    ):
        assert field in row


def test_sham_retention_is_higher_than_real_when_artifact_present():
    """The sham must remove less than the real correction, or it is not a control."""
    rng = np.random.default_rng(1)
    sfreq = 1000.0
    beats = 5.0 + np.arange(120) * 0.9
    clean = rng.normal(0, 10.0, (4, int(130 * sfreq)))
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = 60.0 * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    dirty = clean.copy()
    for beat in beats:
        start = int(round(beat * sfreq))
        dirty[:, start : start + shape.size] += shape

    rows = correct_cardiac_gaps.benchmark_arrays(
        dirty, beats, sfreq, methods=("aas",), n_components=(4,), n_surrogate=5
    )

    assert rows[0]["sham_band_retained"] > rows[0]["real_band_retained"]


def _recovery(recovered, gap_seconds_before, status="ok"):
    """A BeatRecovery carrying only the fields the status decision reads."""
    from studies.pain_study.analysis.bcg.detect import BeatQuality, BeatRecovery

    nan = float("nan")
    quality = BeatQuality(
        analyzer_lock_ratio=nan,
        recovered_lock_ratio=nan,
        combined_lock_ratio=nan,
        physiological_floor_s=nan,
        rr_median_s=nan,
        rr_min_s=nan,
        rr_max_s=nan,
        implied_bpm=nan,
        refractory_violations=0,
        refractory_rejected=0,
        recovered_beats=recovered,
        gap_seconds_before=gap_seconds_before,
        gap_seconds_after=nan,
        status=status,
    )
    return BeatRecovery(
        analyzer_beats=np.zeros(500),
        recovered_beats=np.zeros(recovered),
        combined_beats=np.zeros(500 + recovered),
        quality=quality,
    )


def test_a_run_without_gaps_is_not_reported_as_a_recovery_failure():
    """Nothing to correct is a different outcome from failing to correct.

    Most skipped runs on this cohort are gap-free, which the cohort report must not read
    as detector failure.
    """
    status = correct_cardiac_gaps.recovery_status(_recovery(0, gap_seconds_before=0.0))

    assert status == "no_gaps"


def test_gaps_that_yield_too_few_beats_stay_a_failure():
    status = correct_cardiac_gaps.recovery_status(_recovery(3, gap_seconds_before=12.3))

    assert status == "too_few_recovered (3)"


def test_enough_recovered_beats_is_ok():
    assert correct_cardiac_gaps.recovery_status(_recovery(30, gap_seconds_before=57.3)) == "ok"


def test_insufficient_seed_beats_is_reported_as_itself():
    """A run Analyzer barely marked is an ECG problem, not a gap-filling outcome."""
    recovery = _recovery(0, gap_seconds_before=0.0, status="insufficient_seed_beats")

    assert correct_cardiac_gaps.recovery_status(recovery) == "insufficient_seed_beats"


def test_scoring_data_excludes_non_eeg_channels():
    """The ECG must never reach the referee: it is the cardiac signal, not artifact.

    Scoring it alongside EEG put R-locked reduction at 0.545 on sub0009 run 1 while the
    best EEG channel sat at 0.004, so the reported figure described the ECG channel.
    """
    import mne

    mne.set_log_level("ERROR")
    names = ["Fp1", "Oz", "ECG"]
    info = mne.create_info(names, 1000.0, ch_types=["eeg", "eeg", "misc"])
    raw = mne.io.RawArray(np.zeros((3, 5000)), info, verbose="ERROR")

    data, picked = correct_cardiac_gaps.scoring_data(raw)

    assert picked == ["Fp1", "Oz"]
    assert data.shape[0] == 2


def test_quality_row_carries_every_beat_quality_field():
    """The spec's BeatQuality table is a report, not an internal structure.

    Every field is a measurement the cohort report is supposed to publish, so the row must
    not quietly drop any of them as the dataclass grows.
    """
    from dataclasses import fields

    from studies.pain_study.analysis.bcg.detect import BeatQuality

    row = correct_cardiac_gaps.quality_row(
        _recovery(30, gap_seconds_before=57.3),
        crosscheck={
            "status": "ok",
            "agreement_fraction": 0.9,
            "crosscheck_beats": 500.0,
            "crosscheck_lock_ratio": 3.2,
        },
    )

    # `status` is deliberately republished as `beat_status`, so it cannot overwrite the
    # run-level status when the row is merged into the report.
    renamed = {"status": "beat_status"}
    for field in fields(BeatQuality):
        name = renamed.get(field.name, field.name)
        assert name in row, f"BeatQuality.{field.name} missing from the report row"
    assert row["crosscheck_agreement_fraction"] == 0.9
    assert row["crosscheck_status"] == "ok"


def test_an_implausible_heart_rate_is_flagged_rather_than_passed_on():
    """A run can pass every structural check and still be unusable.

    sub-0008 run 4 ends with 103 beats across 497 s -- 12.4 bpm. Its recovered beats are
    good (QRS lock ratio 4.53), but Analyzer marked so little that the gap rule, which is
    relative to the run's own median RR, never flags the ~4 s intervals where the rest of
    the beats are hiding. The run is globally under-marked rather than gap-structured, so
    gap filling cannot reach it and Analyzer must not be handed it as if corrected.
    """
    recovery = _recovery(59, gap_seconds_before=347.0)
    object.__setattr__(recovery.quality, "implied_bpm", 12.4)

    assert correct_cardiac_gaps.recovery_status(recovery) == "implausible_rate (12.4 bpm)"


def test_an_ordinary_rate_is_not_flagged():
    recovery = _recovery(30, gap_seconds_before=57.3)
    object.__setattr__(recovery.quality, "implied_bpm", 61.6)

    assert correct_cardiac_gaps.recovery_status(recovery) == "ok"


def test_quality_row_does_not_clobber_the_run_level_status():
    """BeatQuality carries its own `status`, which is not the run's outcome.

    Merging the row straight into the report overwrote the run status with the dataclass's
    own: a cohort pass reported every one of 103 recordings as `ok`, including the 23 with
    no gaps at all.
    """
    row = correct_cardiac_gaps.quality_row(_recovery(0, gap_seconds_before=0.0), crosscheck=None)

    assert "status" not in row
    assert row["beat_status"] == "ok"


def test_quality_row_survives_an_unavailable_crosscheck():
    """A cross-check that could not run is a missing measurement, not a missing row."""
    row = correct_cardiac_gaps.quality_row(_recovery(30, gap_seconds_before=57.3), crosscheck=None)

    assert row["crosscheck_status"] == "not_run"
    assert row["recovered_beats"] == 30


def test_provenance_round_trips(tmp_path):
    written = tmp_path / "run1_sub0009_corrected.vhdr"
    written.write_text("")

    correct_cardiac_gaps.write_provenance(
        written, {"method": "obs", "n_components": 4, "recovered_beats": 44}
    )
    back = correct_cardiac_gaps.read_provenance(written)

    assert back["method"] == "obs"
    assert back["recovered_beats"] == 44
    assert "written_utc" in back


def test_missing_provenance_is_reported_not_assumed(tmp_path):
    """A file with no provenance may predate the current code, so it must not be scored.

    This bit for real: after `apply` failed on every run, `verify` re-scored the files a
    previous run had left behind and reported them as current.
    """
    written = tmp_path / "run1_sub0009_corrected.vhdr"
    written.write_text("")

    assert correct_cardiac_gaps.read_provenance(written) is None


def test_apply_only_changes_gap_stretches(tmp_path):
    """Everything outside a gap must survive byte-for-byte from Analyzer's output."""
    rng = np.random.default_rng(2)
    sfreq = 1000.0
    n = int(200 * sfreq)
    analyzer_corrected = rng.normal(0, 10.0, (4, n))
    uncorrected = analyzer_corrected + rng.normal(0, 1.0, (4, n))

    out = correct_cardiac_gaps.substitute_gap_stretches(
        analyzer_corrected, uncorrected, [(80.0, 95.0)], sfreq, pad_seconds=0.5
    )

    lo, hi = int(79.5 * sfreq), int(95.5 * sfreq)
    assert np.array_equal(out[:, :lo], analyzer_corrected[:, :lo])
    assert np.array_equal(out[:, hi:], analyzer_corrected[:, hi:])
    assert not np.array_equal(out[:, lo:hi], analyzer_corrected[:, lo:hi])
