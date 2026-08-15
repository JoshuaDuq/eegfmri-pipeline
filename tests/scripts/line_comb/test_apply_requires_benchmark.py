"""Writing 11 GB of derived data should require a passing benchmark of the same settings.

``apply`` would run whatever was configured, with no check that a benchmark had been run,
that it had passed, or that it described these settings. Three ways that goes wrong were
all live in this work: a benchmark.tsv on disk from an earlier configuration was read as
if it described the current one; a benchmark that died on its second recording left the
previous run's file in place, so the gates appeared to pass; and an apply was started under
gates later found to be unable to fail.
"""

from __future__ import annotations

import pandas as pd
import pytest

from studies.pain_study.scripts.line_comb import remove as rlc


def _benchmark(
    path,
    fingerprint,
    passed=True,
    n=3,
    seam_observed=None,
    residual_p=None,
    focal_residual_p=None,
):
    observed = [0.5] * n if seam_observed is None else list(seam_observed)
    residual = [0.9] * n if residual_p is None else list(residual_p)
    focal = [0.9] * n if focal_residual_p is None else list(focal_residual_p)
    pd.DataFrame(
        [
            {
                "recording": f"r{i}",
                "settings_fingerprint": fingerprint,
                "input_digest": f"digest-{i}",
                "plan_digest": f"plan-{i}",
                "gate_passed": passed,
                "max_boundary_discontinuity_ratio": observed[i],
                "boundary_discontinuity_max_v": observed[i],
                "boundary_control_maxima_v": ";".join(["1"] * 40),
                "residual_null_p": residual[i],
                "focal_residual_null_p": focal[i],
                "nonline_change_null_p": 0.9,
            }
            for i in range(n)
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_apply_refuses_when_the_cohort_residual_criterion_fails(tmp_path):
    """The residual question is exact against each run's own controls, and gated there.

    It is absent from the per-run gate because about one recording in twenty exceeds by
    construction, so an all-runs rule would reject a faultless cohort. Every row can carry
    ``gate_passed`` true and the cohort still be inadmissible.
    """
    settings = rlc.RemovalSettings()
    path = _benchmark(
        tmp_path / "benchmark.tsv",
        rlc.settings_fingerprint(settings),
        residual_p=[1e-9, 0.9, 0.9],
    )

    with pytest.raises(RuntimeError, match="whole-run residual criterion failed"):
        rlc.require_passing_benchmark(path, settings)


def test_apply_refuses_when_the_focal_residual_criterion_fails(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(
        tmp_path / "benchmark.tsv",
        rlc.settings_fingerprint(settings),
        focal_residual_p=[1e-9, 0.9, 0.9],
    )

    with pytest.raises(RuntimeError, match="focal residual criterion failed"):
        rlc.require_passing_benchmark(path, settings)


def test_apply_refuses_a_benchmark_without_the_residual_probabilities(tmp_path):
    """An older benchmark cannot answer the question, so it cannot certify the cohort."""
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", rlc.settings_fingerprint(settings))
    frame = pd.read_csv(path, sep="\t").drop(columns=["residual_null_p"])
    frame.to_csv(path, sep="\t", index=False)

    with pytest.raises(RuntimeError, match="carries no residual_null_p"):
        rlc.require_passing_benchmark(path, settings)


def test_apply_refuses_when_the_cohort_seam_criterion_fails(tmp_path):
    """The seam criterion is decided over the cohort, so apply has to consult it there.

    Every row can carry ``gate_passed`` true and the cohort still be inadmissible: the
    per-run gate no longer contains the seam check at all, because at 2/41 per run it
    could not survive being applied 90 times.
    """
    settings = rlc.RemovalSettings()
    path = _benchmark(
        tmp_path / "benchmark.tsv",
        rlc.settings_fingerprint(settings),
        seam_observed=[5.0, 0.5, 0.5],
    )
    with pytest.raises(RuntimeError, match="seam"):
        rlc.require_passing_benchmark(path, settings)


def test_apply_refuses_a_benchmark_with_no_seam_measurements(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", rlc.settings_fingerprint(settings))
    frame = pd.read_csv(path, sep="\t").drop(columns=["boundary_control_maxima_v"])
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="seam"):
        rlc.require_passing_benchmark(path, settings)


def test_a_fingerprint_changes_when_a_setting_changes():
    a = rlc.settings_fingerprint(rlc.RemovalSettings())
    b = rlc.settings_fingerprint(rlc.RemovalSettings(detection_min_prominence_db=8.0))
    assert a != b
    assert a == rlc.settings_fingerprint(rlc.RemovalSettings())


def test_apply_refuses_without_a_benchmark(tmp_path):
    with pytest.raises(RuntimeError, match="no benchmark"):
        rlc.require_passing_benchmark(tmp_path / "absent.tsv", rlc.RemovalSettings())


def test_apply_refuses_a_benchmark_of_different_settings(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", "not-this-one")
    with pytest.raises(RuntimeError, match="different settings"):
        rlc.require_passing_benchmark(path, settings)


def test_apply_refuses_a_benchmark_that_failed(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", rlc.settings_fingerprint(settings), passed=False)
    with pytest.raises(RuntimeError, match="did not pass"):
        rlc.require_passing_benchmark(path, settings)


def test_apply_accepts_a_matching_passing_benchmark(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", rlc.settings_fingerprint(settings))
    rlc.require_passing_benchmark(path, settings)  # must not raise


def test_a_benchmark_must_cover_the_exact_apply_recordings(tmp_path):
    """One row per subject must not authorise the other runs."""
    settings = rlc.RemovalSettings()
    path = tmp_path / "benchmark.tsv"
    pd.DataFrame(
        [
            {
                "recording": "run-1",
                "settings_fingerprint": rlc.settings_fingerprint(settings),
                "input_digest": "digest-1",
                "gate_passed": True,
            }
        ]
    ).to_csv(path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="recordings differ"):
        rlc.require_passing_benchmark(
            path,
            settings,
            recordings={"run-1": "digest-1", "run-2": "digest-2"},
        )


def test_a_benchmark_of_changed_input_does_not_authorise_apply(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", rlc.settings_fingerprint(settings), n=1)
    with pytest.raises(RuntimeError, match="input digest"):
        rlc.require_passing_benchmark(
            path,
            settings,
            recordings={"r0": "changed"},
        )


def test_a_benchmark_of_a_different_fitted_plan_does_not_authorise_apply(tmp_path):
    settings = rlc.RemovalSettings()
    path = _benchmark(tmp_path / "benchmark.tsv", rlc.settings_fingerprint(settings), n=1)
    with pytest.raises(RuntimeError, match="removal plan changed"):
        rlc.require_passing_benchmark(
            path,
            settings,
            recordings={"r0": "digest-0"},
            plans={"r0": "changed"},
        )


def test_the_fingerprint_tracks_source_content_not_just_a_commit_id():
    """A commit id plus '+dirty' is the same string for every dirty tree.

    Two different uncommitted implementations would share a benchmark fingerprint, which is
    the failure the fingerprint exists to prevent.
    """
    digest = rlc._source_digest()
    assert len(digest) >= 12
    assert "dirty" not in digest
    assert digest != "unknown"


def test_an_unidentifiable_source_is_an_error_not_a_shrug(monkeypatch):
    monkeypatch.setattr(rlc, "_source_digest", lambda: "unknown")
    with pytest.raises(RuntimeError, match="identify"):
        rlc.settings_fingerprint(rlc.RemovalSettings())


def test_recording_digest_covers_header_data_and_markers(tmp_path):
    vhdr = tmp_path / "run.vhdr"
    eeg = tmp_path / "run.eeg"
    vmrk = tmp_path / "run.vmrk"
    vhdr.write_text("DataFile=run.eeg\nMarkerFile=run.vmrk\n", encoding="utf-8")
    eeg.write_bytes(b"samples-a")
    vmrk.write_text("markers", encoding="utf-8")
    original = rlc.recording_digest(vhdr)

    eeg.write_bytes(b"samples-b")

    assert rlc.recording_digest(vhdr) != original


def _cohort_benchmark(path, fingerprint, seam_observed):
    """A realistic 90-run cohort: 15 sessions of 6, named as the pipeline names them."""
    pd.DataFrame(
        [
            {
                "recording": f"sub-{i // 6:04d}_task-thermalactive_run-{i % 6 + 1}_eeg",
                "settings_fingerprint": fingerprint,
                "input_digest": f"digest-{i}",
                "plan_digest": f"plan-{i}",
                "gate_passed": True,
                "max_boundary_discontinuity_ratio": seam_observed[i],
                "boundary_discontinuity_max_v": seam_observed[i],
                "boundary_control_maxima_v": ";".join(["1"] * 40),
            }
            for i in range(90)
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_the_apply_guard_rejects_systematic_seams_against_the_measured_controls(tmp_path):
    settings = rlc.RemovalSettings()
    fingerprint = rlc.settings_fingerprint(settings)
    systematic = [1.1] * 90

    with pytest.raises(RuntimeError, match="seam"):
        rlc.require_passing_benchmark(
            _cohort_benchmark(tmp_path / "systematic.tsv", fingerprint, systematic), settings
        )
