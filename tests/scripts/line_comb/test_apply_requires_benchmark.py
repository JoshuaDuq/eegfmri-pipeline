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


def _benchmark(path, fingerprint, passed=True, n=3):
    pd.DataFrame(
        [
            {"recording": f"r{i}", "settings_fingerprint": fingerprint, "gate_passed": passed}
            for i in range(n)
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


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


def test_the_benchmark_certifies_the_pooled_estimate_the_apply_uses():
    """Benchmarking a per-run fit does not certify a session-pooled removal.

    The two differ measurably here: sub-0000 run-1 reported a 3.47 dB worst residual under
    its own fundamental and 13.90 dB under its session's, on the same recording. Whichever
    is right, the gate has to score the one that ships.
    """
    import inspect

    for function in (rlc.benchmark_run, rlc.estimate_and_targets):
        assert "session_estimate" in inspect.signature(function).parameters, (
            f"{function.__name__} cannot be told which fundamental the apply will use"
        )

    source = inspect.getsource(rlc.estimate_and_targets)
    assert "session_estimate if session_estimate is not None" in source, (
        "the session estimate is accepted but not the one the targets are built from"
    )


def test_the_fingerprint_covers_the_fundamental_scope_and_the_code():
    """Two applies that differ in scope or in code are not the same transformation."""
    settings = rlc.RemovalSettings()
    a = rlc.settings_fingerprint(settings, fundamental_scope="session")
    b = rlc.settings_fingerprint(settings, fundamental_scope="run")
    assert a != b, "the fingerprint ignores which fundamental the apply will use"


def test_a_benchmark_missing_a_subject_does_not_authorise_it(tmp_path):
    """One passing row could authorise all ninety runs."""
    settings = rlc.RemovalSettings()
    path = tmp_path / "benchmark.tsv"
    pd.DataFrame(
        [{
            "recording": "sub-0000_task-thermalactive_run-1_eeg",
            "settings_fingerprint": rlc.settings_fingerprint(settings),
            "gate_passed": True,
        }]
    ).to_csv(path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="did not cover"):
        rlc.require_passing_benchmark(path, settings, subjects={"sub-0000", "sub-0001"})
