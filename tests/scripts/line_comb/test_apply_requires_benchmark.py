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
