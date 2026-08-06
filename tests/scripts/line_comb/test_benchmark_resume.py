"""An interrupted benchmark must not throw away the recordings it already measured.

Twice in one day a 90-recording run died partway and lost everything: a probe collision at
recording 24 discarded 23 completed runs, and an unconstructible matched null would have
discarded all 90. The frame is only written after the loop, so any raise costs hours.

The fix cannot be "write benchmark.tsv as you go". ``require_passing_benchmark`` exists
partly because a benchmark that died on its second recording once left the previous run's
file in place and the gates appeared to pass. Partial work therefore goes to a file that
``apply`` never reads, and is only promoted to benchmark.tsv when every recording is in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from studies.pain_study.scripts.line_comb import remove as rlc


def _partial(path, fingerprint, recordings):
    pd.DataFrame(
        [
            {
                "recording": recording,
                "settings_fingerprint": fingerprint,
                "input_digest": f"digest-{recording}",
                "plan_digest": f"plan-{recording}",
                "gate_passed": True,
                "max_boundary_discontinuity_ratio": 0.5,
            }
            for recording in recordings
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


def _digests(recordings):
    return (
        {r: f"digest-{r}" for r in recordings},
        {r: f"plan-{r}" for r in recordings},
    )


def test_the_partial_file_is_not_the_benchmark_file(tmp_path):
    """apply reads benchmark.tsv; nothing incomplete may ever be written there."""
    assert rlc.partial_benchmark_path(tmp_path) != tmp_path / "benchmark.tsv"


def test_a_partial_alone_does_not_authorise_apply(tmp_path):
    settings = rlc.RemovalSettings()
    _partial(rlc.partial_benchmark_path(tmp_path), rlc.settings_fingerprint(settings), ["r0", "r1"])

    with pytest.raises(RuntimeError, match="no benchmark"):
        rlc.require_passing_benchmark(tmp_path / "benchmark.tsv", settings)


def test_completed_recordings_are_reused(tmp_path):
    settings = rlc.RemovalSettings()
    fingerprint = rlc.settings_fingerprint(settings)
    path = _partial(rlc.partial_benchmark_path(tmp_path), fingerprint, ["r0", "r1"])
    recordings, plans = _digests(["r0", "r1", "r2"])

    reusable = rlc.resumable_benchmark_rows(path, fingerprint, recordings, plans)

    assert set(reusable) == {"r0", "r1"}


def test_rows_from_different_settings_are_discarded(tmp_path):
    settings = rlc.RemovalSettings()
    path = _partial(rlc.partial_benchmark_path(tmp_path), "an-older-fingerprint", ["r0", "r1"])
    recordings, plans = _digests(["r0", "r1"])

    reusable = rlc.resumable_benchmark_rows(
        path, rlc.settings_fingerprint(settings), recordings, plans
    )

    assert reusable == {}


def test_a_recording_whose_input_changed_is_remeasured(tmp_path):
    fingerprint = rlc.settings_fingerprint(rlc.RemovalSettings())
    path = _partial(rlc.partial_benchmark_path(tmp_path), fingerprint, ["r0", "r1"])
    recordings, plans = _digests(["r0", "r1"])
    recordings["r0"] = "the-file-was-edited"

    reusable = rlc.resumable_benchmark_rows(path, fingerprint, recordings, plans)

    assert set(reusable) == {"r1"}


def test_a_recording_whose_fitted_plan_changed_is_remeasured(tmp_path):
    fingerprint = rlc.settings_fingerprint(rlc.RemovalSettings())
    path = _partial(rlc.partial_benchmark_path(tmp_path), fingerprint, ["r0", "r1"])
    recordings, plans = _digests(["r0", "r1"])
    plans["r1"] = "the-fit-moved"

    reusable = rlc.resumable_benchmark_rows(path, fingerprint, recordings, plans)

    assert set(reusable) == {"r0"}


def test_a_missing_partial_simply_means_no_resume(tmp_path):
    recordings, plans = _digests(["r0"])

    assert rlc.resumable_benchmark_rows(tmp_path / "absent.tsv", "f", recordings, plans) == {}


def test_an_unreadable_partial_costs_work_but_never_correctness(tmp_path):
    """Recomputing is always safe; trusting a damaged journal is not."""
    path = rlc.partial_benchmark_path(tmp_path)
    path.write_text("this is not a table\n", encoding="utf-8")
    recordings, plans = _digests(["r0"])

    assert rlc.resumable_benchmark_rows(path, "f", recordings, plans) == {}


def test_the_loop_journals_each_recording_and_clears_it_only_on_success():
    """Order matters: journal inside the loop, promote once, then discard the journal.

    Promoting before the completeness check, or clearing the journal before promoting,
    would each reintroduce a way for an interrupted run to look like a finished one.
    """
    import inspect

    source = inspect.getsource(rlc.run)
    journal = source.index("_write_tsv_atomic(pd.DataFrame(rows), partial_path)")
    complete = source.index("does not contain exactly one result per plan")
    promote = source.index('_write_tsv_atomic(frame, args.report_dir / "benchmark.tsv")')
    clear = source.index("partial_path.unlink")

    assert journal < complete < promote < clear


def test_the_journal_carries_what_resumption_needs_to_verify(tmp_path):
    """A row without its digests cannot be checked, so it must never be reusable."""
    path = rlc.partial_benchmark_path(tmp_path)
    pd.DataFrame([{"recording": "r0", "settings_fingerprint": "f", "gate_passed": True}]).to_csv(
        path, sep="\t", index=False
    )

    assert rlc.resumable_benchmark_rows(path, "f", {"r0": "d"}, {"r0": "p"}) == {}
