"""Electrode bridges are an input-quality diagnostic, not an interpolation rule."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _raw() -> mne.io.RawArray:
    sfreq = 100.0
    info = mne.create_info(["Fz", "Cz", "Pz"], sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((3, int(240 * sfreq))), info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    raw.info["bads"] = ["Pz"]
    return raw


def test_bridge_review_uses_configured_final_segment_and_includes_marked_bads(
    monkeypatch,
) -> None:
    from eeg_pipeline.preprocessing.report.bridging import compute_bridging_review

    captured = {}

    def fake_compute(raw):
        captured["duration"] = raw.n_times / raw.info["sfreq"]
        captured["bads"] = list(raw.info["bads"])
        return [(0, 1)], np.eye(len(raw.ch_names))

    monkeypatch.setattr(mne.preprocessing, "compute_bridged_electrodes", fake_compute)

    review = compute_bridging_review(
        _raw(),
        recording_id="run-6",
        duration_seconds=120.0,
    )

    assert captured["duration"] == 120.0
    assert captured["bads"] == []
    assert review.pair_names == (("Fz", "Cz"),)
    assert review.previously_bad_channels == ("Pz",)


def test_bridge_review_renders_pair_names_and_scope(monkeypatch) -> None:
    from eeg_pipeline.preprocessing.report.bridging import (
        add_bridging_reviews,
        compute_bridging_review,
    )

    monkeypatch.setattr(
        mne.preprocessing,
        "compute_bridged_electrodes",
        lambda raw: ([(0, 1)], np.eye(len(raw.ch_names))),
    )
    monkeypatch.setattr(
        mne.viz,
        "plot_bridged_electrodes",
        lambda *args, **kwargs: plt.figure(),
    )
    report = mne.Report(title="bridge", verbose="ERROR")
    reviews = [
        compute_bridging_review(_raw(), recording_id="run-1"),
        compute_bridging_review(_raw(), recording_id="run-6"),
    ]

    add_bridging_reviews(report=report, reviews=reviews)

    document = "".join(str(element.html) for element in report._content)
    assert "Fz–Cz" in document
    assert "each run" in document
    assert "run-1" in document
    assert "run-6" in document
    assert len(report._content) == 2
    assert any(element.section == "Electrode bridging" for element in report._content)
    assert document.count("Previously marked bad EEG sites") == 1
    assert "Previously bad sites included" not in document
