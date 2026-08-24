"""Evoked QC uses a fixed, configured sampling grid rather than observed peaks."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np

matplotlib.use("Agg")


def _epochs() -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    info.set_montage("standard_1020")
    data = np.random.default_rng(0).normal(0.0, 1e-6, (8, 3, 401))
    events = np.column_stack((np.arange(8) * 500, np.zeros(8, int), np.ones(8, int)))
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"pain": 1},
        tmin=-1.0,
        verbose="ERROR",
    )


def test_topography_times_are_fixed_by_config_not_evoked_peaks(monkeypatch) -> None:
    from eeg_pipeline.preprocessing.report.evoked import add_evoked_response_review

    captured = []

    def fake_plot_joint(self, *, times, picks, show):
        import matplotlib.pyplot as plt

        captured.append(np.asarray(times))
        return plt.figure()

    monkeypatch.setattr(mne.Evoked, "plot_joint", fake_plot_joint)
    report = mne.Report(title="evoked", verbose="ERROR")

    add_evoked_response_review(
        report=report,
        epochs=_epochs(),
        response_window_s=(0.0, 2.0),
        topomap_count=5,
        analysis_status="Final",
    )

    np.testing.assert_allclose(captured[0], np.linspace(0.0, 2.0, 5))
    document = "".join(str(element.html) for element in report._content)
    assert "Exact event description" not in document
    assert "pain · code 1 · 8 retained trials" in document
