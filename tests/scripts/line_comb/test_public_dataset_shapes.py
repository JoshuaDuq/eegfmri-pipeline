"""Dataset shapes other sites have, which this workflow must not refuse.

The removal was written against one study: one task label, six runs per participant, and
eleven events in every recording. None of that is a property of a line comb. These pin the
shapes a new user actually arrives with -- a single continuous baseline, a session
hierarchy, a task named something else, a 50 Hz country -- so they cannot regress back into
study-specific refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.scripts.line_comb import remove as rlc


def _write_run(root, subject: str, task: str, *, run: str | None, session: str | None = None):
    """One short BrainVision recording in BIDS layout, with no events at all."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    sfreq = 250.0
    times = np.arange(int(sfreq * 120)) / sfreq
    rng = np.random.default_rng(len(subject) + len(task))
    data = rng.normal(scale=1e-6, size=(4, times.size))
    data += 2e-6 * np.sin(2.0 * np.pi * 57.25 * times)
    info = mne.create_info([f"EEG{i:02d}" for i in range(4)], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        session=session,
        datatype="eeg",
        root=root,
        extension=".vhdr",
    )
    write_raw_bids(raw, path, format="BrainVision", allow_preload=True, verbose="ERROR")


class TestDiscovery:
    def test_a_single_continuous_recording_without_a_run_entity_is_found(self, tmp_path):
        """BIDS omits run- when a task was acquired once, which baselines usually are."""
        _write_run(tmp_path, "0001", "rest", run=None)

        found = rlc.discover_runs(tmp_path, subjects=None, task="rest")

        assert len(found) == 1
        assert "_run-" not in found[0].name

    def test_a_session_hierarchy_is_searched(self, tmp_path):
        _write_run(tmp_path, "0001", "rest", run=None, session="01")

        found = rlc.discover_runs(tmp_path, subjects=None, task="rest")

        assert len(found) == 1
        assert "ses-01" in str(found[0])

    def test_subject_filtering_survives_a_session_directory(self, tmp_path):
        _write_run(tmp_path, "0001", "rest", run=None, session="01")
        _write_run(tmp_path, "0002", "rest", run=None, session="01")

        found = rlc.discover_runs(tmp_path, subjects=["sub-0002"], task="rest")

        assert [rlc._subject_of(path) for path in found] == ["sub-0002"]

    def test_another_task_in_the_same_root_is_not_picked_up(self, tmp_path):
        _write_run(tmp_path, "0001", "rest", run=None)
        _write_run(tmp_path, "0001", "oddball", run="1")

        assert len(rlc.discover_runs(tmp_path, subjects=None, task="rest")) == 1
        assert len(rlc.discover_runs(tmp_path, subjects=None, task="oddball")) == 1

    def test_the_refusal_names_the_setting_to_change(self, tmp_path):
        _write_run(tmp_path, "0001", "rest", run=None)

        with pytest.raises(FileNotFoundError, match="dataset.task"):
            rlc.discover_runs(tmp_path, subjects=None, task="thermalactive")


class TestSessionSize:
    def test_one_recording_is_planned_rather_than_refused(self):
        """Cross-recording replication is unavailable, not required.

        Only the single-recording route can fire, and it is the stricter of the two, so a
        lone baseline is planned under a higher bar rather than turned away.
        """
        settings = rlc.RemovalSettings()
        spectra = [_flat_session_run()]

        plans = rlc.automatic_line_plans(spectra, settings)

        assert len(plans) == 1

    def test_two_recordings_are_planned(self):
        settings = rlc.RemovalSettings()

        plans = rlc.automatic_line_plans([_flat_session_run(), _flat_session_run()], settings)

        assert len(plans) == 2


def _flat_session_run():
    """One recording carrying a clean comb and nothing else."""
    freqs = np.arange(1.0, 100.0, 0.002)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355
    for harmonic in range(24, 80):
        spectrum[:] = np.maximum(
            spectrum, 14.0 * np.exp(-0.5 * ((freqs - harmonic * 1.2) / sigma) ** 2)
        )
    scope = (freqs, spectrum, spectrum.copy())
    return rlc.SessionRunSpectra(
        whole=scope,
        windows=(scope, scope, scope),
        bounds=((0, 100), (50, 150), (100, 200)),
    )


class TestSiteConstants:
    def test_the_estimation_window_is_a_setting_not_the_volume_repetition(self):
        settings = rlc.RemovalSettings(estimation_window_s=10.0)

        assert rlc.estimation_window_samples(500.0, settings) == 5_000

    def test_a_window_under_two_samples_is_refused(self):
        settings = rlc.RemovalSettings(estimation_window_s=0.001)

        with pytest.raises(ValueError, match="under two samples"):
            rlc.estimation_window_samples(100.0, settings)

    def test_a_fifty_hertz_site_can_move_the_mains_band(self):
        settings = rlc.RemovalSettings(mains_notch_hz=(49.5, 50.5))

        assert settings.mains_notch_hz == (49.5, 50.5)

    @pytest.mark.parametrize("band", [(60.5, 59.5), (0.0, 50.0), (50.0, 50.0)])
    def test_an_impossible_mains_band_is_refused(self, band):
        with pytest.raises(ValueError, match="mains_notch_hz"):
            rlc.RemovalSettings(mains_notch_hz=band)

    def test_an_empty_task_is_refused(self):
        with pytest.raises(ValueError, match="task"):
            rlc.RemovalSettings(task="  ")


class TestRetiredEpochSettings:
    """A config carrying the removed epoch settings must say so, not ignore them."""

    @pytest.mark.parametrize(
        "setting",
        ["study_event_name", "study_epoch_s", "expected_study_events_per_run"],
    )
    def test_a_stale_epoch_setting_is_refused(self, setting):
        class Config:
            def get(self, key, default=None):
                return {"line_comb_removal": {setting: "anything"}}.get(key, default)

        with pytest.raises(ValueError, match="Exact-epoch scoping was removed"):
            rlc.RemovalSettings.from_config(Config())


class TestSpectralCostIsMeasuredNotBudgeted:
    """No shipped ceiling on how much spectrum the removal may take.

    The cost is the notch width times the number of targets; the width ratio comes from a
    documented sweep and each target from the replication rules, so there is nothing left
    for a ceiling to constrain. Any default would be a number picked after seeing the
    answer, which is what the retired 0.18 was.
    """

    def test_no_ceiling_ships_with_the_settings(self):
        assert rlc.RemovalSettings().max_band_cost is None

    def test_the_gate_carries_no_band_criterion(self):
        from studies.pain_study.analysis.line_comb import removal as lr

        criteria = lr.PreservationGate().evaluate(
            {
                "max_probe_deviation_db": 0.0,
                "max_nonline_change_db": 0.0,
                "intrinsic_energy_ratio": 0.95,
                "burst_correlation": 1.0,
            }
        )

        assert "band_mostly_untouched" not in criteria
        assert "max_band_fraction_removed" not in lr.PreservationGate().__dataclass_fields__

    @pytest.mark.parametrize("declared", [0.0, 1.5, -0.2])
    def test_an_impossible_declaration_is_refused(self, declared):
        with pytest.raises(ValueError, match="max_band_cost"):
            rlc.RemovalSettings(max_band_cost=declared)

    def test_a_declared_budget_is_honoured(self, tmp_path):
        """A study may state a budget; apply then refuses against that, not a default."""
        import pandas as pd

        settings = rlc.RemovalSettings(max_band_cost=0.10)
        path = tmp_path / "benchmark.tsv"
        pd.DataFrame(
            [
                {
                    "recording": "r0",
                    "settings_fingerprint": rlc.settings_fingerprint(settings),
                    "gate_passed": True,
                    "measured_band_attenuated_1db": 0.17,
                    "boundary_discontinuity_max_v": 0.5,
                    "boundary_control_maxima_v": ";".join(["1"] * 40),
                    "residual_null_p": 0.9,
                    "focal_residual_null_p": 0.9,
                    "nonline_change_null_p": 0.9,
                }
            ]
        ).to_csv(path, sep="\t", index=False)

        with pytest.raises(RuntimeError, match="this study declared"):
            rlc.require_passing_benchmark(path, settings)

    def test_an_undeclared_budget_refuses_nothing(self, tmp_path):
        import pandas as pd

        settings = rlc.RemovalSettings()
        path = tmp_path / "benchmark.tsv"
        pd.DataFrame(
            [
                {
                    "recording": "r0",
                    "settings_fingerprint": rlc.settings_fingerprint(settings),
                    "gate_passed": True,
                    "measured_band_attenuated_1db": 0.42,
                    "boundary_discontinuity_max_v": 0.5,
                    "boundary_control_maxima_v": ";".join(["1"] * 40),
                    "residual_null_p": 0.9,
                    "focal_residual_null_p": 0.9,
                    "nonline_change_null_p": 0.9,
                }
            ]
        ).to_csv(path, sep="\t", index=False)

        rlc.require_passing_benchmark(path, settings)
