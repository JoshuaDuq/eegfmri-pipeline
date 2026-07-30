"""Tests for cohort assembly logic in the scanner-harmonic diagnosis."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis import harmonic_cohort as hc


class TestAssignRuns:
    def test_partitions_a_balanced_concatenation(self):
        lengths = [1000] * 6
        samples = np.concatenate([np.arange(5) * 100 + run * 1000 for run in range(6)])
        labels = hc.assign_runs(samples, lengths)
        assert labels.tolist() == sum(([run + 1] * 5 for run in range(6)), [])

    def test_handles_unequal_run_lengths(self):
        lengths = [244800, 246150, 242550]
        samples = np.array([10, 244799, 244800, 490949, 490950, 733499])
        assert hc.assign_runs(samples, lengths).tolist() == [1, 1, 2, 2, 3, 3]

    def test_rejects_events_beyond_the_concatenation(self):
        with pytest.raises(ValueError, match="beyond the concatenated"):
            hc.assign_runs([0, 5000], [1000, 1000])

    def test_rejects_a_run_with_no_epochs(self):
        with pytest.raises(ValueError, match="contributed no epochs"):
            hc.assign_runs([0, 10, 2500], [1000, 1000, 1000])

    def test_rejects_unsorted_events(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            hc.assign_runs([500, 100], [1000])


class TestSegmentPeriodograms:
    def test_averages_blocks_within_a_segment(self):
        sfreq, tr_count = 500.0, 6
        n_times = int(0.9 * sfreq) * tr_count * 4
        times = np.arange(n_times) / sfreq
        signal = np.sin(2 * np.pi * 57.1759 * times)[None, None, :]
        freqs, psd = hc.segment_periodograms(signal, sfreq, tr_count=tr_count)
        assert psd.shape == (1, 1, freqs.size)
        assert freqs[1] == pytest.approx(1 / 5.4)
        assert freqs[int(np.argmax(psd[0, 0]))] == pytest.approx(57.1759, abs=freqs[1])

    def test_matched_grid_is_a_subgrid_of_the_high_resolution_grid(self):
        sfreq = 500.0
        n_times = int(0.9 * sfreq) * 24
        data = np.zeros((1, 1, n_times))
        data[..., 0] = 1.0
        freqs_high, _ = hc.segment_periodograms(data, sfreq, tr_count=24)
        freqs_matched, _ = hc.segment_periodograms(data, sfreq, tr_count=6)
        assert np.allclose(freqs_matched, freqs_high[::4], atol=1e-12)

    def test_volume_comb_lands_on_a_bin_in_both_grids(self):
        sfreq = 500.0
        n_times = int(0.9 * sfreq) * 24
        data = np.zeros((1, 1, n_times))
        for tr_count in (24, 6):
            freqs, _ = hc.segment_periodograms(data, sfreq, tr_count=tr_count)
            for harmonic in (18, 37, 55, 74):
                assert np.min(np.abs(freqs - harmonic / 0.9)) < 1e-9

    def test_rejects_segments_shorter_than_one_block(self):
        with pytest.raises(ValueError, match="shorter than one"):
            hc.segment_periodograms(np.zeros((1, 1, 100)), 500.0, tr_count=6)

    def test_rejects_wrong_dimensionality(self):
        with pytest.raises(ValueError, match="n_segments, n_channels, n_times"):
            hc.segment_periodograms(np.zeros((10, 100)), 500.0, tr_count=1)


class TestTileIntervals:
    def test_overlapping_tiles_cover_the_interval(self):
        tiles = hc.tile_intervals(0, 100, block=40, overlap=0.5)
        assert tiles == [(0, 40), (20, 60), (40, 80), (60, 100)]

    def test_no_tiles_when_the_interval_is_too_short(self):
        assert hc.tile_intervals(0, 30, block=40, overlap=0.5) == []

    def test_non_overlapping_tiling(self):
        assert hc.tile_intervals(10, 90, block=40, overlap=0.0) == [(10, 50), (50, 90)]

    def test_rejects_full_overlap(self):
        with pytest.raises(ValueError, match="overlap must be inside"):
            hc.tile_intervals(0, 100, block=10, overlap=1.0)


class TestBlockStandardDeviation:
    def test_ignores_channel_dc_offsets(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(scale=2.0, size=(8, 5000))
        offsets = np.array([500.0, -300.0, 900.0, 0.0, 120.0, -50.0, 700.0, 20.0])[:, None]
        plain = hc.block_standard_deviation(noise, 1000.0)
        with_offsets = hc.block_standard_deviation(noise + offsets, 1000.0)
        assert np.allclose(plain, with_offsets)
        assert np.allclose(plain, 2.0, atol=0.3)

    def test_tracks_an_amplitude_step(self):
        data = np.concatenate([np.ones((4, 2000)) * 0.0, np.ones((4, 2000))], axis=1)
        rng = np.random.default_rng(1)
        data += rng.normal(scale=1.0, size=data.shape) * np.concatenate(
            [np.ones((4, 2000)), np.full((4, 2000), 50.0)], axis=1
        )
        sd = hc.block_standard_deviation(data, 1000.0, block_seconds=0.5)
        assert sd[0] < 2.0
        assert sd[-1] > 30.0


class TestDetectQuietInterval:
    def _profile(self, quiet_blocks, loud_blocks, quiet=15.0, loud=650.0):
        return np.concatenate([np.full(quiet_blocks, quiet), np.full(loud_blocks, loud)])

    def test_finds_the_onset_and_leaves_a_guard(self):
        sd = self._profile(40, 60)  # 10 s quiet at 0.25 s blocks
        start, stop = hc.detect_quiet_interval(sd, sfreq=5000.0)
        assert start == 2500  # 0.5 s lead-in
        assert stop == 40 * 1250 - 1250  # onset minus the 0.25 s guard

    def test_uses_the_whole_recording_when_no_onset_occurs(self):
        sd = np.full(80, 14.0)
        _, stop = hc.detect_quiet_interval(sd, sfreq=5000.0)
        assert stop == 80 * 1250 - 1250

    def test_threshold_choice_does_not_matter_inside_the_gap(self):
        sd = self._profile(40, 60)
        first = hc.detect_quiet_interval(sd, sfreq=5000.0, onset_factor=5.0)
        second = hc.detect_quiet_interval(sd, sfreq=5000.0, onset_factor=25.0)
        assert first == second

    def test_finds_a_short_head_it_cannot_read_from_the_opening_blocks(self):
        # Two seconds of quiet followed by ten of gradient: the opening blocks are no
        # longer a majority, so the level has to come from the profile's extremes.
        sd = self._profile(8, 40)
        start, stop = hc.detect_quiet_interval(sd, sfreq=5000.0)
        assert start == 2500
        assert stop == 8 * 1250 - 1250

    def test_immediate_onset_yields_an_empty_window(self):
        sd = self._profile(1, 47)
        start, stop = hc.detect_quiet_interval(sd, sfreq=5000.0)
        assert stop <= start

    def test_rejects_non_positive_values(self):
        with pytest.raises(ValueError, match="finite and positive"):
            hc.detect_quiet_interval(np.zeros(20), sfreq=5000.0)

    def test_rejects_too_few_blocks(self):
        with pytest.raises(ValueError, match="at least eight blocks"):
            hc.detect_quiet_interval(np.full(4, 10.0), sfreq=5000.0)


class TestReadVolumeMarkers:
    def test_reads_positions(self, tmp_path):
        path = tmp_path / "x.vmrk"
        path.write_text(
            "[Marker Infos]\n"
            "Mk1=New Segment,,1,1,0,20260513110146058017\n"
            "Mk2=SyncStatus,Sync On,1,1,0\n"
            "Mk10=Volume,V  1,63136,1,0\n"
            "Mk12=Volume,V  1,67636,1,0\n",
            encoding="utf-8",
        )
        assert hc.read_volume_markers(path) == [63136, 67636]

    def test_returns_empty_when_absent(self, tmp_path):
        path = tmp_path / "y.vmrk"
        path.write_text("Mk1=New Segment,,1,1,0\n", encoding="utf-8")
        assert hc.read_volume_markers(path) == []


class TestChannelHelpers:
    def test_align_channels_reorders(self):
        order = hc.align_channels(["Cz", "Fp1", "Oz"], ["Fp1", "Oz"])
        assert order.tolist() == [1, 2]

    def test_align_channels_ignores_case(self):
        # BrainVision writes FPz; the BIDS conversion writes Fpz.
        order = hc.align_channels(["FPz", "Cz"], ["Fpz", "Cz"])
        assert order.tolist() == [0, 1]

    def test_align_channels_reports_missing(self):
        with pytest.raises(ValueError, match="P7"):
            hc.align_channels(["Cz", "Fp1"], ["Fp1", "P7"])

    def test_good_channel_mask(self):
        mask = hc.good_channel_mask(["Fp1", "FC2", "Cz"], ["FC2"])
        assert mask.tolist() == [True, False, True]

    def test_all_bad_is_an_error(self):
        with pytest.raises(ValueError, match="Every channel is marked bad"):
            hc.good_channel_mask(["Fp1"], ["Fp1"])
