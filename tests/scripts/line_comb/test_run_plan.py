"""Benchmark and apply must consume one immutable adaptive removal plan."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.line_comb import remove as rlc


def _stationary_model() -> lr.AdaptiveCombModel:
    """Two windows sharing one fundamental, so the grid sits at exactly 1.2 * k."""
    return lr.build_adaptive_comb_model(
        _estimate(1.2, 8e-5), (_estimate(1.2, 1e-4), _estimate(1.2, 1e-4))
    )


def _estimate(fundamental_hz: float, uncertainty_hz: float) -> lr.CombEstimate:
    return lr.CombEstimate(
        fundamental_hz=fundamental_hz,
        harmonics_used=tuple(range(24, 80)),
        harmonic_positions_hz=tuple(fundamental_hz * harmonic for harmonic in range(24, 80)),
        residual_rms_hz=0.05,
        max_abs_residual_hz=0.10,
        fundamental_jackknife_se_hz=uncertainty_hz,
        isolated_hz=(57.25,),
        isolated_prominence_db=(15.0,),
    )


def _model() -> lr.AdaptiveCombModel:
    whole = _estimate(1.2, 8e-5)
    return lr.build_adaptive_comb_model(
        whole,
        (_estimate(1.1998, 1e-4), _estimate(1.2002, 2e-4)),
    )


def _isolated_minimum_width(settings: rlc.RemovalSettings) -> float:
    return rlc.spectrum_fit_nominal_resolution_hz(settings.filter_length)


def test_each_adaptive_window_owns_its_targets_and_uncertainty_aware_widths():
    settings = rlc.RemovalSettings()

    plan = rlc.build_removal_plan(
        _model(),
        bounds=((0, 100), (50, 150)),
        narrow_targets_hz=((), ()),
        settings=settings,
    )

    assert len(plan.windows) == 2
    assert plan.windows[0].bounds == (0, 100)
    assert plan.windows[0].targets_hz != plan.windows[1].targets_hz
    for window in plan.windows:
        assert len(window.targets_hz) == len(window.notch_widths_hz)
        assert max(window.notch_widths_hz) > max(window.targets_hz) / settings.notch_width_ratio
        isolated_index = min(
            range(len(window.targets_hz)),
            key=lambda index: abs(window.targets_hz[index] - 57.25),
        )
        assert window.notch_widths_hz[isolated_index] == pytest.approx(
            max(
                57.25 / settings.notch_width_ratio,
                settings.notch_width_min_hz,
                _isolated_minimum_width(settings),
            )
        )


def _fixed_plan():
    settings = rlc.RemovalSettings()
    return rlc.build_removal_plan(
        _stationary_model(),
        bounds=((0, 100), (50, 150)),
        narrow_targets_hz=((), ()),
        settings=settings,
    )


def test_comb_adjacent_targets_keep_a_narrow_line_sized_width():
    settings = rlc.RemovalSettings()

    plan = rlc.build_removal_plan(
        _stationary_model(),
        bounds=((0, 100), (50, 150)),
        narrow_targets_hz=((27.72,), ()),
        settings=settings,
    )

    window = plan.windows[0]
    assert window.narrow_targets_hz == (27.72,)
    index = _nearest(window, 27.72)
    assert window.targets_hz[index] == pytest.approx(27.72)
    assert window.notch_widths_hz[index] == pytest.approx(
        max(27.72 / settings.notch_width_ratio, settings.notch_width_min_hz)
    )
    assert window.notch_widths_hz[index] < lr.LINE_WIDTH_CEILING_HZ


def test_a_resolved_neighbor_outside_an_isolated_fit_is_retained():
    settings = rlc.RemovalSettings()

    plan = rlc.build_removal_plan(
        _stationary_model(),
        bounds=((0, 100), (50, 150)),
        narrow_targets_hz=((57.35,), ()),
        settings=settings,
    )

    window = plan.windows[0]
    assert window.narrow_targets_hz == (57.35,)
    assert sum(abs(target - 57.25) < 0.2 for target in window.targets_hz) == 2


def test_a_narrow_target_is_retained_when_its_support_crosses_the_model_edge():
    settings = rlc.RemovalSettings()

    plan = rlc.build_removal_plan(
        _stationary_model(),
        bounds=((0, 100), (50, 150)),
        narrow_targets_hz=((27.63,), ()),
        settings=settings,
    )

    window = plan.windows[0]
    assert any(abs(target - 27.63) < 1e-9 for target in window.narrow_targets_hz)


def _covers(window, frequency_hz):
    return any(
        abs(frequency_hz - target) <= width / 2.0
        for target, width in zip(window.targets_hz, window.notch_widths_hz)
    )


def test_offset_support_is_covered_where_it_was_observed_and_not_mirrored():
    """Support beside a target costs its own width, not twice its offset.

    A notch is symmetric about its target, so widening the target to reach a peak 0.1 Hz
    below it would also empty 0.1 Hz above it, where the spectrum was flat. The peak is
    covered by a notch centred on the peak instead, with the localization margin included.
    """
    freqs = np.arange(20.0, 30.0, 0.05)
    prominence = np.zeros(freqs.size)
    peak = int(np.argmin(np.abs(freqs - 24.9)))
    prominence[peak] = 15.0
    window = rlc.AdaptiveWindowRemovalPlan(
        bounds=(0, 200),
        estimate=_estimate(1.2, 1e-4),
        targets_hz=(25.0,),
        notch_widths_hz=(0.1,),
        narrow_targets_hz=(),
    )

    expanded = rlc._expand_window_to_observed_support(
        window,
        ((freqs, prominence, prominence),),
        rlc.RemovalSettings(),
        localization_margin_hz=0.025,
    )

    assert _covers(expanded, 24.9)
    # The localization margin is carried on each side of the observed support.
    assert _covers(expanded, 24.9 - 0.02)
    assert _covers(expanded, 24.9 + 0.02)
    # Every notch stays positive even where the support is one bin wide.
    assert all(width > 0.0 for width in expanded.notch_widths_hz)
    # The mirror image of the peak, which the old symmetric widening also removed.
    assert not _covers(expanded, 25.1)
    # The validated target keeps its own width.
    assert expanded.notch_widths_hz[0] == pytest.approx(0.1)


def test_single_bin_support_still_asks_for_a_positive_notch():
    """The production caller passes no localization margin, so the bin span is the floor.

    A peak confined to one bin has ``left == right``, and a notch of the support's bare
    extent would be zero wide -- which AdaptiveWindowRemovalPlan rejects, so the whole
    recording's plan raises rather than the width being quietly wrong.
    """
    freqs = np.arange(20.0, 30.0, 0.05)
    prominence = np.zeros(freqs.size)
    prominence[int(np.argmin(np.abs(freqs - 24.9)))] = 15.0
    window = rlc.AdaptiveWindowRemovalPlan(
        bounds=(0, 200),
        estimate=_estimate(1.2, 1e-4),
        targets_hz=(25.0,),
        notch_widths_hz=(0.1,),
        narrow_targets_hz=(),
    )

    expanded = rlc._expand_window_to_observed_support(
        window,
        ((freqs, prominence, prominence),),
        rlc.RemovalSettings(),
    )

    assert all(width > 0.0 for width in expanded.notch_widths_hz)
    assert _covers(expanded, 24.9)


def _nearest(window, frequency):
    return min(range(len(window.targets_hz)), key=lambda i: abs(window.targets_hz[i] - frequency))


def test_a_window_cannot_move_a_comb_target_onto_a_peak_selected_from_that_window():
    """Independent recurrent lines belong in the session catalogue, not in a comb warp."""
    window = _fixed_plan().windows[0]

    assert window.targets_hz[_nearest(window, 46.8)] == pytest.approx(46.8, abs=1e-9)


def test_isolated_and_comb_widths_remain_distinct():
    settings = rlc.RemovalSettings()
    window = _fixed_plan().windows[0]

    isolated_width = window.notch_widths_hz[_nearest(window, 57.25)]
    expected_isolated_width = max(
        57.25 / settings.notch_width_ratio,
        settings.notch_width_min_hz,
        _isolated_minimum_width(settings),
    )
    assert isolated_width == pytest.approx(expected_isolated_width)
    comb = _nearest(window, 46.8)
    assert window.notch_widths_hz[comb] < isolated_width


def test_benchmark_and_apply_accept_the_same_plan_type():
    assert "plan" in inspect.signature(rlc.benchmark_run).parameters
    assert "plan" in inspect.signature(rlc.apply_run).parameters


def test_production_pipeline_has_one_fundamental_scope():
    assert "fundamental_scope" not in inspect.signature(rlc.settings_fingerprint).parameters
    source = inspect.getsource(rlc.main)
    assert "--fundamental-scope" not in source
    assert "--limit" not in source
    assert "--subjects" not in source


class TestTheAuditRemainsConstructible:
    """The fixed 0.15 Hz residual search must have a complete matched null."""

    def test_a_matched_null_can_actually_be_built_at_the_configured_reach(self):
        settings = rlc.RemovalSettings()
        reach = lr.RESIDUAL_SEARCH_HZ
        freqs = np.arange(3.0, 99.8, 1 / 54.0)
        targets = np.array([k * settings.nominal_fundamental_hz for k in range(22, 84)])
        widths = np.maximum(targets / settings.notch_width_ratio, settings.notch_width_min_hz)
        after = np.random.default_rng(0).normal(size=freqs.size)

        nulls = lr._matched_null_maxima(freqs, after, targets, np.maximum(widths / 2, reach))

        assert len(nulls) == 40
