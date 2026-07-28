"""Detections from the full recording must reach the table that builds the cleaned data.

MNE-BIDS-Pipeline's own ECG step scores components against ~50 s of the first run. The
cardiac review measures every beat of every run, and these tests cover the path that turns
that measurement into exclusions without trampling anything a reviewer or upstream wrote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.ica_cardiac_review import (
    ComponentCardiacReview,
    ctps_promotions,
)
from eeg_pipeline.preprocessing.ica_exclusions import (
    promote_exclusions,
    read_component_statuses,
    reviewed_exclusions,
)


def _write_components(path, statuses: list[str], descriptions: list[str] | None = None):
    pd.DataFrame(
        {
            "component": np.arange(len(statuses)),
            "type": ["ica"] * len(statuses),
            "description": ["Independent Component"] * len(statuses),
            "status": statuses,
            "status_description": descriptions or ["n/a"] * len(statuses),
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def _review(ctps_flags: np.ndarray, run_ids: tuple[str, ...]) -> ComponentCardiacReview:
    run_count, component_count = ctps_flags.shape
    scores = np.zeros((run_count, component_count))
    return ComponentCardiacReview(
        run_ids=run_ids,
        times=np.linspace(-0.4, 0.6, 11),
        run_mean_z=np.zeros((run_count, component_count, 11)),
        correlation_scores=scores,
        ctps_scores=scores,
        correlation_flags=np.zeros_like(ctps_flags),
        ctps_flags=ctps_flags,
        r_locked_epoch_counts=np.full(run_count, 60),
        run_ecg_z=np.zeros((run_count, 11)),
    )


# --------------------------------------------------------------------------------------
# Which components get promoted
# --------------------------------------------------------------------------------------


def test_a_component_flagged_in_a_majority_of_runs_is_promoted() -> None:
    flags = np.zeros((6, 3), dtype=bool)
    flags[:4, 1] = True  # 4 of 6 runs

    promotions = ctps_promotions(
        _review(flags, tuple(f"run-{i}" for i in range(6))), minimum_run_fraction=0.5
    )

    assert set(promotions) == {1}


def test_a_component_flagged_in_one_run_of_six_is_not_promoted() -> None:
    """BCG topography moves with head position, so a lone run is as likely a threshold
    crossing as a cardiac component. That call belongs to the reviewer, not to this."""
    flags = np.zeros((6, 3), dtype=bool)
    flags[2, 0] = True

    assert ctps_promotions(
        _review(flags, tuple(f"run-{i}" for i in range(6))), minimum_run_fraction=0.5
    ) == {}


def test_the_description_names_the_runs_behind_the_call() -> None:
    flags = np.zeros((4, 2), dtype=bool)
    flags[[0, 1, 3], 0] = True

    promotions = ctps_promotions(
        _review(flags, ("run-a", "run-b", "run-c", "run-d")), minimum_run_fraction=0.5
    )

    description = promotions[0]
    assert "3/4" in description
    assert "run-a, run-b, run-d" in description
    assert "run-c" not in description


def test_the_run_fraction_is_applied_against_usable_runs_only() -> None:
    """Runs whose ECG never resolved are absent from the review, so the denominator is
    the runs that produced a beat train rather than the runs that were recorded."""
    flags = np.zeros((2, 2), dtype=bool)
    flags[0, 1] = True  # 1 of the 2 runs that resolved

    promotions = ctps_promotions(_review(flags, ("run-1", "run-4")), minimum_run_fraction=0.5)

    assert set(promotions) == {1}
    assert "1/2" in promotions[1]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_an_out_of_range_run_fraction_is_rejected(fraction) -> None:
    flags = np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="minimum_run_fraction"):
        ctps_promotions(_review(flags, ("a", "b")), minimum_run_fraction=fraction)


def test_a_review_with_no_usable_runs_promotes_nothing() -> None:
    review = _review(np.zeros((0, 3), dtype=bool), ())

    assert ctps_promotions(review, minimum_run_fraction=0.5) == {}


# --------------------------------------------------------------------------------------
# How the component table is updated
# --------------------------------------------------------------------------------------


def test_promotion_marks_the_component_bad_in_the_table(tmp_path) -> None:
    path = _write_components(tmp_path / "components.tsv", ["good", "good", "good"])

    newly = promote_exclusions(path, components={1: "cardiac"}, component_count=3)

    assert newly == [1]
    assert reviewed_exclusions(path, component_count=3) == [1]


def test_promotion_does_not_overwrite_a_row_that_is_already_bad(tmp_path) -> None:
    """Upstream's ICLabel verdict and a reviewer's own edit both land in this column.
    Restating the reason would erase why the component was really excluded."""
    path = _write_components(
        tmp_path / "components.tsv",
        ["bad", "good"],
        ["Auto-detected eye blink (MNE-ICALabel)", "n/a"],
    )

    newly = promote_exclusions(path, components={0: "cardiac", 1: "cardiac"}, component_count=2)

    statuses = read_component_statuses(path, component_count=2)
    assert newly == [1]
    assert statuses.loc[0, "status_description"] == "Auto-detected eye blink (MNE-ICALabel)"
    assert statuses.loc[1, "status_description"] == "cardiac"


def test_promotion_is_idempotent(tmp_path) -> None:
    path = _write_components(tmp_path / "components.tsv", ["good", "good"])

    first = promote_exclusions(path, components={0: "cardiac"}, component_count=2)
    second = promote_exclusions(path, components={0: "cardiac"}, component_count=2)

    assert first == [0]
    assert second == []
    assert reviewed_exclusions(path, component_count=2) == [0]


def test_promoting_nothing_leaves_the_file_untouched(tmp_path) -> None:
    path = _write_components(tmp_path / "components.tsv", ["good", "good"])
    before = path.read_bytes()

    assert promote_exclusions(path, components={}, component_count=2) == []
    assert path.read_bytes() == before


def test_a_component_index_outside_the_decomposition_is_rejected(tmp_path) -> None:
    path = _write_components(tmp_path / "components.tsv", ["good", "good"])

    with pytest.raises(ValueError, match="has 2 components"):
        promote_exclusions(path, components={5: "cardiac"}, component_count=2)
