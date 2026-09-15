from __future__ import annotations

import pandas as pd

from fmri_pipeline.analysis.report import subject


def _frame(n_clusters: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Cluster ID": [str(i + 1) for i in range(n_clusters)],
            "X": [0.0] * n_clusters,
            "Y": [0.0] * n_clusters,
            "Z": [0.0] * n_clusters,
            "Peak Stat": [10.0 - 0.01 * i for i in range(n_clusters)],
            "Cluster Size (mm3)": [100] * n_clusters,
        }
    )


def test_a_long_cluster_table_is_capped_for_display() -> None:
    # Measured on sub-0003 once inference runs on the z map: 288 rows over 224
    # clusters at |z| > 2.3. Every row still reaches the TSV; only the page is capped.
    shown, note = subject.cap_cluster_rows(_frame(224), limit=20)
    assert len(shown) == 20
    assert "224" in note and "20" in note


def test_the_strongest_clusters_are_the_ones_kept() -> None:
    shown, _note = subject.cap_cluster_rows(_frame(50), limit=5)
    assert list(shown["Cluster ID"]) == ["1", "2", "3", "4", "5"]


def test_a_short_table_is_untouched_and_unannotated() -> None:
    shown, note = subject.cap_cluster_rows(_frame(6), limit=20)
    assert len(shown) == 6
    assert note == ""


def test_subpeak_rows_travel_with_their_parent() -> None:
    # nilearn writes sub-peaks as 1a, 1b under their parent. Cutting between a parent
    # and its sub-peaks would leave rows referring to a cluster the table no longer has.
    frame = _frame(3)
    frame = pd.concat(
        [frame, _frame(1).assign(**{"Cluster ID": ["2a"], "Peak Stat": [9.5]})],
        ignore_index=True,
    )
    shown, _note = subject.cap_cluster_rows(frame, limit=2)
    kept = list(shown["Cluster ID"])
    assert "2a" not in kept or "2" in kept
