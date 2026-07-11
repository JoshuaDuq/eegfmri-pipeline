from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study2 import paths
from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.source_family import (
    compute_source_family_inference,
    summarize_source_family,
)
from studies.pain_study.study2.source_vertex_manifest import ensure_common_source_vertices

BANDS = ("alpha", "beta", "gamma")
SUBJECTS = ("sub-0001", "sub-0002", "sub-0003", "sub-0004")


def test_primary_source_reader_computes_fisher_mean_and_corrected_contours(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )

    config = _write_artifacts(tmp_path)
    result = load_primary_source_associations(config)

    expected_alpha = np.tanh(np.mean(_observed_maps()["alpha"], axis=0))
    np.testing.assert_allclose(result.band_results["alpha"].effect_r, expected_alpha)
    assert result.bands == BANDS
    assert result.subject_ids == SUBJECTS
    assert result.n_subjects == 4
    assert result.display_limit == pytest.approx(
        max(np.max(np.abs(band.effect_r)) for band in result.band_results.values())
    )
    assert result.band_results["alpha"].corrected_vertex_mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert not result.band_results["beta"].corrected_vertex_mask.any()
    assert result.band_results["alpha"].holm_q_value == pytest.approx(0.03)
    assert result.band_results["beta"].holm_q_value == pytest.approx(1.0)
    assert set(result.vertices.columns) == {
        "band",
        "hemisphere",
        "vertex_number",
        "effect_r",
        "t_value",
        "cluster_id",
        "corrected_contour",
    }
    assert set(result.clusters.columns) == {
        "band",
        "cluster_id",
        "sign",
        "n_vertices",
        "cluster_mass",
        "max_cluster_p_value",
        "band_holm_q_value",
        "corrected_contour",
    }


def test_primary_source_reader_rejects_inconsistent_fisher_maps(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )

    config = _write_artifacts(tmp_path)
    partial_path = paths.source_stage_partial_r_path(config, band="beta")
    partial = np.load(partial_path)
    partial[0, 0] += 0.1
    np.save(partial_path, partial)

    with pytest.raises(ValueError, match="partial-r and Fisher-z"):
        load_primary_source_associations(config)


def test_primary_source_reader_rejects_different_band_cohorts(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )

    config = _write_artifacts(tmp_path)
    qc_path = paths.source_stage_qc_path(config, band="beta")
    qc = pd.read_csv(qc_path, sep="\t")
    qc.loc[[0, 1], "subject_id"] = qc.loc[[1, 0], "subject_id"].to_numpy()
    qc.to_csv(qc_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="same valid participant cohort"):
        load_primary_source_associations(config)


def test_primary_source_reader_rejects_stale_family_summary(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )

    config = _write_artifacts(tmp_path)
    summary_path = paths.source_family_summary_path(config)
    summary = pd.read_csv(summary_path, sep="\t")
    summary.loc[summary["band"].eq("alpha"), "holm_q_value"] = 0.9
    summary.to_csv(summary_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="does not match recomputed inference"):
        load_primary_source_associations(config)


def test_primary_source_reader_rejects_vertex_manifest_dimension_mismatch(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )

    config = _write_artifacts(tmp_path, vertices=(np.array([0, 1]), np.array([2, 3, 4])))

    with pytest.raises(ValueError, match="vertex manifest"):
        load_primary_source_associations(config)


def test_primary_source_reader_requires_article_cohort(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )

    config = _write_artifacts(tmp_path)
    config["study2"]["source_stage"]["min_source_valid_subjects"] = 5

    with pytest.raises(ValueError, match="at least 5 source-valid participants"):
        load_primary_source_associations(config)


def _write_artifacts(
    tmp_path: Path,
    *,
    vertices: tuple[np.ndarray, np.ndarray] = (np.array([0, 2, 4]), np.array([1, 3, 5])),
) -> dict:
    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    config["study2"]["source_stage"]["min_source_valid_subjects"] = 4
    observed_maps = _observed_maps()
    null_maps = {band: np.zeros((99, 4, 6), dtype=float) for band in BANDS}
    adjacency = _chain_adjacency(6)

    ensure_common_source_vertices(
        array_path=paths.source_vertex_manifest_path(config),
        metadata_path=paths.source_vertex_metadata_path(config),
        vertices=vertices,
        common_subject="fsaverage",
        spacing="oct6",
    )
    paths.source_stage_dir(config).mkdir(parents=True, exist_ok=True)
    paths.inference_dir(config).mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        np.save(paths.source_stage_fisher_z_path(config, band=band), observed_maps[band])
        np.save(paths.source_stage_partial_r_path(config, band=band), np.tanh(observed_maps[band]))
        np.save(paths.null_source_maps_path(config, band=band), null_maps[band])
        pd.DataFrame(
            {
                "subject_id": SUBJECTS,
                "band": ["combined"] * len(SUBJECTS),
                "source_stage_criteria_met": [True] * len(SUBJECTS),
            }
        ).to_csv(paths.source_stage_qc_path(config, band=band), sep="\t", index=False)
    np.save(paths.source_adjacency_path(config), adjacency)
    family = compute_source_family_inference(
        observed_maps_by_band=observed_maps,
        null_maps_by_band=null_maps,
        adjacency=adjacency,
        cluster_forming_p=0.01,
        alpha=0.05,
    )
    summarize_source_family(family).to_csv(
        paths.source_family_summary_path(config),
        sep="\t",
        index=False,
    )
    return config


def _observed_maps() -> dict[str, np.ndarray]:
    alpha = np.array(
        [
            [0.50, 0.45, 0.02, -0.02, 0.01, -0.01],
            [0.52, 0.47, -0.02, 0.02, -0.01, 0.01],
            [0.48, 0.43, 0.01, -0.01, 0.02, -0.02],
            [0.51, 0.46, -0.01, 0.01, -0.02, 0.02],
        ]
    )
    centered = np.array(
        [
            [0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
            [-0.02, 0.02, -0.01, 0.01, -0.02, 0.02],
            [0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
            [-0.01, 0.01, -0.02, 0.02, -0.01, 0.01],
        ]
    )
    return {"alpha": alpha, "beta": centered, "gamma": centered[:, ::-1]}


def _chain_adjacency(n_vertices: int) -> np.ndarray:
    adjacency = np.eye(n_vertices, dtype=bool)
    indices = np.arange(n_vertices - 1)
    adjacency[indices, indices + 1] = True
    adjacency[indices + 1, indices] = True
    return adjacency
