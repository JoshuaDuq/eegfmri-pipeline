"""Strict artifact-contract tests for the Study 2 spatial-convergence figure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study2 import paths
from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.figures.spatial_convergence import (
    AUDIT_COLUMNS,
    load_spatial_convergence,
)
from studies.pain_study.study2.source_vertex_manifest import ensure_common_source_vertices

BANDS = ("alpha", "beta", "gamma")


def test_loader_reconciles_complete_spatial_family(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)

    summary = load_spatial_convergence(config)

    assert summary.bands == BANDS
    assert summary.n_vertices == 6
    assert summary.n_surrogates == 3
    assert summary.mask.sum() == 6
    assert tuple(summary.audit.columns) == AUDIT_COLUMNS


def test_loader_rejects_differing_band_fmri_map(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    beta_path = paths.spatial_fmri_map_path(config, band="beta")
    beta_map = np.load(beta_path, allow_pickle=False)
    beta_map[0] += 1.0
    _write_fmri_map_and_refresh_hash(config, band="beta", values=beta_map)

    with pytest.raises(ValueError, match="do not share one reference map"):
        load_spatial_convergence(config)


def test_loader_rejects_mismatched_saved_adjusted_p_value(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary_path = paths.spatial_correspondence_summary_path(config)
    summary = pd.read_csv(summary_path, sep="\t")
    summary.loc[summary["band"].eq("alpha"), "holm_adjusted_p_value"] = 0.25
    summary.to_csv(summary_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="does not match recomputed inference"):
        load_spatial_convergence(config)


def test_loader_rejects_map_with_wrong_vertex_count(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    np.save(paths.spatial_eeg_map_path(config, band="alpha"), np.ones(5))

    with pytest.raises(ValueError, match="source vertex manifest"):
        load_spatial_convergence(config)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_loader_rejects_nonboolean_stored_mask(
    tmp_path: Path,
    dtype: type[np.generic],
) -> None:
    config = _write_spatial_artifacts(tmp_path)
    np.save(paths.spatial_mask_path(config), np.ones(6, dtype=dtype))

    with pytest.raises(ValueError, match="must have boolean dtype"):
        load_spatial_convergence(config)


def test_loader_rejects_metadata_naming_wrong_manifest(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    metadata_path = paths.spatial_surrogate_metadata_path(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_vertex_manifest_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong vertex manifest"):
        load_spatial_convergence(config)


def test_loader_rejects_fmri_mismatch_above_absolute_tolerance(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    gamma_path = paths.spatial_fmri_map_path(config, band="gamma")
    gamma_map = np.load(gamma_path, allow_pickle=False)
    gamma_map[0] += 1.1e-12
    _write_fmri_map_and_refresh_hash(config, band="gamma", values=gamma_map)

    with pytest.raises(ValueError, match="do not share one reference map"):
        load_spatial_convergence(config)


def test_loader_requires_every_surrogate_artifact(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    paths.spatial_surrogate_maps_path(config, band="gamma").unlink()

    with pytest.raises(FileNotFoundError, match="gamma surrogate maps artifact not found"):
        load_spatial_convergence(config)


def _write_spatial_artifacts(tmp_path: Path) -> dict:
    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    config["study2"]["spatial_comparison"]["brainsmash_surrogates"] = 3
    ensure_common_source_vertices(
        array_path=paths.source_vertex_manifest_path(config),
        metadata_path=paths.source_vertex_metadata_path(config),
        vertices=(np.asarray([0, 2, 4]), np.asarray([1, 3, 5])),
        common_subject="fsaverage",
        spacing="oct6",
    )

    fmri_map = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    eeg_maps = {
        "alpha": fmri_map.copy(),
        "beta": -fmri_map,
        "gamma": np.asarray([1.0, -1.0, 2.0, -2.0, 3.0, -3.0]),
    }
    surrogate_maps = np.asarray(
        [
            [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
            [-2.0, -3.0, 1.0, 3.0, -1.0, 2.0],
            [1.0, 3.0, -2.0, 2.0, -3.0, -1.0],
        ]
    )
    paths.spatial_dir(config).mkdir(parents=True)
    np.save(paths.spatial_mask_path(config), np.ones(6, dtype=np.bool_))
    for band in BANDS:
        np.save(paths.spatial_eeg_map_path(config, band=band), eeg_maps[band])
        np.save(paths.spatial_fmri_map_path(config, band=band), fmri_map)
        np.save(paths.spatial_surrogate_maps_path(config, band=band), surrogate_maps)

    metadata = {
        "schema_version": 1,
        "method": "BrainSMASH Base",
        "n_surrogates": 3,
        "common_subject": "fsaverage",
        "common_source_space_spacing": "oct6",
        "source_vertex_manifest_sha256": _sha256(paths.source_vertex_manifest_path(config)),
        "analysis_mask_sha256": _sha256(paths.spatial_mask_path(config)),
        "bands": {
            band: {
                "seed": 42 + band_index,
                "masked_vertices": 6,
                "fmri_map_sha256": _sha256(paths.spatial_fmri_map_path(config, band=band)),
            }
            for band_index, band in enumerate(BANDS)
        },
    }
    paths.spatial_surrogate_metadata_path(config).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "band": BANDS,
            "spatial_r": [1.0, -1.0, -2.0 / 7.0],
            "p_value": [0.5, 0.5, 1.0],
            "meaningful": [True, True, True],
            "holm_adjusted_p_value": [1.0, 1.0, 1.0],
            "holm_significant": [False, False, False],
        }
    ).to_csv(paths.spatial_correspondence_summary_path(config), sep="\t", index=False)
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fmri_map_and_refresh_hash(
    config: dict,
    *,
    band: str,
    values: np.ndarray,
) -> None:
    fmri_path = paths.spatial_fmri_map_path(config, band=band)
    np.save(fmri_path, values)
    metadata_path = paths.spatial_surrogate_metadata_path(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["bands"][band]["fmri_map_sha256"] = _sha256(fmri_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
