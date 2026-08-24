"""End-to-end wiring: does a first-level run leave behind what the report reads?

The helpers are unit-tested individually, but the defect this guards against was never
in a helper. It was that ``process_subject`` computed the effect and variance maps and
threw them away, recorded a *discovered* single-run brain mask as though it were the
fitted one, and left the design matrices and contrast vector out of the manifest
entirely. Every one of those helpers worked; nothing connected them, so the dual-coded
panel, the standard-error panel and the whole design section could not render from any
real run.

So this exercises ``process_subject`` itself and reads the manifest off disk.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.manifest import MANIFEST_FILENAME, read_manifest
from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline
from tests.utils.pipelines_test_utils import DotConfig

SHAPE = (8, 8, 8)
COLUMNS = ["cond_a", "cond_b", "trans_x", "constant"]


class _ContrastCfg:
    """A stand-in for ContrastBuilderConfig carrying the settings the manifest records."""

    name = "pain_vs_warm"
    output_type = "z-score"
    resample_to_freesurfer = False
    fmriprep_space = "T1w"
    smoothing_fwhm = 6.0
    signal_scaling = False
    write_design_matrix = True
    hrf_model = "spm + derivative"
    drift_model = "cosine"
    high_pass_hz = 0.008
    low_pass_hz = None
    confounds_strategy = "motion24+wmcsf"
    auto_compcor_n = 5
    contrast_type = "t-test"
    formula = None


class _PlotCfg:
    enabled = False
    space = "native"
    include_effect_size = True
    include_standard_error = True
    include_signatures = False
    threshold_mode = "z"
    z_threshold = 2.3
    fdr_q = 0.05
    cluster_min_voxels = 0
    two_sided = True
    radiological = False

    def normalized(self):
        return self


def _image(fill: float = 1.0) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.full(SHAPE, fill, dtype=np.float32), np.eye(4))


def _series(fill: float = 1.0, frames: int = 30) -> nib.Nifti1Image:
    shape = (*SHAPE, frames)
    return nib.Nifti1Image(np.full(shape, fill, dtype=np.float32), np.eye(4))


class _Masker:
    def inverse_transform(self, series: np.ndarray) -> nib.Nifti1Image:
        data = np.asarray(series, dtype=np.float32).T.reshape(*SHAPE, -1)
        return nib.Nifti1Image(data, np.eye(4))


def _design_tsv(directory: Path, run: str) -> Path:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.standard_normal((30, len(COLUMNS))), columns=COLUMNS)
    frame["constant"] = 1.0
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"design_{run}.tsv"
    frame.to_csv(path, sep="\t", index=True, index_label="frame")
    return path


def _fitted_model() -> SimpleNamespace:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.standard_normal((30, len(COLUMNS))), columns=COLUMNS)
    frame["constant"] = 1.0

    def compute_contrast(_argument, output_type: str):
        return _image(2.0 if output_type == "effect_size" else 0.25)

    coefficients = np.zeros((len(COLUMNS), int(np.prod(SHAPE))))
    response = np.full((len(frame), int(np.prod(SHAPE))), 0.25)
    return SimpleNamespace(
        design_matrices_=[frame],
        labels_=[np.zeros(int(np.prod(SHAPE)))],
        results_=[{0.0: SimpleNamespace(theta=coefficients, Y=response)}],
        masker_=_Masker(),
        compute_contrast=compute_contrast,
    )


@pytest.fixture
def exploding_contrast() -> SimpleNamespace:
    """A fitted model whose extra contrasts fail, as a degenerate design can."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.standard_normal((30, len(COLUMNS))), columns=COLUMNS)
    frame["constant"] = 1.0

    def compute_contrast(_argument, output_type: str):
        raise RuntimeError("contrast could not be computed")

    coefficients = np.zeros((len(COLUMNS), int(np.prod(SHAPE))))
    response = np.full((len(frame), int(np.prod(SHAPE))), 0.25)
    return SimpleNamespace(
        design_matrices_=[frame],
        labels_=[np.zeros(int(np.prod(SHAPE)))],
        results_=[{0.0: SimpleNamespace(theta=coefficients, Y=response)}],
        masker_=_Masker(),
        compute_contrast=compute_contrast,
    )


@pytest.fixture
def run_first_level(tmp_path: Path):
    """Run ``process_subject`` against a stubbed GLM and return the manifest."""

    def _run(*, contrast_cfg=None, stats_cfg=None, fitted_model=None):
        cfg = contrast_cfg or _ContrastCfg()
        deriv = tmp_path / "derivatives"
        qc_dir = tmp_path / "qc"
        design_paths = [str(_design_tsv(qc_dir, "run-01"))]

        (tmp_path / "bids").mkdir(parents=True, exist_ok=True)
        deriv.mkdir(parents=True, exist_ok=True)
        (tmp_path / "fs" / "sub-0001").mkdir(parents=True, exist_ok=True)

        # Constructed the way the existing pipeline tests do: the real __init__
        # resolves paths from study config that does not exist here.
        pipeline = object.__new__(FmriAnalysisPipeline)
        pipeline.config = DotConfig(
            {
                "paths": {
                    "bids_fmri_root": str(tmp_path / "bids"),
                    "freesurfer_dir": str(tmp_path / "fs"),
                }
            }
        )
        pipeline.deriv_root = deriv
        pipeline.logger = logging.getLogger("test-first-level")

        glm = SimpleNamespace(
            flm=fitted_model if fitted_model is not None else _fitted_model(),
            mask_img=_image(1.0),
        )
        run_meta = {
            "output_type": "z_score",
            "tr": 0.9,
            "analysis_space": "T1w",
            "confounds_strategy": "motion24+wmcsf",
            "confound_columns": ["trans_x", "white_matter"],
            "design_matrix_tsv_paths": design_paths,
            "included_bold_paths": [str(tmp_path / "sub-0001_run-01_bold.nii.gz")],
            "included_confounds_paths": [],
            "retained_frame_indices": [list(range(30))],
            "skipped_runs": [],
        }

        def build(**_kwargs):
            return _image(3.0), run_meta, glm, "cond_a - cond_b", "z_score"

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=build,
            resample_to_freesurfer=lambda img, _dir, **_kw: img,
            ContrastBuilderConfig=type(cfg),
        )
        with patch.dict(sys.modules, {"fmri_pipeline.analysis.contrast_builder": fake_builder}):
            pipeline.process_subject(
                "0001",
                "heat",
                contrast_cfg=cfg,
                stats_cfg=stats_cfg or _PlotCfg(),
                dry_run=False,
            )

        manifests = sorted(deriv.rglob(MANIFEST_FILENAME))
        assert manifests, "the run wrote no manifest"
        return read_manifest(manifests[0])

    return _run


# --- the maps the report needs --------------------------------------------


def test_the_effect_map_is_written_and_recorded(run_first_level) -> None:
    # Computed off the fitted model and then discarded, so the dual-coded panel --
    # the report's lead figure -- could not render from any real run.
    manifest = run_first_level()
    assert manifest.effect_map is not None
    assert Path(manifest.effect_map).exists()


def test_the_variance_map_is_written_and_recorded(run_first_level) -> None:
    manifest = run_first_level()
    assert manifest.variance_map is not None
    assert Path(manifest.variance_map).exists()


def test_the_model_response_residual_series_is_written_and_recorded(run_first_level) -> None:
    manifest = run_first_level()

    assert len(manifest.residual_paths) == 1
    assert manifest.residual_paths[0].is_file()
    assert nib.load(str(manifest.residual_paths[0])).shape == (*SHAPE, 30)


def test_the_model_response_predicted_series_is_written_and_recorded(run_first_level) -> None:
    manifest = run_first_level()

    assert len(manifest.predicted_paths) == 1
    assert manifest.predicted_paths[0].is_file()
    assert nib.load(str(manifest.predicted_paths[0])).shape == (*SHAPE, 30)


def test_the_fitted_mask_is_written_and_claimed_as_the_fitted_one(run_first_level) -> None:
    # A discovered mask is one run's; the fitted mask is the intersection across runs.
    # Only the second justifies "voxels outside this mask were not tested".
    manifest = run_first_level()
    assert manifest.mask is not None
    assert Path(manifest.mask).exists()
    assert manifest.mask_is_analysis_mask is True


def test_the_saved_maps_share_the_stat_map_s_geometry(run_first_level) -> None:
    # The dual-coded panel refuses a mismatched pair outright, and a mask on the
    # wrong grid is silently ignored -- costing the colour limits their brain with no
    # error anywhere.
    manifest = run_first_level()
    stat = nib.load(str(manifest.stat_map))
    for other in (manifest.effect_map, manifest.variance_map, manifest.mask):
        image = nib.load(str(other))
        assert image.shape == stat.shape
        assert np.allclose(image.affine, stat.affine)


def test_the_maps_stay_co_registered_through_a_freesurfer_resample(run_first_level) -> None:
    class _Resampled(_ContrastCfg):
        resample_to_freesurfer = True

    manifest = run_first_level(contrast_cfg=_Resampled())
    stat = nib.load(str(manifest.stat_map))
    for other in (manifest.effect_map, manifest.variance_map, manifest.mask):
        assert nib.load(str(other)).shape == stat.shape


# --- the design section ----------------------------------------------------


def test_the_design_matrices_are_recorded(run_first_level) -> None:
    # Written to qc/ by the contrast builder and never recorded, so the design
    # section -- matrix, contrast strip, correlation, VIF, efficiency -- was dark.
    manifest = run_first_level()
    assert manifest.design_matrices
    assert all(Path(p).exists() for p in manifest.design_matrices)


def test_the_contrast_expression_is_expanded_against_the_design(run_first_level) -> None:
    manifest = run_first_level()
    assert manifest.contrast_columns == tuple(COLUMNS)
    weights = dict(zip(manifest.contrast_columns, manifest.contrast_vector))
    assert weights["cond_a"] == pytest.approx(1.0)
    assert weights["cond_b"] == pytest.approx(-1.0)
    assert weights["trans_x"] == pytest.approx(0.0)


# --- the configuration -----------------------------------------------------


def test_the_model_settings_are_recorded(run_first_level) -> None:
    settings = dict(run_first_level().model_settings)
    assert settings["HRF model"] == "spm + derivative"
    assert settings["Drift model"] == "cosine"
    assert "125 s" in settings["High-pass cutoff"]


def test_the_resolved_confound_columns_are_recorded(run_first_level) -> None:
    # The strategy name is not the same fact: "auto" resolves differently per run.
    assert run_first_level().confound_columns == ("trans_x", "white_matter")


def test_the_acquisition_facts_survive(run_first_level) -> None:
    manifest = run_first_level()
    assert manifest.t_r == pytest.approx(0.9)
    assert manifest.smoothing_fwhm == pytest.approx(6.0)
    assert manifest.included_runs == ("run-01",)


# --- degradation -----------------------------------------------------------


def test_a_contrast_that_cannot_be_recomputed_surfaces_the_error(
    run_first_level, exploding_contrast
) -> None:
    with pytest.raises(RuntimeError, match="contrast could not be computed"):
        run_first_level(fitted_model=exploding_contrast)


def test_declining_the_detail_maps_leaves_them_unrecorded(run_first_level) -> None:
    # Gated on the intent flags, not on whether plotting is enabled: the report is a
    # separate step and cannot ask for these retroactively.
    class _Declined(_PlotCfg):
        include_effect_size = False
        include_standard_error = False

    manifest = run_first_level(stats_cfg=_Declined())
    assert manifest.effect_map is None
    assert manifest.variance_map is None
    assert manifest.mask_is_analysis_mask is True
