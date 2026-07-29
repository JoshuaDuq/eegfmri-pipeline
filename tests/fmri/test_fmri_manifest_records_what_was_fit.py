"""The manifest has to record what the model actually produced.

Every figure in the report reads the manifest and nothing else. A field the fitting
path leaves empty is a panel that cannot render from any real run, however well the
code that draws it is tested -- which is exactly what happened to the dual-coded
panel, the standard-error panel, and the whole design section.
"""

from __future__ import annotations

import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.pipelines import fmri_analysis as pipeline_module
from fmri_pipeline.pipelines.fmri_analysis import (
    FmriAnalysisPipeline,
    _contrast_vector_for_design,
)


def _design(columns=("pain", "nonpain", "trans_x", "constant"), n=40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.standard_normal((n, len(columns))), columns=list(columns))
    if "constant" in frame.columns:
        frame["constant"] = 1.0
    return frame


def _glm(design_matrices) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        flm=types.SimpleNamespace(design_matrices_=design_matrices)
    )


# --- the contrast vector --------------------------------------------------


def test_a_contrast_expression_becomes_weights_against_the_design_columns() -> None:
    # The manifest carried only the expression string, so the contrast strip beneath
    # the design matrix and the design's efficiency could never be computed.
    vector, columns = _contrast_vector_for_design(
        glm_result=_glm([_design()]), contrast_def="pain - nonpain"
    )
    assert columns == ["pain", "nonpain", "trans_x", "constant"]
    assert vector == pytest.approx([1.0, -1.0, 0.0, 0.0])


def test_an_explicit_vector_passes_through_when_it_fits_the_design() -> None:
    vector, _columns = _contrast_vector_for_design(
        glm_result=_glm([_design()]), contrast_def=[1.0, -1.0, 0.0, 0.0]
    )
    assert vector == pytest.approx([1.0, -1.0, 0.0, 0.0])


def test_a_vector_of_the_wrong_length_is_refused_rather_than_recorded() -> None:
    # A misaligned vector would draw a contrast strip on the wrong regressors, which
    # is invisible in the figure.
    vector, columns = _contrast_vector_for_design(
        glm_result=_glm([_design()]), contrast_def=[1.0, -1.0]
    )
    assert vector is None
    assert columns  # the columns are still worth recording


def test_an_unparseable_expression_costs_the_strip_and_nothing_else() -> None:
    vector, columns = _contrast_vector_for_design(
        glm_result=_glm([_design()]), contrast_def="nonexistent_condition"
    )
    assert vector is None
    assert columns == ["pain", "nonpain", "trans_x", "constant"]


def test_a_model_without_design_matrices_yields_nothing_to_record() -> None:
    vector, columns = _contrast_vector_for_design(
        glm_result=_glm([]), contrast_def="pain - nonpain"
    )
    assert vector is None and columns == []


# --- the effect and variance maps -----------------------------------------


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.design_matrices_ = [_design()]

    def compute_contrast(self, _argument, output_type: str):
        self.calls.append(output_type)
        return nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))


def _pipeline() -> FmriAnalysisPipeline:
    """A pipeline with no study config behind it.

    Bypasses ``__init__`` deliberately. The real constructor resolves ``deriv_root``
    from the study configuration, which on this project points at an external drive;
    these tests exercise two pure helpers and must not fail when it is unplugged.
    """
    import logging

    pipeline = object.__new__(FmriAnalysisPipeline)
    pipeline.logger = logging.getLogger("test-manifest-helpers")
    return pipeline


def test_the_effect_and_variance_maps_come_off_the_already_fitted_model() -> None:
    # Two compute_contrast calls, no refit.
    model = _FakeModel()
    effect, variance = _pipeline()._contrast_detail_maps(
        glm_result=types.SimpleNamespace(flm=model),
        contrast_def="pain - nonpain",
        plotting_cfg=types.SimpleNamespace(),
    )
    assert model.calls == ["effect_size", "effect_variance"]
    assert effect is not None and variance is not None


def test_the_detail_maps_are_gated_on_intent_not_on_plotting_being_enabled() -> None:
    # They used to live inside `if plotting.enabled`. The report is a separate step
    # and cannot ask for them retroactively.
    model = _FakeModel()
    effect, variance = _pipeline()._contrast_detail_maps(
        glm_result=types.SimpleNamespace(flm=model),
        contrast_def="pain - nonpain",
        plotting_cfg=types.SimpleNamespace(
            include_effect_size=False, include_standard_error=False
        ),
    )
    assert model.calls == []
    assert effect is None and variance is None


def test_a_failure_computing_the_detail_maps_does_not_cost_the_contrast() -> None:
    # By this point the GLM is fitted and the map is on disk.
    class _Exploding(_FakeModel):
        def compute_contrast(self, _argument, output_type: str):
            raise RuntimeError("boom")

    effect, variance = _pipeline()._contrast_detail_maps(
        glm_result=types.SimpleNamespace(flm=_Exploding()),
        contrast_def="pain - nonpain",
        plotting_cfg=types.SimpleNamespace(),
    )
    assert effect is None and variance is None


# --- writing ---------------------------------------------------------------


def test_saving_an_absent_image_is_not_an_error(tmp_path: Path) -> None:
    assert _pipeline()._save_optional(None, tmp_path / "nothing.nii.gz") is None


def test_saving_returns_where_the_image_went(tmp_path: Path) -> None:
    img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    path = _pipeline()._save_optional(img, tmp_path / "effect.nii.gz")
    assert path is not None and path.exists()


def test_an_unwritable_path_costs_the_map_and_not_the_run(tmp_path: Path) -> None:
    img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    assert _pipeline()._save_optional(img, tmp_path / "no" / "such" / "dir.nii.gz") is None


# --- the analysis mask claim ----------------------------------------------


def test_a_discovered_mask_is_never_recorded_as_the_fitted_one(tmp_path: Path) -> None:
    # The two differ: the fitted mask is the intersection across runs, a discovered
    # one is a single run's. The coverage panel makes a claim only the first earns.
    from fmri_pipeline.analysis.report.manifest import read_manifest, write_report_manifest

    stat = tmp_path / "z.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(stat))
    written = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat,
        run_meta={"tr": 2.0},
        mask=stat,
        mask_is_analysis_mask=False,
    )
    assert read_manifest(written).mask_is_analysis_mask is False


def test_the_fitted_mask_is_recorded_as_such(tmp_path: Path) -> None:
    from fmri_pipeline.analysis.report.manifest import read_manifest, write_report_manifest

    stat = tmp_path / "z.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(stat))
    written = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat,
        run_meta={"tr": 2.0},
        mask=stat,
        mask_is_analysis_mask=True,
    )
    assert read_manifest(written).mask_is_analysis_mask is True


def test_the_claim_cannot_be_made_without_a_mask(tmp_path: Path) -> None:
    from fmri_pipeline.analysis.report.manifest import read_manifest, write_report_manifest

    stat = tmp_path / "z.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(stat))
    written = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat,
        run_meta={"tr": 2.0},
        mask=None,
        mask_is_analysis_mask=True,
    )
    assert read_manifest(written).mask_is_analysis_mask is False


def test_a_manifest_written_before_the_field_existed_still_loads(tmp_path: Path) -> None:
    import json

    from fmri_pipeline.analysis.report.manifest import read_manifest

    path = tmp_path / "report_manifest.json"
    path.write_text(
        json.dumps(
            {
                "subject": "sub-01",
                "task": "heat",
                "contrast_name": "c",
                "space": "native",
                "stat_map": str(tmp_path / "z.nii.gz"),
                "effect_map": None,
                "variance_map": None,
                "mask": None,
                "threshold_mode": "z",
                "z_threshold": 2.3,
                "fdr_q": 0.05,
                "cluster_min_voxels": 0,
                "two_sided": True,
                "radiological": False,
                "design_matrices": [],
                "contrast_vector": None,
                "contrast_columns": [],
                "included_runs": [],
                "excluded_runs": [],
                "bold_paths": [],
                "confounds_paths": [],
                "t_r": 2.0,
                "smoothing_fwhm": None,
                "signal_scaling": False,
                "confound_strategy": "auto",
            }
        )
    )
    # Loads, and loads as the conservative answer.
    assert read_manifest(path).mask_is_analysis_mask is False


def test_the_module_exposes_the_helpers_the_pipeline_calls() -> None:
    assert hasattr(pipeline_module, "_contrast_vector_for_design")
    assert hasattr(FmriAnalysisPipeline, "_contrast_detail_maps")
    assert hasattr(FmriAnalysisPipeline, "_save_optional")
