"""Permutation inference must retain the parametric model's nuisance space."""

import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.cohort import CohortReportConfig
from fmri_pipeline.analysis.second_level import (
    PreparedSecondLevelInput,
    SecondLevelConfig,
    SecondLevelPermutationConfig,
    run_second_level_analysis,
)


@pytest.mark.parametrize("general_contrast", [False, True])
def test_permutation_statistic_matches_full_parametric_model(
    tmp_path: Path, general_contrast: bool
) -> None:
    design = pd.DataFrame({"group_a": [1.0] * 3 + [0.0] * 5, "group_b": [0.0] * 3 + [1.0] * 5})
    responses = np.array([9.0, 10.0, 11.0, 9.0, 9.5, 10.0, 10.5, 11.0])
    contrast = "group_b - group_a"
    if general_contrast:
        design["covariate"] = [-1.5, 0.5, 1.0, -1.0, 0.0, 1.5, -0.5, 0.0]
        responses = responses + 0.8 * design["covariate"].to_numpy()
        contrast = "2 * group_b - group_a + 0.5 * covariate"
    original_design = design.copy()

    paths = []
    for index, response in enumerate(responses):
        data = np.zeros((5, 5, 5))
        data[1:4, 1:4, 1:4] = response
        path = tmp_path / f"subject-{index}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        paths.append(path)

    prepared = PreparedSecondLevelInput(
        image_paths=tuple(paths),
        design_matrix=design,
        manifest=pd.DataFrame({"subject": [str(i) for i in range(len(paths))]}),
        contrast_spec=contrast,
        stat_type="t",
        output_name="comparison",
        output_dir=tmp_path / "output",
        metadata={},
    )
    config = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        write_design_matrix=False,
        permutation=SecondLevelPermutationConfig(
            enabled=True, n_permutations=10, cluster_forming_p=0.001
        ),
        report=CohortReportConfig(enabled=False),
    ).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level.prepare_second_level_input", return_value=prepared
    ):
        result = run_second_level_analysis(
            config=config,
            subjects=[str(i) for i in range(len(paths))],
            task="pain",
            deriv_root=tmp_path,
        )

    parametric = nib.load(result["saved_maps"]["stat"]).get_fdata()
    permutation = nib.load(result["saved_maps"]["permutation_t"]).get_fdata()
    np.testing.assert_allclose(permutation, parametric, atol=1e-6, rtol=1e-6)
    assert prepared.contrast_spec == contrast
    pd.testing.assert_frame_equal(prepared.design_matrix, original_design)
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert metadata["contrast_spec"] == contrast
    assert metadata["design_columns"] == list(original_design.columns)
