"""FDR displays must retain all Benjamini-Hochberg boundary ties."""

import numpy as np
import pytest

from fmri_pipeline.analysis.report import inference


@pytest.mark.parametrize("two_sided", [False, True])
@pytest.mark.parametrize("peak", [5.0, 40.0])
def test_fdr_height_and_survivor_count_retain_cutoff_ties(two_sided, peak):
    values = np.array([0.0, 1.0, peak, peak, -peak])
    context = inference.threshold_context(
        values, applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=two_sided
    )
    p = inference.p_values(values, two_sided=two_sided)
    cutoff = inference.fdr_p_cutoff(p, q=0.05)
    expected = p <= cutoff
    compared = np.abs(values) if two_sided else values
    np.testing.assert_array_equal(compared > context.fdr, expected)
    assert context.fdr_survivors == np.count_nonzero(expected)


@pytest.mark.parametrize("two_sided", [False, True])
def test_exploratory_fdr_count_includes_boundary_ties(two_sided):
    null = inference.EmpiricalNull(centre=0.5, scale=1.5, n=5)
    values = 0.5 + 1.5 * np.array([0.0, 1.0, 5.0, 5.0, -5.0])
    calibration = inference.empirical_calibration(
        values, null=null, applied_threshold=2.3, fdr_q=0.05, two_sided=two_sided
    )
    assert calibration.fdr_survivors == (3 if two_sided else 2)
