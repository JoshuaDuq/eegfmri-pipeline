import unittest


class TestTrialSignatureExtractionConfig(unittest.TestCase):
    def test_normalized(self):
        from fmri_pipeline.analysis.trial_signatures import TrialSignatureExtractionConfig

        cfg = TrialSignatureExtractionConfig(
            input_source="INVALID",
            fmriprep_space="MNI152NLin2009cAsym",
            require_fmriprep=True,
            runs=None,
            task="thermalactive",
            name="pain_vs_nonpain",
            condition_a_column="pain_binary_coded",
            condition_a_value="1",
            condition_b_column="pain_binary_coded",
            condition_b_value="0",
            hrf_model="spm",
            drift_model="none",
            high_pass_hz=0.008,
            low_pass_hz=0.0,
            smoothing_fwhm=-1.0,
            confounds_strategy="AUTO",
            method="LSS",
            include_other_events=True,
            lss_other_regressors="per-condition",
            fixed_effects_weighting="MEAN",
        )

        n = cfg.normalized()
        self.assertEqual(n.input_source, "fmriprep")
        self.assertIsNone(n.drift_model)
        self.assertIsNone(n.low_pass_hz)
        self.assertIsNone(n.smoothing_fwhm)
        self.assertEqual(n.method, "lss")
        self.assertEqual(n.lss_other_regressors, "per_condition")
        self.assertEqual(n.fixed_effects_weighting, "mean")
        self.assertEqual(n.condition_scope_trial_types, ("stimulation",))

    def test_normalized_all_disables_default_trial_type_scoping(self):
        from fmri_pipeline.analysis.trial_signatures import TrialSignatureExtractionConfig

        cfg = TrialSignatureExtractionConfig(
            input_source="fmriprep",
            fmriprep_space="MNI152NLin2009cAsym",
            require_fmriprep=True,
            runs=None,
            task="thermalactive",
            name="pain_vs_nonpain",
            condition_a_column="pain_binary_coded",
            condition_a_value="1",
            condition_b_column="pain_binary_coded",
            condition_b_value="0",
            hrf_model="spm",
            drift_model=None,
            high_pass_hz=0.008,
            low_pass_hz=None,
            smoothing_fwhm=None,
            confounds_strategy="auto",
            method="lss",
            condition_scope_trial_types=("all",),
        )

        n = cfg.normalized()
        # Explicit "all" should disable scoping and suppress safety defaults.
        self.assertEqual(n.condition_scope_trial_types, ())


if __name__ == "__main__":
    unittest.main()
