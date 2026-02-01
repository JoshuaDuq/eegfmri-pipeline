import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class _DotGetConfig(dict):
    """Minimal config stub that supports dotted keys via .get()."""

    def get(self, key, default=None):  # type: ignore[override]
        if isinstance(key, str) and "." in key:
            cur = self
            for part in key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur
        return super().get(key, default)


class TestFmriAnalysisCliSignatureScopeWiring(unittest.TestCase):
    def test_signature_scope_args_are_wired_to_trial_cfg(self):
        from fmri_pipeline.cli.commands.fmri_analysis import run_fmri_analysis

        with tempfile.TemporaryDirectory() as td:
            bids_root = Path(td) / "bids_fmri_root"
            bids_root.mkdir(parents=True, exist_ok=True)
            (bids_root / "sub-0000").mkdir(parents=True, exist_ok=True)

            config = _DotGetConfig({"paths": {}})

            args = argparse.Namespace(
                # Common CLI args used by run_fmri_analysis
                bids_fmri_root=str(bids_root),
                deriv_root=None,
                freesurfer_dir=None,
                signature_dir=None,
                task="thermalactive",
                group=None,
                all_subjects=True,
                subjects=None,
                subject=None,
                progress_json=False,
                dry_run=True,
                output_dir=None,
                # Mode + modeling
                mode="lss",
                input_source="fmriprep",
                fmriprep_space=None,
                require_fmriprep=True,
                runs=None,
                contrast_name="pain_vs_nonpain",
                contrast_type="t-test",
                formula=None,
                cond_a_column="pain_binary_coded",
                cond_a_value="1",
                cond_b_column="pain_binary_coded",
                cond_b_value="0",
                confounds_strategy="auto",
                write_design_matrix=None,
                hrf_model="spm",
                drift_model="none",
                high_pass_hz=0.008,
                low_pass_hz=None,
                smoothing_fwhm=None,
                # Trial-wise signature options
                include_other_events=None,
                fixed_effects_weighting=None,
                lss_other_regressors=None,
                max_trials_per_run=None,
                signatures=None,
                signature_roi_atlas=None,
                signature_roi_labels=None,
                signature_rois=None,
                signature_group_column=None,
                signature_group_values=None,
                signature_group_scope=None,
                write_trial_betas=None,
                write_trial_variances=None,
                write_condition_betas=None,
                # Scope args (the subject of this test)
                signature_scope_trial_types=["stimulation", "rating"],
                signature_scope_stim_phases=["plateau"],
            )

            with patch("fmri_pipeline.pipelines.fmri_trial_signatures.FmriTrialSignaturePipeline") as MockPipeline:
                instance = MockPipeline.return_value
                instance.run_batch.return_value = None

                run_fmri_analysis(args, [], config)

                call = instance.run_batch.call_args
                self.assertIsNotNone(call)
                trial_cfg = call.kwargs["trial_cfg"]
                self.assertEqual(trial_cfg.condition_scope_trial_types, ("stimulation", "rating"))
                self.assertEqual(trial_cfg.condition_scope_stim_phases, ("plateau",))


if __name__ == "__main__":
    unittest.main()

