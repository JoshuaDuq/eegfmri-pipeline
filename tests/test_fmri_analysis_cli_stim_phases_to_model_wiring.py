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


class TestFmriAnalysisCliStimPhasesToModelWiring(unittest.TestCase):
    def test_first_level_stim_phases_to_model_is_wired_to_contrast_cfg(self):
        from fmri_pipeline.cli.commands.fmri_analysis import run_fmri_analysis

        with tempfile.TemporaryDirectory() as td:
            bids_root = Path(td) / "bids_fmri_root"
            (bids_root / "sub-0000").mkdir(parents=True, exist_ok=True)

            config = _DotGetConfig({"paths": {}})

            args = argparse.Namespace(
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
                mode="first-level",
                input_source="fmriprep",
                fmriprep_space=None,
                require_fmriprep=True,
                runs=None,
                contrast_name="pain_vs_nonpain",
                contrast_type="t-test",
                formula=None,
                cond_a_column="trial_type",
                cond_a_value="stimulation",
                cond_b_column="trial_type",
                cond_b_value="fixation_rest",
                confounds_strategy="auto",
                write_design_matrix=None,
                hrf_model="spm",
                drift_model="none",
                high_pass_hz=0.008,
                low_pass_hz=None,
                smoothing_fwhm=None,
                events_to_model=None,
                # NEW: stimulation phase scoping (subject of this test)
                stim_phases_to_model="plateau",
                # Plotting args (must exist on args namespace)
                plots=False,
                plot_html_report=None,
                plot_formats=None,
                plot_space=None,
                plot_threshold_mode=None,
                plot_z_threshold=None,
                plot_fdr_q=None,
                plot_cluster_min_voxels=None,
                plot_vmax_mode=None,
                plot_vmax=None,
                plot_include_unthresholded=None,
                plot_types=None,
                plot_effect_size=None,
                plot_standard_error=None,
                plot_motion_qc=None,
                plot_carpet_qc=None,
                plot_tsnr_qc=None,
                plot_design_qc=None,
                plot_embed_images=None,
                plot_signatures=None,
                output_type="z-score",
                resample_to_freesurfer=False,
            )

            with patch("fmri_pipeline.pipelines.fmri_analysis.FmriAnalysisPipeline") as MockPipeline:
                instance = MockPipeline.return_value
                instance.run_batch.return_value = None

                run_fmri_analysis(args, [], config)

                call = instance.run_batch.call_args
                self.assertIsNotNone(call)
                contrast_cfg = call.kwargs["contrast_cfg"]
                self.assertEqual(contrast_cfg.stim_phases_to_model, ["plateau"])


if __name__ == "__main__":
    unittest.main()

