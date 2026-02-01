import unittest
from pathlib import Path


class TestTrialSignatureSelectionScoping(unittest.TestCase):
    def test_default_scoping_selects_only_stimulation_plateau(self):
        import pandas as pd  # type: ignore

        from fmri_pipeline.analysis.trial_signatures import TrialSignatureExtractionConfig, _extract_trials_for_run

        events_path = Path("sub-0000_task-thermalactive_run-01_events.tsv")

        rows = []
        # Trial 1 (pain)
        rows.extend(
            [
                {"onset": 0.0, "duration": 1.0, "trial_type": "instruction", "pain_binary_coded": 1, "stim_phase": None},
                {"onset": 1.0, "duration": 1.0, "trial_type": "anticipation", "pain_binary_coded": 1, "stim_phase": None},
                {"onset": 2.0, "duration": 3.0, "trial_type": "stimulation", "pain_binary_coded": 1, "stim_phase": "plateau"},
                {"onset": 5.0, "duration": 1.0, "trial_type": "rating", "pain_binary_coded": 1, "stim_phase": None},
                {"onset": 6.0, "duration": 1.0, "trial_type": "isi", "pain_binary_coded": 1, "stim_phase": None},
            ]
        )
        # Trial 2 (non-pain)
        rows.extend(
            [
                {"onset": 10.0, "duration": 1.0, "trial_type": "instruction", "pain_binary_coded": 0, "stim_phase": None},
                {"onset": 11.0, "duration": 1.0, "trial_type": "anticipation", "pain_binary_coded": 0, "stim_phase": None},
                {"onset": 12.0, "duration": 3.0, "trial_type": "stimulation", "pain_binary_coded": 0, "stim_phase": "plateau"},
                {"onset": 15.0, "duration": 1.0, "trial_type": "rating", "pain_binary_coded": 0, "stim_phase": None},
                {"onset": 16.0, "duration": 1.0, "trial_type": "isi", "pain_binary_coded": 0, "stim_phase": None},
            ]
        )
        events_df = pd.DataFrame(rows)

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
            include_other_events=True,
        ).normalized()

        trials, modeled = _extract_trials_for_run(events_df=events_df, cfg=cfg, events_path=events_path, run_num=1)
        self.assertEqual(len(trials), 2)
        self.assertTrue(all(t.original_trial_type == "stimulation" for t in trials))
        # Safety default: plateau only when stim_phase exists and plateau is present.
        self.assertTrue(all(t.extra.get("stim_phase") == "plateau" for t in trials))
        self.assertGreaterEqual(len(modeled), len(trials))

    def test_explicit_all_disables_stim_phase_default(self):
        import pandas as pd  # type: ignore

        from fmri_pipeline.analysis.trial_signatures import TrialSignatureExtractionConfig, _extract_trials_for_run

        events_path = Path("sub-0000_task-thermalactive_run-01_events.tsv")

        events_df = pd.DataFrame(
            [
                {"onset": 0.0, "duration": 3.0, "trial_type": "stimulation", "pain_binary_coded": 1, "stim_phase": "rise"},
                {"onset": 4.0, "duration": 3.0, "trial_type": "stimulation", "pain_binary_coded": 1, "stim_phase": "plateau"},
            ]
        )

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
            include_other_events=False,
            condition_scope_stim_phases=("all",),
        ).normalized()

        trials, _ = _extract_trials_for_run(events_df=events_df, cfg=cfg, events_path=events_path, run_num=1)
        # If the user explicitly disables stim_phase scoping, both rows remain eligible.
        self.assertEqual(len(trials), 2)


if __name__ == "__main__":
    unittest.main()

