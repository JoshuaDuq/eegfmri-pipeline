import unittest


class TestFmriEventsSelection(unittest.TestCase):
    def test_normalize_trial_type_list_none(self):
        from fmri_pipeline.analysis.events_selection import normalize_trial_type_list

        self.assertIsNone(normalize_trial_type_list(None))

    def test_normalize_trial_type_list_list(self):
        from fmri_pipeline.analysis.events_selection import normalize_trial_type_list

        self.assertEqual(
            normalize_trial_type_list(["stimulation", "", None, " vas_rating ", "stimulation"]),
            ["stimulation", "vas_rating"],
        )

    def test_normalize_trial_type_list_csv(self):
        from fmri_pipeline.analysis.events_selection import normalize_trial_type_list

        self.assertEqual(
            normalize_trial_type_list("stimulation,pain_question, vas_rating"),
            ["stimulation", "pain_question", "vas_rating"],
        )

    def test_filter_trial_types_no_allowlist(self):
        from fmri_pipeline.analysis.events_selection import filter_trial_types

        self.assertEqual(
            filter_trial_types(["a", "b"], allowed=None),
            ["a", "b"],
        )

    def test_filter_trial_types_allowlist(self):
        from fmri_pipeline.analysis.events_selection import filter_trial_types

        self.assertEqual(
            filter_trial_types(["fixation_rest", "stimulation", "vas_rating"], allowed=["stimulation", "vas_rating"]),
            ["stimulation", "vas_rating"],
        )


if __name__ == "__main__":
    unittest.main()

