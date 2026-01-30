import unittest


class TestConfoundsSelection(unittest.TestCase):
    def test_auto_includes_acompcor_by_default(self):
        from fmri_pipeline.analysis.confounds_selection import (
            select_fmriprep_confounds_columns,
        )

        cols = [
            # motion
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x_derivative1",
            "trans_y_derivative1",
            "trans_z_derivative1",
            "rot_x_derivative1",
            "rot_y_derivative1",
            "rot_z_derivative1",
            "trans_x_power2",
            "trans_y_power2",
            "trans_z_power2",
            "rot_x_power2",
            "rot_y_power2",
            "rot_z_power2",
            "trans_x_derivative1_power2",
            "trans_y_derivative1_power2",
            "trans_z_derivative1_power2",
            "rot_x_derivative1_power2",
            "rot_y_derivative1_power2",
            "rot_z_derivative1_power2",
            # tissue + fd
            "white_matter",
            "csf",
            "framewise_displacement",
            # outliers
            "motion_outlier00",
            "non_steady_state_outlier00",
            # compcor
            "a_comp_cor_00",
            "a_comp_cor_01",
            "a_comp_cor_02",
            "a_comp_cor_03",
            "a_comp_cor_04",
            "a_comp_cor_05",
        ]

        selected = select_fmriprep_confounds_columns(cols, strategy="auto")
        # auto should include the first 5 aCompCor components by default
        for k in ["a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04"]:
            self.assertIn(k, selected)
        self.assertNotIn("a_comp_cor_05", selected)

        # And it should still keep key nuisance regressors
        self.assertIn("framewise_displacement", selected)
        self.assertIn("white_matter", selected)
        self.assertIn("csf", selected)
        self.assertIn("motion_outlier00", selected)

    def test_motion24_wmcsf_fd_does_not_force_acompcor(self):
        from fmri_pipeline.analysis.confounds_selection import (
            select_fmriprep_confounds_columns,
        )

        cols = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x_derivative1",
            "trans_y_derivative1",
            "trans_z_derivative1",
            "rot_x_derivative1",
            "rot_y_derivative1",
            "rot_z_derivative1",
            "trans_x_power2",
            "trans_y_power2",
            "trans_z_power2",
            "rot_x_power2",
            "rot_y_power2",
            "rot_z_power2",
            "trans_x_derivative1_power2",
            "trans_y_derivative1_power2",
            "trans_z_derivative1_power2",
            "rot_x_derivative1_power2",
            "rot_y_derivative1_power2",
            "rot_z_derivative1_power2",
            "white_matter",
            "csf",
            "framewise_displacement",
            "a_comp_cor_00",
        ]

        selected = select_fmriprep_confounds_columns(cols, strategy="motion24+wmcsf+fd")
        self.assertIn("framewise_displacement", selected)
        self.assertIn("white_matter", selected)
        self.assertIn("csf", selected)
        # Explicit strategy should not add compcor implicitly
        self.assertNotIn("a_comp_cor_00", selected)

