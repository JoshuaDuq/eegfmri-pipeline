import tempfile
import unittest
from pathlib import Path


class TestPainSignaturesResampling(unittest.TestCase):
    def test_resamples_image_and_mask_to_signature_grid(self):
        import numpy as np  # type: ignore
        import nibabel as nib  # type: ignore

        from fmri_pipeline.analysis.pain_signatures import compute_pain_signature_expression

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "NPS").mkdir(parents=True, exist_ok=True)
            (root / "SIIPS1").mkdir(parents=True, exist_ok=True)

            w_aff = np.diag([2.0, 2.0, 2.0, 1.0])
            w_img = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), w_aff)
            nib.save(w_img, str(root / "NPS" / "weights_NSF_grouppred_cvpcr.nii.gz"))
            nib.save(w_img, str(root / "SIIPS1" / "nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"))

            x_aff = np.diag([1.0, 1.0, 1.0, 1.0])
            x_img = nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.float32), x_aff)
            mask_img = nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.uint8), x_aff)

            results = compute_pain_signature_expression(
                stat_or_effect_img=x_img,
                signature_root=root,
                mask_img=mask_img,
                signatures=("NPS", "SIIPS1"),
                resampling="image_to_weights",
            )

            self.assertEqual({r.name for r in results}, {"NPS", "SIIPS1"})
            for r in results:
                self.assertEqual(r.n_voxels, 27)
                self.assertAlmostEqual(r.dot, 27.0, places=5)


if __name__ == "__main__":
    unittest.main()

