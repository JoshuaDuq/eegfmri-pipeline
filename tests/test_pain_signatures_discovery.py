import tempfile
import unittest
from pathlib import Path


class TestPainSignaturesDiscovery(unittest.TestCase):
    def test_discovers_nps_and_siips1_files(self):
        from fmri_pipeline.analysis.pain_signatures import discover_pain_signature_files

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "NPS").mkdir(parents=True, exist_ok=True)
            (root / "SIIPS1").mkdir(parents=True, exist_ok=True)

            nps = root / "NPS" / "weights_NSF_grouppred_cvpcr.nii.gz"
            siips = root / "SIIPS1" / "nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
            nps.write_bytes(b"fake")
            siips.write_bytes(b"fake")

            files = discover_pain_signature_files(root)
            self.assertEqual(files["NPS"], nps)
            self.assertEqual(files["SIIPS1"], siips)

