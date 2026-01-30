import unittest
from pathlib import Path


class TestFmriReportingHtml(unittest.TestCase):
    def test_report_includes_threshold_and_sections(self):
        from fmri_pipeline.analysis.reporting import ReportSpaceSection, ReportImage, build_fmri_report_html

        tmp_report = Path("report.html")
        # Use non-existent paths; report should fall back to relative paths when embedding fails.
        sections = [
            ReportSpaceSection(space="native", images=(ReportImage(title="A", path=Path("plots/native/a.png")),)),
            ReportSpaceSection(space="mni", images=(ReportImage(title="B", path=Path("plots/mni/b.png")),)),
        ]
        html = build_fmri_report_html(
            report_path=tmp_report,
            subject="sub-0000",
            task="thermalactive",
            contrast_name="pain_vs_nonpain",
            z_threshold=2.3,
            include_unthresholded=True,
            sections=sections,
        )
        self.assertIn("|z| &gt; 2.30", html)
        self.assertIn("Native (subject space)", html)
        self.assertIn("MNI (standard space)", html)
        self.assertIn("plots/native/a.png", html)
        self.assertIn("plots/mni/b.png", html)
