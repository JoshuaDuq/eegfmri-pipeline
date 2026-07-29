"""The report renders from derivatives and never fits a model."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path


def test_rendering_a_report_never_imports_the_model_fitting_modules() -> None:
    """The invariant the whole decoupling exists to establish.

    If importing the report path pulls in the contrast builder or nilearn's GLM,
    then rendering can still reach the model and the separation is nominal.
    """
    script = textwrap.dedent(
        """
        import sys
        import fmri_pipeline.analysis.report.subject  # noqa: F401

        forbidden = [
            name for name in sys.modules
            if "contrast_builder" in name or name.startswith("nilearn.glm")
        ]
        assert not forbidden, forbidden
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_report_mode_is_registered() -> None:
    """``mode`` is an existing positional with a choices list; ``report`` joins it."""
    from fmri_pipeline.cli.commands.fmri_analysis import setup_fmri_analysis

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_fmri_analysis(sub)
    args = parser.parse_args(
        ["fmri-analysis", "report", "--subject", "0001", "--task", "heat"]
    )
    assert args.mode == "report"
    # --subject uses action="append", so this is a list.
    assert args.subject == ["0001"]
    assert args.task == "heat"


def test_a_report_directory_can_be_chosen_separately_from_the_derivatives() -> None:
    """Derivatives frequently live on a read-only or removable volume."""
    from fmri_pipeline.cli.commands.fmri_analysis import setup_fmri_analysis

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_fmri_analysis(sub)
    args = parser.parse_args(
        [
            "fmri-analysis",
            "report",
            "--subject",
            "0001",
            "--task",
            "heat",
            "--report-dir",
            "/tmp/out",
        ]
    )
    assert args.report_dir == "/tmp/out"


def test_the_glm_path_no_longer_calls_into_plotting() -> None:
    """Subject QC was previously regenerated once per contrast because of this call."""
    source = Path("fmri_pipeline/pipelines/fmri_analysis.py").read_text(encoding="utf-8")
    assert "run_fmri_plotting_and_report" not in source
