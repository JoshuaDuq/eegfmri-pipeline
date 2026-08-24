"""The report renders from derivatives and never fits a model."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest


def test_rendering_a_report_never_imports_the_model_fitting_modules() -> None:
    """The invariant the whole decoupling exists to establish.

    If importing the report path pulls in the contrast builder or nilearn's GLM,
    then rendering can still reach the model and the separation is nominal.
    """
    script = textwrap.dedent("""
        import sys
        import fmri_pipeline.analysis.report.subject  # noqa: F401

        forbidden = [
            name for name in sys.modules
            if "contrast_builder" in name or name.startswith("nilearn.glm")
        ]
        assert not forbidden, forbidden
        print("clean")
        """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_report_mode_is_registered() -> None:
    """``mode`` is an existing positional with a choices list; ``report`` joins it."""
    from fmri_pipeline.cli.commands.fmri_analysis import setup_fmri_analysis

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_fmri_analysis(sub)
    args = parser.parse_args(["fmri-analysis", "report", "--subject", "0001", "--task", "heat"])
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


def test_report_mode_uses_the_yaml_report_settings(tmp_path: Path, monkeypatch) -> None:
    from eeg_pipeline.utils.config.loader import ConfigDict
    from fmri_pipeline.analysis.report import manifest, subject
    from fmri_pipeline.cli.commands.fmri_analysis import _run_report_mode

    captured = {}
    monkeypatch.setattr(manifest, "discover_manifests", lambda **_kwargs: [object()])

    def _build(**kwargs):
        captured["cfg"] = kwargs["cfg"]
        path = Path(kwargs["out_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("report", encoding="utf-8")
        return path

    monkeypatch.setattr(subject, "build_subject_report", _build)
    config = ConfigDict(
        {
            "paths": {"deriv_root": str(tmp_path)},
            "fmri_report": {
                "enabled": True,
                "html_report": True,
                "include_carpet_qc": False,
            },
        }
    )

    _run_report_mode(
        types.SimpleNamespace(report_dir=None),
        config,
        subjects=["01"],
        task="heat",
    )

    assert captured["cfg"].include_carpet_qc is False


def test_report_mode_rejects_unknown_yaml_settings(tmp_path: Path) -> None:
    from eeg_pipeline.utils.config.loader import ConfigDict
    from fmri_pipeline.cli.commands.fmri_analysis import _run_report_mode

    config = ConfigDict(
        {
            "paths": {"deriv_root": str(tmp_path)},
            "fmri_report": {"enabled": True, "html_report": True, "typo": True},
        }
    )
    with pytest.raises(ValueError, match="Unknown fmri_report key"):
        _run_report_mode(
            types.SimpleNamespace(report_dir=None),
            config,
            subjects=["01"],
            task="heat",
        )
