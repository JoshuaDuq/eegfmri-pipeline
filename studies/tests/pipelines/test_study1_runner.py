"""Tests for studies.pain_study.study1.runner stage dispatcher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from studies.pain_study.study1.runner import SignaturePredictionRunner
from studies.tests.test_support import DotConfig


def _config() -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": "/tmp/test"},
            "study1": {
                "outputs": {"root_name": "study1"},
                "targets": {"names": ["NPS", "SIIPS1"]},
            },
        }
    )


###################################################################
# All mode dispatches
###################################################################


def test_runner_dispatches_prepare_targets() -> None:
    cfg = _config()
    runner = SignaturePredictionRunner(config=cfg)

    with patch("studies.pain_study.study1.runner.prepare_primary_targets") as mock:
        runner.run(mode="prepare-targets", subjects=["0001"], task="pain")

    mock.assert_called_once_with(
        subjects=["0001"],
        task="pain",
        config=cfg,
        logger=runner.logger,
    )


def test_runner_dispatches_prepare_features() -> None:
    cfg = _config()
    runner = SignaturePredictionRunner(config=cfg)

    with patch("studies.pain_study.study1.runner.prepare_study1_features") as mock:
        runner.run(mode="prepare-features", subjects=["0001"], task="pain")

    mock.assert_called_once_with(
        subjects=["0001"],
        task="pain",
        config=cfg,
        logger=runner.logger,
    )


def test_runner_dispatches_feature_benchmark() -> None:
    cfg = _config()
    runner = SignaturePredictionRunner(config=cfg)

    with patch("studies.pain_study.study1.runner.run_feature_benchmark") as mock:
        runner.run(mode="feature-benchmark", subjects=["0001"], task="pain")

    mock.assert_called_once_with(
        subjects=["0001"],
        task="pain",
        config=cfg,
        logger=runner.logger,
    )


def test_runner_dispatches_deep_regression() -> None:
    cfg = _config()
    runner = SignaturePredictionRunner(config=cfg)

    with patch("studies.pain_study.study1.runner.run_deep_regression") as mock:
        runner.run(mode="deep-regression", subjects=["0001"], task="pain")

    mock.assert_called_once_with(
        subjects=["0001"],
        task="pain",
        config=cfg,
        logger=runner.logger,
    )


def test_runner_dispatches_report() -> None:
    cfg = _config()
    runner = SignaturePredictionRunner(config=cfg)

    with patch("studies.pain_study.study1.runner.write_study1_report") as mock:
        runner.run(mode="report", subjects=[], task="pain")

    mock.assert_called_once_with(
        task="pain",
        config=cfg,
        logger=runner.logger,
    )


def test_runner_rejects_invalid_mode() -> None:
    cfg = _config()
    runner = SignaturePredictionRunner(config=cfg)

    with pytest.raises(ValueError, match="Unsupported Study 1 mode"):
        runner.run(mode="invalid-mode", subjects=[], task="pain")
