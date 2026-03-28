"""Study 1 orchestration entrypoint."""

from __future__ import annotations

import logging
from typing import Any

from studies.pain_study.study1.deep_regression import run_deep_regression
from studies.pain_study.study1.feature_benchmark import run_feature_benchmark
from studies.pain_study.study1.prepare_features import prepare_study1_features
from studies.pain_study.study1.reporting import write_study1_report
from studies.pain_study.study1.targets import prepare_primary_targets


class SignaturePredictionRunner:
    """Stage dispatcher for Study 1."""

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        *,
        mode: str,
        subjects: list[str],
        task: str,
    ) -> None:
        if mode == "prepare-targets":
            prepare_primary_targets(
                subjects=subjects,
                task=task,
                config=self.config,
                logger=self.logger,
            )
            return
        if mode == "prepare-features":
            prepare_study1_features(
                subjects=subjects,
                task=task,
                config=self.config,
                logger=self.logger,
            )
            return
        if mode == "feature-benchmark":
            run_feature_benchmark(
                subjects=subjects,
                task=task,
                config=self.config,
                logger=self.logger,
            )
            return
        if mode == "deep-regression":
            run_deep_regression(
                subjects=subjects,
                task=task,
                config=self.config,
                logger=self.logger,
            )
            return
        if mode == "report":
            write_study1_report(
                task=task,
                config=self.config,
                logger=self.logger,
            )
            return
        raise ValueError(f"Unsupported Study 1 mode: {mode}")
