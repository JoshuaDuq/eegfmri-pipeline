"""Study 2 source-space analysis orchestration entrypoint.

A declarative stage table is the single source of truth for both single-stage
runs and the ``all`` sequence. Before each stage the runner verifies that the
stage's declared input artifacts exist and fails fast otherwise, so stages run
incrementally as their upstream data lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from studies.pain_study.study2.stages import (
    artifact_controls_required_inputs,
    behavioral_convergence_required_inputs,
    band_unique_inference_required_inputs,
    band_unique_stage_required_inputs,
    directional_consistency_required_inputs,
    gate_required_inputs,
    haufe_required_inputs,
    inference_required_inputs,
    point_spread_required_inputs,
    robustness_required_inputs,
    run_artifact_controls,
    run_behavioral_convergence,
    run_band_unique_inference,
    run_band_unique_stage,
    run_directional_consistency,
    run_gate,
    run_haufe,
    run_inference,
    run_point_spread,
    run_robustness,
    run_source_model_qc,
    run_source_power,
    run_source_stage,
    run_spatial_correspondence,
    run_target_permutations,
    source_model_qc_required_inputs,
    source_power_required_inputs,
    source_stage_required_inputs,
    spatial_correspondence_required_inputs,
    target_permutations_required_inputs,
)


@dataclass(frozen=True)
class Study2StageContext:
    """Inputs shared by every Study 2 stage."""

    config: Any
    subjects: tuple[str, ...]
    task: str
    logger: logging.Logger


@dataclass(frozen=True)
class Study2Stage:
    """A single orchestrated Study 2 protocol stage."""

    name: str
    run: Callable[[Study2StageContext], None]
    required_inputs: Callable[[Study2StageContext], tuple[Path, ...]]
    run_by_default: bool = True


STUDY2_STAGES: tuple[Study2Stage, ...] = (
    Study2Stage(name="gate", run=run_gate, required_inputs=gate_required_inputs),
    Study2Stage(name="haufe", run=run_haufe, required_inputs=haufe_required_inputs),
    Study2Stage(
        name="source-model-qc",
        run=run_source_model_qc,
        required_inputs=source_model_qc_required_inputs,
    ),
    Study2Stage(
        name="source-power",
        run=run_source_power,
        required_inputs=source_power_required_inputs,
    ),
    Study2Stage(
        name="point-spread",
        run=run_point_spread,
        required_inputs=point_spread_required_inputs,
    ),
    Study2Stage(
        name="source-stage",
        run=run_source_stage,
        required_inputs=source_stage_required_inputs,
    ),
    Study2Stage(
        name="band-unique-stage",
        run=run_band_unique_stage,
        required_inputs=band_unique_stage_required_inputs,
    ),
    Study2Stage(
        name="target-permutations",
        run=run_target_permutations,
        required_inputs=target_permutations_required_inputs,
    ),
    Study2Stage(
        name="inference",
        run=run_inference,
        required_inputs=inference_required_inputs,
    ),
    Study2Stage(
        name="band-unique-inference",
        run=run_band_unique_inference,
        required_inputs=band_unique_inference_required_inputs,
        run_by_default=False,
    ),
    Study2Stage(
        name="directional-consistency",
        run=run_directional_consistency,
        required_inputs=directional_consistency_required_inputs,
    ),
    Study2Stage(
        name="artifact-controls",
        run=run_artifact_controls,
        required_inputs=artifact_controls_required_inputs,
    ),
    Study2Stage(
        name="robustness",
        run=run_robustness,
        required_inputs=robustness_required_inputs,
    ),
    Study2Stage(
        name="spatial-correspondence",
        run=run_spatial_correspondence,
        required_inputs=spatial_correspondence_required_inputs,
    ),
    Study2Stage(
        name="behavioral-convergence",
        run=run_behavioral_convergence,
        required_inputs=behavioral_convergence_required_inputs,
    ),
)


class Study2Runner:
    """Stage dispatcher for Study 2 source-space analyses."""

    def __init__(
        self,
        config: Any,
        *,
        stages: tuple[Study2Stage, ...] = STUDY2_STAGES,
    ):
        self.config = config
        self.stages = stages
        self.logger = logging.getLogger(__name__)

    def run(self, *, mode: str, subjects: list[str], task: str) -> None:
        context = Study2StageContext(
            config=self.config,
            subjects=tuple(subjects),
            task=task,
            logger=self.logger,
        )
        if mode == "all":
            for stage in self.stages:
                if not stage.run_by_default:
                    continue
                self._run_stage(stage, context)
            return
        self._run_stage(self._stage_by_name(mode), context)

    def _stage_by_name(self, mode: str) -> Study2Stage:
        for stage in self.stages:
            if stage.name == mode:
                return stage
        raise ValueError(f"Unsupported Study 2 mode: {mode}")

    def _run_stage(self, stage: Study2Stage, context: Study2StageContext) -> None:
        missing = [path for path in stage.required_inputs(context) if not path.exists()]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Study 2 stage '{stage.name}' is missing required input artifacts: {formatted}."
            )
        self.logger.info("Running Study 2 stage: %s", stage.name)
        stage.run(context)


__all__ = [
    "STUDY2_STAGES",
    "Study2Runner",
    "Study2Stage",
    "Study2StageContext",
]
