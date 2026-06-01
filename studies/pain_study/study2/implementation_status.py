"""Explicit implementation status for Study 2 protocol components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Study2ImplementationComponent:
    name: str
    implemented: bool
    module: str


STUDY2_IMPLEMENTATION_COMPONENTS = (
    Study2ImplementationComponent(
        name="Study 1 confirmatory gates",
        implemented=True,
        module="studies.pain_study.study2.gates",
    ),
    Study2ImplementationComponent(
        name="band contribution scores",
        implemented=True,
        module="studies.pain_study.study2.contributions",
    ),
    Study2ImplementationComponent(
        name="source-model QC",
        implemented=True,
        module="studies.pain_study.study2.source_model_qc",
    ),
    Study2ImplementationComponent(
        name="source-stage subject and cohort QC",
        implemented=True,
        module="studies.pain_study.study2.source_stage",
    ),
    Study2ImplementationComponent(
        name="subject-level source-power association maps",
        implemented=True,
        module="studies.pain_study.study2.association",
    ),
    Study2ImplementationComponent(
        name="sLORETA source-power extraction",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="source-resolution point-spread report",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="true-target directional-consistency map",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="artifact and robustness interpretation gates",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="target-retrained source permutations",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="group-level cluster inference",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="band-unique specificity inference",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="BrainSMASH spatial comparison",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="behavioral convergence analysis",
        implemented=False,
        module="",
    ),
    Study2ImplementationComponent(
        name="bootstrap reporting intervals",
        implemented=False,
        module="",
    ),
)


def unimplemented_confirmatory_components() -> tuple[str, ...]:
    return tuple(
        component.name
        for component in STUDY2_IMPLEMENTATION_COMPONENTS
        if not component.implemented
    )


def assert_confirmatory_pipeline_ready() -> None:
    missing = unimplemented_confirmatory_components()
    if missing:
        raise NotImplementedError(
            "Study 2 confirmatory source inference is not executable until these "
            f"protocol components are implemented: {', '.join(missing)}."
        )


__all__ = [
    "STUDY2_IMPLEMENTATION_COMPONENTS",
    "Study2ImplementationComponent",
    "assert_confirmatory_pipeline_ready",
    "unimplemented_confirmatory_components",
]
