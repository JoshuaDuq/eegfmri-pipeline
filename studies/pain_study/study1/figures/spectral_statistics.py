"""Shared participant-level spectral statistics for Study 1 figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eeg_pipeline.utils.config.loader import require_config_value


@dataclass(frozen=True)
class ParticipantBootstrapSpecification:
    """Participant-level percentile-bootstrap settings."""

    iterations: int
    confidence_level: float
    seed: int


def validity_bootstrap_specification(config: Any) -> ParticipantBootstrapSpecification:
    """Load participant-bootstrap settings shared by Study 1 validity figures."""
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    return ParticipantBootstrapSpecification(
        iterations=int(bootstrap["iterations"]),
        confidence_level=float(bootstrap["confidence_level"]),
        seed=int(bootstrap["seed"]),
    )


__all__ = [
    "ParticipantBootstrapSpecification",
    "validity_bootstrap_specification",
]
