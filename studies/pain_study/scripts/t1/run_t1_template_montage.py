"""Project a standard EEG montage onto individual full-head T1w anatomy."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from studies.pain_study.scripts.t1.t1_anatomical_reference import (
    AnatomicalReferenceParameters,
)
from studies.pain_study.scripts.t1.t1_electrode_localization_inputs import (
    SubjectLocalizationInput,
    discover_eeg_channel_names,
    load_canonical_t1,
    make_template_positions,
)
from studies.pain_study.scripts.t1.t1_template_montage import (
    TemplateMontageOutputPaths,
    TemplateProjectionParameters,
    infer_template_montage_from_array,
    write_template_montage_outputs,
)

DEFAULT_CONFIG = Path(__file__).with_name("config") / "t1_template_montage.yaml"


@dataclass(frozen=True)
class TemplateProjectionConfiguration:
    """Validated inputs for anatomical montage projection."""

    output_root: Path
    montage_name: str
    parameters: TemplateProjectionParameters
    participants: tuple[SubjectLocalizationInput, ...]


class TemplateProjectionBatchError(RuntimeError):
    """Report participant failures after every configured participant was attempted."""

    def __init__(
        self,
        failures: dict[str, Exception],
        outputs: dict[str, TemplateMontageOutputPaths],
    ) -> None:
        self.failures = failures
        self.outputs = outputs
        details = "; ".join(f"{subject_id}: {error}" for subject_id, error in failures.items())
        super().__init__(f"T1 template montage projection failed for {details}")


def _resolve_path(value: object, config_directory: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field {field!r} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_directory / path


def load_template_projection_configuration(
    config_path: str | Path,
) -> TemplateProjectionConfiguration:
    """Load a strict anatomical template-projection configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Template projection configuration does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Template projection configuration must be a YAML mapping.")
    required_keys = {"output_root", "montage", "parameters", "participants"}
    if set(raw) != required_keys:
        raise ValueError(
            "Template projection configuration must contain exactly "
            f"{sorted(required_keys)}, got {sorted(raw)}."
        )
    if not isinstance(raw["montage"], str) or not raw["montage"].strip():
        raise ValueError("Configuration field 'montage' must be a non-empty string.")
    if not isinstance(raw["parameters"], dict):
        raise ValueError("Configuration field 'parameters' must be a mapping.")
    parameter_values = dict(raw["parameters"])
    anatomical_reference_values = parameter_values.pop("anatomical_reference", {})
    if not isinstance(anatomical_reference_values, dict):
        raise ValueError("Configuration field 'anatomical_reference' must be a mapping.")
    try:
        anatomical_reference = AnatomicalReferenceParameters(**anatomical_reference_values)
        parameters = TemplateProjectionParameters(
            **parameter_values,
            anatomical_reference=anatomical_reference,
        )
    except TypeError as error:
        raise ValueError(f"Invalid template projection parameters: {error}") from error

    participant_mapping = raw["participants"]
    if not isinstance(participant_mapping, dict) or not participant_mapping:
        raise ValueError("Configuration field 'participants' must be a non-empty mapping.")
    participant_fields = {"t1w", "eeg_bids_subject_directory"}
    config_directory = path.parent
    participants: list[SubjectLocalizationInput] = []
    for subject_id, values in participant_mapping.items():
        if not isinstance(subject_id, str) or not subject_id.startswith("sub-"):
            raise ValueError(f"Invalid BIDS participant identifier: {subject_id!r}.")
        if not isinstance(values, dict) or set(values) != participant_fields:
            raise ValueError(
                f"Participant {subject_id} must contain exactly {sorted(participant_fields)}."
            )
        participants.append(
            SubjectLocalizationInput(
                subject_id=subject_id,
                t1w_path=_resolve_path(
                    values["t1w"],
                    config_directory,
                    f"participants.{subject_id}.t1w",
                ),
                eeg_bids_subject_directory=_resolve_path(
                    values["eeg_bids_subject_directory"],
                    config_directory,
                    f"participants.{subject_id}.eeg_bids_subject_directory",
                ),
            )
        )
    return TemplateProjectionConfiguration(
        output_root=_resolve_path(raw["output_root"], config_directory, "output_root"),
        montage_name=raw["montage"],
        parameters=parameters,
        participants=tuple(participants),
    )


def run_template_participant(
    participant: SubjectLocalizationInput,
    output_root: str | Path,
    montage_name: str,
    parameters: TemplateProjectionParameters,
) -> TemplateMontageOutputPaths:
    """Infer and export the recorded montage for one participant."""
    channel_names = discover_eeg_channel_names(participant.eeg_bids_subject_directory)
    template_positions = make_template_positions(channel_names, montage_name)
    t1_data, t1_affine = load_canonical_t1(participant.t1w_path)
    result = infer_template_montage_from_array(
        t1_data,
        t1_affine,
        template_positions,
        parameters,
    )
    return write_template_montage_outputs(
        result,
        Path(output_root) / participant.subject_id,
        participant.subject_id,
        participant.t1w_path,
    )


def run_template_projection_configuration(
    configuration: TemplateProjectionConfiguration,
) -> dict[str, TemplateMontageOutputPaths]:
    """Process every participant and aggregate expected input/processing failures."""
    outputs: dict[str, TemplateMontageOutputPaths] = {}
    failures: dict[str, Exception] = {}
    expected_errors = (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError)
    for participant in configuration.participants:
        try:
            outputs[participant.subject_id] = run_template_participant(
                participant,
                configuration.output_root,
                configuration.montage_name,
                configuration.parameters,
            )
        except expected_errors as error:
            failures[participant.subject_id] = error
    if failures:
        raise TemplateProjectionBatchError(failures, outputs)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Strict template-projection YAML configuration (default: {DEFAULT_CONFIG}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configuration = load_template_projection_configuration(arguments.config)
    outputs = run_template_projection_configuration(configuration)
    for subject_id, paths in outputs.items():
        print(f"{subject_id}: {paths.electrodes_tsv}")
        print(f"{subject_id}: {paths.qc_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
