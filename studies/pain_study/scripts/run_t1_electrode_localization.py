"""Run automatic full-head T1w EEG-electrode localization."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from studies.pain_study.scripts.t1_electrode_localization import (
    LocalizationParameters,
    localize_electrodes_from_array,
)
from studies.pain_study.scripts.t1_electrode_localization_inputs import (
    RunConfiguration,
    SubjectLocalizationInput,
    discover_eeg_channel_names,
    load_canonical_t1,
    load_run_configuration,
    make_template_positions,
)
from studies.pain_study.scripts.t1_electrode_localization_outputs import (
    write_coordinate_files,
    write_mne_head_montage,
    write_qc_render,
)

DEFAULT_CONFIG = Path(__file__).with_name("config") / "t1_electrode_localization.yaml"


@dataclass(frozen=True)
class ParticipantOutputPaths:
    """Files produced for one participant."""

    electrodes_tsv: Path
    coordinate_system_json: Path
    diagnostics_json: Path
    qc_png: Path
    mne_head_montage_fif: Path
    scanner_ras_to_head_json: Path


class ParticipantBatchError(RuntimeError):
    """Aggregate expected failures after all configured participants were attempted."""

    def __init__(
        self,
        failures: dict[str, Exception],
        outputs: dict[str, ParticipantOutputPaths],
    ) -> None:
        self.failures = failures
        self.outputs = outputs
        details = "; ".join(f"{subject_id}: {error}" for subject_id, error in failures.items())
        super().__init__(f"T1 electrode localization failed for {details}")


def run_participant(
    participant: SubjectLocalizationInput,
    output_root: str | Path,
    montage_name: str,
    parameters: LocalizationParameters,
) -> ParticipantOutputPaths:
    """Localize and export all recorded scalp EEG channels for one participant."""
    channel_names = discover_eeg_channel_names(participant.eeg_bids_subject_directory)
    template_positions = make_template_positions(channel_names, montage_name)
    t1_data, t1_affine = load_canonical_t1(participant.t1w_path)
    result = localize_electrodes_from_array(
        t1_data,
        t1_affine,
        template_positions,
        parameters,
    )
    output_directory = Path(output_root) / participant.subject_id
    electrodes, coordinate_system, diagnostics = write_coordinate_files(
        result,
        output_directory,
        participant.subject_id,
        participant.t1w_path,
    )
    qc_render = write_qc_render(result, output_directory, participant.subject_id)
    head_montage, scanner_to_head = write_mne_head_montage(
        result,
        output_directory,
        participant.subject_id,
    )
    return ParticipantOutputPaths(
        electrodes_tsv=electrodes,
        coordinate_system_json=coordinate_system,
        diagnostics_json=diagnostics,
        qc_png=qc_render,
        mne_head_montage_fif=head_montage,
        scanner_ras_to_head_json=scanner_to_head,
    )


def run_configuration(
    configuration: RunConfiguration,
) -> dict[str, ParticipantOutputPaths]:
    """Run every explicitly configured participant in declaration order."""
    outputs: dict[str, ParticipantOutputPaths] = {}
    failures: dict[str, Exception] = {}
    expected_errors = (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError)
    for participant in configuration.participants:
        try:
            outputs[participant.subject_id] = run_participant(
                participant,
                configuration.output_root,
                configuration.montage_name,
                configuration.parameters,
            )
        except expected_errors as error:
            failures[participant.subject_id] = error
    if failures:
        raise ParticipantBatchError(failures, outputs)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Strict localization YAML configuration (default: {DEFAULT_CONFIG}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configuration = load_run_configuration(arguments.config)
    outputs = run_configuration(configuration)
    for subject_id, paths in outputs.items():
        print(f"{subject_id}: {paths.electrodes_tsv}")
        print(f"{subject_id}: {paths.qc_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
