from __future__ import annotations


def coupling_command():
    from eeg_pipeline.cli.commands import Command
    from studies.pain_study.study2.cli.coupling import run_coupling, setup_coupling

    return Command(
        name="coupling",
        setup=setup_coupling,
        run=run_coupling,
    )


def signature_prediction_command():
    from eeg_pipeline.cli.commands import Command
    from studies.pain_study.cli.signature_prediction import (
        run_signature_prediction,
        setup_signature_prediction,
    )

    return Command(
        name="signature-prediction",
        setup=setup_signature_prediction,
        run=run_signature_prediction,
    )
