from __future__ import annotations


def coupling_command():
    from eeg_pipeline.cli.commands import Command
    from studies.pain_study.eeg_coupling.cli.coupling import run_coupling, setup_coupling

    return Command(
        name="coupling",
        setup=setup_coupling,
        run=run_coupling,
    )


def line_comb_command():
    from eeg_pipeline.cli.commands import Command
    from studies.pain_study.cli.line_comb import run_line_comb, setup_line_comb

    return Command(
        name="line-comb",
        setup=setup_line_comb,
        run=run_line_comb,
        requires_subjects=False,
    )


def cardiac_gaps_command():
    from eeg_pipeline.cli.commands import Command
    from studies.pain_study.cli.cardiac_gaps import run_cardiac_gaps, setup_cardiac_gaps

    return Command(
        name="cardiac-gaps",
        setup=setup_cardiac_gaps,
        run=run_cardiac_gaps,
        requires_subjects=False,
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


def source_interpretation_command():
    from eeg_pipeline.cli.commands import Command
    from studies.pain_study.cli.source_interpretation import (
        run_source_interpretation,
        setup_source_interpretation,
    )

    return Command(
        name="source-interpretation",
        setup=setup_source_interpretation,
        run=run_source_interpretation,
    )
