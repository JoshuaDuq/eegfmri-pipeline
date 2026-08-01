"""`--dry-run` must not execute.

The flag's help text promises to "show what would be done without executing". A command
that accepts it and runs anyway is worse than one that never offered it, because the
caller reaches for it precisely when they are unsure -- and `preprocessing` overwrites
derivatives. These tests pin the promise to the only evidence that matters: nothing
appears under the derivatives root.
"""

from __future__ import annotations

import argparse
import io
import contextlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _files_under(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


class DryRunWritesNothingTests(unittest.TestCase):
    """Each command that offers --dry-run must leave the derivatives root untouched."""

    def _roots(self, stack: contextlib.ExitStack) -> tuple[Path, Path]:
        bids_root = Path(stack.enter_context(TemporaryDirectory()))
        deriv_root = Path(stack.enter_context(TemporaryDirectory()))
        return bids_root, deriv_root

    def test_preprocessing_dry_run_writes_nothing_under_deriv_root(self) -> None:
        from eeg_pipeline.cli.commands.preprocessing import setup_preprocessing
        from eeg_pipeline.cli.commands.preprocessing_orchestrator import run_preprocessing

        with contextlib.ExitStack() as stack:
            bids_root, deriv_root = self._roots(stack)
            parser = argparse.ArgumentParser()
            setup_preprocessing(parser.add_subparsers(dest="command"))
            args = parser.parse_args(
                [
                    "preprocessing",
                    "ica",
                    "--subject",
                    "0012",
                    "--task",
                    "thermalactive",
                    "--bids-root",
                    str(bids_root),
                    "--deriv-root",
                    str(deriv_root),
                    "--dry-run",
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_preprocessing(args, ["0012"], {})

            self.assertEqual(_files_under(deriv_root), [])

    def test_features_dry_run_writes_nothing_under_deriv_root(self) -> None:
        from eeg_pipeline.cli.commands.features import setup_features
        from eeg_pipeline.cli.commands.features_orchestrator import run_features

        with contextlib.ExitStack() as stack:
            bids_root, deriv_root = self._roots(stack)
            parser = argparse.ArgumentParser()
            setup_features(parser.add_subparsers(dest="command"))
            args = parser.parse_args(
                [
                    "features",
                    "compute",
                    "--subject",
                    "0012",
                    "--task",
                    "thermalactive",
                    "--bids-root",
                    str(bids_root),
                    "--deriv-root",
                    str(deriv_root),
                    "--dry-run",
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_features(args, ["0012"], {})

            self.assertEqual(_files_under(deriv_root), [])

    def test_coupling_dry_run_writes_nothing_under_deriv_root(self) -> None:
        from studies.pain_study.eeg_coupling.cli.coupling import run_coupling, setup_coupling

        with contextlib.ExitStack() as stack:
            bids_root, deriv_root = self._roots(stack)
            parser = argparse.ArgumentParser()
            setup_coupling(parser.add_subparsers(dest="command"))
            args = parser.parse_args(
                [
                    "coupling",
                    "compute",
                    "--subject",
                    "0012",
                    "--task",
                    "thermalactive",
                    "--bids-root",
                    str(bids_root),
                    "--deriv-root",
                    str(deriv_root),
                    "--dry-run",
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_coupling(args, ["0012"], {})

            self.assertEqual(_files_under(deriv_root), [])


class DryRunReportsThePlanTests(unittest.TestCase):
    """A preview is only useful if it says what would have happened."""

    def test_preprocessing_dry_run_names_subjects_mode_and_roots(self) -> None:
        from eeg_pipeline.cli.commands.preprocessing import setup_preprocessing
        from eeg_pipeline.cli.commands.preprocessing_orchestrator import run_preprocessing

        with contextlib.ExitStack() as stack:
            bids_root, deriv_root = self._roots_for(stack)
            parser = argparse.ArgumentParser()
            setup_preprocessing(parser.add_subparsers(dest="command"))
            args = parser.parse_args(
                [
                    "preprocessing",
                    "ica",
                    "--subject",
                    "0012",
                    "--task",
                    "thermalactive",
                    "--bids-root",
                    str(bids_root),
                    "--deriv-root",
                    str(deriv_root),
                    "--dry-run",
                ]
            )

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                run_preprocessing(args, ["0012"], {})
            reported = stream.getvalue()

        self.assertIn("dry run", reported.lower())
        self.assertIn("0012", reported)
        self.assertIn("ica", reported)
        self.assertIn("thermalactive", reported)
        self.assertIn(str(bids_root), reported)
        self.assertIn(str(deriv_root), reported)

    def _roots_for(self, stack: contextlib.ExitStack) -> tuple[Path, Path]:
        return (
            Path(stack.enter_context(TemporaryDirectory())),
            Path(stack.enter_context(TemporaryDirectory())),
        )


class DryRunSurfaceTests(unittest.TestCase):
    """A command cannot start offering --dry-run without someone deciding to honour it.

    `add_output_format_args` hands the flag to whichever parser calls it, which is how
    `preprocessing` came to promise a preview and run the full pipeline instead. This
    pins the surface: widening it is a deliberate act, and the widening commit is the one
    that has to add the handling.
    """

    HONOURS_DRY_RUN = {
        "behavior",
        "component-tfr",
        "coupling",
        "features",
        "fmri",
        "fmri-analysis",
        "ml",
        "preprocessing",
        "signature-prediction",
        "source-interpretation",
    }

    def _commands_offering_dry_run(self) -> set[str]:
        from eeg_pipeline.cli.main import create_argument_parser

        parser = create_argument_parser()
        subparser_actions = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ]
        self.assertTrue(subparser_actions, "the CLI should expose subcommands")

        offering = set()
        for action in subparser_actions:
            for name, subparser in action.choices.items():
                options = {
                    option
                    for sub_action in subparser._actions
                    for option in sub_action.option_strings
                }
                if "--dry-run" in options:
                    offering.add(name)
        return offering

    def test_only_commands_that_honour_dry_run_offer_it(self) -> None:
        offering = self._commands_offering_dry_run()

        unexpected = offering - self.HONOURS_DRY_RUN
        self.assertEqual(
            unexpected,
            set(),
            f"These commands offer --dry-run but are not known to honour it: "
            f"{sorted(unexpected)}. Handle the flag (see report_dry_run in "
            f"eeg_pipeline/cli/common.py) and add them to HONOURS_DRY_RUN, or stop "
            f"calling add_output_format_args on their parser.",
        )


if __name__ == "__main__":
    unittest.main()
