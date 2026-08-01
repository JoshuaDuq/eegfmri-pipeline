"""Execution for the preflight CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from typing import Any, List


def run_preflight(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Report the configured dataset's shape, reading BIDS metadata only.

    ``subjects`` is unused: preflight is about the dataset as a whole, and which subjects
    are there is one of the things it reports rather than something it is told.
    """
    from eeg_pipeline.utils.data.preflight import format_preflight_report, run_preflight

    report = run_preflight(config)

    if getattr(args, "output_json", False):
        print(
            json_module.dumps(
                {
                    "observations": [
                        {
                            "key": observation.key,
                            "status": observation.status,
                            "message": observation.message,
                        }
                        for observation in report.observations
                    ]
                },
                indent=2,
            )
        )
        return

    print(format_preflight_report(report))
