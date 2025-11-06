import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from machine_learning.models import baseline, covariance, svm_rbf

ModelRunner = Callable[[Optional[Sequence[str]]], None]


MODEL_RUNNERS: Dict[str, ModelRunner] = {
    "baseline": baseline.main,
    "svm_rbf": svm_rbf.main,
    "covariance": covariance.main,
}


def _strip_leading_double_dash(args: Sequence[str]) -> Sequence[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch shallow EEG→NPS decoders (baseline, SVM, covariance) from one script."
    )
    parser.add_argument(
        "model",
        choices=sorted(MODEL_RUNNERS),
        help="Which shallow model to run.",
    )
    parser.add_argument(
        "model_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the selected model.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    runner = MODEL_RUNNERS[args.model]
    model_args = _strip_leading_double_dash(args.model_args)
    runner(model_args if model_args else None)


if __name__ == "__main__":
    main()
