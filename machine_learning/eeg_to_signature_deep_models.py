#!/usr/bin/env python

from __future__ import annotations

import argparse
from typing import Callable, Dict, Optional, Sequence

from machine_learning.models import cnn, cnn_transformer, graph

ModelRunner = Callable[[Optional[Sequence[str]]], None]


MODEL_RUNNERS: Dict[str, ModelRunner] = {
    "cnn": cnn.main,
    "cnn_transformer": cnn_transformer.main,
    "graph": graph.main,
}


def _strip_leading_double_dash(args: Sequence[str]) -> Sequence[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch deep EEG→NPS decoders (cnn, cnn_transformer, graph) from one script."
    )
    parser.add_argument(
        "model",
        choices=sorted(MODEL_RUNNERS),
        help="Which deep model architecture to run.",
    )
    parser.add_argument(
        "model_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the selected deep model script.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    runner = MODEL_RUNNERS[args.model]
    forwarded = list(_strip_leading_double_dash(args.model_args))
    runner(forwarded)


if __name__ == "__main__":
    main()

