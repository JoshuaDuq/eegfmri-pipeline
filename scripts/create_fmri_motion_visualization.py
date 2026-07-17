"""Create an interactive translation and rotation time-series visualization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


RUN_PATTERN = re.compile(r"run-(?P<run>\d+)")
MOTION_COLUMNS = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
ROTATION_COLUMNS = ("rot_x", "rot_y", "rot_z")
TEMPLATE_MARKER = "__MOTION_DATA__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmriprep-root", required=True, type=Path)
    parser.add_argument("--subjects-file", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_subjects(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No subjects found in {path}")
    return [label if label.startswith("sub-") else f"sub-{label}" for label in labels]


def load_motion_data(root: Path, subjects: list[str], task: str) -> dict:
    motion_data = {}
    for subject in subjects:
        paths = sorted(
            (root / subject / "func").glob(
                f"{subject}_task-{task}_run-*_desc-confounds_timeseries.tsv"
            )
        )
        if not paths:
            raise FileNotFoundError(f"No task-{task} confounds found for {subject}")

        subject_runs = {}
        for path in paths:
            match = RUN_PATTERN.search(path.name)
            if match is None:
                raise ValueError(f"Cannot identify run from {path}")
            confounds = pd.read_csv(path, sep="\t", usecols=list(MOTION_COLUMNS))
            confounds.loc[:, ROTATION_COLUMNS] = confounds.loc[
                :, ROTATION_COLUMNS
            ].apply(lambda values: values * 180 / 3.141592653589793)
            sampled_confounds = confounds.loc[::2, MOTION_COLUMNS]
            subject_runs[str(int(match.group("run")))] = [
                [index, *[round(value, 4) for value in row]]
                for index, row in zip(
                    sampled_confounds.index,
                    sampled_confounds.itertuples(index=False, name=None),
                    strict=True,
                )
            ]
        motion_data[subject] = subject_runs
    return motion_data


def render_fragment(template_path: Path, output_path: Path, motion_data: dict) -> None:
    template = template_path.read_text()
    if template.count(TEMPLATE_MARKER) != 1:
        raise ValueError(f"Expected one {TEMPLATE_MARKER} marker in {template_path}")
    fragment = template.replace(
        TEMPLATE_MARKER, json.dumps(motion_data, separators=(",", ":"), allow_nan=False)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(fragment)


def main() -> None:
    args = parse_args()
    subjects = load_subjects(args.subjects_file)
    motion_data = load_motion_data(args.fmriprep_root, subjects, args.task)
    template_path = Path(__file__).with_name("fmri_motion_timeseries.fragment.html")
    render_fragment(template_path, args.output, motion_data)


if __name__ == "__main__":
    main()
