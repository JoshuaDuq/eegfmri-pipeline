"""Summarize fMRIPrep head-motion estimates for a task fMRI cohort."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FD_THRESHOLDS_MM = (0.2, 0.5)
RUN_PATTERN = re.compile(r"run-(?P<run>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmriprep-root", required=True, type=Path)
    parser.add_argument("--subjects-file", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_subjects(path: Path) -> list[str]:
    subjects = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not subjects:
        raise ValueError(f"No subjects found in {path}")
    return [subject if subject.startswith("sub-") else f"sub-{subject}" for subject in subjects]


def discover_confounds(root: Path, subjects: list[str], task: str) -> list[Path]:
    paths: list[Path] = []
    for subject in subjects:
        subject_paths = sorted(
            (root / subject / "func").glob(
                f"{subject}_task-{task}_run-*_desc-confounds_timeseries.tsv"
            )
        )
        if not subject_paths:
            raise FileNotFoundError(f"No task-{task} confounds found for {subject}")
        paths.extend(subject_paths)
    return paths


def summarize_run(path: Path) -> dict[str, float | int | str]:
    confounds = pd.read_csv(path, sep="\t")
    required = {
        "framewise_displacement",
        "std_dvars",
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
    }
    missing = required.difference(confounds.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    subject = path.name.split("_", maxsplit=1)[0]
    run_match = RUN_PATTERN.search(path.name)
    if run_match is None:
        raise ValueError(f"Cannot identify run from {path}")

    fd = confounds["framewise_displacement"].dropna()
    translations = confounds[["trans_x", "trans_y", "trans_z"]]
    rotations_degrees = np.rad2deg(confounds[["rot_x", "rot_y", "rot_z"]])
    summary: dict[str, float | int | str] = {
        "subject": subject,
        "run": int(run_match.group("run")),
        "n_volumes": len(confounds),
        "n_fd_observations": len(fd),
        "mean_fd_mm": fd.mean(),
        "median_fd_mm": fd.median(),
        "p95_fd_mm": fd.quantile(0.95),
        "max_fd_mm": fd.max(),
        "mean_std_dvars": confounds["std_dvars"].mean(),
        "max_translation_range_mm": (translations.max() - translations.min()).max(),
        "max_rotation_range_degrees": (
            rotations_degrees.max() - rotations_degrees.min()
        ).max(),
    }
    for threshold in FD_THRESHOLDS_MM:
        suffix = str(threshold).replace(".", "p")
        count = int(fd.gt(threshold).sum())
        summary[f"n_fd_gt_{suffix}_mm"] = count
        summary[f"percent_fd_gt_{suffix}_mm"] = 100 * count / len(fd)
    return summary


def summarize_participants(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, runs in run_summary.groupby("subject", sort=True):
        fd_count = int(runs["n_fd_observations"].sum())
        row: dict[str, float | int | str] = {
            "subject": subject,
            "n_runs": len(runs),
            "n_volumes": int(runs["n_volumes"].sum()),
            "mean_fd_mm": np.average(
                runs["mean_fd_mm"], weights=runs["n_fd_observations"]
            ),
            "worst_run_mean_fd_mm": runs["mean_fd_mm"].max(),
            "max_fd_mm": runs["max_fd_mm"].max(),
            "max_translation_range_mm": runs["max_translation_range_mm"].max(),
            "max_rotation_range_degrees": runs["max_rotation_range_degrees"].max(),
        }
        for threshold in FD_THRESHOLDS_MM:
            suffix = str(threshold).replace(".", "p")
            count = int(runs[f"n_fd_gt_{suffix}_mm"].sum())
            row[f"n_fd_gt_{suffix}_mm"] = count
            row[f"percent_fd_gt_{suffix}_mm"] = 100 * count / fd_count
        rows.append(row)
    return pd.DataFrame(rows)


def plot_participant_summary(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("mean_fd_mm")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].barh(ordered["subject"], ordered["mean_fd_mm"], color="#3572A5")
    axes[0].axvline(0.2, color="#B22222", linestyle="--", label="0.2 mm")
    axes[0].set(xlabel="Mean framewise displacement (mm)", ylabel="Participant")
    axes[0].legend(frameon=False)

    axes[1].barh(
        ordered["subject"], ordered["percent_fd_gt_0p5_mm"], color="#D98E32"
    )
    axes[1].set(
        xlabel="Volumes with FD > 0.5 mm (%)",
        ylabel="",
        xlim=(0, max(1, ordered["percent_fd_gt_0p5_mm"].max() * 1.1)),
    )
    figure.suptitle("Study 1 thermal-task head motion across six runs")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_run_heatmap(run_summary: pd.DataFrame, output_path: Path) -> None:
    matrix = run_summary.pivot(index="subject", columns="run", values="mean_fd_mm")
    figure, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    axis.set(
        xlabel="Run",
        ylabel="Participant",
        xticks=np.arange(len(matrix.columns)),
        xticklabels=matrix.columns,
        yticks=np.arange(len(matrix.index)),
        yticklabels=matrix.index,
        title="Mean framewise displacement by run (mm)",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean FD (mm)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    subjects = load_subjects(args.subjects_file)
    paths = discover_confounds(args.fmriprep_root, subjects, args.task)
    run_summary = pd.DataFrame(summarize_run(path) for path in paths)
    participant_summary = summarize_participants(run_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(args.output_dir / "run_motion_summary.tsv", sep="\t", index=False)
    participant_summary.to_csv(
        args.output_dir / "participant_motion_summary.tsv", sep="\t", index=False
    )
    plot_participant_summary(
        participant_summary, args.output_dir / "participant_motion_summary.png"
    )
    plot_run_heatmap(run_summary, args.output_dir / "run_mean_fd_heatmap.png")


if __name__ == "__main__":
    main()
