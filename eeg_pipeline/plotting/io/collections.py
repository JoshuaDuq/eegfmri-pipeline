"""Utilities for collecting and organizing generated plots."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir, deriv_plots_path
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.plotting.io.figures import get_default_config as _get_default_config
from eeg_pipeline.plotting.config import get_plot_config


def _extract_significant_records(
    result_sets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract all significant records from result sets."""
    significant_records: List[Dict[str, Any]] = []
    for result_set in result_sets:
        if isinstance(result_set, dict) and "significant" in result_set:
            significant_records.extend(result_set["significant"])
    return significant_records


def _get_plot_formats(plot_config: Any) -> List[str]:
    """Get plot file formats from configuration."""
    if hasattr(plot_config, "formats"):
        return plot_config.formats
    return ["png", "svg"]


def _sanitize_filename(name: str) -> str:
    """Replace problematic characters in filename."""
    return name.replace(" ", "_").replace("/", "_")


def _build_plot_filename(
    feature: str,
    roi: str,
    target: str,
    correlation: float,
    p_value: float,
    format_extension: str,
) -> str:
    """Build standardized filename for significant plot."""
    base_name = (
        f"{feature}_{roi}_{target}_r{correlation:.2f}_p{p_value:.3f}.{format_extension}"
    )
    return _sanitize_filename(base_name)


def _copy_plot_file(
    source_path: Path,
    destination_path: Path,
    logger: Any,
) -> bool:
    """Copy a single plot file, returning True if successful."""
    try:
        shutil.copy2(source_path, destination_path)
        return True
    except (OSError, shutil.Error) as error:
        logger.warning(f"Failed to copy {source_path}: {error}")
        return False


def _copy_significant_plots(
    significant_records: List[Dict[str, Any]],
    significant_dir: Path,
    plot_formats: List[str],
    logger: Any,
) -> int:
    """Copy all significant plot files to destination directory."""
    copied_count = 0

    for record in significant_records:
        source_path = Path(record.get("path", ""))
        if not source_path.name:
            continue

        feature = record.get("feature", "unknown")
        roi = record.get("roi", "unknown")
        target = record.get("target", "unknown")
        correlation = record.get("r", 0.0)
        p_value = record.get("p", 1.0)

        for format_extension in plot_formats:
            source_file = source_path.with_suffix(f".{format_extension}")
            if not source_file.exists():
                continue

            destination_filename = _build_plot_filename(
                feature, roi, target, correlation, p_value, format_extension
            )
            destination_file = significant_dir / destination_filename

            if _copy_plot_file(source_file, destination_file, logger):
                copied_count += 1

    return copied_count


def _create_summary_dataframe(
    significant_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Create summary DataFrame from significant records."""
    summary_data = []
    for record in significant_records:
        summary_data.append(
            {
                "feature": record.get("feature", ""),
                "roi": record.get("roi", ""),
                "target": record.get("target", ""),
                "r": record.get("r", float("nan")),
                "p": record.get("p", float("nan")),
                "n": record.get("n", 0),
            }
        )
    return pd.DataFrame(summary_data)


def _save_summary_table(
    significant_records: List[Dict[str, Any]],
    subject: str,
    significant_dir: Path,
    logger: Any,
) -> None:
    """Save summary TSV file of significant correlations."""
    if not significant_records:
        return

    summary_dataframe = _create_summary_dataframe(significant_records)
    summary_dataframe = summary_dataframe.sort_values("p", ascending=True)

    summary_path = significant_dir / f"sub-{subject}_significant_correlations.tsv"
    summary_dataframe.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Saved significant correlations summary: {summary_path}")


def collect_significant_plots(
    subject: str,
    deriv_root: Path,
    all_results: List[Dict[str, Any]],
    config=None,
    alpha: float = 0.05,
) -> Path:
    """Collect and organize significant correlation plots.

    Extracts significant records from results, copies plot files to a
    dedicated directory, and creates a summary TSV file.

    Args:
        subject: Subject identifier.
        deriv_root: Root directory for derivatives.
        all_results: List of result dictionaries containing significant records.
        config: Optional configuration dictionary.
        alpha: Significance threshold (currently unused, preserved for API compatibility).

    Returns:
        Path to the directory containing collected significant plots.
    """
    if not subject:
        raise ValueError("Subject identifier cannot be empty")
    if not isinstance(deriv_root, Path):
        raise TypeError("deriv_root must be a Path object")
    if not isinstance(all_results, list):
        raise TypeError("all_results must be a list")

    config = config or _get_default_config()
    logger = get_subject_logger("behavior_analysis", subject)

    plot_config = get_plot_config(config)
    behavioral_config = plot_config.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")

    plots_directory = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    significant_directory = plots_directory / "significant"
    ensure_dir(significant_directory)

    significant_records = _extract_significant_records(all_results)

    if not significant_records:
        logger.info("No significant correlations found to collect")
        return significant_directory

    plot_formats = _get_plot_formats(plot_config)
    copied_count = _copy_significant_plots(
        significant_records, significant_directory, plot_formats, logger
    )

    _save_summary_table(significant_records, subject, significant_directory, logger)

    logger.info(
        f"Collected {len(significant_records)} significant plots "
        f"({copied_count} files) to {significant_directory}"
    )
    return significant_directory


__all__ = [
    "collect_significant_plots",
]
