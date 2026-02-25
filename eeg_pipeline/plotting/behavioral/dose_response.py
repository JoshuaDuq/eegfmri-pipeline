"""
Dose-response visualization (visualize.py-style)
================================================

This module intentionally replaces the prior dose-response implementation with
the plotting approach from the repository-root `visualize` script:

- Mean ± SEM curves by dose level (typically predictor)
- ROI × band panels
- Optional rating (response) vs dose and binary-outcome probability vs dose

Critical differences vs the standalone script
--------------------------------------------
- **ROIs** come from the user's TUI-defined ROIs (CLI `--rois`), i.e.
  `time_frequency_analysis.rois` in config (no hardcoded ROI lists).
- **Bands** come from the user's TUI-defined/selected bands (CLI `--bands`,
  `--frequency-bands`), i.e. `time_frequency_analysis.bands` in config.
- **Column selection** (dose/response/binary outcome) is configurable per-plot via
  `--plot-item-config behavior_dose_response ...` (TUI plot-specific settings).
"""

from __future__ import annotations

import difflib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.analysis.behavior.orchestration import CATEGORY_PREFIX_MAP, _find_trial_table_path
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.paths import deriv_plots_path, deriv_stats_path, ensure_dir
from eeg_pipeline.infra.tsv import read_table, write_tsv
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.utils.analysis.tfr import get_rois
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names
from eeg_pipeline.utils.data.alignment import get_aligned_events
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.manipulation import find_column


_DEFAULT_SEGMENT = ""
_DEFAULT_STAT = "mean"
_MAX_ROI_PAIR_PLOTS = 30

# Stat equivalence groups: stats that can substitute for each other.
# Used to make plotting resilient to baseline-normalized feature names.
_STAT_EQUIVALENTS: dict[str, list[str]] = {
    # Per-trial channel power is often stored as `logratio` (not `mean`),
    # while ROI/global aggregates may use `logratio_mean`. Treat both as
    # compatible with a requested `mean` stat for plotting purposes.
    "mean": ["logratio_mean", "logratio", "percent_mean", "db_mean"],
    "std": ["logratio_std", "percent_std", "db_std"],
    # Explicit transforms should not back-match broad `mean`/`std` buckets.
    "logratio": ["logratio_mean"],
}


@dataclass(frozen=True)
class DoseResponseColumns:
    dose: str
    responses: list[str]
    binary_outcome: Optional[str]


@dataclass(frozen=True)
class ParsedFeature:
    category: str
    column: str
    group: str
    segment: str
    band: str
    scope: str
    identifier: Optional[str]
    stat: str


@contextmanager
def _visualize_style() -> Iterable[None]:
    """Apply the `visualize` script styling locally (temporary rcParams)."""
    original = dict(plt.rcParams)
    try:
        plt.rcParams.update(
            {
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
                "axes.grid": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.frameon": False,
                "figure.dpi": 200,
                "savefig.dpi": 600,
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.linewidth": 0.9,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 4.0,
                "ytick.major.size": 4.0,
                "xtick.major.width": 0.9,
                "ytick.major.width": 0.9,
                "lines.linewidth": 1.6,
                "lines.markersize": 5.0,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "font.family": "DejaVu Sans",
            }
        )
        yield
    finally:
        plt.rcParams.update(original)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _save_fig_scientific(
    fig: plt.Figure,
    out_path: Path,
    *,
    plot_cfg: Any,
    config: Any,
    has_suptitle: bool = False,
) -> None:
    rect = tuple(plot_cfg.get_layout_rect())
    if has_suptitle and len(rect) == 4:
        rect = (rect[0], rect[1], rect[2], min(rect[3], 0.94))

    save_fig(
        fig,
        out_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=max(float(plot_cfg.pad_inches), 0.08),
        tight_layout_rect=rect,
        config=config,
    )


def _mean_sem_by_x(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        raise ValueError("No finite observations for mean±SEM summary.")
    g = df.groupby("x")["y"].agg(["count", "mean", "std"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"])
    g = g.rename(columns={"count": "n"})
    return g.sort_values("x").reset_index(drop=True)


def _spearman_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = np.asarray(x[mask], dtype=float)
    y2 = np.asarray(y[mask], dtype=float)
    n = int(len(x2))
    if n < 3:
        raise ValueError(f"Spearman undefined: need >=3 paired observations, got n={n}.")
    if pd.Series(x2).nunique() < 2 or pd.Series(y2).nunique() < 2:
        raise ValueError("Spearman undefined: x and y must each have >=2 unique values.")
    rho, p = stats.spearmanr(x2, y2)
    if not np.isfinite(rho) or not np.isfinite(p):
        raise ValueError("Spearman correlation failed (non-finite result).")
    return float(rho), float(p), n


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = np.asarray(x[mask], dtype=float)
    y2 = np.asarray(y[mask], dtype=float)
    if len(x2) < 2:
        raise ValueError("Slope undefined: need >=2 observations.")
    x2c = x2 - float(np.mean(x2))
    denom = float(np.sum(x2c**2))
    if denom <= 0.0:
        raise ValueError("Slope undefined: zero variance in x.")
    return float(np.sum(x2c * (y2 - float(np.mean(y2)))) / denom)


def _require_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ValueError(f"Missing required column: {name}. Available: {list(df.columns)}")
    return df[name]


def _suggest_column_names(
    columns: Iterable[str],
    *,
    query: str,
    limit: int = 12,
) -> list[str]:
    """Suggest likely intended column names (for error messages only)."""
    q = str(query).strip()
    if not q:
        return []

    cols = [str(c) for c in columns]
    ql = q.lower()

    substring = [c for c in cols if ql in c.lower()]
    if substring:
        return substring[:limit]

    return difflib.get_close_matches(q, cols, n=limit, cutoff=0.55)


def _sanitize_filename_fragment(value: str, *, fallback: str = "unknown") -> str:
    text = str(value).strip()
    if not text:
        return fallback
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    out = "_".join([p for p in out.split("_") if p])
    return out[:120] if out else fallback


def _present_feature_categories(columns: Iterable[str]) -> list[str]:
    cats: set[str] = set()
    for col in columns:
        s = str(col)
        if "_" not in s:
            continue
        cats.add(s.split("_", 1)[0])
    return sorted(cats)


def _parse_feature_column_for_category(col: str, *, category: str) -> Optional[ParsedFeature]:
    prefix = f"{category}_"
    if not str(col).startswith(prefix):
        return None

    col_str = str(col)
    remainder = col_str[len(prefix) :]
    parsed = NamingSchema.parse(remainder)
    if parsed.get("valid") and str(parsed.get("band") or ""):
        # Shape (A): prefix + full NamingSchema feature (e.g., connectivity_conn_active_alpha_global_...)
        return ParsedFeature(
            category=category,
            column=col_str,
            group=str(parsed.get("group") or category),
            segment=str(parsed.get("segment") or ""),
            band=str(parsed.get("band") or ""),
            scope=str(parsed.get("scope") or ""),
            identifier=str(parsed.get("identifier")) if "identifier" in parsed else None,
            stat=str(parsed.get("stat") or ""),
        )

    parsed_full = NamingSchema.parse(col_str)
    if parsed_full.get("valid"):
        # Shape (B): prefix == group (e.g., power_baseline_alpha_roi_...)
        return ParsedFeature(
            category=category,
            column=col_str,
            group=str(parsed_full.get("group") or category),
            segment=str(parsed_full.get("segment") or ""),
            band=str(parsed_full.get("band") or ""),
            scope=str(parsed_full.get("scope") or ""),
            identifier=str(parsed_full.get("identifier")) if "identifier" in parsed_full else None,
            stat=str(parsed_full.get("stat") or ""),
        )

    if not parsed.get("valid"):
        return None

    # Two shapes exist in trial tables:
    # (A) prefix + full NamingSchema feature (e.g., connectivity_conn_active_alpha_global_...)
    # (B) prefix == group and the remainder omits group (e.g., power_baseline_alpha_roi_...)
    #
    # For (B), stripping the prefix shifts fields: parsed.group becomes the true segment
    # and parsed.segment becomes the true band, with parsed.band empty due to scope token.
    parsed_band = str(parsed.get("band") or "")
    if parsed_band == "":
        segment = str(parsed.get("group") or "")
        band = str(parsed.get("segment") or "")
        group = category
    else:
        group = str(parsed.get("group") or category)
        segment = str(parsed.get("segment") or "")
        band = parsed_band

    return ParsedFeature(
        category=category,
        column=col_str,
        group=group,
        segment=segment,
        band=band,
        scope=str(parsed.get("scope") or ""),
        identifier=str(parsed.get("identifier")) if "identifier" in parsed else None,
        stat=str(parsed.get("stat") or ""),
    )


def _stat_matches_request(*, requested: str, feature_stat: str, scope: str) -> bool:
    req = str(requested).strip()
    fst = str(feature_stat).strip()
    if not req or not fst:
        return False

    candidates = [req] + _STAT_EQUIVALENTS.get(req, [])
    if scope == "global":
        return any(fst == c or fst.endswith(f"_{c}") for c in candidates)
    return fst in candidates


def _collect_category_features(
    trials: pd.DataFrame,
    *,
    category: str,
    segment: str,
    stat: str,
    logger: logging.Logger,
) -> list[ParsedFeature]:
    prefix = f"{category}_"
    candidate_cols = [str(c) for c in trials.columns if str(c).startswith(prefix)]
    if not candidate_cols:
        present = _present_feature_categories(trials.columns)
        raise ValueError(
            f"No columns found for feature category {category!r} (prefix {prefix!r}). "
            f"Present prefixes include: {present}"
        )

    parsed_all = [
        pf
        for pf in (
            _parse_feature_column_for_category(c, category=category) for c in candidate_cols
        )
        if pf is not None
    ]
    if not parsed_all:
        raise ValueError(
            f"Could not parse any columns for category={category!r} with NamingSchema. "
            f"Example columns: {candidate_cols[:8]}"
        )

    seg = str(segment).strip().lower()
    seg_matched = [pf for pf in parsed_all if str(pf.segment).strip().lower() == seg]
    if not seg_matched:
        found_segments = sorted({str(pf.segment) for pf in parsed_all if str(pf.segment).strip()})
        raise ValueError(
            f"No {category!r} features found for segment={segment!r}. "
            f"Available segments for this category: {found_segments}"
        )

    requested = str(stat).strip()
    stat_matched = [
        pf for pf in seg_matched if _stat_matches_request(requested=requested, feature_stat=pf.stat, scope=pf.scope)
    ]
    if stat_matched:
        return stat_matched

    logger.debug(
        "Dose-response: no %s features matched stat=%r for segment=%r; plotting all stats for that segment.",
        category,
        stat,
        segment,
    )
    return seg_matched


def _resolve_dose_response_segment(
    trials: pd.DataFrame,
    *,
    config: Any,
    logger: logging.Logger,
) -> str:
    """Resolve plotting segment with config-first, paradigm-agnostic fallback."""
    segment = str(
        get_config_value(config, "plotting.plots.behavior.dose_response.segment", _DEFAULT_SEGMENT)
    ).strip()
    if segment:
        return segment

    comparison_segment = str(
        get_config_value(config, "plotting.comparisons.comparison_segment", "")
    ).strip()
    if comparison_segment:
        return comparison_segment

    segments: set[str] = set()
    for category in CATEGORY_PREFIX_MAP.keys():
        prefix = f"{category}_"
        for col in trials.columns:
            col_str = str(col)
            if not col_str.startswith(prefix):
                continue
            parsed = _parse_feature_column_for_category(col_str, category=category)
            if parsed and str(parsed.segment).strip():
                segments.add(str(parsed.segment).strip())

    if not segments:
        raise ValueError(
            "dose_response.segment is empty and no feature segments were discoverable. "
            "Set plotting.plots.behavior.dose_response.segment explicitly."
        )

    non_baseline = sorted([s for s in segments if s.strip().lower() != "baseline"], key=str.lower)
    if non_baseline:
        chosen = non_baseline[0]
    else:
        chosen = sorted(segments, key=str.lower)[0]

    logger.info(
        "Dose-response: segment not configured; auto-selected segment=%r from available segments=%s",
        chosen,
        sorted(segments, key=str.lower),
    )
    return chosen


def _resolve_dose_response_columns(
    trials: pd.DataFrame,
    config: Any,
) -> DoseResponseColumns:
    def resolve(
        *,
        override_key: str,
        candidate_key: str,
        defaults: list[str],
        required: bool,
    ) -> Optional[str]:
        override = str(get_config_value(config, override_key, "") or "").strip()
        if override:
            if override not in trials.columns:
                suggestions = _suggest_column_names(trials.columns, query=override)
                suffix = (
                    ""
                    if not suggestions
                    else f" Close matches: {suggestions}."
                )
                raise ValueError(
                    f"Configured column not found in trial table: {override!r} (key={override_key}). "
                    "Ensure this column exists in the subject/task clean events.tsv "
                    "(or regenerate preprocessing / the trial table)."
                    f"{suffix}"
                )
            return override

        candidates = list(get_config_value(config, candidate_key, []) or []) + defaults
        resolved = find_column(trials, candidates)
        if resolved is None:
            if required:
                raise ValueError(
                    f"Could not resolve required column from trial table. "
                    f"Tried candidates={candidates} (key={candidate_key})."
                )
            return None
        return str(resolved)

    dose_col = resolve(
        override_key="plotting.plots.behavior.dose_response.dose_column",
        candidate_key="event_columns.predictor",
        defaults=["predictor"],
        required=True,
    )
    raw_response = get_config_value(config, "plotting.plots.behavior.dose_response.response_column", None)
    if isinstance(raw_response, (list, tuple)):
        response_overrides = [str(x).strip() for x in raw_response if str(x).strip()]
    elif raw_response is None:
        response_overrides = []
    else:
        response_overrides = [str(raw_response).strip()] if str(raw_response).strip() else []

    if response_overrides:
        detected_categories = set(_present_feature_categories(trials.columns))
        unknown = [
            r
            for r in response_overrides
            if r not in trials.columns and r not in detected_categories
        ]
        if unknown:
            raise ValueError(
                "Configured response selection contains unknown entries: "
                f"{unknown}. Available feature categories: {sorted(detected_categories)}. "
                f"Available columns include: {list(trials.columns)}"
            )
        responses = response_overrides
    else:
        responses = []

    # Dose-response plotting itself does not require a binary-outcome column
    # (probability plot is separate).
    return DoseResponseColumns(dose=str(dose_col), responses=responses, binary_outcome=None)


def _resolve_dose_binary_outcome_columns(trials: pd.DataFrame, config: Any) -> tuple[str, str]:
    dose_col = str(
        get_config_value(config, "plotting.plots.behavior.dose_response.dose_column", "") or ""
    ).strip()
    if dose_col:
        if dose_col not in trials.columns:
            raise ValueError(f"Dose column not found in table: {dose_col!r}.")
    else:
        dose_candidates = list(get_config_value(config, "event_columns.predictor", []) or [])
        resolved = find_column(trials, dose_candidates)
        if resolved is None:
            raise ValueError(f"Could not resolve dose column from candidates={dose_candidates}.")
        dose_col = str(resolved)

    binary_outcome_col = str(
        get_config_value(config, "plotting.plots.behavior.dose_response.binary_outcome_column", "") or ""
    ).strip()
    if binary_outcome_col:
        if binary_outcome_col not in trials.columns:
            raise ValueError(f"Binary outcome column not found in table: {binary_outcome_col!r}.")
    else:
        outcome_candidates = list(get_config_value(config, "event_columns.binary_outcome", []) or []) + ["binary_outcome"]
        resolved = find_column(trials, outcome_candidates)
        if resolved is None:
            raise ValueError(
                f"Could not resolve binary outcome column from candidates={outcome_candidates}. "
                "Provide `dose_response_binary_outcome_column` or set `event_columns.binary_outcome`."
            )
        binary_outcome_col = str(resolved)

    return dose_col, binary_outcome_col


def _find_feature_column(
    columns: list[str],
    *,
    group: str,
    segment: str,
    band: str,
    scope: str,
    identifier: str,
    stat: str,
) -> str:
    """Find a feature column matching the given criteria.
    
    Implements smart stat matching: if no exact match is found for the
    requested stat, tries common equivalents. For example, when looking
    for 'mean' in non-baseline segments, also tries 'logratio_mean'
    (baseline-normalized mean values).
    """
    def _match_columns(target_stat: str) -> list[str]:
        matched: list[str] = []
        for col in columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != group:
                continue
            if parsed.get("segment") != segment:
                continue
            if parsed.get("band") != band:
                continue
            if parsed.get("scope") != scope:
                continue
            if scope in {"ch", "roi"} and parsed.get("identifier") != identifier:
                continue
            if parsed.get("stat") != target_stat:
                continue
            matched.append(str(col))
        return matched
    
    # Try exact stat first
    matches = _match_columns(stat)
    
    # If no exact match, try equivalent stats
    if not matches:
        for alt_stat in _STAT_EQUIVALENTS.get(stat, []):
            matches = _match_columns(alt_stat)
            if matches:
                break

    if not matches:
        raise ValueError(
            "No feature column matched "
            f"group={group!r}, segment={segment!r}, band={band!r}, scope={scope!r}, "
            f"identifier={identifier!r}, stat={stat!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous feature column match (expected 1): "
            f"group={group!r}, segment={segment!r}, band={band!r}, scope={scope!r}, "
            f"identifier={identifier!r}, stat={stat!r}. Matches: {matches}"
        )
    return matches[0]


def _roi_power_series(
    trials: pd.DataFrame,
    *,
    roi_name: str,
    roi_patterns: Optional[object] = None,
    band: str,
    segment: str,
    stat: str,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.Series, list[str]]:
    cols = [str(c) for c in trials.columns]

    try:
        roi_col = _find_feature_column(
            cols,
            group="power",
            segment=segment,
            band=band,
            scope="roi",
            identifier=roi_name,
            stat=stat,
        )
        return pd.to_numeric(trials[roi_col], errors="coerce"), [roi_col]
    except ValueError:
        pass

    if roi_patterns is None:
        raise ValueError(
            f"Missing ROI-level power feature and no ROI definition provided (roi={roi_name!r}, band={band!r})."
        )

    patterns: list[str] = []
    if isinstance(roi_patterns, str):
        patterns = [p.strip() for p in roi_patterns.split(",") if p.strip()]
    elif isinstance(roi_patterns, (list, tuple, set)):
        patterns = [str(p).strip() for p in roi_patterns if str(p).strip()]
    else:
        patterns = [str(roi_patterns).strip()] if str(roi_patterns).strip() else []

    if not patterns:
        raise ValueError(
            f"Missing ROI-level power feature and ROI definition is empty (roi={roi_name!r}, band={band!r})."
        )

    # Fallback: approximate ROI power as the mean across matching channel features.
    # This supports cases where ROI aggregation was not enabled during extraction,
    # but channel-level power features are present in the trial table.
    candidate_stats = [stat] + _STAT_EQUIVALENTS.get(stat, [])
    ch_cols_by_stat: dict[str, dict[str, str]] = {}
    for col in cols:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "power":
            continue
        if parsed.get("segment") != segment:
            continue
        if parsed.get("band") != band:
            continue
        if parsed.get("scope") != "ch":
            continue
        identifier = parsed.get("identifier")
        if not identifier:
            continue
        st = parsed.get("stat") or ""
        ch_cols_by_stat.setdefault(str(identifier), {})[str(st)] = str(col)

    def matches_any_pattern(channel: str) -> bool:
        channel_str = str(channel)
        for pat in patterns:
            if channel_str.lower() == pat.lower():
                return True
            try:
                if re.match(pat, channel_str, flags=re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False

    used_cols: list[str] = []
    for ch, by_stat in ch_cols_by_stat.items():
        if not matches_any_pattern(ch):
            continue
        selected = None
        for st in candidate_stats:
            if st in by_stat:
                selected = by_stat[st]
                break
        if selected is None and by_stat:
            # Last resort: pick any available stat for that channel.
            selected = next(iter(by_stat.values()))
        if selected:
            used_cols.append(selected)

    # Preserve deterministic order for reproducibility.
    used_cols = sorted(set(used_cols))
    if not used_cols:
        raise ValueError(
            "No channel-level power features matched ROI definition "
            f"(roi={roi_name!r}, band={band!r}, patterns={patterns!r})."
        )

    if logger is not None:
        logger.debug(
            "Dose-response: computed ROI=%s band=%s from %d channel feature(s).",
            roi_name,
            band,
            len(used_cols),
        )

    y = trials[used_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return y, used_cols


def _save_summary_table(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    subject: str,
    roi: str,
    band: str,
    dose_col: str,
) -> Path:
    stats_dir = out_dir / "stats"
    ensure_dir(stats_dir)
    out = stats_dir / f"sub-{subject}_roi-{roi}_band-{band}_dose_response_summary.tsv"
    write_tsv(df, out)
    return out


def _save_category_summary_table(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    subject: str,
    category: str,
    scope: str,
    identifier: str,
    band: str,
    metric: str,
    dose_col: str,
) -> Path:
    stats_dir = out_dir / "stats"
    ensure_dir(stats_dir)
    subject_f = _sanitize_filename_fragment(subject)
    category_f = _sanitize_filename_fragment(category)
    scope_f = _sanitize_filename_fragment(scope)
    identifier_f = _sanitize_filename_fragment(identifier)
    band_f = _sanitize_filename_fragment(band, fallback="broadband")
    metric_f = _sanitize_filename_fragment(metric, fallback="value")
    out = stats_dir / (
        f"sub-{subject_f}_{category_f}_scope-{scope_f}_id-{identifier_f}_band-{band_f}_{metric_f}_dose_response_summary.tsv"
    )
    df_out = df.rename(columns={"x": dose_col}).copy()
    write_tsv(df_out, out)
    return out


def _sorted_bands(bands: Iterable[str], *, preferred_order: list[str]) -> list[str]:
    seen = set()
    items = [str(b) for b in bands if str(b)]
    pref = [b for b in preferred_order if b in items]
    rest = sorted([b for b in items if b not in set(preferred_order)])
    out: list[str] = []
    for b in pref + rest:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _plot_category_feature_panels(
    trials: pd.DataFrame,
    *,
    subject: str,
    category: str,
    dose_col: str,
    scope: str,
    identifier: str,
    band_metric_cols: dict[str, dict[str, list[str]]],
    out_path: Path,
    summary_dir: Path,
    config: Any,
    logger: logging.Logger,
) -> dict[str, Path]:
    bands = list(band_metric_cols.keys())
    if not bands:
        return {}

    saved: dict[str, Path] = {}
    with _visualize_style():
        plot_cfg = get_plot_config(config)
        fig_size = plot_cfg.get_figure_size("small", plot_type="behavioral")
        for band in bands:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
            _style_axes(ax)

            metrics = band_metric_cols.get(band, {})
            if not metrics:
                continue

            for metric, cols in metrics.items():
                y = trials[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                d = pd.DataFrame({dose_col: pd.to_numeric(trials[dose_col], errors="coerce"), "_y": y}).dropna()
                if d.empty:
                    continue

                summ = _mean_sem_by_x(d[dose_col].to_numpy(float), d["_y"].to_numpy(float))
                label = metric if len(metrics) > 1 else f"sub-{subject}"
                ax.errorbar(
                    summ["x"],
                    summ["mean"],
                    yerr=summ["sem"],
                    fmt="o",
                    linestyle="-",
                    capsize=3.0,
                    color="#333333",
                    ecolor="#333333",
                    markerfacecolor="white",
                    markeredgecolor="#333333",
                    markeredgewidth=0.9,
                    label=label,
                )

                summary_path = _save_category_summary_table(
                    summ,
                    summary_dir,
                    subject=subject,
                    category=category,
                    scope=scope,
                    identifier=identifier,
                    band=band,
                    metric=metric,
                    dose_col=dose_col,
                )
                saved[f"summary_{category}_scope-{scope}_id-{identifier}_band-{band}_{metric}"] = summary_path

            ax.set_xlabel(dose_col)
            ax.set_ylabel(f"{category} (mean ± SEM)")
            ax.set_title(f"{band} — {identifier}")

            # Keep dose levels visible as discrete ticks (visualize.py style).
            try:
                unique_x = np.sort(pd.unique(pd.to_numeric(trials[dose_col], errors="coerce").dropna()))
                if unique_x.size:
                    ax.set_xticks(unique_x)
                    ax.set_xticklabels([f"{v:.1f}" for v in unique_x])
            except Exception as exc:
                logger.debug("Dose tick formatting failed: %s", exc)

            ax.legend(frameon=False, loc="best")

            fig.suptitle(
                f"Dose-response: {category} vs {dose_col} (sub-{subject})",
                y=1.02,
            )
            band_f = _sanitize_filename_fragment(band, fallback="broadband")
            base = out_path.name
            stem = base[:-len("_dose_response")] if base.endswith("_dose_response") else base
            band_out_path = out_path.parent / f"{stem}_band-{band_f}_dose_response"
            _save_fig_scientific(fig, band_out_path, plot_cfg=plot_cfg, config=config, has_suptitle=True)
            saved[f"{category}_scope-{scope}_id-{identifier}_band-{band}_dose_response"] = band_out_path

    return saved


def _plot_category_features_vs_dose_single_subject(
    trials: pd.DataFrame,
    *,
    subject: str,
    category: str,
    dose_col: str,
    segment: str,
    stat: str,
    roi_defs: dict[str, Any],
    preferred_band_order: list[str],
    allowed_scopes: Optional[set[str]],
    allowed_rois: Optional[set[str]],
    out_dir: Path,
    config: Any,
    logger: logging.Logger,
) -> dict[str, Path]:
    feats = _collect_category_features(trials, category=category, segment=segment, stat=stat, logger=logger)

    cat_dir = out_dir / category
    ensure_dir(cat_dir)

    roi_names = [str(k) for k in roi_defs.keys()]

    roi_feats = [pf for pf in feats if pf.scope == "roi" and pf.identifier]
    global_feats = [pf for pf in feats if pf.scope == "global"]
    ch_feats = [pf for pf in feats if pf.scope == "ch" and pf.identifier]
    chpair_feats = [pf for pf in feats if pf.scope == "chpair" and pf.identifier]

    saved: dict[str, Path] = {}
    include_global = not allowed_scopes or "global" in allowed_scopes
    include_roi = not allowed_scopes or "roi" in allowed_scopes
    include_ch = not allowed_scopes or "ch" in allowed_scopes
    include_chpair = not allowed_scopes or "chpair" in allowed_scopes or "roipair" in allowed_scopes

    def add_panels(scope: str, identifier: str, features: list[ParsedFeature]) -> None:
        bands_found = _sorted_bands({pf.band for pf in features if pf.band}, preferred_order=preferred_band_order)
        band_metric_cols: dict[str, dict[str, list[str]]] = {b: {} for b in bands_found}
        for pf in features:
            band_metric_cols.setdefault(pf.band, {}).setdefault(pf.stat, []).append(pf.column)
        # Filter empty bands
        band_metric_cols = {b: m for b, m in band_metric_cols.items() if m}
        if not band_metric_cols:
            return

        out_path = cat_dir / f"sub-{_sanitize_filename_fragment(subject)}_{_sanitize_filename_fragment(category)}_{scope}-{_sanitize_filename_fragment(identifier)}_dose_response"
        saved.update(
            _plot_category_feature_panels(
                trials,
                subject=subject,
                category=category,
                dose_col=dose_col,
                scope=scope,
                identifier=identifier,
                band_metric_cols=band_metric_cols,
                out_path=out_path,
                summary_dir=cat_dir,
                config=config,
                logger=logger,
            )
        )

    # ROI-scoped features: one figure per configured ROI (if present), else all.
    if include_roi and roi_feats:
        by_roi: dict[str, list[ParsedFeature]] = {}
        for pf in roi_feats:
            by_roi.setdefault(str(pf.identifier), []).append(pf)

        roi_order = [r for r in roi_names if r in by_roi] or sorted(by_roi.keys())
        if allowed_rois:
            roi_order = [r for r in roi_order if r in allowed_rois]
        for roi in roi_order:
            add_panels("roi", roi, by_roi[roi])

    # Channel-scoped features: aggregate channels into ROIs if possible (keeps output manageable).
    if include_ch and ch_feats and roi_defs and (not roi_feats or "ch" in (allowed_scopes or set())):
        channel_to_rois: dict[str, list[str]] = {}
        for roi, chans in roi_defs.items():
            for ch in chans or []:
                channel_to_rois.setdefault(str(ch), []).append(str(roi))

        by_roi: dict[str, list[ParsedFeature]] = {}
        for pf in ch_feats:
            for roi in channel_to_rois.get(str(pf.identifier), []):
                by_roi.setdefault(roi, []).append(pf)

        roi_order = [r for r in roi_names if r in by_roi] or sorted(by_roi.keys())
        if allowed_rois:
            roi_order = [r for r in roi_order if r in allowed_rois]
        for roi in roi_order:
            # Within each ROI, average across all channel-level columns per (band, metric).
            roi_features = by_roi[roi]
            grouped: dict[tuple[str, str], list[str]] = {}
            for pf in roi_features:
                grouped.setdefault((pf.band, pf.stat), []).append(pf.column)
            bands_found = _sorted_bands({b for (b, _) in grouped.keys() if b}, preferred_order=preferred_band_order)
            band_metric_cols: dict[str, dict[str, list[str]]] = {b: {} for b in bands_found}
            for (band, metric), cols in grouped.items():
                band_metric_cols.setdefault(band, {}).setdefault(metric, []).extend(cols)
            band_metric_cols = {b: m for b, m in band_metric_cols.items() if m}
            if not band_metric_cols:
                continue

            out_path = cat_dir / f"sub-{_sanitize_filename_fragment(subject)}_{_sanitize_filename_fragment(category)}_roi-{_sanitize_filename_fragment(roi)}_dose_response"
            saved.update(
                _plot_category_feature_panels(
                    trials,
                    subject=subject,
                    category=category,
                    dose_col=dose_col,
                    scope="roi",
                    identifier=roi,
                    band_metric_cols=band_metric_cols,
                    out_path=out_path,
                    summary_dir=cat_dir,
                    config=config,
                    logger=logger,
                )
            )

    # ROI-pair aggregation for chpair features (optional, capped).
    if include_chpair and chpair_feats and roi_defs:
        channel_to_rois: dict[str, list[str]] = {}
        for roi, chans in roi_defs.items():
            for ch in chans or []:
                channel_to_rois.setdefault(str(ch), []).append(str(roi))

        def parse_pair(pair: str) -> Optional[tuple[str, str]]:
            text = str(pair)
            if "-" not in text:
                return None
            left, right = text.split("-", 1)
            left = left.strip()
            right = right.strip()
            if not left or not right:
                return None
            return left, right

        by_pair: dict[tuple[str, str], list[ParsedFeature]] = {}
        for pf in chpair_feats:
            pair = parse_pair(str(pf.identifier))
            if pair is None:
                continue
            ch1, ch2 = pair
            rois1 = channel_to_rois.get(ch1, [])
            rois2 = channel_to_rois.get(ch2, [])
            for r1 in rois1:
                for r2 in rois2:
                    a, b = sorted([str(r1), str(r2)])
                    by_pair.setdefault((a, b), []).append(pf)

        pairs = sorted(by_pair.keys())
        if allowed_rois:
            pairs = [(a, b) for (a, b) in pairs if a in allowed_rois and b in allowed_rois]
        if len(pairs) > _MAX_ROI_PAIR_PLOTS:
            logger.warning(
                "Dose-response: capping ROI-pair plots for category=%s at %d (requested %d).",
                category,
                _MAX_ROI_PAIR_PLOTS,
                len(pairs),
            )
            pairs = pairs[:_MAX_ROI_PAIR_PLOTS]

        for a, b in pairs:
            add_panels("roipair", f"{a}__{b}", by_pair[(a, b)])

    # Global features: one figure for all metrics by band.
    if include_global and global_feats:
        add_panels("global", "global", global_feats)

    return saved

def _plot_roi_bands_vs_dose_single_subject(
    df: pd.DataFrame,
    *,
    subject: str,
    roi_name: str,
    bands: list[str],
    dose_col: str,
    out_dir: Path,
    config: Any,
    logger: logging.Logger,
) -> dict[str, Path]:
    if not bands:
        raise ValueError("No bands provided for dose-response plotting.")
    if dose_col not in df.columns:
        raise ValueError(f"Dose column {dose_col!r} missing from table.")

    saved: dict[str, Path] = {}

    with _visualize_style():
        plot_cfg = get_plot_config(config)
        fig_size = plot_cfg.get_figure_size("small", plot_type="behavioral")
        for band in bands:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
            _style_axes(ax)

            y_col = f"roi_power_{band}"
            if y_col not in df.columns:
                logger.warning(
                    "Skipping ROI power plot (roi=%s, band=%s): missing column %s.",
                    roi_name,
                    band,
                    y_col,
                )
                plt.close(fig)
                continue

            band_df = df[[dose_col, y_col]].copy()
            band_df = band_df.dropna()
            if band_df.empty:
                logger.warning(
                    "Skipping ROI power plot (roi=%s, band=%s): no valid rows after dropping NaNs.",
                    roi_name,
                    band,
                )
                plt.close(fig)
                continue

            summ = _mean_sem_by_x(
                band_df[dose_col].to_numpy(dtype=float),
                band_df[y_col].to_numpy(dtype=float),
            )
            ax.errorbar(
                summ["x"],
                summ["mean"],
                yerr=summ["sem"],
                fmt="o",
                linestyle="-",
                capsize=3.0,
                color="#333333",
                ecolor="#333333",
                markerfacecolor="white",
                markeredgecolor="#333333",
                markeredgewidth=0.9,
                label=f"sub-{subject}",
            )
            ax.axhline(0.0, linewidth=0.9, alpha=0.8, linestyle="--")

            ax.set_xlabel(dose_col)
            ax.set_ylabel("ROI power (mean ± SEM)")
            ax.set_title(f"{band} — {roi_name}")

            unique_x = summ["x"].to_numpy(dtype=float)
            ax.set_xticks(unique_x)
            ax.set_xticklabels([f"{v:.1f}" for v in unique_x])

            # Effect sizes computed on predictor means (avoid trials-as-independent)
            try:
                rho, p, n = _spearman_xy(summ["x"].to_numpy(float), summ["mean"].to_numpy(float))
                slope = _linear_slope(summ["x"].to_numpy(float), summ["mean"].to_numpy(float))
                ax.text(
                    0.02,
                    0.02,
                    f"Spearman ρ={rho:.2f}, p={p:.3g}\nSlope={slope:.3f}/unit (n={n})",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                )
            except Exception as exc:
                logger.debug(
                    "Dose-response effect size unavailable (roi=%s band=%s): %s",
                    roi_name,
                    band,
                    exc,
                )

            ax.legend(frameon=False, loc="best")

            summary_path = _save_summary_table(summ.rename(columns={"x": dose_col}), out_dir, subject=subject, roi=roi_name, band=band, dose_col=dose_col)
            saved[f"summary_roi-{roi_name}_band-{band}"] = summary_path
            fig.suptitle(f"Dose-response: ROI power vs {dose_col} (sub-{subject})", y=1.02)
            out_path = out_dir / f"sub-{subject}_roi-{roi_name}_band-{band}_dose_response"
            _save_fig_scientific(fig, out_path, plot_cfg=plot_cfg, config=config, has_suptitle=True)
            saved[f"roi-{roi_name}_band-{band}_dose_response"] = out_path

    return saved


def _plot_xy_mean_sem(
    df: pd.DataFrame,
    *,
    subject: str,
    x_col: str,
    y_col: str,
    out_path: Path,
    config: Any,
) -> None:
    d = df[[x_col, y_col]].copy().dropna()
    if d.empty:
        raise ValueError(f"No valid rows for x={x_col!r}, y={y_col!r}.")

    with _visualize_style():
        plot_cfg = get_plot_config(config)
        fig = plt.figure(figsize=plot_cfg.get_figure_size("medium", plot_type="behavioral"))
        ax = fig.add_subplot(111)
        _style_axes(ax)

        summ = _mean_sem_by_x(d[x_col].to_numpy(float), d[y_col].to_numpy(float))
        ax.errorbar(
            summ["x"],
            summ["mean"],
            yerr=summ["sem"],
            fmt="o",
            linestyle="-",
            capsize=3.0,
            color="#333333",
            ecolor="#333333",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.9,
            label=f"sub-{subject}",
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col} (sub-{subject})")
        ax.legend(frameon=False, loc="best")

        _save_fig_scientific(fig, out_path, plot_cfg=plot_cfg, config=config, has_suptitle=False)


def _plot_binary_outcome_probability_vs_predictor(
    df: pd.DataFrame,
    *,
    subject: str,
    dose_col: str,
    binary_outcome_col: str,
    out_path: Path,
    config: Any,
) -> None:
    d = df[[dose_col, binary_outcome_col]].copy().dropna()
    if d.empty:
        raise ValueError(f"No valid rows for binary outcome probability with dose={dose_col!r}.")

    binary_outcome = (pd.to_numeric(d[binary_outcome_col], errors="coerce") > 0).astype(float)
    d = d.assign(_binary_outcome=binary_outcome).dropna()
    if d.empty:
        raise ValueError(f"No valid numeric rows for binary outcome column {binary_outcome_col!r}.")

    with _visualize_style():
        plot_cfg = get_plot_config(config)
        fig = plt.figure(figsize=plot_cfg.get_figure_size("medium", plot_type="behavioral"))
        ax = fig.add_subplot(111)
        _style_axes(ax)

        g = d.groupby(dose_col)["_binary_outcome"].agg(["count", "mean"]).reset_index().rename(columns={"count": "n"})
        g["sem_binomial"] = np.sqrt(g["mean"] * (1.0 - g["mean"]) / g["n"].clip(lower=1))

        ax.errorbar(
            g[dose_col].to_numpy(float),
            g["mean"].to_numpy(float),
            yerr=g["sem_binomial"].to_numpy(float),
            fmt="o",
            linestyle="-",
            capsize=3.0,
            color="#333333",
            ecolor="#333333",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.9,
            label=f"sub-{subject}",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(dose_col)
        ax.set_ylabel(f"P({binary_outcome_col}=1)")
        ax.set_title(f"Binary outcome probability vs {dose_col} (sub-{subject})")
        ax.legend(frameon=False, loc="best")

        _save_fig_scientific(fig, out_path, plot_cfg=plot_cfg, config=config, has_suptitle=False)


def _resolve_trial_table_feature_files(config: Any) -> Optional[list[str]]:
    feature_files = get_config_value(config, "behavior_analysis.feature_files", None)
    if isinstance(feature_files, str):
        return [feature_files]
    if isinstance(feature_files, (list, tuple)):
        items = [str(x).strip() for x in feature_files if str(x).strip()]
        return items or None

    feature_categories = get_config_value(config, "behavior_analysis.feature_categories", None)
    if isinstance(feature_categories, str):
        return [feature_categories]
    if isinstance(feature_categories, (list, tuple)):
        items = [str(x).strip() for x in feature_categories if str(x).strip()]
        return items or None
    return None


def _resolve_optional_str_list(config: Any, key: str) -> Optional[list[str]]:
    raw = get_config_value(config, key, None)
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else None
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return items or None
    return None


def _load_trial_table(deriv_root: Path, subject: str, config: Any) -> tuple[pd.DataFrame, Path]:
    stats_dir = deriv_stats_path(deriv_root, subject)
    feature_files = _resolve_trial_table_feature_files(config)
    trial_path = _find_trial_table_path(stats_dir, feature_files=feature_files)
    if trial_path is None:
        suffix_hint = (
            f" for feature_files={feature_files!r}" if feature_files is not None else ""
        )
        raise FileNotFoundError(
            f"Missing trial table under {stats_dir}{suffix_hint}. "
            "Run the behavior computation 'trial_table' first."
        )
    df = read_table(trial_path)
    if df.empty:
        raise ValueError(f"Trial table is empty: {trial_path}")
    return df, trial_path


def visualize_dose_response(
    subject: str,
    deriv_root: Path,
    task: str,
    config: Any,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Generate dose-response plots for a single subject."""
    if not isinstance(subject, str) or not subject:
        raise ValueError("subject must be a non-empty string")
    if not isinstance(deriv_root, Path):
        raise ValueError("deriv_root must be a Path")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")

    plot_cfg = get_plot_config(config)
    plot_subdir = plot_cfg.get_behavioral_config().get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    out_dir = plots_dir / "dose_response"
    ensure_dir(out_dir)

    trials, trial_path = _load_trial_table(deriv_root, subject, config)
    if "task" in trials.columns:
        tasks = sorted({str(x) for x in trials["task"].dropna().unique() if str(x)})
        if tasks and task not in tasks:
            raise ValueError(
                f"Trial table task mismatch for sub-{subject}: expected task={task!r}, found tasks={tasks}. "
                f"trial_table={trial_path}"
            )

    bands = list(get_frequency_band_names(config))
    selected_bands = _resolve_optional_str_list(config, "plotting.plots.behavior.dose_response.bands")
    if selected_bands:
        selected_band_set = {str(b).strip() for b in selected_bands if str(b).strip()}
        bands = [b for b in bands if str(b) in selected_band_set]
    roi_defs = get_rois(config)
    roi_names = [str(k) for k in roi_defs.keys()]
    selected_rois = _resolve_optional_str_list(config, "plotting.plots.behavior.dose_response.rois")
    allowed_roi_set: Optional[set[str]] = None
    if selected_rois:
        allowed_roi_set = {str(r).strip() for r in selected_rois if str(r).strip()}
        roi_names = [r for r in roi_names if r in allowed_roi_set]
    selected_scopes = _resolve_optional_str_list(config, "plotting.plots.behavior.dose_response.scopes")
    allowed_scope_set: Optional[set[str]] = None
    if selected_scopes:
        allowed_scope_set = {str(s).strip() for s in selected_scopes if str(s).strip()}

    segment = _resolve_dose_response_segment(trials, config=config, logger=logger)
    stat = str(
        get_config_value(config, "plotting.plots.behavior.dose_response.stat", _DEFAULT_STAT)
    ).strip()
    if not stat:
        raise ValueError("dose_response.stat must be non-empty.")

    cols = _resolve_dose_response_columns(trials, config)

    table = trials.copy()

    # Resolve dose values
    table[cols.dose] = pd.to_numeric(_require_column(table, cols.dose), errors="coerce")

    saved: dict[str, Path] = {}
    errors: list[Exception] = []

    known_categories = set(CATEGORY_PREFIX_MAP.keys())
    selected_categories = [r for r in (cols.responses or []) if r in known_categories]
    generic_responses = [r for r in (cols.responses or []) if r in table.columns]

    # Optional generic response plots:
    # - If a selected entry matches a raw trial-table column name: plot it vs dose.
    if generic_responses:
        for resp in generic_responses:
            if resp not in table.columns:
                continue
            response_series = pd.to_numeric(_require_column(table, resp), errors="coerce")
            table[resp] = response_series
            out_path = out_dir / f"sub-{subject}_dose_response_{resp}_vs_{cols.dose}"
            _plot_xy_mean_sem(table, subject=subject, x_col=cols.dose, y_col=resp, out_path=out_path, config=config)
            saved[f"{resp}_vs_{cols.dose}"] = out_path

    # ROI × band power vs dose (main figure set)
    try:
        if not bands:
            raise ValueError("No frequency bands available in config (time_frequency_analysis.bands).")
        if allowed_scope_set is not None and "roi" not in allowed_scope_set:
            raise ValueError("ROI scope not selected.")
        if not roi_names:
            raise ValueError("No ROIs configured in time_frequency_analysis.rois.")

        roi_dir = out_dir / "roi_power"
        ensure_dir(roi_dir)
        for roi_name_str in roi_names:
            roi_table = pd.DataFrame({cols.dose: table[cols.dose]})
            roi_patterns = roi_defs.get(roi_name_str)
            used_bands: list[str] = []
            for band in bands:
                try:
                    y, used_cols = _roi_power_series(
                        table,
                        roi_name=roi_name_str,
                        roi_patterns=roi_patterns,
                        band=str(band),
                        segment=segment,
                        stat=stat,
                        logger=logger,
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping ROI power series (roi=%s, band=%s): %s",
                        roi_name_str,
                        band,
                        exc,
                    )
                    continue
                if not used_cols:
                    logger.warning(
                        "Skipping ROI power series (roi=%s, band=%s): used 0 columns.",
                        roi_name_str,
                        band,
                    )
                    continue
                roi_table[f"roi_power_{band}"] = y
                used_bands.append(str(band))
            roi_table = roi_table.dropna(subset=[cols.dose])
            if not used_bands:
                logger.warning(
                    "Skipping ROI power dose-response plots for roi=%s: no usable features found.",
                    roi_name_str,
                )
                continue

            roi_saved = _plot_roi_bands_vs_dose_single_subject(
                roi_table,
                subject=subject,
                roi_name=roi_name_str,
                bands=used_bands,
                dose_col=cols.dose,
                out_dir=roi_dir,
                config=config,
                logger=logger,
            )
            saved.update(roi_saved)
    except Exception as exc:
        errors.append(exc)
        logger.warning("Skipping ROI power dose-response plots: %s", exc)

    # Category-driven plots (e.g., connectivity, aperiodic, erp, ...).
    categories = list(dict.fromkeys(selected_categories))
    if not categories and not cols.responses and not saved:
        present = set(_present_feature_categories(table.columns))
        categories = sorted((present & set(CATEGORY_PREFIX_MAP.keys())) - {"power"})
    for category in categories:
        try:
            cat_saved = _plot_category_features_vs_dose_single_subject(
                table,
                subject=subject,
                category=category,
                dose_col=cols.dose,
                segment=segment,
                stat=stat,
                roi_defs=roi_defs,
                preferred_band_order=[str(b) for b in bands],
                allowed_scopes=allowed_scope_set,
                allowed_rois=allowed_roi_set,
                out_dir=out_dir,
                config=config,
                logger=logger,
            )
            saved.update(cat_saved)
        except Exception as exc:
            errors.append(exc)
            logger.warning("Skipping %s dose-response plots: %s", category, exc)

    if not saved and errors:
        raise errors[-1]

    logger.info("Created %d dose-response outputs under %s", len(saved), out_dir)
    return saved


def visualize_binary_outcome_probability(
    subject: str,
    deriv_root: Path,
    task: str,
    config: Any,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Generate binary outcome probability vs dose as a standalone plot."""
    if not isinstance(subject, str) or not subject:
        raise ValueError("subject must be a non-empty string")
    if not isinstance(deriv_root, Path):
        raise ValueError("deriv_root must be a Path")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")

    plot_cfg = get_plot_config(config)
    plot_subdir = plot_cfg.get_behavioral_config().get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    out_dir = plots_dir / "binary_outcome_probability"
    ensure_dir(out_dir)

    epochs, _ = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        logger=logger,
        config=config,
    )
    if epochs is None:
        raise FileNotFoundError(f"Could not locate clean epochs for sub-{subject}, task-{task}")

    aligned_events = get_aligned_events(
        epochs,
        subject,
        task,
        strict=True,
        logger=logger,
        config=config,
    )
    if aligned_events is None or aligned_events.empty:
        raise ValueError(f"Failed to load aligned events for sub-{subject}, task-{task}")

    dose_col, binary_outcome_col = _resolve_dose_binary_outcome_columns(aligned_events, config)

    table = aligned_events.reset_index(drop=True).copy()
    table[dose_col] = pd.to_numeric(_require_column(table, dose_col), errors="coerce")
    table[binary_outcome_col] = pd.to_numeric(_require_column(table, binary_outcome_col), errors="coerce")

    out_path = out_dir / f"sub-{subject}_binary_outcome_probability_vs_{dose_col}"
    _plot_binary_outcome_probability_vs_predictor(
        table,
        subject=subject,
        dose_col=dose_col,
        binary_outcome_col=binary_outcome_col,
        out_path=out_path,
        config=config,
    )

    logger.info("Created binary outcome probability plot under %s", out_dir)
    return {f"binary_outcome_probability_vs_{dose_col}": out_path}


__all__ = ["visualize_dose_response", "visualize_binary_outcome_probability"]
