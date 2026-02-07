from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.analysis.features.selection import resolve_feature_categories
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.utils.validation import validate_epochs
from eeg_pipeline.utils.progress import PipelineProgress
from eeg_pipeline.utils.parallel import get_n_jobs
from eeg_pipeline.analysis.features.results import (
    FeatureSet,
    ExtractionResult,
    FeatureExtractionResult,
)
from eeg_pipeline.analysis.features.precomputed.erds import extract_erds_from_precomputed
from eeg_pipeline.analysis.features.precomputed.extras import (
    extract_band_ratios_from_precomputed,
    extract_asymmetry_from_precomputed,
)
from eeg_pipeline.analysis.features.spectral import (
    extract_spectral_features,
    extract_power_features,
    extract_power_from_precomputed,
)
from eeg_pipeline.analysis.features.phase import (
    compute_pac_comodulograms,
    extract_itpc_from_precomputed,
    extract_phase_features,
    extract_pac_from_precomputed,
)
from eeg_pipeline.analysis.features.aperiodic import (
    extract_aperiodic_features,
)
from eeg_pipeline.analysis.features.complexity import extract_complexity_from_precomputed
from eeg_pipeline.analysis.features.erp import extract_erp_features
from eeg_pipeline.analysis.features.bursts import extract_burst_features
from eeg_pipeline.analysis.features.connectivity import (
    extract_connectivity_features,
    extract_connectivity_from_precomputed,
    extract_directed_connectivity_features,
    extract_directed_connectivity_from_precomputed,
)
from eeg_pipeline.analysis.features.source_localization import (
    extract_source_localization_features,
)
from eeg_pipeline.analysis.features.preparation import precompute_data
from eeg_pipeline.analysis.features.quality import (
    compute_trial_quality_metrics,
    extract_quality_features,
)

from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_for_subject,
    get_tfr_config,
    compute_complex_tfr,
    save_tfr_with_sidecar,
)


def _prepare_working_epochs(
    ctx: FeatureContext,
    tmin: Optional[float],
    tmax: Optional[float],
) -> "mne.Epochs":
    """Prepare epochs for feature extraction, cropping to time range if specified."""
    from eeg_pipeline.utils.analysis.spatial import crop_epochs_to_time_range

    if tmin is not None or tmax is not None:
        return crop_epochs_to_time_range(ctx.epochs, tmin, tmax, ctx.logger)
    return ctx.epochs


def _prepare_precomputed_data(
    ctx: FeatureContext,
    working_epochs: "mne.Epochs",
    power_bands: List[str],
    tmin: Optional[float],
    tmax: Optional[float],
) -> Optional[PrecomputedData]:
    """Prepare or compute precomputed intermediate data for feature extraction."""
    # Categories that use precomputed band data (ensures IAF-adjusted bands apply consistently)
    precompute_categories = {
        "connectivity",
        "directedconnectivity",
        "directed_connectivity",
        "erds",
        "ratios",
        "asymmetry",
        "complexity",
        "bursts",
        "spectral",
        "aperiodic",
    }
    baseline_dependent_categories = {"erds", "bursts"}

    needs_precompute = bool(precompute_categories & set(ctx.feature_categories))
    if not needs_precompute:
        return None

    needs_baseline = bool(baseline_dependent_categories & set(ctx.feature_categories))
    has_time_range = tmin is not None or tmax is not None

    cached = None
    getter = getattr(ctx, "get_precomputed_for_family", None)
    if callable(getter):
        cached = getter("spectral")
    if cached is None:
        cached = ctx.precomputed

    if cached is not None:
        precomputed_data = cached

        if has_time_range and not needs_baseline:
            ctx.logger.info(f"Cropping precomputed intermediates to range [{tmin}, {tmax}]")
            precomputed_data = precomputed_data.crop(tmin, tmax)
        elif needs_baseline and has_time_range:
            ctx.logger.info(
                f"Keeping full precomputed data for baseline normalization "
                f"(target range: [{tmin}, {tmax}])"
            )
        return precomputed_data

    if working_epochs is None:
        raise ValueError("Missing epochs; cannot precompute feature intermediates.")

    epochs_for_precompute = working_epochs
    if has_time_range and needs_baseline:
        epochs_for_precompute = getattr(ctx, "_original_epochs", None) or working_epochs

    precomputed_evoked_subtracted = False
    precomputed_evoked_subtracted_conditionwise = False
    pre_cfg = ctx.config.get("feature_engineering.precomputed", {}) if hasattr(ctx.config, "get") else {}
    subtract_evoked_cfg = pre_cfg.get("subtract_evoked", None)
    if subtract_evoked_cfg is None:
        subtract_evoked_cfg = (
            ctx.config.get("feature_engineering.power.subtract_evoked", False)
            if hasattr(ctx.config, "get")
            else False
        )
    want_induced_precomputed = bool(subtract_evoked_cfg)

    if want_induced_precomputed:
        analysis_mode = str(getattr(ctx, "analysis_mode", "") or "").strip().lower()
        train_mask = getattr(ctx, "train_mask", None)
        if analysis_mode == "trial_ml_safe" and train_mask is None:
            raise ValueError(
                "Precomputed subtract_evoked requested in trial_ml_safe mode without train_mask. "
                "Evoked subtraction uses cross-trial averages and can leak in CV. "
                "Provide train_mask or disable subtract_evoked."
            )
        else:
            from eeg_pipeline.utils.analysis.spectral import subtract_evoked

            condition_labels = None
            if isinstance(ctx.aligned_events, pd.DataFrame):
                for candidate in ("condition", "trial_type"):
                    if candidate in ctx.aligned_events.columns:
                        condition_labels = ctx.aligned_events[candidate].to_numpy()
                        break

            induced_epochs = epochs_for_precompute.copy()
            data = induced_epochs.get_data()
            min_trials = int(
                ctx.config.get("feature_engineering.power.min_trials_per_condition", 2)
                if hasattr(ctx.config, "get")
                else 2
            )
            induced = subtract_evoked(
                data,
                condition_labels=condition_labels,
                train_mask=train_mask,
                min_trials_per_condition=min_trials,
            )
            induced_epochs._data = induced
            epochs_for_precompute = induced_epochs
            precomputed_evoked_subtracted = True
            precomputed_evoked_subtracted_conditionwise = condition_labels is not None

    if not epochs_for_precompute.preload:
        ctx.logger.info("Preloading epochs data...")
        epochs_for_precompute.load_data()

    relevant_categories = sorted(list(precompute_categories & set(ctx.feature_categories)))
    ctx.logger.info(f"Computing shared intermediate data for: {', '.join(relevant_categories)}...")

    needs_psd = any(
        category in ctx.feature_categories
        for category in ["erds", "aperiodic", "spectral"]
    )

    precomputed_data = precompute_data(
        epochs_for_precompute,
        power_bands,
        ctx.config,
        ctx.logger,
        windows_spec=ctx.windows,
        compute_psd_data=needs_psd,
        frequency_bands_override=getattr(ctx, "frequency_bands", None),
        feature_family="spectral",
        train_mask=getattr(ctx, "train_mask", None),
    )
    precomputed_data.evoked_subtracted = bool(precomputed_evoked_subtracted)
    precomputed_data.evoked_subtracted_conditionwise = bool(precomputed_evoked_subtracted_conditionwise)
    setter = getattr(ctx, "set_precomputed_for_family", None)
    if callable(setter):
        setter("spectral", precomputed_data)
    else:
        ctx.set_precomputed(precomputed_data)
    return precomputed_data


def _align_precomputed_windows(
    ctx: FeatureContext,
    precomputed_data: PrecomputedData,
) -> Optional[PrecomputedData]:
    """Align precomputed data windows with context windows if possible."""
    if ctx.windows is None:
        return precomputed_data

    ctx_times = getattr(ctx.windows, "times", None)
    if (
        ctx_times is not None
        and len(precomputed_data.times) == len(ctx_times)
        and np.allclose(precomputed_data.times, ctx_times, atol=0, rtol=0)
    ):
        aligned_windows = ctx.windows
    else:
        ranges = getattr(ctx.windows, "ranges", None)
        if ranges:
            explicit = [
                {"name": name, "tmin": rng[0], "tmax": rng[1]}
                for name, rng in ranges.items()
                if isinstance(rng, (list, tuple)) and len(rng) >= 2
            ]
        else:
            explicit = None

        if explicit:
            from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec

            spec = TimeWindowSpec(
                times=precomputed_data.times,
                config=ctx.config,
                sampling_rate=float(getattr(precomputed_data, "sfreq", 1.0)),
                logger=ctx.logger,
                name=ctx.name,
                explicit_windows=explicit,
            )
            aligned_windows = time_windows_from_spec(spec, logger=ctx.logger, strict=False)
        else:
            aligned_windows = None

    if aligned_windows is not None:
        precomputed_data = precomputed_data.with_windows(aligned_windows)
        ctx.set_precomputed(precomputed_data)

    return precomputed_data


def _compute_tfr_for_features(
    ctx: FeatureContext,
    tmin: Optional[float],
    tmax: Optional[float],
) -> tuple[Any, Optional[pd.DataFrame], List[str], Optional[float], Optional[float]]:
    """Compute TFR and baseline data for feature extraction.
    
    IMPORTANT:
    - Power TFR is computed from the original (uncropped) epochs so baseline windows
      remain available for normalization.
    - Complex TFR is expensive; for multi-range extraction it can be provided by the
      pipeline via ctx.tfr_complex to avoid recomputing it for every time range.
    """
    tfr_complex = None
    tfr_power = None
    epochs_for_complex = None
    
    epochs_for_tfr = getattr(ctx, "_original_epochs", None) or ctx.epochs
    epochs_for_power_tfr = epochs_for_tfr

    # Evoked subtraction for induced power features (pain paradigms)
    power_cfg = ctx.config.get("feature_engineering.power", {}) if hasattr(ctx.config, "get") else {}
    want_induced_power = bool(power_cfg.get("subtract_evoked", False))
    ctx.power_evoked_subtracted = False
    ctx.power_evoked_subtracted_conditionwise = False
    if want_induced_power:
        analysis_mode = str(getattr(ctx, "analysis_mode", "") or "").strip().lower()
        train_mask = getattr(ctx, "train_mask", None)
        if analysis_mode == "trial_ml_safe" and train_mask is None:
            raise ValueError(
                "Power subtract_evoked=True in trial_ml_safe mode without train_mask. "
                "Evoked subtraction uses cross-trial averages and can leak in CV. "
                "Provide train_mask or disable subtract_evoked."
            )
        else:
            from eeg_pipeline.utils.analysis.spectral import subtract_evoked

            condition_labels = None
            if isinstance(ctx.aligned_events, pd.DataFrame):
                for candidate in ("condition", "trial_type"):
                    if candidate in ctx.aligned_events.columns:
                        condition_labels = ctx.aligned_events[candidate].to_numpy()
                        break

            induced_epochs = epochs_for_tfr.copy()
            data = induced_epochs.get_data()
            induced = subtract_evoked(
                data,
                condition_labels=condition_labels,
                train_mask=train_mask,
                min_trials_per_condition=int(power_cfg.get("min_trials_per_condition", 2)),
            )
            induced_epochs._data = induced
            epochs_for_power_tfr = induced_epochs
            ctx.power_evoked_subtracted = True
            ctx.power_evoked_subtracted_conditionwise = condition_labels is not None

    pac_needs_complex = False
    if "pac" in ctx.feature_categories:
        pac_cfg = ctx.config.get("feature_engineering.pac", {}) if hasattr(ctx.config, "get") else {}
        pac_source = str(pac_cfg.get("source", "precomputed")).strip().lower()
        pac_needs_complex = pac_source != "precomputed"

    needs_complex = ("itpc" in ctx.feature_categories) or pac_needs_complex
    if needs_complex:
        from eeg_pipeline.analysis.features.preparation import (
            _apply_spatial_transform,
            _get_spatial_transform_type,
        )

        # Fast-path reuse: if the pipeline provided a precomputed complex TFR for these
        # epochs (e.g., multi-range extraction), reuse it without recomputing transforms.
        existing_complex = getattr(ctx, "tfr_complex", None)
        if existing_complex is not None:
            try:
                if (
                    hasattr(existing_complex, "times")
                    and hasattr(epochs_for_tfr, "times")
                    and len(existing_complex.times) == len(epochs_for_tfr.times)
                    and np.isclose(float(existing_complex.times[0]), float(epochs_for_tfr.times[0]))
                    and np.isclose(float(existing_complex.times[-1]), float(epochs_for_tfr.times[-1]))
                ):
                    tfr_complex = existing_complex
            except Exception:
                tfr_complex = None

        if tfr_complex is None:
            phase_family = "itpc" if "itpc" in ctx.feature_categories else "pac"
            phase_transform = _get_spatial_transform_type(ctx.config, feature_family=phase_family)
            epochs_for_complex = epochs_for_tfr
            if phase_transform in {"csd", "laplacian"}:
                epochs_for_complex = epochs_for_tfr.copy().pick_types(
                    eeg=True, meg=False, eog=False, stim=False, exclude="bads"
                )
                epochs_for_complex = _apply_spatial_transform(
                    epochs_for_complex, phase_transform, ctx.config, ctx.logger
                )
            tfr_complex = compute_complex_tfr(epochs_for_complex, ctx.config, ctx.logger)
        if tfr_complex is not None:
            ctx.tfr_complex = tfr_complex
            if (
                tfr_power is None
                and not ctx.power_evoked_subtracted
                and epochs_for_complex is epochs_for_tfr
            ):
                ctx.logger.info("Deriving power TFR from complex TFR...")
                tfr_power = tfr_complex.copy()
                tfr_power.data = np.abs(tfr_complex.data) ** 2
                tfr_power.comment = "derived_from_complex"

    if tfr_power is None and ctx.tfr is not None and not ctx.power_evoked_subtracted:
        ctx.logger.info("Using pre-computed TFR from context")
        tfr_power = ctx.tfr

    if tfr_power is None and ctx.power_evoked_subtracted:
        from eeg_pipeline.utils.analysis.tfr import compute_tfr_morlet

        ctx.logger.info("Computing induced power TFR (evoked subtracted)...")
        tfr_power = compute_tfr_morlet(epochs_for_power_tfr, ctx.config, logger=ctx.logger)
        if tfr_power is not None:
            tfr_power.comment = "evoked_subtracted"

    baseline_override = None
    if ctx.windows is not None:
        baseline_start, baseline_end = ctx.windows.baseline_range
        if np.isfinite(baseline_start) and np.isfinite(baseline_end):
            baseline_override = (float(baseline_start), float(baseline_end))

    tfr, baseline_df, baseline_cols, baseline_start, baseline_end = compute_tfr_for_subject(
        epochs_for_power_tfr,
        ctx.aligned_events,
        ctx.subject,
        ctx.task,
        ctx.config,
        ctx.deriv_root,
        ctx.logger,
        tfr_computed=tfr_power,
        baseline_window=baseline_override,
    )

    if tfr is None:
        raise ValueError("TFR computation failed; aborting feature extraction.")

    if tmin is not None or tmax is not None:
        crop_min = tmin if tmin is not None else tfr.times[0]
        crop_max = tmax if tmax is not None else tfr.times[-1]

        if crop_min is not None and crop_max is not None and crop_min > crop_max:
            crop_min, crop_max = crop_max, crop_min

        crop_min = max(crop_min, tfr.times[0]) if crop_min is not None else tfr.times[0]
        crop_max = min(crop_max, tfr.times[-1]) if crop_max is not None else tfr.times[-1]

        if crop_min < crop_max:
            ctx.logger.info(f"Cropping TFR to range [{crop_min:.3f}, {crop_max:.3f}]")
            tfr = tfr.copy().crop(crop_min, crop_max)
        else:
            ctx.logger.warning(
                f"Requested TFR range [{tmin}, {tmax}] is invalid or outside available data; "
                "skipping crop."
            )

    if ctx.config.get("feature_engineering.save_tfr_with_sidecar", False):
        # In multi-range extraction, the per-range TFR is cropped to the current window.
        # Applying a baseline correction after cropping can fail for non-baseline windows
        # because the baseline interval is outside the cropped time axis.
        # Save the TFR only once (for the baseline window if present).
        explicit_windows = getattr(ctx, "explicit_windows", None)
        multi_range = isinstance(explicit_windows, (list, tuple)) and len(explicit_windows) > 1
        name = str(getattr(ctx, "name", "") or "").strip().lower()
        if multi_range and name not in {"", "baseline"}:
            ctx.logger.info(
                "Skipping TFR save for range '%s' during multi-range extraction; saving only baseline TFR.",
                getattr(ctx, "name", None),
            )
        else:
            _save_tfr_with_sidecar(ctx, tfr, baseline_start, baseline_end)

    return tfr, baseline_df, baseline_cols, baseline_start, baseline_end


def _save_tfr_with_sidecar(
    ctx: FeatureContext,
    tfr: Any,
    baseline_start: Optional[float],
    baseline_end: Optional[float],
) -> None:
    """Save baselined TFR with sidecar metadata if configured."""
    if tfr is None or baseline_start is None or baseline_end is None:
        return

    baseline_mode = str(ctx.config.get("time_frequency_analysis.baseline_mode", "logratio"))
    tfr_to_save = tfr.copy()
    tfr_to_save.apply_baseline(baseline=(baseline_start, baseline_end), mode=baseline_mode)
    tfr_to_save.comment = f"BASELINED:mode={baseline_mode};win=({baseline_start:.3f},{baseline_end:.3f})"
    tfr_output_path = (
        ctx.deriv_root
        / f"sub-{ctx.subject}"
        / "eeg"
        / f"sub-{ctx.subject}_task-{ctx.task}_power_epo-tfr.h5"
    )
    save_tfr_with_sidecar(
        tfr_to_save,
        tfr_output_path,
        (baseline_start, baseline_end),
        baseline_mode,
        ctx.logger,
        ctx.config,
    )


def _extract_pac_features(
    ctx: FeatureContext,
    precomputed_data: Optional[PrecomputedData],
    tfr_complex: Optional[Any],
) -> tuple[
    Optional[pd.DataFrame],
    Optional[Any],
    Optional[Any],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """Extract PAC features using precomputed data or TFR-based computation."""
    pac_config = ctx.config.get("feature_engineering.pac", {}) if hasattr(ctx.config, "get") else {}
    pac_source = str(pac_config.get("source", "precomputed")).strip().lower()

    if pac_source == "precomputed":
        precomputed_pac = None
        getter = getattr(ctx, "get_precomputed_for_family", None)
        if callable(getter):
            precomputed_pac = getter("pac")
        if precomputed_pac is None:
            precomputed_pac = precomputed_data

        # Scientific validity: Hilbert PAC should respect the PAC family spatial transform
        # (e.g., CSD). If we reuse a precomputed object built for a different family, it
        # may have spatial_transform='none' and bias phase-based features.
        expected_transform = "none"
        try:
            from eeg_pipeline.analysis.features.preparation import _get_spatial_transform_type

            expected_transform = _get_spatial_transform_type(ctx.config, feature_family="pac")
        except Exception:
            expected_transform = "none"

        current_transform = str(getattr(precomputed_pac, "spatial_transform", "none")).strip().lower()
        if precomputed_pac is not None and expected_transform in {"csd", "laplacian"}:
            if current_transform != expected_transform:
                ctx.logger.warning(
                    "PAC: existing precomputed intermediates have spatial_transform='%s' but PAC requires '%s'; "
                    "recomputing precomputed intermediates for PAC.",
                    current_transform,
                    expected_transform,
                )
                precomputed_pac = None

        if precomputed_pac is None and getattr(ctx, "epochs", None) is not None:
            from eeg_pipeline.utils.config.loader import get_frequency_band_names

            bands_for_pac = ctx.bands if ctx.bands else get_frequency_band_names(ctx.config)
            precomputed_pac = precompute_data(
                ctx.epochs,
                bands_for_pac,
                ctx.config,
                ctx.logger,
                windows_spec=ctx.windows,
                feature_family="pac",
                train_mask=getattr(ctx, "train_mask", None),
            )
            setter = getattr(ctx, "set_precomputed_for_family", None)
            if callable(setter):
                setter("pac", precomputed_pac)
            else:
                ctx.set_precomputed(precomputed_pac)

        if precomputed_pac is not None:
            ctx.logger.info(
                "Using Hilbert PAC from precomputed analytic signals "
                "(recommended for scientific validity)"
            )
            pac_trials_df, _pac_cols = extract_pac_from_precomputed(precomputed_pac, ctx.config)
            if pac_trials_df is not None and not pac_trials_df.empty:
                return None, None, None, pac_trials_df, None
            return None, None, None, None, None
        raise ValueError(
            "PAC source='precomputed' requested but precomputed data is unavailable. "
            "Compute precomputed intermediates first or change feature_engineering.pac.source."
        )

    if tfr_complex is None:
        tfr_complex = compute_complex_tfr(ctx.epochs, ctx.config, ctx.logger)
        if tfr_complex is not None:
            ctx.tfr_complex = tfr_complex

    if tfr_complex is None:
        raise ValueError("PAC requested but complex TFR computation failed.")

    freq_min, freq_max, n_freqs, *_ = get_tfr_config(ctx.config)
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

    segment_window = None
    segment_label = ctx.name or getattr(ctx.windows, "name", None) or "full"
    segment_range = None
    
    if ctx.windows is not None:
        if ctx.name and ctx.windows.ranges.get(ctx.name) is not None:
            segment_range = ctx.windows.ranges.get(ctx.name)
        elif hasattr(ctx.windows, "active_range") and ctx.windows.active_range is not None:
            active_range = ctx.windows.active_range
            if np.isfinite(active_range[0]) and np.isfinite(active_range[1]):
                segment_range = active_range

        if segment_range is not None:
            segment_window = (float(segment_range[0]), float(segment_range[1]))

    pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = (
        compute_pac_comodulograms(
            tfr_complex,
            frequencies,
            tfr_complex.times,
            ctx.epochs.info,
            ctx.config,
            ctx.logger,
            segment_name=segment_label,
            segment_window=segment_window,
            spatial_modes=ctx.spatial_modes,
        )
    )
    return pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df


def _extract_feature_with_error_handling(
    ctx: FeatureContext,
    feature_name: str,
    extractor_func: callable,
    expected_trials: int,
    progress: PipelineProgress,
    *args,
    **kwargs,
) -> tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[Any]]:
    """Extract a feature category with standardized error handling and validation."""
    progress.step(message=f"Extracting {feature_name} features...")
    extraction_result = extractor_func(*args, **kwargs)
    if isinstance(extraction_result, tuple):
        if len(extraction_result) == 2:
            df, cols = extraction_result
            qc = None
        elif len(extraction_result) == 3:
            df, cols, qc = extraction_result
        else:
            raise ValueError(f"Unexpected return tuple length: {len(extraction_result)}")
    else:
        df, cols, qc = extraction_result, [], None

    if df is not None and not df.empty and len(df) != expected_trials:
        raise ValueError(f"{feature_name} length mismatch: {len(df)} vs {expected_trials}")

    return df, cols, qc


def _apply_spatial_filtering_to_results(
    ctx: FeatureContext,
    results: FeatureExtractionResult,
) -> None:
    """Apply spatial mode filtering to all feature DataFrames in results."""
    if not ctx.spatial_modes:
        return

    ctx.logger.info(f"Filtering features by spatial modes: {ctx.spatial_modes}")

    feature_attributes = [
        ("pow_df", "pow_cols"),
        ("erp_df", "erp_cols"),
        ("aper_df", "aper_cols"),
        ("conn_df", "conn_cols"),
        ("comp_df", "comp_cols"),
        ("bursts_df", "bursts_cols"),
        ("spectral_df", "spectral_cols"),
    ]

    for df_attr, cols_attr in feature_attributes:
        df = getattr(results, df_attr, None)
        if df is not None:
            filtered_df = filter_features_by_spatial_modes(df, ctx.spatial_modes, ctx.config)
            setattr(results, df_attr, filtered_df)
            setattr(results, cols_attr, list(filtered_df.columns) if filtered_df is not None else [])


def filter_features_by_spatial_modes(
    df: Optional[pd.DataFrame],
    spatial_modes: List[str],
    config: Any,
) -> Optional[pd.DataFrame]:
    """Filter feature DataFrame columns to only include those matching spatial_modes.
    
    Feature columns are identified by naming patterns:
    - '_ch_': per-channel features (include if 'channels' in spatial_modes)
    - '_chpair_': channel pair features (include if 'channels' in spatial_modes)
    - '_global_' or '_global': global features (include if 'global' in spatial_modes)
    - '_roi_' or ROI name patterns: ROI features (include if 'roi' in spatial_modes)
    
    Non-spatial features are always included.
    """
    if not spatial_modes or df is None or getattr(df, "empty", True):
        return df

    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions

    roi_definitions = get_roi_definitions(config)
    roi_names = list(roi_definitions.keys()) if roi_definitions else []

    columns_to_keep: List[str] = []

    for column in df.columns:
        column_str = str(column)

        is_per_channel = "_ch_" in column_str
        is_channel_pair = "_chpair_" in column_str
        is_global = "_global_" in column_str or column_str.endswith("_global")
        is_roi = "_roi_" in column_str or any(
            f"_{roi}_" in column_str or column_str.endswith(f"_{roi}") for roi in roi_names
        )

        is_spatial = is_per_channel or is_channel_pair or is_global or is_roi

        if not is_spatial:
            columns_to_keep.append(column)
            continue

        if (is_per_channel or is_channel_pair) and "channels" in spatial_modes:
            columns_to_keep.append(column)
        elif is_global and "global" in spatial_modes:
            columns_to_keep.append(column)
        elif is_roi and "roi" in spatial_modes:
            columns_to_keep.append(column)

    return df[columns_to_keep] if columns_to_keep else pd.DataFrame()


def extract_all_features(
    ctx: FeatureContext,
) -> FeatureExtractionResult:
    """Extract all requested features from epochs using TFR-based pipeline."""
    tmin, tmax = ctx.tmin, ctx.tmax
    if tmin is not None or tmax is not None:
        ctx.logger.info(f"Feature extraction restricted to time range: tmin={tmin}, tmax={tmax}")

    if ctx.spatial_modes:
        ctx.logger.info(f"Spatial aggregation modes: {', '.join(ctx.spatial_modes)}")

    power_bands = ctx.bands if ctx.bands else get_frequency_band_names(ctx.config)
    ctx._original_epochs = ctx.epochs
    windows_full = ctx.windows
    working_epochs = _prepare_working_epochs(ctx, tmin, tmax)
    expected_n_trials = len(working_epochs)
    ctx.epochs = working_epochs

    # Rebase masks onto cropped time axis while preserving original ranges.
    # Prevents mask/time-length mismatches while allowing baseline-dependent
    # computations to reference the original baseline range.
    if windows_full is not None and getattr(windows_full, "ranges", None) is not None:
        from eeg_pipeline.types import TimeWindows

        ranges_full = dict(getattr(windows_full, "ranges", {}) or {})
        working_times = getattr(working_epochs, "times", None)
        new_times = np.asarray(working_times if working_times is not None else [], dtype=float)
        new_masks = {}
        for win_name, rng in ranges_full.items():
            try:
                start, end = float(rng[0]), float(rng[1])
            except (TypeError, ValueError, IndexError):
                new_masks[win_name] = np.zeros_like(new_times, dtype=bool)
                continue
            if not (np.isfinite(start) and np.isfinite(end) and end > start):
                new_masks[win_name] = np.zeros_like(new_times, dtype=bool)
                continue
            new_masks[win_name] = (new_times >= start) & (new_times < end)

        baseline_range = ranges_full.get("baseline", (np.nan, np.nan))
        active_key = None
        if ctx.name and ctx.name in new_masks:
            active_key = ctx.name
        elif "active" in new_masks:
            active_key = "active"
        active_range = ranges_full.get(active_key, (np.nan, np.nan)) if active_key else (np.nan, np.nan)

        empty_mask = np.zeros_like(new_times, dtype=bool)
        ctx._windows = TimeWindows(
            baseline_mask=new_masks.get("baseline", empty_mask),
            active_mask=new_masks.get(active_key, empty_mask) if active_key else empty_mask,
            baseline_range=baseline_range,
            active_range=active_range,
            masks=new_masks,
            ranges=ranges_full,
            clamped=getattr(windows_full, "clamped", False),
            valid=getattr(windows_full, "valid", True),
            errors=list(getattr(windows_full, "errors", []) or []),
            times=new_times,
            name=ctx.name,
        )

    precomputed_data = _prepare_precomputed_data(ctx, working_epochs, power_bands, tmin, tmax)
    if precomputed_data is not None:
        precomputed_data = _align_precomputed_windows(ctx, precomputed_data)
        precomputed_data.spatial_modes = list(ctx.spatial_modes) if ctx.spatial_modes else None
        ctx.set_precomputed(precomputed_data)

    validation = validate_epochs(working_epochs, ctx.config, logger=ctx.logger)
    if not validation.valid:
        ctx.logger.warning(f"Validation issues: {validation.issues}")
        if validation.critical:
            raise ValueError(f"Critical errors: {validation.critical}")

    results = FeatureExtractionResult()

    tfr_dependent_categories = {"power", "itpc", "pac"}
    needs_tfr = bool(tfr_dependent_categories & set(ctx.feature_categories))

    tfr_complex = None
    if needs_tfr:
        tfr, baseline_df, baseline_cols, baseline_start, baseline_end = _compute_tfr_for_features(
            ctx, tmin, tmax
        )
        results.tfr = tfr
        results.baseline_df = baseline_df
        results.baseline_cols = baseline_cols

        ctx.results["tfr"] = tfr
        ctx.tfr = tfr
        ctx.baseline_df = baseline_df
        ctx.baseline_cols = baseline_cols

        if any(category in ctx.feature_categories for category in ["itpc", "pac"]):
            tfr_complex = ctx.tfr_complex
    else:
        ctx.logger.info("Skipping TFR computation (not needed for requested feature categories)")

    progress = PipelineProgress(total=len(ctx.feature_categories), logger=ctx.logger, desc="Features")
    progress.start()

    if "power" in ctx.feature_categories:
        pow_df, pow_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "power",
            extract_power_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.pow_df = pow_df
        results.pow_cols = pow_cols or []

    if "connectivity" in ctx.feature_categories:
        conn_df, conn_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "connectivity",
            extract_connectivity_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.conn_df = conn_df
        results.conn_cols = conn_cols or []

    if "directedconnectivity" in ctx.feature_categories:
        dconn_df, dconn_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "directedconnectivity",
            extract_directed_connectivity_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.dconn_df = dconn_df
        results.dconn_cols = dconn_cols or []

    if "sourcelocalization" in ctx.feature_categories:
        source_df, source_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "sourcelocalization",
            extract_source_localization_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.source_df = source_df
        results.source_cols = source_cols or []

    if "aperiodic" in ctx.feature_categories:
        aper_df, aper_cols, qc_payload = _extract_feature_with_error_handling(
            ctx,
            "aperiodic",
            extract_aperiodic_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.aper_df = aper_df
        results.aper_cols = aper_cols or []
        results.aper_qc = qc_payload

    if "erp" in ctx.feature_categories:
        erp_df, erp_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "ERP",
            extract_erp_features,
            expected_n_trials,
            progress,
            ctx,
        )
        results.erp_df = erp_df
        results.erp_cols = erp_cols or []

    if "complexity" in ctx.feature_categories and precomputed_data is not None:
        comp_df, comp_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "complexity",
            extract_complexity_from_precomputed,
            expected_n_trials,
            progress,
            precomputed_data,
        )
        results.comp_df = comp_df
        results.comp_cols = comp_cols or []

    if "bursts" in ctx.feature_categories:
        bursts_df, bursts_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "burst",
            extract_burst_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.bursts_df = bursts_df
        results.bursts_cols = bursts_cols or []

    if "itpc" in ctx.feature_categories or "phase" in ctx.feature_categories:
        phase_df, phase_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "phase",
            extract_phase_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        results.phase_df = phase_df
        results.phase_cols = phase_cols or []

    if "pac" in ctx.feature_categories:
        progress.step(message="Computing PAC features...")
        pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = _extract_pac_features(
            ctx, precomputed_data, tfr_complex
        )

        if pac_trials_df is not None:
            if len(pac_trials_df) != expected_n_trials:
                raise ValueError(
                    f"PAC trials length mismatch: {len(pac_trials_df)} vs {expected_n_trials}"
                )
            results.pac_trials_df = pac_trials_df
            results.pac_df = pac_df
            results.pac_phase_freqs = pac_phase_freqs
            results.pac_amp_freqs = pac_amp_freqs
            results.pac_time_df = pac_time_df

            if pac_time_df is not None and len(pac_time_df) != expected_n_trials:
                raise ValueError(
                    f"PAC time-resolved length mismatch: {len(pac_time_df)} vs {expected_n_trials}"
                )

    if "erds" in ctx.feature_categories and precomputed_data is not None:
        erds_df, erds_cols, _erds_qc = _extract_feature_with_error_handling(
            ctx,
            "ERDS",
            extract_erds_from_precomputed,
            expected_n_trials,
            progress,
            precomputed_data,
            power_bands,
        )
        if erds_df is not None and not erds_df.empty:
            results.erds_df = erds_df
            results.erds_cols = erds_cols or []

    if "spectral" in ctx.feature_categories:
        spectral_df, spectral_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "spectral",
            extract_spectral_features,
            expected_n_trials,
            progress,
            ctx,
            power_bands,
        )
        if spectral_df is not None and not spectral_df.empty:
            results.spectral_df = spectral_df
            results.spectral_cols = spectral_cols or []

    if "ratios" in ctx.feature_categories and precomputed_data is not None:
        ratio_df, ratio_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "band ratio",
            extract_band_ratios_from_precomputed,
            expected_n_trials,
            progress,
            precomputed_data,
            ctx.config,
        )
        if ratio_df is not None and not ratio_df.empty:
            results.ratios_df = ratio_df
            results.ratios_cols = ratio_cols or []

    if "asymmetry" in ctx.feature_categories and precomputed_data is not None:
        n_jobs = int(ctx.config.get("feature_engineering.parallel.n_jobs_bands", -1))
        asym_df, asym_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "asymmetry",
            extract_asymmetry_from_precomputed,
            expected_n_trials,
            progress,
            precomputed_data,
            n_jobs=n_jobs,
        )
        if asym_df is not None and not asym_df.empty:
            results.asymmetry_df = asym_df
            results.asymmetry_cols = asym_cols or []

    if "quality" in ctx.feature_categories:
        qual_df, qual_cols, _ = _extract_feature_with_error_handling(
            ctx,
            "quality",
            extract_quality_features,
            expected_n_trials,
            progress,
            ctx,
        )
        results.quality_df = qual_df
        results.quality_cols = qual_cols or []

    _apply_spatial_filtering_to_results(ctx, results)
    _add_change_scores_to_results(ctx, results)

    progress.finish()
    return results


def _add_change_scores_to_results(
    ctx: FeatureContext,
    results: FeatureExtractionResult,
) -> None:
    """Compute and add change scores (active - baseline) to feature DataFrames.
    
    Change scores are computed once at feature extraction time and saved,
    eliminating redundant computation in downstream pipelines.
    """
    from eeg_pipeline.utils.analysis.stats.transforms import compute_change_features

    add_change = ctx.config.get("feature_engineering.compute_change_scores", True)
    if not add_change:
        return

    feature_dfs = [
        ("pow_df", "pow_cols"),
        ("conn_df", "conn_cols"),
        ("dconn_df", "dconn_cols"),
        ("source_df", "source_cols"),
        ("aper_df", "aper_cols"),
        ("phase_df", "phase_cols"),
        ("pac_trials_df", None),
        ("comp_df", "comp_cols"),
        ("erds_df", "erds_cols"),
        ("spectral_df", "spectral_cols"),
        ("ratios_df", "ratios_cols"),
        ("asymmetry_df", "asymmetry_cols"),
    ]

    n_added = 0
    for df_attr, cols_attr in feature_dfs:
        df = getattr(results, df_attr, None)
        if df is None or df.empty:
            continue

        change_df = compute_change_features(df)
        if change_df.empty:
            continue

        new_cols = [c for c in change_df.columns if c not in df.columns]
        if not new_cols:
            continue

        combined = pd.concat([df, change_df[new_cols]], axis=1)
        setattr(results, df_attr, combined)

        if cols_attr:
            existing_cols = getattr(results, cols_attr, []) or []
            setattr(results, cols_attr, existing_cols + new_cols)

        n_added += len(new_cols)

    if n_added > 0:
        ctx.logger.info("Added %d change score columns to feature results", n_added)


def _extract_precomputed_feature_group(
    feature_name: str,
    extractor_func: callable,
    precomputed: PrecomputedData,
    logger: Any,
    result: ExtractionResult,
    *args,
    **kwargs,
) -> None:
    """Extract a feature group from precomputed data and add to result."""
    logger.info(f"Extracting {feature_name} features...")
    extraction_result = extractor_func(precomputed, *args, **kwargs)
    if isinstance(extraction_result, tuple):
        if len(extraction_result) == 2:
            df, cols = extraction_result
        elif len(extraction_result) == 3:
            df, cols, qc = extraction_result
            result.qc[feature_name] = qc
        else:
            raise ValueError(f"Unexpected return tuple length: {len(extraction_result)}")
    else:
        df, cols = extraction_result, []

    if not df.empty:
        result.features[feature_name] = FeatureSet(df, cols, feature_name)
    else:
        result.qc[feature_name] = {"skipped_reason": "empty_result"}


def extract_precomputed_features(
    epochs: "mne.Epochs",
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    feature_groups: Optional[List[str]] = None,
    events_df: Optional[pd.DataFrame] = None,
    precomputed: Optional[PrecomputedData] = None,
) -> ExtractionResult:
    """Extract features from precomputed intermediate data."""
    if feature_groups is None:
        feature_groups = ["erds", "spectral"]

    band_dependent_groups = ["erds", "spectral", "connectivity", "pac", "ratios"]
    needs_bands = any(group in feature_groups for group in band_dependent_groups)
    needs_psd = any(group in feature_groups for group in ["aperiodic", "spectral"])

    if precomputed is None:
        logger.info("Precomputing intermediate data (bands=%s, psd=%s)...", needs_bands, needs_psd)
        precomputed = precompute_data(
            epochs,
            bands,
            config,
            logger,
            compute_bands=needs_bands,
            compute_psd_data=needs_psd,
        )
    else:
        logger.info("Using provided precomputed intermediates")

    result = ExtractionResult(precomputed=precomputed)

    if events_df is not None:
        if len(events_df) != precomputed.data.shape[0]:
            logger.warning(
                "Precomputed: events_df length (%d) != n_epochs (%d); skipping metadata/condition labels.",
                len(events_df),
                precomputed.data.shape[0],
            )
        else:
            precomputed.metadata = events_df.reset_index(drop=True).copy()
            if "condition" in events_df.columns:
                result.condition = events_df["condition"].to_numpy()
            elif "trial_type" in events_df.columns:
                result.condition = events_df["trial_type"].to_numpy()
            precomputed.condition_labels = result.condition

    if "erds" in feature_groups:
        _extract_precomputed_feature_group(
            "erds", extract_erds_from_precomputed, precomputed, logger, result, bands
        )

    if "spectral" in feature_groups:
        _extract_precomputed_feature_group(
            "spectral", extract_power_from_precomputed, precomputed, logger, result, bands
        )

    if "aperiodic" in feature_groups:
        from eeg_pipeline.analysis.features.aperiodic import extract_aperiodic_from_precomputed
        _extract_precomputed_feature_group(
            "aperiodic", extract_aperiodic_from_precomputed, precomputed, logger, result, bands
        )

    if "connectivity" in feature_groups:
        _extract_precomputed_feature_group(
            "connectivity", extract_connectivity_from_precomputed, precomputed, logger, result
        )

    if "directed_connectivity" in feature_groups:
        def extract_directed_connectivity_wrapper(precomputed, *args, **kwargs):
            return extract_directed_connectivity_from_precomputed(
                precomputed, config=config, logger=logger
            )

        _extract_precomputed_feature_group(
            "directed_connectivity",
            extract_directed_connectivity_wrapper,
            precomputed,
            logger,
            result,
        )

    if "ratios" in feature_groups:
        _extract_precomputed_feature_group(
            "ratios",
            extract_band_ratios_from_precomputed,
            precomputed,
            logger,
            result,
            config,
        )

    if "pac" in feature_groups:
        _extract_precomputed_feature_group(
            "pac", extract_pac_from_precomputed, precomputed, logger, result, config
        )

    if "itpc" in feature_groups:
        n_jobs_itpc = get_n_jobs(config, default=-1, config_path="feature_engineering.parallel.n_jobs_itpc")
        _extract_precomputed_feature_group(
            "itpc", extract_itpc_from_precomputed, precomputed, logger, result, n_jobs=n_jobs_itpc
        )

    if "asymmetry" in feature_groups:
        n_jobs = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
        _extract_precomputed_feature_group(
            "asymmetry", extract_asymmetry_from_precomputed, precomputed, logger, result, n_jobs=n_jobs
        )

    if "complexity" in feature_groups:
        n_jobs = int(config.get("feature_engineering.parallel.n_jobs_complexity", -1))
        _extract_precomputed_feature_group(
            "complexity", extract_complexity_from_precomputed, precomputed, logger, result, n_jobs=n_jobs
        )

    if "quality" in feature_groups:
        logger.info("Computing trial-level quality metrics...")
        qual_df = compute_trial_quality_metrics(epochs, config)
        qual_cols = list(qual_df.columns)
        if not qual_df.empty:
            result.features["quality"] = FeatureSet(qual_df, qual_cols, "quality")
        else:
            result.qc["quality"] = {"skipped_reason": "empty_result"}

    return result

__all__ = [
    "extract_all_features",
    "extract_precomputed_features",
    "resolve_feature_categories",
]
