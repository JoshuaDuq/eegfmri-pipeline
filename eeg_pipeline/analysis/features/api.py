from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.analysis.features.selection import (
    resolve_feature_categories,
    resolve_precomputed_groups,
)
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.data.features import load_group_microstate_templates
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.utils.validation import validate_epochs
from eeg_pipeline.utils.progress import PipelineProgress
from eeg_pipeline.analysis.features.results import (
    FeatureSet,
    ExtractionResult,
    FeatureExtractionResult,
    combine_feature_groups,
)
from eeg_pipeline.analysis.features.precomputed.erds import extract_erds_from_precomputed
from eeg_pipeline.analysis.features.precomputed.spectral import extract_power_from_precomputed
from eeg_pipeline.analysis.features.precomputed.spectral import (
    extract_spectral_extras_from_precomputed,
    extract_segment_power_from_precomputed,
)
from eeg_pipeline.analysis.features.precomputed.gfp import extract_gfp_from_precomputed
from eeg_pipeline.analysis.features.precomputed.roi import extract_roi_features_from_precomputed
from eeg_pipeline.analysis.features.precomputed.temporal import extract_temporal_features_from_precomputed
from eeg_pipeline.analysis.features.precomputed.ratios import extract_band_ratios_from_precomputed
from eeg_pipeline.analysis.features.precomputed.asymmetry import extract_asymmetry_from_precomputed
from eeg_pipeline.analysis.features.dynamics import extract_dynamics_from_precomputed
from eeg_pipeline.analysis.features.aperiodic import (
    extract_aperiodic_features,
    extract_aperiodic_features_from_epochs,
)
from eeg_pipeline.analysis.features.cfc import (
    extract_all_cfc_features,
    extract_pac_from_precomputed,
)
from eeg_pipeline.analysis.features.complexity import (
    extract_dynamics_features,
    extract_complexity_from_precomputed,
)
from eeg_pipeline.analysis.features.connectivity import (
    extract_connectivity_features,
    extract_connectivity_from_precomputed,
)
from eeg_pipeline.analysis.features.microstates import (
    extract_microstate_features,
    extract_microstate_features_from_epochs,
)
from eeg_pipeline.analysis.features.phase import (
    compute_pac_comodulograms,
    extract_itpc_from_precomputed,
    extract_phase_features,
)
from eeg_pipeline.analysis.features.power import extract_power_features
from eeg_pipeline.analysis.features.precompute import precompute_data
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


def extract_all_features(
    ctx: FeatureContext,
    *,
    precomputed_groups_override: Optional[List[str]] = None,
) -> FeatureExtractionResult:
    power_bands = get_frequency_band_names(ctx.config)
    n_microstates = int(ctx.config.get("feature_engineering.microstates.n_states", 4))
    expected_n_trials = len(ctx.epochs)

    precomputed_data = None
    needs_precompute = any(
        c in ctx.feature_categories for c in ["precomputed", "cfc", "dynamics_advanced"]
    )
    if needs_precompute and ctx.ensure_precomputed():
        precomputed_data = ctx.precomputed

    def _check_length(name: str, df: Optional[pd.DataFrame]) -> None:
        if df is None or getattr(df, "empty", False):
            return
        if len(df) != expected_n_trials:
            raise ValueError(f"{name} length mismatch: {len(df)} vs {expected_n_trials}")

    def _resolve_precomputed_groups() -> List[str]:
        try:
            return resolve_precomputed_groups(ctx.config, precomputed_groups_override)
        except Exception as exc:
            ctx.logger.error(f"Precomputed groups unavailable: {exc}")
            return []

    validation = validate_epochs(ctx.epochs, ctx.config, logger=ctx.logger)
    if not validation.valid:
        ctx.logger.warning(f"Validation issues: {validation.issues}")
        if validation.critical:
            raise ValueError(f"Critical errors: {validation.critical}")

    results = FeatureExtractionResult()

    tfr_dependent_categories = {"power", "connectivity", "itpc", "pac"}
    needs_tfr = bool(tfr_dependent_categories & set(ctx.feature_categories))

    tfr_complex = None
    tfr_power = None
    tfr = None
    baseline_df = None
    baseline_cols: List[str] = []
    b_start, b_end = None, None

    if needs_tfr:
        if any(c in ctx.feature_categories for c in ["itpc", "pac"]):
            tfr_complex = compute_complex_tfr(ctx.epochs, ctx.config, ctx.logger)
            if tfr_complex is not None:
                ctx.tfr_complex = tfr_complex
                ctx.logger.info("Deriving power TFR from complex TFR...")
                tfr_power = tfr_complex.copy()
                tfr_power.data = np.abs(tfr_complex.data) ** 2
                tfr_power.comment = "derived_from_complex"

        tfr, baseline_df, baseline_cols, b_start, b_end = compute_tfr_for_subject(
            ctx.epochs,
            ctx.aligned_events,
            ctx.subject,
            ctx.task,
            ctx.config,
            ctx.deriv_root,
            ctx.logger,
            tfr_computed=tfr_power,
        )
        if tfr is None:
            raise ValueError("TFR computation failed; aborting feature extraction.")
        results.tfr = tfr
        results.baseline_df = baseline_df
        results.baseline_cols = baseline_cols

        ctx.results["tfr"] = tfr
        ctx.tfr = tfr
        ctx.baseline_df = baseline_df
        ctx.baseline_cols = baseline_cols

        if ctx.config.get("feature_engineering.save_tfr_with_sidecar", False):
            try:
                if tfr is not None and b_start is not None and b_end is not None:
                    baseline_mode = str(ctx.config.get("time_frequency_analysis.baseline_mode", "logratio"))
                    tfr_to_save = tfr.copy()
                    tfr_to_save.apply_baseline(baseline=(b_start, b_end), mode=baseline_mode)
                    tfr_to_save.comment = (
                        f"BASELINED:mode={baseline_mode};win=({b_start:.3f},{b_end:.3f})"
                    )
                    tfr_out = (
                        ctx.deriv_root
                        / f"sub-{ctx.subject}"
                        / "eeg"
                        / f"sub-{ctx.subject}_task-{ctx.task}_power_epo-tfr.h5"
                    )
                    save_tfr_with_sidecar(
                        tfr_to_save,
                        tfr_out,
                        (b_start, b_end),
                        baseline_mode,
                        ctx.logger,
                        ctx.config,
                    )
            except Exception as exc:
                ctx.logger.warning("Failed to save baselined TFR with sidecar: %s", exc)
    else:
        ctx.logger.info("Skipping TFR computation (not needed for requested feature categories)")

    progress = PipelineProgress(total=len(ctx.feature_categories), logger=ctx.logger, desc="Features")
    progress.start()

    if "power" in ctx.feature_categories:
        progress.step(message="Extracting power features...")
        pow_df, pow_cols = extract_power_features(ctx, power_bands)
        if pow_df is not None and not pow_df.empty and len(pow_df) != expected_n_trials:
            raise ValueError(f"Power length mismatch: {len(pow_df)} vs {expected_n_trials}")
        results.pow_df = pow_df
        results.pow_cols = pow_cols

    if "connectivity" in ctx.feature_categories:
        progress.step(message="Extracting connectivity features...")
        conn_df, conn_cols = extract_connectivity_features(ctx, power_bands)
        if conn_df is not None and not conn_df.empty and len(conn_df) != expected_n_trials:
            raise ValueError(f"Connectivity length mismatch: {len(conn_df)} vs {expected_n_trials}")
        results.conn_df = conn_df
        results.conn_cols = conn_cols

    if "microstates" in ctx.feature_categories:
        progress.step(message="Extracting microstate features...")
        use_fixed = bool(ctx.config.get("feature_engineering.microstates.use_fixed_templates", False))
        use_group = bool(ctx.config.get("feature_engineering.microstates.use_group_templates", False))

        if use_group and not use_fixed:
            group_templates, group_ch_names = load_group_microstate_templates(
                ctx.deriv_root, n_microstates, ctx.logger
            )
            if group_templates is not None:
                ctx.fixed_templates = group_templates
                ctx.fixed_template_ch_names = group_ch_names
                use_fixed = True
                ctx.logger.info("Using group-level microstate templates")

        ms_df, ms_cols, ms_templates = extract_microstate_features(ctx)
        _check_length("Microstates", ms_df)
        results.ms_df = ms_df
        results.ms_cols = ms_cols
        results.ms_templates = ms_templates

    if "aperiodic" in ctx.feature_categories:
        progress.step(message="Extracting aperiodic features...")
        aper_df, aper_cols, qc_payload = extract_aperiodic_features(ctx, power_bands)
        _check_length("Aperiodic", aper_df)
        results.aper_df = aper_df
        results.aper_cols = aper_cols
        results.aper_qc = qc_payload

    if "complexity" in ctx.feature_categories:
        progress.step(message="Extracting complexity features...")
        comp_df, comp_cols = extract_dynamics_features(ctx, power_bands)
        _check_length("Complexity", comp_df)
        results.comp_df = comp_df
        results.comp_cols = comp_cols

    if "itpc" in ctx.feature_categories or "phase" in ctx.feature_categories:
        progress.step(message="Extracting phase features...")
        phase_df, phase_cols = extract_phase_features(ctx, power_bands)
        _check_length("Phase", phase_df)
        results.phase_df = phase_df
        results.phase_cols = phase_cols

    if "pac" in ctx.feature_categories:
        progress.step(message="Computing PAC features...")
        if tfr_complex is None:
            tfr_complex = compute_complex_tfr(ctx.epochs, ctx.config, ctx.logger)
            if tfr_complex is not None:
                ctx.tfr_complex = tfr_complex

        if tfr_complex is not None:
            freq_min, freq_max, n_freqs, *_ = get_tfr_config(ctx.config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

            plateau_meta = ctx.windows.metadata.get("plateau") if ctx.windows is not None else None
            plateau_window = None
            if plateau_meta is not None and getattr(plateau_meta, "valid", False):
                plateau_window = (float(plateau_meta.start), float(plateau_meta.end))

            pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = compute_pac_comodulograms(
                tfr_complex,
                freqs,
                tfr_complex.times,
                ctx.epochs.info,
                ctx.config,
                ctx.logger,
                segment_name="plateau",
                segment_window=plateau_window,
            )
            _check_length("PAC trials", pac_trials_df)
            _check_length("PAC time-resolved", pac_time_df)
            results.pac_df = pac_df
            results.pac_phase_freqs = pac_phase_freqs
            results.pac_amp_freqs = pac_amp_freqs
            results.pac_trials_df = pac_trials_df
            results.pac_time_df = pac_time_df

    precomputed_result = None
    configured_groups = _resolve_precomputed_groups() if "precomputed" in ctx.feature_categories else []
    requested_precomputed_groups = configured_groups.copy()
    if "cfc" in ctx.feature_categories:
        requested_precomputed_groups.append("cfc")
    if "dynamics_advanced" in ctx.feature_categories:
        requested_precomputed_groups.append("dynamics_advanced")
    requested_precomputed_groups = list(dict.fromkeys(requested_precomputed_groups))

    def _ensure_precomputed_result() -> Optional[Any]:
        nonlocal precomputed_result
        if precomputed_result is not None or not requested_precomputed_groups:
            return precomputed_result
        precomputed_result = extract_precomputed_features(
            ctx.epochs,
            power_bands,
            ctx.config,
            ctx.logger,
            feature_groups=requested_precomputed_groups,
            n_plateau_windows=int(ctx.config.get("feature_engineering.erds.n_temporal_windows", 5)),
            events_df=ctx.aligned_events,
            precomputed=precomputed_data,
        )
        if ctx.precomputed is None and precomputed_result is not None:
            ctx.precomputed = precomputed_result.precomputed
        return precomputed_result

    if "precomputed" in ctx.feature_categories:
        progress.step(message="Extracting precomputed features...")
        precomputed_result = _ensure_precomputed_result()
        if precomputed_result is not None:
            precomputed_df, precomputed_cols = combine_feature_groups(precomputed_result, configured_groups)
            results.precomputed_df = precomputed_df
            results.precomputed_cols = precomputed_cols
            _check_length("Precomputed", precomputed_df)

    if "cfc" in ctx.feature_categories:
        progress.step(message="Extracting CFC features...")
        precomputed_result = _ensure_precomputed_result()
        if precomputed_result is not None:
            cfc_df = precomputed_result.get_feature_group_df("cfc")
            _check_length("CFC", cfc_df)
            if cfc_df is not None and not cfc_df.empty:
                cfc_fs = precomputed_result.features.get("cfc")
                if cfc_fs is not None:
                    results.cfc_df = cfc_df
                    results.cfc_cols = list(cfc_fs.columns)

    if "dynamics_advanced" in ctx.feature_categories:
        progress.step(message="Extracting advanced dynamics features...")
        precomputed_result = _ensure_precomputed_result()
        if precomputed_result is not None:
            dyn_df = precomputed_result.get_feature_group_df("dynamics_advanced")
            _check_length("Advanced dynamics", dyn_df)
            if dyn_df is not None and not dyn_df.empty:
                dyn_fs = precomputed_result.features.get("dynamics_advanced")
                if dyn_fs is not None:
                    results.dynamics_df = dyn_df
                    results.dynamics_cols = list(dyn_fs.columns)

    if "quality" in ctx.feature_categories:
        progress.step(message="Computing trial quality metrics...")
        qual_df, qual_cols = extract_quality_features(ctx)
        _check_length("Quality metrics", qual_df)
        results.quality_df = qual_df
        results.quality_cols = qual_cols

    progress.finish()
    return results


def extract_precomputed_features(
    epochs: "mne.Epochs",
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    feature_groups: Optional[List[str]] = None,
    events_df: Optional[pd.DataFrame] = None,
    n_plateau_windows: int = 5,
    precomputed: Optional[PrecomputedData] = None,
) -> ExtractionResult:
    from eeg_pipeline.utils.config.loader import get_config_value

    if feature_groups is None:
        feature_groups = ["erds", "spectral"]

    compute_bands = any(
        g in feature_groups
        for g in [
            "erds",
            "spectral",
            "connectivity",
            "pac",
            "dynamics_advanced",
            "roi",
            "ratios",
        ]
    )
    compute_psd_data = any(g in feature_groups for g in ["aperiodic", "spectral"])

    if precomputed is None:
        logger.info("Precomputing intermediate data (bands=%s, psd=%s)...", compute_bands, compute_psd_data)
        precomputed = precompute_data(
            epochs,
            bands,
            config,
            logger,
            compute_bands=compute_bands,
            compute_psd_data=compute_psd_data,
            n_plateau_windows=n_plateau_windows,
        )
    else:
        logger.info("Using provided precomputed intermediates")

    result = ExtractionResult(precomputed=precomputed)

    if events_df is not None and len(events_df) == precomputed.data.shape[0]:
        if "condition" in events_df.columns:
            result.condition = events_df["condition"].to_numpy()
        elif "trial_type" in events_df.columns:
            result.condition = events_df["trial_type"].to_numpy()

    if "erds" in feature_groups:
        logger.info("Extracting ERDS features...")
        df, cols, qc = extract_erds_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["erds"] = FeatureSet(df, cols, "erds")
            result.qc["erds"] = qc
        else:
            result.qc["erds"] = {"skipped_reason": "empty_result"}

    if "spectral" in feature_groups:
        logger.info("Extracting spectral power features...")
        n_jobs_spectral = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
        df, cols, qc = extract_power_from_precomputed(precomputed, bands)
        extra_df, extra_cols = extract_spectral_extras_from_precomputed(precomputed, bands, n_jobs=n_jobs_spectral)
        seg_df, seg_cols = extract_segment_power_from_precomputed(precomputed, bands, n_jobs=n_jobs_spectral)
        if not extra_df.empty:
            df = pd.concat([df, extra_df], axis=1) if not df.empty else extra_df
            cols.extend(extra_cols)
        if not seg_df.empty:
            df = pd.concat([df, seg_df], axis=1) if not df.empty else seg_df
            cols.extend(seg_cols)
        if not df.empty:
            result.features["spectral"] = FeatureSet(df, cols, "spectral")
            result.qc["spectral"] = qc
        else:
            result.qc["spectral"] = {"skipped_reason": "empty_result"}

    if "aperiodic" in feature_groups:
        logger.info("Extracting aperiodic features...")
        baseline_window = getattr(precomputed.windows, "baseline_range", None)
        if baseline_window is None or np.isnan(baseline_window).any():
            baseline_window = get_config_value(config, "time_frequency_analysis.baseline_window", [-3.0, -0.5])
            baseline_window = tuple(baseline_window)

        df, cols, qc = extract_aperiodic_features_from_epochs(
            epochs,
            baseline_window,
            bands,
            config,
            logger,
            events_df=events_df,
        )
        if not df.empty:
            result.features["aperiodic"] = FeatureSet(df, cols, "aperiodic")
            result.qc["aperiodic"] = qc
        else:
            result.qc["aperiodic"] = {"skipped_reason": "empty_result"}

    if "connectivity" in feature_groups:
        logger.info("Extracting connectivity features (PLV) from precomputed phases...")
        conn_df, conn_cols = extract_connectivity_from_precomputed(precomputed)
        if not conn_df.empty:
            result.features["connectivity"] = FeatureSet(conn_df, conn_cols, "connectivity")
        else:
            result.qc["connectivity"] = {"skipped_reason": "empty_result"}

    if "gfp" in feature_groups:
        logger.info("Extracting GFP features...")
        gfp_df, gfp_cols = extract_gfp_from_precomputed(precomputed, config)
        if not gfp_df.empty:
            result.features["gfp"] = FeatureSet(gfp_df, gfp_cols, "gfp")
        else:
            result.qc["gfp"] = {"skipped_reason": "empty_result"}

    if "roi" in feature_groups:
        logger.info("Extracting ROI-aggregated features...")
        roi_df, roi_cols = extract_roi_features_from_precomputed(precomputed, bands, config)
        if not roi_df.empty:
            result.features["roi"] = FeatureSet(roi_df, roi_cols, "roi")
        else:
            result.qc["roi"] = {"skipped_reason": "empty_result"}

    if "temporal" in feature_groups:
        logger.info("Extracting temporal (time-domain) features...")
        tmp_df, tmp_cols = extract_temporal_features_from_precomputed(precomputed, config)
        if not tmp_df.empty:
            result.features["temporal"] = FeatureSet(tmp_df, tmp_cols, "temporal")
        else:
            result.qc["temporal"] = {"skipped_reason": "empty_result"}

    if "ratios" in feature_groups:
        logger.info("Extracting band ratio features...")
        ratio_df, ratio_cols = extract_band_ratios_from_precomputed(precomputed, config)
        if not ratio_df.empty:
            result.features["ratios"] = FeatureSet(ratio_df, ratio_cols, "ratios")
        else:
            result.qc["ratios"] = {"skipped_reason": "empty_result"}

    if "microstates" in feature_groups:
        logger.info("Extracting microstate features...")
        n_states = int(config.get("feature_engineering.microstates.n_states", 4))
        ms_df, ms_cols, _ = extract_microstate_features_from_epochs(
            epochs,
            n_states,
            config,
            logger,
        )
        if not ms_df.empty:
            result.features["microstates"] = FeatureSet(ms_df, ms_cols, "microstates")
        else:
            result.qc["microstates"] = {"skipped_reason": "empty_result"}

    if "pac" in feature_groups:
        logger.info("Extracting PAC features from precomputed analytic signals...")
        pac_df, pac_cols = extract_pac_from_precomputed(precomputed, config)
        if not pac_df.empty:
            result.features["pac"] = FeatureSet(pac_df, pac_cols, "pac")
        else:
            result.qc["pac"] = {"skipped_reason": "empty_result"}

    if "cfc" in feature_groups:
        logger.info("Extracting CFC features...")
        cfc_df, cfc_cols = extract_all_cfc_features(
            epochs,
            config,
            logger,
            include_pac=True,
            include_aac=True,
            include_ppc=True,
        )
        if not cfc_df.empty:
            result.features["cfc"] = FeatureSet(cfc_df, cfc_cols, "cfc")
        else:
            result.qc["cfc"] = {"skipped_reason": "empty_result"}

    if "dynamics_advanced" in feature_groups:
        logger.info("Extracting advanced dynamics from precomputed data...")
        n_jobs_dynamics = int(config.get("feature_engineering.parallel.n_jobs_temporal", -1))
        dyn_df, dyn_cols = extract_dynamics_from_precomputed(precomputed, n_jobs=n_jobs_dynamics)
        if not dyn_df.empty:
            result.features["dynamics_advanced"] = FeatureSet(dyn_df, dyn_cols, "dynamics_advanced")
        else:
            result.qc["dynamics_advanced"] = {"skipped_reason": "empty_result"}

    if "itpc" in feature_groups:
        logger.info("Extracting ITPC from precomputed phases...")
        itpc_df, itpc_cols = extract_itpc_from_precomputed(precomputed)
        if not itpc_df.empty:
            result.features["itpc"] = FeatureSet(itpc_df, itpc_cols, "itpc")
        else:
            result.qc["itpc"] = {"skipped_reason": "empty_result"}

    if "asymmetry" in feature_groups:
        logger.info("Computing hemispheric asymmetry features...")
        n_jobs_asym = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
        asym_df, asym_cols = extract_asymmetry_from_precomputed(precomputed, n_jobs=n_jobs_asym)
        if not asym_df.empty:
            result.features["asymmetry"] = FeatureSet(asym_df, asym_cols, "asymmetry")
        else:
            result.qc["asymmetry"] = {"skipped_reason": "empty_result"}

    if "complexity" in feature_groups:
        logger.info("Computing complexity metrics (LZC, entropy, DFA-lite)...")
        n_jobs_complexity = int(config.get("feature_engineering.parallel.n_jobs_complexity", -1))
        comp_df, comp_cols = extract_complexity_from_precomputed(precomputed, n_jobs=n_jobs_complexity)
        if not comp_df.empty:
            result.features["complexity"] = FeatureSet(comp_df, comp_cols, "complexity")
        else:
            result.qc["complexity"] = {"skipped_reason": "empty_result"}

    if "quality" in feature_groups:
        logger.info("Computing trial-level quality metrics...")
        qual_df = compute_trial_quality_metrics(epochs, config, logger)
        qual_cols = list(qual_df.columns)
        if not qual_df.empty:
            result.features["quality"] = FeatureSet(qual_df, qual_cols, "quality")
        else:
            result.qc["quality"] = {"skipped_reason": "empty_result"}

    return result


def extract_fmri_prediction_features(
    epochs: "mne.Epochs",
    config: Any,
    logger: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> ExtractionResult:
    feature_groups = ["spectral", "aperiodic", "erds"]
    bands = get_frequency_band_names(config)

    return extract_precomputed_features(
        epochs,
        bands,
        config,
        logger,
        feature_groups=feature_groups,
        events_df=events_df,
    )


__all__ = [
    "extract_all_features",
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    "resolve_feature_categories",
]
