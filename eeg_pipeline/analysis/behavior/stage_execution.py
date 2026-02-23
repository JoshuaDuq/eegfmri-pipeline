from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


_STAGE_TO_ATTR_MAP = {
    "trial_table": "trial_table_path",
    "correlate_fdr": "correlations",
    "predictor_sensitivity": "predictor_sensitivity",
    "regression": "regression",
    "models": "models",
    "stability": "stability",
    "consistency": "consistency",
    "influence": "influence",
    "condition_column": "condition_effects",
    "condition_window": "condition_effects_window",
    "temporal_tfr": "tf",
    "temporal_stats": "temporal",
    "cluster": "cluster",
    "mediation": "mediation",
    "moderation": "moderation",
    "mixed_effects": "mixed_effects",
    "report": "report_path",
}


def update_results_from_stage_impl(results: Any, stage_name: str, output: Any) -> None:
    """Update BehaviorPipelineResults from stage output."""
    attr = _STAGE_TO_ATTR_MAP.get(stage_name)
    if attr and hasattr(results, attr):
        setattr(results, attr, output)


def log_stage_outcome_impl(
    logger: Any,
    stage_name: str,
    output: Any,
    elapsed: float,
    step_num: int,
    total_steps: int,
) -> None:
    """Log concise outcome for a completed behavior stage."""
    detail = ""
    if isinstance(output, pd.DataFrame) and not output.empty:
        detail = f" ({len(output)} rows, {output.shape[1]} cols)"
    elif isinstance(output, dict):
        n_keys = len(output)
        if n_keys > 0:
            detail = f" ({n_keys} outputs)"
    elif isinstance(output, (str, Path)) and output:
        detail = f" -> {Path(str(output)).name}"
    logger.info(
        "[%d/%d] \u2713 %s%s (%.1fs)",
        step_num,
        total_steps,
        stage_name,
        detail,
        elapsed,
    )


def run_selected_stages_impl(
    *,
    ctx: Any,
    config: Any,
    selected_stages: List[str],
    stage_registry: Any,
    stage_runners: Dict[str, Callable[[Any, Any, Dict[str, Any]], Any]],
    is_stage_enabled_by_config_fn: Callable[[str, Any], bool],
    update_results_from_stage_fn: Callable[[Any, str, Any], None],
    log_stage_outcome_fn: Callable[[Any, str, Any, float, int, int], None],
    results: Optional[Any] = None,
    progress: Optional[Any] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute selected stages in dependency order using the stage registry."""
    resolved = stage_registry.auto_resolve_stages(selected_stages)

    enabled_stages = [s for s in resolved if is_stage_enabled_by_config_fn(s, ctx.config)]

    skipped = set(resolved) - set(enabled_stages)
    if skipped:
        ctx.logger.info("Auto-skipped stages (disabled by config): %s", ", ".join(skipped))

    ctx.logger.info("Running %d stages: %s", len(enabled_stages), ", ".join(enabled_stages))

    if dry_run:
        return stage_registry.dry_run(enabled_stages)

    outputs: Dict[str, Any] = {}
    progress_steps = stage_registry.compute_progress_steps(enabled_stages)

    def _run_stage(name: str) -> Any:
        runner = stage_runners.get(name)
        if runner is None:
            raise KeyError(f"Stage '{name}' has no implementation in STAGE_RUNNERS")
        return runner(ctx, config, outputs)

    import time as _time

    for step in progress_steps:
        stage_name = step["name"]

        if progress is not None:
            progress.step(stage_name, current=step["index"] + 1, total=step["total"])

        t0 = _time.perf_counter()
        try:
            output = _run_stage(stage_name)
            stage_elapsed = _time.perf_counter() - t0
            outputs[stage_name] = output

            if results is not None:
                update_results_from_stage_fn(results, stage_name, output)

            log_stage_outcome_fn(
                ctx.logger,
                stage_name,
                output,
                stage_elapsed,
                step["index"] + 1,
                step["total"],
            )
        except Exception as exc:
            ctx.logger.error("Stage '%s' failed after %.1fs: %s", stage_name, _time.perf_counter() - t0, exc)
            raise

    return outputs


def run_behavior_stages_impl(
    *,
    ctx: Any,
    pipeline_config: Any,
    config_to_stage_names_fn: Callable[[Any], List[str]],
    run_selected_stages_fn: Callable[..., Dict[str, Any]],
    results: Optional[Any] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run behavior pipeline stages based on pipeline config."""
    stages = config_to_stage_names_fn(pipeline_config)
    return run_selected_stages_fn(ctx, pipeline_config, stages, results, progress)
