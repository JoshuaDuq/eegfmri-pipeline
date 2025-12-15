from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional


def safe_plot(
    ctx,
    saved_files: Dict[str, Path],
    name: str,
    subdir: str,
    filename: Optional[str],
    plot_func: Callable[..., Any],
    *args,
    **kwargs,
) -> None:
    try:
        if filename:
            path = ctx.subdir(subdir) / filename
            plot_func(path, *args, **kwargs)
            saved_files[name] = path
            ctx.logger.info(f"Created: {name}")
        else:
            res = plot_func(*args, **kwargs)
            if res:
                if isinstance(res, (str, Path)):
                    saved_files[name] = Path(res)
                elif isinstance(res, dict):
                    saved_files.update(res)
            ctx.logger.info(f"Executed: {name}")
    except Exception as e:
        ctx.logger.warning(f"Failed to create {name}: {e}")
