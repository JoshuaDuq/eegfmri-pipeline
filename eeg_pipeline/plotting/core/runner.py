from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol


class PlotContext(Protocol):
    """Protocol for plot context objects with subdir and logger."""

    def subdir(self, name: str) -> Path:
        """Return a subdirectory path."""
        ...

    @property
    def logger(self) -> Any:
        """Return a logger instance."""
        ...


def safe_plot(
    ctx: PlotContext,
    saved_files: Dict[str, Path],
    name: str,
    subdir: str,
    filename: Optional[str],
    plot_func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Execute a plot function safely and record the output path.

    If filename is provided, the plot function is called with the constructed
    path as the first argument. Otherwise, the plot function is called directly
    and its return value (if any) is recorded in saved_files.

    Args:
        ctx: Context object with subdir() method and logger attribute.
        saved_files: Dictionary to record created plot file paths.
        name: Name identifier for this plot.
        subdir: Subdirectory name for file-based plots.
        filename: Optional filename for file-based plots.
        plot_func: Function to execute for plotting.
        *args: Positional arguments to pass to plot_func.
        **kwargs: Keyword arguments to pass to plot_func.
    """
    try:
        if filename is not None:
            output_path = ctx.subdir(subdir) / filename
            plot_func(output_path, *args, **kwargs)
            saved_files[name] = output_path
            ctx.logger.info(f"Created: {name}")
        else:
            plot_result = plot_func(*args, **kwargs)
            if plot_result:
                if isinstance(plot_result, (str, Path)):
                    saved_files[name] = Path(plot_result)
                elif isinstance(plot_result, dict):
                    saved_files.update(plot_result)
            ctx.logger.info(f"Executed: {name}")
    except Exception as exc:
        ctx.logger.warning(f"Failed to create {name}: {exc}")
