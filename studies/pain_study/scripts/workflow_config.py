"""Configuration shared by the paradigm-specific workflow folders.

Each workflow folder that has settings of its own -- ``line_comb/`` and ``cardiac_gaps/``
-- carries a ``config.yaml`` next to the code it configures. Those files hold the numbers
only that workflow has an opinion about: which harmonics the comb removal touches, how
wide a notch costs, how many OBS components the correction fits.

Paths are deliberately *not* copied into them. ``bids_root``, ``source_data`` and
``deriv_root`` name the same three directories for every stage of every pipeline, and are
already answered by the core ``eeg_config.yaml``. A copy per folder would mean five files
to edit the day the drive changes and five chances for them to disagree -- which is the
failure the absolute ``/Volumes/...`` constants in these scripts already were. A workflow
config may still override a root when it genuinely needs a different one, but the default
is to inherit.

Resolution order, highest priority first:

1. a command-line flag
2. ``--config PATH``, or the ``EEG_PIPELINE_<WORKFLOW>_CONFIG`` environment variable
3. the workflow folder's own ``config.yaml``
4. ``paths.*`` in the core ``eeg_config.yaml`` (roots only)
5. the default written in the code

Path values may refer to a root with a placeholder -- ``"<deriv_root>/preprocessed/eeg"``
-- so a folder can name a subdirectory without repeating the drive. Relative paths stay
relative to the working directory, which is what the ``outputs/`` defaults have always
meant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent

#: Roots the core config already answers. A workflow inherits these unless it says otherwise.
INHERITED_ROOTS = ("bids_root", "deriv_root", "source_data")


def _env_var(workflow: str) -> str:
    return f"EEG_PIPELINE_{workflow.upper()}_CONFIG"


def _resolve_config_path(workflow: str, config_path: str | Path | None) -> Path:
    """Locate the YAML for ``workflow``: explicit path, then env var, then the folder's own."""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(_env_var(workflow))
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (SCRIPTS_DIR / workflow / "config.yaml").resolve()


@dataclass(frozen=True)
class WorkflowConfig:
    """The settings and paths one workflow folder runs on."""

    name: str
    source: Path | None
    data: dict[str, Any] = field(default_factory=dict)
    core: Any = None

    def get(self, key: str, default: Any = None) -> Any:
        """Read a dotted key out of the workflow's own YAML.

        Falls back to the core config so a study that has not yet moved a block out of
        ``eeg_config.yaml`` keeps working.
        """
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
        if self.core is not None:
            return self.core.get(key, default)
        return default

    def root(self, name: str) -> Path:
        """Resolve one of the inherited roots, honouring a workflow-level override."""
        if name not in INHERITED_ROOTS:
            raise KeyError(f"{name!r} is not an inherited root; use path() instead")
        override = (self.data.get("paths") or {}).get(name)
        if override:
            return Path(str(override)).expanduser()
        if self.core is None:
            raise KeyError(f"no core config available to resolve paths.{name}")
        value = self.core.get(f"paths.{name}")
        if not value:
            raise KeyError(f"paths.{name} is not set in the core config")
        return Path(str(value)).expanduser()

    def path(self, name: str, *, override: str | Path | None = None) -> Path:
        """Resolve a named path: CLI override, then the workflow YAML, then an inherited root.

        Placeholders of the form ``<deriv_root>`` are expanded against :meth:`root`, so a
        workflow can name a subdirectory without restating the drive.
        """
        if override is not None:
            return Path(override).expanduser()
        if name in INHERITED_ROOTS and not (self.data.get("paths") or {}).get(name):
            return self.root(name)
        raw = (self.data.get("paths") or {}).get(name)
        if raw is None:
            raise KeyError(f"paths.{name} is not set in {self.source or 'the workflow config'}")
        text = str(raw)
        for token in INHERITED_ROOTS:
            placeholder = f"<{token}>"
            if placeholder in text:
                text = text.replace(placeholder, str(self.root(token)))
        return Path(text).expanduser()


def load_workflow_config(
    workflow: str,
    config_path: str | Path | None = None,
    *,
    core_config: Any = None,
) -> WorkflowConfig:
    """Load ``workflow``'s YAML and pair it with the core config it inherits paths from."""
    resolved = _resolve_config_path(workflow, config_path)
    data: dict[str, Any] = {}
    if resolved.exists():
        with open(resolved, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    if core_config is None:
        from eeg_pipeline.utils.config.loader import load_config

        core_config = load_config()

    return WorkflowConfig(
        name=workflow,
        source=resolved if resolved.exists() else None,
        data=data,
        core=core_config,
    )
