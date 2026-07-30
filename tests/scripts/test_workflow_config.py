"""Path resolution for the paradigm workflow folders.

These scripts used to carry absolute ``/Volumes/KINGSTON/...`` constants. The point of the
workflow config is that the drive is named once, in the core config, and everything else
inherits it -- so the tests that matter are about *precedence*, not about any one value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from studies.pain_study.scripts.workflow_config import (
    INHERITED_ROOTS,
    WorkflowConfig,
    load_workflow_config,
)


class FakeCore:
    """Stands in for the core ConfigDict: dotted lookups over a flat mapping."""

    def __init__(self, values: dict):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


CORE = FakeCore(
    {
        "paths.bids_root": "/core/bids",
        "paths.deriv_root": "/core/derivatives",
        "paths.source_data": "/core/source",
    }
)


def _config(data: dict) -> WorkflowConfig:
    return WorkflowConfig(name="line_comb", source=None, data=data, core=CORE)


class TestPrecedence:
    def test_a_cli_override_wins_over_everything(self) -> None:
        config = _config({"paths": {"bids_root": "/from/yaml"}})

        assert config.path("bids_root", override="/from/flag") == Path("/from/flag")

    def test_the_workflow_yaml_wins_over_the_core_config(self) -> None:
        config = _config({"paths": {"bids_root": "/from/yaml"}})

        assert config.path("bids_root") == Path("/from/yaml")

    def test_a_null_root_inherits_the_core_config(self) -> None:
        config = _config({"paths": {"bids_root": None}})

        assert config.path("bids_root") == Path("/core/bids")

    def test_an_absent_root_inherits_the_core_config(self) -> None:
        config = _config({"paths": {}})

        assert config.path("bids_root") == Path("/core/bids")


class TestPlaceholders:
    @pytest.mark.parametrize("root", INHERITED_ROOTS)
    def test_every_inherited_root_can_be_referenced(self, root: str) -> None:
        config = _config({"paths": {"derived": f"<{root}>/sub"}})

        assert config.path("derived") == Path(f"{CORE.get(f'paths.{root}')}/sub")

    def test_a_placeholder_follows_a_workflow_level_override(self) -> None:
        """Overriding a root must move everything derived from it, not just the root."""
        config = _config(
            {"paths": {"deriv_root": "/elsewhere", "derived": "<deriv_root>/preprocessed/eeg"}}
        )

        assert config.path("derived") == Path("/elsewhere/preprocessed/eeg")

    def test_a_relative_path_stays_relative_to_the_working_directory(self) -> None:
        """``outputs/...`` is cited by name in the docs and must not become absolute."""
        config = _config({"paths": {"report": "outputs/line_comb_removal"}})

        assert config.path("report") == Path("outputs/line_comb_removal")


class TestFailures:
    def test_an_unknown_path_names_the_file_it_was_not_found_in(self) -> None:
        config = WorkflowConfig(name="line_comb", source=Path("/x/config.yaml"), data={}, core=CORE)

        with pytest.raises(KeyError, match="config.yaml"):
            config.path("nope")

    def test_asking_root_for_a_non_root_is_refused(self) -> None:
        with pytest.raises(KeyError, match="not an inherited root"):
            _config({}).root("diagnosis_dir")


class TestSettings:
    def test_a_dotted_setting_reads_from_the_workflow_yaml(self) -> None:
        config = _config({"line_comb_removal": {"notch_width_ratio": 450}})

        assert config.get("line_comb_removal.notch_width_ratio") == 450

    def test_a_missing_setting_falls_back_to_the_core_config(self) -> None:
        config = _config({})

        assert config.get("paths.bids_root") == "/core/bids"


class TestShippedConfigs:
    """The files that actually ship must resolve, or every stage fails at startup."""

    @pytest.mark.parametrize(
        "workflow,keys",
        [
            (
                "line_comb",
                ["bids_root", "source_data", "preprocessed_eeg", "output_root", "diagnosis_dir"],
            ),
            (
                "cardiac_gaps",
                ["uncorrected_root", "corrected_root", "output_root", "report_dir"],
            ),
        ],
    )
    def test_every_declared_path_resolves(self, workflow: str, keys: list[str]) -> None:
        config = load_workflow_config(workflow, core_config=CORE)

        assert config.source is not None, f"{workflow}/config.yaml is missing"
        for key in keys:
            assert isinstance(config.path(key), Path)

    @pytest.mark.parametrize("workflow", ["line_comb", "cardiac_gaps"])
    def test_no_shipped_config_hardcodes_a_drive(self, workflow: str) -> None:
        """A drive path in a committed config is the failure this refactor removed."""
        path = (
            Path(__file__).resolve().parents[2]
            / "studies/pain_study/scripts"
            / workflow
            / "config.yaml"
        )
        declared = yaml.safe_load(path.read_text())["paths"]

        for key, value in declared.items():
            assert "/Volumes/" not in str(value), f"{workflow}/config.yaml pins a drive at {key}"
