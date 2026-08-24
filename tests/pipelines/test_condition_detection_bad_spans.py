"""A trial type carrying a bad-span tag is not a task condition.

MNE treats ``/`` as a tag separator, so ``Trig_therm/T  1`` already matches
``BAD_restart/Trig_therm/T  1``. Admitting the compound name as its own condition selects
those events a second time, under a second label, from data marked bad. On this cohort it
reached ``conditions = ['BAD_restart/Trig_therm/T  1', 'Trig_therm/T  1']`` because the
exclusion list tested ``startswith("Bad")`` against a string spelled ``BAD_``.
"""

from __future__ import annotations

import inspect
import sys
from unittest.mock import Mock

import mne_bids
import numpy as np
import pytest

_REAL_IMPORTS = (
    "eeg_pipeline.pipelines.preprocessing",
    "eeg_pipeline.pipelines.base",
    "eeg_pipeline.pipelines.progress",
    "eeg_pipeline.utils.config.roots",
    "eeg_pipeline.utils.config.loader",
)


@pytest.fixture
def pipeline(tmp_path):
    preexisting = {name: sys.modules.get(name) for name in _REAL_IMPORTS}
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    from eeg_pipeline.utils.config.loader import load_config

    instance = object.__new__(PreprocessingPipeline)
    instance.config = load_config()
    instance.logger = Mock()
    instance.bids_root = tmp_path / "bids"
    instance.deriv_root = tmp_path / "deriv"
    yield instance
    for name, module in preexisting.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _write_events(bids_root, trial_types: list[str]) -> None:
    directory = bids_root / "sub-0001" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["onset\tduration\ttrial_type"]
    lines += [f"{i}.0\t0.001\t{t}" for i, t in enumerate(trial_types)]
    (directory / "sub-0001_task-thermalactive_run-1_events.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_a_bad_span_tag_is_not_admitted_as_its_own_condition(pipeline) -> None:
    pipeline.config["preprocessing.condition_preferred_prefixes"] = []
    _write_events(
        pipeline.bids_root,
        ["Trig_therm/T  1", "BAD_restart/Trig_therm/T  1", "Volume/V  1", "Pulse Artifact/R"],
    )

    assert pipeline._detect_conditions_from_bids("thermalactive") == ["Trig_therm/T  1"]


def test_the_rule_is_leading_anchored_exactly_as_mne_applies_it(pipeline) -> None:
    """A non-leading bad tag is kept, because MNE keeps it.

    ``events_from_annotations`` anchors its bad/edge lookahead at the start of the
    description, so ``Trig_therm/BAD_restart/T  1`` *does* become an event upstream.
    Excluding it here would leave epochs with no matching row in the events table —
    the same misalignment as admitting a leading bad tag, in the other direction.
    """
    pipeline.config["preprocessing.condition_preferred_prefixes"] = []
    _write_events(pipeline.bids_root, ["Trig_therm/T  1", "Trig_therm/BAD_restart/T  1"])

    assert pipeline._detect_conditions_from_bids("thermalactive") == [
        "Trig_therm/BAD_restart/T  1",
        "Trig_therm/T  1",
    ]


def test_a_configured_prefix_does_not_readmit_a_leading_bad_tag(pipeline) -> None:
    """The preferred-prefix branch must apply the same exclusion as the fallback."""
    pipeline.config["preprocessing.condition_preferred_prefixes"] = ["BAD_restart", "Trig_"]
    _write_events(pipeline.bids_root, ["Trig_therm/T  1", "BAD_restart/Trig_therm/T  1"])

    assert pipeline._detect_conditions_from_bids("thermalactive") == ["Trig_therm/T  1"]


def test_the_bad_edge_rule_is_read_from_mne_not_restated(pipeline) -> None:
    """Two hand-maintained copies of the same rule is how this bug happened.

    If MNE's default stops being readable, that must raise here rather than fall back to
    a guess — a silent fallback restores the duplicate-rule problem invisibly.
    """
    import re
    import mne

    from eeg_pipeline.pipelines.preprocessing import _mne_annotation_event_pattern

    expected = inspect.signature(mne.events_from_annotations).parameters["regexp"].default
    assert _mne_annotation_event_pattern().pattern == expected
    assert re.match(expected, "BAD_restart/Trig_therm/T  1") is None
    assert re.match(expected, "EDGE boundary") is None


def test_an_unreadable_mne_signature_raises_rather_than_guessing(monkeypatch) -> None:
    import mne

    from eeg_pipeline.pipelines import preprocessing as module

    module._mne_annotation_event_pattern.cache_clear()
    monkeypatch.setattr(mne, "events_from_annotations", lambda raw, **kwargs: None, raising=True)
    try:
        with pytest.raises(RuntimeError, match="MNE's signature has changed"):
            module._mne_annotation_event_pattern()
    finally:
        module._mne_annotation_event_pattern.cache_clear()


def test_ordinary_conditions_are_untouched(pipeline) -> None:
    pipeline.config["preprocessing.condition_preferred_prefixes"] = []
    _write_events(pipeline.bids_root, ["painful", "neutral", "Volume/V  1"])

    assert pipeline._detect_conditions_from_bids("thermalactive") == ["neutral", "painful"]


def test_analysis_metadata_columns_do_not_replace_bids_event_descriptions(pipeline) -> None:
    """Conditions must name the annotations MNE-BIDS creates, not arbitrary metadata."""
    directory = pipeline.bids_root / "sub-0001" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sub-0001_task-thermalactive_run-1_events.tsv").write_text(
        "onset\tduration\ttrial_type\tcondition\n"
        "0\t0\tStimulus\thigh-pain\n"
        "1\t0\tStimulus\tlow-pain\n",
        encoding="utf-8",
    )
    pipeline.config["preprocessing.condition_column"] = "condition"

    assert pipeline._detect_conditions_from_bids("thermalactive") == ["Stimulus"]


def test_ambiguous_bids_values_use_mne_bids_hierarchical_descriptions(pipeline) -> None:
    """Mirror the public annotation names produced by ``read_raw_bids``."""
    directory = pipeline.bids_root / "sub-0001" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sub-0001_task-thermalactive_run-1_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\n" "0\t0\tStimulus\t1\n" "1\t0\tStimulus\t2\n",
        encoding="utf-8",
    )

    assert pipeline._detect_conditions_from_bids("thermalactive") == [
        "Stimulus/1",
        "Stimulus/2",
    ]


def test_event_descriptions_are_taken_from_public_mne_bids_api(
    pipeline,
    monkeypatch,
) -> None:
    """Do not maintain a second, subtly different implementation of MNE-BIDS rules."""
    directory = pipeline.bids_root / "sub-0001" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    events_path = directory / "sub-0001_task-thermalactive_run-1_events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\tvalue\n0\t0\tnan\t1\n1\t0\tnan\t2\n",
        encoding="utf-8",
    )
    calls = []

    def annotation_kwargs(path, *, verbose):
        calls.append((path, verbose))
        return {"description": np.array(["nan/1", "nan/2"])}

    monkeypatch.setattr(mne_bids, "events_file_to_annotation_kwargs", annotation_kwargs)

    assert pipeline._detect_conditions_from_bids("thermalactive") == ["nan/1", "nan/2"]
    assert calls == [(events_path, "ERROR")]
