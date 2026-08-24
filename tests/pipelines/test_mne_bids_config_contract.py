"""The generated MNE-BIDS-Pipeline config must only set options that actually exist.

Asserting on substrings of the generated text proves the pipeline wrote what we meant, not
that MNE-BIDS-Pipeline understands it. An option that is renamed or removed upstream would
pass a substring test and then fail at runtime — or, worse, be ignored — so this checks the
emitted names against the real upstream schema.
"""

from __future__ import annotations

import ast
import sys
from unittest.mock import Mock

import pytest

_config = pytest.importorskip("mne_bids_pipeline._config")

#: Modules this file imports for real. ``tests/pipelines/test_pipeline_preprocessing.py``
#: exercises the same pipeline against stubbed dependencies, and stubbing only works on a
#: module that has not already been imported with its real ones bound at module level.
#: Leaving these cached would silently disarm that file's stubs depending on test order.
_REAL_IMPORTS = (
    "eeg_pipeline.pipelines.preprocessing",
    "eeg_pipeline.pipelines.base",
    "eeg_pipeline.pipelines.progress",
    "eeg_pipeline.utils.config.roots",
    "eeg_pipeline.utils.config.loader",
)


def _assigned_names(config_source: str) -> set[str]:
    module = ast.parse(config_source)
    return {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


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
    # The BIDS tree is empty here, so conditions cannot be auto-detected from events.
    instance.config["epochs.conditions"] = ["painful", "neutral"]

    yield instance

    for name, module in preexisting.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.mark.parametrize("task_is_rest", [False, True])
def test_every_generated_option_exists_upstream(pipeline, task_is_rest) -> None:
    source = pipeline._generate_mne_bids_config(
        "preprocessing/_06a1_fit_ica",
        subjects=["0001"],
        task=None if task_is_rest else "thermalactive",
        task_is_rest=task_is_rest,
    )

    known = {name for name in dir(_config) if not name.startswith("_")}
    unknown = sorted(_assigned_names(source) - known)
    assert unknown == [], (
        f"Generated config sets options MNE-BIDS-Pipeline does not define: {unknown}. "
        "Upstream renamed or removed them."
    )


def test_n_jobs_is_emitted_because_upstream_still_defaults_to_serial(pipeline) -> None:
    """Omitting ``n_jobs`` is not a neutral omission; it is a request for one core.

    MNE-BIDS-Pipeline runs the loop over subjects and runs serially at ``n_jobs = 1``,
    and hands that same 1 to the autoreject threshold search in ``_06a1_fit_ica`` and
    ``_09_ptp_reject``. A generated config that leaves the option out therefore still
    finishes, just on one core, which is why nothing caught it for so long. If upstream
    ever changes its default, this assertion is the notice that the emission below is no
    longer load-bearing.
    """
    assert _config.n_jobs == 1

    source = pipeline._generate_mne_bids_config(
        "preprocessing/_06a1_fit_ica",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
        n_jobs=4,
    )

    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    assert namespace["n_jobs"] == 4


def test_generated_config_is_valid_python_with_the_settings_iclabel_requires(
    pipeline,
) -> None:
    source = pipeline._generate_mne_bids_config(
        "preprocessing/_07_make_epochs",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
    )
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)

    # ICLabel is only valid on average-referenced data decomposed by extended infomax.
    assert namespace["eeg_reference"] == "average"
    assert namespace["ica_algorithm"] == "extended_infomax"
    # The criterion must not be a variance fraction. MNE resolves ``None`` to 0.999999,
    # which is its documented guard against rank-deficient whitening, so ``None`` is the
    # rank-safe choice rather than "every component of a rank-deficient matrix". A float
    # below that is a *variance* criterion, and it stops early exactly when artifact owns
    # the variance: on sub-0015, blink and cardiac held 85% of it and 0.99 fitted 22
    # components of a rank-62 recording, leaving 40 dimensions that ICA.apply restores
    # unmodified and no exclusion can reach.
    n_components = namespace["ica_n_components"]
    assert n_components is None or isinstance(n_components, int), (
        f"ica_n_components must be None or an explicit component count, got "
        f"{n_components!r}; a variance fraction collapses when artifact dominates."
    )


def test_generated_config_preserves_native_upstream_configuration_types(pipeline) -> None:
    """Lists and mappings must not become strings or lose values at the wrapper."""
    pipeline.config["eeg.reference"] = ["P9", "P10"]
    pipeline.config["eeg.eog_channels"] = {
        "default": ["Fp1", "Fp2"],
        "sub-0002": None,
    }
    pipeline.config["epochs.conditions"] = {
        "painful": "stimulus/thermal/painful",
        "neutral": "stimulus/thermal/neutral",
    }

    source = pipeline._generate_mne_bids_config(
        "preprocessing/_07_make_epochs",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
    )
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)

    assert namespace["eeg_reference"] == ["P9", "P10"]
    assert namespace["eog_channels"] == {
        "default": ["Fp1", "Fp2"],
        "sub-0002": None,
    }
    assert namespace["conditions"] == {
        "painful": "stimulus/thermal/painful",
        "neutral": "stimulus/thermal/neutral",
    }


def test_documented_ica_algorithm_is_the_single_generated_source(pipeline) -> None:
    pipeline.config["ica.algorithm"] = "picard-extended_infomax"

    source = pipeline._generate_mne_bids_config(
        "preprocessing/_06a1_fit_ica",
        subjects=["0001"],
        task="thermalactive",
        task_is_rest=False,
    )
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)

    assert namespace["ica_algorithm"] == "picard-extended_infomax"


def test_removed_ica_method_key_fails_loudly(pipeline) -> None:
    pipeline.config["ica.method"] = "fastica"

    with pytest.raises(ValueError, match=r"ica\.method.*ica\.algorithm"):
        pipeline._generate_mne_bids_config(
            "preprocessing/_06a1_fit_ica",
            subjects=["0001"],
            task="thermalactive",
            task_is_rest=False,
        )
