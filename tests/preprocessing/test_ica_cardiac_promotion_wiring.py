"""The cardiac review's detections must reach the file that decides the exclusions.

``ctps_promotions`` and ``promote_exclusions`` are covered on their own in
``test_ica_exclusion_promotion.py``. What is left, and what these cover, is the wiring
inside ``generate_ica_cardiac_review``: that promotion happens before the evidence table is
built, that the report is redrawn against the exclusions it just added rather than the ones
it was opened with, and that the redraw does not recurse.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.ica_cardiac_report import generate_ica_cardiac_review
from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings
from eeg_pipeline.preprocessing.ica_exclusions import (
    read_component_statuses,
    reviewed_exclusions,
)
from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION

SFREQ = 250.0
DURATION_S = 120.0
BEAT_INTERVAL_S = 1.0  # 60 bpm, inside the plausible range the QC enforces
CHANNELS = ["Fz", "Cz", "Pz", "C3", "C4", "Oz"]


def _cardiac_raw(seed: int) -> mne.io.BaseRaw:
    """EEG carrying one strongly R-locked component, plus the ECG and R markers.

    The markers are what ``detect_ecg_events`` prefers, so the beat train is exact and the
    test does not depend on a peak detector finding a synthetic QRS.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(DURATION_S * SFREQ)
    times = np.arange(n_samples) / SFREQ
    beat_times = np.arange(1.0, DURATION_S - 1.0, BEAT_INTERVAL_S)

    # A biphasic deflection locked to every beat, projected onto a fixed topography.
    pulse = np.zeros(n_samples)
    for onset in beat_times:
        start = int(onset * SFREQ)
        width = int(0.12 * SFREQ)
        shape = np.sin(np.linspace(0, 2 * np.pi, width))
        pulse[start : start + width] += shape
    topography = np.array([0.9, 1.0, 0.6, 0.3, 0.35, 0.1])

    eeg = rng.normal(scale=1.0, size=(len(CHANNELS), n_samples))
    eeg += np.outer(topography, pulse) * 12.0
    # A second structured source, so the decomposition has something that is not cardiac.
    eeg += np.outer([0.1, 0.2, 0.4, 0.9, 0.85, 0.3], np.sin(2 * np.pi * 9.0 * times)) * 4.0

    ecg = pulse * 40.0 + rng.normal(scale=0.5, size=n_samples)

    info = mne.create_info(
        CHANNELS + ["ECG"],
        SFREQ,
        ["eeg"] * len(CHANNELS) + ["ecg"],
    )
    data = np.vstack([eeg * 1e-6, ecg[None, :] * 1e-6])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=beat_times,
            duration=np.zeros_like(beat_times),
            description=[PULSE_MARKER_DESCRIPTION] * len(beat_times),
        )
    )
    return raw


@pytest.fixture
def review_inputs(tmp_path) -> dict:
    """Two filtered runs, a fitted ICA, an all-good component table and a report."""
    prefix = "sub-0001"
    raw_paths = []
    raws = []
    for run in (1, 2):
        raw = _cardiac_raw(seed=run)
        path = tmp_path / f"{prefix}_task-thermalactive_run-{run}_proc-filt_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
        raw_paths.append(path)
        raws.append(raw)

    ica = mne.preprocessing.ICA(n_components=4, random_state=0, max_iter=1000)
    ica.fit(mne.concatenate_raws([r.copy() for r in raws], verbose="ERROR"), verbose="ERROR")
    ica_path = tmp_path / f"{prefix}_proc-ica_ica.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")

    components_path = tmp_path / f"{prefix}_proc-ica_components.tsv"
    import pandas as pd

    pd.DataFrame(
        {
            "component": np.arange(ica.n_components_),
            "type": ["ica"] * ica.n_components_,
            "description": ["Independent Component"] * ica.n_components_,
            "status": ["good"] * ica.n_components_,
            "status_description": ["n/a"] * ica.n_components_,
        }
    ).to_csv(components_path, sep="\t", index=False)

    # The cardiac section is inserted ahead of upstream's component review, so the report
    # has to carry that anchor exactly as MNE-BIDS-Pipeline leaves it.
    report = mne.Report(title="sub-0001")
    report.add_html(
        "<p>placeholder</p>",
        title="ICA components",
        section="ICA: components",
        tags=("ica", "ica-component-review"),
    )
    report_path = tmp_path / f"{prefix}_report.h5"
    report.save(report_path, overwrite=True, verbose="ERROR")

    return {
        "filtered_raw_paths": raw_paths,
        "ica_path": ica_path,
        "components_path": components_path,
        "report_path": report_path,
        "output_path": tmp_path / f"{prefix}_desc-icaecg_components.tsv",
        "component_count": int(ica.n_components_),
    }


def _settings(**overrides) -> CardiacReviewSettings:
    return CardiacReviewSettings(
        enabled=True,
        ecg_channel="ECG",
        epoch_window=(-0.4, 0.6),
        baseline=(-0.4, -0.1),
        measurement_window=(0.0, 0.4),
        ctps_threshold="auto",
        representative_window_seconds=10.0,
        marker_description="Pulse Artifact/R",
        **overrides,
    )


def test_a_cardiac_component_is_written_into_the_component_table(review_inputs) -> None:
    generate_ica_cardiac_review(
        filtered_raw_paths=review_inputs["filtered_raw_paths"],
        ica_path=review_inputs["ica_path"],
        report_path=review_inputs["report_path"],
        output_path=review_inputs["output_path"],
        settings=_settings(promote_exclusions=True, promotion_minimum_run_fraction=0.5),
    )

    excluded = reviewed_exclusions(
        review_inputs["components_path"], component_count=review_inputs["component_count"]
    )
    assert excluded, "The R-locked component was detected but never reached the table."

    statuses = read_component_statuses(
        review_inputs["components_path"], component_count=review_inputs["component_count"]
    )
    description = statuses.loc[excluded[0], "status_description"]
    assert "full-recording CTPS" in description
    assert "run-1" in description


def test_review_only_is_the_default_and_changes_no_exclusion(review_inputs) -> None:
    """The knob is what makes this change the cleaned data; without it the review is
    evidence, exactly as its docstring has always promised."""
    before = review_inputs["components_path"].read_bytes()

    generate_ica_cardiac_review(
        filtered_raw_paths=review_inputs["filtered_raw_paths"],
        ica_path=review_inputs["ica_path"],
        report_path=review_inputs["report_path"],
        output_path=review_inputs["output_path"],
        settings=_settings(),
    )

    assert review_inputs["components_path"].read_bytes() == before


def test_the_evidence_table_shows_the_promoted_status_not_the_pre_promotion_one(
    review_inputs,
) -> None:
    """The evidence a reviewer reads has to describe the exclusions that were just made,
    or the report contradicts the derivative it documents."""
    import pandas as pd

    generate_ica_cardiac_review(
        filtered_raw_paths=review_inputs["filtered_raw_paths"],
        ica_path=review_inputs["ica_path"],
        report_path=review_inputs["report_path"],
        output_path=review_inputs["output_path"],
        settings=_settings(promote_exclusions=True, promotion_minimum_run_fraction=0.5),
    )

    excluded = reviewed_exclusions(
        review_inputs["components_path"], component_count=review_inputs["component_count"]
    )
    evidence = pd.read_csv(review_inputs["output_path"], sep="\t")
    promoted_rows = evidence[evidence["component"].isin(excluded)]
    assert (promoted_rows["current_ica_status"] == "bad").all()


def test_rerunning_the_review_promotes_nothing_further(review_inputs) -> None:
    settings = _settings(promote_exclusions=True, promotion_minimum_run_fraction=0.5)
    call = dict(
        filtered_raw_paths=review_inputs["filtered_raw_paths"],
        ica_path=review_inputs["ica_path"],
        report_path=review_inputs["report_path"],
        output_path=review_inputs["output_path"],
        settings=settings,
    )

    generate_ica_cardiac_review(**call)
    after_first = review_inputs["components_path"].read_bytes()
    generate_ica_cardiac_review(**call)

    assert review_inputs["components_path"].read_bytes() == after_first


def test_promotion_cannot_tell_a_cleared_component_from_an_unreviewed_one(
    review_inputs,
) -> None:
    """Pinning the reason the pipeline gates this on ``ica.manual_review_complete``.

    A ``good`` row means "not excluded"; nothing in the table distinguishes a component
    nobody has looked at from one a reviewer deliberately cleared. So promotion re-applies
    here, and the caller is responsible for not running it over a signed-off review —
    which ``PreprocessingPipeline._run_ica_cardiac_review`` enforces.
    """
    import pandas as pd

    call = dict(
        filtered_raw_paths=review_inputs["filtered_raw_paths"],
        ica_path=review_inputs["ica_path"],
        report_path=review_inputs["report_path"],
        output_path=review_inputs["output_path"],
        settings=_settings(promote_exclusions=True, promotion_minimum_run_fraction=0.5),
    )
    generate_ica_cardiac_review(**call)

    statuses = pd.read_csv(review_inputs["components_path"], sep="\t")
    cleared = int(statuses.loc[statuses["status"] == "bad", "component"].iloc[0])
    statuses.loc[statuses["component"] == cleared, "status"] = "good"
    statuses.to_csv(review_inputs["components_path"], sep="\t", index=False)

    generate_ica_cardiac_review(**call)

    final = read_component_statuses(
        review_inputs["components_path"], component_count=review_inputs["component_count"]
    )
    assert final.loc[cleared, "status"] == "bad"


def test_a_subject_with_bad_channels_still_renders_the_review(tmp_path) -> None:
    """Regression: the before/after evidence must span the ICA's channels, not the cap.

    ``picks="eeg"`` keeps bad channels, which are absent from the decomposition and which
    ``ICA.apply`` returns untouched. The evoked array then has more rows than ``ica.info``
    has channels, and drawing a topography from it raised
    "Number of channels in the Info object (60) and the data array (63) do not match" —
    aborting the whole cohort at the review stage for every subject with a bad channel.
    """
    import pandas as pd

    prefix = "sub-0002"
    raw_paths = []
    raws = []
    for run in (1, 2):
        raw = _cardiac_raw(seed=run + 10)
        # Exactly the shape of the real failure: PyPREP marked channels, so the ICA spans
        # fewer channels than the recording carries.
        raw.info["bads"] = ["C4", "Oz"]
        path = tmp_path / f"{prefix}_task-thermalactive_run-{run}_proc-filt_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
        raw_paths.append(path)
        raws.append(raw)

    ica = mne.preprocessing.ICA(n_components=3, random_state=0, max_iter=1000)
    ica.fit(mne.concatenate_raws([r.copy() for r in raws], verbose="ERROR"), verbose="ERROR")
    assert len(ica.ch_names) == len(CHANNELS) - 2, "fixture must exclude the bad channels"

    ica_path = tmp_path / f"{prefix}_proc-ica_ica.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")
    components_path = tmp_path / f"{prefix}_proc-ica_components.tsv"
    pd.DataFrame(
        {
            "component": np.arange(ica.n_components_),
            "type": ["ica"] * ica.n_components_,
            "description": ["Independent Component"] * ica.n_components_,
            "status": ["good"] * ica.n_components_,
            "status_description": ["n/a"] * ica.n_components_,
        }
    ).to_csv(components_path, sep="\t", index=False)

    report = mne.Report(title=prefix)
    report.add_html(
        "<p>placeholder</p>",
        title="ICA components",
        section="ICA: components",
        tags=("ica", "ica-component-review"),
    )
    report_path = tmp_path / f"{prefix}_report.h5"
    report.save(report_path, overwrite=True, verbose="ERROR")

    written = generate_ica_cardiac_review(
        filtered_raw_paths=raw_paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / f"{prefix}_desc-icaecg_components.tsv",
        settings=_settings(promote_exclusions=True, promotion_minimum_run_fraction=0.5),
    )

    assert Path(written).is_file()
