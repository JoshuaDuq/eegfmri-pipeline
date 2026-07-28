"""The exclusions a report renders must be the ones that built the cleaned data."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.ica_exclusions import (
    components_path_for_ica,
    read_component_statuses,
    read_ica_with_reviewed_exclusions,
    reviewed_exclusions,
)


def _write_components(path, statuses: list[str]) -> None:
    pd.DataFrame(
        {
            "component": np.arange(len(statuses)),
            "status": statuses,
            "status_description": ["" for _ in statuses],
        }
    ).to_csv(path, sep="\t", index=False)


def _fitted_ica(tmp_path, n_components: int = 3):
    rng = np.random.default_rng(0)
    info = mne.create_info(["C3", "Cz", "C4", "Pz"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.normal(size=(4, 3000)) * 1e-5, info, verbose=False)
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")
    # A stale exclusion set persisted in the file, exactly what must not be trusted.
    ica.exclude = [2]
    ica_path = tmp_path / "sub-0001_proc-ica_ica.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")
    return ica_path


def test_exclusions_come_from_the_component_table_not_the_ica_file(tmp_path) -> None:
    ica_path = _fitted_ica(tmp_path)
    _write_components(components_path_for_ica(ica_path), ["bad", "good", "good"])

    ica = read_ica_with_reviewed_exclusions(ica_path)

    # The file said [2]; the reviewed table says component 0. The table wins.
    assert ica.exclude == [0]


def test_an_empty_review_excludes_nothing_rather_than_the_stale_file_set(tmp_path) -> None:
    ica_path = _fitted_ica(tmp_path)
    _write_components(components_path_for_ica(ica_path), ["good", "good", "good"])

    assert read_ica_with_reviewed_exclusions(ica_path).exclude == []


def test_a_missing_component_table_fails_rather_than_silently_excluding_nothing(
    tmp_path,
) -> None:
    ica_path = _fitted_ica(tmp_path)

    with pytest.raises(FileNotFoundError, match="component status table"):
        read_ica_with_reviewed_exclusions(ica_path)


def test_component_table_path_keeps_the_session_entities(tmp_path) -> None:
    ica_path = tmp_path / "sub-0001_ses-01_proc-ica_ica.fif"

    assert components_path_for_ica(ica_path).name == "sub-0001_ses-01_proc-ica_components.tsv"


def test_a_path_that_is_not_an_ica_solution_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="ICA path"):
        components_path_for_ica(tmp_path / "sub-0001_proc-icafit_epo.fif")


def test_unrecognized_status_values_are_rejected(tmp_path) -> None:
    ica_path = _fitted_ica(tmp_path)
    _write_components(components_path_for_ica(ica_path), ["bad", "maybe", "good"])

    with pytest.raises(ValueError, match="unrecognized status"):
        reviewed_exclusions(components_path_for_ica(ica_path), component_count=3)


def test_component_table_must_cover_every_component(tmp_path) -> None:
    ica_path = _fitted_ica(tmp_path)
    _write_components(components_path_for_ica(ica_path), ["bad", "good"])

    with pytest.raises(ValueError, match="invalid"):
        read_component_statuses(components_path_for_ica(ica_path), component_count=3)


def test_run_evidence_cleans_each_run_with_the_reviewed_exclusions(monkeypatch, tmp_path) -> None:
    """The per-run panels must clean with the exclusions the caller resolved.

    If they applied an empty set, every before/after panel would compare a run against an
    identical copy of itself and render that as a clean recording.
    """
    from eeg_pipeline.preprocessing.report import run_evidence
    from eeg_pipeline.preprocessing.report.settings import ReportSettings

    applied_exclusions = []

    class _FakeIca:
        exclude = [0, 3]

        def apply(self, inst, exclude=None, verbose=None):
            applied_exclusions.append(list(exclude))
            return inst

    info = mne.create_info(["C3", "Cz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((2, 500)), info, verbose=False)
    monkeypatch.setattr(mne.io, "read_raw_fif", lambda *_a, **_k: raw.copy())
    monkeypatch.setattr(run_evidence, "compute_run_spectra", lambda *_a, **_k: "spectra")
    monkeypatch.setattr(run_evidence, "compute_run_continuity", lambda *_a, **_k: "continuity")
    monkeypatch.setattr(run_evidence, "compute_rr_intervals", lambda *_a, **_k: None)
    monkeypatch.setattr(run_evidence, "measure_volume_timing", lambda *_a, **_k: None)

    run_path = tmp_path / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
    evidence = run_evidence.measure_runs(
        filtered_raw_paths=[run_path],
        ica=_FakeIca(),
        settings=ReportSettings(),
    )

    assert applied_exclusions == [[0, 3]]
    assert evidence.spectra == ["spectra"]
