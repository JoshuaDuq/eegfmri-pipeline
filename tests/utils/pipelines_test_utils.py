from __future__ import annotations


class DotConfig(dict):
    """Config helper supporting dotted key access and attribute access."""

    def get(self, key, default=None):  # type: ignore[override]
        if isinstance(key, str) and "." in key:
            cur = self
            for part in key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur
        return super().get(key, default)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)


class DummyProgress:
    def start(self, *_args, **_kwargs):
        return None

    def step(self, *_args, **_kwargs):
        return None

    def subject_start(self, *_args, **_kwargs):
        return None

    def subject_done(self, *_args, **_kwargs):
        return None

    def complete(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class NoopProgress:
    def start(self, *_a, **_k):
        return None

    def complete(self, *_a, **_k):
        return None

    def error(self, *_a, **_k):
        return None

    def step(self, *_a, **_k):
        return None

    def subject_start(self, *_a, **_k):
        return None

    def subject_done(self, *_a, **_k):
        return None


class NoopBatchProgress:
    def __init__(self, subjects, logger, desc):
        self.subjects = subjects

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _tb):
        return False

    def start_subject(self, _subject):
        return 0.0

    def finish_subject(self, _subject, _start_time):
        return None


def make_mock_fitted_model(
    runs: int = 1,
    frames: int = 30,
    n_voxels: int = 64,
    compute_contrast=None,
):
    from types import SimpleNamespace
    import nibabel as nib
    import numpy as np
    import pandas as pd

    cols = ["cond_a", "constant"]
    designs = [pd.DataFrame(np.ones((frames, len(cols))), columns=cols) for _ in range(runs)]
    labels = [np.zeros(n_voxels) for _ in range(runs)]
    coefficients = np.zeros((len(cols), n_voxels))
    response = np.full((frames, n_voxels), 0.25)
    results = [{0.0: SimpleNamespace(theta=coefficients, Y=response)} for _ in range(runs)]

    class _Masker:
        def inverse_transform(self, series):
            series_array = np.asarray(series)
            if series_array.ndim == 1:
                data = series_array.reshape(4, 4, 4)
            else:
                data = series_array.T.reshape(4, 4, 4, -1)
            return nib.Nifti1Image(data.astype(np.float32), np.eye(4))

    default_compute = compute_contrast or (
        lambda *a, **k: nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    )

    return SimpleNamespace(
        design_matrices_=designs,
        labels_=labels,
        results_=results,
        masker_=_Masker(),
        compute_contrast=default_compute,
    )


def make_mock_run_meta(
    *,
    subject: str = "sub-0001",
    runs: int = 1,
    output_type: str = "z_score",
) -> dict:
    return {
        "output_type": output_type,
        "tr": 0.9,
        "analysis_space": "T1w",
        "confounds_strategy": "motion24+wmcsf",
        "confound_columns": ["trans_x", "white_matter"],
        "design_matrix_tsv_paths": [f"/tmp/{subject}_run-0{i+1}_design.tsv" for i in range(runs)],
        "included_bold_paths": [f"/tmp/{subject}_run-0{i+1}_bold.nii.gz" for i in range(runs)],
        "included_confounds_paths": [],
        "retained_frame_indices": [list(range(30)) for _ in range(runs)],
        "skipped_runs": [],
    }
