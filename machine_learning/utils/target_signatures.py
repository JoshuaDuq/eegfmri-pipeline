from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

###################################################################
# Data Structures
###################################################################

@dataclass
class SubjectDataset:
    subject: str
    data: pd.DataFrame
    feature_columns: List[str]
    dropped_trials: List[Dict[str, float]]
    target_column: str


###################################################################
# Target Signature Definitions
###################################################################

@dataclass(frozen=True)
class TrialFileSpec:
    subdir_parts: Tuple[str, ...]
    filename: str
    score_columns: Tuple[str, ...]

    def candidate_path(self, root: Path, subject: str) -> Path:
        return root.joinpath(*self.subdir_parts, subject, self.filename)


@dataclass(frozen=True)
class TargetSignature:
    key: str
    display_name: str
    short_name: str
    score_column: str
    fmri_output_subdirs: Tuple[Tuple[str, ...], ...]
    trial_sources: Tuple[TrialFileSpec, ...]
    description: str

    def resolve_trial_file(self, root: Path, subject: str) -> Tuple[Path, str]:
        for spec in self.trial_sources:
            path = spec.candidate_path(root, subject)
            if not path.exists():
                continue
            try:
                available_columns = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
            except Exception:
                available_columns = []
            for column in spec.score_columns:
                if column in available_columns:
                    return path, column
        searched = [
            (spec.candidate_path(root, subject), spec.score_columns) for spec in self.trial_sources
        ]
        search_summary = "; ".join(f"{path} (columns {cols})" for path, cols in searched)
        raise FileNotFoundError(
            f"Could not locate trial-level scores for subject {subject} matching target '{self.key}'. "
            f"Searched: {search_summary}"
        )


TARGET_SIGNATURES: Dict[str, TargetSignature] = {
    "nps": TargetSignature(
        key="nps",
        display_name="Neurologic Pain Signature (NPS)",
        short_name="NPS",
        score_column="nps_score",
        fmri_output_subdirs=(
            ("NPS", "outputs"),
            ("NPS", "data_percent_signal_false", "outputs"),
        ),
        trial_sources=(
            TrialFileSpec(
                subdir_parts=("signatures", "nps", "scores"),
                filename="trial_nps.tsv",
                score_columns=("nps_score", "br_score"),
            ),
            TrialFileSpec(
                subdir_parts=("nps_scores",),
                filename="trial_br.tsv",
                score_columns=("br_score",),
            ),
        ),
        description="Trial-wise Neurologic Pain Signature beta values derived from fMRI.",
    ),
    "siips1": TargetSignature(
        key="siips1",
        display_name="Stimulus Intensity Independent Pain Signature (SIIPS1)",
        short_name="SIIPS1",
        score_column="siips1_score",
        fmri_output_subdirs=(
            ("NPS", "outputs"),
            ("NPS", "data_percent_signal_false", "outputs"),
            ("SIIPS1", "outputs"),
        ),
        trial_sources=(
            TrialFileSpec(
                subdir_parts=("signatures", "siips1", "scores"),
                filename="trial_siips1.tsv",
                score_columns=("siips1_score",),
            ),
        ),
        description="Trial-wise SIIPS1 beta values capturing expectancy-related pain modulation.",
    ),
}

DEFAULT_TARGET_KEY = "nps"


def get_target_signature(key: Optional[str]) -> TargetSignature:
    resolved_key = (key or DEFAULT_TARGET_KEY).lower()
    if resolved_key not in TARGET_SIGNATURES:
        valid = ", ".join(sorted(TARGET_SIGNATURES))
        raise ValueError(f"Unsupported target signature '{key}'. Available options: {valid}.")
    return TARGET_SIGNATURES[resolved_key]


def resolve_fmri_outputs_root(repo_root: Path, target: TargetSignature) -> Path:
    fmri_root = repo_root / "fmri_pipeline"
    candidates = [fmri_root.joinpath(*parts) for parts in target.fmri_output_subdirs]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate fMRI outputs for target '{target.key}'. Checked: {searched or '[no candidates]'}"
    )

