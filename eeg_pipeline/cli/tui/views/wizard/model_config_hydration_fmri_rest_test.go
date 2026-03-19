package wizard

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestApplyConfigKeys_HydratesFmriRestConfig(t *testing.T) {
	m := New(types.PipelineFmriAnalysis, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"fmri_resting_state.task_is_rest":       true,
		"fmri_resting_state.input_source":       "bids_raw",
		"fmri_resting_state.fmriprep_space":     "MNI152NLin2009cAsym",
		"fmri_resting_state.require_fmriprep":   false,
		"fmri_resting_state.runs":               []interface{}{1, 3},
		"fmri_resting_state.confounds_strategy": "motion24+wmcsf+fd",
		"fmri_resting_state.high_pass_hz":       0.01,
		"fmri_resting_state.low_pass_hz":        0.08,
		"fmri_resting_state.smoothing_fwhm":     6.0,
		"fmri_resting_state.atlas_labels_img":   "/tmp/atlas.nii.gz",
		"fmri_resting_state.atlas_labels_tsv":   "/tmp/atlas.tsv",
		"fmri_resting_state.connectivity_kind":  "correlation",
		"fmri_resting_state.standardize":        false,
		"fmri_resting_state.detrend":            false,
	})

	if m.modeOptions[m.modeIndex] != "rest" {
		t.Fatalf("expected fmri-analysis mode to hydrate to rest, got %q", m.modeOptions[m.modeIndex])
	}
	if m.fmriAnalysisInputSourceIndex != 1 {
		t.Fatalf("expected fmriAnalysisInputSourceIndex=1 (bids_raw), got %d", m.fmriAnalysisInputSourceIndex)
	}
	if m.fmriAnalysisFmriprepSpace != "MNI152NLin2009cAsym" {
		t.Fatalf("expected fmriAnalysisFmriprepSpace to hydrate, got %q", m.fmriAnalysisFmriprepSpace)
	}
	if m.fmriAnalysisRequireFmriprep {
		t.Fatalf("expected fmriAnalysisRequireFmriprep=false")
	}
	if m.fmriAnalysisRunsSpec != "1 3" {
		t.Fatalf("expected fmriAnalysisRunsSpec='1 3', got %q", m.fmriAnalysisRunsSpec)
	}
	if m.fmriAnalysisConfoundsStrategy != 6 {
		t.Fatalf("expected fmriAnalysisConfoundsStrategy=6, got %d", m.fmriAnalysisConfoundsStrategy)
	}
	if m.fmriAnalysisHighPassHz != 0.01 {
		t.Fatalf("expected fmriAnalysisHighPassHz=0.01, got %v", m.fmriAnalysisHighPassHz)
	}
	if m.fmriAnalysisLowPassHz != 0.08 {
		t.Fatalf("expected fmriAnalysisLowPassHz=0.08, got %v", m.fmriAnalysisLowPassHz)
	}
	if m.fmriAnalysisSmoothingFwhm != 6.0 {
		t.Fatalf("expected fmriAnalysisSmoothingFwhm=6.0, got %v", m.fmriAnalysisSmoothingFwhm)
	}
	if m.fmriAnalysisAtlasLabelsImg != "/tmp/atlas.nii.gz" {
		t.Fatalf("expected fmriAnalysisAtlasLabelsImg to hydrate, got %q", m.fmriAnalysisAtlasLabelsImg)
	}
	if m.fmriAnalysisAtlasLabelsTsv != "/tmp/atlas.tsv" {
		t.Fatalf("expected fmriAnalysisAtlasLabelsTsv to hydrate, got %q", m.fmriAnalysisAtlasLabelsTsv)
	}
	if m.fmriAnalysisConnectivityKind != 0 {
		t.Fatalf("expected fmriAnalysisConnectivityKind=0, got %d", m.fmriAnalysisConnectivityKind)
	}
	if m.fmriAnalysisStandardize {
		t.Fatalf("expected fmriAnalysisStandardize=false")
	}
	if m.fmriAnalysisDetrend {
		t.Fatalf("expected fmriAnalysisDetrend=false")
	}
}

func TestApplyConfigKeys_HydratesFmriPreprocessingRestToggle(t *testing.T) {
	m := New(types.PipelineFmri, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"fmri_preprocessing.task_is_rest": true,
	})

	if !m.fmriTaskIsRest {
		t.Fatalf("expected fmriTaskIsRest=true")
	}
}
