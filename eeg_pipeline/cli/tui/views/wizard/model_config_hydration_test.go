package wizard

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestApplyConfigKeys_HydratesSourceLocFmriContrastConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	values := map[string]interface{}{
		"feature_engineering.sourcelocalization.mode":                                 "fmri_informed",
		"feature_engineering.sourcelocalization.fmri.enabled":                         true,
		"feature_engineering.sourcelocalization.fmri.contrast.type":                   "custom",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_a.column":     "event_group",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_b.column":     "group_label",
		"feature_engineering.sourcelocalization.fmri.contrast.phase_scope_column":     "phase_scope_col",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_scope_column": "scope_col",
		"feature_engineering.sourcelocalization.fmri.contrast.stim_phases_to_model":   []interface{}{"ramp", "plateau"},
		"feature_engineering.sourcelocalization.fmri.contrast.condition_scope_trial_types": []interface{}{
			"stim",
			"outcome",
		},
	}

	m.ApplyConfigKeys(values)

	if m.sourceLocMode != 1 {
		t.Fatalf("expected sourceLocMode=1 (fmri_informed), got %d", m.sourceLocMode)
	}
	if !m.sourceLocFmriEnabled {
		t.Fatalf("expected sourceLocFmriEnabled=true")
	}
	if m.sourceLocFmriContrastType != 3 {
		t.Fatalf("expected sourceLocFmriContrastType=3 (custom), got %d", m.sourceLocFmriContrastType)
	}
	if m.sourceLocFmriCondAColumn != "event_group" {
		t.Fatalf("expected condition A column event_group, got %q", m.sourceLocFmriCondAColumn)
	}
	if m.sourceLocFmriCondBColumn != "group_label" {
		t.Fatalf("expected condition B column group_label, got %q", m.sourceLocFmriCondBColumn)
	}
	if m.sourceLocFmriPhaseScopeColumn != "phase_scope_col" {
		t.Fatalf("expected phase scope column phase_scope_col, got %q", m.sourceLocFmriPhaseScopeColumn)
	}
	if m.sourceLocFmriConditionScopeColumn != "scope_col" {
		t.Fatalf("expected condition scope column scope_col, got %q", m.sourceLocFmriConditionScopeColumn)
	}
	if m.sourceLocFmriStimPhasesToModel != "ramp plateau" {
		t.Fatalf("expected stim phases 'ramp plateau', got %q", m.sourceLocFmriStimPhasesToModel)
	}
	if m.sourceLocFmriConditionScopeTrialTypes != "stim outcome" {
		t.Fatalf(
			"expected condition scope values 'stim outcome', got %q",
			m.sourceLocFmriConditionScopeTrialTypes,
		)
	}
}

func TestApplyConfigKeys_HydratesFmriThreadConfig(t *testing.T) {
	m := New(types.PipelineFmri, ".")
	values := map[string]interface{}{
		"fmri_preprocessing.fmriprep.nthreads":     12,
		"fmri_preprocessing.fmriprep.omp_nthreads": float64(6),
	}

	m.ApplyConfigKeys(values)

	if m.fmriNThreads != 12 {
		t.Fatalf("expected fmriNThreads=12, got %d", m.fmriNThreads)
	}
	if m.fmriOmpNThreads != 6 {
		t.Fatalf("expected fmriOmpNThreads=6, got %d", m.fmriOmpNThreads)
	}
}
