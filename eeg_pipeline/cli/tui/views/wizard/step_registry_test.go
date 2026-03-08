package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestRenderStepContent_RoutesSubjectSelectionThroughRegistry(t *testing.T) {
	m := New(types.PipelineBehavior, "/tmp")
	m.height = 40
	m.contentWidth = 100
	m.CurrentStep = types.StepSelectSubjects
	m.SetSubjectLoadError("discovery failed")

	rendered := m.renderStepContent()

	if !strings.Contains(rendered, "Subject discovery failed") {
		t.Fatalf("expected subject selection renderer output, got: %s", rendered)
	}
}

func TestValidateCurrentStep_RoutesBandValidationThroughRegistry(t *testing.T) {
	m := New(types.PipelineFeatures, "/tmp")
	m.CurrentStep = types.StepSelectBands
	m.bandSelected = make(map[int]bool)

	errors := m.validateCurrentStep()

	if len(errors) != 1 || errors[0] != "Select at least one frequency band" {
		t.Fatalf("expected band-selection validation error, got: %#v", errors)
	}
}

func TestHandleEnter_RunsSelectModeExitHookForFmriTrialSignatures(t *testing.T) {
	m := New(types.PipelineFmriAnalysis, "/tmp")
	m.CurrentStep = types.StepSelectMode
	m.modeIndex = 2
	m.fmriAnalysisFmriprepSpace = "T1w"

	next, _ := m.handleEnter()
	updated := next.(Model)

	if updated.CurrentStep != types.StepSelectSubjects {
		t.Fatalf("expected to advance to subject selection, got %v", updated.CurrentStep)
	}
	if updated.fmriAnalysisFmriprepSpace != "MNI152NLin2009cAsym" {
		t.Fatalf("expected trial-signature mode to default to MNI space, got %q", updated.fmriAnalysisFmriprepSpace)
	}
}

func TestHandleEnter_RunsConfigureOptionsEnterHook(t *testing.T) {
	m := New(types.PipelineFeatures, "/tmp")
	m.SetSubjects([]types.SubjectStatus{
		{
			ID:        "sub-0001",
			HasEpochs: true,
			FeatureAvailability: &types.FeatureAvailability{
				Features: map[string]types.AvailabilityInfo{
					"power": {Available: true},
				},
			},
		},
	})
	m.CurrentStep = types.StepSelectSubjects
	m.featureAvailability = make(map[string]bool)

	next, _ := m.handleEnter()
	updated := next.(Model)

	if updated.CurrentStep != types.StepConfigureOptions {
		t.Fatalf("expected to advance to category selection, got %v", updated.CurrentStep)
	}
	if !updated.featureAvailability["power"] {
		t.Fatalf("expected configure-options enter hook to refresh feature availability")
	}
}
