package wizard

import (
	"regexp"
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

var wizardHeaderANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripWizardHeaderANSI(s string) string {
	return wizardHeaderANSIPattern.ReplaceAllString(s, "")
}

func TestBuildStepPillContainsStepFraction(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.CurrentStep = types.StepSelectComputations
	m.stepIndex = 1

	pill := stripWizardHeaderANSI(m.buildStepPill())

	if !strings.Contains(pill, "2/") {
		t.Fatalf("expected step pill to contain fraction like '2/N', got %q", pill)
	}
}

func TestBuildProgressBarFillsCurrentStep(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.stepIndex = 0

	progress := stripWizardHeaderANSI(m.buildProgressBar(20))
	filledCount := strings.Count(progress, "█")
	emptyCount := strings.Count(progress, "░")

	if filledCount == 0 {
		t.Fatalf("expected current step to contribute visible filled progress, got %q", progress)
	}
	if emptyCount == 0 {
		t.Fatalf("expected first step to leave remaining empty progress, got %q", progress)
	}
}

func TestBuildProgressBarFullyFillsFinalStep(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.stepIndex = len(m.steps) - 1

	progress := stripWizardHeaderANSI(m.buildProgressBar(20))
	emptyCount := strings.Count(progress, "░")

	if emptyCount != 0 {
		t.Fatalf("expected final step to fully fill progress bar, got %q", progress)
	}
}
