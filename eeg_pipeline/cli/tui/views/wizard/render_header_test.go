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

func TestRenderHeaderRhythmTitleSpacerRailSpacerStepper(t *testing.T) {
	m := New(types.PipelineBehavior, ".")

	header := stripWizardHeaderANSI(m.renderHeader(100))
	lines := strings.Split(header, "\n")

	// Five-line wizard chrome: title, spacer, breadcrumb rail, spacer, stepper.
	if len(lines) != 5 {
		t.Fatalf("expected header to use 5 lines, got %d: %q", len(lines), header)
	}
	if strings.TrimSpace(lines[0]) == "" {
		t.Fatalf("expected title row first, got blank line: %q", header)
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("expected blank spacer after title row, got %q", lines[1])
	}
	if strings.TrimSpace(lines[2]) == "" {
		t.Fatalf("expected breadcrumb line after spacer, got blank: %q", header)
	}
	if strings.TrimSpace(lines[3]) != "" {
		t.Fatalf("expected blank spacer before stepper bar, got %q", lines[3])
	}
	if strings.TrimSpace(lines[4]) == "" {
		t.Fatalf("expected progress line last, got blank: %q", header)
	}
}

func TestRenderValidationErrorsHasNoTrailingBlankLine(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.validationErrors = []string{
		"Select at least one analysis to run",
		"Select at least one valid subject",
	}

	rendered := m.renderValidationErrors()
	if strings.HasSuffix(rendered, "\n") {
		t.Fatalf("expected validation errors to avoid a trailing blank line, got %q", rendered)
	}
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

	// The stepper bar uses a heavy rule (━) for completed cells and a light
	// rule (─) for remaining cells; both glyphs share the same vertical
	// centreline so the bar reads as one continuous line whose stroke
	// thickens into the completed region. On the first step at width 20 we
	// expect a small filled run plus a longer remainder.
	progress := stripWizardHeaderANSI(m.buildProgressBar(20))
	filled := strings.Count(progress, "━")
	empty := strings.Count(progress, "─")
	if filled+empty != 20 {
		t.Fatalf("expected 20 rule cells total, got %d filled + %d empty (%q)", filled, empty, progress)
	}
	if filled == 0 {
		t.Fatalf("expected at least one filled cell on first step, got %q", progress)
	}
	if empty == 0 {
		t.Fatalf("expected unfilled cells to remain on first step, got %q", progress)
	}
}

func TestBuildProgressBarFullyFillsFinalStep(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.stepIndex = len(m.steps) - 1

	progress := stripWizardHeaderANSI(m.buildProgressBar(20))
	if got := strings.Count(progress, "━"); got != 20 {
		t.Fatalf("expected 20 fully-filled rule cells, got %d (%q)", got, progress)
	}
	if strings.Contains(progress, "─") {
		t.Fatalf("expected no unfilled cells on final step, got %q", progress)
	}
}
