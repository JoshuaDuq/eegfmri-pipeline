package wizard

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/types"
)

func TestRenderContentUsesReviewPanelOnlyWhenWide(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.width = 140
	m.height = 34
	m.CurrentStep = types.StepSelectComputations
	m.validationErrors = []string{"Select at least one analysis to run"}
	m.SetSubjects([]types.SubjectStatus{
		{ID: "sub-01", HasSourceData: true, HasBids: true, HasDerivatives: true},
		{ID: "sub-02", HasSourceData: true, HasBids: true, HasDerivatives: false},
	})

	wide := stripWizardHeaderANSI(m.renderContent(126, 14))
	if !strings.Contains(wide, "REVIEW") {
		t.Fatalf("expected wide wizard content to include review panel, got %q", wide)
	}
	if !strings.Contains(wide, "Subjects") || !strings.Contains(wide, "2 selected") {
		t.Fatalf("expected review panel to summarize selected subjects, got %q", wide)
	}
	if !strings.Contains(wide, "Task") ||
		!strings.Contains(wide, "not configured") ||
		!strings.Contains(wide, "Paths") ||
		!strings.Contains(wide, "missing BIDS") {
		t.Fatalf("expected review panel to show task and path readiness, got %q", wide)
	}
	if !strings.Contains(wide, "VALIDATION") ||
		!strings.Contains(wide, "Select at least one analysis to run") {
		t.Fatalf("expected review panel to expose validation errors, got %q", wide)
	}
	if !strings.Contains(wide, "COMMAND") || !strings.Contains(wide, "eeg-pipeline behavior") {
		t.Fatalf("expected review panel to show command preview, got %q", wide)
	}

	narrow := stripWizardHeaderANSI(m.renderContent(82, 14))
	if strings.Contains(narrow, "REVIEW") {
		t.Fatalf("expected narrow wizard content to keep single-pane layout, got %q", narrow)
	}
}

func TestRenderReviewPanelFitsRequestedFrame(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.CurrentStep = types.StepSelectBands
	m.stepIndex = 2
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"
	m.validationErrors = []string{
		"Select at least one frequency band",
		"Select at least one valid subject",
	}
	m.SetSubjects([]types.SubjectStatus{
		{ID: "sub-01", HasSourceData: true, HasBids: true, HasDerivatives: true},
	})

	panel := m.renderReviewPanel(38, 12)
	lines := strings.Split(panel, "\n")
	if len(lines) != 12 {
		t.Fatalf("expected review panel height 12, got %d: %q", len(lines), panel)
	}
	for i, line := range lines {
		if got := lipgloss.Width(line); got != 38 {
			t.Fatalf("expected line %d width 38, got %d: %q", i, got, line)
		}
	}
}

func TestViewWithReviewPanelFitsTerminalWidth(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.width = 140
	m.height = 34
	m.validationErrors = []string{"Select at least one analysis to run"}
	m.SetSubjects([]types.SubjectStatus{
		{ID: "sub-01", HasSourceData: true, HasBids: true, HasDerivatives: true},
	})

	view := m.View()
	for i, line := range strings.Split(view, "\n") {
		if got := lipgloss.Width(line); got > m.width {
			t.Fatalf("expected view line %d to fit width %d, got %d: %q", i, m.width, got, line)
		}
	}
}

func TestTallReviewPanelIncludesModeTaskAndPaths(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"

	panel := stripWizardHeaderANSI(m.renderReviewPanel(48, 20))
	for _, want := range []string{"Mode", "Task", "thermalactive", "Paths", "configured"} {
		if !strings.Contains(panel, want) {
			t.Fatalf("expected tall review panel to include %q, got %q", want, panel)
		}
	}
}

func TestMediumReviewPanelPrioritizesTaskAndPathsOverMode(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"

	panel := stripWizardHeaderANSI(m.renderReviewPanel(48, 16))
	for _, want := range []string{"Task", "thermalactive", "Paths", "configured"} {
		if !strings.Contains(panel, want) {
			t.Fatalf("expected medium review panel to include %q, got %q", want, panel)
		}
	}
	if strings.Contains(panel, "Mode") {
		t.Fatalf("expected medium review panel to omit lower-priority mode row, got %q", panel)
	}
}

func TestReviewPanelWrapsCommandPreviewAcrossReadableLines(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"
	m.SetSubjects([]types.SubjectStatus{
		{ID: "sub-01", HasSourceData: true, HasBids: true, HasDerivatives: true},
	})

	lines := m.commandPreviewLines(42, 3)
	if len(lines) < 2 {
		t.Fatalf("expected wrapped command preview to use multiple lines, got %q", lines)
	}
	if !strings.Contains(stripWizardHeaderANSI(strings.Join(lines, "\n")), "eeg-pipeline features compute") {
		t.Fatalf("expected first command tokens to remain readable, got %q", lines)
	}
	for i, line := range lines {
		if got := lipgloss.Width(line); got > 42 {
			t.Fatalf("expected command line %d to fit width 42, got %d: %q", i, got, line)
		}
	}
}

func TestReviewPanelCommandPreviewUsesContinuationMarkerWhenClipped(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"

	lines := m.commandPreviewLines(28, 1)
	if len(lines) != 1 {
		t.Fatalf("expected one clipped command line, got %d: %q", len(lines), lines)
	}
	if !strings.HasSuffix(stripWizardHeaderANSI(lines[0]), "...") {
		t.Fatalf("expected clipped command preview to end with ellipsis, got %q", lines[0])
	}
}

func TestReviewPanelShowsNextActionFromValidationState(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.validationErrors = []string{"Select at least one analysis to run"}

	blocked := stripWizardHeaderANSI(m.renderReviewPanel(48, 18))
	if !strings.Contains(blocked, "NEXT") ||
		!strings.Contains(blocked, "Fix: Select at least one analysis to run") {
		t.Fatalf("expected blocked review panel to show concrete next action, got %q", blocked)
	}

	m.validationErrors = nil
	for i, step := range m.steps {
		if step == types.StepAdvancedConfig {
			m.stepIndex = i
			m.CurrentStep = step
			break
		}
	}
	ready := stripWizardHeaderANSI(m.renderReviewPanel(48, 18))
	if !strings.Contains(ready, "NEXT") || !strings.Contains(ready, "Enter to run") {
		t.Fatalf("expected final-step review panel to show run action, got %q", ready)
	}
}
