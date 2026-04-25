package wizard

import (
	"errors"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/types"
)

func TestClipboardResultMsgShowsSuccessToast(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	updated, _ := m.Update(executor.ClipboardResultMsg{Error: nil})
	got := updated.(Model)
	if got.toastMessage == "" {
		t.Fatal("expected success toast after clipboard copy, got empty toast")
	}
	if !strings.Contains(strings.ToLower(got.toastMessage), "copied") {
		t.Fatalf("expected toast to mention copy success, got %q", got.toastMessage)
	}
	if got.toastType != "clipboard" {
		t.Fatalf("expected toast type 'clipboard', got %q", got.toastType)
	}
}

func TestClipboardResultMsgShowsErrorToast(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	updated, _ := m.Update(executor.ClipboardResultMsg{Error: errors.New("xclip missing")})
	got := updated.(Model)
	if !strings.Contains(got.toastMessage, "xclip missing") {
		t.Fatalf("expected toast to surface clipboard error, got %q", got.toastMessage)
	}
	if got.toastType != "clipboard-error" {
		t.Fatalf("expected toast type 'clipboard-error', got %q", got.toastType)
	}
}

func TestScrollCommandPreviewClampsToMaxOffsetSoUpScrollsImmediately(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.width = 160
	m.height = 40
	m.contentWidth = 140
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"

	maxOffset := m.commandPreviewMaxOffset()
	if maxOffset == 0 {
		t.Fatal("expected the features command preview to be longer than the visible budget for this test")
	}

	// Hammer the down-scroll well past the end.
	for i := 0; i < maxOffset+50; i++ {
		m.scrollCommandPreview(1)
	}
	if m.cmdScrollOffset != maxOffset {
		t.Fatalf("expected stored offset to be clamped at %d after over-scrolling, got %d", maxOffset, m.cmdScrollOffset)
	}

	// A single up-scroll must move the viewport, not just decrement a stale
	// stash of unreachable offsets.
	m.scrollCommandPreview(-1)
	if m.cmdScrollOffset != maxOffset-1 {
		t.Fatalf("expected one [ press to land at offset %d, got %d", maxOffset-1, m.cmdScrollOffset)
	}
}

func TestReviewPanelSurfacesClipboardSuccessInCommandHeader(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.task = "thermalactive"
	m.bidsRoot = "C:/study/bids"
	m.derivRoot = "C:/study/derivatives"
	m.toastMessage = "Command copied to clipboard"
	m.toastType = "clipboard"

	panel := stripWizardHeaderANSI(m.renderReviewPanel(48, 20))
	if !strings.Contains(panel, "COPIED") {
		t.Fatalf("expected review panel to surface clipboard success in COMMAND header, got %q", panel)
	}
	if strings.Contains(panel, "COMMAND  \u2500") || strings.Contains(strings.ToUpper(panel), "COMMAND  ─") {
		// During the toast the label should NOT read as the standard
		// "COMMAND" rule; we want the transient COPIED variant in its
		// place.
		t.Fatalf("expected COMMAND label to be replaced while clipboard toast is active, got %q", panel)
	}
}

func TestReviewPanelSuppressesFooterClipboardToastWhenPanelVisible(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.contentWidth = 130
	m.toastMessage = "Command copied to clipboard"
	m.toastType = "clipboard"

	if status := m.renderFooterStatus(120); status != "" {
		t.Fatalf("expected footer to suppress clipboard toast while review panel is visible, got %q", status)
	}

	m.contentWidth = 60
	if status := m.renderFooterStatus(60); status == "" {
		t.Fatal("expected footer to surface clipboard toast when review panel is hidden")
	}
}

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

func TestReviewPanelOmitsRedundantNextRowWhenBlocked(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.validationErrors = []string{"Select at least one analysis to run"}

	blocked := stripWizardHeaderANSI(m.renderReviewPanel(48, 18))
	if !strings.Contains(blocked, "VALIDATION") ||
		!strings.Contains(blocked, "Select at least one analysis to run") {
		t.Fatalf("expected blocked review panel to surface the validation error, got %q", blocked)
	}
	if strings.Contains(blocked, "NEXT") || strings.Contains(blocked, "Fix:") {
		t.Fatalf("expected blocked review panel to omit redundant NEXT/Fix row, got %q", blocked)
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

func TestReviewPanelShowsSecondErrorInsteadOfPlusOneCounter(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.validationErrors = []string{
		"Select at least one analysis to run",
		"Select at least one valid subject",
	}

	rendered := stripWizardHeaderANSI(m.renderReviewPanel(48, 20))
	if !strings.Contains(rendered, "Select at least one analysis to run") ||
		!strings.Contains(rendered, "Select at least one valid subject") {
		t.Fatalf("expected both validation errors inline, got %q", rendered)
	}
	if strings.Contains(rendered, "+1 more issue") {
		t.Fatalf("expected the second error inline rather than a +1 counter, got %q", rendered)
	}
}

func TestReviewPanelSubjectsRowSilentWhileLoading(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.SetSubjectsLoading()

	rendered := stripWizardHeaderANSI(m.renderReviewPanel(48, 20))
	if !strings.Contains(rendered, "loading") {
		t.Fatalf("expected subjects row to surface loading state, got %q", rendered)
	}
	if strings.Contains(rendered, "none selected") {
		t.Fatalf("expected subjects row to suppress 'none selected' while loading, got %q", rendered)
	}
}
