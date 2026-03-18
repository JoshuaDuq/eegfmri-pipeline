package wizard

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/types"
)

func TestNormalizeContentFrameTruncatesAndPads(t *testing.T) {
	content := "123456\nabc"
	framed := normalizeContentFrame(content, 4, 3)
	lines := strings.Split(framed, "\n")

	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d: %q", len(lines), framed)
	}

	for i, line := range lines {
		if got := lipgloss.Width(line); got != 4 {
			t.Fatalf("expected line %d width 4, got %d: %q", i, got, line)
		}
	}
	if !strings.Contains(lines[0], "...") {
		t.Fatalf("expected first line to be truncated, got %q", lines[0])
	}
	if !strings.HasPrefix(lines[1], "abc") {
		t.Fatalf("expected second line to preserve content, got %q", lines[1])
	}
}

func TestEffectiveDimensionsDefaultAndCustom(t *testing.T) {
	m := Model{}
	w, h := m.effectiveDimensions()
	if w != defaultWidth || h != defaultHeight {
		t.Fatalf("expected defaults %dx%d, got %dx%d", defaultWidth, defaultHeight, w, h)
	}

	m.width = 80
	m.height = 22
	w, h = m.effectiveDimensions()
	if w != 80 || h != 22 {
		t.Fatalf("expected custom dimensions 80x22, got %dx%d", w, h)
	}
}

func TestContainerDimensionsClamp(t *testing.T) {
	m := Model{}
	cw, ch := m.containerDimensions(50, 10)
	if cw != 48 {
		t.Fatalf("expected width to clamp to w-2, got %d", cw)
	}
	if ch != 8 {
		t.Fatalf("expected height to clamp to h-2, got %d", ch)
	}

	cw, ch = m.containerDimensions(200, 60)
	if cw != 184 {
		t.Fatalf("expected width to be 92%% of 200, got %d", cw)
	}
	if ch != 58 {
		t.Fatalf("expected height to be h-2, got %d", ch)
	}
}

func TestBuildBreadcrumbRowCompactAndWide(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.steps = []types.WizardStep{
		types.StepSelectMode,
		types.StepSelectBands,
		types.StepAdvancedConfig,
	}
	m.stepIndex = 1

	compact := stripWizardHeaderANSI(m.buildBreadcrumbRow(80))
	if strings.Contains(compact, "›") {
		t.Fatalf("expected compact breadcrumb to omit connector, got %q", compact)
	}
	if strings.Contains(compact, "Mode") {
		t.Fatalf("expected compact breadcrumb to omit completed step name, got %q", compact)
	}
	if !strings.Contains(compact, "Bands") {
		t.Fatalf("expected current step name in compact breadcrumb, got %q", compact)
	}

	wide := stripWizardHeaderANSI(m.buildBreadcrumbRow(120))
	if !strings.Contains(wide, "›") {
		t.Fatalf("expected wide breadcrumb to include connector, got %q", wide)
	}
	if !strings.Contains(wide, "Mode") {
		t.Fatalf("expected wide breadcrumb to include completed step name, got %q", wide)
	}
}

func TestRenderValidationSummaryIncludesCount(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.validationErrors = []string{
		"first error",
		"second error",
		"third error",
	}

	summary := stripWizardHeaderANSI(m.renderValidationSummary(200))
	if !strings.Contains(summary, "first error") {
		t.Fatalf("expected summary to include first error, got %q", summary)
	}
	if !strings.Contains(summary, "(+2 more)") {
		t.Fatalf("expected summary to include count, got %q", summary)
	}
}

func TestRenderToastIncludesMessage(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.toastMessage = "Saved"
	m.toastType = "success"

	toast := stripWizardHeaderANSI(m.renderToast(40))
	if !strings.Contains(toast, "Saved") {
		t.Fatalf("expected toast to include message, got %q", toast)
	}
}

func TestBuildSubjectBadgeStates(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.subjectSelected["sub-01"] = true
	m.subjectSelected["sub-02"] = true

	badge := stripWizardHeaderANSI(m.buildSubjectBadge())
	if !strings.Contains(badge, "2 subjects") {
		t.Fatalf("expected subject count badge, got %q", badge)
	}

	m.subjectSelected = map[string]bool{}
	m.subjects = []types.SubjectStatus{{ID: "sub-01"}}
	badge = stripWizardHeaderANSI(m.buildSubjectBadge())
	if !strings.Contains(badge, "no subjects") {
		t.Fatalf("expected no-subjects badge, got %q", badge)
	}

	m.subjects = nil
	if badge = stripWizardHeaderANSI(m.buildSubjectBadge()); badge != "" {
		t.Fatalf("expected empty badge when no subjects present, got %q", badge)
	}
}

