package mainmenu

import (
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/eeg-pipeline/tui/types"
)

var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripANSI(s string) string {
	return ansiPattern.ReplaceAllString(s, "")
}

func normalizeWhitespace(s string) string {
	s = stripANSI(s)
	s = strings.NewReplacer(
		"│", " ",
		"╭", " ",
		"╮", " ",
		"╰", " ",
		"╯", " ",
		"─", " ",
		"━", " ",
	).Replace(s)
	return strings.Join(strings.Fields(s), " ")
}

func TestHandleEnter_SelectsPlottingPipelineFromUtilities(t *testing.T) {
	m := New()
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPlotting

	next, _ := m.handleEnter()
	updated, ok := next.(Model)
	if !ok {
		t.Fatalf("expected Model, got %T", next)
	}
	if updated.SelectedPipeline != int(types.PipelinePlotting) {
		t.Fatalf("expected SelectedPipeline=%d, got %d", int(types.PipelinePlotting), updated.SelectedPipeline)
	}
	if updated.SelectedUtility != -1 {
		t.Fatalf("expected no utility selection, got %d", updated.SelectedUtility)
	}
}

func TestHandleEnter_SelectsPipelineSmokeUtility(t *testing.T) {
	m := New()
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	next, _ := m.handleEnter()
	updated, ok := next.(Model)
	if !ok {
		t.Fatalf("expected Model, got %T", next)
	}
	if updated.SelectedUtility != UtilityPipelineSmokeTest {
		t.Fatalf("expected SelectedUtility=%d, got %d", UtilityPipelineSmokeTest, updated.SelectedUtility)
	}
	if updated.SelectedPipeline != -1 {
		t.Fatalf("expected no pipeline selection, got %d", updated.SelectedPipeline)
	}
}

func TestSetCursor_PlottingPipelineTargetsUtilitiesSection(t *testing.T) {
	m := New()

	m.SetCursor(int(types.PipelinePlotting))

	if m.currentSection != SectionUtilities {
		t.Fatalf("expected currentSection=%d, got %d", SectionUtilities, m.currentSection)
	}
	if m.utilityCursor != UtilityPlotting {
		t.Fatalf("expected utilityCursor=%d, got %d", UtilityPlotting, m.utilityCursor)
	}
}

func TestUpdate_RKeyResumesLastSession(t *testing.T) {
	m := New()
	m.SetLastPipeline(int(types.PipelineBehavior))

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
	got := updated.(Model)

	if got.SelectedPipeline != int(types.PipelineBehavior) {
		t.Fatalf("expected resume to select pipeline %d, got %d", int(types.PipelineBehavior), got.SelectedPipeline)
	}
}

func TestView_WideLayoutShowsPipelinePreviewPane(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36
	m.Task = "stroop"
	m.currentSection = SectionAnalysis
	m.analysisCursor = 1

	view := normalizeWhitespace(m.View())

	required := []string{
		"Behavior",
		"DETAILS",
		"eeg-pipeline behavior",
		"Task stroop",
		"FOCUS",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected wide layout to contain %q\nview:\n%s", item, view)
		}
	}

	unwanted := []string{"Context", "Project", "Recent Runs"}
	for _, item := range unwanted {
		if strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("did not expect wide layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_WideLayoutShowsUtilityPreviewPane(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := normalizeWhitespace(m.View())

	required := []string{
		"Pipeline Smoke Test",
		"scripts/tui_pipeline_smoke.py",
		"CLI entrypoints",
		"Verify parser wiring",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected utility preview to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_WideLayoutMarksLastUsedPipeline(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36

	m.SetCursor(int(types.PipelineFmri))

	view := normalizeWhitespace(m.View())

	if !strings.Contains(view, normalizeWhitespace("last used")) {
		t.Fatalf("expected preview pane to mark the restored pipeline as last used\nview:\n%s", view)
	}
}

func TestView_WideLayoutShowsSessionSummaryAndRecentRuns(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36
	m.currentSection = SectionAnalysis
	m.analysisCursor = 1
	m.SetLastPipeline(int(types.PipelineBehavior))
	m.SetConfigSummary(HomeConfigSummary{
		Task:               "stroop",
		DerivRoot:          "/tmp/project/derivatives",
		BidsRoot:           "/tmp/project/bids",
		PreprocessingNJobs: 8,
	})
	m.SetRecentRuns([]RecentRunSummary{
		{
			Pipeline: "behavior",
			Mode:     "compute",
			Duration: "12m",
			Age:      "2 hr ago",
			Success:  true,
		},
		{
			Pipeline: "features",
			Mode:     "compute",
			Duration: "4m",
			Age:      "1 day ago",
			Success:  false,
		},
	})

	view := normalizeWhitespace(m.View())

	required := []string{
		"Behavior",
		"WORKSPACE",
		"stroop",
		"behavior",
		"2 hr ago",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected wide layout to contain %q\nview:\n%s", item, view)
		}
	}

	unwanted := []string{"Project", "Recent Runs", "Context"}
	for _, item := range unwanted {
		if strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("did not expect wide layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_CompactLayoutFitsSmallWindow(t *testing.T) {
	m := New()
	m.width = 60
	m.height = 20
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := stripANSI(m.View())
	lines := strings.Split(view, "\n")
	if len(lines) > m.height {
		t.Fatalf("expected compact layout to fit within %d lines, got %d\nview:\n%s", m.height, len(lines), view)
	}

	normalized := normalizeWhitespace(view)
	required := []string{
		"Analysis",
		"Utilities",
		"Pipeline Smoke Test",
		"more",
	}
	for _, item := range required {
		if !strings.Contains(normalized, item) {
			t.Fatalf("expected compact layout to contain %q\nview:\n%s", item, view)
		}
	}

	if strings.Contains(normalized, "Selected") {
		t.Fatalf("did not expect the wide preview pane in compact layout\nview:\n%s", view)
	}
}

func TestShortPathUsesNativeHomePrefix(t *testing.T) {
	m := New()
	home := homeDir()
	if home == "" {
		t.Skip("home directory unavailable")
	}

	got := m.shortPath(filepath.Join(home, "project", "data"))
	want := filepath.Join("~", "project", "data")
	if got != want {
		t.Fatalf("shortPath() = %q, want %q", got, want)
	}
}

func TestHomeRelativePathRejectsFalsePrefixOnWindows(t *testing.T) {
	home := `C:\Users\Jo`
	path := `C:\Users\John\project`

	if relative, ok := homeRelativePath("windows", home, path); ok {
		t.Fatalf("expected false-prefix path to be rejected, got ok=true with relative=%q", relative)
	}
}

func TestView_CompactLayoutScrollsToSelectedItem(t *testing.T) {
	m := New()
	m.width = 60
	m.height = 18
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := stripANSI(m.View())
	normalized := normalizeWhitespace(view)

	if !strings.Contains(normalized, "Pipeline Smoke Test") {
		t.Fatalf("expected compact layout to keep the selected item visible\nview:\n%s", view)
	}
	if !strings.Contains(normalized, "more") {
		t.Fatalf("expected compact layout to show a scroll indicator when content overflows\nview:\n%s", view)
	}
}

func TestView_CompactLayoutShowsSelectionContextWhenSpaceAllows(t *testing.T) {
	m := New()
	m.width = 72
	m.height = 24
	m.Task = "stroop"
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := normalizeWhitespace(m.View())

	required := []string{
		"Pipeline Smoke Test",
		"CLI entrypoints",
		"scripts/tui_pipeline_smoke.py",
		"Task stroop",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected compact layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_CompactLayoutShowsFocusGuidanceWhenTallEnough(t *testing.T) {
	m := New()
	m.width = 72
	m.height = 28
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := normalizeWhitespace(m.View())

	required := []string{
		"FOCUS",
		"Verify parser wiring for major CLI command families",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected tall compact layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_DoesNotShowSectionItemCounts(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36

	view := normalizeWhitespace(m.View())

	unwanted := []string{
		"Preprocessing 2",
		"Analysis 4",
		"Utilities 3",
	}
	for _, item := range unwanted {
		if strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("did not expect view to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_TallMidWidthLayoutUsesWidePreviewPane(t *testing.T) {
	m := New()
	m.width = 98
	m.height = 54
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPlotting
	m.SetConfigSummary(HomeConfigSummary{
		Task:      "thermalactive",
		DerivRoot: "/tmp/project/derivatives",
	})

	view := normalizeWhitespace(m.View())

	required := []string{
		"WORKSPACE",
		"thermalactive",
		"../project/derivatives",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected tall mid-width layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestRenderCompactDetailPane_InsertsSpacerBeforeFocusSection(t *testing.T) {
	m := New()
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := stripANSI(m.renderCompactDetailPane(72, 10))
	lines := strings.Split(view, "\n")

	focusIdx := -1
	for idx, line := range lines {
		if strings.Contains(line, "FOCUS") {
			focusIdx = idx
			break
		}
	}
	if focusIdx <= 0 {
		t.Fatalf("expected compact detail pane to contain a focus section\nview:\n%s", view)
	}
	if strings.TrimSpace(lines[focusIdx-1]) != "" {
		t.Fatalf("expected a spacer line before focus section\nview:\n%s", view)
	}
}

func TestRenderCompactDetailPane_InsertsSpacerAfterDescriptionWhenRoomAllows(t *testing.T) {
	m := New()
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := stripANSI(m.renderCompactDetailPane(72, 11))
	lines := strings.Split(view, "\n")

	if len(lines) < 5 {
		t.Fatalf("expected enough lines to inspect description spacing\nview:\n%s", view)
	}
	if strings.TrimSpace(lines[3]) != "" {
		t.Fatalf("expected a spacer line after description\nview:\n%s", view)
	}
}

func TestView_TallCompactLayoutUsesSeparateMenuAndDetailPanels(t *testing.T) {
	m := New()
	m.width = 84
	m.height = 40
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPlotting

	view := stripANSI(m.View())

	if strings.Count(view, "╭") < 2 {
		t.Fatalf("expected tall compact layout to render separate menu and detail panels\nview:\n%s", view)
	}
}
