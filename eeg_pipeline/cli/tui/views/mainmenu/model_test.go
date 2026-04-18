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

func TestUpdate_MouseWheelDownMovesSelection(t *testing.T) {
	m := New()

	updated, _ := m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelDown})
	got := updated.(Model)

	if got.currentSection != SectionPreprocessing {
		t.Fatalf("expected preprocessing section to remain active, got %d", got.currentSection)
	}
	if got.prepCursor != 1 {
		t.Fatalf("expected wheel down to move to second preprocessing pipeline, got %d", got.prepCursor)
	}
}

func TestUpdate_MouseClickActivatesMenuItem(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36

	lines := strings.Split(stripANSI(m.View()), "\n")
	targetLine := -1
	for i, line := range lines {
		if strings.Contains(line, "Pipeline Smoke Test") {
			targetLine = i
			break
		}
	}
	if targetLine < 0 {
		t.Fatalf("could not find pipeline smoke test row in view:\n%s", m.View())
	}

	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      targetLine,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if got.SelectedUtility != UtilityPipelineSmokeTest {
		t.Fatalf("expected click to activate pipeline smoke test utility, got %d", got.SelectedUtility)
	}
	if got.SelectedPipeline != -1 {
		t.Fatalf("expected no pipeline selection, got %d", got.SelectedPipeline)
	}
}

func TestUpdate_MouseMotionHighlightsMenuItem(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36

	lines := strings.Split(stripANSI(m.View()), "\n")
	targetLine := -1
	for i, line := range lines {
		if strings.Contains(line, "Pipeline Smoke Test") {
			targetLine = i
			break
		}
	}
	if targetLine < 0 {
		t.Fatalf("could not find pipeline smoke test row in view:\n%s", m.View())
	}

	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      targetLine,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionMotion,
	})
	got := updated.(Model)

	if got.currentSection != SectionUtilities {
		t.Fatalf("expected hover to switch to utilities section, got %d", got.currentSection)
	}
	if got.utilityCursor != UtilityPipelineSmokeTest {
		t.Fatalf("expected hover to move cursor to pipeline smoke test, got %d", got.utilityCursor)
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

	unwanted := []string{"Context", "Project", "Recent Runs", "Quick Actions", "Ctrl+K"}
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

func TestView_WideLayoutShowsSessionSummary(t *testing.T) {
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

	view := normalizeWhitespace(m.View())

	required := []string{
		"Behavior",
		"WORKSPACE",
		"stroop",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected wide layout to contain %q\nview:\n%s", item, view)
		}
	}

	unwanted := []string{"Project", "Recent Runs", "Context", "Quick Actions", "Ctrl+K"}
	for _, item := range unwanted {
		if strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("did not expect wide layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestMainMenuColumnWidths_FitMinimumTerminal(t *testing.T) {
	m := New()

	leftWidth, rightWidth := m.mainMenuColumnWidths(56)

	if leftWidth <= 0 || rightWidth <= 0 {
		t.Fatalf("expected positive column widths, got left=%d right=%d", leftWidth, rightWidth)
	}
	if leftWidth+rightWidth+mainMenuColumnGap != 56 {
		t.Fatalf("expected columns to fill available width, got left=%d right=%d", leftWidth, rightWidth)
	}
	if leftWidth < mainMenuMenuMinWidth {
		t.Fatalf("expected left column to keep at least %d cells, got %d", mainMenuMenuMinWidth, leftWidth)
	}
	if rightWidth < mainMenuPreviewMinWidth {
		t.Fatalf("expected right column to keep at least %d cells, got %d", mainMenuPreviewMinWidth, rightWidth)
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
		filepath.Join("..", "project", "derivatives"),
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected tall mid-width layout to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_WideLayoutMovesDescriptionsOutOfMenuRows(t *testing.T) {
	m := New()
	m.width = 140
	m.height = 36
	m.currentSection = SectionAnalysis
	m.analysisCursor = 1
	m.SetConfigSummary(HomeConfigSummary{
		Task:      "stroop",
		DerivRoot: "/tmp/project/derivatives",
		BidsRoot:  "/tmp/project/bids",
	})

	view := normalizeWhitespace(m.View())

	required := []string{
		"Behavior",
		"EEG-behavior analysis",
		"DETAILS",
		"WORKSPACE",
		"FOCUS",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected redesigned main menu to contain %q\nview:\n%s", item, view)
		}
	}

	unwanted := []string{
		"Bad channels ICA epochs",
		"Preprocess fMRI fMRIPrep style",
		"LOSO regression classification",
		"Curate and export visualization suites",
		"Run quick parser runtime checks across pipeline commands",
	}
	for _, item := range unwanted {
		if strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("did not expect menu rows to repeat description text %q\nview:\n%s", item, view)
		}
	}
}

func TestView_SmallWindowKeepsPreviewPane(t *testing.T) {
	m := New()
	m.width = 60
	m.height = 20
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	view := normalizeWhitespace(m.View())

	required := []string{
		"Pipeline Smoke Test",
		"DETAILS",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected small-window view to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_SmallWindowStillKeepsSelectionContext(t *testing.T) {
	m := New()
	m.width = 60
	m.height = 20
	m.Task = "stroop"
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest
	m.SetConfigSummary(HomeConfigSummary{
		DerivRoot: "/tmp/project/derivatives",
		BidsRoot:  "/tmp/project/bids",
	})

	view := normalizeWhitespace(m.View())

	required := []string{
		"Task stroop",
		"WORKSPACE",
	}
	for _, item := range required {
		if !strings.Contains(view, normalizeWhitespace(item)) {
			t.Fatalf("expected small-window view to contain %q\nview:\n%s", item, view)
		}
	}
}

func TestView_SmallWindowRendersEditorialPanels(t *testing.T) {
	m := New()
	m.width = 60
	m.height = 20
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPlotting

	view := stripANSI(m.View())

	if strings.Count(view, "╭") < 2 {
		t.Fatalf("expected narrow editorial layout to render separate menu and detail panels\nview:\n%s", view)
	}
}
