package dashboard

import (
	"regexp"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripANSI(s string) string {
	return ansiPattern.ReplaceAllString(s, "")
}

func newTestModel() Model {
	m := New("")
	m.loading = false
	m.stats = StatsData{
		TotalSubjects:          12,
		BidsSubjects:           12,
		EegPrepSubjects:        10,
		FmriPrepSubjects:       9,
		EpochsSubjects:         8,
		FeaturesSubjects:       7,
		FmriFirstLevelSubjects: 6,
		FmriBetaSeriesSubjects: 5,
		FmriLssSubjects:        4,
		FeatureCategories: map[string]int{
			"frontal_power": 7,
			"erp":           3,
		},
		Task: "stroop",
	}
	return m
}

func hasLineWithSections(view string) bool {
	for _, line := range strings.Split(view, "\n") {
		if strings.Contains(line, "EEG") && strings.Contains(line, "fMRI") {
			return true
		}
	}
	return false
}

func TestUpdate_WindowSizeMsgStoresDimensions(t *testing.T) {
	m := newTestModel()

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 72, Height: 24})
	got := updated.(Model)

	if got.width != 72 || got.height != 24 {
		t.Fatalf("expected stored size 72x24, got %dx%d", got.width, got.height)
	}
}

func TestView_WideLayoutShowsSideBySideSections(t *testing.T) {
	m := newTestModel()
	m.width = 140
	m.height = 32

	view := stripANSI(m.View())

	if !hasLineWithSections(view) {
		t.Fatalf("expected wide layout to place EEG and fMRI on the same row\nview:\n%s", view)
	}
}

func TestView_NarrowLayoutFitsWindowWidth(t *testing.T) {
	m := newTestModel()
	m.width = 72
	m.height = 24

	view := stripANSI(m.View())

	if hasLineWithSections(view) {
		t.Fatalf("expected narrow layout to stack EEG and fMRI sections\nview:\n%s", view)
	}

	for _, line := range strings.Split(view, "\n") {
		if lipgloss.Width(line) > m.width {
			t.Fatalf("expected line to fit width %d, got %d\nline: %q\nview:\n%s", m.width, lipgloss.Width(line), line, view)
		}
	}
}
