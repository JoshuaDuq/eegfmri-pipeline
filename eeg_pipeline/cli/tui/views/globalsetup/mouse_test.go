package globalsetup

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestUpdateMouseMotionHighlightsField(t *testing.T) {
	m := New(".")
	m.isLoading = false

	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionMotion,
		Y:      m.fieldRowStartY() + 1,
	})
	got := updated.(*Model)
	if got.fieldCursor != 1 {
		t.Fatalf("expected hover to move field cursor to 1, got %d", got.fieldCursor)
	}
}

func TestUpdateMouseClickStartsEditingField(t *testing.T) {
	m := New(".")
	m.isLoading = false

	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
		Y:      m.fieldRowStartY(),
	})
	got := updated.(*Model)
	if !got.editingText {
		t.Fatal("expected click to start editing")
	}
	if got.editingField != fieldTask {
		t.Fatalf("expected task field to be editing, got %v", got.editingField)
	}
}

func TestUpdateMouseClickSwitchesSection(t *testing.T) {
	m := New(".")
	m.isLoading = false

	line := lineIndexContainingGlobalSetup(t, m.View(), "PATHS")
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
		X:      strings.Index(stripGlobalSetupANSI(strings.Split(m.View(), "\n")[line]), "PATHS") + 1,
		Y:      line,
	})
	got := updated.(*Model)
	if got.sectionIndex != int(sectionPaths) {
		t.Fatalf("expected click to switch to paths, got %d", got.sectionIndex)
	}
	if got.fieldCursor != 0 {
		t.Fatalf("expected section switch to reset field cursor, got %d", got.fieldCursor)
	}
}

func lineIndexContainingGlobalSetup(t *testing.T, view, needle string) int {
	t.Helper()
	for i, line := range strings.Split(view, "\n") {
		if strings.Contains(stripGlobalSetupANSI(line), needle) {
			return i
		}
	}
	t.Fatalf("did not find %q in view:\n%s", needle, view)
	return -1
}
