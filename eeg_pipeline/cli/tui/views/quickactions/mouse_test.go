package quickactions

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestUpdateMouseMotionHighlightsAction(t *testing.T) {
	m := New()
	m.Show()

	line := lineIndexContaining(t, m.View(), quickActions[3].Name)
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionMotion,
		Y:      line,
	})
	got := updated.(Model)
	if got.cursor != 3 {
		t.Fatalf("expected hover to move cursor to 3, got %d", got.cursor)
	}
}

func TestUpdateMouseClickSelectsAction(t *testing.T) {
	m := New()
	m.Show()

	line := lineIndexContaining(t, m.View(), quickActions[1].Name)
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
		Y:      line,
	})
	got := updated.(Model)
	if !got.Done {
		t.Fatal("expected click to complete selection")
	}
	if got.SelectedAction != quickActions[1].Type {
		t.Fatalf("expected selected action %v, got %v", quickActions[1].Type, got.SelectedAction)
	}
}

func TestUpdateMouseWheelWrapsCursor(t *testing.T) {
	m := New()
	m.Show()
	m.cursor = 0

	updated, _ := m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
	got := updated.(Model)
	if got.cursor != len(quickActions)-1 {
		t.Fatalf("expected wheel up to wrap to last item, got %d", got.cursor)
	}
}

func lineIndexContaining(t *testing.T, view, needle string) int {
	t.Helper()
	for i, line := range strings.Split(view, "\n") {
		if strings.Contains(stripQuickActionsANSI(line), needle) {
			return i
		}
	}
	t.Fatalf("did not find %q in view:\n%s", needle, view)
	return -1
}
