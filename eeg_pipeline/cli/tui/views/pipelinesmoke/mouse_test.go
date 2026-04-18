package pipelinesmoke

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestUpdateMouseMotionHighlightsSmokeItem(t *testing.T) {
	m := New("task")

	line := lineIndexContainingSmoke(t, m.View(), smokeItems[4].Name)
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionMotion,
		Y:      line,
	})
	got := updated.(Model)
	if got.cursor != 4 {
		t.Fatalf("expected hover to move cursor to 4, got %d", got.cursor)
	}
}

func TestUpdateMouseClickTogglesSmokeItem(t *testing.T) {
	m := New("task")
	m.cursor = 0

	line := lineIndexContainingSmoke(t, m.View(), smokeItems[1].Name)
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
		Y:      line,
	})
	got := updated.(Model)
	if got.selected[smokeItems[1].ID] {
		t.Fatal("expected click to toggle the item off")
	}
	if got.cursor != 1 {
		t.Fatalf("expected click to move cursor to 1, got %d", got.cursor)
	}
}

func TestUpdateMouseWheelWrapsSmokeCursor(t *testing.T) {
	m := New("task")
	m.cursor = 0

	updated, _ := m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
	got := updated.(Model)
	if got.cursor != len(smokeItems)-1 {
		t.Fatalf("expected wheel up to wrap to last item, got %d", got.cursor)
	}
}

func lineIndexContainingSmoke(t *testing.T, view, needle string) int {
	t.Helper()
	for i, line := range strings.Split(view, "\n") {
		if strings.Contains(stripPipelineSmokeANSI(line), needle) {
			return i
		}
	}
	t.Fatalf("did not find %q in view:\n%s", needle, view)
	return -1
}
