package history

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func TestUpdateMouseMotionHighlightsRecord(t *testing.T) {
	m := New(".")
	m.loading = false
	m.records = []ExecutionRecord{
		{Pipeline: "preprocessing", Mode: "batch", Duration: 90, StartTime: time.Now().Add(-2 * time.Hour)},
		{Pipeline: "plotting", Mode: "interactive", Duration: 120, StartTime: time.Now().Add(-1 * time.Hour)},
	}

	line := lineIndexContainingHistory(t, m.View(), m.records[1].Pipeline)
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionMotion,
		Y:      line,
	})
	got := updated.(Model)
	if got.cursor != 1 {
		t.Fatalf("expected hover to move cursor to 1, got %d", got.cursor)
	}
}

func TestUpdateMouseClickSelectsRecord(t *testing.T) {
	m := New(".")
	m.loading = false
	m.records = []ExecutionRecord{
		{Pipeline: "preprocessing", Mode: "batch", Duration: 90, StartTime: time.Now().Add(-2 * time.Hour)},
		{Pipeline: "plotting", Mode: "interactive", Duration: 120, StartTime: time.Now().Add(-1 * time.Hour)},
	}

	line := lineIndexContainingHistory(t, m.View(), m.records[1].Pipeline)
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
		Y:      line,
	})
	got := updated.(Model)
	if got.cursor != 1 {
		t.Fatalf("expected click to move cursor to 1, got %d", got.cursor)
	}
}

func lineIndexContainingHistory(t *testing.T, view, needle string) int {
	t.Helper()
	for i, line := range strings.Split(view, "\n") {
		if strings.Contains(stripHistoryANSI(line), needle) {
			return i
		}
	}
	t.Fatalf("did not find %q in view:\n%s", needle, view)
	return -1
}
