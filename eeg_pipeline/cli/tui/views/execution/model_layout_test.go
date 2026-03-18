package execution

import (
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestUpdateLayout_AlwaysKeepsExecutionStacked(t *testing.T) {
	m := New("echo test")

	if m.useTwoCol {
		t.Fatalf("expected execution layout to stay stacked by default")
	}

	m.SetSize(150, 55)
	if m.useTwoCol {
		t.Fatalf("expected execution layout to remain stacked on wide terminals")
	}
}

var executionANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripExecutionANSI(s string) string {
	return executionANSIPattern.ReplaceAllString(s, "")
}

func TestUpdateViewportSize_DoesNotForceMinWidthPastTerminal(t *testing.T) {
	m := New("echo test")

	m.SetSize(72, 24)

	if m.logViewport.Width > 64 {
		t.Fatalf("expected log viewport width to fit terminal, got %d", m.logViewport.Width)
	}
}

func TestView_StackedCompletionKeepsLogsVisible(t *testing.T) {
	m := New("echo test")
	m.SetSize(100, 24)
	m.AddOutput("[12:00:00] starting")
	m.AddOutput("[12:00:01] processing")
	m.AddOutput("[12:00:02] finished")
	m.SetStatus(StatusSuccess)
	m.StartTime = time.Now().Add(-2 * time.Minute)
	m.EndTime = time.Now()

	view := stripExecutionANSI(m.View())
	if !strings.Contains(view, "Log [3 lines") {
		t.Fatalf("expected completed stacked layout to keep log header visible\nview:\n%s", view)
	}
	if !strings.Contains(view, "[12:00:01] processing") {
		t.Fatalf("expected completed stacked layout to keep log content visible\nview:\n%s", view)
	}

	lines := strings.Split(strings.TrimRight(view, "\n"), "\n")
	if len(lines) > m.height+1 {
		t.Fatalf("expected execution view to fit within %d lines, got %d\nview:\n%s", m.height+1, len(lines), view)
	}
}

func TestView_StackedRunningKeepsVisibleLogHeight(t *testing.T) {
	m := New("echo test")
	m.SetSize(100, 24)
	m.AddOutput("[12:00:00] starting")
	m.AddOutput("[12:00:01] processing")
	m.AddOutput("[12:00:02] finished")
	m.SetStatus(StatusRunning)

	if m.logViewport.Height < executionMinVisibleLogLines {
		t.Fatalf("expected running stacked layout to keep at least %d log lines, got %d", executionMinVisibleLogLines, m.logViewport.Height)
	}

	view := stripExecutionANSI(m.View())
	if !strings.Contains(view, "[12:00:01] processing") {
		t.Fatalf("expected running stacked layout to keep log content visible\nview:\n%s", view)
	}
}

func TestView_CompactModeAlwaysShowsLogs(t *testing.T) {
	m := New("echo test")
	m.SetSize(50, 16)
	m.AddOutput("[12:00:00] starting")
	m.AddOutput("[12:00:01] processing")
	m.SetStatus(StatusRunning)

	view := stripExecutionANSI(m.View())
	if !strings.Contains(view, "Execution Log") {
		t.Fatalf("expected compact execution header\nview:\n%s", view)
	}
	if !strings.Contains(view, "[12:00:01] processing") {
		t.Fatalf("expected compact execution mode to show logs\nview:\n%s", view)
	}
	if strings.Contains(view, "Pipeline Complete") || strings.Contains(view, "Progress") {
		t.Fatalf("did not expect compact execution mode to spend space on secondary panels\nview:\n%s", view)
	}
}

func TestView_WideLayoutStillShowsLogsFirst(t *testing.T) {
	m := New("echo test")
	m.SetSize(150, 55)
	m.AddOutput("[12:00:00] starting")
	m.AddOutput("[12:00:01] processing")
	m.SetStatus(StatusRunning)

	view := stripExecutionANSI(m.View())
	if !strings.Contains(view, "Log [2 lines") {
		t.Fatalf("expected wide execution layout to show log header\nview:\n%s", view)
	}
	if !strings.Contains(view, "[12:00:01] processing") {
		t.Fatalf("expected wide execution layout to show log content\nview:\n%s", view)
	}
}
