package wizard

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/types"
)

func TestRenderFooterHintsFitsNarrowWidth(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.CurrentStep = types.StepSelectComputations

	footer := m.renderFooter(60)
	lines := strings.Split(footer, "\n")
	if len(lines) != 3 {
		t.Fatalf("expected divider, padding, and single footer line, got %d lines: %q", len(lines), footer)
	}
	if got := lipgloss.Width(lines[2]); got > 60 {
		t.Fatalf("expected footer hint line to fit width, got %d > 60: %q", got, lines[2])
	}
}
