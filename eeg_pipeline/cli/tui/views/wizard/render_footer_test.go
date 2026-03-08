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
	// lines: ["", divider, hint]
	if len(lines) != 3 {
		t.Fatalf("expected blank, divider, and hint line, got %d lines: %q", len(lines), footer)
	}
	if got := lipgloss.Width(lines[2]); got > 60 {
		t.Fatalf("expected footer hint line to fit width, got %d > 60: %q", got, lines[2])
	}
}

func TestRenderFooterShowsValidationSummaryWithinWidth(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.CurrentStep = types.StepSelectComputations
	m.validationErrors = []string{
		"Select at least one analysis to run",
		"Select at least one valid subject",
	}

	footer := m.renderFooter(60)
	lines := strings.Split(footer, "\n")
	// lines: ["", divider, status, hint]
	if len(lines) != 4 {
		t.Fatalf("expected blank, divider, status, and hint line, got %d lines: %q", len(lines), footer)
	}
	if !strings.Contains(lines[2], "Select at least one analysis") {
		t.Fatalf("expected validation summary in footer status line, got %q", lines[2])
	}
	if got := lipgloss.Width(lines[2]); got > 60 {
		t.Fatalf("expected validation line to fit width, got %d > 60: %q", got, lines[2])
	}
}

func TestViewShowsValidationSummaryInShortTerminal(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.width = 80
	m.height = 20
	m.CurrentStep = types.StepSelectComputations
	m.validationErrors = []string{"Select at least one analysis to run"}

	view := m.View()
	if !strings.Contains(view, "Select at least one analysis to run") {
		t.Fatalf("expected short-terminal view to include validation summary, got: %q", view)
	}
}
