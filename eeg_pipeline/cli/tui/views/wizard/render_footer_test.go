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
	// lines: [divider, hint]
	if len(lines) != 2 {
		t.Fatalf("expected divider and hint line, got %d lines: %q", len(lines), footer)
	}
	if got := lipgloss.Width(lines[1]); got > 60 {
		t.Fatalf("expected footer hint line to fit width, got %d > 60: %q", got, lines[1])
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
	// lines: [divider, status, hint]
	if len(lines) != 3 {
		t.Fatalf("expected divider, status, and hint line, got %d lines: %q", len(lines), footer)
	}
	if !strings.Contains(lines[1], "Select at least one analysis") {
		t.Fatalf("expected validation summary in footer status line, got %q", lines[1])
	}
	if got := lipgloss.Width(lines[1]); got > 60 {
		t.Fatalf("expected validation line to fit width, got %d > 60: %q", got, lines[1])
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

func TestRenderFooterOmitsLeadingBlankLine(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.CurrentStep = types.StepSelectComputations

	footer := m.renderFooter(60)
	lines := strings.Split(footer, "\n")

	if len(lines) != 2 {
		t.Fatalf("expected footer to render divider and hints only, got %d lines: %q", len(lines), footer)
	}
	if strings.TrimSpace(lines[0]) == "" {
		t.Fatalf("expected footer to start with the divider, got blank line: %q", footer)
	}
}

func TestRenderFooterLeftAlignsHintRow(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.CurrentStep = types.StepSelectComputations

	footer := m.renderFooter(60)
	lines := strings.Split(footer, "\n")

	if len(lines) != 2 {
		t.Fatalf("expected footer divider and hints only, got %d lines: %q", len(lines), footer)
	}
	if strings.HasPrefix(lines[1], " ") {
		t.Fatalf("expected footer hints to start flush left, got %q", lines[1])
	}
}
