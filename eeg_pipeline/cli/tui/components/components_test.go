package components

import (
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

var componentsANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripComponentsANSI(s string) string {
	return componentsANSIPattern.ReplaceAllString(s, "")
}

func TestToastLifecycle(t *testing.T) {
	toast := NewToast("Saved", ToastSuccess, 1)
	if !toast.Visible || toast.TicksLeft != 1 {
		t.Fatalf("unexpected toast state: %+v", toast)
	}

	view := stripComponentsANSI(toast.View())
	if !strings.Contains(view, "Saved") || !strings.Contains(view, "✓") {
		t.Fatalf("unexpected toast view: %q", view)
	}

	toast.Tick()
	if toast.Visible {
		t.Fatal("expected toast to hide after duration expires")
	}
	if got := toast.View(); got != "" {
		t.Fatalf("expected hidden toast to render empty string, got %q", got)
	}
}

func TestHelpOverlayView(t *testing.T) {
	help := NewHelpOverlay("Wizard Shortcuts", 40)
	help.AddSection("Navigation", []HelpItem{{Key: "↑/↓", Description: "Move cursor"}})
	help.AddSection("Selection", []HelpItem{{Key: "Space", Description: "Toggle"}})

	if got := help.View(); got != "" {
		t.Fatalf("expected hidden help overlay to render empty string, got %q", got)
	}

	help.Toggle()
	view := stripComponentsANSI(help.View())
	if !strings.Contains(view, "Wizard Shortcuts") || !strings.Contains(view, "Navigation") || !strings.Contains(view, "Selection") {
		t.Fatalf("unexpected help overlay view: %q", view)
	}
}

func TestSpinnerAndScrollIndicator(t *testing.T) {
	spinner := NewSpinner("Loading")
	first := stripComponentsANSI(spinner.View())
	if !strings.Contains(first, "Loading") {
		t.Fatalf("unexpected spinner view: %q", first)
	}
	spinner.Tick()
	second := stripComponentsANSI(spinner.View())
	if first == second {
		t.Fatal("expected spinner frame to advance after Tick")
	}

	indicator := ScrollIndicator{Current: 2, Total: 10, ViewHeight: 5}
	if !indicator.CanScrollUp() || !indicator.CanScrollDown() {
		t.Fatalf("unexpected scroll indicator state: %+v", indicator)
	}
	if got := stripComponentsANSI(indicator.View()); !strings.Contains(got, "▲") || !strings.Contains(got, "▼") {
		t.Fatalf("unexpected scroll indicator view: %q", got)
	}
}

func TestInfoPanelView(t *testing.T) {
	panel := NewInfoPanel("Summary", 12)
	panel.AddRow("Alpha", "1")
	panel.AddStyledRow("Beta", "2", lipgloss.NewStyle().Bold(true))

	view := stripComponentsANSI(panel.View())
	if !strings.Contains(view, "Summary") || !strings.Contains(view, "Alpha") || !strings.Contains(view, "Beta") {
		t.Fatalf("unexpected info panel view: %q", view)
	}
}

