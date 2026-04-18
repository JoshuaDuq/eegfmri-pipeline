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
	if !strings.Contains(view, "Wizard Shortcuts") || !strings.Contains(view, "NAVIGATION") || !strings.Contains(view, "SELECTION") {
		t.Fatalf("unexpected help overlay view: %q", view)
	}
}

func TestHelpOverlayViewDoesNotWrapLongDescriptions(t *testing.T) {
	help := NewHelpOverlay("Wizard Shortcuts", 40)
	help.AddSection("Navigation", []HelpItem{{
		Key:         "Ctrl+Shift+Alt+N",
		Description: "Move cursor through an intentionally long description that should truncate instead of wrapping",
	}})
	help.Toggle()

	view := stripComponentsANSI(help.View())
	lines := strings.Split(strings.TrimSuffix(view, "\n"), "\n")
	if len(lines) != 11 {
		t.Fatalf("expected 11 overlay lines without wrapped continuations, got %d\nview:\n%s", len(lines), view)
	}
	if !strings.Contains(view, "...") {
		t.Fatalf("expected truncated long help text, got:\n%s", view)
	}
}

func TestSpinnerAndScrollIndicator(t *testing.T) {
	spinner := NewSpinner("Loading")
	first := stripComponentsANSI(spinner.View())
	if !strings.Contains(first, "Loading") {
		t.Fatalf("unexpected spinner view: %q", first)
	}
	if !containsAny(first, spinnerFrames) {
		t.Fatalf("expected spinner to render a braille glyph, got %q", first)
	}

	for i := 0; i < spinnerFrameTicks; i++ {
		spinner.Tick()
	}
	second := stripComponentsANSI(spinner.View())
	if first == second {
		t.Fatal("expected spinner frame to advance after enough ticks")
	}
	if !containsAny(second, spinnerFrames) {
		t.Fatalf("expected spinner to render a braille glyph after Tick, got %q", second)
	}

	bare := stripComponentsANSI(NewSpinner("").View())
	if strings.ContainsRune(bare, ' ') && strings.TrimSpace(bare) == "" {
		t.Fatalf("expected unlabeled spinner to render only the glyph, got %q", bare)
	}

	indicator := ScrollIndicator{Current: 2, Total: 10, ViewHeight: 5}
	if !indicator.CanScrollUp() || !indicator.CanScrollDown() {
		t.Fatalf("unexpected scroll indicator state: %+v", indicator)
	}
	if got := stripComponentsANSI(indicator.View()); !strings.Contains(got, "▲") || !strings.Contains(got, "▼") {
		t.Fatalf("unexpected scroll indicator view: %q", got)
	}
}

func containsAny(s string, options []string) bool {
	for _, opt := range options {
		if strings.Contains(s, opt) {
			return true
		}
	}
	return false
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

func TestDotsLoader(t *testing.T) {
	loader := NewDotsLoader("Computing")

	// Frame 0: label + "   " (blank dots)
	view0 := stripComponentsANSI(loader.View())
	if !strings.Contains(view0, "Computing") {
		t.Fatalf("DotsLoader frame 0 = %q, want label present", view0)
	}

	// Advance 3 ticks → frame 1: "·  "
	for i := 0; i < 3; i++ {
		loader.Advance()
	}
	view1 := stripComponentsANSI(loader.View())
	if !strings.Contains(view1, "·") {
		t.Fatalf("DotsLoader frame 1 = %q, want at least one dot", view1)
	}

	// Advance 3 more ticks → frame 2: "·· "
	for i := 0; i < 3; i++ {
		loader.Advance()
	}
	view2 := stripComponentsANSI(loader.View())
	if strings.Count(view2, "·") < 2 {
		t.Fatalf("DotsLoader frame 2 = %q, want at least two dots", view2)
	}

	// Advance 3 more ticks → frame 3: "···"
	for i := 0; i < 3; i++ {
		loader.Advance()
	}
	view3 := stripComponentsANSI(loader.View())
	if strings.Count(view3, "·") < 3 {
		t.Fatalf("DotsLoader frame 3 = %q, want three dots", view3)
	}

	// Advance 3 more ticks → wraps back to frame 0
	for i := 0; i < 3; i++ {
		loader.Advance()
	}
	view4 := stripComponentsANSI(loader.View())
	if view4 != view0 {
		t.Fatalf("DotsLoader wrap: frame 4 = %q, want same as frame 0 %q", view4, view0)
	}
}
