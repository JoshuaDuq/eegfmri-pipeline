package dashboard

import (
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

var footerVisualAnsi = regexp.MustCompile(`\x1b\[[0-9;]*m`)

// Two-hint footers must split across the full width (Refresh left, Back
// right) instead of clustering both hints at the left edge with only a
// 9-cell middot gap between them.
func TestFooterAnchorsRefreshLeftAndBackRight(t *testing.T) {
	m := New(".")
	m.width = 120
	m.height = 32

	footer := m.renderFooter(m.contentWidth())
	lines := strings.Split(footer, "\n")
	if len(lines) < 2 {
		t.Fatalf("expected divider + bar, got %q", footer)
	}
	bar := footerVisualAnsi.ReplaceAllString(lines[1], "")

	if !strings.HasPrefix(strings.TrimLeft(bar, " "), "[R] Refresh") {
		t.Fatalf("expected Refresh anchored left, got %q", bar)
	}
	if !strings.HasSuffix(strings.TrimRight(bar, " "), "[Esc] Back") {
		t.Fatalf("expected Back anchored right, got %q", bar)
	}
	if w := lipgloss.Width(lines[1]); w < 100 {
		t.Fatalf("expected bar to span footer width (>=100), got %d", w)
	}
}
