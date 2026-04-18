package globalsetup

import (
	"regexp"
	"strings"
	"testing"
)

var globalSetupRenderANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripGlobalSetupRenderANSI(s string) string {
	return globalSetupRenderANSIPattern.ReplaceAllString(s, "")
}

func TestView_UsesEditorialHeaderAndUppercaseTabs(t *testing.T) {
	m := New(".")
	m.width = 100
	m.height = 28

	view := stripGlobalSetupRenderANSI(m.View())

	required := []string{
		"Global Setup",
		"PROJECT",
		"PATHS",
	}
	for _, item := range required {
		if !strings.Contains(view, item) {
			t.Fatalf("expected view to contain %q, got:\n%s", item, view)
		}
	}
	if strings.Contains(view, "Project") || strings.Contains(view, "Paths") {
		t.Fatalf("expected section tabs to render uppercase labels, got:\n%s", view)
	}
}
