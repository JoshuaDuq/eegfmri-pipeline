package execution

import (
	"strings"
	"testing"
	"time"
)

func TestRenderProgressSection_HidesCommandPreview(t *testing.T) {
	m := New("eeg-pipeline features compute --subject 0001")
	m.StartTime = time.Now().Add(-30 * time.Second)

	section := m.renderProgressSection()

	if strings.Contains(section, "Cmd") {
		t.Fatalf("expected command preview to be hidden, but found Cmd label in progress section")
	}
}
