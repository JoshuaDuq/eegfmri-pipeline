package execution

import (
	"strings"
	"testing"
	"time"
)

func TestRenderInfoPanel_HidesCommandPreview(t *testing.T) {
	m := New("eeg-pipeline features compute --subject 0001")
	m.StartTime = time.Now().Add(-30 * time.Second)

	panel := m.renderInfoPanel()

	if strings.Contains(panel, "Cmd") {
		t.Fatalf("expected command preview to be hidden, but found Cmd label in info panel")
	}
}
