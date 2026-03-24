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

func TestRenderProgressSection_HidesETA(t *testing.T) {
	m := New("eeg-pipeline features compute --subject 0001")
	m.Status = StatusRunning
	m.StartTime = time.Now().Add(-2 * time.Minute)
	m.SubjectTotal = 4
	m.SubjectCurrent = 2
	m.SubjectDurations = []time.Duration{time.Minute, 90 * time.Second}

	section := m.renderProgressSection()

	if strings.Contains(section, "ETA") {
		t.Fatalf("expected ETA to be hidden, got %q", section)
	}
}
