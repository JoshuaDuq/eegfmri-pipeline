package execution

import (
	"strings"
	"testing"
	"time"

	"github.com/eeg-pipeline/tui/messages"
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

func TestProcessOutputLine_SubjectDoneFalseMarksSubjectFailed(t *testing.T) {
	m := New("eeg-pipeline features compute --subject 0001")

	m.processOutputLine(`{"event":"start","subjects":["sub-0001"],"total_subjects":1}`)
	m.processOutputLine(`{"event":"subject_start","subject":"sub-0001"}`)
	m.processOutputLine(`{"event":"subject_done","subject":"sub-0001","success":false}`)

	if got := m.subjectStatus("sub-0001"); got != subjectFailed {
		t.Fatalf("expected failed subject status, got %q", got)
	}
	if len(m.FailedSubjects) != 1 || m.FailedSubjects[0] != "sub-0001" {
		t.Fatalf("expected failed subject to be recorded, got %#v", m.FailedSubjects)
	}
}

func TestProcessOutputLine_CompleteFalseDoesNotForceFullProgress(t *testing.T) {
	m := New("eeg-pipeline features compute --subject 0001")
	m.Progress = 0.25

	m.processOutputLine(`{"event":"complete","success":false}`)

	if m.Progress != 0.25 {
		t.Fatalf("expected failed completion to preserve progress, got %.2f", m.Progress)
	}
}

func TestUpdate_CommandFailureDoesNotForceFullProgress(t *testing.T) {
	m := New("eeg-pipeline features compute --subject 0001")
	m.Status = StatusRunning
	m.Progress = 0.25

	updatedModel, _ := m.Update(messages.CommandDoneMsg{ExitCode: 1, Success: false})
	updated := updatedModel.(Model)

	if updated.Progress != 0.25 {
		t.Fatalf("expected command failure to preserve progress, got %.2f", updated.Progress)
	}
}

func TestProcessOutputLine_AccurateStepProgress(t *testing.T) {
	m := New("eeg-pipeline features compute")
	m.processOutputLine(`{"event":"start","subjects":["sub-01","sub-02"],"total_subjects":2}`)
	m.processOutputLine(`{"event":"subject_start","subject":"sub-01"}`)

	if m.Progress != 0.0 {
		t.Fatalf("expected progress at start to be 0.0, got %.2f", m.Progress)
	}

	// 1st step out of 5: stepFraction = (1-1)/5 = 0.0. stepContrib = 0.0. m.Progress = 0.0
	m.processOutputLine(`{"event":"progress","step":"Loading epochs","current":1,"total":5}`)
	if m.Progress != 0.0 {
		t.Fatalf("expected progress at 1st step to be 0.0, got %.2f", m.Progress)
	}

	// 5th step out of 5: stepFraction = (5-1)/5 = 0.8. stepContrib = 0.8/2 = 0.4. m.Progress = 0.4
	m.processOutputLine(`{"event":"progress","step":"ICA","current":5,"total":5}`)
	if m.Progress != 0.4 {
		t.Fatalf("expected progress at 5th step to be 0.4, got %.2f", m.Progress)
	}

	// Subject 1 finishes: m.Progress = 1/2 = 0.5
	m.processOutputLine(`{"event":"subject_done","subject":"sub-01","success":true}`)
	if m.Progress != 0.5 {
		t.Fatalf("expected progress after sub-01 done to be 0.5, got %.2f", m.Progress)
	}

	// Subject 2 starts: m.Progress = (2-1)/2 = 0.5. Seamless transition, no jumping backward!
	m.processOutputLine(`{"event":"subject_start","subject":"sub-02"}`)
	if m.Progress != 0.5 {
		t.Fatalf("expected progress at start of sub-02 to remain 0.5, got %.2f", m.Progress)
	}
}

func TestProcessOutputLine_ParallelProgressJumpingPrevention(t *testing.T) {
	m := New("eeg-pipeline preprocessing --subjects sub-01,sub-02")
	m.processOutputLine(`{"event":"start","subjects":["sub-01","sub-02"],"total_subjects":2}`)

	m.processOutputLine(`{"event":"progress","step":"Loading epochs","current":1,"total":5}`)
	if m.Progress != 0.0 {
		t.Fatalf("expected parallel progress at 1st step to be 0.0, got %.2f", m.Progress)
	}

	m.processOutputLine(`{"event":"progress","step":"ICA","current":3,"total":5}`)
	if m.Progress != 0.4 {
		t.Fatalf("expected parallel progress at 3rd step to be 0.4, got %.2f", m.Progress)
	}
}
