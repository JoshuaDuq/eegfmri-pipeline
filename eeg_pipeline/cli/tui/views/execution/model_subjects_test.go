package execution

import (
	"strings"
	"testing"
)

func TestProcessOutputLine_TracksOrderedSubjectsAndAvoidsDuplicateCompletion(t *testing.T) {
	m := New("eeg-pipeline features compute")

	m.processOutputLine(`{"event":"start","total_subjects":3,"subjects":["sub-01","sub-02","sub-03"]}`)
	m.processOutputLine(`{"event":"subject_start","subject":"sub-01"}`)
	m.processOutputLine(`{"event":"subject_done","subject":"sub-01"}`)
	m.processOutputLine(`{"event":"subject_start","subject":"sub-02"}`)

	if got, want := strings.Join(m.SubjectOrder, ","), "sub-01,sub-02,sub-03"; got != want {
		t.Fatalf("expected ordered subjects %q, got %q", want, got)
	}
	if got := m.subjectStatus("sub-01"); got != subjectDone {
		t.Fatalf("expected sub-01 to be done, got %q", got)
	}
	if got := m.subjectStatus("sub-02"); got != subjectRunning {
		t.Fatalf("expected sub-02 to be running, got %q", got)
	}
	if len(m.SubjectDurations) != 1 {
		t.Fatalf("expected one completed-subject duration, got %d", len(m.SubjectDurations))
	}
}

func TestRenderProgressSection_ShowsSubjectIDsAndFailures(t *testing.T) {
	m := New("eeg-pipeline features compute")
	m.SetSize(120, 32)
	m.Status = StatusRunning
	m.Progress = 0.5
	m.SubjectTotal = 5
	m.SubjectCurrent = 3
	m.CurrentSubject = "sub-03"
	m.CurrentOperation = "features"
	m.OperationCurrent = 2
	m.OperationTotal = 4
	m.SubjectOrder = []string{"sub-01", "sub-02", "sub-03", "sub-04", "sub-05"}
	m.SubjectStatuses["sub-01"] = string(subjectDone)
	m.SubjectStatuses["sub-02"] = string(subjectFailed)
	m.SubjectStatuses["sub-03"] = string(subjectRunning)
	m.SubjectStatuses["sub-04"] = string(subjectPending)
	m.SubjectStatuses["sub-05"] = string(subjectPending)
	m.FailedSubjects = []string{"sub-02"}

	section := stripANSI(m.renderProgressSection())

	required := []string{
		"Subjects 3/5 60%",
		"sub-02",
		"sub-03",
		"Failed: sub-02",
	}
	for _, item := range required {
		if !strings.Contains(section, item) {
			t.Fatalf("expected progress section to contain %q\nsection:\n%s", item, section)
		}
	}
}
