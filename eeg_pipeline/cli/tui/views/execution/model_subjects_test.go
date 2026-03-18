package execution

import (
	"reflect"
	"strings"
	"testing"
)

func TestRecordFailedSubject_DeduplicatesSubjects(t *testing.T) {
	m := Model{}

	m.recordFailedSubject("sub-01")
	m.recordFailedSubject("sub-01")
	m.recordFailedSubject("sub-02")

	want := []string{"sub-01", "sub-02"}
	if !reflect.DeepEqual(m.FailedSubjects, want) {
		t.Fatalf("unexpected failed subjects: got=%#v want=%#v", m.FailedSubjects, want)
	}
}

func TestSubjectAnchorIndex_PrioritizesCurrentSubjectAndCursorBounds(t *testing.T) {
	m := Model{
		SubjectOrder: []string{"sub-01", "sub-02", "sub-03", "sub-04"},
	}

	tests := []struct {
		name         string
		current      string
		subjectIndex int
		want         int
	}{
		{
			name:    "current subject wins",
			current: "sub-03",
			want:    2,
		},
		{
			name:         "cursor below range clamps to start",
			subjectIndex: -1,
			want:         0,
		},
		{
			name:         "cursor in range is used directly",
			subjectIndex: 2,
			want:         2,
		},
		{
			name:         "cursor past range clamps to end",
			subjectIndex: 99,
			want:         3,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m.CurrentSubject = tc.current
			m.SubjectCurrent = tc.subjectIndex

			if got := m.subjectAnchorIndex(); got != tc.want {
				t.Fatalf("unexpected anchor index: got=%d want=%d", got, tc.want)
			}
		})
	}
}

func TestVisibleSubjectWindow_ClampsToAvailableSubjects(t *testing.T) {
	m := Model{
		SubjectOrder: []string{"sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"},
	}

	tests := []struct {
		name         string
		current      string
		subjectIndex int
		maxVisible   int
		wantStart    int
		wantEnd      int
	}{
		{
			name:       "all subjects fit",
			maxVisible: 10,
			wantStart:  0,
			wantEnd:    7,
		},
		{
			name:       "window centers on current subject",
			current:    "sub-04",
			maxVisible: 3,
			wantStart:  2,
			wantEnd:    5,
		},
		{
			name:         "window clamps near end",
			subjectIndex: 6,
			maxVisible:   3,
			wantStart:    4,
			wantEnd:      7,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m.CurrentSubject = tc.current
			m.SubjectCurrent = tc.subjectIndex

			start, end := m.visibleSubjectWindow(tc.maxVisible)
			if start != tc.wantStart || end != tc.wantEnd {
				t.Fatalf("unexpected window: got=(%d,%d) want=(%d,%d)", start, end, tc.wantStart, tc.wantEnd)
			}
		})
	}
}

func TestRenderSubjectSummary_ClampsToTotal(t *testing.T) {
	m := Model{
		SubjectTotal:   3,
		SubjectCurrent: 8,
	}

	got := stripExecutionANSI(m.renderSubjectSummary(80))
	if !strings.Contains(got, "Subjects 3/3 100%") {
		t.Fatalf("unexpected subject summary: %q", got)
	}
}

func TestRenderSubjectLane_ReturnsEmptyWhenUnrenderable(t *testing.T) {
	if got := (Model{}).renderSubjectLane(0); got != "" {
		t.Fatalf("expected empty lane for zero width, got %q", got)
	}
	if got := (Model{}).renderSubjectLane(40); got != "" {
		t.Fatalf("expected empty lane without subjects, got %q", got)
	}
}

func TestRenderSubjectLane_RendersAnchoredWindow(t *testing.T) {
	m := Model{
		SubjectOrder: []string{
			"sub-01",
			"sub-02",
			"sub-03",
			"sub-04",
			"sub-05",
			"sub-06",
			"sub-07",
		},
		SubjectCurrent:  3,
		SubjectStatuses: map[string]string{},
	}

	got := stripExecutionANSI(m.renderSubjectLane(60))
	if got == "" {
		t.Fatalf("expected rendered lane, got empty string")
	}
	if !strings.Contains(got, "sub-04") {
		t.Fatalf("expected anchored subject to remain visible, got %q", got)
	}
	if !strings.Contains(got, "+1") {
		t.Fatalf("expected truncated lane to show remaining subjects, got %q", got)
	}
}
