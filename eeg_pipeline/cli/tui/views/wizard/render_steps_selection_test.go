package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestRenderSubjectSelection_ShowsExistingSubjectsWhileRefreshing(t *testing.T) {
	m := New(types.PipelineFeatures, "/tmp")
	m.height = 40
	m.contentWidth = 100
	m.SetSubjects([]types.SubjectStatus{
		{ID: "sub-0001", HasDerivatives: true},
	})
	m.SetSubjectsLoading()

	rendered := m.renderSubjectSelection()

	if !strings.Contains(rendered, "Refreshing subject status") {
		t.Fatalf("expected refresh status to be shown while subjects are loading")
	}
	if !strings.Contains(rendered, "sub-0001") {
		t.Fatalf("expected subject list to remain visible during refresh")
	}
}

func TestSetSubjectLoadError_ClearsStaleSubjectState(t *testing.T) {
	m := New(types.PipelineBehavior, "/tmp")
	m.height = 40
	m.contentWidth = 100
	m.SetSubjects([]types.SubjectStatus{
		{
			ID:             "sub-0001",
			HasDerivatives: true,
			FeatureAvailability: &types.FeatureAvailability{
				Features: map[string]types.AvailabilityInfo{
					"power": {Available: true},
				},
				Computations: map[string]types.AvailabilityInfo{
					"correlations": {Available: true},
				},
			},
		},
	})

	m.SetSubjectLoadError("discovery failed")

	if len(m.subjects) != 0 {
		t.Fatalf("expected subjects to be cleared after load error")
	}
	if len(m.subjectSelected) != 0 {
		t.Fatalf("expected selected-subject state to be cleared after load error")
	}
	if len(m.featureAvailability) != 0 {
		t.Fatalf("expected feature availability to be cleared after load error")
	}
	if len(m.computationAvailability) != 0 {
		t.Fatalf("expected computation availability to be cleared after load error")
	}

	rendered := m.renderSubjectSelection()
	if !strings.Contains(rendered, "Subject discovery failed") {
		t.Fatalf("expected discovery error message in render, got: %s", rendered)
	}
	if strings.Contains(rendered, "sub-0001") {
		t.Fatalf("did not expect stale subject to remain visible after load error")
	}
}
