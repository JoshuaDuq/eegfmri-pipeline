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

