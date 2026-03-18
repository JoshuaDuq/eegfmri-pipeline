package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/styles"
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

func TestFormatChannelList_WrapsLongChannelLists(t *testing.T) {
	m := Model{contentWidth: 20}
	channels := []string{"A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"}

	rendered := m.formatChannelList(channels, styles.Accent)
	if rendered == "" {
		t.Fatalf("expected wrapped channel list, got empty string")
	}
	if !strings.Contains(rendered, "\n") {
		t.Fatalf("expected wrapped channel list to span multiple lines, got %q", rendered)
	}
	if !strings.Contains(rendered, "A01") || !strings.Contains(rendered, "A08") {
		t.Fatalf("expected channel list to preserve first and last entries, got %q", rendered)
	}
}

func TestFormatChannelList_ReturnsEmptyForNoChannels(t *testing.T) {
	m := Model{contentWidth: 80}

	if got := m.formatChannelList(nil, styles.Accent); got != "" {
		t.Fatalf("expected empty channel list for nil input, got %q", got)
	}
}

func TestRenderSpatialSelection_ShowsSelectionCountAndModes(t *testing.T) {
	m := Model{
		contentWidth:    80,
		spatialSelected: map[int]bool{0: true, 1: false, 2: true},
		spatialCursor:   1,
	}

	rendered := m.renderSpatialSelection()
	if !strings.Contains(rendered, "2/3") {
		t.Fatalf("expected selection count in spatial renderer, got %q", rendered)
	}
	if !strings.Contains(rendered, "ROI") ||
		!strings.Contains(rendered, "All Channels") ||
		!strings.Contains(rendered, "Global") {
		t.Fatalf("expected all spatial modes to be rendered, got %q", rendered)
	}
}

func TestRenderFeatureFileSelection_ShowsApplicableFilesAndAvailability(t *testing.T) {
	m := New(types.PipelineBehavior, "/tmp")
	m.contentWidth = 100
	for i := range m.computationSelected {
		m.computationSelected[i] = false
	}
	for i, comp := range m.computations {
		if comp.Key == "temporal" {
			m.computationSelected[i] = true
			break
		}
	}
	m.featureAvailability = map[string]bool{
		"power": true,
		"itpc":  false,
		"erds":  true,
	}
	m.featureLastModified = map[string]string{
		"power": "",
		"erds":  "",
	}
	m.featureFileCursor = 99

	rendered := m.renderFeatureFileSelection()
	if !strings.Contains(rendered, "Showing applicable features only") {
		t.Fatalf("expected applicable-feature notice, got %q", rendered)
	}
	if !strings.Contains(rendered, "3/3") {
		t.Fatalf("expected selected count for applicable files, got %q", rendered)
	}
	if !strings.Contains(rendered, "2 available") {
		t.Fatalf("expected available count for applicable files, got %q", rendered)
	}
	if !strings.Contains(rendered, "Power Features") ||
		!strings.Contains(rendered, "ITPC") ||
		!strings.Contains(rendered, "ERDS") {
		t.Fatalf("expected applicable feature names to be rendered, got %q", rendered)
	}
	if !strings.Contains(rendered, styles.CheckMark) {
		t.Fatalf("expected available feature marker, got %q", rendered)
	}
	if !strings.Contains(rendered, styles.CrossMark) {
		t.Fatalf("expected unavailable feature marker, got %q", rendered)
	}
}
