package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
)

func TestRenderTimeRange_RestingStateShowsImplicitFullEpochGuidance(t *testing.T) {
	model := New(types.PipelineFeatures, ".")
	model.contentWidth = 100
	model.modeIndex = 0
	model.modeOptions = []string{styles.ModeCompute}
	model.prepTaskIsRest = true

	rendered := model.renderTimeRange()

	if strings.HasPrefix(rendered, "\n") {
		t.Fatalf("expected time range view to start without a blank line, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "The pipeline defaults to a full-epoch analysis window when no explicit time range is provided.") {
		t.Fatalf("expected resting-state full-epoch guidance in time range view, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "No explicit time ranges defined. Press [A] to add one.") {
		t.Fatalf("expected resting-state empty-state copy in time range view, got:\n%s", rendered)
	}
}
