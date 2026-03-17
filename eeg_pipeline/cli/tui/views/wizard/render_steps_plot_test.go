package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
)

func TestRenderTimeRange_RestingStateShowsImplicitFullEpochGuidance(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.contentWidth = 100
	m.modeIndex = 0
	m.modeOptions = []string{styles.ModeCompute}
	m.prepTaskIsRest = true

	rendered := m.renderTimeRange()

	if !strings.Contains(rendered, "The pipeline defaults to a full-epoch analysis window when no explicit time range is provided.") {
		t.Fatalf("expected resting-state full-epoch guidance in time range view, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "No explicit time ranges defined. Press [A] to add one.") {
		t.Fatalf("expected resting-state empty-state copy in time range view, got:\n%s", rendered)
	}
}

func TestPlotConfigFields_PowerTimecourseIncludesConditionAndROIFields(t *testing.T) {
	m := New(types.PipelinePlotting, ".")
	fields := m.plotConfigFields(PlotItem{ID: "power_timecourse", Group: "power"})

	requiredFields := []plotItemConfigField{
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
		plotItemConfigFieldComparisonROIs,
	}

	for _, requiredField := range requiredFields {
		found := false
		for _, field := range fields {
			if field == requiredField {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected power_timecourse config fields to include %v, got %v", requiredField, fields)
		}
	}
}
