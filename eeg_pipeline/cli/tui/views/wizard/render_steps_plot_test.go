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

func TestPlotConfigFields_CrossFrequencyPowerCorrelationIncludesComparisonControls(t *testing.T) {
	m := New(types.PipelinePlotting, ".")
	fields := m.plotConfigFields(PlotItem{ID: "cross_frequency_power_correlation", Group: "power"})

	requiredFields := []plotItemConfigField{
		plotItemConfigFieldComparisonSegment,
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
			t.Fatalf("expected cross_frequency_power_correlation config fields to include %v, got %v", requiredField, fields)
		}
	}
}

func TestRenderPlotSelection_RestingStateShowsPowerCompatibilityBadges(t *testing.T) {
	m := New(types.PipelinePlotting, ".")
	m.contentWidth = 120
	m.prepTaskIsRest = true
	m.plotItems = []PlotItem{
		{ID: "power_spectral_density", Group: "power", Name: "PSD Summary", RestCompatibility: plotRestCompatible},
		{ID: "power_by_condition", Group: "power", Name: "Condition Comparison", RestCompatibility: plotRestTaskOnly},
	}
	m.plotSelected = map[int]bool{0: true, 1: true}

	rendered := m.renderPlotSelection()

	if !strings.Contains(rendered, "rest-compatible") {
		t.Fatalf("expected resting-state compatible badge in plot selection render, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "task-only") {
		t.Fatalf("expected task-only badge in plot selection render, got:\n%s", rendered)
	}
}

func TestDefaultPlotItems_ClassifyConnectivityRestCompatibility(t *testing.T) {
	plotByID := make(map[string]PlotItem, len(defaultPlotItems))
	for _, plot := range defaultPlotItems {
		plotByID[plot.ID] = plot
	}

	if plotByID["connectivity_circle"].RestCompatibility != plotRestCompatible {
		t.Fatalf(
			"expected connectivity_circle to be rest-compatible, got %q",
			plotByID["connectivity_circle"].RestCompatibility,
		)
	}

	if plotByID["connectivity_by_condition"].RestCompatibility != plotRestTaskOnly {
		t.Fatalf(
			"expected connectivity_by_condition to be task-only, got %q",
			plotByID["connectivity_by_condition"].RestCompatibility,
		)
	}
	if plotByID["connectivity_circle_condition"].RestCompatibility != plotRestTaskOnly {
		t.Fatalf(
			"expected connectivity_circle_condition to be task-only, got %q",
			plotByID["connectivity_circle_condition"].RestCompatibility,
		)
	}
	if plotByID["connectivity_heatmap"].RestCompatibility != plotRestCompatible {
		t.Fatalf(
			"expected connectivity_heatmap to be rest-compatible, got %q",
			plotByID["connectivity_heatmap"].RestCompatibility,
		)
	}
	if plotByID["connectivity_network"].RestCompatibility != plotRestCompatible {
		t.Fatalf(
			"expected connectivity_network to be rest-compatible, got %q",
			plotByID["connectivity_network"].RestCompatibility,
		)
	}
}

func TestPlotConfigFields_ConnectivityCircleIncludesOnlySummaryControls(t *testing.T) {
	m := New(types.PipelinePlotting, ".")
	fields := m.plotConfigFields(PlotItem{ID: "connectivity_circle", Group: "connectivity"})

	requiredFields := []plotItemConfigField{
		plotItemConfigFieldConnectivityCircleTopFraction,
		plotItemConfigFieldConnectivityCircleMinLines,
		plotItemConfigFieldComparisonSegment,
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
			t.Fatalf("expected connectivity_circle config fields to include %v, got %v", requiredField, fields)
		}
	}

	for _, forbiddenField := range []plotItemConfigField{
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	} {
		for _, field := range fields {
			if field == forbiddenField {
				t.Fatalf("did not expect connectivity_circle config fields to include %v; got %v", forbiddenField, fields)
			}
		}
	}
}
