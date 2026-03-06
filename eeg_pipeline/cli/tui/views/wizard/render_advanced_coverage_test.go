package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestFeaturesAdvancedConfigRendersMaximalState(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.height = 200
	for i := range m.categories {
		m.selected[i] = true
	}
	m.featGroupConnectivityExpanded = true
	m.featGroupDirectedConnExpanded = true
	m.featGroupPACExpanded = true
	m.featGroupAperiodicExpanded = true
	m.featGroupComplexityExpanded = true
	m.featGroupBurstsExpanded = true
	m.featGroupPowerExpanded = true
	m.featGroupSpectralExpanded = true
	m.featGroupERPExpanded = true
	m.featGroupRatiosExpanded = true
	m.featGroupAsymmetryExpanded = true
	m.featGroupQualityExpanded = true
	m.featGroupMicrostatesExpanded = true
	m.featGroupERDSExpanded = true
	m.featGroupSpatialTransformExpanded = true
	m.featGroupSourceLocExpanded = true
	m.featGroupITPCExpanded = true
	m.featGroupTFRExpanded = true
	m.featGroupStorageExpanded = true
	m.featGroupExecutionExpanded = true
	m.connGranularity = 1
	m.itpcMethod = 3
	m.iafEnabled = true
	m.featComputeChangeScores = true
	m.sourceLocMode = 1
	m.sourceLocMethod = 1
	m.sourceLocContrastEnabled = true
	m.sourceLocCreateTrans = false
	m.sourceLocCreateBemSolution = false
	m.sourceLocFmriEnabled = true
	m.sourceLocFmriMinClusterMM3 = 0
	m.sourceLocFmriContrastEnabled = true
	m.sourceLocFmriContrastType = 0
	m.sourceLocFmriAutoDetectRuns = false
	m.sourceLocFmriClusterCorrection = true

	rendered := m.renderFeaturesAdvancedConfig()
	if strings.Contains(rendered, "Unknown:") {
		t.Fatalf("features advanced config rendered an unknown option:\n%s", rendered)
	}
}

func TestBehaviorAdvancedConfigRendersMaximalState(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.height = 200
	for i := range m.computations {
		m.computationSelected[i] = true
	}
	m.behaviorGroupGeneralExpanded = true
	m.behaviorGroupStatsExpanded = true
	m.behaviorGroupTrialTableExpanded = true
	m.behaviorGroupPredictorResidualExpanded = true
	m.behaviorGroupCorrelationsExpanded = true
	m.behaviorGroupRegressionExpanded = true
	m.behaviorGroupConditionExpanded = true
	m.behaviorGroupTemporalExpanded = true
	m.behaviorGroupClusterExpanded = true
	m.behaviorGroupReportExpanded = true
	m.behaviorGroupOutputExpanded = true
	m.behaviorGroupAdvancedExpanded = true
	m.runAdjustmentEnabled = true
	m.predictorResidualEnabled = true
	m.predictorResidualCrossfitEnabled = true
	m.predictorType = 0
	m.regressionTempControl = 2
	m.clusterCorrectionEnabled = true
	m.featureFileSelected["itpc"] = true
	m.featureFileSelected["erds"] = true

	rendered := m.renderBehaviorAdvancedConfig()
	if strings.Contains(rendered, "Unknown:") {
		t.Fatalf("behavior advanced config rendered an unknown option:\n%s", rendered)
	}
}

func TestPlottingGlobalOptionsAreRendered(t *testing.T) {
	m := New(types.PipelinePlotting, ".")
	m.plotGroupDefaultsExpanded = true
	m.plotGroupFontsExpanded = true
	m.plotGroupLayoutExpanded = true
	m.plotGroupFigureSizesExpanded = true
	m.plotGroupColorsExpanded = true
	m.plotGroupAlphaExpanded = true
	m.plotGroupTopomapExpanded = true
	m.plotGroupTFRExpanded = true
	m.plotGroupSourceLocExpanded = true

	for _, opt := range m.getGlobalStylingOptions() {
		rendered := flattenRenderLines(m.renderOption(opt, 24, false))
		if strings.Contains(rendered, "(unwired)") {
			t.Fatalf("plotting option %v rendered as unwired: %s", opt, rendered)
		}
	}
}

func TestPlottingPerPlotFieldsAreRendered(t *testing.T) {
	m := New(types.PipelinePlotting, ".")

	for _, plot := range defaultPlotItems {
		for _, field := range m.plotConfigFields(plot) {
			row := plottingAdvancedRow{
				kind:      plottingRowPlotField,
				plotID:    plot.ID,
				plotField: field,
			}
			rendered := flattenRenderLines(m.renderPlotField(row, 24, false))
			if strings.Contains(rendered, "Unknown plot field") {
				t.Fatalf("plot %q field %v is not rendered: %s", plot.ID, field, rendered)
			}
		}
	}
}

func flattenRenderLines(lines []renderLine) string {
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		parts = append(parts, line.text)
	}
	return strings.Join(parts, "\n")
}
