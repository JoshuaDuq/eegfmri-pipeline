package mainmenu

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestHandleEnter_SelectsPlottingPipelineFromUtilities(t *testing.T) {
	m := New()
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPlotting

	next, _ := m.handleEnter()
	updated, ok := next.(Model)
	if !ok {
		t.Fatalf("expected Model, got %T", next)
	}
	if updated.SelectedPipeline != int(types.PipelinePlotting) {
		t.Fatalf("expected SelectedPipeline=%d, got %d", int(types.PipelinePlotting), updated.SelectedPipeline)
	}
	if updated.SelectedUtility != -1 {
		t.Fatalf("expected no utility selection, got %d", updated.SelectedUtility)
	}
}

func TestHandleEnter_SelectsPipelineSmokeUtility(t *testing.T) {
	m := New()
	m.currentSection = SectionUtilities
	m.utilityCursor = UtilityPipelineSmokeTest

	next, _ := m.handleEnter()
	updated, ok := next.(Model)
	if !ok {
		t.Fatalf("expected Model, got %T", next)
	}
	if updated.SelectedUtility != UtilityPipelineSmokeTest {
		t.Fatalf("expected SelectedUtility=%d, got %d", UtilityPipelineSmokeTest, updated.SelectedUtility)
	}
	if updated.SelectedPipeline != -1 {
		t.Fatalf("expected no pipeline selection, got %d", updated.SelectedPipeline)
	}
}

func TestSetCursor_PlottingPipelineTargetsUtilitiesSection(t *testing.T) {
	m := New()

	m.SetCursor(int(types.PipelinePlotting))

	if m.currentSection != SectionUtilities {
		t.Fatalf("expected currentSection=%d, got %d", SectionUtilities, m.currentSection)
	}
	if m.utilityCursor != UtilityPlotting {
		t.Fatalf("expected utilityCursor=%d, got %d", UtilityPlotting, m.utilityCursor)
	}
}
