package mainmenu

import "testing"

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
