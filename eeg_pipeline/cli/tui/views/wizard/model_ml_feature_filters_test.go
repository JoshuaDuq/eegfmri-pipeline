package wizard

import (
	"slices"
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func optionIndex(items []optionType, want optionType) int {
	for i, item := range items {
		if item == want {
			return i
		}
	}
	return -1
}

func itemIndex(items []string, want string) int {
	for i, item := range items {
		if item == want {
			return i
		}
	}
	return -1
}

func TestToggleMLAdvancedOption_FeatureFamiliesUsesExpandedList(t *testing.T) {
	m := New(types.PipelineML, ".")
	idx := optionIndex(m.getMLOptions(), optMLFeatureFamilies)
	if idx < 0 {
		t.Fatalf("optMLFeatureFamilies missing from ML options")
	}

	m.advancedCursor = idx
	m.toggleMLAdvancedOption()

	if m.expandedOption != expandedMLFeatureFamilies {
		t.Fatalf("expected expandedMLFeatureFamilies, got %d", m.expandedOption)
	}
	if m.editingText {
		t.Fatalf("did not expect text-edit mode for feature families")
	}
}

func TestGetExpandedListItems_MLFeatureScopes(t *testing.T) {
	m := New(types.PipelineML, ".")
	m.expandedOption = expandedMLFeatureScopes

	items := m.getExpandedListItems()
	want := []string{"(none)", "global", "roi", "ch", "chpair"}
	for _, expected := range want {
		if !slices.Contains(items, expected) {
			t.Fatalf("expected %q in scopes options, got %#v", expected, items)
		}
	}
}

func TestHandleExpandedListToggle_MLFeatureSegmentsMultiSelectAndClear(t *testing.T) {
	m := New(types.PipelineML, ".")
	m.availableWindows = []string{"baseline", "active"}
	m.expandedOption = expandedMLFeatureSegments

	items := m.getExpandedListItems()
	baselineIdx := itemIndex(items, "baseline")
	activeIdx := itemIndex(items, "active")
	noneIdx := itemIndex(items, "(none)")
	if baselineIdx < 0 || activeIdx < 0 || noneIdx < 0 {
		t.Fatalf("expected baseline/active/(none) in segment options, got %#v", items)
	}

	m.subCursor = baselineIdx
	m.handleExpandedListToggle()
	if m.mlFeatureSegmentsSpec != "baseline" {
		t.Fatalf("expected baseline selected, got %q", m.mlFeatureSegmentsSpec)
	}
	if m.expandedOption != expandedMLFeatureSegments {
		t.Fatalf("expected expanded list to remain open for multi-select")
	}

	m.subCursor = activeIdx
	m.handleExpandedListToggle()
	if m.mlFeatureSegmentsSpec != "baseline active" {
		t.Fatalf("expected baseline active selected, got %q", m.mlFeatureSegmentsSpec)
	}

	m.subCursor = noneIdx
	m.handleExpandedListToggle()
	if m.mlFeatureSegmentsSpec != "" {
		t.Fatalf("expected segments to clear via (none), got %q", m.mlFeatureSegmentsSpec)
	}
}

func TestRenderMLAdvancedConfig_ShowsScrollIndicatorWithOffset(t *testing.T) {
	m := New(types.PipelineML, ".")
	m.height = 14
	m.contentWidth = 100
	m.expandedOption = expandedMLFeatureStats
	m.advancedOffset = 3

	out := m.renderMLAdvancedConfig()
	if !strings.Contains(out, "more items above") {
		t.Fatalf("expected scroll indicator in ML advanced render, got:\n%s", out)
	}
}
