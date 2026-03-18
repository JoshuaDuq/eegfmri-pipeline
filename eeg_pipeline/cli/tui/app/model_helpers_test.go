package app

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/wizard"
)

func TestFindRepoRootFromPath(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "eeg_pipeline"), 0o755); err != nil {
		t.Fatalf("mkdir repo marker: %v", err)
	}

	start := filepath.Join(root, "a", "b")
	if err := os.MkdirAll(start, 0o755); err != nil {
		t.Fatalf("mkdir start path: %v", err)
	}

	if got := findRepoRootFromPath(start); got != root {
		t.Fatalf("findRepoRootFromPath() = %q, want %q", got, root)
	}
}

func TestFindRepoRootFromPathFallsBackWhenMarkerMissing(t *testing.T) {
	start := t.TempDir()
	if got := findRepoRootFromPath(start); got != start {
		t.Fatalf("findRepoRootFromPath() = %q, want %q", got, start)
	}
}

func TestSanitizeCachedROIs(t *testing.T) {
	rois := []ROIState{
		{Key: "a", Name: "A", Channels: "Fp1, Fp1, C4, O1"},
		{Key: "b", Name: "B", Channels: "CP1, Cz"},
		{Key: "c", Name: "C", Channels: " , "},
	}

	sanitized, changed := sanitizeCachedROIs(rois)
	if !changed {
		t.Fatal("expected sanitizeCachedROIs to report a change")
	}
	if len(sanitized) != 1 {
		t.Fatalf("expected one ROI to survive sanitation, got %d", len(sanitized))
	}
	if sanitized[0].Channels != "O1" {
		t.Fatalf("unexpected sanitized channels: %q", sanitized[0].Channels)
	}
}

func TestModelHelpers(t *testing.T) {
	m := Model{}
	if !m.isValidPipelineIndex(0) || !m.isValidPipelineIndex(maxPipelineIndex) {
		t.Fatal("expected valid pipeline indices to be accepted")
	}
	if m.isValidPipelineIndex(-1) || m.isValidPipelineIndex(maxPipelineIndex+1) {
		t.Fatal("expected out-of-range pipeline indices to be rejected")
	}

	if gotPipeline, gotMode := m.extractCommandParts("eeg-pipeline features visualize"); gotPipeline != "features" || gotMode != "visualize" {
		t.Fatalf("extractCommandParts() = %q %q", gotPipeline, gotMode)
	}
	if gotPipeline, gotMode := m.extractCommandParts("invalid"); gotPipeline != "unknown" || gotMode != "" {
		t.Fatalf("unexpected fallback command parts: %q %q", gotPipeline, gotMode)
	}

	if !m.shouldUpdateTask("task") || m.shouldUpdateTask("") {
		t.Fatal("unexpected shouldUpdateTask behavior")
	}
	m.task = "task"
	if m.shouldUpdateTask("task") {
		t.Fatal("expected identical task to be rejected")
	}
}

func TestMessageConverters(t *testing.T) {
	lastModified := "2026-03-18T12:00:00Z"
	m := Model{}
	subjects := m.convertSubjects([]messages.SubjectInfo{
		{
			ID:            "sub-01",
			HasEpochs:     true,
			HasFeatures:   true,
			AvailableBands: []string{"alpha"},
			FeatureAvailability: &messages.FeatureAvailability{
				Features: map[string]messages.AvailabilityInfo{
					"power": {Available: true, LastModified: &lastModified},
				},
				Computations: map[string]messages.AvailabilityInfo{
					"correlations": {Available: true},
				},
			},
		},
	})
	if len(subjects) != 1 || subjects[0].ID != "sub-01" {
		t.Fatalf("unexpected converted subjects: %+v", subjects)
	}
	if subjects[0].FeatureAvailability == nil {
		t.Fatal("expected feature availability to be converted")
	}
	if got := subjects[0].FeatureAvailability.Features["power"].LastModified; got != lastModified {
		t.Fatalf("unexpected feature timestamp: %q", got)
	}

	feat := m.convertFeatureAvailability(&messages.FeatureAvailability{
		Features: map[string]messages.AvailabilityInfo{
			"power": {Available: true, LastModified: &lastModified},
		},
	})
	if feat == nil || !feat.Features["power"].Available {
		t.Fatalf("unexpected converted feature availability: %+v", feat)
	}
	if got := m.convertAvailabilityInfo(messages.AvailabilityInfo{Available: true, LastModified: &lastModified}); got.LastModified != lastModified {
		t.Fatalf("unexpected availability info: %+v", got)
	}

	plotters := m.convertPlotters(map[string][]messages.PlotterInfo{
		"features": []messages.PlotterInfo{{ID: "p1", Category: "features", Name: "Plot 1"}},
	})
	if len(plotters["features"]) != 1 || plotters["features"][0].ID != "p1" {
		t.Fatalf("unexpected converted plotters: %+v", plotters)
	}
}

func TestHandleSubjectsLoadedRoutesDiscovery(t *testing.T) {
	m := Model{
		selectedPipeline: types.PipelinePlotting,
		repoRoot:         t.TempDir(),
		task:             "task",
		wizard:           wizard.New(types.PipelinePlotting, "."),
	}

	_, cmd := m.handleSubjectsLoaded(messages.SubjectsLoadedMsg{
		Subjects: []messages.SubjectInfo{{ID: "sub-01"}},
	})
	if cmd == nil {
		t.Fatal("expected plotting subjects load to trigger follow-up discovery")
	}
}
