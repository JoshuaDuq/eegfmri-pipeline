package wizard

import (
	"testing"
	"time"

	"github.com/eeg-pipeline/tui/types"
)

func TestFormatRelativeTimeAndTimestampParsing(t *testing.T) {
	nano := time.Now().UTC().Add(-90 * time.Minute).Format(time.RFC3339Nano)
	parsedNano := parseISOTimestamp(nano)
	if parsedNano.IsZero() {
		t.Fatal("expected RFC3339Nano timestamp to parse")
	}
	if got := formatRelativeTime(nano); got != "1 hour ago" {
		t.Fatalf("expected 1 hour ago, got %q", got)
	}

	sec := time.Now().UTC().Add(-30 * time.Second).Format(time.RFC3339)
	parsedSec := parseISOTimestamp(sec)
	if parsedSec.IsZero() {
		t.Fatal("expected RFC3339 timestamp to parse")
	}
	if got := formatRelativeTime(sec); got != "just now" {
		t.Fatalf("expected just now, got %q", got)
	}

	if !parseISOTimestamp("not-a-timestamp").IsZero() {
		t.Fatal("expected invalid timestamp to return zero time")
	}
	if got := formatRelativeTime(""); got != "" {
		t.Fatalf("expected empty relative time for empty input, got %q", got)
	}
	if got := formatTimeAgo(30 * time.Second); got != "just now" {
		t.Fatalf("expected just now, got %q", got)
	}
	if got := formatMinutes(1 * time.Minute); got != "1 min ago" {
		t.Fatalf("expected singular minute label, got %q", got)
	}
	if got := formatMinutes(2 * time.Minute); got != "2 mins ago" {
		t.Fatalf("expected plural minute label, got %q", got)
	}
	if got := formatHours(1 * time.Hour); got != "1 hour ago" {
		t.Fatalf("expected singular hour label, got %q", got)
	}
	if got := formatHours(2 * time.Hour); got != "2 hours ago" {
		t.Fatalf("expected plural hour label, got %q", got)
	}
	if got := formatDays(24 * time.Hour); got != "1 day ago" {
		t.Fatalf("expected singular day label, got %q", got)
	}
	if got := formatDays(48 * time.Hour); got != "2 days ago" {
		t.Fatalf("expected plural day label, got %q", got)
	}
}

func TestFeatureCategoryLabelCoversUnknownAndEmpty(t *testing.T) {
	if got := featureCategoryLabel(""); got != "" {
		t.Fatalf("expected empty category to stay empty, got %q", got)
	}
	if got := featureCategoryLabel("alpha"); got != "Alpha" {
		t.Fatalf("expected generic category label to capitalize, got %q", got)
	}
}

func TestValidationHelpers(t *testing.T) {
	t.Run("basic selection validators", func(t *testing.T) {
		cases := []struct {
			name string
			got  []string
			want string
		}{
			{
				name: "computations",
				got: func() []string {
					m := New(types.PipelineBehavior, ".")
					m.computationSelected = make(map[int]bool)
					return m.validateComputationSelectionStep()
				}(),
				want: "Select at least one analysis to run",
			},
			{
				name: "feature-files",
				got: func() []string {
					m := New(types.PipelineFeatures, ".")
					m.featureFileSelected = make(map[string]bool)
					return m.validateFeatureFileSelectionStep()
				}(),
				want: "Select at least one feature file to load",
			},
			{
				name: "categories",
				got: func() []string {
					m := New(types.PipelineFeatures, ".")
					m.selected = make(map[int]bool)
					return m.validateCategorySelectionStep()
				}(),
				want: "Select at least one category",
			},
			{
				name: "bands",
				got: func() []string {
					m := New(types.PipelineFeatures, ".")
					m.bandSelected = make(map[int]bool)
					return m.validateBandSelectionStep()
				}(),
				want: "Select at least one frequency band",
			},
			{
				name: "spatial",
				got: func() []string {
					m := New(types.PipelineFeatures, ".")
					m.spatialSelected = make(map[int]bool)
					return m.validateSpatialSelectionStep()
				}(),
				want: "Select at least one spatial mode",
			},
			{
				name: "preprocessing-stages",
				got: func() []string {
					m := New(types.PipelinePreprocessing, ".")
					m.prepStageSelected = make(map[int]bool)
					return m.validatePreprocessingStageSelectionStep()
				}(),
				want: "Select at least one preprocessing stage",
			},
			{
				name: "plot-config",
				got: func() []string {
					m := New(types.PipelinePlotting, ".")
					m.plotFormatSelected = make(map[string]bool)
					return m.validatePlotConfigStep()
				}(),
				want: "Select at least one output format (PNG, SVG, or PDF)",
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if len(tc.got) != 1 || tc.got[0] != tc.want {
					t.Fatalf("unexpected validation errors: %#v", tc.got)
				}
			})
		}
	})

	t.Run("plot validators", func(t *testing.T) {
		m := New(types.PipelinePlotting, ".")
		m.plotItems = []PlotItem{{ID: "plot-1", Group: "power"}}
		m.plotSelected = map[int]bool{0: false}
		if got := m.validatePlotSelectionStep(); len(got) != 1 || got[0] != "Select at least one plot to generate" {
			t.Fatalf("unexpected plot validation errors: %#v", got)
		}

		m = New(types.PipelineFeatures, ".")
		m.plotItems = []PlotItem{{ID: "features_power", Group: "features"}}
		m.plotSelected = map[int]bool{0: true}
		m.featurePlotterError = ""
		if got := m.validateFeaturePlotterSelectionStep(); len(got) != 1 || got[0] != "Feature plot list is still loading" {
			t.Fatalf("unexpected loading validation errors: %#v", got)
		}

		m.featurePlotters = map[string][]PlotterInfo{
			"power": {
				{ID: "power.topo", Category: "power", Name: "Power Topography"},
			},
		}
		m.featurePlotterSelected = map[string]bool{}
		if got := m.validateFeaturePlotterSelectionStep(); len(got) != 1 || got[0] != "Select at least one feature plot" {
			t.Fatalf("unexpected feature plotter validation errors: %#v", got)
		}
	})

	t.Run("time ranges", func(t *testing.T) {
		m := New(types.PipelineFeatures, ".")
		m.prepTaskIsRest = false
		m.TimeRanges = nil
		if got := m.validateTimeRangeStep(); len(got) != 1 || got[0] != "No time ranges defined" {
			t.Fatalf("unexpected time range validation errors: %#v", got)
		}
	})
}
