package wizard

import (
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestRenderAdvancedConfigDispatchesToPipelineRenderers(t *testing.T) {
	tests := []struct {
		name     string
		pipeline types.Pipeline
		setup    func(*Model)
		want     string
	}{
		{
			name:     "features",
			pipeline: types.PipelineFeatures,
			setup: func(m *Model) {
				for i := range m.categories {
					m.selected[i] = true
				}
			},
			want: "Connectivity",
		},
		{
			name:     "behavior",
			pipeline: types.PipelineBehavior,
			want:     "Execution",
		},
		{
			name:     "plotting",
			pipeline: types.PipelinePlotting,
			want:     "Plot-Specific Settings",
		},
		{
			name:     "machine-learning",
			pipeline: types.PipelineML,
			want:     "Data & Features",
		},
		{
			name:     "preprocessing",
			pipeline: types.PipelinePreprocessing,
			want:     "General",
		},
		{
			name:     "fmri-preprocessing",
			pipeline: types.PipelineFmri,
			want:     "Runtime",
		},
		{
			name:     "fmri-analysis",
			pipeline: types.PipelineFmriAnalysis,
			want:     "Input",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := New(tc.pipeline, ".")
			m.contentWidth = 120
			m.height = 200
			if tc.setup != nil {
				tc.setup(&m)
			}

			rendered := stripWizardHeaderANSI(m.renderAdvancedConfig())
			if !strings.Contains(rendered, tc.want) {
				t.Fatalf("expected %s renderer output to contain %q\nrendered:\n%s", tc.name, tc.want, rendered)
			}
		})
	}
}

func TestRenderAdvancedConfigUsesDefaultViewWhenEnabled(t *testing.T) {
	tests := []struct {
		name     string
		pipeline types.Pipeline
		want     string
	}{
		{
			name:     "features",
			pipeline: types.PipelineFeatures,
			want:     "Using defaults for feature parameters.",
		},
		{
			name:     "behavior",
			pipeline: types.PipelineBehavior,
			want:     "Using defaults for behavior analysis.",
		},
		{
			name:     "plotting",
			pipeline: types.PipelinePlotting,
			want:     "Using defaults. Space to customize.",
		},
		{
			name:     "machine-learning",
			pipeline: types.PipelineML,
			want:     "Using defaults for machine learning.",
		},
		{
			name:     "preprocessing",
			pipeline: types.PipelinePreprocessing,
			want:     "Using defaults for preprocessing.",
		},
		{
			name:     "fmri-preprocessing",
			pipeline: types.PipelineFmri,
			want:     "Using defaults for fMRI preprocessing.",
		},
		{
			name:     "fmri-analysis",
			pipeline: types.PipelineFmriAnalysis,
			want:     "Using defaults for fMRI analysis.",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := New(tc.pipeline, ".")
			m.contentWidth = 120
			m.height = 200
			m.useDefaultAdvanced = true

			rendered := stripWizardHeaderANSI(m.renderAdvancedConfig())
			if !strings.Contains(rendered, tc.want) {
				t.Fatalf("expected %s default view to contain %q\nrendered:\n%s", tc.name, tc.want, rendered)
			}
		})
	}
}

func TestRenderAdvancedConfigFallsBackForUnknownPipeline(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.Pipeline = types.Pipeline(999)
	m.contentWidth = 120
	m.height = 200

	rendered := stripWizardHeaderANSI(m.renderAdvancedConfig())
	if !strings.Contains(rendered, "No advanced options for this pipeline.") {
		t.Fatalf("expected unknown pipeline to use the default advanced view\nrendered:\n%s", rendered)
	}
}

func TestSpatialTransformPerFamilyLabel(t *testing.T) {
	tests := []struct {
		name string
		in   int
		want string
	}{
		{name: "inherit", in: 0, want: "inherit"},
		{name: "none", in: 1, want: "none"},
		{name: "csd", in: 2, want: "csd"},
		{name: "laplacian", in: 3, want: "laplacian"},
		{name: "wraps", in: 5, want: "none"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := spatialTransformPerFamilyLabel(tc.in)
			if got != tc.want {
				t.Fatalf("expected %d to map to %q, got %q", tc.in, tc.want, got)
			}
		})
	}
}
