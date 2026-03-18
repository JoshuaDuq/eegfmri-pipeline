package types

import "testing"

func TestPipelineMetadata(t *testing.T) {
	tests := []struct {
		name          string
		pipeline      Pipeline
		wantName      string
		wantCommand   string
		wantDesc      string
		wantDataSource string
		wantEpochs    bool
		wantFeatures  bool
	}{
		{
			name:           "preprocessing",
			pipeline:       PipelinePreprocessing,
			wantName:       "Preprocessing",
			wantCommand:    "preprocessing",
			wantDesc:       "Bad channels, ICA, epochs",
			wantDataSource: "bids",
		},
		{
			name:           "features",
			pipeline:       PipelineFeatures,
			wantName:       "Features",
			wantCommand:    "features",
			wantDesc:       "Extract EEG features (power, connectivity...)",
			wantDataSource: "epochs",
			wantEpochs:     true,
		},
		{
			name:           "behavior",
			pipeline:       PipelineBehavior,
			wantName:       "Behavior",
			wantCommand:    "behavior",
			wantDesc:       "EEG-behavior analysis",
			wantDataSource: "epochs",
			wantFeatures:   true,
		},
		{
			name:           "ml",
			pipeline:       PipelineML,
			wantName:       "Machine Learning",
			wantCommand:    "ml",
			wantDesc:       "Machine learning: LOSO regression & time generalization",
			wantDataSource: "features",
			wantFeatures:   true,
		},
		{
			name:           "plotting",
			pipeline:       PipelinePlotting,
			wantName:       "Plotting",
			wantCommand:    "plotting",
			wantDesc:       "Generate curated visualization suites",
			wantDataSource: "all",
		},
		{
			name:           "fmri",
			pipeline:       PipelineFmri,
			wantName:       "fMRI",
			wantCommand:    "fmri",
			wantDesc:       "Preprocess fMRI (fMRIPrep-style)",
			wantDataSource: "bids_fmri",
		},
		{
			name:           "fmri-analysis",
			pipeline:       PipelineFmriAnalysis,
			wantName:       "fMRI Analysis",
			wantCommand:    "fmri-analysis",
			wantDesc:       "First-level contrasts + trial-wise signatures",
			wantDataSource: "bids_fmri",
		},
		{
			name:           "invalid",
			pipeline:       Pipeline(-1),
			wantName:       "Unknown",
			wantCommand:    "unknown",
			wantDataSource: "epochs",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.pipeline.String(); got != tc.wantName {
				t.Fatalf("String() = %q, want %q", got, tc.wantName)
			}
			if got := tc.pipeline.CLICommand(); got != tc.wantCommand {
				t.Fatalf("CLICommand() = %q, want %q", got, tc.wantCommand)
			}
			if got := tc.pipeline.Description(); got != tc.wantDesc {
				t.Fatalf("Description() = %q, want %q", got, tc.wantDesc)
			}
			if got := tc.pipeline.GetDataSource(); got != tc.wantDataSource {
				t.Fatalf("GetDataSource() = %q, want %q", got, tc.wantDataSource)
			}
			if got := tc.pipeline.RequiresEpochs(); got != tc.wantEpochs {
				t.Fatalf("RequiresEpochs() = %v, want %v", got, tc.wantEpochs)
			}
			if got := tc.pipeline.RequiresFeatures(); got != tc.wantFeatures {
				t.Fatalf("RequiresFeatures() = %v, want %v", got, tc.wantFeatures)
			}
		})
	}
}

func TestValidateSubject(t *testing.T) {
	subject := SubjectStatus{
		ID:         "sub-01",
		HasEpochs:  true,
		HasFeatures: true,
	}

	if valid, reason := PipelineFeatures.ValidateSubject(subject); !valid || reason != "" {
		t.Fatalf("expected feature subject to be valid, got %v %q", valid, reason)
	}

	subject.HasEpochs = false
	if valid, reason := PipelineFeatures.ValidateSubject(subject); valid || reason != "missing epochs" {
		t.Fatalf("expected missing epochs failure, got %v %q", valid, reason)
	}

	subject.HasEpochs = true
	subject.HasFeatures = false
	if valid, reason := PipelineBehavior.ValidateSubject(subject); valid || reason != "missing features" {
		t.Fatalf("expected missing features failure, got %v %q", valid, reason)
	}
}

func TestWizardStepMetadata(t *testing.T) {
	if got := StepAdvancedConfig.String(); got != "Advanced Config" {
		t.Fatalf("StepAdvancedConfig.String() = %q", got)
	}

	if got := WizardStep(-1).String(); got != "Unknown" {
		t.Fatalf("invalid WizardStep.String() = %q", got)
	}
}
