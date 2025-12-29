package types

// Pipeline represents available pipeline types
type Pipeline int

const (
	PipelinePreprocessing Pipeline = iota
	PipelineFeatures
	PipelineBehavior
	PipelineDecoding
	PipelinePlotting
	PipelineCombineFeatures
	PipelineMergePsychoPyData
	PipelineRawToBIDS
)

// TimeRange represents a named time interval
type TimeRange struct {
	Name string
	Tmin string
	Tmax string
}

func (p Pipeline) String() string {
	names := []string{
		"Preprocessing",
		"Features",
		"Behavior",
		"Decoding",
		"Plotting",
		"Combine Features",
		"Merge PsychoPy Data",
		"Raw to BIDS",
	}
	if int(p) < len(names) {
		return names[p]
	}
	return "Unknown"
}

func (p Pipeline) Description() string {
	descriptions := []string{
		"Bad channels, ICA, epochs",
		"Extract EEG features (power, connectivity...)",
		"EEG-behavior analysis",
		"LOSO regression & time generalization",
		"Generate curated visualization suites",
		"Aggregate feature files into features_all.tsv",
		"Merge PsychoPy data into BIDS events files",
		"Convert raw BrainVision data to BIDS",
	}
	if int(p) < len(descriptions) {
		return descriptions[p]
	}
	return ""
}

// WizardStep represents steps in the pipeline wizard
type WizardStep int

const (
	StepSelectMode           WizardStep = iota
	StepProjectSetup                    // Project/task/path setup
	StepSelectComputations              // For behavior: which analyses to run
	StepSelectFeatureFiles              // For behavior & combine-features: which feature files to load
	StepConfigureOptions                // Category selection (features) or per-computation features
	StepSelectBands                     // Frequency band selection (features)
	StepSelectSpatial                   // Spatial aggregation mode (roi/channels/global)
	StepTimeRange                       // Time range input for feature extraction
	StepAdvancedConfig                  // Advanced pipeline configuration
	StepSelectPlots                     // Plotting: plot selection
	StepSelectPlotCategories            // Plotting: category selection (ERP, TFR, etc.)
	StepPlotConfig                      // Plotting: output config
	StepSelectSubjects
	StepReviewExecute
)

func (s WizardStep) String() string {
	names := []string{
		"Select Mode",
		"Project Setup",
		"Select Computations",
		"Select Feature Files",
		"Configure Options",
		"Select Bands",
		"Select Spatial",
		"Time Range",
		"Advanced Config",
		"Select Plots",
		"Select Plot Categories",
		"Plot Config",
		"Select Subjects",
		"Review & Execute",
	}
	if int(s) < len(names) {
		return names[s]
	}
	return "Unknown"
}

// AvailabilityInfo holds availability status with last modified timestamp
type AvailabilityInfo struct {
	Available    bool
	LastModified string // ISO timestamp
}

// FeatureAvailability holds per-feature, per-band, and computation availability with timestamps
type FeatureAvailability struct {
	Features     map[string]AvailabilityInfo
	Bands        map[string]AvailabilityInfo
	Computations map[string]AvailabilityInfo // Behavior computation outputs
}

// SubjectStatus represents processing status for a subject
type SubjectStatus struct {
	ID                  string
	HasEpochs           bool
	HasFeatures         bool
	HasStats            bool
	AvailableBands      []string
	FeatureAvailability *FeatureAvailability
	EpochMetadata       map[string]float64 `json:"epoch_metadata"`
}

// Pipeline validation methods

// RequiresEpochs returns true if the pipeline needs epoched data
func (p Pipeline) RequiresEpochs() bool {
	switch p {
	case PipelinePreprocessing, PipelineRawToBIDS:
		return false // Works with raw BIDS data or source data
	case PipelineFeatures:
		return true // Need epochs
	case PipelineBehavior, PipelineDecoding, PipelinePlotting, PipelineCombineFeatures, PipelineMergePsychoPyData:
		return false // Need features or other raw data
	default:
		return false
	}
}

// RequiresFeatures returns true if the pipeline needs extracted features
func (p Pipeline) RequiresFeatures() bool {
	switch p {
	case PipelineBehavior, PipelineDecoding, PipelineCombineFeatures:
		return true // Need pre-computed features
	default:
		return false
	}
}

// ValidateSubject checks if a subject meets the pipeline's data requirements
func (p Pipeline) ValidateSubject(s SubjectStatus) (valid bool, reason string) {
	if p.RequiresEpochs() && !s.HasEpochs {
		return false, "missing epochs"
	}
	if p.RequiresFeatures() && !s.HasFeatures {
		return false, "missing features"
	}
	return true, ""
}

// GetDataSource returns the appropriate data source for subject discovery
func (p Pipeline) GetDataSource() string {
	switch p {
	case PipelinePreprocessing, PipelineMergePsychoPyData:
		return "bids" // Raw BIDS data or source folder (handled by CLI)
	case PipelineRawToBIDS:
		return "source_data" // Raw source data
	case PipelineFeatures:
		return "epochs" // Epoched data
	case PipelineBehavior:
		return "epochs" // Subject discovery needs epochs; feature availability checked separately
	case PipelineDecoding, PipelineCombineFeatures:
		return "features" // Extracted features
	case PipelinePlotting:
		return "all" // Mixed plot types across derivatives
	default:
		return "epochs" // Default to epochs
	}
}
