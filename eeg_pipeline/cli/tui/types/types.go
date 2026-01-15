package types

// Pipeline represents available pipeline types
type Pipeline int

const (
	PipelinePreprocessing Pipeline = iota
	PipelineFeatures
	PipelineBehavior
	PipelineML
	PipelinePlotting
	PipelineFmri
	PipelineMergePsychoPyData
	PipelineRawToBIDS
	pipelineCount // Sentinel for bounds checking
)

// TimeRange represents a named time interval
type TimeRange struct {
	Name string
	Tmin string
	Tmax string
}

// pipelineNames maps pipeline types to their display names
var pipelineNames = [pipelineCount]string{
	PipelinePreprocessing:     "Preprocessing",
	PipelineFeatures:          "Features",
	PipelineBehavior:          "Behavior",
	PipelineML:                "Machine Learning",
	PipelinePlotting:          "Plotting",
	PipelineFmri:              "fMRI",
	PipelineMergePsychoPyData: "Merge PsychoPy Data",
	PipelineRawToBIDS:         "Raw to BIDS",
}

// pipelineCommands maps pipeline types to their CLI subcommand names
var pipelineCommands = [pipelineCount]string{
	PipelinePreprocessing:     "preprocessing",
	PipelineFeatures:          "features",
	PipelineBehavior:          "behavior",
	PipelineML:                "ml",
	PipelinePlotting:          "plotting",
	PipelineFmri:              "fmri",
	PipelineMergePsychoPyData: "utilities",
	PipelineRawToBIDS:         "utilities",
}

// pipelineDescriptions maps pipeline types to their descriptions
var pipelineDescriptions = [pipelineCount]string{
	PipelinePreprocessing:     "Bad channels, ICA, epochs",
	PipelineFeatures:          "Extract EEG features (power, connectivity...)",
	PipelineBehavior:          "EEG-behavior analysis",
	PipelineML:                "Machine learning: LOSO regression & time generalization",
	PipelinePlotting:          "Generate curated visualization suites",
	PipelineFmri:              "Preprocess fMRI (fMRIPrep-style)",
	PipelineMergePsychoPyData: "Merge PsychoPy data into BIDS events files",
	PipelineRawToBIDS:         "Convert raw BrainVision data to BIDS",
}

// String returns the display name for the pipeline
func (p Pipeline) String() string {
	if p.isValid() {
		return pipelineNames[p]
	}
	return "Unknown"
}

// CLICommand returns the canonical CLI subcommand name for this pipeline.
// This is the actual command accepted by the Python CLI (eeg-pipeline <cmd>).
func (p Pipeline) CLICommand() string {
	if p.isValid() {
		return pipelineCommands[p]
	}
	return "unknown"
}

// Description returns a human-readable description of what the pipeline does
func (p Pipeline) Description() string {
	if p.isValid() {
		return pipelineDescriptions[p]
	}
	return ""
}

// isValid checks if the pipeline value is within valid bounds
func (p Pipeline) isValid() bool {
	return p >= 0 && p < pipelineCount
}

// WizardStep represents steps in the pipeline wizard
type WizardStep int

const (
	StepSelectMode WizardStep = iota
	StepSelectComputations
	StepSelectFeatureFiles
	StepConfigureOptions
	StepSelectBands
	StepSelectROIs
	StepSelectSpatial
	StepTimeRange
	StepAdvancedConfig
	StepSelectPlots
	StepSelectFeaturePlotters
	StepSelectPlotCategories
	StepPlotConfig
	StepSelectSubjects
	StepSelectPreprocessingStages
	StepPreprocessingFiltering
	StepPreprocessingICA
	StepPreprocessingEpochs
	StepReviewExecute
	wizardStepCount // Sentinel for bounds checking
)

// wizardStepNames maps wizard steps to their display names
var wizardStepNames = [wizardStepCount]string{
	StepSelectMode:                "Select Mode",
	StepSelectComputations:        "Select Computations",
	StepSelectFeatureFiles:        "Select Feature Files",
	StepConfigureOptions:          "Configure Options",
	StepSelectBands:               "Select Bands",
	StepSelectROIs:                "Select ROIs",
	StepSelectSpatial:             "Select Spatial",
	StepTimeRange:                 "Time Range",
	StepAdvancedConfig:            "Advanced Config",
	StepSelectPlots:               "Select Plots",
	StepSelectFeaturePlotters:     "Select Feature Plotters",
	StepSelectPlotCategories:      "Select Plot Categories",
	StepPlotConfig:                "Plot Config",
	StepSelectSubjects:            "Subjects",
	StepSelectPreprocessingStages: "Stages",
	StepPreprocessingFiltering:    "Filtering",
	StepPreprocessingICA:          "ICA",
	StepPreprocessingEpochs:       "Epochs",
	StepReviewExecute:             "", // Deprecated - no longer used
}

// String returns the display name for the wizard step
func (s WizardStep) String() string {
	if s.isValid() {
		return wizardStepNames[s]
	}
	return "Unknown"
}

// isValid checks if the wizard step value is within valid bounds
func (s WizardStep) isValid() bool {
	return s >= 0 && s < wizardStepCount
}

// AvailabilityInfo holds availability status with last modified timestamp
type AvailabilityInfo struct {
	Available    bool
	LastModified string
}

// FeatureAvailability holds per-feature, per-band, and computation availability with timestamps
type FeatureAvailability struct {
	Features     map[string]AvailabilityInfo
	Bands        map[string]AvailabilityInfo
	Computations map[string]AvailabilityInfo
}

// SubjectStatus represents processing status for a subject
type SubjectStatus struct {
	ID                  string
	HasEpochs           bool
	HasPreprocessing    bool
	HasFeatures         bool
	HasStats            bool
	AvailableBands      []string
	FeatureAvailability *FeatureAvailability
	EpochMetadata       map[string]float64 `json:"epoch_metadata"`
}

// RequiresEpochs returns true if the pipeline needs epoched data
func (p Pipeline) RequiresEpochs() bool {
	switch p {
	case PipelineFeatures:
		return true
	case PipelinePreprocessing, PipelineRawToBIDS, PipelineBehavior, PipelineML, PipelinePlotting, PipelineMergePsychoPyData, PipelineFmri:
		return false
	default:
		return false
	}
}

// RequiresFeatures returns true if the pipeline needs extracted features
func (p Pipeline) RequiresFeatures() bool {
	switch p {
	case PipelineBehavior, PipelineML:
		return true
	default:
		return false
	}
}

// ValidateSubject checks if a subject meets the pipeline's data requirements.
// Returns true if valid, false with a reason string if invalid.
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
		return "bids"
	case PipelineRawToBIDS:
		return "source_data"
	case PipelineFeatures, PipelineBehavior:
		return "epochs"
	case PipelineML:
		return "features"
	case PipelinePlotting:
		return "all"
	case PipelineFmri:
		return "bids_fmri"
	default:
		return "epochs"
	}
}
