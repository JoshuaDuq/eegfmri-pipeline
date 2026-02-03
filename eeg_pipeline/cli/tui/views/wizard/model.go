package wizard

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	tea "github.com/charmbracelet/bubbletea"
)

///////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////

const (
	// Time range editing states
	noRangeEditing     = -1
	fieldName          = 0
	fieldTmin          = 1
	fieldTmax          = 2
	numTimeRangeFields = 3

	// Scroll offset calculation
	minVisibleLines       = 8
	defaultTerminalHeight = 40

	// Minimum buffer length for single character input
	singleCharLength = 1
)

///////////////////////////////////////////////////////////////////
// Data Definitions
///////////////////////////////////////////////////////////////////

type Computation struct {
	Key         string
	Name        string
	Description string
	Group       string // Core, Advanced, DataPrep, Quality
}

type FeatureCategory struct {
	Key         string
	Name        string
	Description string
}

var behaviorComputations = []Computation{
	// Data Preparation
	{"trial_table", "Trial Table", "Build canonical per-trial analysis table", "DataPrep"},
	{"lag_features", "Lag Features", "Add temporal dynamics (prev_*, delta_*) for habituation", "DataPrep"},
	{"pain_residual", "Pain Residual + Diagnostics", "Compute pain_residual and temperature model diagnostics", "DataPrep"},

	// Core Analyses
	{"correlations", "Correlations", "EEG-rating correlations with bootstrap CIs", "Core"},
	{"multilevel_correlations", "Group Multilevel Correlations", "Group-level correlations with block-restricted permutations", "Core"},
	{"regression", "Regression", "Feature regression with optional permutation + model sensitivity", "Core"},
	{"condition", "Condition Comparison", "Compare conditions (e.g., pain vs non-pain)", "Core"},
	{"temporal", "Temporal Correlations", "Time-resolved correlation analysis", "Core"},
	{"pain_sensitivity", "Pain Sensitivity", "Individual pain sensitivity (temperature→rating slope)", "Core"},
	{"cluster", "Cluster Permutation", "Cluster-based permutation tests", "Core"},

	// Advanced/Causal Analyses
	{"mediation", "Mediation Analysis", "Path analysis: does EEG mediate temperature→rating?", "Advanced"},
	{"moderation", "Moderation Analysis", "Does EEG moderate the temperature→rating effect?", "Advanced"},
	{"mixed_effects", "Mixed Effects", "Mixed-effects modeling (group-level)", "Advanced"},

	// Quality & Validation
	{"stability", "Stability (Run/Block)", "Within-subject stability diagnostics", "Quality"},
	{"validation", "Validation", "Effect consistency + influence diagnostics", "Quality"},
	{"report", "Subject Report", "Single-subject summary report", "Quality"},
}

type FrequencyBand struct {
	Key         string
	Name        string
	Description string
	LowHz       float64
	HighHz      float64
}

type behaviorSection struct {
	Key     string
	Label   string
	Enabled bool
}

var frequencyBands = []FrequencyBand{
	{"delta", "Delta", "Delta band", 1.0, 3.9},
	{"theta", "Theta", "Theta band", 4.0, 7.9},
	{"alpha", "Alpha", "Alpha band", 8.0, 12.9},
	{"beta", "Beta", "Beta band", 13.0, 30.0},
	{"gamma", "Gamma", "Gamma band", 30.1, 80.0},
}

type ROIDefinition struct {
	Key      string
	Name     string
	Channels string
}

var defaultROIs = []ROIDefinition{
	{"Frontal", "Frontal", "Fp1,Fp2,Fpz,AF3,AF4,AF7,AF8,F1,F2,F3,F4,F5,F6,F7,F8"},
	{"Sensorimotor_Right", "Sensorimotor Right", "FC2,FC4,FC6,C2,C4,C6,CP2,CP4,CP6"},
	{"Sensorimotor_Left", "Sensorimotor Left", "FC1,FC3,FC5,C1,C3,C5,CP1,CP3,CP5"},
	{"Temporal_Right", "Temporal Right", "FT8,FT10,T8,TP8,TP10"},
	{"Temporal_Left", "Temporal Left", "FT7,FT9,T7,TP7,TP9"},
	{"ParOccipital_Right", "ParOccipital Right", "P2,P4,P6,P8,PO4,PO8,O2"},
	{"ParOccipital_Left", "ParOccipital Left", "P1,P3,P5,P7,PO3,PO7,O1"},
	{"ParOccipital_Midline", "ParOccipital Midline", "Pz,POz,Oz"},
	{"Midline_FrontalCentral", "Midline Frontal-Central", "Fz,Cz,CPz"},
}

// PreprocessingStage represents a preprocessing stage
type PreprocessingStage struct {
	Key         string
	Name        string
	Description string
}

var preprocessingStages = []PreprocessingStage{
	{"bad_channels", "Bad Channels", "Detect and mark bad channels"},
	{"filtering", "Filtering", "Apply frequency filters"},
	{"ica", "ICA", "Independent component analysis"},
	{"epochs", "Epochs", "Create epoched data"},
}

// FilteringOptions represents filtering configuration
type FilteringOptions struct {
	ResampleFreq *float64 // Hz, nil means use config default
	HighPassFreq *float64 // Hz (l_freq)
	LowPassFreq  *float64 // Hz (h_freq)
	NotchFreq    *int     // Hz
	LineFreq     *int     // Hz (50 or 60)
}

// ICAOptions represents ICA configuration
type ICAOptions struct {
	Method               *string  // fastica, infomax, picard
	Components           *float64 // int or variance fraction
	ProbabilityThreshold *float64 // 0-1
	LabelsToKeep         *string  // comma-separated labels
	UseICALabel          *bool    // true/false
	KeepMNEBIDS          *bool    // keep MNE-BIDS flagged bads
}

// EpochOptions represents epoch configuration
type EpochOptions struct {
	Tmin     *float64    // epoch start time (s)
	Tmax     *float64    // epoch end time (s)
	Baseline *[2]float64 // [start, end] in seconds, nil means no baseline
	Reject   *float64    // peak-to-peak amplitude threshold (µV)
}

// PreprocessingAdvancedOptions represents advanced preprocessing configuration
type PreprocessingAdvancedOptions struct {
	UsePyPREP            *bool // use PyPREP for bad channel detection
	RANSAC               *bool // use RANSAC
	Repeats              *int  // number of bad channel detection iterations
	AverageReref         *bool // average re-reference before detection
	ConsiderPreviousBads *bool // keep previously marked bad channels
	OverwriteChannelsTSV *bool // overwrite channels.tsv file
	DeleteBreaks         *bool // delete breaks in data
	BreaksMinLength      *int  // minimum break duration (s)
	TStartAfterPrevious  *int  // time after previous event (s)
	TStopBeforeNext      *int  // time before next event (s)
}

type SpatialMode struct {
	Key         string
	Name        string
	Description string
}

var spatialModes = []SpatialMode{
	{"roi", "ROI", "Aggregate by region of interest"},
	{"channels", "All Channels", "Compute per-channel features"},
	{"global", "Global", "Mean across all channels"},
}

// Connectivity measures for features pipeline
type ConnectivityMeasure struct {
	Key         string
	Name        string
	Description string
}

var connectivityMeasures = []ConnectivityMeasure{
	{"wpli", "wPLI", "Weighted phase lag index"},
	{"aec", "AEC", "Amplitude envelope correlation"},
	{"plv", "PLV", "Phase locking value"},
	{"pli", "PLI", "Phase lag index"},
}

// Directed connectivity measures for features pipeline
type DirectedConnectivityMeasure struct {
	Key         string
	Name        string
	Description string
}

var directedConnectivityMeasures = []DirectedConnectivityMeasure{
	{"psi", "PSI", "Phase Slope Index (direction of information flow)"},
	{"dtf", "DTF", "Directed Transfer Function (MVAR-based causality)"},
	{"pdc", "PDC", "Partial Directed Coherence (direct causal influence)"},
}

type MLCVScope int

const (
	MLCVScopeGroup MLCVScope = iota
	MLCVScopeSubject
)

func (s MLCVScope) CLIValue() string {
	switch s {
	case MLCVScopeSubject:
		return "subject"
	default:
		return "group"
	}
}

type PlottingScope int

const (
	PlottingScopeSubject PlottingScope = iota
	PlottingScopeGroup
)

func (s PlottingScope) CLIValue() string {
	switch s {
	case PlottingScopeGroup:
		return "group"
	default:
		return "subject"
	}
}

type MLFeatureHarmonization int

const (
	MLFeatureHarmonizationDefault MLFeatureHarmonization = iota
	MLFeatureHarmonizationIntersection
	MLFeatureHarmonizationUnionImpute
)

func (h MLFeatureHarmonization) CLIValue() string {
	switch h {
	case MLFeatureHarmonizationIntersection:
		return "intersection"
	case MLFeatureHarmonizationUnionImpute:
		return "union_impute"
	default:
		return ""
	}
}

func (h MLFeatureHarmonization) Display() string {
	v := h.CLIValue()
	if v == "" {
		return "(default)"
	}
	return v
}

func (h MLFeatureHarmonization) Next() MLFeatureHarmonization {
	return MLFeatureHarmonization((int(h) + 1) % 3)
}

type MLRegressionModel int

const (
	MLRegressionElasticNet MLRegressionModel = iota
	MLRegressionRidge
	MLRegressionRF
)

func (r MLRegressionModel) CLIValue() string {
	switch r {
	case MLRegressionRidge:
		return "ridge"
	case MLRegressionRF:
		return "rf"
	default:
		return "elasticnet"
	}
}

func (r MLRegressionModel) Display() string {
	return r.CLIValue()
}

func (r MLRegressionModel) Next() MLRegressionModel {
	return MLRegressionModel((int(r) + 1) % 3)
}

type MLClassificationModel int

const (
	MLClassificationDefault MLClassificationModel = iota
	MLClassificationSVM
	MLClassificationLR
	MLClassificationRF
)

func (c MLClassificationModel) CLIValue() string {
	switch c {
	case MLClassificationSVM:
		return "svm"
	case MLClassificationLR:
		return "lr"
	case MLClassificationRF:
		return "rf"
	default:
		return ""
	}
}

func (c MLClassificationModel) Display() string {
	v := c.CLIValue()
	if v == "" {
		return "(default)"
	}
	return v
}

func (c MLClassificationModel) Next() MLClassificationModel {
	return MLClassificationModel((int(c) + 1) % 4)
}

// Feature file selection for behavior pipeline
type FeatureFile struct {
	Key         string
	Name        string
	Description string
}

var featureFileOptions = []FeatureFile{
	{"power", "Power Features", "EEG power spectral features"},
	{"connectivity", "Connectivity", "Functional connectivity features"},
	{"directedconnectivity", "Directed Connectivity", "Effective connectivity (PSI, DTF, PDC)"},
	{"sourcelocalization", "Source Localization", "LCMV/eLORETA source estimates"},
	{"aperiodic", "Aperiodic (1/f)", "Aperiodic spectral features"},
	{"itpc", "ITPC", "Inter-trial phase coherence"},
	{"pac", "PAC", "Phase-amplitude coupling"},
	{"complexity", "Complexity", "Complexity/entropy features"},
	{"ratios", "Ratios", "Band power ratios"},
	{"asymmetry", "Asymmetry", "Hemispheric asymmetry"},
	{"quality", "Quality", "Trial quality metrics"},
	{"erds", "ERDS", "Event-related desynchronization/sync"},
	{"spectral", "Spectral", "Peak frequency, spectral edge"},
}

// Behavior scatter feature types for plotting
var behaviorScatterFeatureTypes = []string{
	"power",
	"connectivity",
	"aperiodic",
	"complexity",
	"itpc",
	"pac",
	"erds",
	"spectral",
	"ratios",
	"asymmetry",
}

// Behavior scatter aggregation modes
var behaviorScatterAggregationModes = []string{
	"roi",
	"global",
	"channel",
}

// computationApplicableFeatures maps each computation to the feature files it can use.
// Features not in this list for a given computation won't be shown in the feature selection.
var computationApplicableFeatures = map[string][]string{
	// Correlations can use all standard EEG features
	"correlations": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	// Multilevel correlations uses the same features as correlations (group-level)
	"multilevel_correlations": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	// Pain sensitivity uses the same features as correlations
	"pain_sensitivity": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	// Condition comparison uses trial-level features
	"condition": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	// Temporal: power, itpc, erds are the temporal-specific features computed from epochs
	// User selects which to compute in step 3 (feature selection)
	"temporal": {"power", "itpc", "erds"},
	// Cluster permutation uses TFR data directly
	"cluster": {"power"},
	// Mediation/moderation use correlations features
	"mediation":     {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"moderation":    {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"mixed_effects": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	// Quality & Validation analyses
	"regression":  {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"stability":   {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"validation":  {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"consistency": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"influence":   {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
	"report":      {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
}

type PlotItem struct {
	ID               string
	Group            string
	Name             string
	Description      string
	RequiredFiles    []string
	RequiresEpochs   bool
	RequiresFeatures bool
	RequiresStats    bool
	Dependencies     []string // Other plots this plot depends on
}

type PlotterInfo struct {
	ID       string
	Category string
	Name     string
}

// PlotItemConfig stores advanced settings scoped to a specific plot ID.
// These are passed to the CLI as per-plot overrides.
type PlotItemConfig struct {
	// TFR
	TfrDefaultBaselineWindowSpec string

	// Comparisons
	CompareWindows        *bool
	ComparisonWindowsSpec string
	CompareColumns        *bool
	ComparisonSegment     string
	ComparisonColumn      string
	ComparisonValuesSpec  string
	ComparisonLabelsSpec  string
	ComparisonROIsSpec    string

	// Topomaps
	TopomapWindowsSpec string

	// TFR Topomap
	TfrTopomapActiveWindow         string
	TfrTopomapWindowSizeMs         string
	TfrTopomapWindowCount          string
	TfrTopomapLabelXPosition       string
	TfrTopomapLabelYPositionBottom string
	TfrTopomapLabelYPosition       string
	TfrTopomapTitleY               string
	TfrTopomapTitlePad             string
	TfrTopomapSubplotsRight        string
	TfrTopomapTemporalHspace       string
	TfrTopomapTemporalWspace       string

	// Connectivity
	ConnectivityCircleTopFraction  string
	ConnectivityCircleMinLines     string
	ConnectivityNetworkTopFraction string

	// ITPC
	ItpcSharedColorbar *bool

	// Behavior Scatter
	BehaviorScatterFeaturesSpec         string
	BehaviorScatterColumnsSpec          string
	BehaviorScatterAggregationModesSpec string
	BehaviorScatterSegmentSpec          string

	// Behavior temporal topomaps
	BehaviorTemporalStatsFeatureFolder string

	// Behavior dose response
	DoseResponseDoseColumn     string
	DoseResponseResponseColumn string
	DoseResponsePainColumn     string
	DoseResponseSegment        string
}

type plotItemConfigField int

const (
	plotItemConfigFieldNone plotItemConfigField = iota
	plotItemConfigFieldTfrDefaultBaselineWindow
	plotItemConfigFieldCompareWindows
	plotItemConfigFieldComparisonWindows
	plotItemConfigFieldCompareColumns
	plotItemConfigFieldComparisonSegment
	plotItemConfigFieldComparisonColumn
	plotItemConfigFieldComparisonValues
	plotItemConfigFieldComparisonLabels
	plotItemConfigFieldComparisonROIs
	plotItemConfigFieldTopomapWindow
	plotItemConfigFieldTfrTopomapActiveWindow
	plotItemConfigFieldTfrTopomapWindowSizeMs
	plotItemConfigFieldTfrTopomapWindowCount
	plotItemConfigFieldTfrTopomapLabelXPosition
	plotItemConfigFieldTfrTopomapLabelYPositionBottom
	plotItemConfigFieldTfrTopomapLabelYPosition
	plotItemConfigFieldTfrTopomapTitleY
	plotItemConfigFieldTfrTopomapTitlePad
	plotItemConfigFieldTfrTopomapSubplotsRight
	plotItemConfigFieldTfrTopomapTemporalHspace
	plotItemConfigFieldTfrTopomapTemporalWspace
	plotItemConfigFieldConnectivityCircleTopFraction
	plotItemConfigFieldConnectivityCircleMinLines
	plotItemConfigFieldConnectivityNetworkTopFraction
	plotItemConfigFieldItpcSharedColorbar
	// Behavior Scatter
	plotItemConfigFieldBehaviorScatterFeatures
	plotItemConfigFieldBehaviorScatterColumns
	plotItemConfigFieldBehaviorScatterAggregationModes
	plotItemConfigFieldBehaviorScatterSegment
	// Behavior temporal topomaps
	plotItemConfigFieldBehaviorTemporalStatsFeatureFolder
	// Behavior dose response
	plotItemConfigFieldDoseResponseDoseColumn
	plotItemConfigFieldDoseResponseResponseColumn
	plotItemConfigFieldDoseResponsePainColumn
	plotItemConfigFieldDoseResponseSegment
)

type textField int

const (
	textFieldNone textField = iota
	textFieldTask
	textFieldBidsRoot
	textFieldBidsFmriRoot
	textFieldDerivRoot
	textFieldSourceRoot
	// fMRI preprocessing text fields
	textFieldFmriFmriprepImage
	textFieldFmriFmriprepOutputDir
	textFieldFmriFmriprepWorkDir
	textFieldFmriFreesurferLicenseFile
	textFieldFmriFreesurferSubjectsDir
	textFieldFmriOutputSpaces
	textFieldFmriIgnore
	textFieldFmriBidsFilterFile
	textFieldFmriExtraArgs
	textFieldFmriSkullStripTemplate
	textFieldFmriTaskId
	// fMRI analysis text fields
	textFieldFmriAnalysisFmriprepSpace
	textFieldFmriAnalysisRuns
	textFieldFmriAnalysisCondAColumn
	textFieldFmriAnalysisCondAValue
	textFieldFmriAnalysisCondBColumn
	textFieldFmriAnalysisCondBValue
	textFieldFmriAnalysisContrastName
	textFieldFmriAnalysisFormula
	textFieldFmriAnalysisEventsToModel
	textFieldFmriAnalysisStimPhasesToModel
		textFieldFmriAnalysisOutputDir
		textFieldFmriAnalysisFreesurferDir
		textFieldFmriAnalysisSignatureDir
		textFieldFmriTrialSigGroupColumn
		textFieldFmriTrialSigGroupValues
		textFieldFmriTrialSigScopeStimPhases
		textFieldRawMontage
	textFieldPrepMontage
	textFieldPrepChTypes
	textFieldPrepEegReference
	textFieldPrepEogChannels
	textFieldPrepConditions
	textFieldPrepFileExtension
	textFieldPrepRenameAnotDict
	textFieldPrepCustomBadDict
	textFieldRawEventPrefixes
	textFieldMergeEventPrefixes
	textFieldMergeEventTypes
	// fMRI raw-to-bids text fields
	textFieldFmriRawSession
	textFieldFmriRawRestTask
	textFieldFmriRawDcm2niixPath
	textFieldFmriRawDcm2niixArgs
	// Behavior advanced config text fields
	textFieldConditionCompareColumn
	textFieldConditionCompareWindows
	textFieldConditionCompareValues
	textFieldTemporalConditionColumn
	textFieldTemporalConditionValues
	textFieldTemporalTargetColumn
	textFieldTfHeatmapFreqs
	textFieldRunAdjustmentColumn
	textFieldPainResidualCrossfitGroupColumn
	textFieldClusterConditionColumn
	textFieldClusterConditionValues
	textFieldCorrelationsTargetColumn
	textFieldCorrelationsTypes
	// Frequency band editing text fields
	textFieldBandName
	textFieldBandLowHz
	textFieldBandHighHz
	// Features advanced config text fields
	textFieldPACPairs
	textFieldBurstBands
	textFieldSpectralRatioPairs
	textFieldAsymmetryChannelPairs
	textFieldAsymmetryActivationBands
	textFieldIAFRois
	textFieldERPComponents
	textFieldERDSBands
	// ITPC condition-based text fields
	textFieldItpcConditionColumn
	textFieldItpcConditionValues
	// Connectivity condition-based text fields
	textFieldConnConditionColumn
	textFieldConnConditionValues
	// Source localization advanced config text fields
	textFieldSourceLocSubject
	textFieldSourceLocTrans
	textFieldSourceLocBem
	textFieldSourceLocFmriStatsMap
	// fMRI contrast builder text fields
	textFieldSourceLocFmriCondAColumn
	textFieldSourceLocFmriCondAValue
	textFieldSourceLocFmriCondBColumn
	textFieldSourceLocFmriCondBValue
	textFieldSourceLocFmriContrastFormula
	textFieldSourceLocFmriContrastName
	textFieldSourceLocFmriRunsToInclude
	textFieldSourceLocFmriStimPhasesToModel
	textFieldSourceLocFmriWindowAName
	textFieldSourceLocFmriWindowBName
	textFieldPainResidualSplineDfCandidates
	textFieldPainResidualModelComparePolyDegrees
	// Plotting advanced config text fields
	textFieldPlotBboxInches
	textFieldPlotFontFamily
	textFieldPlotFontWeight
	textFieldPlotLayoutTightRect
	textFieldPlotLayoutTightRectMicrostate
	textFieldPlotGridSpecWidthRatios
	textFieldPlotGridSpecHeightRatios
	textFieldPlotFigureSizeStandard
	textFieldPlotFigureSizeMedium
	textFieldPlotFigureSizeSmall
	textFieldPlotFigureSizeSquare
	textFieldPlotFigureSizeWide
	textFieldPlotFigureSizeTFR
	textFieldPlotFigureSizeTopomap
	textFieldPlotColorPain
	textFieldPlotColorNonpain
	textFieldPlotColorSignificant
	textFieldPlotColorNonsignificant
	textFieldPlotColorGray
	textFieldPlotColorLightGray
	textFieldPlotColorBlack
	textFieldPlotColorBlue
	textFieldPlotColorRed
	textFieldPlotColorNetworkNode
	textFieldPlotScatterEdgecolor
	textFieldPlotHistEdgecolor
	textFieldPlotKdeColor
	textFieldPlotTopomapColormap
	textFieldPlotTopomapSigMaskMarker
	textFieldPlotTopomapSigMaskMarkerFaceColor
	textFieldPlotTopomapSigMaskMarkerEdgeColor
	textFieldPlotTfrDefaultBaselineWindow
	textFieldPlotComparisonWindows
	textFieldPlotComparisonSegment
	textFieldPlotComparisonColumn
	textFieldPlotComparisonValues
	textFieldPlotComparisonLabels
	textFieldPlotComparisonROIs
	textFieldPlotPacCmap
	textFieldPlotPacPairs
	textFieldPlotConnectivityMeasures
	textFieldPlotSpectralMetrics
	textFieldPlotBurstsMetrics
	textFieldPlotTemporalTimeBins
	textFieldPlotTemporalTimeLabels
	textFieldPlotAsymmetryStat
	// Machine Learning advanced config text fields
	textFieldMLTarget
	textFieldMLFmriSigContrastName
	textFieldMLFeatureFamilies
	textFieldMLFeatureBands
	textFieldMLFeatureSegments
	textFieldMLFeatureScopes
	textFieldMLFeatureStats
	textFieldMLCovariates
	textFieldMLBaselinePredictors
	textFieldElasticNetAlphaGrid
	textFieldElasticNetL1RatioGrid
	textFieldRidgeAlphaGrid
	textFieldRfMaxDepthGrid
	textFieldVarianceThresholdGrid
	// Preprocessing advanced config text fields
	textFieldIcaLabelsToKeep
)

var defaultPlotItems = []PlotItem{
	// Power
	{ID: "power_by_condition", Group: "power", Name: "Condition Comparison", Description: "Power differences between conditions", RequiredFiles: []string{"features_power*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "band_power_topomaps", Group: "power", Name: "Topomaps", Description: "Band power topographic maps for selected time window", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif", "events.tsv"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "cross_frequency_power_correlation", Group: "power", Name: "Cross-Frequency Correlation", Description: "Correlation matrix between frequency bands", RequiredFiles: []string{"features_power*.tsv"}, RequiresFeatures: true},
	{ID: "power_spectral_density", Group: "power", Name: "PSD Summary", Description: "Power spectral density curves", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Connectivity
	{ID: "connectivity_by_condition", Group: "connectivity", Name: "Condition Comparison", Description: "Connectivity differences between conditions", RequiredFiles: []string{"features_connectivity*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "connectivity_circle_condition", Group: "connectivity", Name: "Circle by Condition", Description: "Connectivity circles per measure and band by condition", RequiredFiles: []string{"features_connectivity*.tsv", "epochs/*.fif", "events.tsv"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "connectivity_heatmap", Group: "connectivity", Name: "Heatmaps", Description: "Connectivity heatmaps per measure and band", RequiredFiles: []string{"features_connectivity*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "connectivity_network", Group: "connectivity", Name: "Networks", Description: "Connectivity network visualizations per measure and band", RequiredFiles: []string{"features_connectivity*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	// Aperiodic
	{ID: "aperiodic_topomaps", Group: "aperiodic", Name: "Topomaps", Description: "Topographic maps of slope and offset", RequiredFiles: []string{"features_aperiodic*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "aperiodic_by_condition", Group: "aperiodic", Name: "Condition Comparison", Description: "Aperiodic differences between conditions", RequiredFiles: []string{"features_aperiodic*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Phase (ITPC/PAC)
	{ID: "itpc_topomaps", Group: "phase", Name: "ITPC Topomaps", Description: "Topographic maps of phase coherence", RequiredFiles: []string{"features_itpc*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "itpc_by_condition", Group: "phase", Name: "ITPC Condition Comparison", Description: "Phase coherence differences between conditions", RequiredFiles: []string{"features_itpc*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "pac_by_condition", Group: "phase", Name: "PAC Condition Comparison", Description: "PAC differences between conditions", RequiredFiles: []string{"features_pac*.tsv", "events.tsv"}, RequiresFeatures: true},
	// ERDS
	{ID: "erds_by_condition", Group: "erds", Name: "Condition Comparison", Description: "ERDS differences between conditions", RequiredFiles: []string{"features_erds*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Complexity
	{ID: "complexity_by_condition", Group: "complexity", Name: "Condition Comparison", Description: "Complexity differences between conditions", RequiredFiles: []string{"features_complexity*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Spectral
	{ID: "spectral_by_condition", Group: "spectral", Name: "Condition Comparison", Description: "Spectral differences between conditions", RequiredFiles: []string{"features_spectral*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Ratios
	{ID: "ratios_by_condition", Group: "ratios", Name: "Condition Comparison", Description: "Ratio differences between conditions", RequiredFiles: []string{"features_ratios*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Asymmetry
	{ID: "asymmetry_by_condition", Group: "asymmetry", Name: "Condition Comparison", Description: "Asymmetry differences between conditions", RequiredFiles: []string{"features_asymmetry*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Bursts
	{ID: "bursts_by_condition", Group: "bursts", Name: "Condition Comparison", Description: "Burst differences between conditions", RequiredFiles: []string{"features_bursts*.tsv", "events.tsv"}, RequiresFeatures: true},
	// ERP
	{ID: "erp_butterfly", Group: "erp", Name: "Butterfly", Description: "Butterfly ERP plots (all channels)", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_roi", Group: "erp", Name: "ROI Waveforms", Description: "ROI-based ERP waveforms with error bars", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_contrast", Group: "erp", Name: "Contrast", Description: "ERP condition contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	// TFR
	{ID: "tfr_scalpmean", Group: "tfr", Name: "Scalp-Mean TFR", Description: "Scalp-mean time-frequency representation", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_scalpmean_contrast", Group: "tfr", Name: "Scalp-Mean Contrast", Description: "Pain vs non-pain scalp-mean TFR contrast", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_channels", Group: "tfr", Name: "Channel TFRs", Description: "Time-frequency per channel", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_channels_contrast", Group: "tfr", Name: "Channel Contrasts", Description: "Pain vs non-pain channel TFR contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_rois", Group: "tfr", Name: "ROI TFRs", Description: "Time-frequency per ROI", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_rois_contrast", Group: "tfr", Name: "ROI Contrasts", Description: "Pain vs non-pain ROI TFR contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_topomaps", Group: "tfr", Name: "TFR Topomaps", Description: "Time-frequency topographic maps", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_band_evolution", Group: "tfr", Name: "Band Evolution", Description: "Frequency band power evolution over time", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Behavior
	{ID: "behavior_psychometrics", Group: "behavior", Name: "Psychometrics", Description: "Rating distributions and psychometrics", RequiredFiles: []string{"events.tsv"}},
	{ID: "behavior_scatter", Group: "behavior", Name: "Feature-Behavior Scatter", Description: "Configurable scatter plots correlating any EEG feature with behavioral columns", RequiredFiles: []string{"features_*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "behavior_temporal_topomaps", Group: "behavior", Name: "Temporal Topomaps", Description: "Temporal correlation topomaps", RequiredFiles: []string{"stats/temporal_correlations*/*/temporal_correlations_by_condition*.npz"}, RequiresStats: true},
	{ID: "behavior_dose_response", Group: "behavior", Name: "Dose Response", Description: "Dose-response curves and contrasts", RequiredFiles: []string{"stats/trial_table*/*/trials*.tsv", "stats/trial_table*/*/trials*.parquet"}, RequiresStats: true},
	{ID: "behavior_pain_probability", Group: "behavior", Name: "Pain Probability", Description: "Pain probability vs dose (binary outcome vs temperature)", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Machine Learning
	{ID: "ml_regression_plots", Group: "machine_learning", Name: "Regression Plots", Description: "LOSO regression diagnostics", RequiredFiles: []string{"machine_learning/regression/loso_predictions.tsv"}},
	{ID: "ml_timegen_plots", Group: "machine_learning", Name: "Time-Generalization", Description: "Time-generalization matrices", RequiredFiles: []string{"machine_learning/time_generalization/time_generalization_regression.npz"}},
	{ID: "ml_classification_plots", Group: "machine_learning", Name: "Classification", Description: "LOSO classification diagnostics", RequiredFiles: []string{"machine_learning/classification/loso_predictions.tsv"}},
	{ID: "ml_within_subject_regression_plots", Group: "machine_learning", Name: "Within-Subject Regression", Description: "Block-aware within-subject regression diagnostics", RequiredFiles: []string{"machine_learning/within_subject_regression/cv_predictions.tsv"}},
	{ID: "ml_within_subject_classification_plots", Group: "machine_learning", Name: "Within-Subject Classification", Description: "Block-aware within-subject classification diagnostics", RequiredFiles: []string{"machine_learning/within_subject_classification/cv_predictions.tsv"}},
}

var defaultPlotCategories = []FeatureCategory{
	{"power", "Power", "Band power features and topomaps"},
	{"connectivity", "Connectivity", "Functional connectivity measures and networks"},
	{"aperiodic", "Aperiodic", "1/f spectral slope and offset features"},
	{"phase", "Phase (ITPC/PAC)", "Phase coherence and phase-amplitude coupling"},
	{"erds", "ERDS", "Event-related desynchronization/synchronization"},
	{"complexity", "Complexity", "Lempel-Ziv complexity and permutation entropy"},
	{"spectral", "Spectral", "Peak frequency, spectral edge, and entropy"},
	{"ratios", "Ratios", "Band power ratios (theta/beta, alpha/beta, etc.)"},
	{"asymmetry", "Asymmetry", "Hemispheric asymmetry indices"},
	{"bursts", "Bursts", "Oscillatory burst dynamics"},
	{"quality", "Quality", "Data quality diagnostics and outlier detection"},
	{"erp", "ERP", "Event-related potential waveforms and topographies"},
	{"tfr", "Time-Frequency", "Time-frequency representations and contrasts"},
	{"behavior", "Behavior", "EEG-behavior correlations and temporal stats"},
	{"machine_learning", "Machine Learning", "Machine learning regression, time-generalization, and classification"},
}

type plotCatalogPayload struct {
	Groups []plotGroupPayload `json:"groups"`
	Plots  []plotItemPayload  `json:"plots"`
}

type plotGroupPayload struct {
	Key         string `json:"key"`
	Label       string `json:"label"`
	Description string `json:"description"`
}

type plotItemPayload struct {
	ID               string   `json:"id"`
	Group            string   `json:"group"`
	Label            string   `json:"label"`
	Description      string   `json:"description"`
	RequiredFiles    []string `json:"required_files"`
	RequiresEpochs   bool     `json:"requires_epochs"`
	RequiresFeatures bool     `json:"requires_features"`
	RequiresStats    bool     `json:"requires_stats"`
}

func loadPlotCatalog(repoRoot string) ([]PlotItem, []FeatureCategory, error) {
	catalogPath := filepath.Join(repoRoot, "eeg_pipeline", "plotting", "plot_catalog.json")
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		return nil, nil, err
	}

	var payload plotCatalogPayload
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, nil, err
	}

	items := make([]PlotItem, 0, len(payload.Plots))
	for _, plot := range payload.Plots {
		items = append(items, PlotItem{
			ID:               plot.ID,
			Group:            plot.Group,
			Name:             plot.Label,
			Description:      plot.Description,
			RequiredFiles:    plot.RequiredFiles,
			RequiresEpochs:   plot.RequiresEpochs,
			RequiresFeatures: plot.RequiresFeatures,
			RequiresStats:    plot.RequiresStats,
		})
	}

	categories := make([]FeatureCategory, 0, len(payload.Groups))
	for _, group := range payload.Groups {
		categories = append(categories, FeatureCategory{
			Key:         group.Key,
			Name:        group.Label,
			Description: group.Description,
		})
	}

	return items, categories, nil
}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	Pipeline    types.Pipeline
	CurrentStep types.WizardStep
	steps       []types.WizardStep
	stepIndex   int

	// Project setup (received from global config)
	task         string
	bidsRoot     string
	bidsFmriRoot string
	derivRoot    string
	sourceRoot   string
	repoRoot     string // Project repository root for running Python commands

	// Mode selection
	modeOptions      []string
	modeDescriptions []string
	modeIndex        int

	// Computation selection (for behavior)
	computations        []Computation
	computationSelected map[int]bool
	computationCursor   int
	computationOffset   int // Scroll offset for computations list

	// Category selection (for features pipeline)
	categories    []string
	categoryDescs []string
	categoryIndex int
	selected      map[int]bool

	// Preprocessing stage selection (for preprocessing pipeline)
	prepStages        []PreprocessingStage
	prepStageSelected map[int]bool
	prepStageCursor   int

	// Band selection (for features and behavior pipeline)
	bands        []FrequencyBand
	bandSelected map[int]bool
	bandCursor   int

	// Band editing state
	editingBandIdx   int    // Which band is being edited (-1 for none)
	editingBandField int    // 0: name, 1: lowHz, 2: highHz
	bandEditBuffer   string // Buffer for the value being typed

	// ROI selection (for features pipeline)
	rois        []ROIDefinition
	roiSelected map[int]bool
	roiCursor   int

	// ROI editing state
	editingROIIdx   int    // Which ROI is being edited (-1 for none)
	editingROIField int    // 0: name, 1: channels
	roiEditBuffer   string // Buffer for the value being typed

	// Spatial mode selection (for features pipeline)
	spatialSelected map[int]bool
	spatialCursor   int

	// Time range input (for features pipeline)
	TimeRanges      []types.TimeRange
	timeRangeCursor int // Which range is focused
	editingRangeIdx int // Which range is being edited (noRangeEditing for none)
	editingField    int // fieldName, fieldTmin, or fieldTmax

	// Feature availability with timestamps
	featureAvailability map[string]bool
	featureLastModified map[string]string

	// Computation availability with timestamps (for behavior pipeline)
	computationAvailability map[string]bool
	computationLastModified map[string]string

	// Feature file selection (for behavior pipeline)
	featureFiles        []FeatureFile
	featureFileSelected map[string]bool
	featureFileCursor   int

	// Plotting pipeline selection
	plotCategories         []FeatureCategory
	plotItems              []PlotItem
	plotSelected           map[int]bool
	plotCursor             int
	plotOffset             int // Scroll offset for plots
	featurePlotters        map[string][]PlotterInfo
	featurePlotterSelected map[string]bool
	featurePlotterCursor   int
	featurePlotterOffset   int
	featurePlotterError    string

	// Plotting output configuration
	plottingScope       PlottingScope
	plotFormats         []string
	plotFormatSelected  map[string]bool
	plotDpiOptions      []int
	plotDpiIndex        int
	plotSavefigDpiIndex int
	plotSharedColorbar  bool
	plotConfigCursor    int

	// fMRI preprocessing (fMRIPrep-style) configuration
	fmriEngineIndex           int    // 0: docker, 1: apptainer
	fmriFmriprepImage         string // Docker image or Apptainer URI/path
	fmriFmriprepOutputDir     string // Host path; default: deriv_root
	fmriFmriprepWorkDir       string // Host path; default: deriv_root/work/fmriprep
	fmriFreesurferLicenseFile string // Host path to license.txt
	fmriFreesurferSubjectsDir string // Host path (optional)
	fmriOutputSpacesSpec      string // Space-separated (e.g., "T1w MNI152NLin2009cAsym")
	fmriIgnoreSpec            string // Space-separated (e.g., "fieldmaps slicetiming")
	fmriBidsFilterFile        string // Host path to BIDS filter JSON (optional)
	fmriExtraArgs             string // Passed to fMRIPrep via shlex splitting
	fmriUseAroma              bool
	fmriSkipBidsValidation    bool
	fmriStopOnFirstCrash      bool
	fmriCleanWorkdir          bool
	fmriSkipReconstruction    bool // fMRIPrep: --fs-no-reconall
	fmriMemMb                 int
	// Additional fMRIPrep options
	fmriNThreads            int     // --nthreads (max threads across all processes)
	fmriOmpNThreads         int     // --omp-nthreads (max threads per process)
	fmriLowMem              bool    // --low-mem (reduce memory usage)
	fmriLongitudinal        bool    // --longitudinal (unbiased structural template)
	fmriCiftiOutputIndex    int     // 0: disabled, 1: 91k, 2: 170k
	fmriSkullStripTemplate  string  // --skull-strip-template (default: OASIS30ANTs)
	fmriSkullStripFixedSeed bool    // --skull-strip-fixed-seed (reproducibility)
	fmriRandomSeed          int     // --random-seed (run-to-run replicability)
	fmriDummyScans          int     // --dummy-scans (non-steady state volumes)
	fmriBold2T1wInitIndex   int     // 0: register (default), 1: header
	fmriBold2T1wDof         int     // --bold2t1w-dof (degrees of freedom, default: 6)
	fmriSliceTimeRef        float64 // --slice-time-ref (0=start, 0.5=middle, 1=end)
	fmriFdSpikeThreshold    float64 // --fd-spike-threshold (default: 0.5)
	fmriDvarsSpikeThreshold float64 // --dvars-spike-threshold (default: 1.5)
	fmriMeOutputEchos       bool    // --me-output-echos (multi-echo: output each echo)
	fmriMedialSurfaceNan    bool    // --medial-surface-nan (fill medial with NaN)
	fmriNoMsm               bool    // --no-msm (disable MSM-Sulc alignment)
	fmriLevelIndex          int     // 0: full (default), 1: resampling, 2: minimal
	fmriTaskId              string  // --task-id (process only specific task)

	// fMRI UI group expansion states (for collapsible sections)
	fmriGroupRuntimeExpanded     bool
	fmriGroupOutputExpanded      bool
	fmriGroupPerformanceExpanded bool
	fmriGroupAnatomicalExpanded  bool
	fmriGroupBoldExpanded        bool
	fmriGroupQcExpanded          bool
	fmriGroupDenoisingExpanded   bool
	fmriGroupSurfaceExpanded     bool
	fmriGroupMultiechoExpanded   bool
	fmriGroupReproExpanded       bool
	fmriGroupValidationExpanded  bool
	fmriGroupAdvancedExpanded    bool

	// fMRI analysis (first-level GLM + contrasts) configuration
	fmriAnalysisInputSourceIndex  int    // 0: fmriprep, 1: bids_raw
	fmriAnalysisFmriprepSpace     string // e.g., "T1w"
	fmriAnalysisRequireFmriprep   bool   // Fail if fMRIPrep outputs missing
	fmriAnalysisRunsSpec          string // Space-separated ints (e.g., "1 2 3") or empty for auto
	fmriAnalysisContrastType      int    // 0: t-test, 1: custom
	fmriAnalysisCondAColumn       string // Condition A: events column (e.g. trial_type, pain_binary_coded)
	fmriAnalysisCondAValue        string // Condition A: value in that column
	fmriAnalysisCondBColumn       string // Condition B: events column
	fmriAnalysisCondBValue        string // Condition B: value in that column
	fmriAnalysisContrastName      string // e.g., "pain_vs_nonpain"
	fmriAnalysisFormula           string // Custom formula
	fmriAnalysisEventsToModel     string // Optional: comma-separated list of trial_type values to include (first-level only)
	fmriAnalysisStimPhasesToModel string // "": auto (plateau if present), "all", or specific phase(s)
	fmriAnalysisHrfModel          int    // 0: spm, 1: flobs, 2: fir
	fmriAnalysisDriftModel        int    // 0: none, 1: cosine, 2: polynomial
	fmriAnalysisHighPassHz        float64
	fmriAnalysisLowPassHz         float64
	fmriAnalysisSmoothingFwhm     float64 // mm; 0 disables
	fmriAnalysisOutputType        int     // 0: z-score, 1: t-stat, 2: cope, 3: beta
	fmriAnalysisOutputDir         string  // Optional output directory override
	fmriAnalysisResampleToFS      bool
	fmriAnalysisFreesurferDir     string // Optional FreeSurfer SUBJECTS_DIR override
	fmriAnalysisConfoundsStrategy int    // 0..N (see render/command builder options)
	fmriAnalysisWriteDesignMatrix bool   // Write design matrices to <output>/qc/

	// fMRI analysis UI group expansion states
	fmriAnalysisGroupInputExpanded     bool
	fmriAnalysisGroupContrastExpanded  bool
	fmriAnalysisGroupGLMExpanded       bool
	fmriAnalysisGroupConfoundsExpanded bool
	fmriAnalysisGroupOutputExpanded    bool
	fmriAnalysisGroupPlottingExpanded  bool

	// fMRI analysis plotting/report configuration
	fmriAnalysisPlotsEnabled             bool
	fmriAnalysisPlotHTML                 bool
	fmriAnalysisPlotSpaceIndex           int     // 0: both, 1: native, 2: mni
	fmriAnalysisPlotThresholdModeIndex   int     // 0: z, 1: fdr, 2: none
	fmriAnalysisPlotZThreshold           float64 // default: 2.3
	fmriAnalysisPlotFdrQ                 float64 // default: 0.05
	fmriAnalysisPlotClusterMinVoxels     int     // default: 0
	fmriAnalysisPlotVmaxModeIndex        int     // 0: per-space robust, 1: shared robust, 2: manual
	fmriAnalysisPlotVmaxManual           float64 // default used when manual
	fmriAnalysisPlotIncludeUnthresholded bool
	fmriAnalysisPlotFormatPNG            bool
	fmriAnalysisPlotFormatSVG            bool
	fmriAnalysisPlotTypeSlices           bool
	fmriAnalysisPlotTypeGlass            bool
	fmriAnalysisPlotTypeHist             bool
	fmriAnalysisPlotTypeClusters         bool
	fmriAnalysisPlotEffectSize           bool
	fmriAnalysisPlotStandardError        bool
	fmriAnalysisPlotMotionQC             bool
	fmriAnalysisPlotCarpetQC             bool
	fmriAnalysisPlotTSNRQC               bool
	fmriAnalysisPlotDesignQC             bool
	fmriAnalysisPlotEmbedImages          bool
	fmriAnalysisPlotSignatures           bool
	fmriAnalysisSignatureDir             string // optional override; empty => auto

	// fMRI trial-wise signatures configuration (beta-series, LSS)
	fmriTrialSigGroupExpanded           bool
	fmriTrialSigMethodIndex             int // 0: beta-series, 1: lss
	fmriTrialSigIncludeOtherEvents      bool
	fmriTrialSigMaxTrialsPerRun         int // 0 = no cap
	fmriTrialSigFixedEffectsWeighting   int // 0: variance, 1: mean
	fmriTrialSigWriteTrialBetas         bool
	fmriTrialSigWriteTrialVariances     bool
		fmriTrialSigWriteConditionBetas     bool
		fmriTrialSigSignatureNPS            bool
		fmriTrialSigSignatureSIIPS1         bool
		fmriTrialSigLssOtherRegressorsIndex int // 0: per-condition, 1: all
		// Signature grouping (compute signatures for specific values within an events column)
		fmriTrialSigGroupColumn     string // e.g., temperature
		fmriTrialSigGroupValuesSpec string // space-separated values (e.g., "44.3 45.3 46.3")
		fmriTrialSigGroupScopeIndex int    // 0: across-runs (average), 1: per-run
	fmriTrialSigScopeStimPhases string // "": auto (plateau if present), "all", or specific phase(s)

	// Plotting advanced configuration (wizard overrides for `eeg-pipeline plotting visualize`)
	plotGroupDefaultsExpanded    bool
	plotGroupFontsExpanded       bool
	plotGroupLayoutExpanded      bool
	plotGroupFigureSizesExpanded bool
	plotGroupColorsExpanded      bool
	plotGroupAlphaExpanded       bool
	plotGroupScatterExpanded     bool
	plotGroupBarExpanded         bool
	plotGroupLineExpanded        bool
	plotGroupHistogramExpanded   bool
	plotGroupKDEExpanded         bool
	plotGroupErrorbarExpanded    bool
	plotGroupTextExpanded        bool
	plotGroupValidationExpanded  bool
	plotGroupTFRMiscExpanded     bool

	plotGroupTopomapExpanded     bool
	plotGroupTFRExpanded         bool
	plotGroupSizingExpanded      bool
	plotGroupSelectionExpanded   bool
	plotGroupComparisonsExpanded bool

	// Global styling panel state (shown in plot categories page)
	showGlobalStyling    bool
	globalStylingCursor  int
	globalStylingOptions []optionType

	// Per-plot advanced configuration (wizard overrides scoped to plot IDs)
	plotItemConfigs        map[string]PlotItemConfig
	plotItemConfigExpanded map[string]bool

	// Cached discovery for behavior_temporal_topomaps stats feature folders
	temporalTopomapsStatsFeatureFolders      []string
	temporalTopomapsStatsFeatureFoldersError string

	plotBboxInches string
	plotPadInches  float64

	plotFontFamily          string
	plotFontWeight          string
	plotFontSizeSmall       int
	plotFontSizeMedium      int
	plotFontSizeLarge       int
	plotFontSizeTitle       int
	plotFontSizeAnnotation  int
	plotFontSizeLabel       int
	plotFontSizeYLabel      int
	plotFontSizeSuptitle    int
	plotFontSizeFigureTitle int

	plotLayoutTightRectSpec           string
	plotLayoutTightRectMicrostateSpec string
	plotGridSpecWidthRatiosSpec       string
	plotGridSpecHeightRatiosSpec      string
	plotGridSpecHspace                float64
	plotGridSpecWspace                float64
	plotGridSpecLeft                  float64
	plotGridSpecRight                 float64
	plotGridSpecTop                   float64
	plotGridSpecBottom                float64

	plotFigureSizeStandardSpec string
	plotFigureSizeMediumSpec   string
	plotFigureSizeSmallSpec    string
	plotFigureSizeSquareSpec   string
	plotFigureSizeWideSpec     string
	plotFigureSizeTFRSpec      string
	plotFigureSizeTopomapSpec  string

	plotColorPain           string
	plotColorNonpain        string
	plotColorSignificant    string
	plotColorNonsignificant string
	plotColorGray           string
	plotColorLightGray      string
	plotColorBlack          string
	plotColorBlue           string
	plotColorRed            string
	plotColorNetworkNode    string

	plotAlphaGrid       float64
	plotAlphaFill       float64
	plotAlphaCI         float64
	plotAlphaCILine     float64
	plotAlphaTextBox    float64
	plotAlphaViolinBody float64
	plotAlphaRidgeFill  float64

	plotScatterMarkerSizeSmall   int
	plotScatterMarkerSizeLarge   int
	plotScatterMarkerSizeDefault int
	plotScatterAlpha             float64
	plotScatterEdgeColor         string
	plotScatterEdgeWidth         float64

	plotBarAlpha        float64
	plotBarWidth        float64
	plotBarCapsize      int
	plotBarCapsizeLarge int

	plotLineWidthThin       float64
	plotLineWidthStandard   float64
	plotLineWidthThick      float64
	plotLineWidthBold       float64
	plotLineAlphaStandard   float64
	plotLineAlphaDim        float64
	plotLineAlphaZeroLine   float64
	plotLineAlphaFitLine    float64
	plotLineAlphaDiagonal   float64
	plotLineAlphaReference  float64
	plotLineRegressionWidth float64
	plotLineResidualWidth   float64
	plotLineQQWidth         float64

	plotHistBins           int
	plotHistBinsBehavioral int
	plotHistBinsResidual   int
	plotHistBinsTFR        int
	plotHistEdgeColor      string
	plotHistEdgeWidth      float64
	plotHistAlpha          float64
	plotHistAlphaResidual  float64
	plotHistAlphaTFR       float64

	plotKdePoints    int
	plotKdeColor     string
	plotKdeLinewidth float64
	plotKdeAlpha     float64

	plotErrorbarMarkerSize   int
	plotErrorbarCapsize      int
	plotErrorbarCapsizeLarge int

	plotTextStatsX             float64
	plotTextStatsY             float64
	plotTextPvalueX            float64
	plotTextPvalueY            float64
	plotTextBootstrapX         float64
	plotTextBootstrapY         float64
	plotTextChannelAnnotationX float64
	plotTextChannelAnnotationY float64
	plotTextTitleY             float64
	plotTextResidualQcTitleY   float64

	plotValidationMinBinsForCalibration int
	plotValidationMaxBinsForCalibration int
	plotValidationSamplesPerBin         int
	plotValidationMinRoisForFDR         int
	plotValidationMinPvaluesForFDR      int

	plotTfrDefaultBaselineWindowSpec string

	plotTopomapContours               int
	plotTopomapColormap               string
	plotTopomapColorbarFraction       float64
	plotTopomapColorbarPad            float64
	plotTopomapDiffAnnotation         *bool
	plotTopomapAnnotateDesc           *bool
	plotTopomapSigMaskMarker          string
	plotTopomapSigMaskMarkerFaceColor string
	plotTopomapSigMaskMarkerEdgeColor string
	plotTopomapSigMaskLinewidth       float64
	plotTopomapSigMaskMarkerSize      float64

	plotTFRLogBase              float64
	plotTFRPercentageMultiplier float64

	plotTFRTopomapWindowSizeMs         float64
	plotTFRTopomapWindowCount          int
	plotTFRTopomapLabelXPosition       float64
	plotTFRTopomapLabelYPositionBottom float64
	plotTFRTopomapLabelYPosition       float64
	plotTFRTopomapTitleY               float64
	plotTFRTopomapTitlePad             int
	plotTFRTopomapSubplotsRight        float64
	plotTFRTopomapTemporalHspace       float64
	plotTFRTopomapTemporalWspace       float64

	plotRoiWidthPerBand   float64
	plotRoiWidthPerMetric float64
	plotRoiHeightPerRoi   float64

	plotPowerWidthPerBand     float64
	plotPowerHeightPerSegment float64

	plotItpcWidthPerBin     float64
	plotItpcHeightPerBand   float64
	plotItpcWidthPerBandBox float64
	plotItpcHeightBox       float64

	plotPacCmap        string
	plotPacWidthPerRoi float64
	plotPacHeightBox   float64

	plotAperiodicWidthPerColumn float64
	plotAperiodicHeightPerRow   float64
	plotAperiodicNPerm          int

	plotComplexityWidthPerMeasure  float64
	plotComplexityHeightPerSegment float64

	plotConnectivityWidthPerCircle     float64
	plotConnectivityWidthPerBand       float64
	plotConnectivityHeightPerMeasure   float64
	plotConnectivityCircleTopFraction  float64
	plotConnectivityCircleMinLines     int
	plotConnectivityNetworkTopFraction float64

	plotPacPairsSpec           string
	plotSpectralMetricsSpec    string
	plotBurstsMetricsSpec      string
	plotTemporalTimeBinsSpec   string
	plotTemporalTimeLabelsSpec string
	plotAsymmetryStatSpec      string

	// Plotting comparisons (global)
	plotCompareWindows        *bool
	plotComparisonWindowsSpec string
	plotCompareColumns        *bool
	plotComparisonSegment     string
	plotComparisonColumn      string
	plotComparisonValuesSpec  string
	plotComparisonLabelsSpec  string
	plotComparisonROIsSpec    string
	plotOverwrite             *bool // Overwrite existing plot files

	// Subject selection
	subjects                  []types.SubjectStatus
	subjectSelected           map[string]bool
	subjectCursor             int
	subjectsLoading           bool
	subjectFilter             string
	filteringSubject          bool
	availableWindows          []string
	availableWindowsByFeature map[string][]string // Windows per feature group (e.g., "itpc", "power")
	availableColumns          []string
	availableChannels         []string // EEG channels from electrodes.tsv
	unavailableChannels       []string // Bad channels from preprocessing log

	// Review/Execute
	ReadyToExecute    bool
	ConfirmingExecute bool
	DryRunMode        bool // If true, append --dry-run to command

	// Validation
	validationErrors []string

	// Help overlay
	helpOverlay components.HelpOverlay
	showHelp    bool

	// Animation
	ticker                int
	animQueue             animation.Queue
	subjectLoadingSpinner components.Spinner
	plotLoadingSpinner    components.Spinner

	width  int
	height int

	// Advanced configuration (shared)
	useDefaultAdvanced bool // True = skip advanced config customization
	advancedCursor     int  // Which config option is focused
	advancedOffset     int  // Scroll offset for advanced config lists

	// Multi-select expansion state for advanced config
	expandedOption int // expandedNone = none expanded
	subCursor      int // Cursor within the expanded list

	// Text input mode for numeric config values
	editingNumber bool   // True when typing a number
	numberBuffer  string // Buffer for the number being typed

	// Text input mode for string config values
	editingText      bool
	textBuffer       string
	editingTextField textField
	editingPlotID    string

	// File browsing mode for path selection
	browsingField            string  // Field being browsed (e.g., "sourceLocTrans", "sourceLocBem", "sourceLocFmriStatsMap")
	pendingFileCmd           tea.Cmd // Pending file picker command to execute
	pendingFmriConditionsCmd tea.Cmd // Pending fMRI conditions discovery command
	editingPlotField         plotItemConfigField

	// Features pipeline advanced config
	connectivityMeasures map[int]bool // Selected connectivity measures
	// Features advanced config section expansion (collapsed by default for compact UI)
	featGroupConnectivityExpanded     bool
	featGroupPACExpanded              bool
	featGroupAperiodicExpanded        bool
	featGroupComplexityExpanded       bool
	featGroupBurstsExpanded           bool
	featGroupPowerExpanded            bool
	featGroupSpectralExpanded         bool
	featGroupERPExpanded              bool
	featGroupRatiosExpanded           bool
	featGroupAsymmetryExpanded        bool
	featGroupSpatialTransformExpanded bool
	featGroupStorageExpanded          bool
	featGroupExecutionExpanded        bool
	featGroupTFRExpanded              bool

	// PAC/CFC configuration
	pacPhaseMin  float64 // Min phase frequency (Hz)
	pacPhaseMax  float64 // Max phase frequency (Hz)
	pacAmpMin    float64 // Min amplitude frequency (Hz)
	pacAmpMax    float64 // Max amplitude frequency (Hz)
	pacMethod    int     // 0: mvl, 1: kl, 2: tort, 3: ozkurt
	pacMinEpochs int
	pacPairsSpec string // e.g. theta:gamma,alpha:gamma

	// Aperiodic configuration
	aperiodicFmin      float64 // Min frequency for aperiodic fit
	aperiodicFmax      float64 // Max frequency for aperiodic fit
	aperiodicPeakZ     float64
	aperiodicMinR2     float64
	aperiodicMinPoints int

	// Complexity configuration
	complexityPEOrder int // Permutation entropy order (3-7)
	complexityPEDelay int

	// ERP configuration
	erpBaselineCorrection bool
	erpAllowNoBaseline    bool
	erpComponentsSpec     string  // e.g. n1=0.10-0.20,n2=0.20-0.35,p2=0.35-0.50
	erpSmoothMs           float64 // Smoothing window in ms (0.0 = no smoothing)
	erpPeakProminenceUv   float64 // Peak prominence threshold in µV (0 = use default)
	erpLowpassHz          float64 // Low-pass filter before ERP peak detection (Hz)

	// Burst configuration
	burstThresholdZ            float64
	burstThresholdMethod       int     // 0: percentile, 1: zscore, 2: mad
	burstThresholdPercentile   float64 // Percentile threshold (for percentile method)
	burstThresholdReference    int     // 0: trial, 1: subject, 2: condition
	burstMinTrialsPerCondition int     // used when threshold_reference="condition"
	burstMinSegmentSec         float64 // Minimum segment duration (sec) before attempting bursts
	burstSkipInvalidSegments   bool    // Skip invalid segments
	burstMinDuration           int     // ms
	burstMinCycles             float64 // Minimum oscillatory cycles for burst detection
	burstBandsSpec             string  // e.g. beta,gamma

	// Power configuration
	powerBaselineMode          int // 0: logratio, 1: mean, 2: ratio, 3: zscore, 4: zlogratio
	powerRequireBaseline       bool
	powerSubtractEvoked        bool
	powerMinTrialsPerCondition int
	powerExcludeLineNoise      bool
	powerLineNoiseFreq         float64
	powerLineNoiseWidthHz      float64
	powerLineNoiseHarmonics    int
	powerEmitDb                bool

	// Spectral configuration
	spectralEdgePercentile     float64
	spectralRatioPairsSpec     string // e.g. theta:beta,alpha:beta
	spectralPsdAdaptive        bool
	spectralMultitaperAdaptive bool

	// Aggregation
	aggregationMethod int // 0: mean, 1: median

	// Generic
	minEpochsForFeatures int

	// Execution (features pipeline)
	featAnalysisMode        int // 0: group_stats, 1: trial_ml_safe
	featComputeChangeScores bool
	featSaveTfrWithSidecar  bool
	featNJobsBands          int
	featNJobsConnectivity   int
	featNJobsAperiodic      int
	featNJobsComplexity     int

	// Storage configuration
	saveSubjectLevelFeatures bool
	featAlsoSaveCsv          bool // Also save feature tables as CSV files

	// Asymmetry
	asymmetryChannelPairsSpec         string  // e.g. F3:F4,C3:C4
	asymmetryMinSegmentSec            float64 // Minimum segment duration
	asymmetryMinCyclesAtFmin          float64 // Minimum cycles at lowest frequency
	asymmetrySkipInvalidSegments      bool    // Skip invalid segments
	asymmetryEmitActivationConvention bool
	asymmetryActivationBandsSpec      string // e.g. alpha,beta

	// Ratios
	ratiosMinSegmentSec       float64 // Minimum segment duration
	ratiosMinCyclesAtFmin     float64 // Minimum cycles at lowest frequency
	ratiosSkipInvalidSegments bool    // Skip invalid segments

	// Quality group expanded state
	featGroupQualityExpanded bool

	// ERDS group expanded state
	featGroupERDSExpanded bool

	// Connectivity configuration
	connOutputLevel  int // 0: full, 1: global_only
	connGraphMetrics bool
	connGraphProp    float64
	connWindowLen    float64
	connWindowStep   float64
	connAECMode      int // 0: orth, 1: none, 2: sym

	// Scientific validity options (new)
	itpcMethod             int     // 0: global, 1: fold_global, 2: loo, 3: condition
	aperiodicMinSegmentSec float64 // Minimum segment duration for aperiodic fits
	connAECOutput          int     // 0: r only, 1: z only, 2: both r and z
	connForceWithinEpochML bool    // Force within_epoch for CV/machine learning
	ratioSource            int     // 0: raw, 1: powcorr (aperiodic-adjusted)

	// Spectral validity options
	aperiodicSubtractEvoked bool // Subtract evoked response for induced spectra

	// ITPC additional options
	itpcAllowUnsafeLoo        bool   // Allow unsafe LOO ITPC
	itpcBaselineCorrection    int    // 0: none, 1: subtract
	itpcConditionColumn       string // Column for condition-based ITPC (avoids pseudo-replication)
	itpcConditionValues       string // Values to compute ITPC for (space-separated)
	itpcMinTrialsPerCondition int    // Minimum trials per condition for reliable ITPC
	itpcNJobs                 int    // Parallel jobs for ITPC computation (-1 = all CPUs)

	// Spectral advanced options
	spectralIncludeLogRatios   bool    // Include log ratios
	spectralPsdMethod          int     // 0: multitaper, 1: welch
	spectralFmin               float64 // Min frequency for spectral
	spectralFmax               float64 // Max frequency for spectral
	spectralExcludeLineNoise   bool    // Exclude line noise
	spectralLineNoiseFreq      float64 // Line noise frequency (50 or 60)
	spectralLineNoiseWidthHz   float64 // Line noise frequency band width to exclude
	spectralLineNoiseHarmonics int     // Number of line noise harmonics to exclude
	spectralMinSegmentSec      float64 // Minimum segment duration
	spectralMinCyclesAtFmin    float64 // Minimum cycles at lowest frequency

	// Band envelope options
	bandEnvelopePadSec    float64 // Padding in seconds
	bandEnvelopePadCycles float64 // Padding in cycles

	// IAF (Individualized Alpha Frequency) options
	iafEnabled                  bool    // Use individualized bands
	iafAlphaWidthHz             float64 // Alpha band width
	iafSearchRangeMin           float64 // IAF search range min
	iafSearchRangeMax           float64 // IAF search range max
	iafMinProminence            float64 // IAF peak prominence threshold
	iafRoisSpec                 string  // IAF ROIs (comma-separated)
	iafMinCyclesAtFmin          float64 // Minimum cycles at iaf_search_range_hz[0]
	iafMinBaselineSec           float64 // Additional absolute minimum baseline duration (sec)
	iafAllowFullFallback        bool    // If baseline missing, allow full segment (not recommended)
	iafAllowAllChannelsFallback bool    // If ROIs missing, allow all channels fallback

	// Aperiodic advanced options
	aperiodicModel              int     // 0: fixed
	aperiodicPsdMethod          int     // 0: multitaper, 1: welch
	aperiodicPsdBandwidth       float64 // PSD bandwidth (0 = use default)
	aperiodicMaxRms             float64 // Max RMS for aperiodic fit (0 = no limit)
	aperiodicExcludeLineNoise   bool    // Exclude line noise
	aperiodicLineNoiseFreq      float64 // Line noise frequency
	aperiodicLineNoiseWidthHz   float64 // Line noise frequency band width to exclude
	aperiodicLineNoiseHarmonics int     // Number of line noise harmonics to exclude

	// Spatial transform options (for volume conduction reduction)
	spatialTransform          int     // 0: none, 1: csd, 2: laplacian
	spatialTransformLambda2   float64 // Lambda2 regularization
	spatialTransformStiffness float64 // Stiffness parameter

	// Connectivity advanced options
	connGranularity            int     // 0: trial, 1: condition, 2: subject
	connConditionColumn        string  // Condition grouping column (events.tsv)
	connConditionValues        string  // Allowed condition values (comma/space-separated)
	connMinEpochsPerGroup      int     // Minimum epochs per group
	connMinCyclesPerBand       float64 // Minimum cycles per band
	connWarnNoSpatialTransform bool    // Warn if no spatial transform
	connPhaseEstimator         int     // 0: within_epoch, 1: across_epochs
	connMinSegmentSec          float64 // Minimum segment duration

	// Directed connectivity options (PSI, DTF, PDC)
	directedConnMeasures          map[int]bool // Selected directed connectivity measures
	directedConnEnabled           bool         // Enable directed connectivity extraction
	directedConnOutputLevel       int          // 0: full, 1: global_only
	directedConnMvarOrder         int          // MVAR model order for DTF/PDC
	directedConnNFreqs            int          // Number of frequency bins
	directedConnMinSegSamples     int          // Minimum segment samples
	featGroupDirectedConnExpanded bool         // UI expansion state

	// Source localization options (LCMV, eLORETA)
	sourceLocEnabled           bool    // Enable source localization
	sourceLocMode              int     // 0: EEG-only (template), 1: fMRI-informed (subject-specific)
	sourceLocMethod            int     // 0: lcmv, 1: eloreta
	sourceLocSpacing           int     // 0: oct5, 1: oct6, 2: ico4, 3: ico5
	sourceLocParc              int     // 0: aparc, 1: aparc.a2009s, 2: HCPMMP1
	sourceLocReg               float64 // LCMV regularization
	sourceLocSnr               float64 // eLORETA SNR
	sourceLocLoose             float64 // eLORETA loose constraint
	sourceLocDepth             float64 // eLORETA depth weighting
	sourceLocConnMethod        int     // 0: aec, 1: wpli, 2: plv
	sourceLocSubject           string  // FreeSurfer subject name (e.g., sub-0001)
	sourceLocTrans             string  // EEG↔MRI transform .fif
	sourceLocBem               string  // BEM solution .fif
	sourceLocMindistMm         float64 // MNE mindist (mm)
	sourceLocFmriEnabled       bool    // Enable fMRI-informed source localization
	sourceLocFmriStatsMap      string  // Path to fMRI stats map NIfTI
	sourceLocFmriProvenance    int     // 0: independent, 1: same_dataset
	sourceLocFmriRequireProv   bool    // Require explicit provenance
	sourceLocFmriThreshold     float64 // Threshold (e.g., z>=3.1)
	sourceLocFmriTail          int     // 0: pos, 1: abs
	sourceLocFmriMinClusterVox int     // Minimum cluster size (voxels)
	sourceLocFmriMinClusterMM3 float64 // Minimum cluster volume (mm^3); preferred when > 0
	sourceLocFmriMaxClusters   int     // Max clusters retained
	sourceLocFmriMaxVoxPerClus int     // Max voxels sampled per cluster
	sourceLocFmriMaxTotalVox   int     // Max total voxels across clusters
	sourceLocFmriRandomSeed    int     // Random seed for voxel subsampling

	// BEM/Trans generation options (Docker-based)
	sourceLocCreateTrans        bool // Auto-create coregistration transform via Docker
	sourceLocAllowIdentityTrans bool // Allow creating identity transform (DEBUG ONLY)
	sourceLocCreateBemModel     bool // Auto-create BEM model via Docker
	sourceLocCreateBemSolution  bool // Auto-create BEM solution via Docker

	// fMRI GLM contrast builder (for fMRI-informed mode)
	sourceLocFmriContrastEnabled   bool     // Build contrast from BOLD data (vs. load pre-computed)
	sourceLocFmriContrastType      int      // 0: t-test, 1: paired t-test, 2: F-test, 3: custom formula
	sourceLocFmriCondAColumn       string   // Condition A column (e.g., "trial_type", "pain_binary")
	sourceLocFmriCondAValue        string   // Condition A value (e.g., "temp49p3", "1")
	sourceLocFmriCondBColumn       string   // Condition B column
	sourceLocFmriCondBValue        string   // Condition B value
	sourceLocFmriConditions        []string // Discovered conditions from fMRI events files (for backward compat)
	sourceLocFmriCondIdx1          int      // Index into discovered conditions for Condition A
	sourceLocFmriCondIdx2          int      // Index into discovered conditions for Condition B
	sourceLocFmriContrastFormula   string   // Custom formula (e.g., "pain_high - pain_low")
	sourceLocFmriContrastName      string   // Contrast name (e.g., "pain_vs_baseline")
	sourceLocFmriRunsToInclude     string   // Comma-separated runs (e.g., "1,2,3")
	sourceLocFmriAutoDetectRuns    bool     // Auto-detect available BOLD runs
	sourceLocFmriHrfModel          int      // 0: SPM, 1: FLOBS, 2: FIR
	sourceLocFmriDriftModel        int      // 0: none, 1: cosine, 2: polynomial
	sourceLocFmriHighPassHz        float64  // High-pass cutoff (Hz)
	sourceLocFmriLowPassHz         float64  // Low-pass cutoff (Hz)
	sourceLocFmriStimPhasesToModel string   // "": auto (plateau if present), "all", or specific phase(s)
	sourceLocFmriClusterCorrection bool     // Enable cluster-extent filtering heuristic (NOT cluster-level FWE correction)
	sourceLocFmriClusterPThreshold float64  // Cluster-forming p-threshold
	sourceLocFmriOutputType        int      // 0: z-score, 1: t-stat, 2: cope, 3: beta
	sourceLocFmriResampleToFS      bool     // Auto-resample to FreeSurfer space

	// fMRI-specific time windows (independent of EEG feature extraction windows)
	sourceLocFmriWindowAName string  // Name for window A (e.g., "plateau")
	sourceLocFmriWindowATmin float64 // Start time for window A (seconds)
	sourceLocFmriWindowATmax float64 // End time for window A (seconds)
	sourceLocFmriWindowBName string  // Name for window B (e.g., "baseline")
	sourceLocFmriWindowBTmin float64 // Start time for window B (seconds)
	sourceLocFmriWindowBTmax float64 // End time for window B (seconds)

	// Source localization UI expansion states
	featGroupSourceLocExpanded         bool // UI expansion state for source localization
	featGroupSourceLocFmriExpanded     bool // UI expansion state for fMRI constraint
	featGroupSourceLocContrastExpanded bool // UI expansion state for contrast builder
	featGroupSourceLocGLMExpanded      bool // UI expansion state for GLM config
	featGroupITPCExpanded              bool // UI expansion state for ITPC options

	// PAC advanced options
	pacSource              int     // 0: precomputed
	pacNormalize           bool    // Normalize PAC values
	pacNSurrogates         int     // Number of surrogates (0=none)
	pacAllowHarmonicOvrlap bool    // Allow harmonic overlap
	pacMaxHarmonic         int     // Maximum harmonic to check
	pacHarmonicToleranceHz float64 // Harmonic tolerance in Hz
	pacComputeWaveformQC   bool    // Compute waveform QC
	pacWaveformOffsetMs    float64 // Waveform offset in ms
	pacRandomSeed          int     // Random seed for PAC surrogate testing

	// Complexity advanced options
	complexitySignalBasis   int     // 0: filtered, 1: envelope
	complexityMinSegmentSec float64 // Minimum segment duration (sec)
	complexityMinSamples    int     // Minimum samples per segment
	complexityZscore        bool    // Apply z-score normalization

	// Quality feature options
	qualityPsdMethod          int     // 0: welch, 1: multitaper
	qualityFmin               float64 // Min frequency
	qualityFmax               float64 // Max frequency
	qualityNfft               int     // FFT size
	qualityExcludeLineNoise   bool    // Exclude line noise
	qualityLineNoiseFreq      float64 // Line noise frequency (50 or 60)
	qualityLineNoiseWidthHz   float64 // Line noise width
	qualityLineNoiseHarmonics int     // Line noise harmonics
	qualitySnrSignalBandMin   float64 // SNR signal band min
	qualitySnrSignalBandMax   float64 // SNR signal band max
	qualitySnrNoiseBandMin    float64 // SNR noise band min
	qualitySnrNoiseBandMax    float64 // SNR noise band max
	qualityMuscleBandMin      float64 // Muscle band min
	qualityMuscleBandMax      float64 // Muscle band max

	// ERDS advanced options
	erdsUseLogRatio      bool    // Use dB instead of percent
	erdsMinBaselinePower float64 // Minimum baseline power
	erdsMinActivePower   float64 // Minimum active power
	erdsMinSegmentSec    float64 // Minimum segment duration
	erdsBandsSpec        string  // Bands for ERDS (comma-separated)

	// Temporal feature selection (behavior pipeline)
	temporalFeaturePowerEnabled bool // Power temporal enabled
	temporalFeatureITPCEnabled  bool // ITPC temporal enabled
	temporalFeatureERDSEnabled  bool // ERDS temporal enabled

	// Time-frequency heatmap options
	tfHeatmapEnabled   bool   // Enable TF heatmap
	tfHeatmapFreqsSpec string // Frequencies for heatmap
	tfHeatmapTimeResMs int    // Time resolution in ms

	// TFR advanced options
	tfrMaxCycles  float64 // Maximum cycles for wavelets
	tfrDecimPower int     // Decimation for power TFR
	tfrDecimPhase int     // Decimation for phase TFR

	// Behavior pipeline advanced config
	correlationMethod     string  // "spearman" or "pearson"
	robustCorrelation     int     // 0=none, 1=percentage_bend, 2=winsorized, 3=shepherd
	bootstrapSamples      int     // 0 = disabled, 1000+ recommended
	nPermutations         int     // For cluster tests
	rngSeed               int     // 0 = use project default
	controlTemperature    bool    // Include temperature as covariate
	controlTrialOrder     bool    // Include trial order as covariate
	fdrAlpha              float64 // FDR correction threshold
	behaviorConfigSection int
	behaviorNJobs         int // -1 = all

	behaviorComputeChangeScores  bool
	behaviorComputeBayesFactors  bool
	behaviorComputeLosoStability bool

	// Run adjustment (subject-level; optional)
	runAdjustmentEnabled               bool
	runAdjustmentColumn                string
	runAdjustmentIncludeInCorrelations bool
	runAdjustmentMaxDummies            int

	// Output options
	alsoSaveCsv       bool // Also save output tables as CSV files
	behaviorOverwrite bool // Overwrite existing output folders (if false, append timestamp)

	// Behavior advanced config section expansion (collapsed by default for compact UI)
	behaviorGroupGeneralExpanded      bool
	behaviorGroupTrialTableExpanded   bool
	behaviorGroupPainResidualExpanded bool
	behaviorGroupCorrelationsExpanded bool
	behaviorGroupPainSensExpanded     bool
	behaviorGroupRegressionExpanded   bool
	behaviorGroupModelsExpanded       bool
	behaviorGroupStabilityExpanded    bool
	behaviorGroupConsistencyExpanded  bool
	behaviorGroupInfluenceExpanded    bool
	behaviorGroupReportExpanded       bool
	behaviorGroupConditionExpanded    bool
	behaviorGroupTemporalExpanded     bool
	behaviorGroupClusterExpanded      bool
	behaviorGroupMediationExpanded    bool
	behaviorGroupModerationExpanded   bool
	behaviorGroupMixedEffectsExpanded bool
	behaviorGroupOutputExpanded       bool

	// Trial table / pain residual config (subject-level)
	trialTableFormat         int // 0=parquet, 1=tsv
	trialTableAddLagFeatures bool

	// Trial order validation (used when controlTrialOrder=true)
	trialOrderMaxMissingFraction float64

	featureSummariesEnabled bool

	// Feature QC (pre-statistics gating)
	featureQCEnabled                bool
	featureQCMaxMissingPct          float64
	featureQCMinVariance            float64
	featureQCCheckWithinRunVariance bool

	painResidualEnabled                 bool
	painResidualMethod                  int // 0=spline, 1=poly
	painResidualPolyDegree              int
	painResidualSplineDfCandidates      string // Comma-separated list (e.g., "3,4,5")
	painResidualModelCompareEnabled     bool
	painResidualModelComparePolyDegrees string // Comma-separated list (e.g., "2,3")
	painResidualBreakpointEnabled       bool
	painResidualBreakpointCandidates    int
	painResidualBreakpointQlow          float64
	painResidualBreakpointQhigh         float64

	// Pain residual cross-fit (out-of-run prediction)
	painResidualCrossfitEnabled     bool
	painResidualCrossfitGroupColumn string
	painResidualCrossfitNSplits     int
	painResidualCrossfitMethod      int // 0=spline, 1=poly
	painResidualCrossfitSplineKnots int

	// Regression
	regressionOutcome            int // 0=rating, 1=pain_residual, 2=temperature
	regressionIncludeTemperature bool
	regressionTempControl        int // 0=linear, 1=rating_hat, 2=spline
	regressionTempSplineKnots    int
	regressionTempSplineQlow     float64
	regressionTempSplineQhigh    float64
	regressionIncludeTrialOrder  bool
	regressionIncludePrev        bool
	regressionIncludeRunBlock    bool
	regressionIncludeInteraction bool
	regressionStandardize        bool
	regressionPermutations       int
	regressionMaxFeatures        int // 0 = no limit

	// Models
	modelsIncludeTemperature  bool
	modelsTempControl         int // 0=linear, 1=rating_hat, 2=spline
	modelsTempSplineKnots     int
	modelsTempSplineQlow      float64
	modelsTempSplineQhigh     float64
	modelsIncludeTrialOrder   bool
	modelsIncludePrev         bool
	modelsIncludeRunBlock     bool
	modelsIncludeInteraction  bool
	modelsStandardize         bool
	modelsMaxFeatures         int
	modelsOutcomeRating       bool
	modelsOutcomePainResidual bool
	modelsOutcomeTemperature  bool
	modelsOutcomePainBinary   bool
	modelsFamilyOLS           bool
	modelsFamilyRobust        bool
	modelsFamilyQuantile      bool
	modelsFamilyLogit         bool
	modelsBinaryOutcome       int // 0=pain_binary, 1=rating_median

	// Stability
	stabilityMethod      int // 0=spearman, 1=pearson
	stabilityOutcome     int // 0=auto, 1=rating, 2=pain_residual
	stabilityGroupColumn int // 0=auto, 1=run, 2=block
	stabilityPartialTemp bool
	stabilityMaxFeatures int
	stabilityAlpha       float64

	// Consistency & influence
	consistencyEnabled           bool
	influenceOutcomeRating       bool
	influenceOutcomePainResidual bool
	influenceOutcomeTemperature  bool
	influenceMaxFeatures         int
	influenceIncludeTemperature  bool
	influenceTempControl         int // 0=linear, 1=rating_hat, 2=spline
	influenceTempSplineKnots     int
	influenceTempSplineQlow      float64
	influenceTempSplineQhigh     float64
	influenceTempSplineMinN      int
	influenceIncludeTrialOrder   bool
	influenceIncludeRunBlock     bool
	influenceIncludeInteraction  bool
	influenceStandardize         bool
	influenceCooksThreshold      float64 // 0 = default
	influenceLeverageThreshold   float64 // 0 = default

	// Correlations (trial-table)
	correlationsTargetRating        bool
	correlationsTargetTemperature   bool
	correlationsTargetPainResidual  bool
	correlationsTypesSpec           string // Comma-separated list (e.g., "partial_cov_temp,raw")
	correlationsPreferPainResidual  bool
	correlationsUseCrossfitResidual bool
	correlationsPrimaryUnit         int // 0=trial, 1=run_mean
	correlationsPermutationPrimary  bool
	correlationsTargetColumn        string // Custom target column from events (dropdown)
	groupLevelBlockPermutation      bool   // Use block-restricted permutations when block/run is available

	// Pain sensitivity

	// Report
	reportTopN int

	// Temporal
	temporalResolutionMs       int
	temporalSmoothMs           int
	temporalTimeMinMs          int
	temporalTimeMaxMs          int
	temporalTargetColumn       string // events.tsv column to correlate against (empty=default rating)
	temporalSplitByCondition   bool   // If true, compute separate correlations per condition value
	temporalConditionColumn    string // Column to split by (empty = use event_columns.pain_binary)
	temporalConditionValues    string // Values to compute (empty = all unique values)
	temporalIncludeROIAverages bool   // Include ROI-averaged rows in output
	temporalIncludeTFGrid      bool   // Include individual frequency (TF grid) rows
	// ITPC-specific parameters
	temporalITPCBaselineCorrection bool    // Subtract baseline ITPC
	temporalITPCBaselineMin        float64 // Baseline window start
	temporalITPCBaselineMax        float64 // Baseline window end
	// ERDS-specific parameters
	temporalERDSBaselineMin float64 // ERDS baseline window start (seconds)
	temporalERDSBaselineMax float64 // ERDS baseline window end (seconds)
	temporalERDSMethod      int     // 0=percent, 1=zscore

	// Mixed effects (group-level; still configurable)
	mixedEffectsType int // 0=intercept, 1=intercept_slope

	// Mediation
	mediationMinEffect float64

	// Condition extras
	conditionFailFast           bool
	conditionPermutationPrimary bool
	conditionWindowPrimaryUnit  int // 0=trial, 1=run_mean
	// Cluster-specific
	clusterThreshold       float64 // Forming threshold for clusters
	clusterMinSize         int     // Minimum cluster size
	clusterTail            int     // 0=two-tailed, 1=upper, -1=lower
	clusterConditionColumn string  // events.tsv column to split by (empty=event_columns.pain_binary)
	clusterConditionValues string  // Exactly 2 values (space/comma-separated) to compare
	// Mediation-specific
	mediationBootstrap           int  // Bootstrap iterations for mediation
	mediationMaxMediatorsEnabled bool // Enable max mediators limit
	mediationMaxMediators        int  // Max mediators to test
	mediationPermutations        int  // Permutation iterations for mediation (0=disabled)
	// Moderation-specific
	moderationMaxFeaturesEnabled bool // Enable max features limit
	moderationMaxFeatures        int  // Max features for moderation
	moderationPermutations       int  // Permutation iterations for moderation (0=disabled)
	// Mixed effects-specific
	mixedMaxFeatures int // Max features for mixed effects
	// Condition-specific
	conditionEffectThreshold float64 // Min effect size to report
	conditionCompareColumn   string  // Column to use for condition split (empty=event_columns.pain_binary)
	conditionCompareWindows  string  // Time windows to compare (e.g., "baseline active")
	conditionCompareValues   string  // Values in the column to compare (e.g., "0,1" or "pain,nonpain")
	conditionOverwrite       bool    // Overwrite existing condition effects files

	// Column discovery (populated from events files)
	discoveredColumns       []string            // Available columns from events/trial table
	discoveredColumnValues  map[string][]string // Values for each column
	columnDiscoveryDone     bool                // Whether discovery has been completed
	columnDiscoveryError    string              // Error message if discovery failed
	columnDiscoverySource   string              // "events" or "trial_table"
	selectedColumnCursor    int                 // Cursor for column selection
	selectedValueCursors    map[string]int      // Cursor for value selection per column
	expandedColumnSelection string              // Currently expanded column for value selection

	// Trial table discovery (separate from events discovery; includes feature columns)
	trialTableColumns           []string            // Columns from trial table (including features)
	trialTableColumnValues      map[string][]string // Values for each column (if discovered)
	trialTableFeatureCategories []string            // Detected feature categories from trial table columns
	trialTableDiscoveryDone     bool
	trialTableDiscoveryError    string

	// Condition effects discovery (populated from condition effects files for plotting)
	conditionEffectsColumns        []string            // Available columns from condition effects files
	conditionEffectsColumnValues   map[string][]string // Values for each condition effects column
	conditionEffectsWindows        []string            // Available windows from condition effects files
	conditionEffectsDiscoveryDone  bool                // Whether condition effects discovery has been completed
	conditionEffectsDiscoveryError string              // Error message if condition effects discovery failed

	// fMRI column discovery (separate from EEG events - for fMRI contrast builder)
	fmriDiscoveredColumns      []string            // Available columns from fMRI events files
	fmriDiscoveredColumnValues map[string][]string // Values for each fMRI column
	fmriColumnDiscoveryDone    bool                // Whether fMRI discovery has been completed
	fmriColumnDiscoveryError   string              // Error message if fMRI discovery failed

	// Multigroup stats discovery (populated from precomputed stats)
	multigroupStatsAvailable     bool     // Whether multigroup stats are available
	multigroupStatsGroups        []string // Group labels from precomputed stats
	multigroupStatsNFeatures     int      // Number of features with stats
	multigroupStatsNSignificant  int      // Number of FDR-significant comparisons
	multigroupStatsFile          string   // Path to stats file
	multigroupStatsDiscoveryDone bool     // Whether discovery has been completed

	// ROI discovery (populated from feature data)
	discoveredROIs    []string // Available ROIs from feature parquet files
	roiDiscoveryDone  bool     // Whether ROI discovery has been completed
	roiDiscoveryError string   // Error message if ROI discovery failed

	// Machine Learning pipeline advanced config
	mlNPerm     int // Permutations for significance test
	innerSplits int // CV inner splits
	outerJobs   int // Number of parallel jobs for outer CV
	mlScope     MLCVScope
	mlTarget    string

	// ML: fMRI signature target settings (used when mlTarget == "fmri_signature")
	mlFmriSigGroupExpanded      bool
	mlFmriSigMethodIndex        int    // 0: beta-series, 1: lss
	mlFmriSigContrastName       string // e.g., pain_vs_nonpain
	mlFmriSigSignatureIndex     int    // 0: NPS, 1: SIIPS1
	mlFmriSigMetricIndex        int    // 0: dot, 1: cosine, 2: pearson_r
	mlFmriSigNormalizationIndex int    // 0: none, 1..: zscore/robust options
	mlFmriSigRoundDecimals      int    // default: 3

	mlBinaryThresholdEnabled bool
	mlBinaryThreshold        float64

	mlFeatureFamiliesSpec    string
	mlFeatureBandsSpec       string
	mlFeatureSegmentsSpec    string
	mlFeatureScopesSpec      string
	mlFeatureStatsSpec       string
	mlFeatureHarmonization   MLFeatureHarmonization
	mlCovariatesSpec         string
	mlBaselinePredictorsSpec string

	mlRegressionModel     MLRegressionModel
	mlClassificationModel MLClassificationModel
	mlRequireTrialMlSafe  bool

	mlUncertaintyAlpha float64
	mlPermNRepeats     int

	// Machine Learning model hyperparameters
	elasticNetAlphaGrid   string // alpha grid as comma-separated values
	elasticNetL1RatioGrid string // l1_ratio grid as comma-separated values
	ridgeAlphaGrid        string // ridge alpha grid as comma-separated values
	rfNEstimators         int    // Random forest n_estimators
	rfMaxDepthGrid        string // max_depth grid as comma-separated values (use "null" for None)
	varianceThresholdGrid string // variance_threshold grid (e.g. 0.0 or 0.0,0.01,0.1); use 0.0 only for small train folds

	// TFR parameters (for features pipeline)
	tfrFreqMin       float64 // Min frequency for TFR
	tfrFreqMax       float64 // Max frequency for TFR
	tfrNFreqs        int     // Number of frequency bins
	tfrMinCycles     float64 // Minimum cycles for wavelet
	tfrNCyclesFactor float64 // Cycles factor
	tfrDecim         int     // Decimation for power TFR
	tfrWorkers       int     // Workers for parallel TFR (-1 = all)

	// System/global settings
	systemNJobs      int  // Global n_jobs (-1 = all)
	systemStrictMode bool // Strict mode for validation
	loggingLevel     int  // 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

	// ICA labels to keep
	icaLabelsToKeep string // Comma-separated ICA labels (e.g., "brain,other")

	// Preprocessing pipeline advanced config
	prepUsePyprep    bool
	prepUseIcalabel  bool
	prepNJobs        int
	prepMontage      string // EEG montage (e.g., easycap-M1)
	prepResample     int
	prepLFreq        float64
	prepHFreq        float64
	prepNotch        int
	prepLineFreq     int     // Line frequency (50 or 60 Hz)
	prepChTypes      string  // Channel types (e.g., "eeg")
	prepEegReference string  // EEG reference (e.g., "average")
	prepEogChannels  string  // EOG channels (e.g., "Fp1,Fp2")
	prepRandomState  int     // Random seed for reproducibility
	prepTaskIsRest   bool    // Whether task is resting-state
	prepZaplineFline float64 // Zapline filtering frequency (Hz)
	prepFindBreaks   bool    // Find breaks in data
	// PyPREP advanced options
	prepRansac               bool   // Use RANSAC for bad channel detection
	prepRepeats              int    // Number of detection iterations
	prepAverageReref         bool   // Average re-reference before detection
	prepFileExtension        string // File extension (e.g., .vhdr)
	prepConsiderPreviousBads bool   // Keep previously marked bad channels
	prepOverwriteChansTsv    bool   // Overwrite channels.tsv file
	prepDeleteBreaks         bool   // Delete breaks in data
	prepBreaksMinLength      int    // Minimum break duration (seconds)
	prepTStartAfterPrevious  int    // Time after previous event (seconds)
	prepTStopBeforeNext      int    // Time before next event (seconds)
	prepRenameAnotDict       string // Dictionary to rename annotations (config-only)
	prepCustomBadDict        string // Dictionary of custom bad channels (config-only)
	// ICA options
	prepSpatialFilter   int     // 0: ica, 1: ssp
	prepICAAlgorithm    int     // 0: extended_infomax, 1: fastica, 2: infomax, 3: picard
	prepICAComp         float64 // ICA components (variance fraction)
	prepICALFreq        float64 // ICA high-pass filter frequency
	prepICARejThresh    float64 // ICA rejection threshold (µV)
	prepProbThresh      float64 // ICA label probability threshold
	prepKeepMnebidsBads bool    // Keep MNE-BIDS flagged bad ICAs
	// Epoching options
	prepConditions          string // Epoching conditions (comma-separated)
	prepEpochsTmin          float64
	prepEpochsTmax          float64
	prepEpochsBaselineStart float64 // Epoch baseline start (seconds)
	prepEpochsBaselineEnd   float64 // Epoch baseline end (seconds)
	prepEpochsNoBaseline    bool    // Disable baseline correction
	prepEpochsReject        float64 // Peak-to-peak rejection threshold (µV)
	prepRejectMethod        int     // 0: none, 1: autoreject_local, 2: autoreject_global
	prepRunSourceEstimation bool    // Run source estimation
	// Clean events.tsv options
	prepWriteCleanEvents     bool // Write clean events.tsv aligned to kept epochs
	prepOverwriteCleanEvents bool // Overwrite existing clean events.tsv
	prepCleanEventsStrict    bool // Fail if clean events.tsv cannot be written

	// Preprocessing UI group expansion states (for collapsible sections)
	prepGroupStagesExpanded    bool
	prepGroupGeneralExpanded   bool
	prepGroupFilteringExpanded bool
	prepGroupPyprepExpanded    bool
	prepGroupICAExpanded       bool
	prepGroupEpochingExpanded  bool

	// Utilities (raw-to-bids/merge) advanced config
	rawMontage           string
	rawLineFreq          int
	rawOverwrite         bool
	rawTrimToFirstVolume bool
	rawEventPrefixes     string
	rawKeepAnnotations   bool
	mergeEventPrefixes   string
	mergeEventTypes      string
	// Utilities (fMRI raw-to-bids) advanced config
	fmriRawSession          string
	fmriRawRestTask         string
	fmriRawIncludeRest      bool
	fmriRawIncludeFieldmaps bool
	fmriRawDicomModeIndex   int // 0=symlink, 1=copy, 2=skip
	fmriRawOverwrite        bool
	fmriRawCreateEvents     bool
	fmriRawEventGranularity int     // 0=phases, 1=trial
	fmriRawOnsetRefIndex    int     // 0=as_is, 1=first_iti_start, 2=first_stim_start
	fmriRawOnsetOffsetS     float64 // seconds
	fmriRawDcm2niixPath     string
	fmriRawDcm2niixArgs     string // Comma-separated

	// Preset system
	activePreset   string // Name of currently applied preset (empty if custom)
	showPresetMenu bool   // Show preset selection overlay
	presetCursor   int    // Cursor in preset menu

	// Toast notifications
	toastMessage string // Current toast message
	toastType    string // "success", "error", "warning", "info"
	toastTicker  int    // Countdown for toast visibility

	// Subject selection enhancements
	showOnlyValid    bool // Filter to show only valid subjects
	subjectViewMode  int  // 0=list, 1=grid
	subjectSortMode  int  // 0=id, 1=status, 2=date
	subjectScrollTop int  // Scroll offset for subject list
}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New(pipeline types.Pipeline, repoRoot string) Model {
	help := components.NewHelpOverlay("Wizard Shortcuts", 55)
	help.AddSection("Navigation", []components.HelpItem{
		{Key: "↑/↓ or j/k", Description: "Move cursor"},
		{Key: "←/→ or h/l", Description: "Switch tab (behavior config)"},
		{Key: "Tab", Description: "Next computation"},
	})
	help.AddSection("Selection", []components.HelpItem{
		{Key: "Space", Description: "Toggle selection"},
		{Key: "A", Description: "Select all"},
		{Key: "N", Description: "Select none"},
		{Key: "/", Description: "Filter subjects"},
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Proceed to next step"},
		{Key: "?", Description: "Toggle help"},
		{Key: "R", Description: "Refresh subjects"},
	})
	help.AddSection("General", []components.HelpItem{
		{Key: "Esc", Description: "Go back / Cancel"},
	})

	m := Model{
		Pipeline:            pipeline,
		selected:            make(map[int]bool),
		subjectSelected:     make(map[string]bool),
		computationSelected: make(map[int]bool),
		bands:               append([]FrequencyBand{}, frequencyBands...),
		bandSelected:        make(map[int]bool),
		editingBandIdx:      -1,
		rois:                append([]ROIDefinition{}, defaultROIs...),
		roiSelected:         make(map[int]bool),
		editingROIIdx:       -1,
		spatialSelected:     make(map[int]bool),
		helpOverlay:         help,
		// Advanced config defaults (shared)
		useDefaultAdvanced:                false,
		expandedOption:                    expandedNone,
		connectivityMeasures:              make(map[int]bool),
		featGroupConnectivityExpanded:     false,
		featGroupPACExpanded:              false,
		featGroupAperiodicExpanded:        false,
		featGroupComplexityExpanded:       false,
		featGroupBurstsExpanded:           false,
		featGroupPowerExpanded:            false,
		featGroupSpectralExpanded:         false,
		featGroupERPExpanded:              false,
		featGroupRatiosExpanded:           false,
		featGroupAsymmetryExpanded:        false,
		featGroupSpatialTransformExpanded: false,
		featGroupStorageExpanded:          true,
		featGroupExecutionExpanded:        true,
		plotItemConfigs:                   make(map[string]PlotItemConfig),
		plotItemConfigExpanded:            make(map[string]bool),
		// PAC/CFC defaults (from config)
		pacPhaseMin:  4.0,
		pacPhaseMax:  8.0,
		pacAmpMin:    30.0,
		pacAmpMax:    80.0,
		pacMethod:    0,
		pacMinEpochs: 2,
		pacPairsSpec: "theta:gamma,alpha:gamma",
		// Aperiodic defaults
		aperiodicFmin:      1.0,
		aperiodicFmax:      80.0,
		aperiodicPeakZ:     3.5,
		aperiodicMinR2:     0.6,
		aperiodicMinPoints: 5,
		// Complexity defaults
		complexityPEOrder: 3,
		complexityPEDelay: 1,
		// ERP defaults
		erpBaselineCorrection: true,
		erpAllowNoBaseline:    false,
		erpComponentsSpec:     "n1=0.10-0.20,n2=0.20-0.35,p2=0.35-0.50",
		erpSmoothMs:           0.0,
		erpPeakProminenceUv:   0.0, // 0 = use default from config
		erpLowpassHz:          30.0,
		// Burst defaults
		burstThresholdZ:            2.0,
		burstThresholdMethod:       0, // 0: percentile
		burstThresholdPercentile:   95.0,
		burstThresholdReference:    0, // 0: trial
		burstMinTrialsPerCondition: 10,
		burstMinSegmentSec:         2.0,
		burstSkipInvalidSegments:   true,
		burstMinDuration:           50,
		burstMinCycles:             3.0,
		burstBandsSpec:             "beta,gamma",
		// Power defaults
		powerBaselineMode:          0,
		powerRequireBaseline:       true,
		powerSubtractEvoked:        true,
		powerMinTrialsPerCondition: 2,
		powerExcludeLineNoise:      true,
		powerLineNoiseFreq:         60.0,
		powerLineNoiseWidthHz:      1.0,
		powerLineNoiseHarmonics:    3,
		powerEmitDb:                true,
		// Spectral defaults
		spectralEdgePercentile:     0.95,
		spectralRatioPairsSpec:     "theta:beta,theta:alpha,alpha:beta,delta:alpha,delta:theta",
		spectralPsdAdaptive:        false,
		spectralMultitaperAdaptive: false,
		// Connectivity defaults
		connOutputLevel:  0,
		connGraphMetrics: false,
		connGraphProp:    0.1,
		connWindowLen:    1.0,
		connWindowStep:   0.5,
		connAECMode:      0,
		// Scientific validity defaults (new)
		itpcMethod:                1,    // 1: fold_global (CV-safe default)
		itpcConditionColumn:       "",   // Empty = not using condition-based ITPC
		itpcConditionValues:       "",   // Empty = all unique values
		itpcMinTrialsPerCondition: 10,   // Minimum trials per condition
		aperiodicMinSegmentSec:    2.0,  // 2.0s minimum for stable fits
		connAECOutput:             0,    // 0: r only (raw)
		connForceWithinEpochML:    true, // Force within_epoch for CV-safety
		ratioSource:               0,    // 0: raw (default)

		// ITPC additional defaults
		itpcAllowUnsafeLoo:     false,
		itpcBaselineCorrection: 0,  // 0: none
		itpcNJobs:              -1, // -1 = all CPUs

		// Spectral advanced defaults
		spectralIncludeLogRatios:   true,
		spectralPsdMethod:          0, // 0: multitaper
		spectralFmin:               1.0,
		spectralFmax:               80.0,
		spectralExcludeLineNoise:   true,
		spectralLineNoiseFreq:      60.0,
		spectralLineNoiseWidthHz:   1.0,
		spectralLineNoiseHarmonics: 3,
		spectralMinSegmentSec:      2.0,
		spectralMinCyclesAtFmin:    3.0,

		// Band envelope defaults
		bandEnvelopePadSec:    0.5,
		bandEnvelopePadCycles: 3.0,

		// IAF defaults
		iafEnabled:                  false,
		iafAlphaWidthHz:             2.0,
		iafSearchRangeMin:           7.0,
		iafSearchRangeMax:           13.0,
		iafMinProminence:            0.05,
		iafRoisSpec:                 "ParOccipital_Midline,ParOccipital_Left,ParOccipital_Right",
		iafMinCyclesAtFmin:          5.0,
		iafMinBaselineSec:           0.0,
		iafAllowFullFallback:        false,
		iafAllowAllChannelsFallback: false,

		// Aperiodic advanced defaults
		aperiodicModel:              0,   // 0: fixed
		aperiodicPsdMethod:          0,   // 0: multitaper
		aperiodicPsdBandwidth:       0.0, // 0 = use default
		aperiodicMaxRms:             0.0, // 0 = no limit
		aperiodicExcludeLineNoise:   true,
		aperiodicLineNoiseFreq:      60.0,
		aperiodicLineNoiseWidthHz:   1.0,
		aperiodicLineNoiseHarmonics: 3,

		// Spatial transform defaults
		spatialTransform:          0,    // 0: none
		spatialTransformLambda2:   1e-5, // Default lambda2
		spatialTransformStiffness: 4.0,  // Default stiffness

		// Connectivity advanced defaults
		connGranularity:            0, // 0: trial
		connConditionColumn:        "",
		connConditionValues:        "",
		connMinEpochsPerGroup:      5,
		connMinCyclesPerBand:       3.0,
		connWarnNoSpatialTransform: true,
		connPhaseEstimator:         0, // 0: within_epoch
		connMinSegmentSec:          1.0,

		// Directed connectivity defaults
		directedConnMeasures:          make(map[int]bool),
		directedConnEnabled:           false, // Disabled by default (opt-in)
		directedConnOutputLevel:       0,     // 0: full
		directedConnMvarOrder:         10,    // MVAR model order
		directedConnNFreqs:            16,    // Number of frequency bins
		directedConnMinSegSamples:     100,   // Minimum segment samples
		featGroupDirectedConnExpanded: false,

		// Source localization defaults
		sourceLocEnabled:           false, // Disabled by default (opt-in, requires fsaverage)
		sourceLocMethod:            0,     // 0: lcmv
		sourceLocSpacing:           1,     // 1: oct6 (default)
		sourceLocParc:              0,     // 0: aparc (Desikan-Killiany)
		sourceLocReg:               0.05,  // LCMV regularization
		sourceLocSnr:               3.0,   // eLORETA SNR
		sourceLocLoose:             0.2,   // eLORETA loose constraint
		sourceLocDepth:             0.8,   // eLORETA depth weighting
		sourceLocConnMethod:        0,     // 0: aec
		sourceLocSubject:           "",
		sourceLocTrans:             "",
		sourceLocBem:               "",
		sourceLocMindistMm:         5.0,
		sourceLocFmriEnabled:       false,
		sourceLocFmriStatsMap:      "",
		sourceLocFmriProvenance:    0,
		sourceLocFmriRequireProv:   true,
		sourceLocFmriThreshold:     3.1,
		sourceLocFmriTail:          0, // 0: pos
		sourceLocFmriMinClusterVox: 50,
		sourceLocFmriMinClusterMM3: 400.0,
		sourceLocFmriMaxClusters:   20,
		sourceLocFmriMaxVoxPerClus: 2000,
		sourceLocFmriMaxTotalVox:   20000,
		sourceLocFmriRandomSeed:    0,

		// fMRI GLM contrast builder defaults
		sourceLocFmriContrastEnabled:   false,
		sourceLocFmriContrastType:      0, // 0: t-test
		sourceLocFmriCondAColumn:       "trial_type",
		sourceLocFmriCondAValue:        "",
		sourceLocFmriCondBColumn:       "trial_type",
		sourceLocFmriCondBValue:        "",
		sourceLocFmriContrastFormula:   "",
		sourceLocFmriContrastName:      "pain_vs_baseline",
		sourceLocFmriRunsToInclude:     "",
		sourceLocFmriAutoDetectRuns:    true,
		sourceLocFmriHrfModel:          0,     // 0: SPM
		sourceLocFmriDriftModel:        1,     // 1: cosine
		sourceLocFmriHighPassHz:        0.008, // 128s period
		sourceLocFmriLowPassHz:         0.0,
		sourceLocFmriStimPhasesToModel: "", // auto (plateau-only default when present)
		sourceLocFmriClusterCorrection: true,
		sourceLocFmriClusterPThreshold: 0.001,
		sourceLocFmriOutputType:        0, // 0: z-score
		sourceLocFmriResampleToFS:      true,
		sourceLocFmriWindowAName:       "plateau",
		sourceLocFmriWindowATmin:       5.0,
		sourceLocFmriWindowATmax:       10.0,
		sourceLocFmriWindowBName:       "baseline",
		sourceLocFmriWindowBTmin:       -2.0,
		sourceLocFmriWindowBTmax:       0.0,

		// Source localization UI expansion states
		featGroupSourceLocExpanded:         false,
		featGroupSourceLocFmriExpanded:     false,
		featGroupSourceLocContrastExpanded: false,
		featGroupSourceLocGLMExpanded:      false,
		featGroupITPCExpanded:              false,

		// PAC advanced defaults
		pacSource:              0, // 0: precomputed
		pacNormalize:           true,
		pacNSurrogates:         0,
		pacAllowHarmonicOvrlap: false,
		pacMaxHarmonic:         6,
		pacHarmonicToleranceHz: 1.0,
		pacRandomSeed:          0,
		pacComputeWaveformQC:   true,
		pacWaveformOffsetMs:    5.0,

		// Complexity advanced defaults
		complexitySignalBasis:   0,
		complexityMinSegmentSec: 2.0,
		complexityMinSamples:    200,
		complexityZscore:        true,

		// Quality defaults
		qualityPsdMethod:        0, // 0: welch
		qualityFmin:             1.0,
		qualityFmax:             100.0,
		qualityNfft:             256,
		qualityExcludeLineNoise: true,
		qualitySnrSignalBandMin: 1.0,
		qualitySnrSignalBandMax: 30.0,
		qualitySnrNoiseBandMin:  40.0,
		qualitySnrNoiseBandMax:  80.0,
		qualityMuscleBandMin:    30.0,
		qualityMuscleBandMax:    80.0,

		// ERDS defaults
		erdsUseLogRatio:      false,
		erdsMinBaselinePower: 1.0e-12,
		erdsMinActivePower:   1.0e-12,
		erdsMinSegmentSec:    0.5,
		erdsBandsSpec:        "alpha,beta",

		// Temporal feature selection defaults
		temporalFeaturePowerEnabled: true,
		temporalFeatureITPCEnabled:  false,
		temporalFeatureERDSEnabled:  false,

		// Time-frequency heatmap defaults
		tfHeatmapEnabled:   true,
		tfHeatmapFreqsSpec: "4,8,13,30,45",
		tfHeatmapTimeResMs: 100,

		// TFR advanced defaults
		tfrMaxCycles:  15.0,
		tfrDecimPower: 4,
		tfrDecimPhase: 1,

		// Validation & Generic
		minEpochsForFeatures:     10,
		featAnalysisMode:         0,
		featComputeChangeScores:  true,
		featSaveTfrWithSidecar:   false,
		featNJobsBands:           -1,
		featNJobsConnectivity:    -1,
		featNJobsAperiodic:       -1,
		featNJobsComplexity:      -1,
		featAlsoSaveCsv:          false,
		saveSubjectLevelFeatures: true,

		// Asymmetry defaults
		asymmetryChannelPairsSpec:         "F3:F4,F7:F8,C3:C4,P3:P4,O1:O2",
		asymmetryMinSegmentSec:            1.0,
		asymmetryMinCyclesAtFmin:          3.0,
		asymmetrySkipInvalidSegments:      true,
		asymmetryEmitActivationConvention: false,
		asymmetryActivationBandsSpec:      "alpha",

		// Ratios defaults
		ratiosMinSegmentSec:       1.0,
		ratiosMinCyclesAtFmin:     3.0,
		ratiosSkipInvalidSegments: true,

		// Quality line noise defaults
		qualityLineNoiseFreq:      60.0,
		qualityLineNoiseWidthHz:   1.0,
		qualityLineNoiseHarmonics: 3,
		// Behavior defaults
		correlationMethod:     "spearman",
		robustCorrelation:     0,
		bootstrapSamples:      1000,
		nPermutations:         1000,
		rngSeed:               0,
		controlTemperature:    true,
		controlTrialOrder:     true,
		fdrAlpha:              0.05,
		behaviorConfigSection: 0,
		behaviorNJobs:         -1,

		behaviorComputeChangeScores:        true,
		behaviorComputeBayesFactors:        false,
		behaviorComputeLosoStability:       true,
		behaviorOverwrite:                  true, // Default: overwrite existing outputs
		runAdjustmentEnabled:               false,
		runAdjustmentColumn:                "run_id",
		runAdjustmentIncludeInCorrelations: true,
		runAdjustmentMaxDummies:            20,

		trialTableFormat:             1,
		trialTableAddLagFeatures:     true,
		trialOrderMaxMissingFraction: 0.1,

		featureSummariesEnabled:         true,
		featureQCEnabled:                false,
		featureQCMaxMissingPct:          0.2,
		featureQCMinVariance:            1e-10,
		featureQCCheckWithinRunVariance: true,

		painResidualEnabled:                 true,
		painResidualMethod:                  0,
		painResidualPolyDegree:              2,
		painResidualSplineDfCandidates:      "3,4,5",
		painResidualModelCompareEnabled:     true,
		painResidualModelComparePolyDegrees: "2,3",
		painResidualBreakpointEnabled:       true,
		painResidualBreakpointCandidates:    15,
		painResidualBreakpointQlow:          0.15,
		painResidualBreakpointQhigh:         0.85,
		painResidualCrossfitEnabled:         false,
		painResidualCrossfitGroupColumn:     "",
		painResidualCrossfitNSplits:         5,
		painResidualCrossfitMethod:          0,
		painResidualCrossfitSplineKnots:     5,

		regressionOutcome:            0,
		regressionIncludeTemperature: true,
		regressionTempControl:        0,
		regressionTempSplineKnots:    4,
		regressionTempSplineQlow:     0.05,
		regressionTempSplineQhigh:    0.95,
		regressionIncludeTrialOrder:  true,
		regressionIncludePrev:        false,
		regressionIncludeRunBlock:    true,
		regressionIncludeInteraction: true,
		regressionStandardize:        true,
		regressionPermutations:       0,
		regressionMaxFeatures:        0,

		modelsIncludeTemperature:  true,
		modelsTempControl:         0,
		modelsTempSplineKnots:     4,
		modelsTempSplineQlow:      0.05,
		modelsTempSplineQhigh:     0.95,
		modelsIncludeTrialOrder:   true,
		modelsIncludePrev:         false,
		modelsIncludeRunBlock:     true,
		modelsIncludeInteraction:  true,
		modelsStandardize:         true,
		modelsMaxFeatures:         100,
		modelsOutcomeRating:       true,
		modelsOutcomePainResidual: true,
		modelsOutcomeTemperature:  false,
		modelsOutcomePainBinary:   false,
		modelsFamilyOLS:           true,
		modelsFamilyRobust:        true,
		modelsFamilyQuantile:      true,
		modelsFamilyLogit:         true,
		modelsBinaryOutcome:       0,

		stabilityMethod:      0,
		stabilityOutcome:     0,
		stabilityGroupColumn: 0,
		stabilityPartialTemp: true,
		stabilityMaxFeatures: 50,
		stabilityAlpha:       0.05,

		consistencyEnabled:           true,
		influenceOutcomeRating:       true,
		influenceOutcomePainResidual: true,
		influenceOutcomeTemperature:  false,
		influenceMaxFeatures:         20,
		influenceIncludeTemperature:  true,
		influenceTempControl:         0,
		influenceTempSplineKnots:     4,
		influenceTempSplineQlow:      0.05,
		influenceTempSplineQhigh:     0.95,
		influenceIncludeTrialOrder:   true,
		influenceIncludeRunBlock:     true,
		influenceIncludeInteraction:  false,
		influenceStandardize:         true,
		influenceCooksThreshold:      0.0,
		influenceLeverageThreshold:   0.0,

		correlationsTargetRating:        true,
		correlationsTargetTemperature:   true,
		correlationsTargetPainResidual:  true,
		correlationsTypesSpec:           "partial_cov_temp",
		correlationsPreferPainResidual:  true,
		correlationsUseCrossfitResidual: false,
		correlationsPrimaryUnit:         0,
		correlationsPermutationPrimary:  false,
		groupLevelBlockPermutation:      true,

		reportTopN:                     15,
		temporalResolutionMs:           50,
		temporalSmoothMs:               100,
		temporalTimeMinMs:              -200,
		temporalTimeMaxMs:              1000,
		temporalTargetColumn:           "",
		temporalSplitByCondition:       true,
		temporalConditionColumn:        "",
		temporalConditionValues:        "",
		temporalIncludeROIAverages:     true,
		temporalIncludeTFGrid:          true,
		temporalITPCBaselineCorrection: true,
		temporalITPCBaselineMin:        -0.5,
		temporalITPCBaselineMax:        -0.01,
		// ERDS defaults
		temporalERDSBaselineMin: -0.5,
		temporalERDSBaselineMax: -0.1,
		temporalERDSMethod:      0, // 0=percent, 1=zscore
		mixedEffectsType:        0,
		mediationMinEffect:      0.05,
		// Cluster defaults
		clusterThreshold:       0.05,
		clusterMinSize:         2,
		clusterTail:            0,
		clusterConditionColumn: "",
		clusterConditionValues: "",
		// Mediation defaults
		mediationBootstrap:           1000,
		mediationMaxMediatorsEnabled: true,
		mediationMaxMediators:        20,
		mediationPermutations:        0, // Disabled by default
		// Moderation defaults
		moderationMaxFeaturesEnabled: true,
		moderationMaxFeatures:        50,
		moderationPermutations:       0, // Disabled by default
		// Mixed effects defaults
		mixedMaxFeatures: 50,
		// Condition defaults
		conditionEffectThreshold:    0.5,
		conditionFailFast:           true,
		conditionPermutationPrimary: false,
		conditionWindowPrimaryUnit:  0,
		// Column discovery defaults
		discoveredColumns:              []string{},
		trialTableColumns:              []string{},
		conditionEffectsColumns:        []string{},
		conditionEffectsColumnValues:   make(map[string][]string),
		conditionEffectsWindows:        []string{},
		conditionEffectsDiscoveryDone:  false,
		conditionEffectsDiscoveryError: "",
		discoveredColumnValues:         make(map[string][]string),
		trialTableColumnValues:         make(map[string][]string),
		selectedValueCursors:           make(map[string]int),
		availableWindowsByFeature:      make(map[string][]string),
		// Machine Learning defaults
		mlNPerm:                     0,
		innerSplits:                 3,
		outerJobs:                   1,
		mlScope:                     MLCVScopeGroup,
		mlTarget:                    "",
		mlFmriSigGroupExpanded:      true,
		mlFmriSigMethodIndex:        0,
		mlFmriSigContrastName:       "pain_vs_nonpain",
		mlFmriSigSignatureIndex:     0,
		mlFmriSigMetricIndex:        0,
		mlFmriSigNormalizationIndex: 0,
		mlFmriSigRoundDecimals:      3,
		mlBinaryThresholdEnabled:    false,
		mlBinaryThreshold:           0.0,
		mlFeatureFamiliesSpec:       "",
		mlFeatureBandsSpec:          "",
		mlFeatureSegmentsSpec:       "",
		mlFeatureScopesSpec:         "",
		mlFeatureStatsSpec:          "",
		mlFeatureHarmonization:      MLFeatureHarmonizationDefault,
		mlCovariatesSpec:            "",
		mlBaselinePredictorsSpec:    "",
		mlRegressionModel:           MLRegressionElasticNet,
		mlClassificationModel:       MLClassificationDefault,
		mlRequireTrialMlSafe:        false,
		mlUncertaintyAlpha:          0.1,
		mlPermNRepeats:              10,
		// Hyperparameter defaults mirror eeg_pipeline/utils/config/eeg_config.yaml
		elasticNetAlphaGrid:   "0.001,0.01,0.1,1,10",
		elasticNetL1RatioGrid: "0.2,0.5,0.8",
		varianceThresholdGrid: "0.0,0.01,0.1",
		ridgeAlphaGrid:        "0.01,0.1,1,10,100",
		rfNEstimators:         500,
		rfMaxDepthGrid:        "5,10,20,null",
		// TFR defaults (from config)
		tfrFreqMin:       1.0,
		tfrFreqMax:       100.0,
		tfrNFreqs:        40,
		tfrMinCycles:     3.0,
		tfrNCyclesFactor: 2.0,
		tfrWorkers:       -1,
		// System defaults
		systemNJobs:      -1,
		systemStrictMode: true,
		loggingLevel:     1, // INFO
		// ICA defaults
		icaLabelsToKeep:        "brain,other",
		plotSelected:           make(map[int]bool),
		featurePlotterSelected: make(map[string]bool),
		plottingScope:          PlottingScopeSubject,
		plotFormats:            []string{"png", "svg", "pdf"},
		plotFormatSelected: map[string]bool{
			"png": true,
			"svg": true,
		},
		plotDpiOptions:      []int{150, 300, 600},
		plotDpiIndex:        1,
		plotSavefigDpiIndex: 2,

		// Preprocessing defaults
		prepUsePyprep:    true,
		prepUseIcalabel:  true,
		prepNJobs:        -1,
		prepMontage:      "easycap-M1",
		prepResample:     500,
		prepLFreq:        0.1,
		prepHFreq:        100.0,
		prepNotch:        60,
		prepLineFreq:     60,
		prepChTypes:      "eeg",
		prepEegReference: "average",
		prepEogChannels:  "",
		prepRandomState:  42,
		prepTaskIsRest:   false,
		prepZaplineFline: 60.0,
		prepFindBreaks:   true,
		// PyPREP advanced defaults
		prepRansac:               true,
		prepRepeats:              3,
		prepAverageReref:         false,
		prepFileExtension:        ".vhdr",
		prepConsiderPreviousBads: true,
		prepOverwriteChansTsv:    true,
		prepDeleteBreaks:         false,
		prepBreaksMinLength:      20,
		prepTStartAfterPrevious:  2,
		prepTStopBeforeNext:      2,
		prepRenameAnotDict:       "",
		prepCustomBadDict:        "",
		// ICA defaults
		prepSpatialFilter:   0,
		prepICAAlgorithm:    0,
		prepICAComp:         0.99,
		prepICALFreq:        1.0,
		prepICARejThresh:    500.0,
		prepProbThresh:      0.8,
		prepKeepMnebidsBads: false,
		// Epoching defaults
		prepConditions:           "",
		prepEpochsTmin:           -5.0,
		prepEpochsTmax:           15.0,
		prepEpochsBaselineStart:  -2.0,
		prepEpochsBaselineEnd:    0.0,
		prepEpochsNoBaseline:     false,
		prepEpochsReject:         0.0,
		prepRejectMethod:         1,
		prepRunSourceEstimation:  false,
		prepWriteCleanEvents:     true,
		prepOverwriteCleanEvents: true,
		prepCleanEventsStrict:    true,

		// Preprocessing group expansion defaults (all collapsed for compact view)
		prepGroupStagesExpanded:    false,
		prepGroupGeneralExpanded:   true, // General is expanded by default (most common options)
		prepGroupFilteringExpanded: false,
		prepGroupPyprepExpanded:    false,
		prepGroupICAExpanded:       false,
		prepGroupEpochingExpanded:  false,

		// Utilities defaults
		rawMontage:           "easycap-M1",
		rawLineFreq:          60,
		rawOverwrite:         false,
		rawTrimToFirstVolume: true,
		rawEventPrefixes:     "",
		rawKeepAnnotations:   false,
		mergeEventPrefixes:   "",
		mergeEventTypes:      "",
		// fMRI raw-to-bids defaults
		fmriRawSession:          "",
		fmriRawRestTask:         "rest",
		fmriRawIncludeRest:      true,
		fmriRawIncludeFieldmaps: true,
		fmriRawDicomModeIndex:   0, // symlink
		fmriRawOverwrite:        false,
		fmriRawCreateEvents:     true,
		fmriRawEventGranularity: 0, // phases
		fmriRawOnsetRefIndex:    1, // first_iti_start (recommended for simultaneous EEG-fMRI)
		fmriRawOnsetOffsetS:     0.0,
		fmriRawDcm2niixPath:     "",
		fmriRawDcm2niixArgs:     "",
	}

	// Align default measure selections with eeg_config.yaml.
	// - feature_engineering.connectivity.measures: ["wpli", "aec"]
	// - feature_engineering.directedconnectivity: enable_psi=true, enable_dtf=false, enable_pdc=false
	for i, measure := range connectivityMeasures {
		if measure.Key == "wpli" || measure.Key == "aec" {
			m.connectivityMeasures[i] = true
		}
	}
	for i, measure := range directedConnectivityMeasures {
		if measure.Key == "psi" {
			m.directedConnMeasures[i] = true
		}
	}

	// Time ranges
	m.TimeRanges = []types.TimeRange{}
	m.editingRangeIdx = noRangeEditing

	switch pipeline {
	case types.PipelineFeatures:
		m.modeOptions = []string{styles.ModeCompute}
		m.modeDescriptions = []string{
			"Extract EEG feature sets",
		}
		m.categories = []string{
			"power", "spectral", "aperiodic", "erp", "erds", "ratios", "asymmetry",
			"connectivity", "directedconnectivity", "itpc", "pac",
			"complexity", "bursts", "quality", "sourcelocalization",
		}
		m.categoryDescs = []string{
			"Band power (log-ratio)",
			"Peak frequency, IAF",
			"1/f spectral slope",
			"ERP/LEP time-domain features",
			"Event-related desync/sync",
			"Band power ratios",
			"Hemispheric asymmetry",
			"Functional connectivity",
			"Directed connectivity (PSI, DTF, PDC)",
			"Inter-trial phase coh.",
			"Phase-amplitude coupling",
			"Signal complexity",
			"Oscillatory burst dynamics",
			"Trial quality metrics",
			"Source localization (LCMV, eLORETA)",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepConfigureOptions, // Category selection
			types.StepSelectBands,
			types.StepSelectROIs,
			types.StepSelectSpatial,
			types.StepTimeRange,
			types.StepAdvancedConfig,
		}
		for i := range frequencyBands {
			m.bandSelected[i] = true
		}
		for i := range defaultROIs {
			m.roiSelected[i] = true
		}
		// Default spatial modes: roi and global
		m.spatialSelected[0] = true // roi
		m.spatialSelected[2] = true // global

	case types.PipelineBehavior:
		m.modeOptions = []string{styles.ModeCompute}
		m.modeDescriptions = []string{
			"Compute EEG-behavior correlations",
		}
		m.computations = behaviorComputations

		// Default selections organized by analysis purpose
		defaultComps := map[string]bool{
			// Data Preparation
			"trial_table":   false,
			"lag_features":  false,
			"pain_residual": false,

			// Core Analyses
			"correlations":            false,
			"multilevel_correlations": false,
			"regression":              false,
			"condition":               false,
			"temporal":                false,
			"pain_sensitivity":        false,
			"cluster":                 false,

			// Advanced/Causal Analyses
			"mediation":     false,
			"moderation":    false,
			"mixed_effects": false,

			// Quality & Validation
			"stability":  false,
			"validation": false,
			"report":     false,
		}
		for i, c := range behaviorComputations {
			m.computationSelected[i] = defaultComps[c.Key]
		}

		// Initialize feature file selection (all selected by default)
		m.featureFiles = featureFileOptions
		m.featureFileSelected = make(map[string]bool)
		for _, file := range featureFileOptions {
			m.featureFileSelected[file.Key] = true
		}

		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectComputations,
			types.StepSelectFeatureFiles,
			types.StepAdvancedConfig,
		}

	case types.PipelineML:
		m.modeOptions = []string{
			"regression",
			"timegen",
			"classify",
			"model_comparison",
			"incremental_validity",
			"uncertainty",
			"shap",
			"permutation",
		}
		m.modeDescriptions = []string{
			"LOSO regression",
			"Time generalization",
			"Binary classification",
			"Compare ElasticNet vs Ridge vs RF",
			"Δ performance from EEG over baseline",
			"Conformal prediction intervals",
			"SHAP feature importance",
			"Permutation feature importance",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectMode,
			types.StepAdvancedConfig,
		}

	case types.PipelinePreprocessing:
		m.modeOptions = []string{"full", "partial"}
		m.modeDescriptions = []string{
			"Run all preprocessing stages",
			"Select specific preprocessing stages",
		}
		m.prepStages = preprocessingStages
		m.prepStageSelected = make(map[int]bool)
		// Default: select all stages
		for i := range preprocessingStages {
			m.prepStageSelected[i] = true
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectMode,
			types.StepAdvancedConfig,
		}

	case types.PipelineFmri:
		m.modeOptions = []string{"preprocess"}
		m.modeDescriptions = []string{
			"Run fMRIPrep-style preprocessing (containerized)",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
		}

		// Defaults mirror fmri_pipeline/utils/config/fmri_config.yaml (fmri_preprocessing.*)
		m.fmriEngineIndex = 0 // docker
		m.fmriFmriprepImage = "nipreps/fmriprep:25.2.4"
		m.fmriFmriprepOutputDir = ""
		m.fmriFmriprepWorkDir = ""
		m.fmriFreesurferLicenseFile = ""
		m.fmriFreesurferSubjectsDir = ""
		m.fmriOutputSpacesSpec = "MNI152NLin2009cAsym T1w"
		m.fmriIgnoreSpec = ""
		m.fmriBidsFilterFile = ""
		m.fmriExtraArgs = ""
		m.fmriUseAroma = false
		m.fmriSkipBidsValidation = false
		m.fmriStopOnFirstCrash = false
		m.fmriCleanWorkdir = true
		m.fmriSkipReconstruction = false
		m.fmriMemMb = 0 // 0 = fMRIPrep default
		// Additional fMRIPrep options defaults
		m.fmriNThreads = 0            // 0 = fMRIPrep default (all available)
		m.fmriOmpNThreads = 0         // 0 = fMRIPrep default
		m.fmriLowMem = false          // standard memory usage
		m.fmriLongitudinal = false    // single-session processing
		m.fmriCiftiOutputIndex = 0    // disabled
		m.fmriSkullStripTemplate = "" // default: OASIS30ANTs
		m.fmriSkullStripFixedSeed = false
		m.fmriRandomSeed = 0         // 0 = non-deterministic
		m.fmriDummyScans = 0         // 0 = auto-detect from metadata
		m.fmriBold2T1wInitIndex = 0  // register (default)
		m.fmriBold2T1wDof = 6        // 6 DOF rigid-body (default)
		m.fmriSliceTimeRef = 0.5     // middle of acquisition (default)
		m.fmriFdSpikeThreshold = 0.5 // mm (default)
		m.fmriDvarsSpikeThreshold = 1.5
		m.fmriMeOutputEchos = false
		m.fmriMedialSurfaceNan = false
		m.fmriNoMsm = false
		m.fmriLevelIndex = 0 // full (default)
		m.fmriTaskId = ""    // all tasks

		// fMRI group expansion defaults (collapsed for compact view, Runtime expanded by default)
		m.fmriGroupRuntimeExpanded = true
		m.fmriGroupOutputExpanded = false
		m.fmriGroupPerformanceExpanded = false
		m.fmriGroupAnatomicalExpanded = false
		m.fmriGroupBoldExpanded = false
		m.fmriGroupQcExpanded = false
		m.fmriGroupDenoisingExpanded = false
		m.fmriGroupSurfaceExpanded = false
		m.fmriGroupMultiechoExpanded = false
		m.fmriGroupReproExpanded = false
		m.fmriGroupValidationExpanded = false
		m.fmriGroupAdvancedExpanded = false

	case types.PipelineFmriAnalysis:
		m.modeOptions = []string{"first-level", "trial-signatures"}
		m.modeDescriptions = []string{
			"First-level GLM contrasts (per subject)",
			"Trial-wise betas + NPS/SIIPS1 readouts (beta-series or LSS)",
		}
		m.steps = []types.WizardStep{
			types.StepSelectMode,
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
		}

		// Defaults match fmri_pipeline/cli/commands/fmri_analysis.py
		m.fmriAnalysisInputSourceIndex = 0 // fmriprep
		m.fmriAnalysisFmriprepSpace = "T1w"
		m.fmriAnalysisRequireFmriprep = true
		m.fmriAnalysisRunsSpec = ""    // auto-detect
		m.fmriAnalysisContrastType = 0 // t-test
		m.fmriAnalysisCondAColumn = "trial_type"
		m.fmriAnalysisCondAValue = ""
		m.fmriAnalysisCondBColumn = "trial_type"
		m.fmriAnalysisCondBValue = ""
		m.fmriAnalysisContrastName = "pain_vs_nonpain"
		m.fmriAnalysisFormula = ""
		m.fmriAnalysisEventsToModel = ""
		m.fmriAnalysisStimPhasesToModel = "" // auto (plateau-only default when present)
		m.fmriAnalysisHrfModel = 0           // spm
		m.fmriAnalysisDriftModel = 1         // cosine
		m.fmriAnalysisHighPassHz = 0.008
		m.fmriAnalysisLowPassHz = 0.0
		m.fmriAnalysisSmoothingFwhm = 5.0
		m.fmriAnalysisOutputType = 0 // z-score
		m.fmriAnalysisOutputDir = ""
		m.fmriAnalysisResampleToFS = false
		m.fmriAnalysisFreesurferDir = ""
		m.fmriAnalysisConfoundsStrategy = 0 // auto
		m.fmriAnalysisWriteDesignMatrix = false

		m.fmriAnalysisGroupInputExpanded = true
		m.fmriAnalysisGroupContrastExpanded = true
		m.fmriAnalysisGroupGLMExpanded = false
		m.fmriAnalysisGroupConfoundsExpanded = false
		m.fmriAnalysisGroupOutputExpanded = false
		m.fmriAnalysisGroupPlottingExpanded = true

		// Plotting/report defaults (TUI convenience defaults; CLI defaults are "off")
		m.fmriAnalysisPlotsEnabled = true
		m.fmriAnalysisPlotHTML = true
		m.fmriAnalysisPlotSpaceIndex = 0         // both
		m.fmriAnalysisPlotThresholdModeIndex = 0 // z
		m.fmriAnalysisPlotZThreshold = 2.3
		m.fmriAnalysisPlotFdrQ = 0.05
		m.fmriAnalysisPlotClusterMinVoxels = 50
		m.fmriAnalysisPlotVmaxModeIndex = 0 // per-space robust
		m.fmriAnalysisPlotVmaxManual = 5.0
		m.fmriAnalysisPlotIncludeUnthresholded = true
		m.fmriAnalysisPlotFormatPNG = true
		m.fmriAnalysisPlotFormatSVG = true
		m.fmriAnalysisPlotTypeSlices = true
		m.fmriAnalysisPlotTypeGlass = true
		m.fmriAnalysisPlotTypeHist = true
		m.fmriAnalysisPlotTypeClusters = true
		m.fmriAnalysisPlotEffectSize = true
		m.fmriAnalysisPlotStandardError = true
		m.fmriAnalysisPlotMotionQC = true
		m.fmriAnalysisPlotCarpetQC = true
		m.fmriAnalysisPlotTSNRQC = true
		m.fmriAnalysisPlotDesignQC = true
		m.fmriAnalysisPlotEmbedImages = true
		m.fmriAnalysisPlotSignatures = true
		m.fmriAnalysisSignatureDir = ""

		// Trial-wise signature defaults (used by beta-series / lss modes)
		m.fmriTrialSigGroupExpanded = true
		m.fmriTrialSigMethodIndex = 0 // beta-series
		m.fmriTrialSigIncludeOtherEvents = true
		m.fmriTrialSigMaxTrialsPerRun = 0
		m.fmriTrialSigFixedEffectsWeighting = 0 // variance
		m.fmriTrialSigWriteTrialBetas = false
		m.fmriTrialSigWriteTrialVariances = false
		m.fmriTrialSigWriteConditionBetas = true
			m.fmriTrialSigSignatureNPS = true
			m.fmriTrialSigSignatureSIIPS1 = true
			m.fmriTrialSigLssOtherRegressorsIndex = 0 // per-condition
			m.fmriTrialSigGroupColumn = ""
			m.fmriTrialSigGroupValuesSpec = ""
			m.fmriTrialSigGroupScopeIndex = 0  // across-runs (average)
			m.fmriTrialSigScopeStimPhases = "" // auto (plateau-only default when present)

		// Backward-compat: older configs used modeIndex 2=lss.
		// New mode options are 0=first-level, 1=trial-signatures, with method selected separately.
		if m.modeIndex == 2 {
			m.modeIndex = 1
			m.fmriTrialSigMethodIndex = 1 // lss
		}
		if m.modeIndex < 0 || m.modeIndex >= len(m.modeOptions) {
			m.modeIndex = 0
		}

	case types.PipelineMergePsychoPyData:
		m.modeOptions = []string{"merge-psychopy"}
		m.modeDescriptions = []string{"Merge PsychoPy TrialSummary into BIDS events.tsv"}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
		}

	case types.PipelineRawToBIDS:
		m.modeOptions = []string{"raw-to-bids"}
		m.modeDescriptions = []string{"Convert raw EEG data to BIDS format"}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
		}

	case types.PipelineFmriRawToBIDS:
		m.modeOptions = []string{"fmri-raw-to-bids"}
		m.modeDescriptions = []string{"Convert raw fMRI DICOM series to BIDS format"}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
		}

	case types.PipelinePlotting:
		m.modeOptions = []string{styles.ModeVisualize}
		m.modeDescriptions = []string{"Generate selected visualization suites"}
		plotItems, plotCategories, err := loadPlotCatalog(repoRoot)
		if err != nil || len(plotItems) == 0 || len(plotCategories) == 0 {
			plotItems = defaultPlotItems
			plotCategories = defaultPlotCategories
		}
		m.plotItems = plotItems
		m.plotCategories = plotCategories
		for i := range m.plotItems {
			m.plotSelected[i] = true
		}
		m.plotSharedColorbar = true

		// Initialize categories for plotting
		m.categories = make([]string, len(m.plotCategories))
		m.categoryDescs = make([]string, len(m.plotCategories))
		for i, cat := range m.plotCategories {
			m.categories[i] = cat.Name
			m.categoryDescs[i] = cat.Description
			m.selected[i] = true // Default to all categories
		}

		// Initialize ROI selection for plotting (same as features pipeline)
		for i := range defaultROIs {
			m.roiSelected[i] = true
		}

		// Initialize band selection for plotting (same as features pipeline)
		for i := range frequencyBands {
			m.bandSelected[i] = true
		}

		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectPlotCategories,
			types.StepSelectPlots,
			types.StepSelectFeaturePlotters,
			types.StepSelectBands,
			types.StepSelectROIs,
			types.StepAdvancedConfig,
			types.StepPlotConfig,
		}

	default:
		m.modeOptions = []string{styles.ModeCompute}
		m.modeDescriptions = []string{"Run computation"}
		m.steps = []types.WizardStep{types.StepSelectSubjects}
	}

	if len(m.steps) > 0 {
		m.CurrentStep = m.steps[0]
	}

	// Set repository root for running Python commands
	m.repoRoot = repoRoot

	m.animQueue.Push(animation.CursorBlinkLoop())
	m.subjectLoadingSpinner = components.NewSpinner("Loading subjects...")
	m.plotLoadingSpinner = components.NewSpinner("Loading available feature plots...")
	return m
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

type tickMsg struct{}

func (m Model) Init() tea.Cmd {
	return tea.Batch(m.tick(), m.immediateTick())
}

func (m Model) immediateTick() tea.Cmd {
	return tea.Tick(0, func(t time.Time) tea.Msg { return tickMsg{} })
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*styles.TickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

// CursorBlinkVisible returns true when the cursor should be shown (blink on phase).
func (m Model) CursorBlinkVisible() bool {
	return m.animQueue.CursorVisible()
}

// IsEditing reports whether the wizard is currently in any interactive
// editing mode. When true, global keybindings like quitting should be
// suppressed so that edit-specific handlers can consume the keys.
func (m Model) IsEditing() bool {
	if m.editingText || m.editingNumber {
		return true
	}

	if m.editingBandIdx >= 0 {
		return true
	}

	if m.editingROIIdx >= 0 {
		return true
	}

	if m.editingRangeIdx != noRangeEditing {
		return true
	}

	return false
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
		m.animQueue.Tick()
		m.subjectLoadingSpinner.Tick()
		m.plotLoadingSpinner.Tick()
		m.TickToast()
		return m, m.tick()

	case executor.PickFileMsg:
		// Handle file picker result
		if msg.Error == nil && msg.Path != "" {
			switch msg.Field {
			case "sourceLocTrans":
				m.sourceLocTrans = msg.Path
			case "sourceLocBem":
				m.sourceLocBem = msg.Path
			case "sourceLocFmriStatsMap":
				m.sourceLocFmriStatsMap = msg.Path
			}
		}
		m.browsingField = ""
		return m, nil

	case messages.FmriConditionsDiscoveredMsg:
		// Handle discovered fMRI conditions
		if msg.Error == nil && len(msg.Conditions) > 0 {
			m.sourceLocFmriConditions = msg.Conditions
			// Reset indices if out of bounds
			if m.sourceLocFmriCondIdx1 >= len(msg.Conditions) {
				m.sourceLocFmriCondIdx1 = 0
			}
			if m.sourceLocFmriCondIdx2 >= len(msg.Conditions) {
				m.sourceLocFmriCondIdx2 = 0
			}
			// Set condition strings from indices
			m.sourceLocFmriCondAValue = msg.Conditions[m.sourceLocFmriCondIdx1]
			if len(msg.Conditions) > 1 && m.sourceLocFmriCondIdx2 < len(msg.Conditions) {
				m.sourceLocFmriCondBValue = msg.Conditions[m.sourceLocFmriCondIdx2]
			}

		}
		return m, nil

	case tea.KeyMsg:
		if m.showHelp {
			if msg.String() == "?" || msg.String() == "esc" {
				m.showHelp = false
				m.helpOverlay.Visible = false
			}
			return m, nil
		}

		if m.filteringSubject {
			switch msg.String() {
			case "esc":
				m.filteringSubject = false
				m.subjectFilter = ""
			case "enter":
				m.filteringSubject = false
			case "backspace":
				if hasContent := len(m.subjectFilter) > 0; hasContent {
					m.subjectFilter = m.subjectFilter[:len(m.subjectFilter)-1]
				}
			default:
				if isSingleChar := len(msg.String()) == singleCharLength; isSingleChar {
					m.subjectFilter += msg.String()
				}
			}
			return m, nil
		}

		if m.ConfirmingExecute {
			switch msg.String() {
			case "y", "Y", "enter":
				m.ConfirmingExecute = false
				m.DryRunMode = false
				m.ReadyToExecute = true
				return m, nil
			case "d", "D":
				// Dry-run mode - execute with --dry-run flag
				m.ConfirmingExecute = false
				m.DryRunMode = true
				m.ReadyToExecute = true
				return m, nil
			case "n", "N", "esc":
				m.ConfirmingExecute = false
				return m, nil
			}
			return m, nil
		}

		// Handle text input mode for editable fields
		if m.editingText {
			switch msg.String() {
			case "esc":
				m.editingText = false
				m.textBuffer = ""
				m.editingTextField = textFieldNone
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
			case "enter":
				m.commitTextInput()
				m.editingText = false
				m.textBuffer = ""
				m.editingTextField = textFieldNone
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				return m, nil
			case "backspace":
				if hasContent := len(m.textBuffer) > 0; hasContent {
					m.textBuffer = m.textBuffer[:len(m.textBuffer)-1]
				}
			default:
				if isSingleChar := len(msg.String()) == singleCharLength; isSingleChar {
					m.textBuffer += msg.String()
				}
			}
			return m, nil
		}

		// Handle number input mode for advanced config
		if m.editingNumber {
			switch msg.String() {
			case "esc":
				m.editingNumber = false
				m.numberBuffer = ""
			case "enter":
				m.commitNumberInput()
				m.editingNumber = false
				m.numberBuffer = ""
			case "backspace":
				if hasContent := len(m.numberBuffer) > 0; hasContent {
					m.numberBuffer = m.numberBuffer[:len(m.numberBuffer)-1]
				}
			default:
				// Accept digits, decimal point, and minus sign
				char := msg.String()
				isValidNumericChar := len(char) == singleCharLength &&
					(char >= "0" && char <= "9" || char == "." || char == "-")
				if isValidNumericChar {
					m.numberBuffer += char
				}
			}
			return m, nil
		}

		// Handle band editing mode
		isEditingBand := m.editingBandIdx >= 0 && m.editingBandIdx < len(m.bands)
		if isEditingBand {
			switch msg.String() {
			case "esc":
				m.editingBandIdx = -1
				m.bandEditBuffer = ""
			case "enter":
				m.commitBandEdit()
				// Move to next field or exit
				if m.editingBandField < 2 {
					m.editingBandField++
					m.initBandEditBuffer()
				} else {
					m.editingBandIdx = -1
					m.bandEditBuffer = ""
				}
			case "tab":
				m.commitBandEdit()
				m.editingBandField = (m.editingBandField + 1) % 3
				m.initBandEditBuffer()
			case "backspace":
				if len(m.bandEditBuffer) > 0 {
					m.bandEditBuffer = m.bandEditBuffer[:len(m.bandEditBuffer)-1]
				}
			default:
				char := msg.String()
				if len(char) == singleCharLength {
					if m.editingBandField == 0 {
						// Name field: accept alphanumeric
						m.bandEditBuffer += char
					} else {
						// Frequency fields: accept digits and decimal
						isNumericChar := (char >= "0" && char <= "9") || char == "."
						if isNumericChar {
							m.bandEditBuffer += char
						}
					}
				}
			}
			return m, nil
		}

		// Handle ROI editing mode
		isEditingROI := m.editingROIIdx >= 0 && m.editingROIIdx < len(m.rois)
		if isEditingROI {
			switch msg.String() {
			case "esc":
				m.editingROIIdx = -1
				m.roiEditBuffer = ""
			case "enter":
				m.commitROIEdit()
				// Move to next field or exit
				if m.editingROIField < 1 {
					m.editingROIField++
					m.initROIEditBuffer()
				} else {
					m.editingROIIdx = -1
					m.roiEditBuffer = ""
				}
			case "tab":
				m.commitROIEdit()
				m.editingROIField = (m.editingROIField + 1) % 2
				m.initROIEditBuffer()
			case "backspace":
				if len(m.roiEditBuffer) > 0 {
					m.roiEditBuffer = m.roiEditBuffer[:len(m.roiEditBuffer)-1]
				}
			default:
				char := msg.String()
				if len(char) == singleCharLength {
					// Accept alphanumeric and comma for channels
					m.roiEditBuffer += char
				}
			}
			return m, nil
		}

		// Handle time range input for tmin/tmax
		isEditingRange := m.editingRangeIdx >= 0 && m.editingRangeIdx < len(m.TimeRanges)
		if isEditingRange {
			switch msg.String() {
			case "esc":
				m.editingRangeIdx = noRangeEditing
			case "enter":
				// Commit and move to next field, or exit if at end
				isLastField := m.editingField >= fieldTmax
				if isLastField {
					m.editingRangeIdx = noRangeEditing
					m.editingField = fieldName
				} else {
					m.editingField++
				}
			case "tab":
				// Cycle through fields
				m.editingField = (m.editingField + 1) % numTimeRangeFields
			case "backspace":
				ref := &m.TimeRanges[m.editingRangeIdx]
				isNameField := m.editingField == fieldName
				isTminField := m.editingField == fieldTmin
				isTmaxField := m.editingField == fieldTmax

				if isNameField && len(ref.Name) > 0 {
					ref.Name = ref.Name[:len(ref.Name)-1]
				} else if isTminField && len(ref.Tmin) > 0 {
					ref.Tmin = ref.Tmin[:len(ref.Tmin)-1]
				} else if isTmaxField && len(ref.Tmax) > 0 {
					ref.Tmax = ref.Tmax[:len(ref.Tmax)-1]
				}
			default:
				char := msg.String()
				if len(char) == singleCharLength {
					ref := &m.TimeRanges[m.editingRangeIdx]
					if m.editingField == fieldName {
						ref.Name += char
					} else {
						// For numeric fields, only accept digits, dot, minus
						isNumericChar := (char >= "0" && char <= "9") || char == "." || char == "-"
						if isNumericChar {
							if m.editingField == fieldTmin {
								ref.Tmin += char
							} else {
								ref.Tmax += char
							}
						}
					}
				}
			}
			return m, nil
		}

		switch msg.String() {
		case "?":
			m.showHelp = true
			m.helpOverlay.Visible = true
		case "/":
			if m.CurrentStep == types.StepSelectSubjects {
				m.filteringSubject = true
				m.subjectFilter = ""
			}
		case "up", "k":
			m.handleUp()
		case "down", "j":
			m.handleDown()
		case " ":
			// Space to toggle selections
			m.handleSpace()
			// Check for pending file picker command (set by file path options)
			if m.pendingFileCmd != nil {
				cmd := m.pendingFileCmd
				m.pendingFileCmd = nil
				return m, cmd
			}
			// Check for pending fMRI conditions discovery command
			if m.pendingFmriConditionsCmd != nil {
				cmd := m.pendingFmriConditionsCmd
				m.pendingFmriConditionsCmd = nil
				return m, cmd
			}
		case "enter":
			// If an expanded list is open, toggle the current item (like Space does)
			// This allows multi-select fields to work properly
			if m.CurrentStep == types.StepAdvancedConfig && m.expandedOption >= 0 {
				m.handleExpandedListToggle()
				return m, nil
			}
			return m.handleEnter()
		case "tab":
			m.handleTab()
		case "a":
			// Handle expanded lists in advanced config first
			if m.CurrentStep == types.StepAdvancedConfig && m.expandedOption >= 0 {
				switch m.expandedOption {
				case expandedConnectivityMeasures:
					for i := range connectivityMeasures {
						m.connectivityMeasures[i] = true
					}
				case expandedDirectedConnMeasures:
					for i := range directedConnectivityMeasures {
						m.directedConnMeasures[i] = true
					}
				default:
					// For other expanded lists, use default selectAll behavior
					m.selectAll()
				}
			} else {
				// "A" is now only for "Select All"
				m.selectAll()
			}
		case "d", "x":
			isTimeRangeStep := m.CurrentStep == types.StepTimeRange
			isNotEditing := m.editingRangeIdx == noRangeEditing
			hasRanges := len(m.TimeRanges) > 0
			if isTimeRangeStep && isNotEditing && hasRanges {
				idx := m.timeRangeCursor
				m.TimeRanges = append(m.TimeRanges[:idx], m.TimeRanges[idx+1:]...)
				if m.timeRangeCursor >= len(m.TimeRanges) {
					m.timeRangeCursor = len(m.TimeRanges) - 1
				}
				if m.timeRangeCursor < 0 {
					m.timeRangeCursor = 0
				}
			} else {
				switch m.CurrentStep {
				case types.StepSelectBands:
					m.removeBand()
				case types.StepSelectROIs:
					m.removeROI()
				}
			}
		case "n":
			m.selectNone()

		case "g", "G":
			// Toggle global styling panel in plot categories page
			if m.CurrentStep == types.StepSelectPlotCategories && m.Pipeline == types.PipelinePlotting {
				m.showGlobalStyling = !m.showGlobalStyling
				if m.showGlobalStyling {
					m.globalStylingCursor = 0
					m.globalStylingOptions = m.getGlobalStylingOptions()
				}
			}

		case "e", "E":
			// Edit band frequencies or ROI channels
			switch m.CurrentStep {
			case types.StepSelectBands:
				m.startBandEdit()
			case types.StepSelectROIs:
				m.startROIEdit()
			}

		case "+", "=":
			// Add new item (time range, band, or ROI)
			isTimeRangeStep := m.CurrentStep == types.StepTimeRange
			isNotEditing := m.editingRangeIdx == noRangeEditing
			if isTimeRangeStep && isNotEditing {
				newName := fmt.Sprintf("range%d", len(m.TimeRanges)+1)
				m.TimeRanges = append(m.TimeRanges, types.TimeRange{Name: newName, Tmin: "", Tmax: ""})
				m.timeRangeCursor = len(m.TimeRanges) - 1
				m.editingRangeIdx = m.timeRangeCursor
				m.editingField = fieldName
			} else {
				switch m.CurrentStep {
				case types.StepSelectBands:
					m.addNewBand()
				case types.StepSelectROIs:
					m.addNewROI()
				}
			}

		case "-", "_":
			switch m.CurrentStep {
			case types.StepSelectBands:
				m.removeBand()
			case types.StepSelectROIs:
				m.removeROI()
			}

		case "R":
			// Refresh subjects (only meaningful on subject selection step)
			if m.CurrentStep == types.StepSelectSubjects {
				m.subjectsLoading = true
				return m, func() tea.Msg { return messages.RefreshSubjectsMsg{} }
			}
		case "ctrl+r":
			// Hidden alias for refresh (useful on keyboards without function keys)
			if m.CurrentStep == types.StepSelectSubjects {
				m.subjectsLoading = true
				return m, func() tea.Msg { return messages.RefreshSubjectsMsg{} }
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		maxOverlayWidth := m.width - 4
		if maxOverlayWidth < 1 {
			maxOverlayWidth = 1
		}
		m.helpOverlay.Width = min(50, maxOverlayWidth)
	}

	// Always update plot offset if in that step to ensure it's in sync
	if m.CurrentStep == types.StepSelectPlots {
		m.UpdatePlotOffset()
	}
	if m.CurrentStep == types.StepSelectFeaturePlotters {
		m.UpdateFeaturePlotterOffset()
	}
	if m.CurrentStep == types.StepSelectComputations {
		m.UpdateComputationOffset()
	}
	if m.CurrentStep == types.StepAdvancedConfig {
		m.UpdateAdvancedOffset()
	}

	return m, nil
}

// UpdateComputationOffset calculates and updates the scrolling offset for the computations list.
func (m *Model) UpdateComputationOffset() {
	// Match overhead with renderComputationSelection (12 lines)
	overheadLines := 12
	maxVisibleLines := m.height - overheadLines
	if maxVisibleLines < minVisibleLines {
		maxVisibleLines = minVisibleLines
	}

	totalLines := len(m.computations)
	m.computationOffset = calculateScrollOffset(
		m.computationCursor,
		m.computationOffset,
		totalLines,
		maxVisibleLines,
	)
}

// UpdateAdvancedOffset calculates and updates the scrolling offset for advanced config lists.
func (m *Model) UpdateAdvancedOffset() {
	// Use a fallback height if terminal size not yet received
	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	// Total height minus overhead - must match render functions
	overheadLines := 10
	maxLines := effectiveHeight - overheadLines
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	totalLines := 0
	cursorLine := 0

	switch m.Pipeline {
	case types.PipelineBehavior:
		options := m.getBehaviorOptions()
		totalLines = len(options)
		// Note: expanded list items are rendered inline, so cursorLine stays as advancedCursor
		cursorLine = m.advancedCursor

	case types.PipelineFeatures:
		options := m.getFeaturesOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

		if m.expandedOption == expandedConnectivityMeasures {
			expandedIdx := -1
			for i, opt := range options {
				if opt == optConnectivity {
					expandedIdx = i
					break
				}
			}
			if expandedIdx >= 0 {
				totalLines += len(connectivityMeasures)
				cursorLine = expandedIdx + 1 + m.subCursor
			}
		}

	case types.PipelinePlotting:
		rows := m.getPlottingAdvancedRows()
		totalLines = len(rows)
		cursorLine = m.advancedCursor

	case types.PipelinePreprocessing:
		options := m.getPreprocessingOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	case types.PipelineFmri:
		options := m.getFmriPreprocessingOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	case types.PipelineFmriAnalysis:
		options := m.getFmriAnalysisOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	case types.PipelineML:
		options := m.getMLOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	default:
		totalLines = 0
		cursorLine = 0
	}

	if totalLines <= 0 {
		m.advancedOffset = 0
		return
	}

	m.advancedOffset = calculateScrollOffset(
		cursorLine,
		m.advancedOffset,
		totalLines,
		maxLines,
	)
}

// UpdatePlotOffset calculates and updates the scrolling offset for the plots list
func (m *Model) UpdatePlotOffset() {
	// Match overhead with renderPlotSelection (10-14 lines)
	overheadLines := 10
	maxLines := m.height - overheadLines
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	// Reconstruct the list logic to find cursor position
	currentGroup := ""
	lineIdx := 0
	cursorLine := -1

	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}

		if plot.Group != currentGroup {
			lineIdx++ // Group header
			currentGroup = plot.Group
		}

		if i == m.plotCursor {
			cursorLine = lineIdx
		}
		lineIdx++ // Item line
	}

	if cursorLine == -1 {
		return
	}

	m.plotOffset = calculateScrollOffset(
		cursorLine,
		m.plotOffset,
		lineIdx,
		maxLines,
	)
}

func (m Model) selectedFeaturePlotterCategories() []string {
	var ordered []string
	seen := make(map[string]bool)
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] || !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		if plot.Group != "features" {
			continue
		}
		id := strings.TrimSpace(plot.ID)
		if !strings.HasPrefix(id, "features_") {
			continue
		}
		cat := strings.TrimPrefix(id, "features_")
		if cat == "" || seen[cat] {
			continue
		}
		seen[cat] = true
		ordered = append(ordered, cat)
	}
	return ordered
}

func (m Model) featurePlotterItems() []PlotterInfo {
	if m.featurePlotters == nil {
		return nil
	}
	var items []PlotterInfo
	for _, category := range m.selectedFeaturePlotterCategories() {
		items = append(items, m.featurePlotters[category]...)
	}
	return items
}

func (m *Model) UpdateFeaturePlotterOffset() {
	// Match overhead with renderFeaturePlotterSelection (10 lines)
	overheadLines := 10
	maxLines := m.height - overheadLines
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	items := m.featurePlotterItems()
	if len(items) == 0 {
		m.featurePlotterOffset = 0
		return
	}

	currentCategory := ""
	lineIdx := 0
	cursorLine := -1
	for i, p := range items {
		if p.Category != currentCategory {
			lineIdx++
			currentCategory = p.Category
		}
		if i == m.featurePlotterCursor {
			cursorLine = lineIdx
		}
		lineIdx++
	}
	if cursorLine < 0 {
		return
	}

	m.featurePlotterOffset = calculateScrollOffset(
		cursorLine,
		m.featurePlotterOffset,
		lineIdx,
		maxLines,
	)
}

// calculateScrollOffset computes the scroll offset to keep the cursor visible.
// It ensures the cursor stays within the visible area when scrolling.
func calculateScrollOffset(cursorLine, currentOffset, totalLines, maxVisibleLines int) int {
	if totalLines <= 0 {
		return 0
	}

	// Clamp cursor to valid range
	if cursorLine < 0 {
		cursorLine = 0
	}
	if cursorLine >= totalLines {
		cursorLine = totalLines - 1
	}

	// Adjust offset to keep cursor visible
	if cursorLine < currentOffset {
		currentOffset = cursorLine
	} else if cursorLine >= currentOffset+maxVisibleLines {
		currentOffset = cursorLine - maxVisibleLines + 1
	}

	// Ensure offset is non-negative
	if currentOffset < 0 {
		currentOffset = 0
	}

	// Ensure offset doesn't exceed maximum
	maxOffset := totalLines - maxVisibleLines
	if maxOffset > 0 && currentOffset > maxOffset {
		currentOffset = maxOffset
	}

	return currentOffset
}

func (m Model) plotCountsForGroup(group string) (total int, selected int) {
	for i, plot := range m.plotItems {
		if strings.EqualFold(plot.Group, group) {
			total++
			if m.plotSelected[i] {
				selected++
			}
		}
	}
	return total, selected
}

// isPlotGroupSelected checks if any plot from the given group(s) is currently selected.
// Accepts one or more group names and returns true if any plot in those groups is selected.
func (m Model) isPlotGroupSelected(groups ...string) bool {
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] {
			continue
		}
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		for _, group := range groups {
			if strings.EqualFold(plot.Group, group) {
				return true
			}
		}
	}
	return false
}

func (m Model) plotAvailabilitySummary(plot PlotItem) (int, int, map[string]int) {
	missing := make(map[string]int)
	total := 0
	available := 0

	for _, s := range m.subjects {
		if !m.subjectSelected[s.ID] {
			continue
		}
		total++

		hasEpochs := !plot.RequiresEpochs || s.HasEpochs
		hasFeatures := !plot.RequiresFeatures || s.HasFeatures
		hasStats := !plot.RequiresStats || s.HasStats
		isAvailable := hasEpochs && hasFeatures && hasStats

		if !hasEpochs {
			missing["epochs"]++
		}
		if !hasFeatures {
			missing["features"]++
		}
		if !hasStats {
			missing["stats"]++
		}

		if isAvailable {
			available++
		}
	}

	return available, total, missing
}

func (m Model) discoverTemporalTopomapsStatsFeatureFolders() ([]string, error) {
	derivRoot := strings.TrimSpace(m.derivRoot)
	if derivRoot == "" {
		return nil, fmt.Errorf("deriv_root is not set")
	}

	selectedSubjects := make([]string, 0, len(m.subjects))
	for _, s := range m.subjects {
		if m.subjectSelected[s.ID] {
			selectedSubjects = append(selectedSubjects, s.ID)
		}
	}
	if len(selectedSubjects) == 0 {
		return nil, fmt.Errorf("no subjects selected")
	}

	var intersection map[string]struct{}
	for _, subjID := range selectedSubjects {
		statsDir := filepath.Join(derivRoot, fmt.Sprintf("sub-%s", subjID), "eeg", "stats")
		kindDirs, _ := filepath.Glob(filepath.Join(statsDir, "temporal_correlations*"))
		perSubject := make(map[string]struct{})

		for _, kindDir := range kindDirs {
			entries, err := os.ReadDir(kindDir)
			if err != nil {
				continue
			}
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				featureFolder := entry.Name()
				featureDir := filepath.Join(kindDir, featureFolder)
				matches, _ := filepath.Glob(filepath.Join(featureDir, "temporal_correlations_by_condition*.npz"))
				if len(matches) > 0 {
					perSubject[featureFolder] = struct{}{}
				}
			}
		}

		if intersection == nil {
			intersection = perSubject
			continue
		}
		for k := range intersection {
			if _, ok := perSubject[k]; !ok {
				delete(intersection, k)
			}
		}
	}

	if len(intersection) == 0 {
		return nil, fmt.Errorf("no temporal_correlations feature folders found (expected NPZ in stats/temporal_correlations*/<feature>/)")
	}

	out := make([]string, 0, len(intersection))
	for k := range intersection {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}

///////////////////////////////////////////////////////////////////
// Setters
///////////////////////////////////////////////////////////////////

func (m *Model) SetSubjects(subjects []types.SubjectStatus) {
	m.subjects = subjects
	m.subjectsLoading = false
	for _, s := range subjects {
		m.subjectSelected[s.ID] = true
	}

	// Calculate feature and computation availability based on all subjects
	m.updateFeatureAvailability()
	m.updateComputationAvailability()
}

// updateFeatureAvailability recalculates feature availability based on selected subjects
func (m *Model) updateFeatureAvailability() {
	m.featureAvailability = make(map[string]bool)
	m.featureLastModified = make(map[string]string)

	for _, s := range m.subjects {
		isSelected := m.subjectSelected[s.ID]
		if !isSelected {
			continue
		}

		if s.FeatureAvailability == nil {
			continue
		}

		for cat, info := range s.FeatureAvailability.Features {
			if info.Available {
				m.featureAvailability[cat] = true
				hasLastModified := info.LastModified != ""
				if hasLastModified {
					existing, exists := m.featureLastModified[cat]
					isNewer := !exists || info.LastModified > existing
					if isNewer {
						m.featureLastModified[cat] = info.LastModified
					}
				}
			}
		}
	}
}

// updateComputationAvailability recalculates computation availability based on selected subjects
func (m *Model) updateComputationAvailability() {
	m.computationAvailability = make(map[string]bool)
	m.computationLastModified = make(map[string]string)

	for _, s := range m.subjects {
		isSelected := m.subjectSelected[s.ID]
		if !isSelected {
			continue
		}

		if s.FeatureAvailability == nil || s.FeatureAvailability.Computations == nil {
			continue
		}

		for comp, info := range s.FeatureAvailability.Computations {
			if info.Available {
				m.computationAvailability[comp] = true
				hasLastModified := info.LastModified != ""
				if hasLastModified {
					existing, exists := m.computationLastModified[comp]
					isNewer := !exists || info.LastModified > existing
					if isNewer {
						m.computationLastModified[comp] = info.LastModified
					}
				}
			}
		}
	}
}

func (m *Model) SetSubjectsLoading() {
	m.subjectsLoading = true
}

func (m *Model) SetTimeRanges(ranges []types.TimeRange) {
	if len(ranges) > 0 {
		m.TimeRanges = ranges
	}
}

// SetBands restores band definitions and selection states from persisted state.
func (m *Model) SetBands(bands []FrequencyBand, selected []bool) {
	if len(bands) > 0 {
		m.bands = bands
		m.bandSelected = make(map[int]bool)
		for i, sel := range selected {
			if i < len(bands) {
				m.bandSelected[i] = sel
			}
		}
	}
}

// GetBands returns the current band definitions for persistence.
func (m Model) GetBands() []FrequencyBand {
	return m.bands
}

// GetBandSelected returns the band selection states for persistence.
func (m Model) GetBandSelected() []bool {
	result := make([]bool, len(m.bands))
	for i := range m.bands {
		result[i] = m.bandSelected[i]
	}
	return result
}

// SetROIs restores ROI definitions and selection states from persisted state.
func (m *Model) SetROIs(rois []ROIDefinition, selected []bool) {
	if len(rois) > 0 {
		m.rois = rois
		m.roiSelected = make(map[int]bool)
		for i, sel := range selected {
			if i < len(rois) {
				m.roiSelected[i] = sel
			}
		}
	}
}

// GetROIs returns the current ROI definitions for persistence.
func (m Model) GetROIs() []ROIDefinition {
	return m.rois
}

// GetROISelected returns the ROI selection states for persistence.
func (m Model) GetROISelected() []bool {
	result := make([]bool, len(m.rois))
	for i := range m.rois {
		result[i] = m.roiSelected[i]
	}
	return result
}

// SetSpatialSelected restores spatial mode selection from persisted state.
func (m *Model) SetSpatialSelected(selected []bool) {
	if len(selected) > 0 {
		for i, sel := range selected {
			if i < len(spatialModes) {
				m.spatialSelected[i] = sel
			}
		}
	}
}

// GetSpatialSelected returns spatial selection states for persistence.
func (m Model) GetSpatialSelected() []bool {
	result := make([]bool, len(spatialModes))
	for i := range spatialModes {
		result[i] = m.spatialSelected[i]
	}
	return result
}

// SetAvailableMetadata stores runtime-derived metadata (e.g., discovered time
// windows / event columns) for use in UI hints and lightweight validation.
func (m *Model) SetAvailableMetadata(windows []string, eventColumns []string) {
	m.availableWindows = append([]string(nil), windows...)
	m.availableColumns = append([]string(nil), eventColumns...)
}

// SetAvailableWindowsByFeature stores windows discovered per feature group.
func (m *Model) SetAvailableWindowsByFeature(windowsByFeature map[string][]string) {
	if m.availableWindowsByFeature == nil {
		m.availableWindowsByFeature = make(map[string][]string)
	}
	for feature, windows := range windowsByFeature {
		m.availableWindowsByFeature[feature] = append([]string(nil), windows...)
	}
}

// SetChannelInfo stores available and unavailable EEG channels from BIDS data
// and preprocessing logs. Used by ROI selection to validate channel names.
func (m *Model) SetChannelInfo(available, unavailable []string) {
	m.availableChannels = append([]string(nil), available...)
	m.unavailableChannels = append([]string(nil), unavailable...)
}

func (m *Model) SetFeaturePlotters(plotters map[string][]PlotterInfo) {
	m.featurePlotters = plotters
	m.featurePlotterError = ""
	if m.featurePlotterSelected == nil {
		m.featurePlotterSelected = make(map[string]bool)
	}
	// Default: select all discovered plotters (the selection step can narrow).
	for _, entries := range plotters {
		for _, p := range entries {
			if p.ID != "" {
				m.featurePlotterSelected[p.ID] = true
			}
		}
	}
}

func (m *Model) SetFeaturePlottersError(err error) {
	if err == nil {
		return
	}
	m.featurePlotterError = err.Error()
}

// SetDiscoveredColumns sets the columns and values discovered from events/trial tables
func (m *Model) SetDiscoveredColumns(columns []string, values map[string][]string, source string) {
	m.discoveredColumns = columns
	m.discoveredColumnValues = values
	m.columnDiscoverySource = source
	m.columnDiscoveryDone = true
	m.columnDiscoveryError = ""
}

func (m *Model) SetTrialTableColumns(columns []string, values map[string][]string) {
	m.trialTableColumns = columns
	m.trialTableColumnValues = values
	m.trialTableFeatureCategories = detectFeatureCategoriesFromColumns(columns)
	m.trialTableDiscoveryDone = true
	m.trialTableDiscoveryError = ""
}

// SetColumnsDiscoveryError sets the error from column discovery
func (m *Model) SetColumnsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.columnDiscoveryError = err.Error()
	m.columnDiscoveryDone = true
}

func (m *Model) SetTrialTableDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.trialTableDiscoveryError = err.Error()
	m.trialTableDiscoveryDone = true
}

// GetDiscoveredColumnValues returns the unique values for a column.
// Checks primary discovered columns first, then trial table columns as fallback.
func (m Model) GetDiscoveredColumnValues(column string) []string {
	// First check the primary discovery source
	if m.discoveredColumnValues != nil {
		if vals, ok := m.discoveredColumnValues[column]; ok && len(vals) > 0 {
			return vals
		}
	}
	// Fallback to trial table values (in case column came from trial table discovery)
	if m.trialTableColumnValues != nil {
		if vals, ok := m.trialTableColumnValues[column]; ok && len(vals) > 0 {
			return vals
		}
	}
	return nil
}

// SetConditionEffectsColumns sets the columns, values, and windows discovered from condition effects files
func (m *Model) SetConditionEffectsColumns(columns []string, values map[string][]string, windows []string) {
	m.conditionEffectsColumns = columns
	m.conditionEffectsColumnValues = values
	m.conditionEffectsWindows = windows
	m.conditionEffectsDiscoveryDone = true
	m.conditionEffectsDiscoveryError = ""
}

// SetConditionEffectsDiscoveryError sets the error from condition effects discovery
func (m *Model) SetConditionEffectsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.conditionEffectsDiscoveryError = err.Error()
	m.conditionEffectsDiscoveryDone = true
}

// GetConditionEffectsColumnValues returns the unique values for a condition effects column
func (m Model) GetConditionEffectsColumnValues(column string) []string {
	if m.conditionEffectsColumnValues == nil {
		return nil
	}
	return m.conditionEffectsColumnValues[column]
}

// GetAvailableColumns returns the best available columns for dropdowns.
// Prefers discovered event columns, then falls back to lightweight columns
// surfaced by the subjects endpoint.
func (m Model) GetAvailableColumns() []string {
	if len(m.discoveredColumns) > 0 {
		return m.discoveredColumns
	}
	return m.availableColumns
}

// GetPlottingComparisonColumns returns columns for plotting comparison from trial table (events.tsv).
// Feature plots (connectivity, power, etc.) use trial-level columns for condition comparisons,
// not condition effects stats columns.
func (m Model) GetPlottingComparisonColumns() []string {
	// Prefer columns discovered from trial tables (richer, includes value maps),
	// but fall back to the lightweight columns surfaced by the subjects endpoint
	// so dropdowns still work even if discovery hasn't completed yet.
	if len(m.discoveredColumns) > 0 {
		return m.discoveredColumns
	}
	return m.availableColumns
}

func (m Model) GetTrialTableFeatureColumns() []string {
	if len(m.trialTableColumns) == 0 {
		return nil
	}

	// Keep in sync with eeg_pipeline.analysis.behavior.orchestration.FEATURE_COLUMN_PREFIXES
	featurePrefixes := []string{
		"power_",
		"connectivity_",
		"directedconnectivity_",
		"sourcelocalization_",
		"aperiodic_",
		"erp_",
		"itpc_",
		"pac_",
		"complexity_",
		"bursts_",
		"quality_",
		"erds_",
		"spectral_",
		"ratios_",
		"asymmetry_",
		"temporal_",
	}

	out := make([]string, 0, len(m.trialTableColumns))
	for _, col := range m.trialTableColumns {
		c := strings.TrimSpace(col)
		if c == "" {
			continue
		}
		for _, p := range featurePrefixes {
			if strings.HasPrefix(c, p) {
				out = append(out, c)
				break
			}
		}
	}
	return out
}

func (m Model) GetTrialTableFeatureCategories() []string {
	if len(m.trialTableFeatureCategories) == 0 {
		return nil
	}
	out := make([]string, 0, len(m.trialTableFeatureCategories))
	out = append(out, m.trialTableFeatureCategories...)
	return out
}

func detectFeatureCategoriesFromColumns(columns []string) []string {
	// Keep in sync with eeg_pipeline.analysis.behavior.orchestration.FEATURE_COLUMN_PREFIXES
	featurePrefixes := []string{
		"power_",
		"connectivity_",
		"directedconnectivity_",
		"sourcelocalization_",
		"aperiodic_",
		"erp_",
		"itpc_",
		"pac_",
		"complexity_",
		"bursts_",
		"quality_",
		"erds_",
		"spectral_",
		"ratios_",
		"asymmetry_",
		"temporal_",
	}

	prefixSet := make(map[string]bool, len(featurePrefixes))
	for _, p := range featurePrefixes {
		prefixSet[p] = true
	}

	seen := make(map[string]bool, len(featurePrefixes))
	for _, col := range columns {
		c := strings.TrimSpace(col)
		if c == "" {
			continue
		}
		idx := strings.Index(c, "_")
		if idx <= 0 {
			continue
		}
		prefix := c[:idx+1]
		if !prefixSet[prefix] {
			continue
		}
		seen[prefix] = true
	}

	out := make([]string, 0, len(featurePrefixes))
	for _, p := range featurePrefixes {
		if seen[p] {
			out = append(out, strings.TrimSuffix(p, "_"))
		}
	}
	return out
}

// GetPlottingComparisonWindows returns windows for plotting comparison from computed feature data.
// If featureGroup is provided, returns only windows for that feature group.
func (m Model) GetPlottingComparisonWindows(featureGroup ...string) []string {
	if len(featureGroup) > 0 && featureGroup[0] != "" {
		// Prefer feature-specific windows (more accurate), but fall back to the global
		// discovered windows list so selection dropdowns still work even when the
		// executor didn't provide per-feature windows.
		//
		// This is a UI convenience fallback only; downstream plotting code will still
		// validate window availability when reading feature files.
		if m.availableWindowsByFeature != nil {
			if windows, ok := m.availableWindowsByFeature[featureGroup[0]]; ok && len(windows) > 0 {
				return windows
			}
		}
		return m.availableWindows
	}
	return m.availableWindows
}

// getPlotByID returns the PlotItem for the given plot ID.
func (m Model) getPlotByID(plotID string) PlotItem {
	for _, plot := range m.plotItems {
		if plot.ID == plotID {
			return plot
		}
	}
	return PlotItem{Group: ""}
}

// getFeatureGroupForPlot returns the feature group name used in column names for a given plot.
// Maps plot Group/ID to the actual feature group in NamingSchema (e.g., "phase" -> "itpc" or "pac").
func (m Model) getFeatureGroupForPlot(plotID string) string {
	plot := m.getPlotByID(plotID)

	// Map plot IDs to their feature groups
	switch plotID {
	case "itpc_topomaps", "itpc_by_condition":
		return "itpc"
	case "pac_by_condition":
		return "pac"
	}

	// For most plots, Group matches the feature group name
	// But handle special cases
	switch plot.Group {
	case "phase":
		// Phase group contains both ITPC and PAC - determine from plot ID
		if strings.HasPrefix(plotID, "itpc_") {
			return "itpc"
		}
		if strings.HasPrefix(plotID, "pac_") {
			return "pac"
		}
		return "itpc" // Default for phase plots
	}

	// Default: use Group as feature group (works for power, connectivity, aperiodic, etc.)
	return plot.Group
}

// GetPlottingComparisonColumnValues returns values for a column in plotting comparison from trial table.
// Feature plots use trial-level columns from events.tsv for condition comparisons.
func (m Model) GetPlottingComparisonColumnValues(column string) []string {
	return m.GetDiscoveredColumnValues(column)
}

// SetFmriDiscoveredColumns sets the columns and values discovered from fMRI events files
func (m *Model) SetFmriDiscoveredColumns(columns []string, values map[string][]string, source string) {
	m.fmriDiscoveredColumns = columns
	m.fmriDiscoveredColumnValues = values
	m.fmriColumnDiscoveryDone = true
	m.fmriColumnDiscoveryError = ""
}

// SetFmriColumnsDiscoveryError sets the error from fMRI column discovery
func (m *Model) SetFmriColumnsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.fmriColumnDiscoveryError = err.Error()
	m.fmriColumnDiscoveryDone = true
}

// GetFmriDiscoveredColumnValues returns the unique values for an fMRI column
func (m Model) GetFmriDiscoveredColumnValues(column string) []string {
	if m.fmriDiscoveredColumnValues == nil {
		return nil
	}
	return m.fmriDiscoveredColumnValues[column]
}

// SetMultigroupStats sets the multigroup stats discovered from precomputed stats
func (m *Model) SetMultigroupStats(available bool, groups []string, nFeatures int, nSignificant int, file string) {
	m.multigroupStatsAvailable = available
	m.multigroupStatsGroups = groups
	m.multigroupStatsNFeatures = nFeatures
	m.multigroupStatsNSignificant = nSignificant
	m.multigroupStatsFile = file
	m.multigroupStatsDiscoveryDone = true
}

// HasMultigroupStats returns whether multigroup stats are available
func (m Model) HasMultigroupStats() bool {
	return m.multigroupStatsAvailable && len(m.multigroupStatsGroups) > 0
}

// GetMultigroupStatsGroups returns the group labels from precomputed multigroup stats
func (m Model) GetMultigroupStatsGroups() []string {
	return m.multigroupStatsGroups
}

// SetDiscoveredROIs sets the ROIs discovered from feature parquet files
func (m *Model) SetDiscoveredROIs(rois []string) {
	m.discoveredROIs = rois
	m.roiDiscoveryDone = true
	m.roiDiscoveryError = ""
}

// SetROIDiscoveryError sets the error from ROI discovery
func (m *Model) SetROIDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.roiDiscoveryError = err.Error()
	m.roiDiscoveryDone = true
}

// getExpandedListLength returns the length of the currently expanded list
func (m Model) getExpandedListLength() int {
	switch m.expandedOption {
	case expandedConnectivityMeasures:
		return len(connectivityMeasures)
	case expandedDirectedConnMeasures:
		return len(directedConnectivityMeasures)
	case expandedConditionCompareColumn, expandedTemporalConditionColumn, expandedClusterConditionColumn:
		return len(m.GetAvailableColumns())
	case expandedConnConditionColumn:
		return len(m.GetAvailableColumns())
	case expandedConditionCompareValues:
		if m.conditionCompareColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.conditionCompareColumn))
	case expandedTemporalConditionValues:
		if m.temporalConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.temporalConditionColumn))
	case expandedClusterConditionValues:
		if m.clusterConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.clusterConditionColumn))
	case expandedPlotComparisonColumn:
		if m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn {
			return len(m.GetTrialTableFeatureCategories())
		}
		return len(m.GetPlottingComparisonColumns())
	case expandedPlotComparisonValues:
		// Check plot-specific column first, then fall back to global
		col := ""
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok && cfg.ComparisonColumn != "" {
				col = cfg.ComparisonColumn
			}
		}
		if col == "" {
			col = m.plotComparisonColumn
		}
		if col == "" {
			return 0
		}
		return len(m.GetPlottingComparisonColumnValues(col))
	case expandedConditionCompareWindows:
		return len(m.availableWindows)
	case expandedPlotComparisonWindows:
		if m.editingPlotID != "" {
			if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
				// Dose-response segment uses global windows list (same as renderer),
				// not feature-group-specific windows.
				return len(m.GetPlottingComparisonWindows())
			}
			featureGroup := m.getFeatureGroupForPlot(m.editingPlotID)
			return len(m.GetPlottingComparisonWindows(featureGroup))
		}
		return len(m.GetPlottingComparisonWindows())
	case expandedPlotComparisonROIs:
		return len(m.discoveredROIs)
	case expandedRunAdjustmentColumn:
		return len(m.GetAvailableColumns())
	case expandedCorrelationsTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(none)" option
	case expandedTemporalTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(default)" option
	case expandedMLTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(stage default)" option
	case expandedItpcConditionColumn:
		return len(m.GetAvailableColumns())
	case expandedConnConditionValues:
		if m.connConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.connConditionColumn))
	case expandedFmriCondAColumn, expandedFmriCondBColumn:
		return len(m.fmriDiscoveredColumns)
	case expandedFmriCondAValue:
		return len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn))
	case expandedFmriCondBValue:
		return len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn))
	case expandedFmriAnalysisCondAColumn, expandedFmriAnalysisCondBColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisCondAValue:
		n := len(m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondAColumn))
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisCondBValue:
		n := len(m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondBColumn))
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisStimPhases:
		return len(m.getExpandedListItems())
	case expandedFmriTrialSigGroupColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriTrialSigGroupValues:
		n := len(m.GetFmriDiscoveredColumnValues(m.fmriTrialSigGroupColumn))
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriTrialSigStimPhases:
		return len(m.getExpandedListItems())
	case expandedSourceLocFmriStimPhases:
		return len(m.getExpandedListItems())
	case expandedItpcConditionValues:
		if m.itpcConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.itpcConditionColumn))
	case expandedBehaviorScatterFeatures:
		return len(behaviorScatterFeatureTypes)
	case expandedBehaviorScatterColumns:
		return len(m.GetPlottingComparisonColumns())
	case expandedBehaviorScatterAggregation:
		return len(behaviorScatterAggregationModes)
	case expandedBehaviorScatterSegment:
		return len(m.GetPlottingComparisonWindows())
	case expandedTemporalTopomapsFeatureDir:
		return len(m.temporalTopomapsStatsFeatureFolders)
	case expandedPainResidualCrossfitGroupColumn:
		if len(m.GetAvailableColumns()) == 0 {
			return 2
		}
		return len(m.GetAvailableColumns()) + 1
	case expandedStabilityGroupColumn:
		return 3
	}
	return 0
}

// getExpandedListItems returns the items in the currently expanded list
func (m Model) getExpandedListItems() []string {
	switch m.expandedOption {
	case expandedConnectivityMeasures:
		items := make([]string, len(connectivityMeasures))
		for i, measure := range connectivityMeasures {
			items[i] = measure.Key
		}
		return items
	case expandedDirectedConnMeasures:
		items := make([]string, len(directedConnectivityMeasures))
		for i, measure := range directedConnectivityMeasures {
			items[i] = measure.Key
		}
		return items
	case expandedConditionCompareColumn, expandedTemporalConditionColumn, expandedClusterConditionColumn:
		return m.GetAvailableColumns()
	case expandedConnConditionColumn:
		return m.GetAvailableColumns()
	case expandedConditionCompareValues:
		if m.conditionCompareColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.conditionCompareColumn)
	case expandedTemporalConditionValues:
		if m.temporalConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.temporalConditionColumn)
	case expandedClusterConditionValues:
		if m.clusterConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.clusterConditionColumn)
	case expandedPlotComparisonColumn:
		if m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn {
			return m.GetTrialTableFeatureCategories()
		}
		return m.GetPlottingComparisonColumns()
	case expandedPlotComparisonValues:
		// Check plot-specific column first, then fall back to global
		col := ""
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok && cfg.ComparisonColumn != "" {
				col = cfg.ComparisonColumn
			}
		}
		if col == "" {
			col = m.plotComparisonColumn
		}
		if col == "" {
			return nil
		}
		return m.GetPlottingComparisonColumnValues(col)
	case expandedConditionCompareWindows:
		return m.availableWindows
	case expandedPlotComparisonWindows:
		if m.editingPlotID != "" {
			if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
				// Dose-response segment uses global windows list (same as renderer),
				// not feature-group-specific windows.
				return m.GetPlottingComparisonWindows()
			}
			featureGroup := m.getFeatureGroupForPlot(m.editingPlotID)
			return m.GetPlottingComparisonWindows(featureGroup)
		}
		return m.GetPlottingComparisonWindows()
	case expandedPlotComparisonROIs:
		return m.discoveredROIs
	case expandedRunAdjustmentColumn:
		return m.GetAvailableColumns()
	case expandedCorrelationsTargetColumn:
		return append([]string{"(none)"}, m.GetAvailableColumns()...)
	case expandedTemporalTargetColumn:
		return append([]string{"(default)"}, m.GetAvailableColumns()...)
	case expandedMLTargetColumn:
		return append([]string{"(stage default)"}, m.GetAvailableColumns()...)
	case expandedItpcConditionColumn:
		return m.GetAvailableColumns()
	case expandedFmriCondAColumn, expandedFmriCondBColumn:
		return m.fmriDiscoveredColumns
	case expandedFmriCondAValue:
		return m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn)
	case expandedFmriCondBValue:
		return m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn)
	case expandedFmriAnalysisCondAColumn, expandedFmriAnalysisCondBColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriAnalysisCondAValue:
		vals := m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondAColumn)
		if len(vals) == 0 {
			return []string{"(type manually)"}
		}
		return vals
	case expandedFmriAnalysisCondBValue:
		vals := m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondBColumn)
		if len(vals) == 0 {
			return []string{"(type manually)"}
		}
		return vals
	case expandedFmriAnalysisStimPhases:
		items := []string{"(auto)", "(all)"}
		vals := m.GetFmriDiscoveredColumnValues("stim_phase")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriTrialSigGroupColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriTrialSigGroupValues:
		if m.fmriTrialSigGroupColumn == "" {
			return nil
		}
		vals := m.GetFmriDiscoveredColumnValues(m.fmriTrialSigGroupColumn)
		if len(vals) == 0 {
			return []string{"(type manually)"}
		}
		return vals
	case expandedFmriTrialSigStimPhases:
		items := []string{"(auto)", "(all)"}
		vals := m.GetFmriDiscoveredColumnValues("stim_phase")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedSourceLocFmriStimPhases:
		items := []string{"(auto)", "(all)"}
		vals := m.GetFmriDiscoveredColumnValues("stim_phase")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedItpcConditionValues:
		if m.itpcConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.itpcConditionColumn)
	case expandedConnConditionValues:
		if m.connConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.connConditionColumn)
	case expandedBehaviorScatterFeatures:
		return behaviorScatterFeatureTypes
	case expandedBehaviorScatterColumns:
		return m.GetPlottingComparisonColumns()
	case expandedBehaviorScatterAggregation:
		return behaviorScatterAggregationModes
	case expandedBehaviorScatterSegment:
		return m.GetPlottingComparisonWindows()
	case expandedTemporalTopomapsFeatureDir:
		return m.temporalTopomapsStatsFeatureFolders
	case expandedPainResidualCrossfitGroupColumn:
		items := []string{"(default: run column)"}
		cols := m.GetAvailableColumns()
		if len(cols) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, cols...)
	case expandedStabilityGroupColumn:
		return []string{"(auto)", "run", "block"}
	}
	return nil
}

// isColumnValueSelected checks if a value is selected for the current column context
func (m Model) isColumnValueSelected(value string) bool {
	var selectedValues string
	switch m.expandedOption {
	case expandedConditionCompareValues:
		selectedValues = m.conditionCompareValues
	case expandedTemporalConditionValues:
		selectedValues = m.temporalConditionValues
	case expandedClusterConditionValues:
		selectedValues = m.clusterConditionValues
	case expandedPlotComparisonValues:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.ComparisonValuesSpec
			}
		}
		if selectedValues == "" {
			selectedValues = m.plotComparisonValuesSpec
		}
	case expandedConditionCompareWindows:
		selectedValues = m.conditionCompareWindows
	case expandedPlotComparisonWindows:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				if m.editingPlotField == plotItemConfigFieldComparisonSegment {
					selectedValues = cfg.ComparisonSegment
				} else if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
					selectedValues = cfg.DoseResponseSegment
				} else if m.editingPlotField == plotItemConfigFieldTopomapWindow {
					selectedValues = cfg.TopomapWindowsSpec
				} else {
					selectedValues = cfg.ComparisonWindowsSpec
				}
			}
		}
		if selectedValues == "" {
			selectedValues = m.plotComparisonWindowsSpec
		}
	case expandedPlotComparisonROIs:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.ComparisonROIsSpec
			}
		}
	case expandedItpcConditionValues:
		selectedValues = m.itpcConditionValues
	case expandedConnConditionValues:
		selectedValues = m.connConditionValues
	case expandedFmriTrialSigGroupValues:
		selectedValues = m.fmriTrialSigGroupValuesSpec
	case expandedBehaviorScatterFeatures:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterFeaturesSpec
			}
		}
	case expandedBehaviorScatterColumns:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterColumnsSpec
			}
		}
	case expandedBehaviorScatterAggregation:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterAggregationModesSpec
			}
		}
	case expandedBehaviorScatterSegment:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterSegmentSpec
			}
		}
	case expandedPlotComparisonColumn:
		if m.editingPlotID != "" && m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.DoseResponseResponseColumn
			}
		}
	default:
		return false
	}
	if selectedValues == "" {
		return false
	}
	// Check if value is in space or comma-separated list
	for _, v := range strings.Fields(selectedValues) {
		if v == value {
			return true
		}
	}
	for _, v := range strings.Split(selectedValues, ",") {
		if strings.TrimSpace(v) == value {
			return true
		}
	}
	return false
}

// handleExpandedListToggle handles toggling items in expanded column/value lists
func (m *Model) handleExpandedListToggle() {
	items := m.getExpandedListItems()
	if m.subCursor < 0 || m.subCursor >= len(items) {
		return
	}

	selectedItem := items[m.subCursor]

	switch m.expandedOption {
	case expandedConnectivityMeasures:
		m.connectivityMeasures[m.subCursor] = !m.connectivityMeasures[m.subCursor]

	case expandedDirectedConnMeasures:
		m.directedConnMeasures[m.subCursor] = !m.directedConnMeasures[m.subCursor]

	case expandedConditionCompareColumn:
		m.conditionCompareColumn = selectedItem
		m.conditionCompareValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedTemporalConditionColumn:
		m.temporalConditionColumn = selectedItem
		m.temporalConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedClusterConditionColumn:
		m.clusterConditionColumn = selectedItem
		m.clusterConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedConditionCompareValues:
		m.toggleColumnValue(selectedItem, &m.conditionCompareValues)

	case expandedTemporalConditionValues:
		m.toggleColumnValue(selectedItem, &m.temporalConditionValues)

	case expandedClusterConditionValues:
		m.toggleColumnValue(selectedItem, &m.clusterConditionValues)

	case expandedPlotComparisonColumn:
		// Update plot-specific config if editing a plot field, otherwise global
		if m.editingPlotID != "" {
			plotID := m.editingPlotID
			cfg := m.ensurePlotItemConfig(plotID)

			switch m.editingPlotField {
			case plotItemConfigFieldDoseResponseDoseColumn:
				cfg.DoseResponseDoseColumn = selectedItem
			case plotItemConfigFieldDoseResponseResponseColumn:
				// Multi-select feature categories (space-separated)
				m.toggleSpaceValue(selectedItem, &cfg.DoseResponseResponseColumn)
				m.plotItemConfigs[plotID] = cfg
				return
			case plotItemConfigFieldDoseResponsePainColumn:
				cfg.DoseResponsePainColumn = selectedItem
			default:
				cfg.ComparisonColumn = selectedItem
				cfg.ComparisonValuesSpec = "" // Reset values when column changes
			}

			m.plotItemConfigs[plotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
		} else {
			m.plotComparisonColumn = selectedItem
			m.plotComparisonValuesSpec = "" // Reset values when column changes
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedPlotComparisonValues:
		// Update plot-specific config if editing a plot field, otherwise global
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.ComparisonValuesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		} else {
			m.toggleSpaceValue(selectedItem, &m.plotComparisonValuesSpec)
		}

	case expandedConditionCompareWindows:
		m.toggleSpaceValue(selectedItem, &m.conditionCompareWindows)

	case expandedPlotComparisonWindows:
		// Update plot-specific config if editing a plot field, otherwise global
		if m.editingPlotID != "" {
			plotID := m.editingPlotID
			cfg := m.ensurePlotItemConfig(plotID)
			if m.editingPlotField == plotItemConfigFieldComparisonSegment {
				// Segment is single-select
				cfg.ComparisonSegment = selectedItem
				m.plotItemConfigs[plotID] = cfg
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				m.expandedOption = expandedNone
				m.subCursor = 0
			} else if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
				// Dose-response segment is single-select
				cfg.DoseResponseSegment = selectedItem
				m.plotItemConfigs[plotID] = cfg
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				m.expandedOption = expandedNone
				m.subCursor = 0
			} else if m.editingPlotField == plotItemConfigFieldTopomapWindow {
				// TopomapWindowsSpec is multi-select
				m.toggleSpaceValue(selectedItem, &cfg.TopomapWindowsSpec)
				m.plotItemConfigs[plotID] = cfg
			} else {
				// Windows is multi-select
				m.toggleSpaceValue(selectedItem, &cfg.ComparisonWindowsSpec)
				m.plotItemConfigs[plotID] = cfg
			}
		} else {
			m.toggleSpaceValue(selectedItem, &m.plotComparisonWindowsSpec)
		}

	case expandedPlotComparisonROIs:
		// Update plot-specific config - ROIs is multi-select
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.ComparisonROIsSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedRunAdjustmentColumn:
		m.runAdjustmentColumn = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedCorrelationsTargetColumn:
		if selectedItem == "(none)" {
			m.correlationsTargetColumn = ""
		} else {
			m.correlationsTargetColumn = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedTemporalTargetColumn:
		if selectedItem == "(default)" {
			m.temporalTargetColumn = ""
		} else {
			m.temporalTargetColumn = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedMLTargetColumn:
		if selectedItem == "(stage default)" {
			m.mlTarget = ""
		} else {
			m.mlTarget = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedItpcConditionColumn:
		m.itpcConditionColumn = selectedItem
		m.itpcConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedConnConditionColumn:
		m.connConditionColumn = selectedItem
		m.connConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondAColumn:
		m.sourceLocFmriCondAColumn = selectedItem
		m.sourceLocFmriCondAValue = "" // Reset value when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondAValue:
		m.sourceLocFmriCondAValue = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondBColumn:
		m.sourceLocFmriCondBColumn = selectedItem
		m.sourceLocFmriCondBValue = "" // Reset value when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondBValue:
		m.sourceLocFmriCondBValue = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedSourceLocFmriStimPhases:
		switch selectedItem {
		case "(auto)":
			m.sourceLocFmriStimPhasesToModel = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(all)":
			m.sourceLocFmriStimPhasesToModel = "all"
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriStimPhasesToModel)
		default:
			m.sourceLocFmriStimPhasesToModel = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondAColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondAColumn)
		} else {
			m.fmriAnalysisCondAColumn = selectedItem
			m.fmriAnalysisCondAValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondAValue:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondAValue)
		} else {
			m.fmriAnalysisCondAValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondBColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondBColumn)
		} else {
			m.fmriAnalysisCondBColumn = selectedItem
			m.fmriAnalysisCondBValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondBValue:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondBValue)
		} else {
			m.fmriAnalysisCondBValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisStimPhases:
		switch selectedItem {
		case "(auto)":
			m.fmriAnalysisStimPhasesToModel = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(all)":
			m.fmriAnalysisStimPhasesToModel = "all"
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisStimPhasesToModel)
		default:
			m.fmriAnalysisStimPhasesToModel = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriTrialSigGroupColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigGroupColumn)
		} else {
			m.fmriTrialSigGroupColumn = selectedItem
			m.fmriTrialSigGroupValuesSpec = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriTrialSigGroupValues:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigGroupValues)
		} else {
			m.toggleSpaceValue(selectedItem, &m.fmriTrialSigGroupValuesSpec)
		}
	case expandedFmriTrialSigStimPhases:
		switch selectedItem {
		case "(auto)":
			m.fmriTrialSigScopeStimPhases = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(all)":
			m.fmriTrialSigScopeStimPhases = "all"
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigScopeStimPhases)
		default:
			m.fmriTrialSigScopeStimPhases = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}

	case expandedItpcConditionValues:
		m.toggleColumnValue(selectedItem, &m.itpcConditionValues)
	case expandedConnConditionValues:
		m.toggleColumnValue(selectedItem, &m.connConditionValues)

	case expandedBehaviorScatterFeatures:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.BehaviorScatterFeaturesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedBehaviorScatterColumns:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.BehaviorScatterColumnsSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedBehaviorScatterAggregation:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.BehaviorScatterAggregationModesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedBehaviorScatterSegment:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			// Segment is single-select
			cfg.BehaviorScatterSegmentSpec = selectedItem
			m.plotItemConfigs[m.editingPlotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedTemporalTopomapsFeatureDir:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			cfg.BehaviorTemporalStatsFeatureFolder = selectedItem
			m.plotItemConfigs[m.editingPlotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedPainResidualCrossfitGroupColumn:
		switch selectedItem {
		case "(default: run column)":
			m.painResidualCrossfitGroupColumn = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldPainResidualCrossfitGroupColumn)
		default:
			m.painResidualCrossfitGroupColumn = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedStabilityGroupColumn:
		switch selectedItem {
		case "run":
			m.stabilityGroupColumn = 1
		case "block":
			m.stabilityGroupColumn = 2
		default:
			m.stabilityGroupColumn = 0
		}
		m.expandedOption = expandedNone
		m.subCursor = 0
	}

	m.useDefaultAdvanced = false
}

// shouldRenderExpandedListAfterOption checks if we should render an expanded list after the given option
func (m Model) shouldRenderExpandedListAfterOption(opt optionType) bool {
	switch m.expandedOption {
	case expandedConditionCompareColumn:
		return opt == optConditionCompareColumn
	case expandedConditionCompareValues:
		return opt == optConditionCompareValues
	case expandedConditionCompareWindows:
		return opt == optConditionCompareWindows
	case expandedTemporalConditionColumn:
		return opt == optTemporalConditionColumn
	case expandedTemporalConditionValues:
		return opt == optTemporalConditionValues
	case expandedClusterConditionColumn:
		return opt == optClusterConditionColumn
	case expandedClusterConditionValues:
		return opt == optClusterConditionValues
	case expandedPlotComparisonColumn:
		return opt == optPlotComparisonColumn
	case expandedPlotComparisonValues:
		return opt == optPlotComparisonValues
	case expandedPlotComparisonWindows:
		return opt == optPlotComparisonWindows
	case expandedRunAdjustmentColumn:
		return opt == optRunAdjustmentColumn
	case expandedCorrelationsTargetColumn:
		return opt == optCorrelationsTargetColumn
	case expandedTemporalTargetColumn:
		return opt == optTemporalTargetColumn
	case expandedMLTargetColumn:
		return opt == optMLTarget
	case expandedItpcConditionColumn:
		return opt == optItpcConditionColumn
	case expandedConnConditionColumn:
		return opt == optConnConditionColumn
	case expandedFmriCondAColumn:
		return opt == optSourceLocFmriCondAColumn
	case expandedFmriCondAValue:
		return opt == optSourceLocFmriCondAValue
	case expandedFmriCondBColumn:
		return opt == optSourceLocFmriCondBColumn
	case expandedFmriCondBValue:
		return opt == optSourceLocFmriCondBValue
	case expandedFmriAnalysisCondAColumn:
		return opt == optFmriAnalysisCondAColumn
	case expandedFmriAnalysisCondAValue:
		return opt == optFmriAnalysisCondAValue
	case expandedFmriAnalysisCondBColumn:
		return opt == optFmriAnalysisCondBColumn
	case expandedFmriAnalysisCondBValue:
		return opt == optFmriAnalysisCondBValue
	case expandedFmriAnalysisStimPhases:
		return opt == optFmriAnalysisStimPhasesToModel
	case expandedFmriTrialSigGroupColumn:
		return opt == optFmriTrialSigGroupColumn
	case expandedFmriTrialSigGroupValues:
		return opt == optFmriTrialSigGroupValues
	case expandedFmriTrialSigStimPhases:
		return opt == optFmriTrialSigScopeStimPhases
	case expandedItpcConditionValues:
		return opt == optItpcConditionValues
	case expandedConnConditionValues:
		return opt == optConnConditionValues
	case expandedSourceLocFmriStimPhases:
		return opt == optSourceLocFmriStimPhasesToModel
	case expandedPainResidualCrossfitGroupColumn:
		return opt == optPainResidualCrossfitGroupColumn
	case expandedStabilityGroupColumn:
		return opt == optStabilityGroupColumn
	}
	return false
}

// isExpandedItemSelected checks if an item at the given index is selected in the expanded list
func (m Model) isExpandedItemSelected(_ int, item string) bool {
	switch m.expandedOption {
	case expandedConditionCompareColumn:
		return m.conditionCompareColumn == item
	case expandedTemporalConditionColumn:
		return m.temporalConditionColumn == item
	case expandedClusterConditionColumn:
		return m.clusterConditionColumn == item
	case expandedPlotComparisonColumn:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok && cfg.ComparisonColumn != "" {
				return cfg.ComparisonColumn == item
			}
		}
		return m.plotComparisonColumn == item
	case expandedRunAdjustmentColumn:
		return m.runAdjustmentColumn == item
	case expandedCorrelationsTargetColumn:
		if item == "(none)" {
			return m.correlationsTargetColumn == ""
		}
		return m.correlationsTargetColumn == item
	case expandedTemporalTargetColumn:
		if item == "(default)" {
			return m.temporalTargetColumn == ""
		}
		return m.temporalTargetColumn == item
	case expandedMLTargetColumn:
		if item == "(stage default)" {
			return m.mlTarget == ""
		}
		return m.mlTarget == item
	case expandedItpcConditionColumn:
		return m.itpcConditionColumn == item
	case expandedConnConditionColumn:
		return m.connConditionColumn == item
	case expandedFmriCondAColumn:
		return m.sourceLocFmriCondAColumn == item
	case expandedFmriCondAValue:
		return m.sourceLocFmriCondAValue == item
	case expandedFmriCondBColumn:
		return m.sourceLocFmriCondBColumn == item
	case expandedFmriCondBValue:
		return m.sourceLocFmriCondBValue == item
	case expandedSourceLocFmriStimPhases:
		if item == "(auto)" {
			return strings.TrimSpace(m.sourceLocFmriStimPhasesToModel) == ""
		}
		if item == "(all)" {
			return strings.TrimSpace(m.sourceLocFmriStimPhasesToModel) == "all"
		}
		for _, p := range splitSpaceList(m.sourceLocFmriStimPhasesToModel) {
			if p == item {
				return true
			}
		}
		return false
	case expandedFmriAnalysisCondAColumn:
		return m.fmriAnalysisCondAColumn == item
	case expandedFmriAnalysisCondAValue:
		return m.fmriAnalysisCondAValue == item
	case expandedFmriAnalysisCondBColumn:
		return m.fmriAnalysisCondBColumn == item
	case expandedFmriAnalysisCondBValue:
		return m.fmriAnalysisCondBValue == item
	case expandedFmriAnalysisStimPhases:
		if item == "(auto)" {
			return strings.TrimSpace(m.fmriAnalysisStimPhasesToModel) == ""
		}
		if item == "(all)" {
			return strings.TrimSpace(m.fmriAnalysisStimPhasesToModel) == "all"
		}
		for _, p := range splitSpaceList(m.fmriAnalysisStimPhasesToModel) {
			if p == item {
				return true
			}
		}
		return false
	case expandedFmriTrialSigGroupColumn:
		return m.fmriTrialSigGroupColumn == item
	case expandedFmriTrialSigStimPhases:
		if item == "(auto)" {
			return strings.TrimSpace(m.fmriTrialSigScopeStimPhases) == ""
		}
		if item == "(all)" {
			return strings.TrimSpace(m.fmriTrialSigScopeStimPhases) == "all"
		}
		for _, p := range splitSpaceList(m.fmriTrialSigScopeStimPhases) {
			if p == item {
				return true
			}
		}
		return false
	case expandedConditionCompareValues, expandedTemporalConditionValues, expandedClusterConditionValues, expandedPlotComparisonValues,
		expandedConditionCompareWindows, expandedPlotComparisonWindows, expandedItpcConditionValues, expandedConnConditionValues,
		expandedFmriTrialSigGroupValues:
		return m.isColumnValueSelected(item)
	case expandedBehaviorScatterSegment:
		// Check plot-specific config
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				return cfg.BehaviorScatterSegmentSpec == item
			}
		}
		return false
	case expandedPainResidualCrossfitGroupColumn:
		if item == "(default: run column)" {
			return strings.TrimSpace(m.painResidualCrossfitGroupColumn) == ""
		}
		if item == "(type manually)" {
			return false
		}
		return m.painResidualCrossfitGroupColumn == item
	case expandedStabilityGroupColumn:
		switch item {
		case "(auto)":
			return m.stabilityGroupColumn == 0
		case "run":
			return m.stabilityGroupColumn == 1
		case "block":
			return m.stabilityGroupColumn == 2
		}
		return false
	}
	return false
}

// toggleColumnValue toggles a value in a comma-separated list
func (m *Model) toggleColumnValue(value string, target *string) {
	if *target == "" {
		*target = value
		return
	}

	// Parse existing values
	existing := strings.Split(*target, ",")
	var newValues []string
	found := false

	for _, v := range existing {
		v = strings.TrimSpace(v)
		if v == value {
			found = true
		} else if v != "" {
			newValues = append(newValues, v)
		}
	}

	if !found {
		newValues = append(newValues, value)
	}

	*target = strings.Join(newValues, ",")
}

// toggleSpaceValue toggles a value in a space-separated list
func (m *Model) toggleSpaceValue(value string, target *string) {
	if *target == "" {
		*target = value
		return
	}

	existing := strings.Fields(*target)
	var newValues []string
	found := false

	for _, v := range existing {
		if v == value {
			found = true
		} else if v != "" {
			newValues = append(newValues, v)
		}
	}

	if !found {
		newValues = append(newValues, value)
	}

	*target = strings.Join(newValues, " ")
}

func (m *Model) SetConfigSummary(summary messages.ConfigSummary) {
	if m.task == "" && summary.Task != "" {
		m.task = summary.Task
	}
	if m.bidsRoot == "" && summary.BidsRoot != "" {
		m.bidsRoot = summary.BidsRoot
	}
	if m.bidsFmriRoot == "" && summary.BidsFmriRoot != "" {
		m.bidsFmriRoot = summary.BidsFmriRoot
	}
	if m.derivRoot == "" && summary.DerivRoot != "" {
		m.derivRoot = summary.DerivRoot
	}
	if m.sourceRoot == "" && summary.SourceRoot != "" {
		m.sourceRoot = summary.SourceRoot
	}
	if summary.PreprocessingNJobs > 0 {
		m.prepNJobs = summary.PreprocessingNJobs
	}
}

func (m *Model) SetRepoRoot(repoRoot string) {
	m.repoRoot = repoRoot
}

func (m *Model) startTextEdit(field textField) {
	m.editingTextField = field
	m.textBuffer = m.getTextFieldValue(field)
	m.editingPlotID = ""
	m.editingPlotField = plotItemConfigFieldNone
	m.editingText = true
}

func (m *Model) commitTextInput() {
	if m.editingPlotID != "" && m.editingPlotField != plotItemConfigFieldNone {
		m.setPlotItemTextFieldValue(m.editingPlotID, m.editingPlotField, m.textBuffer)
		return
	}
	m.setTextFieldValue(m.editingTextField, m.textBuffer)
}

func (m *Model) startPlotTextEdit(plotID string, field plotItemConfigField) {
	m.editingTextField = textFieldNone
	m.editingPlotID = plotID
	m.editingPlotField = field
	m.textBuffer = m.getPlotItemTextFieldValue(plotID, field)
	m.editingText = true
}

func (m Model) getTextFieldValue(field textField) string {
	switch field {
	case textFieldTask:
		return m.task
	case textFieldBidsRoot:
		return m.bidsRoot
	case textFieldBidsFmriRoot:
		return m.bidsFmriRoot
	case textFieldDerivRoot:
		return m.derivRoot
	case textFieldSourceRoot:
		return m.sourceRoot
	case textFieldFmriFmriprepImage:
		return m.fmriFmriprepImage
	case textFieldFmriFmriprepOutputDir:
		return m.fmriFmriprepOutputDir
	case textFieldFmriFmriprepWorkDir:
		return m.fmriFmriprepWorkDir
	case textFieldFmriFreesurferLicenseFile:
		return m.fmriFreesurferLicenseFile
	case textFieldFmriFreesurferSubjectsDir:
		return m.fmriFreesurferSubjectsDir
	case textFieldFmriOutputSpaces:
		return m.fmriOutputSpacesSpec
	case textFieldFmriIgnore:
		return m.fmriIgnoreSpec
	case textFieldFmriBidsFilterFile:
		return m.fmriBidsFilterFile
	case textFieldFmriExtraArgs:
		return m.fmriExtraArgs
	case textFieldFmriSkullStripTemplate:
		return m.fmriSkullStripTemplate
	case textFieldFmriTaskId:
		return m.fmriTaskId
	case textFieldFmriAnalysisFmriprepSpace:
		return m.fmriAnalysisFmriprepSpace
	case textFieldFmriAnalysisRuns:
		return m.fmriAnalysisRunsSpec
	case textFieldFmriAnalysisCondAColumn:
		return m.fmriAnalysisCondAColumn
	case textFieldFmriAnalysisCondAValue:
		return m.fmriAnalysisCondAValue
	case textFieldFmriAnalysisCondBColumn:
		return m.fmriAnalysisCondBColumn
	case textFieldFmriAnalysisCondBValue:
		return m.fmriAnalysisCondBValue
	case textFieldFmriAnalysisContrastName:
		return m.fmriAnalysisContrastName
	case textFieldFmriAnalysisFormula:
		return m.fmriAnalysisFormula
	case textFieldFmriAnalysisEventsToModel:
		return m.fmriAnalysisEventsToModel
	case textFieldFmriAnalysisStimPhasesToModel:
		return m.fmriAnalysisStimPhasesToModel
	case textFieldFmriAnalysisOutputDir:
		return m.fmriAnalysisOutputDir
	case textFieldFmriAnalysisFreesurferDir:
		return m.fmriAnalysisFreesurferDir
		case textFieldFmriAnalysisSignatureDir:
			return m.fmriAnalysisSignatureDir
		case textFieldFmriTrialSigGroupColumn:
			return m.fmriTrialSigGroupColumn
		case textFieldFmriTrialSigGroupValues:
			return m.fmriTrialSigGroupValuesSpec
	case textFieldFmriTrialSigScopeStimPhases:
		return m.fmriTrialSigScopeStimPhases
	case textFieldRawMontage:
		return m.rawMontage
	case textFieldPrepMontage:
		return m.prepMontage
	case textFieldPrepChTypes:
		return m.prepChTypes
	case textFieldPrepEegReference:
		return m.prepEegReference
	case textFieldPrepEogChannels:
		return m.prepEogChannels
	case textFieldPrepConditions:
		return m.prepConditions
	case textFieldPrepFileExtension:
		return m.prepFileExtension
	case textFieldPrepRenameAnotDict:
		return m.prepRenameAnotDict
	case textFieldPrepCustomBadDict:
		return m.prepCustomBadDict
	case textFieldRawEventPrefixes:
		return m.rawEventPrefixes
	case textFieldMergeEventPrefixes:
		return m.mergeEventPrefixes
	case textFieldMergeEventTypes:
		return m.mergeEventTypes
	case textFieldFmriRawSession:
		return m.fmriRawSession
	case textFieldFmriRawRestTask:
		return m.fmriRawRestTask
	case textFieldFmriRawDcm2niixPath:
		return m.fmriRawDcm2niixPath
	case textFieldFmriRawDcm2niixArgs:
		return m.fmriRawDcm2niixArgs
	case textFieldConditionCompareColumn:
		return m.conditionCompareColumn
	case textFieldConditionCompareWindows:
		return m.conditionCompareWindows
	case textFieldConditionCompareValues:
		return m.conditionCompareValues
	case textFieldTemporalConditionColumn:
		return m.temporalConditionColumn
	case textFieldTemporalConditionValues:
		return m.temporalConditionValues
	case textFieldTemporalTargetColumn:
		return m.temporalTargetColumn
	case textFieldTfHeatmapFreqs:
		return m.tfHeatmapFreqsSpec
	case textFieldRunAdjustmentColumn:
		return m.runAdjustmentColumn
	case textFieldPainResidualCrossfitGroupColumn:
		return m.painResidualCrossfitGroupColumn
	case textFieldPainResidualSplineDfCandidates:
		return m.painResidualSplineDfCandidates
	case textFieldPainResidualModelComparePolyDegrees:
		return m.painResidualModelComparePolyDegrees
	case textFieldClusterConditionColumn:
		return m.clusterConditionColumn
	case textFieldClusterConditionValues:
		return m.clusterConditionValues
	case textFieldCorrelationsTargetColumn:
		return m.correlationsTargetColumn
	case textFieldCorrelationsTypes:
		return m.correlationsTypesSpec
	case textFieldItpcConditionColumn:
		return m.itpcConditionColumn
	case textFieldItpcConditionValues:
		return m.itpcConditionValues
	case textFieldConnConditionColumn:
		return m.connConditionColumn
	case textFieldConnConditionValues:
		return m.connConditionValues
	case textFieldPACPairs:
		return m.pacPairsSpec
	case textFieldBurstBands:
		return m.burstBandsSpec
	case textFieldERDSBands:
		return m.erdsBandsSpec
	case textFieldSpectralRatioPairs:
		return m.spectralRatioPairsSpec
	case textFieldAsymmetryChannelPairs:
		return m.asymmetryChannelPairsSpec
	case textFieldAsymmetryActivationBands:
		return m.asymmetryActivationBandsSpec
	case textFieldIAFRois:
		return m.iafRoisSpec
	case textFieldERPComponents:
		return m.erpComponentsSpec
	case textFieldSourceLocSubject:
		return m.sourceLocSubject
	case textFieldSourceLocTrans:
		return m.sourceLocTrans
	case textFieldSourceLocBem:
		return m.sourceLocBem
	case textFieldSourceLocFmriStatsMap:
		return m.sourceLocFmriStatsMap
	case textFieldSourceLocFmriCondAColumn:
		return m.sourceLocFmriCondAColumn
	case textFieldSourceLocFmriCondAValue:
		return m.sourceLocFmriCondAValue
	case textFieldSourceLocFmriCondBColumn:
		return m.sourceLocFmriCondBColumn
	case textFieldSourceLocFmriCondBValue:
		return m.sourceLocFmriCondBValue
	case textFieldSourceLocFmriContrastFormula:
		return m.sourceLocFmriContrastFormula
	case textFieldSourceLocFmriContrastName:
		return m.sourceLocFmriContrastName
	case textFieldSourceLocFmriRunsToInclude:
		return m.sourceLocFmriRunsToInclude
	case textFieldSourceLocFmriStimPhasesToModel:
		return m.sourceLocFmriStimPhasesToModel
	case textFieldSourceLocFmriWindowAName:
		return m.sourceLocFmriWindowAName
	case textFieldSourceLocFmriWindowBName:
		return m.sourceLocFmriWindowBName
	case textFieldPlotBboxInches:
		return m.plotBboxInches
	case textFieldPlotFontFamily:
		return m.plotFontFamily
	case textFieldPlotFontWeight:
		return m.plotFontWeight
	case textFieldPlotLayoutTightRect:
		return m.plotLayoutTightRectSpec
	case textFieldPlotLayoutTightRectMicrostate:
		return m.plotLayoutTightRectMicrostateSpec
	case textFieldPlotGridSpecWidthRatios:
		return m.plotGridSpecWidthRatiosSpec
	case textFieldPlotGridSpecHeightRatios:
		return m.plotGridSpecHeightRatiosSpec
	case textFieldPlotFigureSizeStandard:
		return m.plotFigureSizeStandardSpec
	case textFieldPlotFigureSizeMedium:
		return m.plotFigureSizeMediumSpec
	case textFieldPlotFigureSizeSmall:
		return m.plotFigureSizeSmallSpec
	case textFieldPlotFigureSizeSquare:
		return m.plotFigureSizeSquareSpec
	case textFieldPlotFigureSizeWide:
		return m.plotFigureSizeWideSpec
	case textFieldPlotFigureSizeTFR:
		return m.plotFigureSizeTFRSpec
	case textFieldPlotFigureSizeTopomap:
		return m.plotFigureSizeTopomapSpec
	case textFieldPlotColorPain:
		return m.plotColorPain
	case textFieldPlotColorNonpain:
		return m.plotColorNonpain
	case textFieldPlotColorSignificant:
		return m.plotColorSignificant
	case textFieldPlotColorNonsignificant:
		return m.plotColorNonsignificant
	case textFieldPlotColorGray:
		return m.plotColorGray
	case textFieldPlotColorLightGray:
		return m.plotColorLightGray
	case textFieldPlotColorBlack:
		return m.plotColorBlack
	case textFieldPlotColorBlue:
		return m.plotColorBlue
	case textFieldPlotColorRed:
		return m.plotColorRed
	case textFieldPlotColorNetworkNode:
		return m.plotColorNetworkNode
	case textFieldPlotScatterEdgecolor:
		return m.plotScatterEdgeColor
	case textFieldPlotHistEdgecolor:
		return m.plotHistEdgeColor
	case textFieldPlotKdeColor:
		return m.plotKdeColor
	case textFieldPlotTopomapColormap:
		return m.plotTopomapColormap
	case textFieldPlotTopomapSigMaskMarker:
		return m.plotTopomapSigMaskMarker
	case textFieldPlotTopomapSigMaskMarkerFaceColor:
		return m.plotTopomapSigMaskMarkerFaceColor
	case textFieldPlotTopomapSigMaskMarkerEdgeColor:
		return m.plotTopomapSigMaskMarkerEdgeColor
	case textFieldPlotTfrDefaultBaselineWindow:
		return m.plotTfrDefaultBaselineWindowSpec
	case textFieldPlotPacCmap:
		return m.plotPacCmap
	case textFieldPlotPacPairs:
		return m.plotPacPairsSpec
	case textFieldPlotConnectivityMeasures:
		return strings.Join(m.selectedConnectivityMeasures(), " ")
	case textFieldPlotSpectralMetrics:
		return m.plotSpectralMetricsSpec
	case textFieldPlotBurstsMetrics:
		return m.plotBurstsMetricsSpec
	case textFieldPlotTemporalTimeBins:
		return m.plotTemporalTimeBinsSpec
	case textFieldPlotTemporalTimeLabels:
		return m.plotTemporalTimeLabelsSpec
	case textFieldPlotAsymmetryStat:
		return m.plotAsymmetryStatSpec
	case textFieldPlotComparisonWindows:
		return m.plotComparisonWindowsSpec
	case textFieldPlotComparisonSegment:
		return m.plotComparisonSegment
	case textFieldPlotComparisonColumn:
		return m.plotComparisonColumn
	case textFieldPlotComparisonValues:
		return m.plotComparisonValuesSpec
	case textFieldPlotComparisonLabels:
		return m.plotComparisonLabelsSpec
	case textFieldPlotComparisonROIs:
		return m.plotComparisonROIsSpec
	// Machine Learning advanced config text fields
	case textFieldMLTarget:
		return m.mlTarget
	case textFieldMLFmriSigContrastName:
		return m.mlFmriSigContrastName
	case textFieldMLFeatureFamilies:
		return m.mlFeatureFamiliesSpec
	case textFieldMLFeatureBands:
		return m.mlFeatureBandsSpec
	case textFieldMLFeatureSegments:
		return m.mlFeatureSegmentsSpec
	case textFieldMLFeatureScopes:
		return m.mlFeatureScopesSpec
	case textFieldMLFeatureStats:
		return m.mlFeatureStatsSpec
	case textFieldMLCovariates:
		return m.mlCovariatesSpec
	case textFieldMLBaselinePredictors:
		return m.mlBaselinePredictorsSpec
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		return m.elasticNetAlphaGrid
	case textFieldElasticNetL1RatioGrid:
		return m.elasticNetL1RatioGrid
	case textFieldRidgeAlphaGrid:
		return m.ridgeAlphaGrid
	case textFieldRfMaxDepthGrid:
		return m.rfMaxDepthGrid
	case textFieldVarianceThresholdGrid:
		return m.varianceThresholdGrid
	// Preprocessing text fields
	case textFieldIcaLabelsToKeep:
		return m.icaLabelsToKeep
	default:
		return ""
	}
}

func (m Model) getPlotItemTextFieldValue(plotID string, field plotItemConfigField) string {
	cfg, ok := m.plotItemConfigs[plotID]
	if !ok {
		return ""
	}
	switch field {
	case plotItemConfigFieldTfrDefaultBaselineWindow:
		return cfg.TfrDefaultBaselineWindowSpec
	case plotItemConfigFieldComparisonWindows:
		return cfg.ComparisonWindowsSpec
	case plotItemConfigFieldComparisonSegment:
		return cfg.ComparisonSegment
	case plotItemConfigFieldComparisonColumn:
		return cfg.ComparisonColumn
	case plotItemConfigFieldComparisonValues:
		return cfg.ComparisonValuesSpec
	case plotItemConfigFieldComparisonLabels:
		return cfg.ComparisonLabelsSpec
	case plotItemConfigFieldComparisonROIs:
		return cfg.ComparisonROIsSpec
	case plotItemConfigFieldTopomapWindow:
		return cfg.TopomapWindowsSpec
	case plotItemConfigFieldTfrTopomapActiveWindow:
		return cfg.TfrTopomapActiveWindow
	case plotItemConfigFieldTfrTopomapWindowSizeMs:
		return cfg.TfrTopomapWindowSizeMs
	case plotItemConfigFieldTfrTopomapWindowCount:
		return cfg.TfrTopomapWindowCount
	case plotItemConfigFieldTfrTopomapLabelXPosition:
		return cfg.TfrTopomapLabelXPosition
	case plotItemConfigFieldTfrTopomapLabelYPositionBottom:
		return cfg.TfrTopomapLabelYPositionBottom
	case plotItemConfigFieldTfrTopomapLabelYPosition:
		return cfg.TfrTopomapLabelYPosition
	case plotItemConfigFieldTfrTopomapTitleY:
		return cfg.TfrTopomapTitleY
	case plotItemConfigFieldTfrTopomapTitlePad:
		return cfg.TfrTopomapTitlePad
	case plotItemConfigFieldTfrTopomapSubplotsRight:
		return cfg.TfrTopomapSubplotsRight
	case plotItemConfigFieldTfrTopomapTemporalHspace:
		return cfg.TfrTopomapTemporalHspace
	case plotItemConfigFieldTfrTopomapTemporalWspace:
		return cfg.TfrTopomapTemporalWspace
	case plotItemConfigFieldConnectivityCircleTopFraction:
		return cfg.ConnectivityCircleTopFraction
	case plotItemConfigFieldConnectivityCircleMinLines:
		return cfg.ConnectivityCircleMinLines
	case plotItemConfigFieldConnectivityNetworkTopFraction:
		return cfg.ConnectivityNetworkTopFraction
	case plotItemConfigFieldBehaviorTemporalStatsFeatureFolder:
		return cfg.BehaviorTemporalStatsFeatureFolder
	case plotItemConfigFieldDoseResponseDoseColumn:
		return cfg.DoseResponseDoseColumn
	case plotItemConfigFieldDoseResponseResponseColumn:
		return cfg.DoseResponseResponseColumn
	case plotItemConfigFieldDoseResponsePainColumn:
		return cfg.DoseResponsePainColumn
	case plotItemConfigFieldDoseResponseSegment:
		return cfg.DoseResponseSegment
	default:
		return ""
	}
}

func (m *Model) ApplyConfigKeys(values map[string]interface{}) {
	asString := func(v interface{}) (string, bool) {
		s, ok := v.(string)
		if !ok {
			return "", false
		}
		return strings.TrimSpace(s), true
	}
	asBool := func(v interface{}) (bool, bool) {
		b, ok := v.(bool)
		return b, ok
	}
	asInt := func(v interface{}) (int, bool) {
		switch n := v.(type) {
		case float64:
			return int(n), true
		case int:
			return n, true
		default:
			return 0, false
		}
	}
	asStringList := func(v interface{}) ([]string, bool) {
		raw, ok := v.([]interface{})
		if !ok {
			return nil, false
		}
		var out []string
		for _, item := range raw {
			s, ok := item.(string)
			if !ok {
				continue
			}
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
		return out, true
	}

	if rawBands, ok := values["time_frequency_analysis.bands"]; ok {
		bands := parseConfigBands(rawBands)
		if len(bands) > 0 {
			m.bands = bands
			m.bandSelected = make(map[int]bool)
			for i := range m.bands {
				m.bandSelected[i] = true
			}
			m.bandCursor = 0
		}
	}

	// fMRI preprocessing defaults (used by the fMRI pipeline wizard)
	if v, ok := values["paths.bids_fmri_root"]; ok {
		if s, ok := asString(v); ok && s != "" {
			m.bidsFmriRoot = s
		}
	}
	if v, ok := values["fmri_preprocessing.engine"]; ok {
		if s, ok := asString(v); ok {
			switch s {
			case "apptainer":
				m.fmriEngineIndex = 1
			case "docker":
				m.fmriEngineIndex = 0
			}
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.image"]; ok {
		if s, ok := asString(v); ok && s != "" {
			m.fmriFmriprepImage = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.output_dir"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFmriprepOutputDir = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.work_dir"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFmriprepWorkDir = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.fs_license_file"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFreesurferLicenseFile = s
		}
	}
	if v, ok := values["paths.freesurfer_license"]; ok {
		if strings.TrimSpace(m.fmriFreesurferLicenseFile) == "" {
			if s, ok := asString(v); ok {
				m.fmriFreesurferLicenseFile = s
			}
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.fs_subjects_dir"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFreesurferSubjectsDir = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.output_spaces"]; ok {
		if list, ok := asStringList(v); ok && len(list) > 0 {
			m.fmriOutputSpacesSpec = strings.Join(list, " ")
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.ignore"]; ok {
		if list, ok := asStringList(v); ok && len(list) > 0 {
			m.fmriIgnoreSpec = strings.Join(list, " ")
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.bids_filter_file"]; ok {
		if s, ok := asString(v); ok {
			m.fmriBidsFilterFile = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.extra_args"]; ok {
		if s, ok := asString(v); ok {
			m.fmriExtraArgs = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.use_aroma"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriUseAroma = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.skip_bids_validation"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriSkipBidsValidation = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.stop_on_first_crash"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriStopOnFirstCrash = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.clean_workdir"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriCleanWorkdir = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.fs_no_reconall"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriSkipReconstruction = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.mem_mb"]; ok {
		if n, ok := asInt(v); ok {
			m.fmriMemMb = n
		}
	}
}

func parseConfigBands(value interface{}) []FrequencyBand {
	raw, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}
	names := make([]string, 0, len(raw))
	for name := range raw {
		names = append(names, name)
	}
	sort.Strings(names)

	var bands []FrequencyBand
	for _, name := range names {
		entry, ok := raw[name].([]interface{})
		if !ok || len(entry) < 2 {
			continue
		}
		var low, high float64
		var okLow, okHigh bool

		// Try to parse as float64 first
		if f, ok := entry[0].(float64); ok {
			low = f
			okLow = true
		} else if i, ok := entry[0].(int); ok {
			low = float64(i)
			okLow = true
		}

		if f, ok := entry[1].(float64); ok {
			high = f
			okHigh = true
		} else if i, ok := entry[1].(int); ok {
			high = float64(i)
			okHigh = true
		}

		if !okLow || !okHigh {
			continue
		}
		bands = append(bands, FrequencyBand{
			Key:         name,
			Name:        titleCase(name),
			Description: fmt.Sprintf("%.1f-%.1f Hz", low, high),
			LowHz:       low,
			HighHz:      high,
		})
	}
	return bands
}

func titleCase(value string) string {
	if value == "" {
		return value
	}
	return strings.ToUpper(value[:1]) + value[1:]
}

func (m Model) behaviorSections() []behaviorSection {
	return []behaviorSection{
		{Key: "general", Label: "General", Enabled: true},
		{Key: "trial_table", Label: "Trial Table", Enabled: m.isComputationSelected("trial_table")},
		{Key: "correlations", Label: "Correlations", Enabled: m.isComputationSelected("correlations")},
		{Key: "pain_sensitivity", Label: "Pain Sensitivity", Enabled: m.isComputationSelected("pain_sensitivity")},
		{Key: "regression", Label: "Regression", Enabled: m.isComputationSelected("regression")},
		{Key: "stability", Label: "Stability", Enabled: m.isComputationSelected("stability")},
		{Key: "consistency", Label: "Consistency", Enabled: m.isComputationSelected("consistency")},
		{Key: "influence", Label: "Influence", Enabled: m.isComputationSelected("influence")},
		{Key: "condition", Label: "Condition", Enabled: m.isComputationSelected("condition")},
		{Key: "temporal", Label: "Temporal", Enabled: m.isComputationSelected("temporal")},
		{Key: "cluster", Label: "Cluster", Enabled: m.isComputationSelected("cluster")},
		{Key: "mediation", Label: "Mediation", Enabled: m.isComputationSelected("mediation")},
		{Key: "moderation", Label: "Moderation", Enabled: m.isComputationSelected("moderation")},
		{Key: "mixed_effects", Label: "Mixed Effects", Enabled: m.isComputationSelected("mixed_effects")},
	}
}

func (m *Model) setTextFieldValue(field textField, value string) {
	value = strings.TrimSpace(value)
	switch field {
	case textFieldTask:
		m.task = value
	case textFieldBidsRoot:
		m.bidsRoot = value
	case textFieldBidsFmriRoot:
		m.bidsFmriRoot = value
	case textFieldDerivRoot:
		m.derivRoot = value
	case textFieldSourceRoot:
		m.sourceRoot = value
	case textFieldFmriFmriprepImage:
		m.fmriFmriprepImage = value
	case textFieldFmriFmriprepOutputDir:
		m.fmriFmriprepOutputDir = value
	case textFieldFmriFmriprepWorkDir:
		m.fmriFmriprepWorkDir = value
	case textFieldFmriFreesurferLicenseFile:
		m.fmriFreesurferLicenseFile = value
	case textFieldFmriFreesurferSubjectsDir:
		m.fmriFreesurferSubjectsDir = value
	case textFieldFmriOutputSpaces:
		m.fmriOutputSpacesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriIgnore:
		m.fmriIgnoreSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriBidsFilterFile:
		m.fmriBidsFilterFile = value
	case textFieldFmriExtraArgs:
		m.fmriExtraArgs = value
	case textFieldFmriSkullStripTemplate:
		m.fmriSkullStripTemplate = strings.TrimSpace(value)
	case textFieldFmriTaskId:
		m.fmriTaskId = strings.TrimSpace(value)
	case textFieldFmriAnalysisFmriprepSpace:
		m.fmriAnalysisFmriprepSpace = strings.TrimSpace(value)
	case textFieldFmriAnalysisRuns:
		m.fmriAnalysisRunsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriAnalysisCondAColumn:
		m.fmriAnalysisCondAColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondAValue:
		m.fmriAnalysisCondAValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondBColumn:
		m.fmriAnalysisCondBColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondBValue:
		m.fmriAnalysisCondBValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisContrastName:
		m.fmriAnalysisContrastName = strings.TrimSpace(value)
	case textFieldFmriAnalysisFormula:
		m.fmriAnalysisFormula = strings.TrimSpace(value)
	case textFieldFmriAnalysisEventsToModel:
		m.fmriAnalysisEventsToModel = strings.TrimSpace(value)
	case textFieldFmriAnalysisStimPhasesToModel:
		m.fmriAnalysisStimPhasesToModel = strings.TrimSpace(value)
	case textFieldFmriAnalysisOutputDir:
		m.fmriAnalysisOutputDir = strings.TrimSpace(value)
	case textFieldFmriAnalysisFreesurferDir:
		m.fmriAnalysisFreesurferDir = strings.TrimSpace(value)
		case textFieldFmriAnalysisSignatureDir:
			m.fmriAnalysisSignatureDir = strings.TrimSpace(value)
		case textFieldFmriTrialSigGroupColumn:
			m.fmriTrialSigGroupColumn = strings.TrimSpace(value)
			m.fmriTrialSigGroupValuesSpec = "" // Reset values when column changes
		case textFieldFmriTrialSigGroupValues:
		m.fmriTrialSigGroupValuesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriTrialSigScopeStimPhases:
		m.fmriTrialSigScopeStimPhases = strings.TrimSpace(value)
	case textFieldRawMontage:
		m.rawMontage = value
	case textFieldPrepMontage:
		m.prepMontage = value
	case textFieldPrepChTypes:
		m.prepChTypes = value
	case textFieldPrepEegReference:
		m.prepEegReference = value
	case textFieldPrepEogChannels:
		m.prepEogChannels = value
	case textFieldPrepConditions:
		m.prepConditions = value
	case textFieldPrepFileExtension:
		m.prepFileExtension = strings.TrimSpace(value)
	case textFieldPrepRenameAnotDict:
		m.prepRenameAnotDict = value
	case textFieldPrepCustomBadDict:
		m.prepCustomBadDict = value
	case textFieldRawEventPrefixes:
		m.rawEventPrefixes = value
	case textFieldMergeEventPrefixes:
		m.mergeEventPrefixes = value
	case textFieldMergeEventTypes:
		m.mergeEventTypes = value
	case textFieldFmriRawSession:
		m.fmriRawSession = value
	case textFieldFmriRawRestTask:
		m.fmriRawRestTask = value
	case textFieldFmriRawDcm2niixPath:
		m.fmriRawDcm2niixPath = value
	case textFieldFmriRawDcm2niixArgs:
		m.fmriRawDcm2niixArgs = value
	case textFieldConditionCompareColumn:
		m.conditionCompareColumn = strings.TrimSpace(value)
	case textFieldConditionCompareWindows:
		m.conditionCompareWindows = strings.TrimSpace(value)
	case textFieldConditionCompareValues:
		m.conditionCompareValues = strings.TrimSpace(value)
	case textFieldTemporalConditionColumn:
		m.temporalConditionColumn = strings.TrimSpace(value)
	case textFieldTemporalConditionValues:
		m.temporalConditionValues = strings.TrimSpace(value)
	case textFieldTemporalTargetColumn:
		m.temporalTargetColumn = strings.TrimSpace(value)
	case textFieldTfHeatmapFreqs:
		m.tfHeatmapFreqsSpec = strings.Join(strings.Fields(value), "")
	case textFieldRunAdjustmentColumn:
		m.runAdjustmentColumn = strings.TrimSpace(value)
	case textFieldPainResidualCrossfitGroupColumn:
		m.painResidualCrossfitGroupColumn = strings.TrimSpace(value)
	case textFieldPainResidualSplineDfCandidates:
		m.painResidualSplineDfCandidates = strings.Join(strings.Fields(value), "")
	case textFieldPainResidualModelComparePolyDegrees:
		m.painResidualModelComparePolyDegrees = strings.Join(strings.Fields(value), "")
	case textFieldClusterConditionColumn:
		m.clusterConditionColumn = strings.TrimSpace(value)
	case textFieldClusterConditionValues:
		m.clusterConditionValues = strings.TrimSpace(value)
	case textFieldCorrelationsTargetColumn:
		m.correlationsTargetColumn = strings.TrimSpace(value)
	case textFieldCorrelationsTypes:
		m.correlationsTypesSpec = strings.Join(strings.Fields(value), "")
	case textFieldItpcConditionColumn:
		m.itpcConditionColumn = strings.TrimSpace(value)
	case textFieldItpcConditionValues:
		m.itpcConditionValues = strings.TrimSpace(value)
	case textFieldConnConditionColumn:
		m.connConditionColumn = strings.TrimSpace(value)
	case textFieldConnConditionValues:
		m.connConditionValues = strings.TrimSpace(value)
	case textFieldPACPairs:
		m.pacPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldBurstBands:
		m.burstBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldERDSBands:
		m.erdsBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSpectralRatioPairs:
		m.spectralRatioPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldAsymmetryChannelPairs:
		m.asymmetryChannelPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldAsymmetryActivationBands:
		m.asymmetryActivationBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldIAFRois:
		m.iafRoisSpec = strings.Join(strings.Fields(value), "")
	case textFieldERPComponents:
		m.erpComponentsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSourceLocSubject:
		m.sourceLocSubject = value
	case textFieldSourceLocTrans:
		m.sourceLocTrans = value
	case textFieldSourceLocBem:
		m.sourceLocBem = value
	case textFieldSourceLocFmriStatsMap:
		m.sourceLocFmriStatsMap = value
	case textFieldSourceLocFmriCondAColumn:
		m.sourceLocFmriCondAColumn = value
	case textFieldSourceLocFmriCondAValue:
		m.sourceLocFmriCondAValue = value
	case textFieldSourceLocFmriCondBColumn:
		m.sourceLocFmriCondBColumn = value
	case textFieldSourceLocFmriCondBValue:
		m.sourceLocFmriCondBValue = value
	case textFieldSourceLocFmriContrastFormula:
		m.sourceLocFmriContrastFormula = value
	case textFieldSourceLocFmriContrastName:
		m.sourceLocFmriContrastName = value
	case textFieldSourceLocFmriRunsToInclude:
		m.sourceLocFmriRunsToInclude = value
	case textFieldSourceLocFmriStimPhasesToModel:
		m.sourceLocFmriStimPhasesToModel = strings.TrimSpace(value)
	case textFieldSourceLocFmriWindowAName:
		m.sourceLocFmriWindowAName = value
	case textFieldSourceLocFmriWindowBName:
		m.sourceLocFmriWindowBName = value
	case textFieldPlotBboxInches:
		m.plotBboxInches = value
	case textFieldPlotFontFamily:
		m.plotFontFamily = value
	case textFieldPlotFontWeight:
		m.plotFontWeight = value
	case textFieldPlotLayoutTightRect:
		m.plotLayoutTightRectSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotLayoutTightRectMicrostate:
		m.plotLayoutTightRectMicrostateSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotGridSpecWidthRatios:
		m.plotGridSpecWidthRatiosSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotGridSpecHeightRatios:
		m.plotGridSpecHeightRatiosSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeStandard:
		m.plotFigureSizeStandardSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeMedium:
		m.plotFigureSizeMediumSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeSmall:
		m.plotFigureSizeSmallSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeSquare:
		m.plotFigureSizeSquareSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeWide:
		m.plotFigureSizeWideSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeTFR:
		m.plotFigureSizeTFRSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeTopomap:
		m.plotFigureSizeTopomapSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotColorPain:
		m.plotColorPain = value
	case textFieldPlotColorNonpain:
		m.plotColorNonpain = value
	case textFieldPlotColorSignificant:
		m.plotColorSignificant = value
	case textFieldPlotColorNonsignificant:
		m.plotColorNonsignificant = value
	case textFieldPlotColorGray:
		m.plotColorGray = value
	case textFieldPlotColorLightGray:
		m.plotColorLightGray = value
	case textFieldPlotColorBlack:
		m.plotColorBlack = value
	case textFieldPlotColorBlue:
		m.plotColorBlue = value
	case textFieldPlotColorRed:
		m.plotColorRed = value
	case textFieldPlotColorNetworkNode:
		m.plotColorNetworkNode = value
	case textFieldPlotScatterEdgecolor:
		m.plotScatterEdgeColor = value
	case textFieldPlotHistEdgecolor:
		m.plotHistEdgeColor = value
	case textFieldPlotKdeColor:
		m.plotKdeColor = value
	case textFieldPlotTopomapColormap:
		m.plotTopomapColormap = value
	case textFieldPlotTopomapSigMaskMarker:
		m.plotTopomapSigMaskMarker = value
	case textFieldPlotTopomapSigMaskMarkerFaceColor:
		m.plotTopomapSigMaskMarkerFaceColor = value
	case textFieldPlotTopomapSigMaskMarkerEdgeColor:
		m.plotTopomapSigMaskMarkerEdgeColor = value
	case textFieldPlotTfrDefaultBaselineWindow:
		m.plotTfrDefaultBaselineWindowSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotPacCmap:
		m.plotPacCmap = value
	case textFieldPlotPacPairs:
		m.plotPacPairsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotConnectivityMeasures:
		// Parse space-separated measure keys (e.g. "aec wpli") and map to the
		// checkbox model used elsewhere in the wizard.
		for i := range connectivityMeasures {
			m.connectivityMeasures[i] = false
		}
		for _, token := range strings.Fields(value) {
			for i, measure := range connectivityMeasures {
				if strings.EqualFold(token, measure.Key) || strings.EqualFold(token, measure.Name) {
					m.connectivityMeasures[i] = true
				}
			}
		}
	case textFieldPlotSpectralMetrics:
		m.plotSpectralMetricsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotBurstsMetrics:
		m.plotBurstsMetricsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotTemporalTimeBins:
		m.plotTemporalTimeBinsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotTemporalTimeLabels:
		m.plotTemporalTimeLabelsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotAsymmetryStat:
		m.plotAsymmetryStatSpec = value
	case textFieldPlotComparisonWindows:
		m.plotComparisonWindowsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotComparisonSegment:
		m.plotComparisonSegment = strings.TrimSpace(value)
	case textFieldPlotComparisonColumn:
		m.plotComparisonColumn = strings.TrimSpace(value)
	case textFieldPlotComparisonValues:
		m.plotComparisonValuesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotComparisonLabels:
		m.plotComparisonLabelsSpec = strings.TrimSpace(value)
	case textFieldPlotComparisonROIs:
		m.plotComparisonROIsSpec = strings.Join(strings.Fields(value), " ")
	// Machine Learning advanced config text fields
	case textFieldMLTarget:
		m.mlTarget = strings.TrimSpace(value)
	case textFieldMLFmriSigContrastName:
		m.mlFmriSigContrastName = strings.TrimSpace(value)
	case textFieldMLFeatureFamilies:
		m.mlFeatureFamiliesSpec = strings.TrimSpace(value)
	case textFieldMLFeatureBands:
		m.mlFeatureBandsSpec = strings.TrimSpace(value)
	case textFieldMLFeatureSegments:
		m.mlFeatureSegmentsSpec = strings.TrimSpace(value)
	case textFieldMLFeatureScopes:
		m.mlFeatureScopesSpec = strings.TrimSpace(value)
	case textFieldMLFeatureStats:
		m.mlFeatureStatsSpec = strings.TrimSpace(value)
	case textFieldMLCovariates:
		m.mlCovariatesSpec = strings.TrimSpace(value)
	case textFieldMLBaselinePredictors:
		m.mlBaselinePredictorsSpec = strings.TrimSpace(value)
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		m.elasticNetAlphaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldElasticNetL1RatioGrid:
		m.elasticNetL1RatioGrid = strings.Join(splitLooseList(value), ",")
	case textFieldRidgeAlphaGrid:
		m.ridgeAlphaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldRfMaxDepthGrid:
		m.rfMaxDepthGrid = strings.Join(splitLooseList(value), ",")
	case textFieldVarianceThresholdGrid:
		m.varianceThresholdGrid = strings.Join(splitLooseList(value), ",")
	// Preprocessing text fields
	case textFieldIcaLabelsToKeep:
		m.icaLabelsToKeep = strings.Join(strings.Fields(value), "")
	}
}

func (m *Model) ensurePlotItemConfig(plotID string) PlotItemConfig {
	if m.plotItemConfigs == nil {
		m.plotItemConfigs = make(map[string]PlotItemConfig)
	}
	cfg, ok := m.plotItemConfigs[plotID]
	if !ok {
		cfg = PlotItemConfig{}
		m.plotItemConfigs[plotID] = cfg
	}
	return cfg
}

func (m *Model) setPlotItemTextFieldValue(plotID string, field plotItemConfigField, value string) {
	cfg := m.ensurePlotItemConfig(plotID)
	switch field {
	case plotItemConfigFieldTfrDefaultBaselineWindow:
		cfg.TfrDefaultBaselineWindowSpec = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldComparisonWindows:
		cfg.ComparisonWindowsSpec = strings.Join(strings.Fields(value), " ")
		featureGroup := m.getFeatureGroupForPlot(plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if len(windows) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.ComparisonWindowsSpec), windows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown window(s): "+strings.Join(unknown, ", "), "warning")
			}
		}
	case plotItemConfigFieldComparisonSegment:
		cfg.ComparisonSegment = strings.TrimSpace(value)
		featureGroup := m.getFeatureGroupForPlot(plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if cfg.ComparisonSegment != "" && len(windows) > 0 {
			unknown := unknownFromList([]string{cfg.ComparisonSegment}, windows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown segment: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldComparisonColumn:
		cfg.ComparisonColumn = strings.TrimSpace(value)
		if cfg.ComparisonColumn != "" && len(m.availableColumns) > 0 {
			unknown := unknownFromList([]string{cfg.ComparisonColumn}, m.availableColumns)
			if len(unknown) > 0 {
				m.ShowToast("Unknown events.tsv column: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldComparisonValues:
		cfg.ComparisonValuesSpec = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldComparisonLabels:
		cfg.ComparisonLabelsSpec = strings.TrimSpace(value)
	case plotItemConfigFieldComparisonROIs:
		cfg.ComparisonROIsSpec = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldTopomapWindow:
		cfg.TopomapWindowsSpec = strings.Join(strings.Fields(value), " ")
		featureGroup := m.getFeatureGroupForPlot(plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if cfg.TopomapWindowsSpec != "" && len(windows) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.TopomapWindowsSpec), windows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown window: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldTfrTopomapActiveWindow:
		cfg.TfrTopomapActiveWindow = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldTfrTopomapWindowSizeMs:
		cfg.TfrTopomapWindowSizeMs = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapWindowCount:
		cfg.TfrTopomapWindowCount = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapLabelXPosition:
		cfg.TfrTopomapLabelXPosition = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapLabelYPositionBottom:
		cfg.TfrTopomapLabelYPositionBottom = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapLabelYPosition:
		cfg.TfrTopomapLabelYPosition = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTitleY:
		cfg.TfrTopomapTitleY = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTitlePad:
		cfg.TfrTopomapTitlePad = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapSubplotsRight:
		cfg.TfrTopomapSubplotsRight = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTemporalHspace:
		cfg.TfrTopomapTemporalHspace = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTemporalWspace:
		cfg.TfrTopomapTemporalWspace = strings.TrimSpace(value)
	case plotItemConfigFieldConnectivityCircleTopFraction:
		cfg.ConnectivityCircleTopFraction = strings.TrimSpace(value)
	case plotItemConfigFieldConnectivityCircleMinLines:
		cfg.ConnectivityCircleMinLines = strings.TrimSpace(value)
	case plotItemConfigFieldConnectivityNetworkTopFraction:
		cfg.ConnectivityNetworkTopFraction = strings.TrimSpace(value)
	case plotItemConfigFieldBehaviorTemporalStatsFeatureFolder:
		cfg.BehaviorTemporalStatsFeatureFolder = strings.TrimSpace(value)
	case plotItemConfigFieldDoseResponseDoseColumn:
		cfg.DoseResponseDoseColumn = strings.TrimSpace(value)
		if cfg.DoseResponseDoseColumn != "" && len(m.availableColumns) > 0 {
			unknown := unknownFromList([]string{cfg.DoseResponseDoseColumn}, m.availableColumns)
			if len(unknown) > 0 {
				m.ShowToast("Unknown events.tsv column: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldDoseResponseResponseColumn:
		cfg.DoseResponseResponseColumn = strings.Join(strings.Fields(value), " ")
		if cfg.DoseResponseResponseColumn != "" && len(m.trialTableFeatureCategories) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.DoseResponseResponseColumn), m.trialTableFeatureCategories)
			if len(unknown) > 0 {
				m.ShowToast("Unknown feature category: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldDoseResponsePainColumn:
		cfg.DoseResponsePainColumn = strings.TrimSpace(value)
		if cfg.DoseResponsePainColumn != "" && len(m.availableColumns) > 0 {
			unknown := unknownFromList([]string{cfg.DoseResponsePainColumn}, m.availableColumns)
			if len(unknown) > 0 {
				m.ShowToast("Unknown events.tsv column: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldDoseResponseSegment:
		cfg.DoseResponseSegment = strings.TrimSpace(value)
		// Segment is a feature window label, not necessarily an events.tsv column.
	default:
		return
	}
	m.plotItemConfigs[plotID] = cfg
}

func unknownFromList(requested []string, available []string) []string {
	availableSet := make(map[string]struct{}, len(available))
	for _, item := range available {
		availableSet[strings.ToLower(strings.TrimSpace(item))] = struct{}{}
	}
	var unknown []string
	for _, item := range requested {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, ok := availableSet[strings.ToLower(item)]; !ok {
			unknown = append(unknown, item)
		}
	}
	return unknown
}

///////////////////////////////////////////////////////////////////
// Advanced Options Helpers
///////////////////////////////////////////////////////////////////

type optionType int

const (
	// Feature Pipeline Advanced Options
	optUseDefaults optionType = iota
	// Features section headers (expand/collapse)
	optFeatGroupConnectivity
	optFeatGroupDirectedConnectivity
	optFeatGroupPAC
	optFeatGroupAperiodic
	optFeatGroupComplexity
	optFeatGroupBursts
	optFeatGroupPower
	optFeatGroupSpectral
	optFeatGroupERP
	optFeatGroupRatios
	optFeatGroupAsymmetry
	optFeatGroupQuality
	optFeatGroupERDS
	optFeatGroupSpatialTransform
	optFeatGroupStorage
	optFeatGroupExecution
	optFeatGroupSourceLoc
	optFeatGroupITPC
	// Behavior section headers (expand/collapse)
	optBehaviorGroupGeneral
	optBehaviorGroupTrialTable
	optBehaviorGroupPainResidual
	optBehaviorGroupCorrelations
	optBehaviorGroupPainSens
	optBehaviorGroupRegression
	optBehaviorGroupModels
	optBehaviorGroupStability
	optBehaviorGroupConsistency
	optBehaviorGroupInfluence
	optBehaviorGroupReport
	optBehaviorGroupCondition
	optBehaviorGroupTemporal
	optBehaviorGroupCluster
	optBehaviorGroupMediation
	optBehaviorGroupModeration
	optBehaviorGroupMixedEffects
	optConnectivity
	optPACPhaseRange
	optPACAmpRange
	optPACMethod
	optPACMinEpochs
	optPACPairs
	optPACSource
	optPACNormalize
	optPACNSurrogates
	optPACAllowHarmonicOverlap
	optPACMaxHarmonic
	optPACHarmonicToleranceHz
	optPACRandomSeed
	optPACComputeWaveformQC
	optPACWaveformOffsetMs
	optAperiodicFmin
	optAperiodicFmax
	optAperiodicPeakZ
	optAperiodicMinR2
	optAperiodicMinPoints
	optAperiodicPsdBandwidth
	optAperiodicMaxRms
	optAperiodicLineNoiseFreq
	optAperiodicLineNoiseWidthHz
	optAperiodicLineNoiseHarmonics
	optAperiodicMinSegmentSec
	optAperiodicModel
	optAperiodicPsdMethod
	optAperiodicExcludeLineNoise
	optPEOrder
	optPEDelay
	optComplexitySignalBasis
	optComplexityMinSegmentSec
	optComplexityMinSamples
	optComplexityZscore
	optBurstThresholdMethod
	optBurstThresholdPercentile
	optBurstThreshold
	optBurstThresholdReference
	optBurstMinTrialsPerCondition
	optBurstMinSegmentSec
	optBurstSkipInvalidSegments
	optBurstMinDuration
	optBurstMinCycles
	optBurstBands
	optERPBaseline
	optERPAllowNoBaseline
	optERPComponents
	optERPSmoothMs
	optERPPeakProminenceUv
	optERPLowpassHz
	optPowerBaselineMode
	optPowerRequireBaseline
	optPowerSubtractEvoked
	optPowerMinTrialsPerCondition
	optPowerExcludeLineNoise
	optPowerLineNoiseFreq
	optPowerLineNoiseWidthHz
	optPowerLineNoiseHarmonics
	optPowerEmitDb
	optSpectralEdge
	optSpectralRatioPairs
	optSpectralIncludeLogRatios
	optSpectralExcludeLineNoise
	optSpectralLineNoiseFreq
	optSpectralLineNoiseWidthHz
	optSpectralLineNoiseHarmonics
	optSpectralPsdMethod
	optSpectralPsdAdaptive
	optSpectralMultitaperAdaptive
	optSpectralFmin
	optSpectralFmax
	optSpectralMinSegmentSec
	optSpectralMinCyclesAtFmin
	optAperiodicSubtractEvoked
	optAsymmetryChannelPairs
	optAsymmetryMinSegmentSec
	optAsymmetryMinCyclesAtFmin
	optAsymmetrySkipInvalidSegments
	optAsymmetryEmitActivationConvention
	optAsymmetryActivationBands
	// Ratios options
	optRatiosMinSegmentSec
	optRatiosMinCyclesAtFmin
	optRatiosSkipInvalidSegments
	// Quality options
	optQualityPsdMethod
	optQualityFmin
	optQualityFmax
	optQualityNFft
	optQualityExcludeLineNoise
	optQualityLineNoiseFreq
	optQualityLineNoiseWidthHz
	optQualityLineNoiseHarmonics
	optQualitySnrSignalBandMin
	optQualitySnrSignalBandMax
	optQualitySnrNoiseBandMin
	optQualitySnrNoiseBandMax
	optQualityMuscleBandMin
	optQualityMuscleBandMax
	// ERDS options
	optERDSUseLogRatio
	optERDSMinBaselinePower
	optERDSMinActivePower
	optERDSMinSegmentSec
	optERDSBands
	optConnOutputLevel
	optConnGraphMetrics
	optConnGraphProp
	optConnWindowLen
	optConnWindowStep
	optConnAECMode
	optConnAECOutput
	optConnForceWithinEpochML
	optConnGranularity
	optConnConditionColumn
	optConnConditionValues
	optConnMinEpochsPerGroup
	optConnMinCyclesPerBand
	optConnWarnNoSpatialTransform
	optConnPhaseEstimator
	optConnMinSegmentSec
	// Directed connectivity options (PSI, DTF, PDC)
	optDirectedConnMeasures
	optDirectedConnOutputLevel
	optDirectedConnMvarOrder
	optDirectedConnNFreqs
	optDirectedConnMinSegSamples
	// Spatial transform options
	optSpatialTransform
	optSpatialTransformLambda2
	optSpatialTransformStiffness
	optMinEpochs
	optFeatAnalysisMode
	optFeatComputeChangeScores
	optFeatSaveTfrWithSidecar
	optFeatNJobsBands
	optFeatNJobsConnectivity
	optFeatNJobsAperiodic
	optFeatNJobsComplexity
	// Source localization options (LCMV, eLORETA)
	optSourceLocMode
	optSourceLocMethod
	optSourceLocSpacing
	optSourceLocParc
	optSourceLocReg
	optSourceLocSnr
	optSourceLocLoose
	optSourceLocDepth
	optSourceLocConnMethod
	optSourceLocSubject
	optSourceLocTrans
	optSourceLocBem
	optSourceLocMindistMm
	optSourceLocFmriEnabled
	optSourceLocFmriStatsMap
	optSourceLocFmriProvenance
	optSourceLocFmriRequireProvenance
	optSourceLocFmriThreshold
	optSourceLocFmriTail
	optSourceLocFmriMinClusterVox
	optSourceLocFmriMinClusterMM3
	optSourceLocFmriMaxClusters
	optSourceLocFmriMaxVoxPerClus
	optSourceLocFmriMaxTotalVox
	optSourceLocFmriRandomSeed
	// BEM/Trans generation options (Docker-based)
	optSourceLocCreateTrans
	optSourceLocCreateBemModel
	optSourceLocCreateBemSolution
	// fMRI GLM contrast builder options
	optSourceLocFmriContrastEnabled
	optSourceLocFmriContrastType
	optSourceLocFmriCondAColumn
	optSourceLocFmriCondAValue
	optSourceLocFmriCondBColumn
	optSourceLocFmriCondBValue
	optSourceLocFmriContrastFormula
	optSourceLocFmriContrastName
	optSourceLocFmriRunsToInclude
	optSourceLocFmriAutoDetectRuns
	optSourceLocFmriHrfModel
	optSourceLocFmriDriftModel
	optSourceLocFmriHighPassHz
	optSourceLocFmriLowPassHz
	optSourceLocFmriStimPhasesToModel
	optSourceLocFmriClusterCorrection
	optSourceLocFmriClusterPThreshold
	optSourceLocFmriOutputType
	optSourceLocFmriResampleToFS
	optSourceLocFmriWindowAName
	optSourceLocFmriWindowATmin
	optSourceLocFmriWindowATmax
	optSourceLocFmriWindowBName
	optSourceLocFmriWindowBTmin
	optSourceLocFmriWindowBTmax
	// ITPC options (condition-based)
	optItpcMethod
	optItpcAllowUnsafeLoo
	optItpcBaselineCorrection
	optItpcConditionColumn
	optItpcConditionValues
	optItpcMinTrialsPerCondition
	optItpcNJobs

	// Storage options
	optSaveSubjectLevelFeatures
	optFeatAlsoSaveCsv
	// Behavior options - General
	optCorrMethod
	optBootstrap
	optNPerm
	optRNGSeed
	optControlTemp
	optControlOrder
	optRunAdjustmentEnabled
	optRunAdjustmentColumn
	optRunAdjustmentIncludeInCorrelations
	optRunAdjustmentMaxDummies
	optFDRAlpha
	// Behavior options - Cluster
	optClusterThreshold
	optClusterMinSize
	optClusterTail
	optClusterConditionColumn
	optClusterConditionValues
	// Behavior options - Mediation
	optMediationBootstrap
	optMediationMaxMediatorsEnabled
	optMediationMaxMediators
	optMediationPermutations
	// Behavior options - Moderation
	optModerationMaxFeaturesEnabled
	optModerationMaxFeatures
	optModerationPermutations
	// Behavior options - Mixed Effects
	optMixedMaxFeatures
	// Behavior options - Condition
	optConditionEffectThreshold
	optConditionFailFast
	optConditionOverwrite
	optConditionCompareColumn
	optConditionCompareWindows
	optConditionCompareValues
	optConditionWindowPrimaryUnit
	optConditionPermutationPrimary
	// Behavior options - Trial table / residual
	optTrialTableFormat
	optTrialTableAddLagFeatures
	optTrialOrderMaxMissingFraction
	optFeatureSummariesEnabled
	optFeatureQCEnabled
	optFeatureQCMaxMissingPct
	optFeatureQCMinVariance
	optFeatureQCCheckWithinRunVariance
	optPainResidualEnabled
	optPainResidualMethod
	optPainResidualPolyDegree
	optPainResidualSplineDfCandidates
	optPainResidualModelCompare
	optPainResidualModelComparePolyDegrees
	optPainResidualBreakpoint
	optPainResidualBreakpointCandidates
	optPainResidualBreakpointQlow
	optPainResidualBreakpointQhigh
	optPainResidualCrossfitEnabled
	optPainResidualCrossfitGroupColumn
	optPainResidualCrossfitNSplits
	optPainResidualCrossfitMethod
	optPainResidualCrossfitSplineKnots
	// Behavior options - General extra
	optRobustCorrelation
	optBehaviorNJobs
	optComputeChangeScores
	optComputeBayesFactors
	optComputeLosoStability
	// Behavior options - Regression
	optRegressionOutcome
	optRegressionIncludeTemperature
	optRegressionTempControl
	optRegressionTempSplineKnots
	optRegressionTempSplineQlow
	optRegressionTempSplineQhigh
	optRegressionIncludeTrialOrder
	optRegressionIncludePrev
	optRegressionIncludeRunBlock
	optRegressionIncludeInteraction
	optRegressionStandardize
	optRegressionPermutations
	optRegressionMaxFeatures
	// Behavior options - Models
	optModelsIncludeTemperature
	optModelsTempControl
	optModelsTempSplineKnots
	optModelsTempSplineQlow
	optModelsTempSplineQhigh
	optModelsIncludeTrialOrder
	optModelsIncludePrev
	optModelsIncludeRunBlock
	optModelsIncludeInteraction
	optModelsStandardize
	optModelsMaxFeatures
	optModelsOutcomeRating
	optModelsOutcomePainResidual
	optModelsOutcomeTemperature
	optModelsOutcomePainBinary
	optModelsFamilyOLS
	optModelsFamilyRobust
	optModelsFamilyQuantile
	optModelsFamilyLogit
	optModelsBinaryOutcome
	// Behavior options - Stability
	optStabilityMethod
	optStabilityOutcome
	optStabilityGroupColumn
	optStabilityPartialTemp
	optStabilityMaxFeatures
	optStabilityAlpha
	// Behavior options - Consistency / Influence
	optConsistencyEnabled
	optInfluenceOutcomeRating
	optInfluenceOutcomePainResidual
	optInfluenceOutcomeTemperature
	optInfluenceMaxFeatures
	optInfluenceIncludeTemperature
	optInfluenceTempControl
	optInfluenceTempSplineKnots
	optInfluenceTempSplineQlow
	optInfluenceTempSplineQhigh
	optInfluenceIncludeTrialOrder
	optInfluenceIncludeRunBlock
	optInfluenceIncludeInteraction
	optInfluenceStandardize
	optInfluenceCooksThreshold
	optInfluenceLeverageThreshold
	// Behavior options - Report
	optReportTopN
	// Behavior options - Correlations / pain sensitivity
	optCorrelationsTargetRating
	optCorrelationsTargetTemperature
	optCorrelationsTargetPainResidual
	optCorrelationsTypes
	optCorrelationsPreferPainResidual
	optCorrelationsUseCrossfitPainResidual
	optCorrelationsMultilevel
	optCorrelationsPrimaryUnit
	optCorrelationsPermutationPrimary
	optCorrelationsTargetColumn
	optGroupLevelBlockPermutation
	// Behavior options - Pain sensitivity / temporal
	optTemporalResolutionMs
	optTemporalTimeMinMs
	optTemporalTimeMaxMs
	optTemporalSmoothMs
	optTemporalTargetColumn
	optTemporalSplitByCondition
	optTemporalConditionColumn
	optTemporalConditionValues
	optTemporalIncludeROIAverages
	optTemporalIncludeTFGrid
	// Temporal feature selection
	optTemporalFeaturePower
	optTemporalFeatureITPC
	optTemporalFeatureERDS
	// ITPC-specific options
	optTemporalITPCBaselineCorrection
	optTemporalITPCBaselineMin
	optTemporalITPCBaselineMax
	// ERDS-specific options
	optTemporalERDSBaselineMin
	optTemporalERDSBaselineMax
	optTemporalERDSMethod
	// TF Heatmap options
	optTemporalTfHeatmapEnabled
	optTemporalTfHeatmapFreqs
	optTemporalTfHeatmapTimeResMs
	// Behavior options - Mixed effects / mediation
	optMixedEffectsType
	optMediationMinEffect
	// Behavior options - Output
	optBehaviorGroupOutput
	optAlsoSaveCsv
	optBehaviorOverwrite
	// Plotting options
	optPlotPNG
	optPlotSVG
	optPlotPDF
	optPlotDPI
	optPlotSaveDPI
	optPlotSharedColorbar
	// Machine Learning options
	optMLNPerm
	optMLInnerSplits
	optMLOuterJobs
	optMLTarget
	optMLFmriSigGroup
	optMLFmriSigMethod
	optMLFmriSigContrastName
	optMLFmriSigSignature
	optMLFmriSigMetric
	optMLFmriSigNormalization
	optMLFmriSigRoundDecimals
	optMLRegressionModel
	optMLClassificationModel
	optMLBinaryThresholdEnabled
	optMLBinaryThreshold
	optMLFeatureFamilies
	optMLFeatureBands
	optMLFeatureSegments
	optMLFeatureScopes
	optMLFeatureStats
	optMLFeatureHarmonization
	optMLCovariates
	optMLBaselinePredictors
	optMLRequireTrialMlSafe
	optMLUncertaintyAlpha
	optMLPermNRepeats
	// Preprocessing group headers (collapsible sections)
	optPrepGroupStages
	optPrepGroupGeneral
	optPrepGroupFiltering
	optPrepGroupPyprep
	optPrepGroupICA
	optPrepGroupEpoching
	// Preprocessing options
	optPrepStageBadChannels
	optPrepStageFiltering
	optPrepStageICA
	optPrepStageEpoching
	optPrepUsePyprep
	optPrepUseIcalabel
	optPrepNJobs
	optPrepMontage
	optPrepResample
	optPrepLFreq
	optPrepHFreq
	optPrepNotch
	optPrepLineFreq
	optPrepChTypes
	optPrepEegReference
	optPrepEogChannels
	optPrepRandomState
	optPrepTaskIsRest
	optPrepZaplineFline
	optPrepFindBreaks
	// PyPREP advanced options
	optPrepRansac
	optPrepRepeats
	optPrepAverageReref
	optPrepFileExtension
	optPrepConsiderPreviousBads
	optPrepOverwriteChansTsv
	optPrepDeleteBreaks
	optPrepBreaksMinLength
	optPrepTStartAfterPrevious
	optPrepTStopBeforeNext
	optPrepRenameAnotDict
	optPrepCustomBadDict
	// ICA options
	optPrepSpatialFilter
	optPrepICAAlgorithm
	optPrepICAComp
	optPrepICALFreq
	optPrepICARejThresh
	optPrepProbThresh
	optPrepKeepMnebidsBads
	optIcaLabelsToKeep
	// Epoching options
	optPrepConditions
	optPrepRejectMethod
	optPrepRunSourceEstimation
	optPrepEpochsTmin
	optPrepEpochsTmax
	optPrepEpochsBaseline
	optPrepEpochsNoBaseline
	optPrepEpochsReject
	optPrepWriteCleanEvents
	optPrepOverwriteCleanEvents
	optPrepCleanEventsStrict
	// Raw-to-BIDS options
	optRawMontage
	optRawLineFreq
	optRawOverwrite
	optRawTrimToFirstVolume
	optRawEventPrefixes
	optRawKeepAnnotations
	// Merge-behavior options
	optMergeEventPrefixes
	optMergeEventTypes
	// fMRI Raw-to-BIDS options
	optFmriRawSession
	optFmriRawRestTask
	optFmriRawIncludeRest
	optFmriRawIncludeFieldmaps
	optFmriRawDicomMode
	optFmriRawOverwrite
	optFmriRawCreateEvents
	optFmriRawEventGranularity
	optFmriRawOnsetReference
	optFmriRawOnsetOffsetS
	optFmriRawDcm2niixPath
	optFmriRawDcm2niixArgs
	// Plotting advanced options (wizard-only)
	optPlotGroupDefaults
	optPlotGroupFonts
	optPlotGroupLayout
	optPlotGroupFigureSizes
	optPlotGroupColors
	optPlotGroupAlpha
	optPlotGroupScatter
	optPlotGroupBar
	optPlotGroupLine
	optPlotGroupHistogram
	optPlotGroupKDE
	optPlotGroupErrorbar
	optPlotGroupText
	optPlotGroupValidation
	optPlotGroupTFRMisc
	optPlotGroupTopomap
	optPlotGroupTFR
	optPlotGroupSizing
	optPlotGroupSelection
	optPlotGroupComparisons
	optPlotBboxInches
	optPlotPadInches
	optPlotFontFamily
	optPlotFontWeight
	optPlotFontSizeSmall
	optPlotFontSizeMedium
	optPlotFontSizeLarge
	optPlotFontSizeTitle
	optPlotFontSizeAnnotation
	optPlotFontSizeLabel
	optPlotFontSizeYLabel
	optPlotFontSizeSuptitle
	optPlotFontSizeFigureTitle
	optPlotLayoutTightRect
	optPlotLayoutTightRectMicrostate
	optPlotGridSpecWidthRatios
	optPlotGridSpecHeightRatios
	optPlotGridSpecHspace
	optPlotGridSpecWspace
	optPlotGridSpecLeft
	optPlotGridSpecRight
	optPlotGridSpecTop
	optPlotGridSpecBottom
	optPlotFigureSizeStandard
	optPlotFigureSizeMedium
	optPlotFigureSizeSmall
	optPlotFigureSizeSquare
	optPlotFigureSizeWide
	optPlotFigureSizeTFR
	optPlotFigureSizeTopomap
	optPlotColorPain
	optPlotColorNonpain
	optPlotColorSignificant
	optPlotColorNonsignificant
	optPlotColorGray
	optPlotColorLightGray
	optPlotColorBlack
	optPlotColorBlue
	optPlotColorRed
	optPlotColorNetworkNode
	optPlotAlphaGrid
	optPlotAlphaFill
	optPlotAlphaCI
	optPlotAlphaCILine
	optPlotAlphaTextBox
	optPlotAlphaViolinBody
	optPlotAlphaRidgeFill
	optPlotScatterMarkerSizeSmall
	optPlotScatterMarkerSizeLarge
	optPlotScatterMarkerSizeDefault
	optPlotScatterAlpha
	optPlotScatterEdgecolor
	optPlotScatterEdgewidth
	optPlotBarAlpha
	optPlotBarWidth
	optPlotBarCapsize
	optPlotBarCapsizeLarge
	optPlotLineWidthThin
	optPlotLineWidthStandard
	optPlotLineWidthThick
	optPlotLineWidthBold
	optPlotLineAlphaStandard
	optPlotLineAlphaDim
	optPlotLineAlphaZeroLine
	optPlotLineAlphaFitLine
	optPlotLineAlphaDiagonal
	optPlotLineAlphaReference
	optPlotLineRegressionWidth
	optPlotLineResidualWidth
	optPlotLineQQWidth
	optPlotHistBins
	optPlotHistBinsBehavioral
	optPlotHistBinsResidual
	optPlotHistBinsTFR
	optPlotHistEdgecolor
	optPlotHistEdgewidth
	optPlotHistAlpha
	optPlotHistAlphaResidual
	optPlotHistAlphaTFR
	optPlotKdePoints
	optPlotKdeColor
	optPlotKdeLinewidth
	optPlotKdeAlpha
	optPlotErrorbarMarkersize
	optPlotErrorbarCapsize
	optPlotErrorbarCapsizeLarge
	optPlotTextStatsX
	optPlotTextStatsY
	optPlotTextPvalueX
	optPlotTextPvalueY
	optPlotTextBootstrapX
	optPlotTextBootstrapY
	optPlotTextChannelAnnotationX
	optPlotTextChannelAnnotationY
	optPlotTextTitleY
	optPlotTextResidualQcTitleY
	optPlotValidationMinBinsForCalibration
	optPlotValidationMaxBinsForCalibration
	optPlotValidationSamplesPerBin
	optPlotValidationMinRoisForFDR
	optPlotValidationMinPvaluesForFDR
	optPlotTfrDefaultBaselineWindow
	optPlotTopomapContours
	optPlotTopomapColormap
	optPlotTopomapColorbarFraction
	optPlotTopomapColorbarPad
	optPlotTopomapDiffAnnotation
	optPlotTopomapAnnotateDescriptive
	optPlotTopomapSigMaskMarker
	optPlotTopomapSigMaskMarkerFaceColor
	optPlotTopomapSigMaskMarkerEdgeColor
	optPlotTopomapSigMaskLinewidth
	optPlotTopomapSigMaskMarkersize
	optPlotTFRLogBase
	optPlotTFRPercentageMultiplier
	optPlotTFRTopomapWindowSizeMs
	optPlotTFRTopomapWindowCount
	optPlotTFRTopomapLabelXPosition
	optPlotTFRTopomapLabelYPositionBottom
	optPlotTFRTopomapLabelYPosition
	optPlotTFRTopomapTitleY
	optPlotTFRTopomapTitlePad
	optPlotTFRTopomapSubplotsRight
	optPlotTFRTopomapTemporalHspace
	optPlotTFRTopomapTemporalWspace
	optPlotRoiWidthPerBand
	optPlotRoiWidthPerMetric
	optPlotRoiHeightPerRoi
	optPlotPowerWidthPerBand
	optPlotPowerHeightPerSegment
	optPlotItpcWidthPerBin
	optPlotItpcHeightPerBand
	optPlotItpcWidthPerBandBox
	optPlotItpcHeightBox
	optPlotPacCmap
	optPlotPacWidthPerRoi
	optPlotPacHeightBox
	optPlotAperiodicWidthPerColumn
	optPlotAperiodicHeightPerRow
	optPlotAperiodicNPerm
	optPlotComplexityWidthPerMeasure
	optPlotComplexityHeightPerSegment
	optPlotConnectivityWidthPerCircle
	optPlotConnectivityWidthPerBand
	optPlotConnectivityHeightPerMeasure
	optPlotConnectivityCircleTopFraction
	optPlotConnectivityCircleMinLines
	optPlotConnectivityNetworkTopFraction
	optPlotPacPairs
	optPlotConnectivityMeasures
	optPlotSpectralMetrics
	optPlotBurstsMetrics
	optPlotAsymmetryStat
	optPlotTemporalTimeBins
	optPlotTemporalTimeLabels
	// Plotting comparisons (global)
	optPlotCompareWindows
	optPlotComparisonWindows
	optPlotCompareColumns
	optPlotComparisonSegment
	optPlotComparisonColumn
	optPlotComparisonValues
	optPlotComparisonLabels
	optPlotComparisonROIs
	optPlotOverwrite
	// TFR parameters (features pipeline)
	optFeatGroupTFR
	optTfrFreqMin
	optTfrFreqMax
	optTfrNFreqs
	optTfrMinCycles
	optTfrNCyclesFactor
	optTfrWorkers
	// Band envelope + IAF (features pipeline)
	optBandEnvelopePadSec
	optBandEnvelopePadCycles
	optIAFEnabled
	optIAFAlphaWidthHz
	optIAFSearchRangeMin
	optIAFSearchRangeMax
	optIAFMinProminence
	optIAFRois
	optIAFMinCyclesAtFmin
	optIAFMinBaselineSec
	optIAFAllowFullFallback
	optIAFAllowAllChannelsFallback
	// Machine Learning model hyperparameters
	optElasticNetAlphaGrid
	optElasticNetL1RatioGrid
	optRidgeAlphaGrid
	optRfNEstimators
	optRfMaxDepthGrid
	optVarianceThresholdGrid
	// fMRI preprocessing group headers (collapsible sections)
	optFmriGroupRuntime
	optFmriGroupOutput
	optFmriGroupPerformance
	optFmriGroupAnatomical
	optFmriGroupBold
	optFmriGroupQc
	optFmriGroupDenoising
	optFmriGroupSurface
	optFmriGroupMultiecho
	optFmriGroupRepro
	optFmriGroupValidation
	optFmriGroupAdvanced
	// fMRI preprocessing (fMRIPrep-style)
	optFmriEngine
	optFmriFmriprepImage
	optFmriFmriprepOutputDir
	optFmriFmriprepWorkDir
	optFmriFreesurferLicenseFile
	optFmriFreesurferSubjectsDir
	optFmriOutputSpaces
	optFmriIgnore
	optFmriBidsFilterFile
	optFmriUseAroma
	optFmriSkipBidsValidation
	optFmriStopOnFirstCrash
	optFmriCleanWorkdir
	optFmriSkipReconstruction
	optFmriMemMb
	optFmriExtraArgs
	// Additional fMRIPrep options
	optFmriNThreads
	optFmriOmpNThreads
	optFmriLowMem
	optFmriLongitudinal
	optFmriCiftiOutput
	optFmriSkullStripTemplate
	optFmriSkullStripFixedSeed
	optFmriRandomSeed
	optFmriDummyScans
	optFmriBold2T1wInit
	optFmriBold2T1wDof
	optFmriSliceTimeRef
	optFmriFdSpikeThreshold
	optFmriDvarsSpikeThreshold
	optFmriMeOutputEchos
	optFmriMedialSurfaceNan
	optFmriNoMsm
	optFmriLevel
	optFmriTaskId
	// fMRI analysis (first-level GLM + contrasts) group headers
	optFmriAnalysisGroupInput
	optFmriAnalysisGroupContrast
	optFmriAnalysisGroupGLM
	optFmriAnalysisGroupConfounds
	optFmriAnalysisGroupOutput
	optFmriAnalysisGroupPlotting
	// fMRI analysis options
	optFmriAnalysisInputSource
	optFmriAnalysisFmriprepSpace
	optFmriAnalysisRequireFmriprep
	optFmriAnalysisRuns
	optFmriAnalysisContrastType
	optFmriAnalysisCondAColumn
	optFmriAnalysisCondAValue
	optFmriAnalysisCondBColumn
	optFmriAnalysisCondBValue
	optFmriAnalysisContrastName
	optFmriAnalysisFormula
	optFmriAnalysisHrfModel
	optFmriAnalysisDriftModel
	optFmriAnalysisHighPassHz
	optFmriAnalysisLowPassHz
	optFmriAnalysisSmoothingFwhm
	optFmriAnalysisEventsToModel
	optFmriAnalysisStimPhasesToModel
	optFmriAnalysisConfoundsStrategy
	optFmriAnalysisWriteDesignMatrix
	optFmriAnalysisOutputType
	optFmriAnalysisOutputDir
	optFmriAnalysisResampleToFS
	optFmriAnalysisFreesurferDir
	// fMRI analysis plotting/report options
	optFmriAnalysisPlotsEnabled
	optFmriAnalysisPlotHTML
	optFmriAnalysisPlotSpace
	optFmriAnalysisPlotThresholdMode
	optFmriAnalysisPlotZThreshold
	optFmriAnalysisPlotFdrQ
	optFmriAnalysisPlotClusterMinVoxels
	optFmriAnalysisPlotVmaxMode
	optFmriAnalysisPlotVmaxManual
	optFmriAnalysisPlotIncludeUnthresholded
	optFmriAnalysisPlotFormatPNG
	optFmriAnalysisPlotFormatSVG
	optFmriAnalysisPlotTypeSlices
	optFmriAnalysisPlotTypeGlass
	optFmriAnalysisPlotTypeHist
	optFmriAnalysisPlotTypeClusters
	optFmriAnalysisPlotEffectSize
	optFmriAnalysisPlotStandardError
	optFmriAnalysisPlotMotionQC
	optFmriAnalysisPlotCarpetQC
	optFmriAnalysisPlotTSNRQC
	optFmriAnalysisPlotDesignQC
	optFmriAnalysisPlotEmbedImages
	optFmriAnalysisPlotSignatures
	optFmriAnalysisSignatureDir
	// fMRI analysis trial-wise signatures options
	optFmriTrialSigGroup
	optFmriTrialSigMethod
	optFmriTrialSigIncludeOtherEvents
	optFmriTrialSigMaxTrialsPerRun
	optFmriTrialSigFixedEffectsWeighting
	optFmriTrialSigWriteTrialBetas
	optFmriTrialSigWriteTrialVariances
	optFmriTrialSigWriteConditionBetas
	optFmriTrialSigSignatureNPS
	optFmriTrialSigSignatureSIIPS1
	optFmriTrialSigLssOtherRegressors
	optFmriTrialSigGroupColumn
	optFmriTrialSigGroupValues
	optFmriTrialSigScopeStimPhases
	optFmriTrialSigGroupScope
	// System/global settings
	optSystemNJobs
	optSystemStrictMode
	optLoggingLevel
)

const (
	expandedNone                            = -1
	expandedConnectivityMeasures            = 0
	expandedConditionCompareColumn          = 1
	expandedConditionCompareValues          = 2
	expandedConditionCompareWindows         = 3
	expandedTemporalConditionColumn         = 4
	expandedTemporalConditionValues         = 5
	expandedClusterConditionColumn          = 6
	expandedClusterConditionValues          = 7
	expandedPlotComparisonColumn            = 8
	expandedPlotComparisonValues            = 9
	expandedPlotComparisonWindows           = 10
	expandedPlotComparisonROIs              = 11
	expandedDirectedConnMeasures            = 12
	expandedRunAdjustmentColumn             = 13
	expandedCorrelationsTargetColumn        = 14
	expandedItpcConditionColumn             = 15
	expandedItpcConditionValues             = 16
	expandedFmriCondAColumn                 = 17
	expandedFmriCondAValue                  = 18
	expandedFmriCondBColumn                 = 19
	expandedFmriCondBValue                  = 20
	expandedBehaviorScatterFeatures         = 21
	expandedBehaviorScatterColumns          = 22
	expandedBehaviorScatterAggregation      = 23
	expandedBehaviorScatterSegment          = 24
	expandedTemporalTargetColumn            = 25
	expandedTemporalTopomapsFeatureDir      = 26
	expandedConnConditionColumn             = 27
	expandedConnConditionValues             = 28
	expandedMLTargetColumn                  = 29
	expandedFmriAnalysisCondAColumn         = 30
	expandedFmriAnalysisCondAValue          = 31
	expandedFmriAnalysisCondBColumn         = 32
	expandedFmriAnalysisCondBValue          = 33
	expandedFmriAnalysisStimPhases          = 34
	expandedFmriTrialSigGroupColumn         = 35
	expandedFmriTrialSigGroupValues         = 36
	expandedSourceLocFmriStimPhases         = 37
	expandedPainResidualCrossfitGroupColumn = 38
	expandedStabilityGroupColumn            = 39
	expandedFmriTrialSigStimPhases          = 40
)

// getFeaturesOptions returns the active advanced options for the features pipeline
func (m Model) getFeaturesOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	if m.isCategorySelected("connectivity") {
		options = append(options, optFeatGroupConnectivity)
		if m.featGroupConnectivityExpanded {
			options = append(options, optConnectivity, optConnOutputLevel, optConnGranularity)
			if m.connGranularity == 1 {
				options = append(options, optConnConditionColumn, optConnConditionValues)
			}
			options = append(
				options,
				optConnPhaseEstimator,
				optConnMinEpochsPerGroup,
				optConnMinCyclesPerBand,
				optConnMinSegmentSec,
				optConnWarnNoSpatialTransform,
				optConnGraphMetrics,
				optConnGraphProp,
				optConnWindowLen,
				optConnWindowStep,
				optConnAECMode,
				optConnAECOutput,
				optConnForceWithinEpochML,
			)
		}
	}

	if m.isCategorySelected("directedconnectivity") {
		options = append(options, optFeatGroupDirectedConnectivity)
		if m.featGroupDirectedConnExpanded {
			options = append(options, optDirectedConnMeasures, optDirectedConnOutputLevel, optDirectedConnMvarOrder, optDirectedConnNFreqs, optDirectedConnMinSegSamples)
		}
	}

	if m.isCategorySelected("pac") {
		options = append(options, optFeatGroupPAC)
		if m.featGroupPACExpanded {
			options = append(options, optPACPhaseRange, optPACAmpRange, optPACMethod, optPACMinEpochs, optPACPairs,
				optPACSource, optPACNormalize, optPACNSurrogates, optPACAllowHarmonicOverlap, optPACMaxHarmonic, optPACHarmonicToleranceHz, optPACRandomSeed, optPACComputeWaveformQC, optPACWaveformOffsetMs)
		}
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optFeatGroupAperiodic)
		if m.featGroupAperiodicExpanded {
			options = append(
				options,
				optAperiodicModel,
				optAperiodicPsdMethod,
				optAperiodicFmin,
				optAperiodicFmax,
				optAperiodicPsdBandwidth,
				optAperiodicMinSegmentSec,
				optAperiodicExcludeLineNoise,
				optAperiodicLineNoiseFreq,
				optAperiodicLineNoiseWidthHz,
				optAperiodicLineNoiseHarmonics,
				optAperiodicPeakZ,
				optAperiodicMinR2,
				optAperiodicMinPoints,
				optAperiodicMaxRms,
				optAperiodicSubtractEvoked,
			)
		}
	}
	if m.isCategorySelected("complexity") {
		options = append(options, optFeatGroupComplexity)
		if m.featGroupComplexityExpanded {
			options = append(options, optPEOrder, optPEDelay, optComplexitySignalBasis, optComplexityMinSegmentSec, optComplexityMinSamples, optComplexityZscore)
		}
	}
	if m.isCategorySelected("erp") {
		options = append(options, optFeatGroupERP)
		if m.featGroupERPExpanded {
			options = append(options, optERPBaseline, optERPAllowNoBaseline, optERPComponents, optERPSmoothMs, optERPPeakProminenceUv, optERPLowpassHz)
		}
	}
	if m.isCategorySelected("bursts") {
		options = append(options, optFeatGroupBursts)
		if m.featGroupBurstsExpanded {
			options = append(
				options,
				optBurstThresholdMethod,
				optBurstThresholdPercentile,
				optBurstThreshold,
				optBurstThresholdReference,
				optBurstMinTrialsPerCondition,
				optBurstMinSegmentSec,
				optBurstSkipInvalidSegments,
				optBurstMinDuration,
				optBurstMinCycles,
				optBurstBands,
			)
		}
	}
	if m.isCategorySelected("power") {
		options = append(options, optFeatGroupPower)
		if m.featGroupPowerExpanded {
			options = append(
				options,
				optPowerRequireBaseline,
				optPowerBaselineMode,
				optPowerSubtractEvoked,
				optPowerMinTrialsPerCondition,
				optPowerExcludeLineNoise,
				optPowerLineNoiseFreq,
				optPowerLineNoiseWidthHz,
				optPowerLineNoiseHarmonics,
				optPowerEmitDb,
			)
		}
	}
	if m.isCategorySelected("spectral") {
		options = append(options, optFeatGroupSpectral)
		if m.featGroupSpectralExpanded {
			options = append(
				options,
				optSpectralIncludeLogRatios,
				optSpectralPsdMethod,
				optSpectralPsdAdaptive,
				optSpectralMultitaperAdaptive,
				optSpectralFmin,
				optSpectralFmax,
				optSpectralMinSegmentSec,
				optSpectralMinCyclesAtFmin,
				optSpectralExcludeLineNoise,
				optSpectralLineNoiseFreq,
				optSpectralLineNoiseWidthHz,
				optSpectralLineNoiseHarmonics,
				optSpectralEdge,
			)
		}
	}
	if m.isCategorySelected("ratios") {
		options = append(options, optFeatGroupRatios)
		if m.featGroupRatiosExpanded {
			options = append(options, optSpectralRatioPairs, optRatiosMinSegmentSec, optRatiosMinCyclesAtFmin, optRatiosSkipInvalidSegments)
		}
	}
	if m.isCategorySelected("asymmetry") {
		options = append(options, optFeatGroupAsymmetry)
		if m.featGroupAsymmetryExpanded {
			options = append(
				options,
				optAsymmetryChannelPairs,
				optAsymmetryMinSegmentSec,
				optAsymmetryMinCyclesAtFmin,
				optAsymmetrySkipInvalidSegments,
				optAsymmetryEmitActivationConvention,
				optAsymmetryActivationBands,
			)
		}
	}
	if m.isCategorySelected("quality") {
		options = append(options, optFeatGroupQuality)
		if m.featGroupQualityExpanded {
			options = append(options, optQualityPsdMethod, optQualityFmin, optQualityFmax, optQualityNFft,
				optQualityExcludeLineNoise, optQualityLineNoiseFreq, optQualityLineNoiseWidthHz, optQualityLineNoiseHarmonics,
				optQualitySnrSignalBandMin, optQualitySnrSignalBandMax, optQualitySnrNoiseBandMin, optQualitySnrNoiseBandMax,
				optQualityMuscleBandMin, optQualityMuscleBandMax)
		}
	}
	if m.isCategorySelected("erds") {
		options = append(options, optFeatGroupERDS)
		if m.featGroupERDSExpanded {
			options = append(options, optERDSUseLogRatio, optERDSMinBaselinePower, optERDSMinActivePower, optERDSMinSegmentSec, optERDSBands)
		}
	}

	// Spatial transform (for volume conduction reduction) - useful when connectivity is selected
	if m.isCategorySelected("connectivity") {
		options = append(options, optFeatGroupSpatialTransform)
		if m.featGroupSpatialTransformExpanded {
			options = append(options, optSpatialTransform, optSpatialTransformLambda2, optSpatialTransformStiffness)
		}
	}

	// Source localization (LCMV, eLORETA)
	if m.isCategorySelected("sourcelocalization") {
		options = append(options, optFeatGroupSourceLoc)
		if m.featGroupSourceLocExpanded {
			// Mode selection: EEG-only vs fMRI-informed
			options = append(options, optSourceLocMode)
			options = append(options, optSourceLocMethod, optSourceLocSpacing, optSourceLocParc)
			// Show method-specific options based on selected method
			if m.sourceLocMethod == 0 { // LCMV
				options = append(options, optSourceLocReg)
			} else { // eLORETA
				options = append(options, optSourceLocSnr, optSourceLocLoose, optSourceLocDepth)
			}
			options = append(options, optSourceLocConnMethod)

			// fMRI-informed mode (mode == 1) requires additional paths
			if m.sourceLocMode == 1 {
				// BEM/Trans generation options (Docker-based)
				// Note: FS License is configured in global paths, subject is from step 1
				options = append(options, optSourceLocCreateTrans, optSourceLocCreateBemModel, optSourceLocCreateBemSolution)
				// If not auto-creating, user must provide paths
				if !m.sourceLocCreateTrans {
					options = append(options, optSourceLocTrans)
				}
				if !m.sourceLocCreateBemSolution {
					options = append(options, optSourceLocBem)
				}
				options = append(options, optSourceLocMindistMm)
				options = append(options, optSourceLocFmriEnabled)
				if m.sourceLocFmriEnabled || strings.TrimSpace(m.sourceLocFmriStatsMap) != "" {
					options = append(options,
						optSourceLocFmriStatsMap,
						optSourceLocFmriProvenance,
						optSourceLocFmriRequireProvenance,
						optSourceLocFmriThreshold,
						optSourceLocFmriTail,
						optSourceLocFmriMaxClusters,
						optSourceLocFmriMaxVoxPerClus,
						optSourceLocFmriMaxTotalVox,
						optSourceLocFmriRandomSeed,
					)
					options = append(options, optSourceLocFmriMinClusterMM3)
					if m.sourceLocFmriMinClusterMM3 <= 0 {
						options = append(options, optSourceLocFmriMinClusterVox)
					}
					options = append(options, optSourceLocFmriContrastEnabled)
					if m.sourceLocFmriContrastEnabled {
						options = append(options, optSourceLocFmriContrastType)
						// Show condition fields based on contrast type
						if m.sourceLocFmriContrastType == 3 { // custom formula
							options = append(options, optSourceLocFmriContrastFormula)
						} else {
							options = append(options, optSourceLocFmriCondAColumn, optSourceLocFmriCondAValue)
							options = append(options, optSourceLocFmriCondBColumn, optSourceLocFmriCondBValue)
						}
						options = append(options, optSourceLocFmriContrastName)
						options = append(options, optSourceLocFmriAutoDetectRuns)
						if !m.sourceLocFmriAutoDetectRuns {
							options = append(options, optSourceLocFmriRunsToInclude)
						}
						options = append(options, optSourceLocFmriHrfModel, optSourceLocFmriDriftModel, optSourceLocFmriStimPhasesToModel)
						options = append(options, optSourceLocFmriHighPassHz, optSourceLocFmriLowPassHz)
						options = append(options, optSourceLocFmriClusterCorrection)
						if m.sourceLocFmriClusterCorrection {
							options = append(options, optSourceLocFmriClusterPThreshold)
						}
						options = append(options, optSourceLocFmriOutputType, optSourceLocFmriResampleToFS)
						// fMRI-specific time windows
						options = append(options, optSourceLocFmriWindowAName, optSourceLocFmriWindowATmin, optSourceLocFmriWindowATmax)
						options = append(options, optSourceLocFmriWindowBName, optSourceLocFmriWindowBTmin, optSourceLocFmriWindowBTmax)
					}
				}
			}
		}
	}

	// ITPC options (condition-based ITPC for avoiding pseudo-replication)
	if m.isCategorySelected("itpc") {
		options = append(options, optFeatGroupITPC)
		if m.featGroupITPCExpanded {
			options = append(options, optItpcMethod, optItpcAllowUnsafeLoo, optItpcBaselineCorrection)
			// Show condition-based options only when method is "condition" (method index 3)
			if m.itpcMethod == 3 {
				options = append(options, optItpcConditionColumn, optItpcConditionValues, optItpcMinTrialsPerCondition)
			}
		}
	}

	// TFR settings (always available for features that use time-frequency)
	options = append(options, optFeatGroupTFR)
	if m.featGroupTFRExpanded {
		options = append(
			options,
			optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrNCyclesFactor, optTfrWorkers,
			optBandEnvelopePadSec, optBandEnvelopePadCycles,
			optIAFEnabled,
		)
		if m.iafEnabled {
			options = append(
				options,
				optIAFAlphaWidthHz,
				optIAFSearchRangeMin,
				optIAFSearchRangeMax,
				optIAFMinProminence,
				optIAFRois,
				optIAFMinCyclesAtFmin,
				optIAFMinBaselineSec,
				optIAFAllowFullFallback,
				optIAFAllowAllChannelsFallback,
			)
		}
	}

	options = append(options, optFeatGroupStorage)
	if m.featGroupStorageExpanded {
		options = append(options, optSaveSubjectLevelFeatures, optFeatAlsoSaveCsv)
	}

	options = append(options, optFeatGroupExecution)
	if m.featGroupExecutionExpanded {
		options = append(
			options,
			optMinEpochs,
			optFeatAnalysisMode,
			optFeatComputeChangeScores,
			optFeatSaveTfrWithSidecar,
			optFeatNJobsBands,
			optFeatNJobsConnectivity,
			optFeatNJobsAperiodic,
			optFeatNJobsComplexity,
		)
		if m.isCategorySelected("itpc") {
			options = append(options, optItpcNJobs)
		}
	}

	return options
}

// getPreprocessingOptions returns advanced options for preprocessing with collapsible groups
func (m Model) getPreprocessingOptions() []optionType {
	isFull := m.modeIndex == 0 || m.modeOptions[m.modeIndex] == "full"
	options := []optionType{optUseDefaults}

	// Stage Selection group (only show if not in full mode)
	if !isFull {
		options = append(options, optPrepGroupStages)
		if m.prepGroupStagesExpanded {
			options = append(options,
				optPrepStageBadChannels,
				optPrepStageFiltering,
				optPrepStageICA,
				optPrepStageEpoching,
			)
		}
	}

	// General Settings group (montage, jobs, etc.)
	options = append(options, optPrepGroupGeneral)
	if m.prepGroupGeneralExpanded {
		options = append(options,
			optPrepMontage,
			optPrepChTypes,
			optPrepEegReference,
			optPrepEogChannels,
			optPrepRandomState,
			optPrepTaskIsRest,
			optPrepNJobs,
			optPrepUsePyprep,
			optPrepUseIcalabel,
		)
	}

	// Filtering group
	if isFull || m.prepStageSelected[1] {
		options = append(options, optPrepGroupFiltering)
		if m.prepGroupFilteringExpanded {
			options = append(options,
				optPrepResample,
				optPrepLFreq,
				optPrepHFreq,
				optPrepNotch,
				optPrepLineFreq,
				optPrepZaplineFline,
				optPrepFindBreaks,
			)
		}
	}

	// PyPREP Advanced group (part of bad channel detection if enabled)
	if (isFull || m.prepStageSelected[0]) && m.prepUsePyprep {
		options = append(options, optPrepGroupPyprep)
		if m.prepGroupPyprepExpanded {
			options = append(options,
				optPrepRansac,
				optPrepRepeats,
				optPrepAverageReref,
				optPrepFileExtension,
				optPrepConsiderPreviousBads,
				optPrepOverwriteChansTsv,
				optPrepDeleteBreaks,
				optPrepBreaksMinLength,
				optPrepTStartAfterPrevious,
				optPrepTStopBeforeNext,
				optPrepRenameAnotDict,
				optPrepCustomBadDict,
			)
		}
	}

	// ICA group
	if isFull || m.prepStageSelected[2] {
		options = append(options, optPrepGroupICA)
		if m.prepGroupICAExpanded {
			options = append(options,
				optPrepSpatialFilter,
				optPrepICAAlgorithm,
				optPrepICAComp,
				optPrepICALFreq,
				optPrepICARejThresh,
				optPrepProbThresh,
				optPrepKeepMnebidsBads,
				optIcaLabelsToKeep,
			)
		}
	}

	// Epoching group
	if isFull || m.prepStageSelected[3] {
		options = append(options, optPrepGroupEpoching)
		if m.prepGroupEpochingExpanded {
			options = append(options,
				optPrepConditions,
				optPrepEpochsTmin,
				optPrepEpochsTmax,
				optPrepEpochsNoBaseline,
				optPrepEpochsBaseline,
				optPrepEpochsReject,
				optPrepRejectMethod,
				optPrepRunSourceEstimation,
				optPrepWriteCleanEvents,
				optPrepOverwriteCleanEvents,
				optPrepCleanEventsStrict,
			)
		}
	}

	return options
}

func (m Model) getFmriPreprocessingOptions() []optionType {
	options := []optionType{optUseDefaults}

	// Runtime group
	options = append(options, optFmriGroupRuntime)
	if m.fmriGroupRuntimeExpanded {
		options = append(options, optFmriEngine, optFmriFmriprepImage)
	}

	// Output group
	options = append(options, optFmriGroupOutput)
	if m.fmriGroupOutputExpanded {
		options = append(options, optFmriOutputSpaces, optFmriIgnore, optFmriLevel, optFmriCiftiOutput, optFmriTaskId)
	}

	// Performance group
	options = append(options, optFmriGroupPerformance)
	if m.fmriGroupPerformanceExpanded {
		options = append(options, optFmriNThreads, optFmriOmpNThreads, optFmriMemMb, optFmriLowMem)
	}

	// Anatomical group
	options = append(options, optFmriGroupAnatomical)
	if m.fmriGroupAnatomicalExpanded {
		options = append(options, optFmriSkipReconstruction, optFmriLongitudinal, optFmriSkullStripTemplate, optFmriSkullStripFixedSeed)
	}

	// BOLD processing group
	options = append(options, optFmriGroupBold)
	if m.fmriGroupBoldExpanded {
		options = append(options, optFmriBold2T1wInit, optFmriBold2T1wDof, optFmriSliceTimeRef, optFmriDummyScans)
	}

	// Quality control group
	options = append(options, optFmriGroupQc)
	if m.fmriGroupQcExpanded {
		options = append(options, optFmriFdSpikeThreshold, optFmriDvarsSpikeThreshold)
	}

	// Denoising group
	options = append(options, optFmriGroupDenoising)
	if m.fmriGroupDenoisingExpanded {
		options = append(options, optFmriUseAroma)
	}

	// Surface group
	options = append(options, optFmriGroupSurface)
	if m.fmriGroupSurfaceExpanded {
		options = append(options, optFmriMedialSurfaceNan, optFmriNoMsm)
	}

	// Multi-echo group
	options = append(options, optFmriGroupMultiecho)
	if m.fmriGroupMultiechoExpanded {
		options = append(options, optFmriMeOutputEchos)
	}

	// Reproducibility group
	options = append(options, optFmriGroupRepro)
	if m.fmriGroupReproExpanded {
		options = append(options, optFmriRandomSeed)
	}

	// Validation group
	options = append(options, optFmriGroupValidation)
	if m.fmriGroupValidationExpanded {
		options = append(options, optFmriSkipBidsValidation, optFmriStopOnFirstCrash, optFmriCleanWorkdir)
	}

	// Advanced group
	options = append(options, optFmriGroupAdvanced)
	if m.fmriGroupAdvancedExpanded {
		options = append(options, optFmriExtraArgs)
	}

	return options
}

func (m Model) getFmriAnalysisOptions() []optionType {
	options := []optionType{optUseDefaults}

	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}
	isFirstLevel := mode == "" || mode == "first-level"

	options = append(options, optFmriAnalysisGroupInput)
	if m.fmriAnalysisGroupInputExpanded {
		options = append(options,
			optFmriAnalysisInputSource,
			optFmriAnalysisFmriprepSpace,
			optFmriAnalysisRequireFmriprep,
			optFmriAnalysisRuns,
		)
	}

	options = append(options, optFmriAnalysisGroupContrast)
	if m.fmriAnalysisGroupContrastExpanded {
		if isFirstLevel {
			options = append(options, optFmriAnalysisContrastType)
		}
		options = append(options,
			optFmriAnalysisCondAColumn,
			optFmriAnalysisCondAValue,
			optFmriAnalysisCondBColumn,
			optFmriAnalysisCondBValue,
			optFmriAnalysisContrastName,
		)
		if isFirstLevel && m.fmriAnalysisContrastType == 1 {
			options = append(options, optFmriAnalysisFormula)
		}
	}

	options = append(options, optFmriAnalysisGroupGLM)
	if m.fmriAnalysisGroupGLMExpanded {
		options = append(options,
			optFmriAnalysisHrfModel,
			optFmriAnalysisDriftModel,
			optFmriAnalysisHighPassHz,
			optFmriAnalysisLowPassHz,
			optFmriAnalysisSmoothingFwhm,
		)
	}

	options = append(options, optFmriAnalysisGroupConfounds)
	if m.fmriAnalysisGroupConfoundsExpanded {
		if isFirstLevel {
			options = append(options, optFmriAnalysisEventsToModel, optFmriAnalysisStimPhasesToModel)
		}
		options = append(options, optFmriAnalysisConfoundsStrategy)
		if isFirstLevel {
			options = append(options, optFmriAnalysisWriteDesignMatrix)
		}
	}

	options = append(options, optFmriAnalysisGroupOutput)
	if m.fmriAnalysisGroupOutputExpanded {
		if isFirstLevel {
			options = append(options,
				optFmriAnalysisOutputType,
				optFmriAnalysisOutputDir,
				optFmriAnalysisResampleToFS,
			)
			if m.fmriAnalysisResampleToFS {
				options = append(options, optFmriAnalysisFreesurferDir)
			}
		} else {
			options = append(options, optFmriAnalysisOutputDir)
		}
	}

	if isFirstLevel {
		options = append(options, optFmriAnalysisGroupPlotting)
		if m.fmriAnalysisGroupPlottingExpanded {
			options = append(options, optFmriAnalysisPlotsEnabled, optFmriAnalysisPlotHTML, optFmriAnalysisPlotSpace)

			// Thresholding
			options = append(options, optFmriAnalysisPlotThresholdMode, optFmriAnalysisPlotZThreshold)
			if m.fmriAnalysisPlotThresholdModeIndex%3 == 1 { // fdr
				options = append(options, optFmriAnalysisPlotFdrQ)
			}
			options = append(options, optFmriAnalysisPlotClusterMinVoxels)

			// Scaling
			options = append(options, optFmriAnalysisPlotVmaxMode)
			if m.fmriAnalysisPlotVmaxModeIndex%3 == 2 { // manual
				options = append(options, optFmriAnalysisPlotVmaxManual)
			}

			// Content
			options = append(options,
				optFmriAnalysisPlotIncludeUnthresholded,
				optFmriAnalysisPlotFormatPNG,
				optFmriAnalysisPlotFormatSVG,
				optFmriAnalysisPlotTypeSlices,
				optFmriAnalysisPlotTypeGlass,
				optFmriAnalysisPlotTypeHist,
				optFmriAnalysisPlotTypeClusters,
				optFmriAnalysisPlotEffectSize,
				optFmriAnalysisPlotStandardError,
				optFmriAnalysisPlotMotionQC,
				optFmriAnalysisPlotCarpetQC,
				optFmriAnalysisPlotTSNRQC,
				optFmriAnalysisPlotDesignQC,
				optFmriAnalysisPlotEmbedImages,
				optFmriAnalysisPlotSignatures,
				optFmriAnalysisSignatureDir,
			)
		}
	} else {
		options = append(options, optFmriTrialSigGroup)
		if m.fmriTrialSigGroupExpanded {
			trialMethod := "beta-series"
			if m.fmriTrialSigMethodIndex%2 == 1 {
				trialMethod = "lss"
			}
				options = append(options,
					optFmriTrialSigMethod,
					optFmriTrialSigIncludeOtherEvents,
					optFmriTrialSigMaxTrialsPerRun,
				optFmriTrialSigFixedEffectsWeighting,
				optFmriTrialSigWriteConditionBetas,
				optFmriTrialSigWriteTrialBetas,
				optFmriTrialSigWriteTrialVariances,
					optFmriTrialSigSignatureNPS,
					optFmriTrialSigSignatureSIIPS1,
					optFmriAnalysisSignatureDir,
					optFmriTrialSigScopeStimPhases,
					optFmriTrialSigGroupColumn,
					optFmriTrialSigGroupValues,
					optFmriTrialSigGroupScope,
			)
			if trialMethod == "lss" {
				options = append(options, optFmriTrialSigLssOtherRegressors)
			}
		}
	}

	return options
}

// getRawToBidsOptions returns advanced options for raw-to-bids
func (m Model) getRawToBidsOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optRawMontage,
		optRawLineFreq,
		optRawOverwrite,
		optRawTrimToFirstVolume,
		optRawEventPrefixes,
		optRawKeepAnnotations,
	}
}

func (m Model) getFmriRawToBidsOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optFmriRawSession,
		optFmriRawRestTask,
		optFmriRawIncludeRest,
		optFmriRawIncludeFieldmaps,
		optFmriRawDicomMode,
		optFmriRawOverwrite,
		optFmriRawCreateEvents,
		optFmriRawEventGranularity,
		optFmriRawOnsetReference,
		optFmriRawOnsetOffsetS,
		optFmriRawDcm2niixPath,
		optFmriRawDcm2niixArgs,
	}
}

// getMergeBehaviorOptions returns advanced options for merge-behavior
func (m Model) getMergeBehaviorOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optMergeEventPrefixes,
		optMergeEventTypes,
	}
}

type plottingAdvancedRowKind int

const (
	plottingRowOption plottingAdvancedRowKind = iota
	plottingRowSection
	plottingRowPlotHeader
	plottingRowPlotField
	plottingRowPlotInfo
)

type plottingAdvancedRow struct {
	kind plottingAdvancedRowKind

	// kind == plottingRowOption
	opt optionType

	// kind == plottingRowPlotHeader/plottingRowPlotField/plottingRowPlotInfo
	plotID string

	// kind == plottingRowPlotField
	plotField plotItemConfigField

	// kind == plottingRowSection/plottingRowPlotInfo
	label string
}

func (m Model) selectedPlotItemsForConfig() []PlotItem {
	items := make([]PlotItem, 0, len(m.plotItems))
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] {
			continue
		}
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		items = append(items, plot)
	}
	return items
}

func (m Model) plotSupportsComparisons(plot PlotItem) bool {
	// Only plots that actually use aligned_events/events_df for condition
	// comparisons should have comparison configs. Based on Python code analysis.
	switch plot.ID {
	// Aperiodic - uses aligned_events
	case "aperiodic_topomaps",
		"aperiodic_by_condition",
		// Connectivity - uses aligned_events
		"connectivity_by_condition",
		"connectivity_circle_condition",
		"connectivity_network",
		// ERDS - uses aligned_events
		"erds_by_condition",
		// Complexity - uses aligned_events
		"complexity_by_condition",
		// Spectral - uses aligned_events
		"spectral_by_condition",
		// Ratios - uses aligned_events
		"ratios_by_condition",
		// Asymmetry - uses aligned_events
		"asymmetry_by_condition",
		// Bursts - uses aligned_events
		"bursts_by_condition",
		// ITPC - uses aligned_events
		"itpc_by_condition",
		// PAC - uses aligned_events
		"pac_by_condition",
		// Power - uses aligned_events
		"power_by_condition",
		"power_spectral_density",
		// ERP - all use conditions
		"erp_butterfly",
		"erp_roi",
		"erp_contrast",
		// TFR contrast plots - use conditions (column comparisons only, no window comparisons)
		"tfr_scalpmean_contrast",
		"tfr_channels_contrast",
		"tfr_rois_contrast",
		"tfr_topomaps",
		"tfr_band_evolution":
		return true
	}
	return false
}

func (m Model) plotConfigFields(plot PlotItem) []plotItemConfigField {
	if plot.ID == "behavior_temporal_topomaps" {
		return []plotItemConfigField{plotItemConfigFieldBehaviorTemporalStatsFeatureFolder}
	}
	fields := make([]plotItemConfigField, 0, 8)
	if plot.ID == "behavior_dose_response" {
		fields = append(fields,
			plotItemConfigFieldDoseResponseDoseColumn,
			plotItemConfigFieldDoseResponseResponseColumn,
			plotItemConfigFieldDoseResponseSegment,
		)
	}
	if plot.ID == "behavior_pain_probability" {
		fields = append(fields,
			plotItemConfigFieldDoseResponseDoseColumn,
			plotItemConfigFieldDoseResponsePainColumn,
		)
	}
	if plot.ID == "band_power_topomaps" {
		fields = append(fields,
			plotItemConfigFieldTopomapWindow,
			plotItemConfigFieldCompareWindows,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonWindows,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	}
	if plot.ID == "connectivity_circle_condition" {
		fields = append(fields,
			plotItemConfigFieldConnectivityCircleTopFraction,
			plotItemConfigFieldConnectivityCircleMinLines,
			plotItemConfigFieldComparisonSegment,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "connectivity_network" {
		fields = append(fields,
			plotItemConfigFieldConnectivityNetworkTopFraction,
			plotItemConfigFieldComparisonSegment,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "erp_butterfly" {
		// erp_butterfly only uses column comparisons, not window/segment/ROI comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "erp_roi" {
		// erp_roi uses column comparisons and ROI filtering, but not window/segment comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
			plotItemConfigFieldComparisonROIs,
		)
	} else if plot.ID == "erp_contrast" {
		// erp_contrast only uses column comparisons, not window/segment/ROI comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "aperiodic_topomaps" {
		// aperiodic_topomaps only uses column comparisons, not window/segment comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "power_spectral_density" {
		// power_spectral_density only uses column comparisons (supports 1+ conditions)
		// CompareColumns toggle not needed - column comparison is always required
		fields = append(fields,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if m.plotSupportsComparisons(plot) {
		// TFR plots only use column comparisons, not window comparisons
		isTfrPlot := plot.Group == "tfr"

		if !isTfrPlot {
			// Feature plots can use window comparisons
			fields = append(fields,
				plotItemConfigFieldCompareWindows,
				plotItemConfigFieldComparisonWindows,
			)
			// Feature plots use comparison_segment to specify which time window to compare
			fields = append(fields, plotItemConfigFieldComparisonSegment)
		}

		// Column comparison fields (used by both TFR and feature plots)
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)

		// ROI field only for feature plots that use ROIs in comparisons
		// TFR topomaps are already spatial, so ROIs don't apply
		if !isTfrPlot && plot.ID != "tfr_rois" && plot.ID != "tfr_channels_contrast" && plot.ID != "tfr_scalpmean_contrast" {
			fields = append(fields, plotItemConfigFieldComparisonROIs)
		}
	}
	if plot.ID == "itpc_topomaps" {
		fields = append(fields, plotItemConfigFieldItpcSharedColorbar)
	}
	if plot.ID == "behavior_scatter" {
		fields = append(fields,
			plotItemConfigFieldBehaviorScatterFeatures,
			plotItemConfigFieldBehaviorScatterColumns,
			plotItemConfigFieldBehaviorScatterAggregationModes,
			plotItemConfigFieldBehaviorScatterSegment,
		)
	}
	if plot.ID == "tfr_topomaps" {
		fields = append(fields,
			plotItemConfigFieldTfrTopomapActiveWindow,
			plotItemConfigFieldTfrTopomapWindowSizeMs,
			plotItemConfigFieldTfrTopomapWindowCount,
			plotItemConfigFieldTfrTopomapLabelXPosition,
			plotItemConfigFieldTfrTopomapLabelYPositionBottom,
			plotItemConfigFieldTfrTopomapLabelYPosition,
			plotItemConfigFieldTfrTopomapTitleY,
			plotItemConfigFieldTfrTopomapTitlePad,
			plotItemConfigFieldTfrTopomapSubplotsRight,
			plotItemConfigFieldTfrTopomapTemporalHspace,
			plotItemConfigFieldTfrTopomapTemporalWspace,
		)
	}
	return fields
}

func (m Model) getPlottingAdvancedRows() []plottingAdvancedRow {
	// When using defaults, keep the list to a single actionable row so
	// navigation matches the minimal renderer.
	if m.useDefaultAdvanced {
		return []plottingAdvancedRow{{kind: plottingRowOption, opt: optUseDefaults}}
	}

	rows := make([]plottingAdvancedRow, 0, 128)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowOption, opt: optUseDefaults})

	// Only show per-plot configs for selected plots
	selectedPlots := m.selectedPlotItemsForConfig()
	if len(selectedPlots) == 0 {
		rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotInfo, label: "No plots selected."})
		return rows
	}

	// Per-plot configs only (global styling moved to categories page)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowSection, label: "Plot-Specific Settings"})
	for _, plot := range selectedPlots {
		rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotHeader, plotID: plot.ID})

		fields := m.plotConfigFields(plot)
		if len(fields) == 0 {
			if m.plotItemConfigExpanded[plot.ID] {
				rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotInfo, plotID: plot.ID, label: "No plot-specific settings."})
			}
			continue
		}
		if !m.plotItemConfigExpanded[plot.ID] {
			continue
		}
		for _, f := range fields {
			rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotField, plotID: plot.ID, plotField: f})
		}
	}

	return rows
}

func (m Model) getGlobalStylingOptions() []optionType {
	// Truly global styling options that apply to ALL plots
	options := []optionType{}

	// Defaults & Output
	options = append(options, optPlotGroupDefaults)
	if m.plotGroupDefaultsExpanded {
		options = append(options, optPlotBboxInches, optPlotPadInches)
	}

	// Fonts
	options = append(options, optPlotGroupFonts)
	if m.plotGroupFontsExpanded {
		options = append(options,
			optPlotFontFamily,
			optPlotFontWeight,
			optPlotFontSizeSmall,
			optPlotFontSizeMedium,
			optPlotFontSizeLarge,
			optPlotFontSizeTitle,
			optPlotFontSizeAnnotation,
			optPlotFontSizeLabel,
			optPlotFontSizeYLabel,
			optPlotFontSizeSuptitle,
			optPlotFontSizeFigureTitle,
		)
	}

	// Layout
	options = append(options, optPlotGroupLayout)
	if m.plotGroupLayoutExpanded {
		options = append(options,
			optPlotLayoutTightRect,
			optPlotLayoutTightRectMicrostate,
			optPlotGridSpecWidthRatios,
			optPlotGridSpecHeightRatios,
			optPlotGridSpecHspace,
			optPlotGridSpecWspace,
			optPlotGridSpecLeft,
			optPlotGridSpecRight,
			optPlotGridSpecTop,
			optPlotGridSpecBottom,
		)
	}

	// Figure Sizes
	options = append(options, optPlotGroupFigureSizes)
	if m.plotGroupFigureSizesExpanded {
		options = append(options,
			optPlotFigureSizeStandard,
			optPlotFigureSizeMedium,
			optPlotFigureSizeSmall,
			optPlotFigureSizeSquare,
			optPlotFigureSizeWide,
			optPlotFigureSizeTFR,
			optPlotFigureSizeTopomap,
		)
	}

	// Colors
	options = append(options, optPlotGroupColors)
	if m.plotGroupColorsExpanded {
		options = append(options,
			optPlotColorPain,
			optPlotColorNonpain,
			optPlotColorSignificant,
			optPlotColorNonsignificant,
			optPlotColorGray,
			optPlotColorLightGray,
			optPlotColorBlack,
			optPlotColorBlue,
			optPlotColorRed,
			optPlotColorNetworkNode,
		)
	}

	// Alpha
	options = append(options, optPlotGroupAlpha)
	if m.plotGroupAlphaExpanded {
		options = append(options,
			optPlotAlphaGrid,
			optPlotAlphaFill,
			optPlotAlphaCI,
			optPlotAlphaCILine,
			optPlotAlphaTextBox,
			optPlotAlphaViolinBody,
			optPlotAlphaRidgeFill,
		)
	}

	// Topomap
	options = append(options, optPlotGroupTopomap)
	if m.plotGroupTopomapExpanded {
		options = append(options,
			optPlotTopomapContours,
			optPlotTopomapColormap,
			optPlotTopomapColorbarFraction,
			optPlotTopomapColorbarPad,
			optPlotTopomapDiffAnnotation,
			optPlotTopomapAnnotateDescriptive,
			optPlotTopomapSigMaskMarker,
			optPlotTopomapSigMaskMarkerFaceColor,
			optPlotTopomapSigMaskMarkerEdgeColor,
			optPlotTopomapSigMaskLinewidth,
			optPlotTopomapSigMaskMarkersize,
		)
	}

	// TFR
	options = append(options, optPlotGroupTFR)
	if m.plotGroupTFRExpanded {
		options = append(options,
			optPlotTFRLogBase,
			optPlotTFRPercentageMultiplier,
			optPlotTFRTopomapWindowSizeMs,
			optPlotTFRTopomapWindowCount,
			optPlotTFRTopomapLabelXPosition,
			optPlotTFRTopomapLabelYPositionBottom,
			optPlotTFRTopomapLabelYPosition,
			optPlotTFRTopomapTitleY,
			optPlotTFRTopomapTitlePad,
			optPlotTFRTopomapSubplotsRight,
			optPlotTFRTopomapTemporalHspace,
			optPlotTFRTopomapTemporalWspace,
		)
	}

	return options
}

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	// Check if any computation is selected
	hasAnyComputation := len(m.SelectedComputations()) > 0

	// General section - only show if at least one computation is selected
	if hasAnyComputation {
		options = append(options, optBehaviorGroupGeneral)
		if m.behaviorGroupGeneralExpanded {
			// RNG Seed and N Jobs are always relevant
			options = append(options, optRNGSeed, optBehaviorNJobs)

			// Correlation method and robust correlation - only for correlations, stability, pain_sensitivity
			needsCorrelationMethod := m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability") ||
				m.isComputationSelected("pain_sensitivity")
			if needsCorrelationMethod {
				options = append(options, optCorrMethod, optRobustCorrelation)
			}

			// Bootstrap - relevant for correlations, stability
			needsBootstrap := m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability")
			if needsBootstrap {
				options = append(options, optBootstrap)
			}

			// FDR Alpha - relevant for correlations, condition, temporal, cluster, regression
			needsFDR := m.isComputationSelected("correlations") ||
				m.isComputationSelected("condition") ||
				m.isComputationSelected("temporal") ||
				m.isComputationSelected("cluster") ||
				m.isComputationSelected("regression")
			if needsFDR {
				options = append(options, optFDRAlpha)
			}

			// N Permutations - relevant for cluster, temporal, regression, correlations, mediation, moderation
			needsPermutations := m.isComputationSelected("cluster") ||
				m.isComputationSelected("temporal") ||
				m.isComputationSelected("regression") ||
				m.isComputationSelected("correlations") ||
				m.isComputationSelected("mediation") ||
				m.isComputationSelected("moderation")
			if needsPermutations {
				options = append(options, optNPerm)
			}

			// Covariate controls - relevant for regression, models, influence, correlations, stability, pain_sensitivity
			needsCovariates := m.isComputationSelected("regression") ||
				m.isComputationSelected("models") ||
				m.isComputationSelected("influence") ||
				m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability") ||
				m.isComputationSelected("pain_sensitivity")
			if needsCovariates {
				options = append(options, optControlTemp, optControlOrder)
			}

			// Run adjustment - relevant for trial_table, correlations
			needsRunAdjustment := m.isComputationSelected("trial_table") ||
				m.isComputationSelected("correlations")
			if needsRunAdjustment {
				options = append(options,
					optRunAdjustmentEnabled,
					optRunAdjustmentColumn,
					optRunAdjustmentIncludeInCorrelations,
					optRunAdjustmentMaxDummies,
				)
			}

			// Change scores, LOSO stability, Bayes factors - relevant for correlations
			if m.isComputationSelected("correlations") {
				options = append(options,
					optComputeChangeScores,
					optComputeLosoStability,
					optComputeBayesFactors,
				)
			}

			// Feature QC (optional gating) - relevant for correlations / multilevel correlations
			if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
				options = append(options,
					optFeatureQCEnabled,
					optFeatureQCMaxMissingPct,
					optFeatureQCMinVariance,
					optFeatureQCCheckWithinRunVariance,
				)
			}
		}
	}

	// Trial table section - only show if trial_table computation is selected
	if m.isComputationSelected("trial_table") {
		options = append(options, optBehaviorGroupTrialTable)
		if m.behaviorGroupTrialTableExpanded {
			options = append(options,
				optTrialTableFormat,
				optTrialTableAddLagFeatures,
				optTrialOrderMaxMissingFraction,
				optFeatureSummariesEnabled,
			)
		}
	}

	// Pain Residual section - only show if pain_residual computation is selected
	if m.isComputationSelected("pain_residual") {
		options = append(options, optBehaviorGroupPainResidual)
		if m.behaviorGroupPainResidualExpanded {
			options = append(options,
				optPainResidualEnabled,
				optPainResidualMethod,
				optPainResidualPolyDegree,
				optPainResidualSplineDfCandidates,
				optPainResidualModelCompare,
				optPainResidualModelComparePolyDegrees,
				optPainResidualBreakpoint,
				optPainResidualBreakpointCandidates,
				optPainResidualBreakpointQlow,
				optPainResidualBreakpointQhigh,
				optPainResidualCrossfitEnabled,
				optPainResidualCrossfitGroupColumn,
				optPainResidualCrossfitNSplits,
				optPainResidualCrossfitMethod,
				optPainResidualCrossfitSplineKnots,
			)
		}
	}

	// Correlations section
	if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
		options = append(options, optBehaviorGroupCorrelations)
		if m.behaviorGroupCorrelationsExpanded {
			if m.isComputationSelected("correlations") {
				options = append(options,
					optCorrelationsTargetRating,
					optCorrelationsTargetTemperature,
					optCorrelationsTargetPainResidual,
					optCorrelationsTargetColumn,
					optCorrelationsPreferPainResidual,
					optCorrelationsTypes,
					optCorrelationsPrimaryUnit,
					optCorrelationsPermutationPrimary,
					optCorrelationsUseCrossfitPainResidual,
				)
			}
			options = append(options, optCorrelationsMultilevel)
			if m.isComputationSelected("multilevel_correlations") {
				options = append(options, optGroupLevelBlockPermutation)
			}
		}
	}

	// Regression section (includes model sensitivity options)
	if m.isComputationSelected("regression") {
		options = append(options, optBehaviorGroupRegression)
		if m.behaviorGroupRegressionExpanded {
			options = append(options,
				optRegressionOutcome,
				optRegressionIncludeTemperature,
				optRegressionTempControl,
			)
			if m.regressionTempControl == 2 {
				options = append(options,
					optRegressionTempSplineKnots,
					optRegressionTempSplineQlow,
					optRegressionTempSplineQhigh,
				)
			}
			options = append(options,
				optRegressionIncludeTrialOrder,
				optRegressionIncludePrev,
				optRegressionIncludeRunBlock,
				optRegressionIncludeInteraction,
				optRegressionStandardize,
				optRegressionPermutations,
				optRegressionMaxFeatures,
			)
			// Model sensitivity options (now part of regression)
			options = append(options,
				optModelsFamilyOLS,
				optModelsFamilyRobust,
				optModelsFamilyQuantile,
				optModelsFamilyLogit,
			)
		}
	}

	// Stability section
	if m.isComputationSelected("stability") {
		options = append(options, optBehaviorGroupStability)
		if m.behaviorGroupStabilityExpanded {
			options = append(options,
				optStabilityMethod,
				optStabilityOutcome,
				optStabilityGroupColumn,
				optStabilityPartialTemp,
				optStabilityMaxFeatures,
				optStabilityAlpha,
			)
		}
	}

	// Consistency section
	if m.isComputationSelected("consistency") {
		options = append(options, optBehaviorGroupConsistency)
		if m.behaviorGroupConsistencyExpanded {
			options = append(options, optConsistencyEnabled)
		}
	}

	// Influence section
	if m.isComputationSelected("influence") {
		options = append(options, optBehaviorGroupInfluence)
		if m.behaviorGroupInfluenceExpanded {
			options = append(options,
				optInfluenceOutcomeRating,
				optInfluenceOutcomePainResidual,
				optInfluenceOutcomeTemperature,
				optInfluenceMaxFeatures,
				optInfluenceIncludeTemperature,
				optInfluenceTempControl,
			)
			if m.influenceTempControl == 2 {
				options = append(options,
					optInfluenceTempSplineKnots,
					optInfluenceTempSplineQlow,
					optInfluenceTempSplineQhigh,
				)
			}
			options = append(options,
				optInfluenceIncludeTrialOrder,
				optInfluenceIncludeRunBlock,
				optInfluenceIncludeInteraction,
				optInfluenceStandardize,
				optInfluenceCooksThreshold,
				optInfluenceLeverageThreshold,
			)
		}
	}

	// Report section
	if m.isComputationSelected("report") {
		options = append(options, optBehaviorGroupReport)
		if m.behaviorGroupReportExpanded {
			options = append(options, optReportTopN)
		}
	}

	// Condition section
	if m.isComputationSelected("condition") {
		options = append(options, optBehaviorGroupCondition)
		if m.behaviorGroupConditionExpanded {
			options = append(options,
				optConditionCompareColumn,
				optConditionCompareValues,
				optConditionCompareWindows,
				optConditionWindowPrimaryUnit,
				optConditionPermutationPrimary,
				optConditionFailFast,
				optConditionEffectThreshold,
				optConditionOverwrite,
			)
		}
	}

	// Temporal section
	if m.isComputationSelected("temporal") {
		options = append(options, optBehaviorGroupTemporal)
		if m.behaviorGroupTemporalExpanded {
			options = append(options,
				optTemporalResolutionMs,
				optTemporalTimeMinMs,
				optTemporalTimeMaxMs,
				optTemporalSmoothMs,
				optTemporalTargetColumn,
				optTemporalSplitByCondition,
				optTemporalConditionColumn,
				optTemporalConditionValues,
				optTemporalIncludeROIAverages,
				optTemporalIncludeTFGrid,
			)
			// Show ITPC-specific options when 'itpc' is selected in step 3 (feature selection)
			if m.featureFileSelected["itpc"] {
				options = append(options,
					optTemporalITPCBaselineCorrection,
					optTemporalITPCBaselineMin,
					optTemporalITPCBaselineMax,
				)
			}
			// Show ERDS-specific options when 'erds' is selected in step 3 (feature selection)
			if m.featureFileSelected["erds"] {
				options = append(options,
					optTemporalERDSBaselineMin,
					optTemporalERDSBaselineMax,
					optTemporalERDSMethod,
				)
			}
			// TF Heatmap options (always visible when temporal is expanded)
			options = append(options,
				optTemporalTfHeatmapEnabled,
			)
			if m.tfHeatmapEnabled {
				options = append(options,
					optTemporalTfHeatmapFreqs,
					optTemporalTfHeatmapTimeResMs,
				)
			}
		}
	}

	// Cluster section
	if m.isComputationSelected("cluster") {
		options = append(options, optBehaviorGroupCluster)
		if m.behaviorGroupClusterExpanded {
			options = append(
				options,
				optClusterThreshold,
				optClusterMinSize,
				optClusterTail,
				optClusterConditionColumn,
				optClusterConditionValues,
			)
		}
	}

	// Mediation section
	if m.isComputationSelected("mediation") {
		options = append(options, optBehaviorGroupMediation)
		if m.behaviorGroupMediationExpanded {
			options = append(options, optMediationBootstrap, optMediationPermutations, optMediationMinEffect, optMediationMaxMediatorsEnabled, optMediationMaxMediators)
		}
	}

	// Moderation section
	if m.isComputationSelected("moderation") {
		options = append(options, optBehaviorGroupModeration)
		if m.behaviorGroupModerationExpanded {
			options = append(options, optModerationMaxFeaturesEnabled, optModerationMaxFeatures, optModerationPermutations)
		}
	}

	// Mixed effects section
	if m.isComputationSelected("mixed_effects") {
		options = append(options, optBehaviorGroupMixedEffects)
		if m.behaviorGroupMixedEffectsExpanded {
			options = append(options, optMixedEffectsType, optMixedMaxFeatures)
		}
	}

	// Output section - only show if at least one computation is selected
	if hasAnyComputation {
		options = append(options, optBehaviorGroupOutput)
		if m.behaviorGroupOutputExpanded {
			options = append(options, optAlsoSaveCsv, optBehaviorOverwrite)
		}
	}

	return options
}

func (m Model) getPlotConfigOptions() []optionType {
	options := []optionType{
		optPlotPNG,
		optPlotSVG,
		optPlotPDF,
		optPlotDPI,
		optPlotSaveDPI,
		optPlotOverwrite,
	}

	// Dynamic options based on selected plots/categories
	if m.IsPlotCategorySelected("tfr") || m.IsPlotCategorySelected("features") {
		// ITPC and PAC settings
		options = append(options, optPlotSharedColorbar)
	}

	return options
}

func (m Model) getMLOptions() []optionType {
	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}

	opts := []optionType{
		optUseDefaults,
		optMLTarget,
		optMLFeatureFamilies,
		optMLFeatureBands,
		optMLFeatureSegments,
		optMLFeatureScopes,
		optMLFeatureStats,
		optMLFeatureHarmonization,
		optMLCovariates,
	}

	if mode == "incremental_validity" {
		opts = append(opts, optMLBaselinePredictors)
	}

	if strings.EqualFold(strings.TrimSpace(m.mlTarget), "fmri_signature") {
		opts = append(opts, optMLFmriSigGroup)
		if m.mlFmriSigGroupExpanded {
			opts = append(opts,
				optMLFmriSigMethod,
				optMLFmriSigContrastName,
				optMLFmriSigSignature,
				optMLFmriSigMetric,
				optMLFmriSigNormalization,
				optMLFmriSigRoundDecimals,
			)
		}
	}

	opts = append(opts, optMLRequireTrialMlSafe)

	if mode == "classify" {
		opts = append(opts, optMLClassificationModel, optMLBinaryThresholdEnabled)
		if m.mlBinaryThresholdEnabled {
			opts = append(opts, optMLBinaryThreshold)
		}
		opts = append(opts, optVarianceThresholdGrid)
	} else if mode != "timegen" && mode != "" {
		// Most non-classification stages use the regression model family (timegen is separate).
		if mode != "model_comparison" {
			opts = append(opts, optMLRegressionModel)
		}
		opts = append(
			opts,
			optElasticNetAlphaGrid,
			optElasticNetL1RatioGrid,
			optRidgeAlphaGrid,
			optRfNEstimators,
			optRfMaxDepthGrid,
			optVarianceThresholdGrid,
		)
	}

	opts = append(opts, optMLNPerm, optMLInnerSplits, optMLOuterJobs, optRNGSeed)

	if mode == "uncertainty" {
		opts = append(opts, optMLUncertaintyAlpha)
	}
	if mode == "permutation" {
		opts = append(opts, optMLPermNRepeats)
	}

	return opts
}

func (m Model) isCurrentlyEditing(opt optionType) bool {

	if !m.editingNumber {
		return false
	}

	// Plotting advanced config uses a mixed row model (per-plot + global options),
	// so the cursor no longer indexes directly into getPlottingOptions().
	if m.Pipeline == types.PipelinePlotting {
		rows := m.getPlottingAdvancedRows()
		if m.advancedCursor < 0 || m.advancedCursor >= len(rows) {
			return false
		}
		return rows[m.advancedCursor].kind == plottingRowOption && rows[m.advancedCursor].opt == opt
	}

	var options []optionType
	switch m.Pipeline {
	case types.PipelineFeatures:
		options = m.getFeaturesOptions()
	case types.PipelineBehavior:
		options = m.getBehaviorOptions()
	case types.PipelineML:
		options = m.getMLOptions()
	case types.PipelinePreprocessing:
		options = m.getPreprocessingOptions()
	case types.PipelineRawToBIDS:
		options = m.getRawToBidsOptions()
	case types.PipelineFmriRawToBIDS:
		options = m.getFmriRawToBidsOptions()
	case types.PipelineFmri:
		options = m.getFmriPreprocessingOptions()
	case types.PipelineFmriAnalysis:
		options = m.getFmriAnalysisOptions()
	default:
		return false
	}
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return false
	}
	return options[m.advancedCursor] == opt
}
