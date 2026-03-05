package wizard

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

// File layout notes:
// - `model.go`: shared types/constants plus constructor and Tea lifecycle core.
// - `model_scroll_plotters.go`: scroll/plotter availability and discovery helpers.
// - `model_state_data.go`: external setters/getters and discovered metadata state.
// - `model_editing_config.go`: expanded-list toggles and text/config mutation logic.
// - `model_options.go`: advanced option list builders and edit-state predicates.

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
	{"predictor_residual", "Residual + Diagnostics", "Compute residualized outcome and predictor model diagnostics", "DataPrep"},

	// Core Analyses
	{"correlations", "Correlations", "EEG-outcome correlations with bootstrap CIs", "Core"},
	{"multilevel_correlations", "Group Multilevel Correlations", "Group-level correlations with block-restricted permutations", "Core"},
	{"regression", "Regression", "Feature regression with optional permutation + model sensitivity", "Core"},
	{"models", "Model Families", "Per-feature model families (OLS/robust/quantile/logit)", "Core"},
	{"condition", "Condition Comparison", "Compare conditions (e.g., condition A vs condition B)", "Core"},
	{"temporal", "Temporal Correlations", "Time-resolved correlation analysis", "Core"},
	{"predictor_sensitivity", "Predictor Sensitivity", "Individual sensitivity (predictor→outcome slope)", "Core"},
	{"cluster", "Cluster Permutation", "Cluster-based permutation tests", "Core"},

	// Advanced/Causal Analyses
	{"mediation", "Mediation Analysis", "Path analysis: does EEG mediate predictor→outcome?", "Advanced"},
	{"moderation", "Moderation Analysis", "Does EEG moderate the predictor→outcome effect?", "Advanced"},
	{"mixed_effects", "Mixed Effects", "Mixed-effects modeling (group-level)", "Advanced"},

	// Quality & Validation
	{"stability", "Stability (Run/Block)", "Within-subject stability diagnostics", "Quality"},
	{"icc", "Reliability (ICC)", "Run-to-run feature reliability", "Quality"},
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
	{"Frontal", "Frontal", "Fp2,Fpz,AF3,AF7,AF8,F1,F2,F3,F5,F6,F7,F8"},
	{"Sensorimotor_Right", "Sensorimotor Right (Contralateral)", "FC4,FC6,C2,C6,CP2,CP4,CP6"},
	{"Sensorimotor_Left", "Sensorimotor Left (Ipsilateral)", "FC5,C1,C3,C5,CP3,CP5"},
	{"Temporal_Right", "Temporal Right", "FT8,FT10,T8,TP8,TP10"},
	{"Temporal_Left", "Temporal Left", "FT7,FT9,T7,TP7,TP9"},
	{"ParOccipital_Right", "ParOccipital Right", "P2,P4,P6,P8,PO4,PO8,O2"},
	{"ParOccipital_Left", "ParOccipital Left", "P1,P3,P5,P7,PO3,PO7,O1"},
	{"ParOccipital_Midline", "ParOccipital Midline", "Pz,POz,Oz"},
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
	{"imcoh", "imCoh", "Imaginary coherence"},
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
	MLClassificationCNN
	MLClassificationEnsemble
)

func (c MLClassificationModel) CLIValue() string {
	switch c {
	case MLClassificationSVM:
		return "svm"
	case MLClassificationLR:
		return "lr"
	case MLClassificationRF:
		return "rf"
	case MLClassificationCNN:
		return "cnn"
	case MLClassificationEnsemble:
		return "ensemble"
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
	return MLClassificationModel((int(c) + 1) % 6)
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
	{"microstates", "Microstates", "EEG microstate dynamics (A-D)"},
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
	"microstates",
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
	"correlations": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	// Multilevel correlations uses the same features as correlations (group-level)
	"multilevel_correlations": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	// Predictor sensitivity uses the same features as correlations
	"predictor_sensitivity": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	// Condition comparison uses trial-level features
	"condition": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	// Temporal: power, itpc, erds are the temporal-specific features computed from epochs
	// User selects which to compute in step 3 (feature selection)
	"temporal": {"power", "itpc", "erds"},
	// Cluster permutation uses TFR data directly
	"cluster": {"power"},
	// Mediation/moderation use correlations features
	"mediation":     {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"moderation":    {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"mixed_effects": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	// Quality & Validation analyses
	"regression":  {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"stability":   {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"validation":  {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"consistency": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"influence":   {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
	"report":      {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "microstates", "erds", "spectral"},
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

	// Source localization
	SourceSegment     string
	SourceHemi        string
	SourceViewsSpec   string
	SourceCortex      string
	SourceSubjectsDir string
	SourceCondition   string
	SourceConditionA  string
	SourceConditionB  string
	SourceBandsSpec   string

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
	DoseResponseDoseColumn          string
	DoseResponseResponseColumn      string
	DoseResponseBinaryOutcomeColumn string
	DoseResponseSegment             string
	DoseResponseBandsSpec           string
	DoseResponseROIsSpec            string
	DoseResponseScopesSpec          string
	DoseResponseStat                string
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
	plotItemConfigFieldSourceSegment
	plotItemConfigFieldSourceHemi
	plotItemConfigFieldSourceViews
	plotItemConfigFieldSourceCortex
	plotItemConfigFieldSourceSubjectsDir
	plotItemConfigFieldSourceCondition
	plotItemConfigFieldSourceConditionA
	plotItemConfigFieldSourceConditionB
	plotItemConfigFieldSourceBands
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
	plotItemConfigFieldDoseResponseBinaryOutcomeColumn
	plotItemConfigFieldDoseResponseSegment
	plotItemConfigFieldDoseResponseBands
	plotItemConfigFieldDoseResponseROIs
	plotItemConfigFieldDoseResponseScopes
	plotItemConfigFieldDoseResponseStat
)

type textField int

const (
	textFieldNone textField = iota
	textFieldTask
	textFieldBidsRoot
	textFieldBidsFmriRoot
	textFieldDerivRoot
	textFieldSourceRoot
	textFieldConfigSetOverrides
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
	textFieldFmriAnalysisScopeColumn
	textFieldFmriAnalysisScopeTrialTypes
	textFieldFmriAnalysisPhaseColumn
	textFieldFmriAnalysisPhaseScopeColumn
	textFieldFmriAnalysisPhaseScopeValue
	textFieldFmriAnalysisStimPhasesToModel
	textFieldFmriAnalysisOutputDir
	textFieldFmriAnalysisFreesurferDir
	textFieldFmriAnalysisSignatureDir
	textFieldFmriAnalysisSignatureMaps
	textFieldFmriTrialSigGroupColumn
	textFieldFmriTrialSigGroupValues
	textFieldFmriTrialSigScopeTrialTypeColumn
	textFieldFmriTrialSigScopePhaseColumn
	textFieldFmriTrialSigScopeTrialTypes
	textFieldFmriTrialSigScopeStimPhases
	textFieldPrepMontage
	textFieldPrepChTypes
	textFieldPrepEegReference
	textFieldPrepEogChannels
	textFieldPrepConditions
	textFieldPrepFileExtension
	textFieldPrepRenameAnotDict
	textFieldPrepCustomBadDict
	// Behavior advanced config text fields
	textFieldConditionCompareColumn
	textFieldConditionCompareWindows
	textFieldConditionCompareValues
	textFieldConditionCompareLabels
	textFieldTemporalConditionColumn
	textFieldTemporalConditionValues
	textFieldTemporalTargetColumn
	textFieldTfHeatmapFreqs
	textFieldRunAdjustmentColumn
	textFieldBehaviorOutcomeColumn
	textFieldBehaviorPredictorColumn
	textFieldPredictorResidualCrossfitGroupColumn
	textFieldClusterConditionColumn
	textFieldClusterConditionValues
	textFieldCorrelationsTargetColumn
	textFieldCorrelationsPowerSegment
	textFieldGroupLevelTarget
	textFieldCorrelationsTypes
	textFieldCorrelationsFeatures
	textFieldPredictorSensitivityFeatures
	textFieldConditionFeatures
	textFieldTemporalFeatures
	textFieldClusterFeatures
	textFieldMediationFeatures
	textFieldModerationFeatures
	// Frequency band editing text fields
	textFieldBandName
	textFieldBandLowHz
	textFieldBandHighHz
	// Features advanced config text fields
	textFieldPACPairs
	textFieldBurstBands
	textFieldSpectralRatioPairs
	textFieldSpectralSegments
	textFieldAsymmetryChannelPairs
	textFieldAsymmetryActivationBands
	textFieldIAFRois
	textFieldERPComponents
	textFieldMicrostatesFixedTemplatesPath
	// ITPC condition-based text fields
	textFieldItpcConditionColumn
	textFieldItpcConditionValues
	// Connectivity condition-based text fields
	textFieldConnConditionColumn
	textFieldConnConditionValues
	// Source localization advanced config text fields
	textFieldSourceLocSubject
	textFieldSourceLocSubjectsDir
	textFieldSourceLocTrans
	textFieldSourceLocBem
	textFieldSourceLocFmriStatsMap
	textFieldSourceLocContrastConditionColumn
	textFieldSourceLocContrastConditionA
	textFieldSourceLocContrastConditionB
	// fMRI contrast builder text fields
	textFieldSourceLocFmriCondAColumn
	textFieldSourceLocFmriCondAValue
	textFieldSourceLocFmriCondBColumn
	textFieldSourceLocFmriCondBValue
	textFieldSourceLocFmriContrastFormula
	textFieldSourceLocFmriContrastName
	textFieldSourceLocFmriRunsToInclude
	textFieldSourceLocFmriConditionScopeColumn
	textFieldSourceLocFmriConditionScopeTrialTypes
	textFieldSourceLocFmriPhaseColumn
	textFieldSourceLocFmriPhaseScopeColumn
	textFieldSourceLocFmriPhaseScopeValue
	textFieldSourceLocFmriStimPhasesToModel
	textFieldPredictorResidualSplineDfCandidates
	textFieldPredictorResidualModelComparePolyDegrees
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
	textFieldPlotColorCondB
	textFieldPlotColorCondA
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
	textFieldPlotSourceHemi
	textFieldPlotSourceViews
	textFieldPlotSourceCortex
	textFieldPlotSourceSubjectsDir
	// Machine Learning advanced config text fields
	textFieldMLTarget
	textFieldMLFmriSigContrastName
	textFieldMLFeatureFamilies
	textFieldMLFeatureBands
	textFieldMLFeatureSegments
	textFieldMLFeatureScopes
	textFieldMLFeatureStats
	textFieldMLCovariates
	textFieldMLSpatialRegionsAllowed
	textFieldMLBaselinePredictors
	textFieldMLPlotFormats
	textFieldElasticNetAlphaGrid
	textFieldElasticNetL1RatioGrid
	textFieldRidgeAlphaGrid
	textFieldRfMaxDepthGrid
	textFieldVarianceThresholdGrid

	// ML new text fields
	textFieldMLSvmCGrid
	textFieldMLSvmGammaGrid
	textFieldMLLrCGrid
	textFieldMLLrL1RatioGrid
	textFieldMLRfMinSamplesSplitGrid
	textFieldMLRfMinSamplesLeafGrid

	// EEG Preprocessing new text fields
	textFieldPrepEcgChannels
	textFieldPrepAutorejectNInterpolate

	// Event Column Mapping text fields
	textFieldEventColPredictor
	textFieldEventColOutcome
	textFieldEventColBinaryOutcome
	textFieldEventColCondition
	textFieldEventColRequired
	textFieldConditionPreferredPrefixes

	// Change Scores text fields
	textFieldChangeScoresWindowPairs

	// ERDS Condition Markers text fields
	textFieldERDSConditionMarkerBands
	textFieldERDSLateralityColumns
	textFieldERDSSomatosensoryLeftChannels
	textFieldERDSSomatosensoryRightChannels

	// Behavior Statistics text fields
	textFieldBehaviorPermGroupColumnPreference
	textFieldBehaviorFeatureRegistryFilesJSON
	textFieldBehaviorFeatureRegistrySourceToTypeJSON
	textFieldBehaviorFeatureRegistryTypeHierarchyJSON
	textFieldBehaviorFeatureRegistryPatternsJSON
	textFieldBehaviorFeatureRegistryClassifiersJSON

	// System / IO text fields
	textFieldIOPredictorRange

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
	{ID: "aperiodic_topomaps", Group: "aperiodic", Name: "Topomaps", Description: "Topographic maps of aperiodic and periodic-peak metrics", RequiredFiles: []string{"features_aperiodic*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "aperiodic_by_condition", Group: "aperiodic", Name: "Condition Comparison", Description: "Aperiodic and oscillatory peak differences between conditions", RequiredFiles: []string{"features_aperiodic*.tsv", "events.tsv"}, RequiresFeatures: true},
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
	// Microstates
	{ID: "microstates_by_condition", Group: "microstates", Name: "Condition Comparison", Description: "Microstate dynamics differences between conditions", RequiredFiles: []string{"features_microstates*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Bursts
	{ID: "bursts_by_condition", Group: "bursts", Name: "Condition Comparison", Description: "Burst differences between conditions", RequiredFiles: []string{"features_bursts*.tsv", "events.tsv"}, RequiresFeatures: true},
	// ERP
	{ID: "erp_butterfly", Group: "erp", Name: "Butterfly", Description: "Butterfly ERP plots (all channels)", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_roi", Group: "erp", Name: "ROI Waveforms", Description: "ROI-based ERP waveforms with error bars", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_contrast", Group: "erp", Name: "Contrast", Description: "ERP condition contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	// TFR
	{ID: "tfr_scalpmean", Group: "tfr", Name: "Scalp-Mean TFR", Description: "Scalp-mean time-frequency representation", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_scalpmean_contrast", Group: "tfr", Name: "Scalp-Mean Contrast", Description: "Condition A vs B scalp-mean TFR contrast", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_channels", Group: "tfr", Name: "Channel TFRs", Description: "Time-frequency per channel", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_channels_contrast", Group: "tfr", Name: "Channel Contrasts", Description: "Condition A vs B channel TFR contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_rois", Group: "tfr", Name: "ROI TFRs", Description: "Time-frequency per ROI", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_rois_contrast", Group: "tfr", Name: "ROI Contrasts", Description: "Condition A vs B ROI TFR contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_topomaps", Group: "tfr", Name: "TFR Topomaps", Description: "Time-frequency topographic maps", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_band_evolution", Group: "tfr", Name: "Band Evolution", Description: "Frequency band power evolution over time", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Behavior
	{ID: "behavior_psychometrics", Group: "behavior", Name: "Psychometrics", Description: "Rating distributions and psychometrics", RequiredFiles: []string{"events.tsv"}},
	{ID: "behavior_scatter", Group: "behavior", Name: "Feature-Behavior Scatter", Description: "Configurable scatter plots correlating any EEG feature with behavioral columns", RequiredFiles: []string{"features_*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "behavior_temporal_topomaps", Group: "behavior", Name: "Temporal Topomaps", Description: "Temporal correlation topomaps", RequiredFiles: []string{"stats/temporal_correlations*/*/temporal_correlations_by_condition*.npz"}, RequiresStats: true},
	{ID: "behavior_dose_response", Group: "behavior", Name: "Dose Response", Description: "Dose-response curves and contrasts", RequiredFiles: []string{"stats/trial_table*/*/trials_*.tsv", "stats/trial_table*/*/trials_*.parquet"}, RequiresStats: true},
	{ID: "behavior_binary_outcome_probability", Group: "behavior", Name: "Binary Outcome Probability", Description: "Binary outcome probability vs predictor (dose-response curve)", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
}

var defaultPlotCategories = []FeatureCategory{
	{"power", "Power", "Band power features and topomaps"},
	{"connectivity", "Connectivity", "Functional connectivity measures and networks"},
	{"aperiodic", "Aperiodic", "1/f spectral slope and offset features"},
	{"phase", "Phase (ITPC/PAC)", "Phase coherence and phase-amplitude coupling"},
	{"erds", "ERDS", "Event-related desynchronization/synchronization"},
	{"complexity", "Complexity", "Lempel-Ziv, permutation entropy, sample entropy, and multiscale entropy"},
	{"spectral", "Spectral", "Peak frequency, spectral edge, and entropy"},
	{"ratios", "Ratios", "Band power ratios (theta/beta, alpha/beta, etc.)"},
	{"asymmetry", "Asymmetry", "Hemispheric asymmetry indices"},
	{"microstates", "Microstates", "EEG microstate dynamics and transitions"},
	{"bursts", "Bursts", "Oscillatory burst dynamics"},
	{"quality", "Quality", "Data quality diagnostics and outlier detection"},
	{"erp", "ERP", "Event-related potential waveforms and topographies"},
	{"tfr", "Time-Frequency", "Time-frequency representations and contrasts"},
	{"behavior", "Behavior", "EEG-behavior correlations and temporal stats"},
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
	task               string
	bidsRoot           string
	bidsFmriRoot       string
	derivRoot          string
	sourceRoot         string
	configSetOverrides string // Semicolon-separated KEY=VALUE config overrides
	repoRoot           string // Project repository root for running Python commands

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
	editingROIIdx    int    // Which ROI is being edited (-1 for none)
	editingROIField  int    // 0: name, 1: channels
	roiEditBuffer    string // Buffer for the value being typed
	roiEditCursorPos int    // Caret index within roiEditBuffer (byte offset)

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
	fmriAnalysisCondAColumn       string // Condition A: events column (config/discovery default when empty)
	fmriAnalysisCondAValue        string // Condition A: value in that column
	fmriAnalysisCondBColumn       string // Condition B: events column
	fmriAnalysisCondBValue        string // Condition B: value in that column
	fmriAnalysisContrastName      string // e.g., "contrast"
	fmriAnalysisFormula           string // Custom formula
	fmriAnalysisEventsToModel     string // Optional: comma-separated list of condition values to include (first-level only)
	fmriAnalysisScopeColumn       string // Events column used by condition scope values
	fmriAnalysisScopeTrialTypes   string // Optional: space-separated allow-list for condition selection
	fmriAnalysisPhaseColumn       string // Events column used by phase scoping values
	fmriAnalysisPhaseScopeColumn  string // Events column used to scope phase filtering
	fmriAnalysisPhaseScopeValue   string // Optional scope value for fmriAnalysisPhaseScopeColumn
	fmriAnalysisStimPhasesToModel string // Optional: comma-separated phase allow-list (empty = no scoping)
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
	fmriAnalysisSignatureMaps            string // "NAME:path NAME2:path2" format; empty => from config

	// fMRI trial-wise signatures configuration (beta-series, LSS)
	fmriTrialSigGroupExpanded           bool
	fmriTrialSigMethodIndex             int // 0: beta-series, 1: lss
	fmriTrialSigIncludeOtherEvents      bool
	fmriTrialSigMaxTrialsPerRun         int // 0 = no cap
	fmriTrialSigFixedEffectsWeighting   int // 0: variance, 1: mean
	fmriTrialSigWriteTrialBetas         bool
	fmriTrialSigWriteTrialVariances     bool
	fmriTrialSigWriteConditionBetas     bool
	fmriTrialSigSignatureOption1        bool
	fmriTrialSigSignatureOption2        bool
	fmriTrialSigLssOtherRegressorsIndex int // 0: per-condition, 1: all
	// Signature grouping (compute signatures for specific values within an events column)
	fmriTrialSigGroupColumn          string // e.g., predictor_column
	fmriTrialSigGroupValuesSpec      string // space-separated values (e.g., "44.3 45.3 46.3")
	fmriTrialSigGroupScopeIndex      int    // 0: across-runs (average), 1: per-run
	fmriTrialSigScopeTrialTypeColumn string // Events column used for trial-type scope values
	fmriTrialSigScopePhaseColumn     string // Events column used for phase scope values
	fmriTrialSigScopeTrialTypes      string // Optional: space-separated allow-list for fmriTrialSigScopeTrialTypeColumn
	fmriTrialSigScopeStimPhases      string // Optional: space-separated phase allow-list (empty = no scoping)

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
	plotGroupTopomapExpanded     bool
	plotGroupTFRExpanded         bool
	plotGroupTFRMiscExpanded     bool
	plotGroupSizingExpanded      bool
	plotGroupSourceLocExpanded   bool
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

	plotColorCondB          string
	plotColorCondA          string
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

	plotSourceHemi        string
	plotSourceViews       string
	plotSourceCortex      string
	plotSourceSubjectsDir string

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

	// Execute
	ReadyToExecute bool
	DryRunMode     bool // If true, append --dry-run to command

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

	width        int
	height       int
	contentWidth int // inner width available for step content (set in View)

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
	featGroupMicrostatesExpanded      bool
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
	complexityPEOrder     int // Permutation entropy order (3-7)
	complexityPEDelay     int
	complexitySampEnOrder int     // Sample entropy embedding dimension
	complexitySampEnR     float64 // Sample entropy tolerance as SD fraction
	complexityMSEScaleMin int     // MSE minimum coarse-graining scale
	complexityMSEScaleMax int     // MSE maximum coarse-graining scale

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
	spectralRatioPairsSpec     string // e.g. theta:beta,alpha:beta
	spectralSegmentsSpec       string // e.g. baseline,active
	spectralPsdAdaptive        bool
	spectralMultitaperAdaptive bool

	// Aggregation
	aggregationMethod int // 0: mean, 1: median
	featureTmin       float64
	featureTmax       float64

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
	featAlsoSaveCsv bool // Also save feature tables as CSV files

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
	connOutputLevel     int // 0: full, 1: global_only
	connGraphMetrics    bool
	connGraphProp       float64
	connWindowLen       float64
	connWindowStep      float64
	connAECMode         int // 0: orth, 1: none, 2: sym
	connMode            int // 0: cwt_morlet, 1: multitaper, 2: fourier
	connAECAbsolute     bool
	connEnableAEC       bool
	connNFreqsPerBand   int
	connNCycles         float64
	connDecim           int
	connMinSegSamples   int
	connSmallWorldNRand int

	// Scientific validity options (new)
	itpcMethod             int     // 0: global, 1: fold_global, 2: loo, 3: condition
	aperiodicMinSegmentSec float64 // Minimum segment duration for aperiodic fits
	connAECOutput          int     // 0: r only, 1: z only, 2: both r and z
	connForceWithinEpochML bool    // Force within_epoch for CV/machine learning

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
	connDynamicEnabled         bool    // Enable dynamic sliding-window connectivity features
	connDynamicMeasures        int     // 0: wpli+aec, 1: wpli, 2: aec
	connDynamicAutocorrLag     int     // Lag (in windows) for dynamic autocorrelation
	connDynamicMinWindows      int     // Minimum windows required for dynamic summaries
	connDynamicIncludeROIPairs bool    // Include ROI-pair dynamic summaries
	connDynamicStateEnabled    bool    // Enable dynamic state-transition metrics
	connDynamicStateNStates    int     // Number of k-means states
	connDynamicStateMinWindows int     // Minimum windows for state metrics
	connDynamicStateRandomSeed int     // Random seed for state clustering (-1 = unset)

	// Directed connectivity options (PSI, DTF, PDC)
	directedConnMeasures          map[int]bool // Selected directed connectivity measures
	directedConnEnabled           bool         // Enable directed connectivity extraction
	directedConnOutputLevel       int          // 0: full, 1: global_only
	directedConnMvarOrder         int          // MVAR model order for DTF/PDC
	directedConnNFreqs            int          // Number of frequency bins
	directedConnMinSegSamples     int          // Minimum segment samples
	featGroupDirectedConnExpanded bool         // UI expansion state

	// Source localization options (LCMV, eLORETA)
	sourceLocEnabled            bool    // Enable source localization
	sourceLocMode               int     // 0: EEG-only (template), 1: fMRI-informed (subject-specific)
	sourceLocMethod             int     // 0: lcmv, 1: eloreta
	sourceLocSpacing            int     // 0: oct5, 1: oct6, 2: ico4, 3: ico5
	sourceLocParc               int     // 0: aparc, 1: aparc.a2009s, 2: HCPMMP1
	sourceLocReg                float64 // LCMV regularization
	sourceLocSnr                float64 // eLORETA SNR
	sourceLocLoose              float64 // eLORETA loose constraint
	sourceLocDepth              float64 // eLORETA depth weighting
	sourceLocSaveStc            bool    // Save STCs for 3D plotting
	sourceLocConnMethod         int     // 0: aec, 1: wpli, 2: plv
	sourceLocSubject            string  // FreeSurfer subject name (e.g., sub-0001)
	sourceLocSubjectsDir        string  // FreeSurfer SUBJECTS_DIR override
	sourceLocTrans              string  // EEG↔MRI transform .fif
	sourceLocBem                string  // BEM solution .fif
	sourceLocMindistMm          float64 // MNE mindist (mm)
	sourceLocContrastEnabled    bool    // Enable condition contrasts from source trial features
	sourceLocContrastCondition  string  // Condition column for source contrasts
	sourceLocContrastA          string  // Condition A value for source contrasts
	sourceLocContrastB          string  // Condition B value for source contrasts
	sourceLocContrastMinTrials  int     // Minimum trials per condition for source contrasts
	sourceLocContrastWelchStats bool    // Emit Welch t/p statistics for source contrasts
	sourceLocFmriEnabled        bool    // Enable fMRI-informed source localization
	sourceLocFmriStatsMap       string  // Path to fMRI stats map NIfTI
	sourceLocFmriProvenance     int     // 0: independent, 1: same_dataset
	sourceLocFmriRequireProv    bool    // Require explicit provenance
	sourceLocFmriThreshold      float64 // Threshold (e.g., z>=3.1)
	sourceLocFmriTail           int     // 0: pos, 1: abs
	sourceLocFmriMinClusterVox  int     // Minimum cluster size (voxels)
	sourceLocFmriMinClusterMM3  float64 // Minimum cluster volume (mm^3); preferred when > 0
	sourceLocFmriMaxClusters    int     // Max clusters retained
	sourceLocFmriMaxVoxPerClus  int     // Max voxels sampled per cluster
	sourceLocFmriMaxTotalVox    int     // Max total voxels across clusters
	sourceLocFmriRandomSeed     int     // Random seed for voxel subsampling
	sourceLocFmriOutputSpace    int     // 0: cluster, 1: atlas, 2: dual

	// BEM/Trans generation options (Docker-based)
	sourceLocCreateTrans        bool // Auto-create coregistration transform via Docker
	sourceLocAllowIdentityTrans bool // Allow creating identity transform (DEBUG ONLY)
	sourceLocCreateBemModel     bool // Auto-create BEM model via Docker
	sourceLocCreateBemSolution  bool // Auto-create BEM solution via Docker

	// fMRI GLM contrast builder (for fMRI-informed mode)
	sourceLocFmriContrastEnabled          bool     // Build contrast from BOLD data (vs. load pre-computed)
	sourceLocFmriContrastType             int      // 0: t-test, 1: paired t-test, 2: F-test, 3: custom formula
	sourceLocFmriCondAColumn              string   // Condition A column (config/discovery default when empty)
	sourceLocFmriCondAValue               string   // Condition A value (e.g., "temp49p3", "1")
	sourceLocFmriCondBColumn              string   // Condition B column
	sourceLocFmriCondBValue               string   // Condition B value
	sourceLocFmriConditions               []string // Discovered conditions from fMRI events files
	sourceLocFmriCondIdx1                 int      // Index into discovered conditions for Condition A
	sourceLocFmriCondIdx2                 int      // Index into discovered conditions for Condition B
	sourceLocFmriContrastFormula          string   // Custom formula (e.g., "cond_a - cond_b")
	sourceLocFmriContrastName             string   // Contrast name (e.g., "contrast")
	sourceLocFmriRunsToInclude            string   // Comma-separated runs (e.g., "1,2,3")
	sourceLocFmriAutoDetectRuns           bool     // Auto-detect available BOLD runs
	sourceLocFmriHrfModel                 int      // 0: SPM, 1: FLOBS, 2: FIR
	sourceLocFmriDriftModel               int      // 0: none, 1: cosine, 2: polynomial
	sourceLocFmriHighPassHz               float64  // High-pass cutoff (Hz)
	sourceLocFmriLowPassHz                float64  // Low-pass cutoff (Hz)
	sourceLocFmriConditionScopeColumn     string   // Events column used by condition scope values
	sourceLocFmriConditionScopeTrialTypes string   // Optional: space-separated allow-list for condition selection
	sourceLocFmriPhaseColumn              string   // Events column used by phase filtering values
	sourceLocFmriPhaseScopeColumn         string   // Events column used to scope phase filtering to subset rows
	sourceLocFmriPhaseScopeValue          string   // Optional value in sourceLocFmriPhaseScopeColumn for scoped phase filtering
	sourceLocFmriStimPhasesToModel        string   // Optional: phase allow-list from sourceLocFmriPhaseColumn (empty = no scoping)
	sourceLocFmriClusterCorrection        bool     // Enable cluster-extent filtering heuristic (NOT cluster-level FWE correction)
	sourceLocFmriClusterPThreshold        float64  // Cluster-forming p-threshold
	sourceLocFmriOutputType               int      // 0: z-score, 1: t-stat, 2: cope, 3: beta
	sourceLocFmriResampleToFS             bool     // Auto-resample to FreeSurfer space
	sourceLocFmriInputSource              int      // 0: fmriprep, 1: bids_raw
	sourceLocFmriRequireFmriprep          bool     // Require fMRIPrep for contrast building

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

	// Microstates options
	microstatesNStates             int
	microstatesMinPeakDistanceMs   float64
	microstatesMaxGfpPeaksPerEpoch int
	microstatesMinDurationMs       float64
	microstatesGfpPeakProminence   float64
	microstatesRandomState         int
	microstatesFixedTemplatesPath  string

	// ERDS advanced options
	erdsUseLogRatio         bool    // Use dB instead of percent
	erdsMinBaselinePower    float64 // Minimum baseline power
	erdsMinActivePower      float64 // Minimum active power
	erdsOnsetThresholdSigma float64 // Baseline SD multiplier for ERD onset threshold
	erdsOnsetMinDurationMs  float64 // Sustained threshold crossing duration
	erdsReboundMinLatencyMs float64 // Minimum latency from ERD peak to rebound search
	erdsInferContralateral  bool    // Infer contralateral side when metadata is missing

	// Temporal feature selection (behavior pipeline)
	temporalFeaturePowerEnabled bool // Power temporal enabled
	temporalFeatureITPCEnabled  bool // ITPC temporal enabled
	temporalFeatureERDSEnabled  bool // ERDS temporal enabled

	// Time-frequency heatmap options
	tfHeatmapEnabled   bool   // Enable TF heatmap
	tfHeatmapFreqsSpec string // Frequencies for heatmap
	tfHeatmapTimeResMs int    // Time resolution in ms

	// Behavior pipeline advanced config
	predictorType           int     // 0=continuous, 1=binary, 2=categorical
	correlationMethod       string  // "spearman" or "pearson"
	robustCorrelation       int     // 0=none, 1=percentage_bend, 2=winsorized, 3=shepherd
	bootstrapSamples        int     // 0 = disabled, 1000+ recommended
	nPermutations           int     // For cluster tests
	rngSeed                 int     // 0 = use project default
	controlPredictor        bool    // Include predictor as covariate
	controlTrialOrder       bool    // Include trial order as covariate
	behaviorOutcomeColumn   string  // Canonical outcome column (blank=auto)
	behaviorPredictorColumn string  // Canonical predictor column (blank=auto)
	behaviorMinSamples      int     // 0=unset; behavior_analysis.min_samples.default
	fdrAlpha                float64 // FDR correction threshold
	behaviorConfigSection   int
	behaviorNJobs           int // -1 = all

	behaviorComputeChangeScores  bool
	behaviorComputeBayesFactors  bool
	behaviorComputeLosoStability bool
	behaviorValidateOnly         bool

	// Run adjustment (subject-level; optional)
	runAdjustmentEnabled               bool
	runAdjustmentColumn                string
	runAdjustmentIncludeInCorrelations bool
	runAdjustmentMaxDummies            int

	// Output options
	alsoSaveCsv       bool // Also save output tables as CSV files
	behaviorOverwrite bool // Overwrite existing output folders (if false, append timestamp)

	// Behavior advanced config section expansion (collapsed by default for compact UI)
	behaviorGroupGeneralExpanded           bool
	behaviorGroupTrialTableExpanded        bool
	behaviorGroupPredictorResidualExpanded bool
	behaviorGroupCorrelationsExpanded      bool
	behaviorGroupPredictorSensExpanded     bool
	behaviorGroupRegressionExpanded        bool
	behaviorGroupModelsExpanded            bool
	behaviorGroupStabilityExpanded         bool
	behaviorGroupConsistencyExpanded       bool
	behaviorGroupInfluenceExpanded         bool
	behaviorGroupReportExpanded            bool
	behaviorGroupConditionExpanded         bool
	behaviorGroupTemporalExpanded          bool
	behaviorGroupClusterExpanded           bool
	behaviorGroupMediationExpanded         bool
	behaviorGroupModerationExpanded        bool
	behaviorGroupMixedEffectsExpanded      bool
	behaviorGroupOutputExpanded            bool
	behaviorGroupStatsExpanded             bool
	behaviorGroupAnalysesExpanded          bool
	behaviorGroupAdvancedExpanded          bool

	// Trial table / predictor residual config (subject-level)
	trialTableFormat                      int // 0=parquet, 1=tsv
	trialTableAddLagFeatures              bool
	trialTableDisallowPositionalAlignment bool

	// Trial order validation (used when controlTrialOrder=true)
	trialOrderMaxMissingFraction float64

	featureSummariesEnabled bool

	// Feature QC (pre-statistics gating)
	featureQCEnabled                bool
	featureQCMaxMissingPct          float64
	featureQCMinVariance            float64
	featureQCCheckWithinRunVariance bool

	predictorResidualEnabled                 bool
	predictorResidualMethod                  int // 0=spline, 1=poly
	predictorResidualPolyDegree              int
	predictorResidualSplineDfCandidates      string // Comma-separated list (e.g., "3,4,5")
	predictorResidualModelCompareEnabled     bool
	predictorResidualModelComparePolyDegrees string // Comma-separated list (e.g., "2,3")
	predictorResidualMinSamples              int
	predictorResidualModelCompareMinSamples  int
	predictorResidualBreakpointEnabled       bool
	predictorResidualBreakpointCandidates    int
	predictorResidualBreakpointMinSamples    int
	predictorResidualBreakpointQlow          float64
	predictorResidualBreakpointQhigh         float64

	// Predictor residual cross-fit (out-of-run prediction)
	predictorResidualCrossfitEnabled     bool
	predictorResidualCrossfitGroupColumn string
	predictorResidualCrossfitNSplits     int
	predictorResidualCrossfitMethod      int // 0=spline, 1=poly
	predictorResidualCrossfitSplineKnots int

	// Regression
	regressionOutcome            int // 0=rating, 1=predictor_residual, 2=predictor
	regressionIncludePredictor   bool
	regressionTempControl        int // 0=linear, 1=rating_hat, 2=spline
	regressionTempSplineKnots    int
	regressionTempSplineQlow     float64
	regressionTempSplineQhigh    float64
	regressionTempSplineMinN     int
	regressionIncludeTrialOrder  bool
	regressionIncludePrev        bool
	regressionIncludeRunBlock    bool
	regressionIncludeInteraction bool
	regressionStandardize        bool
	regressionMinSamples         int
	regressionPrimaryUnit        int // 0=trial, 1=run_mean
	regressionPermutations       int
	regressionMaxFeatures        int // 0 = no limit

	// Models
	modelsIncludePredictor         bool
	modelsTempControl              int // 0=linear, 1=rating_hat, 2=spline
	modelsTempSplineKnots          int
	modelsTempSplineQlow           float64
	modelsTempSplineQhigh          float64
	modelsTempSplineMinN           int
	modelsIncludeTrialOrder        bool
	modelsIncludePrev              bool
	modelsIncludeRunBlock          bool
	modelsIncludeInteraction       bool
	modelsStandardize              bool
	modelsMinSamples               int
	modelsMaxFeatures              int
	modelsOutcomeValue             bool
	modelsOutcomePredictorResidual bool
	modelsOutcomePredictor         bool
	modelsOutcomeBinaryOutcome     bool
	modelsFamilyOLS                bool
	modelsFamilyRobust             bool
	modelsFamilyQuantile           bool
	modelsFamilyLogit              bool
	modelsBinaryOutcome            int // 0=binary_outcome, 1=rating_median
	modelsPrimaryUnit              int // 0=trial, 1=run_mean
	modelsForceTrialIIDAsymptotic  bool

	// Stability
	stabilityMethod      int // 0=spearman, 1=pearson
	stabilityOutcome     int // 0=auto, 1=rating, 2=predictor_residual
	stabilityGroupColumn int // 0=auto, 1=run, 2=block
	stabilityPartialTemp bool
	stabilityMinGroupN   int // 0=unset
	stabilityMaxFeatures int
	stabilityAlpha       float64

	// Consistency & influence
	consistencyEnabled                bool
	influenceOutcomeValue             bool
	influenceOutcomePredictorResidual bool
	influenceOutcomePredictor         bool
	influenceMaxFeatures              int
	influenceIncludePredictor         bool
	influenceTempControl              int // 0=linear, 1=rating_hat, 2=spline
	influenceTempSplineKnots          int
	influenceTempSplineQlow           float64
	influenceTempSplineQhigh          float64
	influenceTempSplineMinN           int
	influenceIncludeTrialOrder        bool
	influenceIncludeRunBlock          bool
	influenceIncludeInteraction       bool
	influenceStandardize              bool
	influenceCooksThreshold           float64 // 0 = default
	influenceLeverageThreshold        float64 // 0 = default

	// Correlations (trial-table)
	correlationsTypesSpec               string // Comma-separated list (e.g., "partial_cov_predictor,raw")
	correlationsUseCrossfitResidual     bool
	correlationsPrimaryUnit             int // 0=trial, 1=run_mean
	correlationsMinRuns                 int // minimum runs for run-mean correlations
	correlationsPreferPredictorResidual bool
	correlationsPermutations            int // 0=use global --n-perm
	correlationsPermutationPrimary      bool
	correlationsTargetColumn            string // Custom target column from events (dropdown)
	correlationsPowerSegment            string // Optional NamingSchema segment for ROI power correlations
	correlationsFeaturesSpec            string // Comma-separated feature filters for correlations
	groupLevelBlockPermutation          bool   // Use block-restricted permutations when block/run is available
	groupLevelTarget                    string // target column for multilevel correlations
	groupLevelControlPredictor          bool
	groupLevelControlTrialOrder         bool
	groupLevelControlRunEffects         bool
	groupLevelMaxRunDummies             int
	groupLevelAllowParametricFallback   bool

	// Predictor sensitivity
	predictorSensitivityMinTrials          int // 0=unset
	predictorSensitivityPrimaryUnit        int // 0=trial, 1=run_mean
	predictorSensitivityPermutations       int // 0=use global permutation setting
	predictorSensitivityPermutationPrimary bool
	predictorSensitivityFeaturesSpec       string // Comma-separated feature filters for predictor sensitivity

	// Report
	reportTopN int

	// Temporal
	temporalResolutionMs       int
	temporalCorrectionMethod   int // 0=fdr, 1=cluster
	temporalSmoothMs           int
	temporalTimeMinMs          int
	temporalTimeMaxMs          int
	temporalTargetColumn       string // events.tsv column to correlate against (empty=default rating)
	temporalSplitByCondition   bool   // If true, compute separate correlations per condition value
	temporalConditionColumn    string // Column to split by (empty = use event_columns.binary_outcome)
	temporalConditionValues    string // Values to compute (empty = all unique values)
	temporalIncludeROIAverages bool   // Include ROI-averaged rows in output
	temporalIncludeTFGrid      bool   // Include individual frequency (TF grid) rows
	temporalFeaturesSpec       string // Comma-separated feature filters for temporal
	// ITPC-specific parameters
	temporalITPCBaselineCorrection bool    // Subtract baseline ITPC
	temporalITPCBaselineMin        float64 // Baseline window start
	temporalITPCBaselineMax        float64 // Baseline window end
	// ERDS-specific parameters
	temporalERDSBaselineMin float64 // ERDS baseline window start (seconds)
	temporalERDSBaselineMax float64 // ERDS baseline window end (seconds)
	temporalERDSMethod      int     // 0=percent, 1=zscore

	// Mixed effects (group-level; still configurable)
	mixedEffectsType      int // 0=intercept, 1=intercept_slope
	mixedIncludePredictor bool

	// Mediation
	mediationMinEffect          float64
	mediationPermutationPrimary bool

	// Condition extras
	conditionFailFast           bool
	conditionPermutationPrimary bool
	conditionPrimaryUnit        int // 0=trial, 1=run_mean
	conditionWindowPrimaryUnit  int // 0=trial, 1=run_mean
	conditionWindowMinSamples   int
	// Cluster-specific
	clusterThreshold       float64 // Forming threshold for clusters
	clusterMinSize         int     // Minimum cluster size
	clusterTail            int     // 0=two-tailed, 1=upper, -1=lower
	clusterConditionColumn string  // events.tsv column to split by (empty=event_columns.binary_outcome)
	clusterConditionValues string  // Exactly 2 values (space/comma-separated) to compare
	clusterFeaturesSpec    string  // Comma-separated feature filters for cluster
	// Mediation-specific
	mediationBootstrap           int    // Bootstrap iterations for mediation
	mediationMaxMediatorsEnabled bool   // Enable max mediators limit
	mediationMaxMediators        int    // Max mediators to test
	mediationPermutations        int    // Permutation iterations for mediation (0=disabled)
	mediationFeaturesSpec        string // Comma-separated feature filters for mediation
	// Moderation-specific
	moderationMaxFeaturesEnabled bool // Enable max features limit
	moderationMaxFeatures        int  // Max features for moderation
	moderationMinSamples         int
	moderationPermutations       int // Permutation iterations for moderation (0=disabled)
	moderationPermutationPrimary bool
	moderationFeaturesSpec       string // Comma-separated feature filters for moderation
	// Mixed effects-specific
	mixedMaxFeatures int // Max features for mixed effects
	// Condition-specific
	conditionEffectThreshold float64 // Min effect size to report
	conditionCompareColumn   string  // Column to use for condition split (empty=event_columns.binary_outcome)
	conditionCompareWindows  string  // Time windows to compare (e.g., "baseline active")
	conditionCompareValues   string  // Values in the column to compare (e.g., "0,1" or "cond_a,cond_b")
	conditionCompareLabels   string  // Optional labels aligned to compare values
	conditionMinTrials       int     // 0=unset; condition.min_trials_per_condition
	conditionOverwrite       bool    // Overwrite existing condition effects files
	conditionFeaturesSpec    string  // Comma-separated feature filters for condition

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
	mlFmriSigContrastName       string // e.g., contrast
	mlFmriSigSignatureIndex     int    // Reserved for signature picker UI; config value is used by default.
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

	mlPlotsEnabled     bool
	mlPlotFormatsSpec  string
	mlPlotDPI          int
	mlPlotTopNFeatures int
	mlPlotDiagnostics  bool

	mlUncertaintyAlpha float64
	mlPermNRepeats     int

	// Machine Learning model hyperparameters
	elasticNetAlphaGrid   string // alpha grid as comma-separated values
	elasticNetL1RatioGrid string // l1_ratio grid as comma-separated values
	ridgeAlphaGrid        string // ridge alpha grid as comma-separated values
	rfNEstimators         int    // Random forest n_estimators
	rfMaxDepthGrid        string // max_depth grid as comma-separated values (use "null" for None)
	varianceThresholdGrid string // variance_threshold grid (e.g. 0.0 or 0.0,0.01,0.1); use 0.0 only for small train folds

	// ML Preprocessing
	mlImputer                     int     // 0: median, 1: mean, 2: most_frequent
	mlPowerTransformerMethod      int     // 0: yeo-johnson, 1: box-cox
	mlPowerTransformerStandardize bool    // Standardize after power transform
	mlPCAEnabled                  bool    // Enable PCA dimensionality reduction
	mlPCANComponents              float64 // Variance threshold (e.g. 0.95) or int
	mlPCAWhiten                   bool    // PCA whitening
	mlPCASvdSolver                int     // 0: auto, 1: full, 2: randomized
	mlPCARngSeed                  int     // PCA random state
	mlDeconfound                  bool    // Enable covariate deconfounding
	mlFeatureSelectionPercentile  float64 // Percentage of features to keep
	mlEnsembleCalibrate           bool    // Calibrate SVM/RF probability outputs using CalibratedClassifierCV
	mlSpatialRegionsAllowed       string  // Comma-separated list of ROIs
	mlClassificationResampler     int     // 0: none, 1: undersample, 2: smote
	mlClassificationResamplerSeed int     // Random seed for resampler
	mlGroupPreprocessingExpanded  bool    // UI expansion state
	mlGroupDataExpanded           bool    // Data & Features section
	mlGroupModelExpanded          bool    // Model & Hyperparameters section
	mlGroupTrainingExpanded       bool    // Training & CV section
	mlGroupOutputExpanded         bool    // Output & Plots section

	// ML Model Hyperparameters - SVM
	mlSvmKernel      int    // 0: rbf, 1: linear, 2: poly
	mlSvmCGrid       string // C grid (comma-separated)
	mlSvmGammaGrid   string // gamma grid (comma-separated)
	mlSvmClassWeight int    // 0: balanced, 1: none

	// ML Model Hyperparameters - Logistic Regression
	mlLrPenalty     int    // 0: l2, 1: l1, 2: elasticnet
	mlLrCGrid       string // C grid (comma-separated)
	mlLrL1RatioGrid string // L1 ratio grid for elasticnet penalty
	mlLrMaxIter     int    // Max iterations
	mlLrClassWeight int    // 0: balanced, 1: none

	// ML Model Hyperparameters - Random Forest extras
	mlRfMinSamplesSplitGrid string // min_samples_split grid (comma-separated)
	mlRfMinSamplesLeafGrid  string // min_samples_leaf grid (comma-separated)
	mlRfBootstrap           bool   // Bootstrap sampling
	mlRfClassWeight         int    // 0: balanced, 1: balanced_subsample, 2: none

	// ML Model Hyperparameters - CNN
	mlGroupCNNExpanded bool    // UI expansion state
	mlCnnFilters1      int     // Conv1 filters
	mlCnnFilters2      int     // Conv2 filters
	mlCnnKernelSize1   int     // Conv1 kernel size
	mlCnnKernelSize2   int     // Conv2 kernel size
	mlCnnPoolSize      int     // Max pool size
	mlCnnDenseUnits    int     // Dense layer units
	mlCnnDropoutConv   float64 // Conv dropout rate
	mlCnnDropoutDense  float64 // Dense dropout rate
	mlCnnBatchSize     int     // Training batch size
	mlCnnEpochs        int     // Training epochs
	mlCnnLearningRate  float64 // Learning rate
	mlCnnPatience      int     // Early stopping patience
	mlCnnMinDelta      float64 // Early stopping min delta
	mlCnnL2Lambda      float64 // L2 regularization
	mlCnnRandomSeed    int     // CNN random seed

	// ML CV / Evaluation / Analysis
	mlCvHygieneEnabled               bool    // CV hygiene toggle
	mlCvPermutationScheme            int     // 0: within_subject, 1: within_subject_within_block
	mlCvMinValidPermFraction         float64 // Min valid permutation fraction
	mlCvDefaultNBins                 int     // Default stratification bins
	mlEvalCIMethod                   int     // 0: bootstrap, 1: fixed_effects
	mlEvalSubjectWeighting           int     // 0: equal, 1: trial_count
	mlEvalBootstrapIterations        int     // Bootstrap iterations
	mlDataCovariatesStrict           bool    // Error on missing covariates
	mlDataMaxExcludedSubjectFraction float64 // Max excluded subject fraction
	mlIncrementalBaselineAlpha       float64 // Baseline model alpha
	mlIncrementalRequireBaselinePred bool    // Require baseline predictors in incremental_validity
	mlInterpretabilityGroupedOutputs bool    // Grouped importance tables
	mlTimeGenMinSubjects             int     // TG min subjects
	mlTimeGenMinValidPermFraction    float64 // TG min valid perm fraction
	mlClassMinSubjectsForAUC         int     // Min subjects for AUC inference
	mlClassMaxFailedFoldFraction     float64 // Max failed fold fraction
	mlTargetsStrictRegressionCont    bool    // Error on binary-like regression target

	// EEG Preprocessing missing
	prepEcgChannels            string // ECG channel list (e.g., "ECG")
	prepAutorejectNInterpolate string // Autoreject interpolation candidates (e.g., "4,8,16")

	// Alignment
	alignAllowMisalignedTrim bool // Allow misaligned trim
	alignMinAlignmentSamples int  // Minimum alignment samples
	alignTrimToFirstVolume   bool // Trim EEG to first volume marker
	alignFmriOnsetReference  int  // 0: as_is, 1: first_volume, 2: scanner_trigger

	// Event Column Mapping
	eventColPredictor          string // Predictor column candidates (comma-separated)
	eventColOutcome            string // Rating column candidates (comma-separated)
	eventColBinaryOutcome      string // Binary outcome column candidates (comma-separated)
	eventColCondition          string // Condition label column candidates (comma-separated)
	eventColRequired           string // Required logical event groups (comma-separated)
	conditionPreferredPrefixes string // Preferred trigger prefixes for auto condition detection (comma-separated)

	// Per-Family Spatial Transforms (0: none, 1: csd, 2: laplacian)
	spatialTransformPerFamilyConnectivity int
	spatialTransformPerFamilyItpc         int
	spatialTransformPerFamilyPac          int
	spatialTransformPerFamilyPower        int
	spatialTransformPerFamilyAperiodic    int
	spatialTransformPerFamilyBursts       int
	spatialTransformPerFamilyErds         int
	spatialTransformPerFamilyComplexity   int
	spatialTransformPerFamilyRatios       int
	spatialTransformPerFamilyAsymmetry    int
	spatialTransformPerFamilySpectral     int
	spatialTransformPerFamilyErp          int
	spatialTransformPerFamilyQuality      int
	spatialTransformPerFamilyMicrostates  int

	// Change Scores Config
	changeScoresTransform   int    // 0: difference, 1: percent, 2: log_ratio
	changeScoresWindowPairs string // Window pairs (e.g., "baseline:active")

	// ITPC/PAC Segment Validity
	itpcMinSegmentSec   float64 // Min segment duration for ITPC
	itpcMinCyclesAtFmin float64 // Min cycles at lowest freq for ITPC
	pacMinSegmentSec    float64 // Min segment duration for PAC
	pacMinCyclesAtFmin  float64 // Min cycles at lowest freq for PAC
	pacSurrogateMethod  int     // 0: trial_shuffle, 1: circular_shift, 2: swap_phase_amp, 3: time_shift

	// Aperiodic Missing
	aperiodicMaxFreqResolutionHz float64 // Max PSD freq resolution threshold
	aperiodicMultitaperAdaptive  bool    // Multitaper adaptive flag

	// Directed Connectivity Missing
	directedConnMinSamplesPerMvarParam int // Auto-reduce MVAR order for short windows

	// ERDS Condition Markers
	erdsConditionMarkerBands       string  // Bands for contralateral condition markers (comma-separated)
	erdsLateralityColumns          string  // Column names for stimulation side (comma-separated)
	erdsSomatosensoryLeftChannels  string  // Left somatosensory channels (comma-separated)
	erdsSomatosensoryRightChannels string  // Right somatosensory channels (comma-separated)
	erdsOnsetMinThresholdPercent   float64 // Absolute min ERD onset threshold
	erdsReboundThresholdSigma      float64 // Rebound threshold sigma
	erdsReboundMinThresholdPercent float64 // Absolute min rebound threshold

	// Microstates Missing
	microstatesAssignFromGfpPeaks bool // Assign states at GFP peaks then backfit

	// Behavior Statistics
	behaviorStatsTempControl               int     // 0: spline, 1: linear
	behaviorStatsAllowIIDTrials            bool    // Allow i.i.d. trial assumptions
	behaviorStatsHierarchicalFDR           bool    // Hierarchical FDR correction
	behaviorStatsComputeReliability        bool    // Compute reliability
	statisticsAlpha                        float64 // Global fallback alpha for shared statistics helpers
	behaviorPermScheme                     int     // 0: circular_shift, 1: shuffle
	behaviorPermGroupColumnPreference      string  // Group column preference order (comma-separated)
	behaviorExcludeNonTrialwiseFeatures    bool    // Drop broadcast features
	behaviorFeatureRegistryFilesJSON       string  // behavior_analysis.feature_registry.files JSON
	behaviorFeatureRegistrySourceJSON      string  // behavior_analysis.feature_registry.source_to_feature_type JSON
	behaviorFeatureRegistryHierarchyJSON   string  // behavior_analysis.feature_registry.feature_type_hierarchy JSON
	behaviorFeatureRegistryPatternsJSON    string  // behavior_analysis.feature_registry.feature_patterns JSON
	behaviorFeatureRegistryClassifiersJSON string  // behavior_analysis.feature_registry.feature_classifiers JSON

	// Global Statistics & Validation
	globalNBootstrap                int     // Global bootstrap count
	clusterCorrectionEnabled        bool    // Global cluster correction enabled
	clusterCorrectionAlpha          float64 // Cluster correction alpha
	clusterCorrectionMinClusterSize int     // Min cluster size
	clusterCorrectionTailGlobal     int     // 0: two-tailed, 1: upper, -1: lower
	validationMinEpochs             int     // Min epochs for validation
	validationMinChannels           int     // Min channels
	validationMaxAmplitudeUv        float64 // Max amplitude threshold

	// System / IO
	ioPredictorRange             string  // Valid predictor range (e.g., "35.0,55.0")
	ioMaxMissingChannelsFraction float64 // Max missing channels fraction

	// TFR parameters (for features pipeline)
	tfrFreqMin       float64 // Min frequency for TFR
	tfrFreqMax       float64 // Max frequency for TFR
	tfrNFreqs        int     // Number of frequency bins
	tfrMinCycles     float64 // Minimum cycles for wavelet
	tfrMaxCycles     float64 // Maximum cycles for wavelet
	tfrNCyclesFactor float64 // Cycles factor
	tfrDecimPower    int     // Decimation for power TFR
	tfrDecimPhase    int     // Decimation for phase TFR
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
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Proceed to next step"},
		{Key: "?", Description: "Toggle help"},
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
		featGroupMicrostatesExpanded:      false,
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
		complexityPEOrder:     3,
		complexityPEDelay:     1,
		complexitySampEnOrder: 2,
		complexitySampEnR:     0.2,
		complexityMSEScaleMin: 1,
		complexityMSEScaleMax: 20,
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
		spectralRatioPairsSpec:     "theta:beta,theta:alpha,alpha:beta,delta:alpha,delta:theta",
		spectralSegmentsSpec:       "baseline",
		spectralPsdAdaptive:        false,
		spectralMultitaperAdaptive: false,
		// Connectivity defaults
		connOutputLevel:     0,
		connGraphMetrics:    false,
		connGraphProp:       0.1,
		connWindowLen:       1.0,
		connWindowStep:      0.5,
		connAECMode:         0,
		connMode:            0,
		connAECAbsolute:     true,
		connEnableAEC:       true,
		connNFreqsPerBand:   8,
		connNCycles:         0.0,
		connDecim:           1,
		connMinSegSamples:   50,
		connSmallWorldNRand: 100,
		// Scientific validity defaults (new)
		itpcMethod:                1,    // 1: fold_global (CV-safe default)
		itpcConditionColumn:       "",   // Empty = not using condition-based ITPC
		itpcConditionValues:       "",   // Empty = all unique values
		itpcMinTrialsPerCondition: 10,   // Minimum trials per condition
		aperiodicMinSegmentSec:    2.0,  // 2.0s minimum for stable fits
		connAECOutput:             0,    // 0: r only (raw)
		connForceWithinEpochML:    true, // Force within_epoch for CV-safety

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
		connDynamicEnabled:         false,
		connDynamicMeasures:        0, // 0: wpli+aec
		connDynamicAutocorrLag:     1,
		connDynamicMinWindows:      3,
		connDynamicIncludeROIPairs: true,
		connDynamicStateEnabled:    true,
		connDynamicStateNStates:    3,
		connDynamicStateMinWindows: 8,
		connDynamicStateRandomSeed: -1,

		// Directed connectivity defaults
		directedConnMeasures:          make(map[int]bool),
		directedConnEnabled:           false, // Disabled by default (opt-in)
		directedConnOutputLevel:       0,     // 0: full
		directedConnMvarOrder:         10,    // MVAR model order
		directedConnNFreqs:            16,    // Number of frequency bins
		directedConnMinSegSamples:     100,   // Minimum segment samples
		featGroupDirectedConnExpanded: false,

		// Source localization defaults
		sourceLocEnabled:            false, // Disabled by default (opt-in, requires fsaverage)
		sourceLocMethod:             0,     // 0: lcmv
		sourceLocSpacing:            1,     // 1: oct6 (default)
		sourceLocParc:               0,     // 0: aparc (Desikan-Killiany)
		sourceLocReg:                0.05,  // LCMV regularization
		sourceLocSnr:                3.0,   // eLORETA SNR
		sourceLocLoose:              0.2,   // eLORETA loose constraint
		sourceLocDepth:              0.8,   // eLORETA depth weighting
		sourceLocConnMethod:         0,     // 0: aec
		sourceLocSubject:            "",
		sourceLocSubjectsDir:        "",
		sourceLocTrans:              "",
		sourceLocBem:                "",
		sourceLocMindistMm:          5.0,
		sourceLocContrastEnabled:    false,
		sourceLocContrastCondition:  "",
		sourceLocContrastA:          "",
		sourceLocContrastB:          "",
		sourceLocContrastMinTrials:  5,
		sourceLocContrastWelchStats: true,
		sourceLocFmriEnabled:        false,
		sourceLocFmriStatsMap:       "",
		sourceLocFmriProvenance:     0,
		sourceLocFmriRequireProv:    true,
		sourceLocFmriThreshold:      3.1,
		sourceLocFmriTail:           0, // 0: pos
		sourceLocFmriMinClusterVox:  50,
		sourceLocFmriMinClusterMM3:  400.0,
		sourceLocFmriMaxClusters:    20,
		sourceLocFmriMaxVoxPerClus:  2000,
		sourceLocFmriMaxTotalVox:    20000,
		sourceLocFmriRandomSeed:     0,
		sourceLocFmriOutputSpace:    2, // 2: dual

		// fMRI GLM contrast builder defaults
		sourceLocFmriContrastEnabled:          false,
		sourceLocFmriContrastType:             0, // 0: t-test
		sourceLocFmriCondAColumn:              "",
		sourceLocFmriCondAValue:               "",
		sourceLocFmriCondBColumn:              "",
		sourceLocFmriCondBValue:               "",
		sourceLocFmriContrastFormula:          "",
		sourceLocFmriContrastName:             "contrast",
		sourceLocFmriRunsToInclude:            "",
		sourceLocFmriAutoDetectRuns:           true,
		sourceLocFmriHrfModel:                 0,     // 0: SPM
		sourceLocFmriDriftModel:               1,     // 1: cosine
		sourceLocFmriHighPassHz:               0.008, // 128s period
		sourceLocFmriLowPassHz:                0.0,
		sourceLocFmriConditionScopeColumn:     "",
		sourceLocFmriConditionScopeTrialTypes: "",
		sourceLocFmriPhaseColumn:              "",
		sourceLocFmriPhaseScopeColumn:         "",
		sourceLocFmriPhaseScopeValue:          "",
		sourceLocFmriStimPhasesToModel:        "", // no default scoping
		sourceLocFmriClusterCorrection:        true,
		sourceLocFmriClusterPThreshold:        0.001,
		sourceLocFmriOutputType:               0, // 0: z-score
		sourceLocFmriResampleToFS:             true,
		sourceLocFmriInputSource:              0, // 0: fmriprep
		sourceLocFmriRequireFmriprep:          true,

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

		// Microstates defaults
		microstatesNStates:             4,
		microstatesMinPeakDistanceMs:   10.0,
		microstatesMaxGfpPeaksPerEpoch: 400,
		microstatesMinDurationMs:       20.0,
		microstatesGfpPeakProminence:   0.0,
		microstatesRandomState:         42,
		microstatesFixedTemplatesPath:  "",

		// ERDS defaults
		erdsUseLogRatio:         false,
		erdsMinBaselinePower:    1.0e-12,
		erdsMinActivePower:      1.0e-12,
		erdsOnsetThresholdSigma: 1.0,
		erdsOnsetMinDurationMs:  30.0,
		erdsReboundMinLatencyMs: 100.0,
		erdsInferContralateral:  true,

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
		minEpochsForFeatures:    10,
		featAnalysisMode:        0,
		aggregationMethod:       0,
		featureTmin:             -7.0,
		featureTmax:             15.0,
		featComputeChangeScores: true,
		featSaveTfrWithSidecar:  false,
		featNJobsBands:          -1,
		featNJobsConnectivity:   -1,
		featNJobsAperiodic:      -1,
		featNJobsComplexity:     -1,
		featAlsoSaveCsv:         false,

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
		predictorType:           0, // continuous
		correlationMethod:       "spearman",
		robustCorrelation:       0,
		bootstrapSamples:        1000,
		nPermutations:           1000,
		rngSeed:                 0,
		controlPredictor:        true,
		controlTrialOrder:       true,
		behaviorOutcomeColumn:   "",
		behaviorPredictorColumn: "",
		behaviorMinSamples:      0,
		fdrAlpha:                0.05,
		behaviorConfigSection:   0,
		behaviorNJobs:           -1,

		behaviorComputeChangeScores:        true,
		behaviorComputeBayesFactors:        false,
		behaviorComputeLosoStability:       true,
		behaviorValidateOnly:               false,
		behaviorOverwrite:                  true, // Default: overwrite existing outputs
		runAdjustmentEnabled:               false,
		runAdjustmentColumn:                "run_id",
		runAdjustmentIncludeInCorrelations: true,
		runAdjustmentMaxDummies:            20,

		trialTableFormat:                      1,
		trialTableAddLagFeatures:              true,
		trialTableDisallowPositionalAlignment: false,
		trialOrderMaxMissingFraction:          0.1,

		featureSummariesEnabled:         true,
		featureQCEnabled:                false,
		featureQCMaxMissingPct:          0.2,
		featureQCMinVariance:            1e-10,
		featureQCCheckWithinRunVariance: true,

		predictorResidualEnabled:                 true,
		predictorResidualMethod:                  0,
		predictorResidualPolyDegree:              2,
		predictorResidualSplineDfCandidates:      "3,4,5",
		predictorResidualModelCompareEnabled:     true,
		predictorResidualModelComparePolyDegrees: "2,3",
		predictorResidualMinSamples:              10,
		predictorResidualModelCompareMinSamples:  10,
		predictorResidualBreakpointEnabled:       true,
		predictorResidualBreakpointCandidates:    15,
		predictorResidualBreakpointMinSamples:    12,
		predictorResidualBreakpointQlow:          0.15,
		predictorResidualBreakpointQhigh:         0.85,
		predictorResidualCrossfitEnabled:         false,
		predictorResidualCrossfitGroupColumn:     "",
		predictorResidualCrossfitNSplits:         5,
		predictorResidualCrossfitMethod:          0,
		predictorResidualCrossfitSplineKnots:     5,

		regressionOutcome:            0,
		regressionIncludePredictor:   true,
		regressionTempControl:        0,
		regressionTempSplineKnots:    4,
		regressionTempSplineQlow:     0.05,
		regressionTempSplineQhigh:    0.95,
		regressionTempSplineMinN:     12,
		regressionIncludeTrialOrder:  true,
		regressionIncludePrev:        false,
		regressionIncludeRunBlock:    true,
		regressionIncludeInteraction: true,
		regressionStandardize:        true,
		regressionMinSamples:         15,
		regressionPrimaryUnit:        0,
		regressionPermutations:       0,
		regressionMaxFeatures:        0,

		modelsIncludePredictor:         true,
		modelsTempControl:              0,
		modelsTempSplineKnots:          4,
		modelsTempSplineQlow:           0.05,
		modelsTempSplineQhigh:          0.95,
		modelsTempSplineMinN:           12,
		modelsIncludeTrialOrder:        true,
		modelsIncludePrev:              false,
		modelsIncludeRunBlock:          true,
		modelsIncludeInteraction:       true,
		modelsStandardize:              true,
		modelsMinSamples:               20,
		modelsMaxFeatures:              100,
		modelsOutcomeValue:             true,
		modelsOutcomePredictorResidual: true,
		modelsOutcomePredictor:         false,
		modelsOutcomeBinaryOutcome:     false,
		modelsFamilyOLS:                true,
		modelsFamilyRobust:             true,
		modelsFamilyQuantile:           true,
		modelsFamilyLogit:              true,
		modelsBinaryOutcome:            0,
		modelsPrimaryUnit:              0,
		modelsForceTrialIIDAsymptotic:  false,

		stabilityMethod:      0,
		stabilityOutcome:     0,
		stabilityGroupColumn: 0,
		stabilityPartialTemp: true,
		stabilityMinGroupN:   0,
		stabilityMaxFeatures: 50,
		stabilityAlpha:       0.05,

		consistencyEnabled:                true,
		influenceOutcomeValue:             true,
		influenceOutcomePredictorResidual: true,
		influenceOutcomePredictor:         false,
		influenceMaxFeatures:              20,
		influenceIncludePredictor:         true,
		influenceTempControl:              0,
		influenceTempSplineKnots:          4,
		influenceTempSplineQlow:           0.05,
		influenceTempSplineQhigh:          0.95,
		influenceTempSplineMinN:           12,
		influenceIncludeTrialOrder:        true,
		influenceIncludeRunBlock:          true,
		influenceIncludeInteraction:       false,
		influenceStandardize:              true,
		influenceCooksThreshold:           0.0,
		influenceLeverageThreshold:        0.0,

		correlationsTypesSpec:                  "partial_cov_predictor",
		correlationsUseCrossfitResidual:        false,
		correlationsPrimaryUnit:                0,
		correlationsMinRuns:                    3,
		correlationsPreferPredictorResidual:    false,
		correlationsPermutations:               0,
		correlationsPermutationPrimary:         false,
		correlationsPowerSegment:               "",
		correlationsFeaturesSpec:               "",
		groupLevelBlockPermutation:             true,
		groupLevelTarget:                       "",
		groupLevelControlPredictor:             true,
		groupLevelControlTrialOrder:            true,
		groupLevelControlRunEffects:            false,
		groupLevelMaxRunDummies:                20,
		groupLevelAllowParametricFallback:      false,
		predictorSensitivityMinTrials:          0,
		predictorSensitivityPrimaryUnit:        0,
		predictorSensitivityPermutations:       0,
		predictorSensitivityPermutationPrimary: true,
		predictorSensitivityFeaturesSpec:       "",

		reportTopN:                     15,
		temporalResolutionMs:           50,
		temporalCorrectionMethod:       0,
		temporalSmoothMs:               100,
		temporalTimeMinMs:              -200,
		temporalTimeMaxMs:              1000,
		temporalTargetColumn:           "",
		temporalSplitByCondition:       true,
		temporalConditionColumn:        "",
		temporalConditionValues:        "",
		temporalIncludeROIAverages:     true,
		temporalIncludeTFGrid:          true,
		temporalFeaturesSpec:           "",
		temporalITPCBaselineCorrection: true,
		temporalITPCBaselineMin:        -0.5,
		temporalITPCBaselineMax:        -0.01,
		// ERDS defaults
		temporalERDSBaselineMin:     -0.5,
		temporalERDSBaselineMax:     -0.1,
		temporalERDSMethod:          0, // 0=percent, 1=zscore
		mixedEffectsType:            0,
		mixedIncludePredictor:       true,
		mediationMinEffect:          0.05,
		mediationPermutationPrimary: true,
		// Cluster defaults
		clusterThreshold:       0.05,
		clusterMinSize:         2,
		clusterTail:            0,
		clusterConditionColumn: "",
		clusterConditionValues: "",
		clusterFeaturesSpec:    "",
		// Mediation defaults
		mediationBootstrap:           1000,
		mediationMaxMediatorsEnabled: true,
		mediationMaxMediators:        20,
		mediationPermutations:        0, // Disabled by default
		mediationFeaturesSpec:        "",
		// Moderation defaults
		moderationMaxFeaturesEnabled: true,
		moderationMaxFeatures:        50,
		moderationMinSamples:         15,
		moderationPermutations:       0, // Disabled by default
		moderationPermutationPrimary: true,
		moderationFeaturesSpec:       "",
		// Mixed effects defaults
		mixedMaxFeatures: 50,
		// Condition defaults
		conditionEffectThreshold:    0.5,
		conditionFailFast:           true,
		conditionPermutationPrimary: false,
		conditionPrimaryUnit:        0,
		conditionWindowPrimaryUnit:  0,
		conditionWindowMinSamples:   10,
		conditionCompareLabels:      "",
		conditionMinTrials:          0,
		conditionFeaturesSpec:       "",
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
		mlFmriSigContrastName:       "contrast",
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
		mlPlotsEnabled:              true,
		mlPlotFormatsSpec:           "png",
		mlPlotDPI:                   300,
		mlPlotTopNFeatures:          20,
		mlPlotDiagnostics:           true,
		mlUncertaintyAlpha:          0.1,
		mlPermNRepeats:              10,
		// Hyperparameter defaults mirror eeg_pipeline/utils/config/eeg_config.yaml
		elasticNetAlphaGrid:   "0.001,0.01,0.1,1,10",
		elasticNetL1RatioGrid: "0.2,0.5,0.8",
		varianceThresholdGrid: "0.0,0.01,0.1",
		ridgeAlphaGrid:        "0.01,0.1,1,10,100",
		rfNEstimators:         500,
		rfMaxDepthGrid:        "5,10,20,null",

		// ML Preprocessing defaults
		mlImputer:                     0, // 0: median
		mlPowerTransformerMethod:      0, // 0: yeo-johnson
		mlPowerTransformerStandardize: true,
		mlPCAEnabled:                  false,
		mlPCANComponents:              0.95,
		mlPCAWhiten:                   false,
		mlPCASvdSolver:                0, // 0: auto
		mlPCARngSeed:                  0,
		mlDeconfound:                  false,
		mlSpatialRegionsAllowed:       "",
		mlClassificationResampler:     0, // 0: none
		mlClassificationResamplerSeed: 42,
		mlGroupDataExpanded:           true,
		mlGroupModelExpanded:          true,
		mlGroupTrainingExpanded:       true,
		mlGroupOutputExpanded:         true,

		// ML SVM defaults
		mlSvmKernel:      0, // 0: rbf
		mlSvmCGrid:       "0.01,0.1,1,10,100",
		mlSvmGammaGrid:   "scale,0.001,0.01,0.1",
		mlSvmClassWeight: 0, // 0: balanced

		// ML Logistic Regression defaults
		mlLrPenalty:     0, // 0: l2
		mlLrCGrid:       "0.01,0.1,1,10,100",
		mlLrL1RatioGrid: "0.1,0.5,0.9",
		mlLrMaxIter:     1000,
		mlLrClassWeight: 0, // 0: balanced

		// ML Random Forest extras defaults
		mlRfMinSamplesSplitGrid: "2,5,10",
		mlRfMinSamplesLeafGrid:  "1,2,4",
		mlRfBootstrap:           true,
		mlRfClassWeight:         0, // 0: balanced

		// ML CNN defaults
		mlCnnFilters1:     32,
		mlCnnFilters2:     64,
		mlCnnKernelSize1:  3,
		mlCnnKernelSize2:  3,
		mlCnnPoolSize:     2,
		mlCnnDenseUnits:   128,
		mlCnnDropoutConv:  0.25,
		mlCnnDropoutDense: 0.5,
		mlCnnBatchSize:    32,
		mlCnnEpochs:       100,
		mlCnnLearningRate: 0.001,
		mlCnnPatience:     10,
		mlCnnMinDelta:     0.001,
		mlCnnL2Lambda:     0.01,
		mlCnnRandomSeed:   42,

		// ML CV / Evaluation / Analysis defaults
		mlCvHygieneEnabled:               true,
		mlCvPermutationScheme:            0, // 0: within_subject
		mlCvMinValidPermFraction:         0.8,
		mlCvDefaultNBins:                 5,
		mlEvalCIMethod:                   0, // 0: bootstrap
		mlEvalSubjectWeighting:           0, // 0: equal
		mlEvalBootstrapIterations:        1000,
		mlDataCovariatesStrict:           false,
		mlDataMaxExcludedSubjectFraction: 0.2,
		mlIncrementalBaselineAlpha:       0.05,
		mlIncrementalRequireBaselinePred: true,
		mlInterpretabilityGroupedOutputs: true,
		mlTimeGenMinSubjects:             5,
		mlTimeGenMinValidPermFraction:    0.5,
		mlClassMinSubjectsForAUC:         10,
		mlClassMaxFailedFoldFraction:     0.2,
		mlTargetsStrictRegressionCont:    true,

		// EEG Preprocessing missing defaults
		prepEcgChannels:            "",
		prepAutorejectNInterpolate: "4,8,16",

		// Alignment defaults
		alignAllowMisalignedTrim: false,
		alignMinAlignmentSamples: 5,
		alignTrimToFirstVolume:   true,
		alignFmriOnsetReference:  0, // 0: as_is

		// Event Column Mapping defaults
		eventColPredictor:          "predictor",
		eventColOutcome:            "outcome",
		eventColBinaryOutcome:      "binary_outcome,outcome_binary,label",
		eventColCondition:          "condition,trial_type",
		eventColRequired:           "outcome",
		conditionPreferredPrefixes: "",

		// Per-Family Spatial Transforms defaults (all 0 = none / inherit global)
		spatialTransformPerFamilyConnectivity: 0,
		spatialTransformPerFamilyItpc:         0,
		spatialTransformPerFamilyPac:          0,
		spatialTransformPerFamilyPower:        0,
		spatialTransformPerFamilyAperiodic:    0,
		spatialTransformPerFamilyBursts:       0,
		spatialTransformPerFamilyErds:         0,
		spatialTransformPerFamilyComplexity:   0,
		spatialTransformPerFamilyRatios:       0,
		spatialTransformPerFamilyAsymmetry:    0,
		spatialTransformPerFamilySpectral:     0,
		spatialTransformPerFamilyErp:          0,
		spatialTransformPerFamilyQuality:      0,
		spatialTransformPerFamilyMicrostates:  0,

		// Change Scores defaults
		changeScoresTransform:   0, // 0: difference
		changeScoresWindowPairs: "baseline:active",

		// ITPC/PAC Segment Validity defaults
		itpcMinSegmentSec:   1.0,
		itpcMinCyclesAtFmin: 3.0,
		pacMinSegmentSec:    2.0,
		pacMinCyclesAtFmin:  3.0,
		pacSurrogateMethod:  0, // 0: trial_shuffle

		// Aperiodic Missing defaults
		aperiodicMaxFreqResolutionHz: 0.5,
		aperiodicMultitaperAdaptive:  false,

		// Directed Connectivity Missing defaults
		directedConnMinSamplesPerMvarParam: 5,

		// ERDS Condition Markers defaults
		erdsConditionMarkerBands:       "alpha,beta",
		erdsLateralityColumns:          "stimulation_side,stim_side",
		erdsSomatosensoryLeftChannels:  "C3,CP3,C5,CP5",
		erdsSomatosensoryRightChannels: "C4,CP4,C6,CP6",
		erdsOnsetMinThresholdPercent:   10.0,
		erdsReboundThresholdSigma:      1.0,
		erdsReboundMinThresholdPercent: 5.0,

		// Microstates Missing defaults
		microstatesAssignFromGfpPeaks: true,

		// Behavior Statistics defaults
		behaviorStatsTempControl:               0, // 0: spline
		behaviorStatsAllowIIDTrials:            false,
		behaviorStatsHierarchicalFDR:           true,
		behaviorStatsComputeReliability:        true,
		statisticsAlpha:                        0.05,
		behaviorPermScheme:                     0, // 0: circular_shift
		behaviorPermGroupColumnPreference:      "run_id,block",
		behaviorExcludeNonTrialwiseFeatures:    true,
		behaviorFeatureRegistryFilesJSON:       "",
		behaviorFeatureRegistrySourceJSON:      "",
		behaviorFeatureRegistryHierarchyJSON:   "",
		behaviorFeatureRegistryPatternsJSON:    "",
		behaviorFeatureRegistryClassifiersJSON: "",

		// Global Statistics & Validation defaults
		globalNBootstrap:                1000,
		clusterCorrectionEnabled:        false,
		clusterCorrectionAlpha:          0.05,
		clusterCorrectionMinClusterSize: 2,
		clusterCorrectionTailGlobal:     0, // two-tailed
		validationMinEpochs:             5,
		validationMinChannels:           10,
		validationMaxAmplitudeUv:        500.0,

		// System / IO defaults
		ioPredictorRange:             "35.0,55.0",
		ioMaxMissingChannelsFraction: 0.3,

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
			"complexity", "bursts", "quality", "microstates", "sourcelocalization",
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
			"EEG microstate dynamics (A-D)",
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
			"trial_table":        false,
			"lag_features":       false,
			"predictor_residual": false,

			// Core Analyses
			"correlations":            false,
			"multilevel_correlations": false,
			"regression":              false,
			"condition":               false,
			"temporal":                false,
			"predictor_sensitivity":   false,
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
			"Trial-wise betas + multivariate signature readouts (beta-series or LSS)",
		}
		m.steps = []types.WizardStep{
			types.StepSelectMode,
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
		}

		// Defaults match fmri_pipeline/cli/commands/fmri_analysis.py
		defaultFmriConditionColumn := m.fmriDefaultConditionColumn()
		m.fmriAnalysisInputSourceIndex = 0 // fmriprep
		m.fmriAnalysisFmriprepSpace = "T1w"
		m.fmriAnalysisRequireFmriprep = true
		m.fmriAnalysisRunsSpec = ""    // auto-detect
		m.fmriAnalysisContrastType = 0 // t-test
		m.fmriAnalysisCondAColumn = defaultFmriConditionColumn
		m.fmriAnalysisCondAValue = ""
		m.fmriAnalysisCondBColumn = defaultFmriConditionColumn
		m.fmriAnalysisCondBValue = ""
		m.fmriAnalysisContrastName = "contrast"
		m.fmriAnalysisFormula = ""
		m.fmriAnalysisEventsToModel = ""
		m.fmriAnalysisScopeColumn = defaultFmriConditionColumn
		m.fmriAnalysisScopeTrialTypes = ""
		m.fmriAnalysisPhaseColumn = m.fmriDefaultPhaseColumn()
		m.fmriAnalysisPhaseScopeColumn = defaultFmriConditionColumn
		m.fmriAnalysisPhaseScopeValue = ""
		m.fmriAnalysisStimPhasesToModel = "" // no default scoping
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
		m.fmriAnalysisSignatureMaps = ""

		// Trial-wise signature defaults (used by beta-series / lss modes)
		m.fmriTrialSigGroupExpanded = true
		m.fmriTrialSigMethodIndex = 0 // beta-series
		m.fmriTrialSigIncludeOtherEvents = true
		m.fmriTrialSigMaxTrialsPerRun = 0
		m.fmriTrialSigFixedEffectsWeighting = 0 // variance
		m.fmriTrialSigWriteTrialBetas = false
		m.fmriTrialSigWriteTrialVariances = false
		m.fmriTrialSigWriteConditionBetas = true
		m.fmriTrialSigSignatureOption1 = true
		m.fmriTrialSigSignatureOption2 = true
		m.fmriTrialSigLssOtherRegressorsIndex = 0 // per-condition
		m.fmriTrialSigGroupColumn = ""
		m.fmriTrialSigGroupValuesSpec = ""
		m.fmriTrialSigGroupScopeIndex = 0 // across-runs (average)
		m.fmriTrialSigScopeTrialTypeColumn = defaultFmriConditionColumn
		m.fmriTrialSigScopePhaseColumn = m.fmriDefaultPhaseColumn()
		m.fmriTrialSigScopeTrialTypes = ""
		m.fmriTrialSigScopeStimPhases = "" // no default scoping

		if m.modeIndex < 0 || m.modeIndex >= len(m.modeOptions) {
			m.modeIndex = 0
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

		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectPlotCategories,
			types.StepSelectPlots,
			types.StepSelectFeaturePlotters,
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
	return tea.Tick(m.tickInterval(), func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) tickInterval() time.Duration {
	// Reduce repaint frequency when the wizard is idle to avoid visible flicker
	// on configuration pages while preserving responsiveness during interactions.
	if m.IsEditing() || m.subjectsLoading || (m.featurePlotters == nil && strings.TrimSpace(m.featurePlotterError) == "") || m.toastTicker > 0 {
		return time.Millisecond * styles.TickIntervalMs
	}
	return 500 * time.Millisecond
}

// CursorBlinkVisible returns true when the cursor should be shown (blink on phase).
func (m Model) CursorBlinkVisible() bool {
	if !m.IsEditing() {
		// Keep cursor steady while browsing options to prevent distracting flicker.
		return true
	}
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
		if m.IsEditing() {
			m.animQueue.Tick()
		}
		if m.subjectsLoading {
			m.subjectLoadingSpinner.Tick()
		}
		if m.featurePlotters == nil && strings.TrimSpace(m.featurePlotterError) == "" {
			m.plotLoadingSpinner.Tick()
		}
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
				m.roiEditCursorPos = 0
			case "enter":
				m.commitROIEdit()
				// Move to next field or exit
				if m.editingROIField < 1 {
					m.editingROIField++
					m.initROIEditBuffer()
				} else {
					m.editingROIIdx = -1
					m.roiEditBuffer = ""
					m.roiEditCursorPos = 0
				}
			case "tab":
				m.commitROIEdit()
				m.editingROIField = (m.editingROIField + 1) % 2
				m.initROIEditBuffer()
			case "left":
				m.moveROIEditCursorLeft()
			case "right":
				m.moveROIEditCursorRight()
			case "backspace":
				m.backspaceROIEditBuffer()
			default:
				char := msg.String()
				if len(char) == singleCharLength {
					// Accept alphanumeric and comma for channels
					m.insertROIEditChar(char)
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
