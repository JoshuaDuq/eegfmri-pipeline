package wizard

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/components"
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

var behaviorComputationGroups = map[string]string{
	"trial_table":      "DataPrep",
	"lag_features":     "DataPrep",
	"pain_residual":    "DataPrep",
	"correlations":     "Core",
	"regression":       "Core",
	"condition":        "Core",
	"temporal":         "Core",
	"pain_sensitivity": "Core",
	"cluster":          "Core",
	"mediation":        "Advanced",
	"moderation":       "Advanced",
	"mixed_effects":    "Advanced",
	"stability":        "Quality",
	"validation":       "Quality",
	"report":           "Quality",
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
	{"Sensorimotor_Contra_R", "Sensorimotor Contra R", "FC2,FC4,FC6,C2,C4,C6,CP2,CP4,CP6"},
	{"Sensorimotor_Ipsi_L", "Sensorimotor Ipsi L", "FC1,FC3,FC5,C1,C3,C5,CP1,CP3,CP5"},
	{"Temporal_Contra_R", "Temporal Contra R", "FT8,FT10,T8,TP8,TP10"},
	{"Temporal_Ipsi_L", "Temporal Ipsi L", "FT7,FT9,T7,TP7,TP9"},
	{"ParOccipital_Contra_R", "ParOccipital Contra R", "P2,P4,P6,P8,PO4,PO8,O2"},
	{"ParOccipital_Ipsi_L", "ParOccipital Ipsi L", "P1,P3,P5,P7,PO3,PO7,O1"},
	{"ParOccipital_Midline", "ParOccipital Midline", "Pz,POz,Oz"},
	{"Midline_ACC_MCC", "Midline ACC/MCC", "Fz,Cz,CPz"},
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

// computationApplicableFeatures maps each computation to the feature files it can use.
// Features not in this list for a given computation won't be shown in the feature selection.
var computationApplicableFeatures = map[string][]string{
	// Correlations can use all standard EEG features
	"correlations": {"power", "connectivity", "directedconnectivity", "sourcelocalization", "aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry", "erds", "spectral"},
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
)

type textField int

const (
	textFieldNone textField = iota
	textFieldTask
	textFieldBidsRoot
	textFieldDerivRoot
	textFieldSourceRoot
	textFieldRawMontage
	textFieldPrepMontage
	textFieldRawEventPrefixes
	textFieldMergeEventPrefixes
	textFieldMergeEventTypes
	// Behavior advanced config text fields
	textFieldTrialTableExtraEventColumns
	textFieldConfoundsQCColumnPatterns
	textFieldConditionCompareColumn
	textFieldConditionCompareWindows
	textFieldConditionCompareValues
	textFieldTemporalConditionColumn
	textFieldTemporalConditionValues
	textFieldTfHeatmapFreqs
	textFieldRunAdjustmentColumn
	textFieldPainResidualCrossfitGroupColumn
	textFieldClusterConditionColumn
	textFieldClusterConditionValues
	textFieldCorrelationsTargetColumn
	// Frequency band editing text fields
	textFieldBandName
	textFieldBandLowHz
	textFieldBandHighHz
	// Features advanced config text fields
	textFieldPACPairs
	textFieldBurstBands
	textFieldSpectralRatioPairs
	textFieldAsymmetryChannelPairs
	textFieldERPComponents
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
	textFieldElasticNetAlphaGrid
	textFieldElasticNetL1RatioGrid
	textFieldRfMaxDepthGrid
	// Preprocessing advanced config text fields
	textFieldIcaLabelsToKeep
)

var defaultPlotItems = []PlotItem{
	// Power
	{ID: "power_by_condition", Group: "power", Name: "Condition Comparison", Description: "Power differences between conditions", RequiredFiles: []string{"features_power*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "power_band_segment_condition", Group: "power", Name: "Band × Segment × Condition", Description: "Power across bands and time segments by condition", RequiredFiles: []string{"features_power*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "band_power_topomaps_baseline", Group: "power", Name: "Topomaps (Baseline)", Description: "Band power topographic maps for baseline period", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "band_power_topomaps_active", Group: "power", Name: "Topomaps (Active)", Description: "Band power topographic maps for active period", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "power_variability_comprehensive", Group: "power", Name: "Variability Analysis", Description: "Comprehensive power variability diagnostics", RequiredFiles: []string{"features_power*.tsv"}, RequiresFeatures: true},
	{ID: "cross_frequency_power_correlation", Group: "power", Name: "Cross-Frequency Correlation", Description: "Correlation matrix between frequency bands", RequiredFiles: []string{"features_power*.tsv"}, RequiresFeatures: true},
	{ID: "power_spectral_density", Group: "power", Name: "PSD Summary", Description: "Power spectral density curves", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "power_topomaps_from_df", Group: "power", Name: "Topomaps from Features", Description: "Power topomaps generated from feature DataFrame", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "spectral_slope_topomap", Group: "power", Name: "Spectral Slope Topomap", Description: "Topographic map of 1/f spectral slope", RequiredFiles: []string{"features_aperiodic*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	// Connectivity
	{ID: "connectivity_by_condition", Group: "connectivity", Name: "Condition Comparison", Description: "Connectivity differences between conditions", RequiredFiles: []string{"features_connectivity*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "connectivity_dynamics", Group: "connectivity", Name: "Sliding Window Dynamics", Description: "Connectivity trajectories over time windows", RequiredFiles: []string{"features_connectivity*.tsv"}, RequiresFeatures: true},
	// Aperiodic
	{ID: "aperiodic_topomaps", Group: "aperiodic", Name: "Topomaps", Description: "Topographic maps of slope and offset", RequiredFiles: []string{"features_aperiodic*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "aperiodic_by_condition", Group: "aperiodic", Name: "Condition Comparison", Description: "Aperiodic differences between conditions", RequiredFiles: []string{"features_aperiodic*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "aperiodic_band_segment_condition", Group: "aperiodic", Name: "Band × Segment × Condition", Description: "Aperiodic features across segments by condition", RequiredFiles: []string{"features_aperiodic*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "aperiodic_temporal_evolution", Group: "aperiodic", Name: "Temporal Evolution", Description: "Aperiodic features over time bins", RequiredFiles: []string{"features_aperiodic*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Phase (ITPC/PAC)
	{ID: "itpc_heatmap", Group: "phase", Name: "ITPC Heatmap", Description: "Inter-trial phase coherence heatmaps", RequiredFiles: []string{"features_itpc*.tsv"}, RequiresFeatures: true},
	{ID: "itpc_topomaps", Group: "phase", Name: "ITPC Topomaps", Description: "Topographic maps of phase coherence", RequiredFiles: []string{"features_itpc*.tsv", "epochs/*.fif"}, RequiresFeatures: true, RequiresEpochs: true},
	{ID: "itpc_by_condition", Group: "phase", Name: "ITPC Condition Comparison", Description: "Phase coherence differences between conditions", RequiredFiles: []string{"features_itpc*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "itpc_band_segment_condition", Group: "phase", Name: "ITPC Band × Segment × Condition", Description: "ITPC across bands and segments by condition", RequiredFiles: []string{"features_itpc*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "itpc_temporal_evolution", Group: "phase", Name: "ITPC Temporal Evolution", Description: "Phase coherence over time bins", RequiredFiles: []string{"features_itpc*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "pac_summary", Group: "phase", Name: "PAC Summary", Description: "Phase-amplitude coupling overview", RequiredFiles: []string{"features_pac*.tsv"}, RequiresFeatures: true},
	{ID: "pac_comodulograms", Group: "phase", Name: "PAC Comodulograms", Description: "Phase-amplitude coupling frequency matrices", RequiredFiles: []string{"features_pac*.tsv"}, RequiresFeatures: true},
	{ID: "pac_by_condition", Group: "phase", Name: "PAC Condition Comparison", Description: "PAC differences between conditions", RequiredFiles: []string{"features_pac*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "pac_time_ribbons", Group: "phase", Name: "PAC Time Ribbons", Description: "PAC temporal dynamics ribbon plots", RequiredFiles: []string{"features_pac*.tsv"}, RequiresFeatures: true},
	// ERDS
	{ID: "erds_temporal_evolution", Group: "erds", Name: "Temporal Evolution", Description: "ERD/ERS time course plots", RequiredFiles: []string{"features_erds*.tsv"}, RequiresFeatures: true},
	{ID: "erds_latency_distribution", Group: "erds", Name: "Latency Distribution", Description: "Distribution of ERD/ERS onset and peak latencies", RequiredFiles: []string{"features_erds*.tsv"}, RequiresFeatures: true},
	{ID: "erds_erd_ers_separation", Group: "erds", Name: "ERD/ERS Separation", Description: "Separate ERD and ERS magnitude/duration", RequiredFiles: []string{"features_erds*.tsv"}, RequiresFeatures: true},
	{ID: "erds_global_summary", Group: "erds", Name: "Global Summary", Description: "Summary statistics across channels", RequiredFiles: []string{"features_erds*.tsv"}, RequiresFeatures: true},
	{ID: "erds_by_condition", Group: "erds", Name: "Condition Comparison", Description: "ERDS differences between conditions", RequiredFiles: []string{"features_erds*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Complexity
	{ID: "complexity_by_band", Group: "complexity", Name: "By Band", Description: "Complexity metrics per frequency band", RequiredFiles: []string{"features_complexity*.tsv"}, RequiresFeatures: true},
	{ID: "complexity_by_condition", Group: "complexity", Name: "Condition Comparison", Description: "Complexity differences between conditions", RequiredFiles: []string{"features_complexity*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "complexity_band_segment_condition", Group: "complexity", Name: "Band × Segment × Condition", Description: "Complexity across bands and segments by condition", RequiredFiles: []string{"features_complexity*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "complexity_temporal_evolution", Group: "complexity", Name: "Temporal Evolution", Description: "Complexity over time bins", RequiredFiles: []string{"features_complexity*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Spectral
	{ID: "spectral_summary", Group: "spectral", Name: "Summary", Description: "Peak frequency and spectral features overview", RequiredFiles: []string{"features_spectral*.tsv"}, RequiresFeatures: true},
	{ID: "spectral_edge_frequency", Group: "spectral", Name: "Edge Frequency", Description: "Spectral edge frequency (95%) distributions", RequiredFiles: []string{"features_spectral*.tsv"}, RequiresFeatures: true},
	{ID: "spectral_by_condition", Group: "spectral", Name: "Condition Comparison", Description: "Spectral differences between conditions", RequiredFiles: []string{"features_spectral*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "spectral_band_segment_condition", Group: "spectral", Name: "Band × Segment × Condition", Description: "Spectral features across bands and segments", RequiredFiles: []string{"features_spectral*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "spectral_temporal_evolution", Group: "spectral", Name: "Temporal Evolution", Description: "Spectral features over time bins", RequiredFiles: []string{"features_spectral*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Ratios
	{ID: "ratios_by_pair", Group: "ratios", Name: "By Pair", Description: "Ratio values for each band pair", RequiredFiles: []string{"features_ratios*.tsv"}, RequiresFeatures: true},
	{ID: "ratios_by_condition", Group: "ratios", Name: "Condition Comparison", Description: "Ratio differences between conditions", RequiredFiles: []string{"features_ratios*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "ratios_band_segment_condition", Group: "ratios", Name: "Band × Segment × Condition", Description: "Ratios across segments by condition", RequiredFiles: []string{"features_ratios*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "ratios_temporal_evolution", Group: "ratios", Name: "Temporal Evolution", Description: "Ratios over time bins", RequiredFiles: []string{"features_ratios*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Asymmetry
	{ID: "asymmetry_by_band", Group: "asymmetry", Name: "By Band", Description: "Asymmetry indices per frequency band", RequiredFiles: []string{"features_asymmetry*.tsv"}, RequiresFeatures: true},
	{ID: "asymmetry_by_condition", Group: "asymmetry", Name: "Condition Comparison", Description: "Asymmetry differences between conditions", RequiredFiles: []string{"features_asymmetry*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "asymmetry_band_segment_condition", Group: "asymmetry", Name: "Band × Segment × Condition", Description: "Asymmetry across bands and segments", RequiredFiles: []string{"features_asymmetry*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "asymmetry_temporal_evolution", Group: "asymmetry", Name: "Temporal Evolution", Description: "Asymmetry over time bins", RequiredFiles: []string{"features_asymmetry*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Bursts
	{ID: "bursts_by_band", Group: "bursts", Name: "By Band", Description: "Burst metrics per frequency band", RequiredFiles: []string{"features_bursts*.tsv"}, RequiresFeatures: true},
	{ID: "bursts_by_condition", Group: "bursts", Name: "Condition Comparison", Description: "Burst differences between conditions", RequiredFiles: []string{"features_bursts*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "burst_band_segment_condition", Group: "bursts", Name: "Band × Segment × Condition", Description: "Burst metrics across bands and segments", RequiredFiles: []string{"features_bursts*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "burst_temporal_evolution", Group: "bursts", Name: "Temporal Evolution", Description: "Burst dynamics over time bins", RequiredFiles: []string{"features_bursts*.tsv", "events.tsv"}, RequiresFeatures: true},
	// Quality
	{ID: "quality_feature_distributions", Group: "quality", Name: "Feature Distributions", Description: "Distribution of quality metrics across trials", RequiredFiles: []string{"features_quality*.tsv"}, RequiresFeatures: true},
	{ID: "quality_outlier_heatmap", Group: "quality", Name: "Outlier Heatmap", Description: "Heatmap of outlier trials and features", RequiredFiles: []string{"features_quality*.tsv"}, RequiresFeatures: true},
	{ID: "quality_snr_distribution", Group: "quality", Name: "SNR Distribution", Description: "Signal-to-noise ratio distributions", RequiredFiles: []string{"features_quality*.tsv"}, RequiresFeatures: true},
	// ERP
	{ID: "erp_butterfly", Group: "erp", Name: "Butterfly", Description: "Butterfly ERP plots (all channels)", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_roi", Group: "erp", Name: "ROI Waveforms", Description: "ROI-based ERP waveforms with error bars", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_contrast", Group: "erp", Name: "Contrast", Description: "ERP condition contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "erp_topomaps", Group: "erp", Name: "Topomaps", Description: "ERP spatial distributions", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// TFR
	{ID: "TFR", Group: "tfr", Name: "Scalp-Mean TFR", Description: "Scalp-mean time-frequency representation", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Behavior
	{ID: "temporal_evolution", Group: "behavior", Name: "Temporal Evolution (All Features)", Description: "Feature evolution over time bins", RequiredFiles: []string{"features_*.tsv", "events.tsv"}, RequiresFeatures: true},
	{ID: "behavior_psychometrics", Group: "behavior", Name: "Psychometrics", Description: "Rating distributions and psychometrics", RequiredFiles: []string{"events.tsv"}},
	{ID: "behavior_power_scatter", Group: "behavior", Name: "Power ROI Scatter", Description: "Power vs behavior scatter plots", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_complexity_scatter", Group: "behavior", Name: "Complexity Scatter", Description: "Complexity vs behavior scatter plots", RequiredFiles: []string{"features_complexity*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_aperiodic_scatter", Group: "behavior", Name: "Aperiodic Scatter", Description: "Aperiodic vs behavior scatter plots", RequiredFiles: []string{"features_aperiodic*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_connectivity_scatter", Group: "behavior", Name: "Connectivity Scatter", Description: "Connectivity vs behavior scatter plots", RequiredFiles: []string{"features_connectivity*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_itpc_scatter", Group: "behavior", Name: "ITPC Scatter", Description: "ITPC vs behavior scatter plots", RequiredFiles: []string{"features_itpc*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_temporal_topomaps", Group: "behavior", Name: "Temporal Topomaps", Description: "Temporal correlation topomaps", RequiredFiles: []string{"stats/temporal_correlations/temporal_correlations_by_condition*.npz"}, RequiresStats: true},
	{ID: "behavior_pain_clusters", Group: "behavior", Name: "Condition Clusters", Description: "Cluster-based temporal contrasts", RequiredFiles: []string{"stats/pain_nonpain_time_clusters_*.tsv"}, RequiresStats: true},
	{ID: "behavior_dose_response", Group: "behavior", Name: "Dose Response", Description: "Dose-response curves and contrasts", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_top_predictors", Group: "behavior", Name: "Top Predictors", Description: "Top predictors summary", RequiredFiles: []string{"stats/correlations*.tsv"}, RequiresStats: true},
	{ID: "behavior_temperature_models", Group: "behavior", Name: "Temperature Models", Description: "Subject-level temperature→rating diagnostics", RequiredFiles: []string{"stats/trials*.tsv"}, RequiresStats: true},
	{ID: "behavior_stability_groupwise", Group: "behavior", Name: "Stability (Run/Block)", Description: "Within-subject stability of feature→outcome associations", RequiredFiles: []string{"stats/stability_groupwise*.tsv"}, RequiresStats: true},
	// Machine Learning
	{ID: "ml_regression_plots", Group: "machine_learning", Name: "Regression Plots", Description: "LOSO regression diagnostics", RequiredFiles: []string{"machine_learning/regression/loso_predictions.tsv"}},
	{ID: "ml_timegen_plots", Group: "machine_learning", Name: "Time-Generalization", Description: "Time-generalization matrices", RequiredFiles: []string{"machine_learning/time_generalization/time_generalization_regression.npz"}},
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
	{"machine_learning", "Machine Learning", "Machine learning regression and time-generalization"},
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
	task       string
	bidsRoot   string
	derivRoot  string
	sourceRoot string

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
	plotFormats         []string
	plotFormatSelected  map[string]bool
	plotDpiOptions      []int
	plotDpiIndex        int
	plotSavefigDpiIndex int
	plotSharedColorbar  bool
	plotConfigCursor    int

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

	// Per-plot advanced configuration (wizard overrides scoped to plot IDs)
	plotItemConfigs        map[string]PlotItemConfig
	plotItemConfigExpanded map[string]bool

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

	plotQualityWidthPerPlot            float64
	plotQualityHeightPerPlot           float64
	plotQualityDistributionNCols       int
	plotQualityDistributionMaxFeatures int
	plotQualityOutlierZThreshold       float64
	plotQualityOutlierMaxFeatures      int
	plotQualityOutlierMaxTrials        int
	plotQualitySnrThresholdDb          float64

	plotComplexityWidthPerMeasure  float64
	plotComplexityHeightPerSegment float64

	plotConnectivityWidthPerCircle    float64
	plotConnectivityWidthPerBand      float64
	plotConnectivityHeightPerMeasure  float64
	plotConnectivityCircleTopFraction float64
	plotConnectivityCircleMinLines    int

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

	// Subject selection
	subjects         []types.SubjectStatus
	subjectSelected  map[string]bool
	subjectCursor    int
	subjectsLoading  bool
	subjectFilter    string
	filteringSubject bool
	availableWindows []string
	availableColumns []string

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
	ticker int

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
	editingPlotField plotItemConfigField

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
	featGroupValidationExpanded       bool
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
	burstThresholdZ          float64
	burstThresholdMethod     int     // 0: percentile, 1: zscore, 2: mad
	burstThresholdPercentile float64 // Percentile threshold (for percentile method)
	burstMinDuration         int     // ms
	burstMinCycles           float64 // Minimum oscillatory cycles for burst detection
	burstBandsSpec           string  // e.g. beta,gamma

	// Power configuration
	powerBaselineMode    int // 0: logratio, 1: mean, 2: ratio, 3: zscore, 4: zlogratio
	powerRequireBaseline bool

	// Spectral configuration
	spectralEdgePercentile float64
	spectralRatioPairsSpec string // e.g. theta:beta,alpha:beta

	// Aggregation
	aggregationMethod int // 0: mean, 1: median

	// Validation & Generic
	minEpochsForFeatures int

	failOnMissingWindows     bool
	failOnMissingNamedWindow bool

	// Storage configuration
	saveSubjectLevelFeatures bool

	// Asymmetry
	asymmetryChannelPairsSpec string // e.g. F3:F4,C3:C4

	// Connectivity configuration
	connOutputLevel  int // 0: full, 1: global_only
	connGraphMetrics bool
	connGraphProp    float64
	connWindowLen    float64
	connWindowStep   float64
	connAECMode      int // 0: orth, 1: none, 2: sym

	// Scientific validity options (new)
	itpcMethod             int     // 0: global, 1: fold_global, 2: loo
	aperiodicMinSegmentSec float64 // Minimum segment duration for aperiodic fits
	connAECOutput          int     // 0: r only, 1: z only, 2: both r and z
	connForceWithinEpochML bool    // Force within_epoch for CV/machine learning
	ratioSource            int     // 0: raw, 1: powcorr (aperiodic-adjusted)

	// ITPC additional options
	itpcAllowUnsafeLoo     bool // Allow unsafe LOO ITPC
	itpcBaselineCorrection int  // 0: none, 1: subtract

	// Spectral advanced options
	spectralIncludeLogRatios   bool    // Include log ratios
	spectralPsdMethod          int     // 0: multitaper, 1: welch
	spectralFmin               float64 // Min frequency for spectral
	spectralFmax               float64 // Max frequency for spectral
	spectralExcludeLineNoise   bool    // Exclude line noise
	spectralLineNoiseFreq      float64 // Line noise frequency (50 or 60)
	spectralLineNoiseWidthHz   float64 // Line noise frequency band width to exclude
	spectralLineNoiseHarmonics int     // Number of line noise harmonics to exclude
	spectralSegments           int     // 0: baseline only, 1: active only, 2: both
	spectralMinSegmentSec      float64 // Minimum segment duration
	spectralMinCyclesAtFmin    float64 // Minimum cycles at lowest frequency

	// Band envelope options
	bandEnvelopePadSec    float64 // Padding in seconds
	bandEnvelopePadCycles float64 // Padding in cycles

	// IAF (Individualized Alpha Frequency) options
	iafEnabled        bool    // Use individualized bands
	iafAlphaWidthHz   float64 // Alpha band width
	iafSearchRangeMin float64 // IAF search range min
	iafSearchRangeMax float64 // IAF search range max
	iafMinProminence  float64 // IAF peak prominence threshold
	iafRoisSpec       string  // IAF ROIs (comma-separated)

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
	sourceLocMethod            int     // 0: lcmv, 1: eloreta
	sourceLocSpacing           int     // 0: oct5, 1: oct6, 2: ico4, 3: ico5
	sourceLocParc              int     // 0: aparc, 1: aparc.a2009s, 2: HCPMMP1
	sourceLocReg               float64 // LCMV regularization
	sourceLocSnr               float64 // eLORETA SNR
	sourceLocLoose             float64 // eLORETA loose constraint
	sourceLocDepth             float64 // eLORETA depth weighting
	sourceLocConnMethod        int     // 0: aec, 1: wpli, 2: plv
	featGroupSourceLocExpanded bool    // UI expansion state

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
	complexityTargetHz       float64 // Target sampling rate
	complexityTargetNSamples int     // Target number of samples
	complexityZscore         bool    // Apply z-score normalization

	// Quality feature options
	qualityPsdMethod        int     // 0: welch, 1: multitaper
	qualityFmin             float64 // Min frequency
	qualityFmax             float64 // Max frequency
	qualityNfft             int     // FFT size
	qualityExcludeLineNoise bool    // Exclude line noise
	qualitySnrSignalBandMin float64 // SNR signal band min
	qualitySnrSignalBandMax float64 // SNR signal band max
	qualitySnrNoiseBandMin  float64 // SNR noise band min
	qualitySnrNoiseBandMax  float64 // SNR noise band max
	qualityMuscleBandMin    float64 // Muscle band min
	qualityMuscleBandMax    float64 // Muscle band max

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
	trialTableOnly        bool    // Skip computations that require epochs/time-frequency arrays
	fdrAlpha              float64 // FDR correction threshold
	behaviorConfigSection int     // Behavior config section index (legacy, kept for compatibility)
	behaviorNJobs         int     // -1 = all

	behaviorComputeChangeScores  bool
	behaviorComputeBayesFactors  bool
	behaviorComputeLosoStability bool

	// Run adjustment (subject-level; optional)
	runAdjustmentEnabled               bool
	runAdjustmentColumn                string
	runAdjustmentIncludeInCorrelations bool
	runAdjustmentMaxDummies            int

	// Output options
	alsoSaveCsv bool // Also save output tables as CSV files

	// Behavior advanced config section expansion (collapsed by default for compact UI)
	behaviorGroupGeneralExpanded      bool
	behaviorGroupTrialTableExpanded   bool
	behaviorGroupCorrelationsExpanded bool
	behaviorGroupPainSensExpanded     bool
	behaviorGroupConfoundsExpanded    bool
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
	trialTableFormat          int // 0=parquet, 1=tsv
	trialTableIncludeFeatures bool
	trialTableIncludeCovars   bool
	trialTableIncludeEvents   bool
	trialTableAddLagFeatures  bool
	trialTableExtraEventCols  string
	trialTableHighMissingFrac float64

	featureSummariesEnabled bool

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
	correlationsPreferPainResidual  bool
	correlationsUseCrossfitResidual bool
	correlationsPrimaryUnit         int // 0=trial, 1=run_mean
	correlationsPermutationPrimary  bool
	correlationsTargetColumn        string // Custom target column from events (dropdown)

	// Pain sensitivity

	// Report
	reportTopN int

	// Temporal
	temporalResolutionMs       int
	temporalSmoothMs           int
	temporalTimeMinMs          int
	temporalTimeMaxMs          int
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

	// Column discovery (populated from events files)
	discoveredColumns       []string            // Available columns from events/trial table
	discoveredColumnValues  map[string][]string // Values for each column
	columnDiscoveryDone     bool                // Whether discovery has been completed
	columnDiscoveryError    string              // Error message if discovery failed
	columnDiscoverySource   string              // "events" or "trial_table"
	selectedColumnCursor    int                 // Cursor for column selection
	selectedValueCursors    map[string]int      // Cursor for value selection per column
	expandedColumnSelection string              // Currently expanded column for value selection

	// Machine Learning pipeline advanced config
	mlNPerm     int  // Permutations for significance test
	innerSplits int  // CV inner splits
	outerJobs   int  // Number of parallel jobs for outer CV
	skipTimeGen bool // Skip time generalization
	mlScope     MLCVScope
	// Machine Learning model hyperparameters
	elasticNetAlphaGrid   string // alpha grid as comma-separated values
	elasticNetL1RatioGrid string // l1_ratio grid as comma-separated values
	rfNEstimators         int    // Random forest n_estimators
	rfMaxDepthGrid        string // max_depth grid as comma-separated values (use "null" for None)

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
	prepUsePyprep           bool
	prepUseIcalabel         bool
	prepNJobs               int
	prepMontage             string // EEG montage (e.g., easycap-M1)
	prepResample            int
	prepLFreq               float64
	prepHFreq               float64
	prepNotch               int
	prepICAMethod           int // 0: fastica, 1: infomax, 2: picard
	prepICAComp             float64
	prepProbThresh          float64
	prepEpochsTmin          float64
	prepEpochsTmax          float64
	prepLineFreq            int     // Line frequency (50 or 60 Hz)
	prepEpochsBaselineStart float64 // Epoch baseline start (seconds)
	prepEpochsBaselineEnd   float64 // Epoch baseline end (seconds)
	prepEpochsNoBaseline    bool    // Disable baseline correction
	prepEpochsReject        float64 // Peak-to-peak rejection threshold (µV)

	// Utilities (raw-to-bids/merge) advanced config
	rawMontage           string
	rawLineFreq          int
	rawOverwrite         bool
	rawZeroBaseOnsets    bool
	rawTrimToFirstVolume bool
	rawEventPrefixes     string
	rawKeepAnnotations   bool
	mergeEventPrefixes   string
	mergeEventTypes      string

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
	help.AddSection("Presets", []components.HelpItem{
		{Key: "Q", Description: "Quick preset"},
		{Key: "F", Description: "Full preset"},
		{Key: "C", Description: "Connectivity (features)"},
		{Key: "S", Description: "Spectral (features)"},
		{Key: "R", Description: "Regression (behavior)"},
		{Key: "T", Description: "Temporal (behavior)"},
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Proceed to next step"},
		{Key: "?", Description: "Toggle help"},
		{Key: "F5", Description: "Refresh subjects"},
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
		featGroupValidationExpanded:       true,
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
		aperiodicFmin:      2.0,
		aperiodicFmax:      40.0,
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
		burstThresholdZ:          2.0,
		burstThresholdMethod:     0, // 0: percentile
		burstThresholdPercentile: 95.0,
		burstMinDuration:         50,
		burstMinCycles:           3.0,
		burstBandsSpec:           "beta,gamma",
		// Power defaults
		powerBaselineMode:    0,
		powerRequireBaseline: true,
		// Spectral defaults
		spectralEdgePercentile: 0.95,
		spectralRatioPairsSpec: "theta:beta,theta:alpha,alpha:beta,delta:alpha,delta:theta",
		// Connectivity defaults
		connOutputLevel:  0,
		connGraphMetrics: true,
		connGraphProp:    0.1,
		connWindowLen:    1.0,
		connWindowStep:   0.5,
		connAECMode:      0,
		// Scientific validity defaults (new)
		itpcMethod:             0,    // 0: global (default)
		aperiodicMinSegmentSec: 2.0,  // 2.0s minimum for stable fits
		connAECOutput:          0,    // 0: r only (raw)
		connForceWithinEpochML: true, // Force within_epoch for CV-safety
		ratioSource:            0,    // 0: raw (default)

		// ITPC additional defaults
		itpcAllowUnsafeLoo:     false,
		itpcBaselineCorrection: 0, // 0: none

		// Spectral advanced defaults
		spectralIncludeLogRatios:   true,
		spectralPsdMethod:          0, // 0: multitaper
		spectralFmin:               1.0,
		spectralFmax:               80.0,
		spectralExcludeLineNoise:   true,
		spectralLineNoiseFreq:      50.0,
		spectralLineNoiseWidthHz:   1.0,
		spectralLineNoiseHarmonics: 3,
		spectralSegments:           0, // 0: baseline only
		spectralMinSegmentSec:      2.0,
		spectralMinCyclesAtFmin:    3.0,

		// Band envelope defaults
		bandEnvelopePadSec:    0.5,
		bandEnvelopePadCycles: 3.0,

		// IAF defaults
		iafEnabled:        false,
		iafAlphaWidthHz:   2.0,
		iafSearchRangeMin: 7.0,
		iafSearchRangeMax: 13.0,
		iafMinProminence:  0.05,
		iafRoisSpec:       "ParOccipital_Midline,ParOccipital_Ipsi_L,ParOccipital_Contra_R",

		// Aperiodic advanced defaults
		aperiodicModel:              0,   // 0: fixed
		aperiodicPsdMethod:          0,   // 0: multitaper
		aperiodicPsdBandwidth:       0.0, // 0 = use default
		aperiodicMaxRms:             0.0, // 0 = no limit
		aperiodicExcludeLineNoise:   true,
		aperiodicLineNoiseFreq:      50.0,
		aperiodicLineNoiseWidthHz:   1.0,
		aperiodicLineNoiseHarmonics: 3,

		// Spatial transform defaults
		spatialTransform:          0,    // 0: none
		spatialTransformLambda2:   1e-5, // Default lambda2
		spatialTransformStiffness: 4.0,  // Default stiffness

		// Connectivity advanced defaults
		connGranularity:            0, // 0: trial
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
		featGroupSourceLocExpanded: false,

		// PAC advanced defaults
		pacSource:              0, // 0: precomputed
		pacNormalize:           true,
		pacNSurrogates:         0,
		pacAllowHarmonicOvrlap: false,
		pacMaxHarmonic:         6,
		pacHarmonicToleranceHz: 1.0,
		pacComputeWaveformQC:   false,
		pacWaveformOffsetMs:    5.0,

		// Complexity advanced defaults
		complexityTargetHz:       100.0,
		complexityTargetNSamples: 500,
		complexityZscore:         true,

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
		minEpochsForFeatures: 10,

		failOnMissingWindows:      false,
		failOnMissingNamedWindow:  true,
		saveSubjectLevelFeatures:  true,
		asymmetryChannelPairsSpec: "",
		// Behavior defaults
		correlationMethod:     "spearman",
		robustCorrelation:     0,
		bootstrapSamples:      1000,
		nPermutations:         1000,
		rngSeed:               0,
		controlTemperature:    true,
		controlTrialOrder:     true,
		trialTableOnly:        true,
		fdrAlpha:              0.05,
		behaviorConfigSection: 0,
		behaviorNJobs:         -1,

		behaviorComputeChangeScores:        true,
		behaviorComputeBayesFactors:        false,
		behaviorComputeLosoStability:       true,
		runAdjustmentEnabled:               false,
		runAdjustmentColumn:                "run_id",
		runAdjustmentIncludeInCorrelations: true,
		runAdjustmentMaxDummies:            20,

		trialTableFormat:          1,
		trialTableIncludeFeatures: true,
		trialTableIncludeCovars:   true,
		trialTableIncludeEvents:   true,
		trialTableAddLagFeatures:  true,
		trialTableExtraEventCols:  "",
		trialTableHighMissingFrac: 0.5,

		featureSummariesEnabled: true,

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
		correlationsPreferPainResidual:  true,
		correlationsUseCrossfitResidual: false,
		correlationsPrimaryUnit:         0,
		correlationsPermutationPrimary:  false,

		reportTopN:                 15,
		temporalResolutionMs:       50,
		temporalSmoothMs:           100,
		temporalTimeMinMs:          -200,
		temporalTimeMaxMs:          1000,
		temporalSplitByCondition:   true,
		temporalConditionColumn:    "",
		temporalConditionValues:    "",
		temporalIncludeROIAverages: true,
		temporalIncludeTFGrid:      true,
		// Temporal feature selection - duplicate defaults removed (using values from temporalFeature*Enabled fields above)
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
		discoveredColumns:      []string{},
		discoveredColumnValues: make(map[string][]string),
		selectedValueCursors:   make(map[string]int),
		// Machine Learning defaults
		mlNPerm:               0,
		innerSplits:           3,
		outerJobs:             1,
		skipTimeGen:           false,
		mlScope:               MLCVScopeGroup,
		elasticNetAlphaGrid:   "0.001,0.01,0.1,1,10",
		elasticNetL1RatioGrid: "0.2,0.5,0.8",
		rfNEstimators:         500,
		rfMaxDepthGrid:        "5,10,20,null",
		// TFR defaults (from config)
		tfrFreqMin:       1.0,
		tfrFreqMax:       100.0,
		tfrNFreqs:        40,
		tfrMinCycles:     3.0,
		tfrNCyclesFactor: 2.0,
		tfrDecim:         4,
		tfrWorkers:       -1,
		// System defaults
		systemNJobs:      -1,
		systemStrictMode: true,
		loggingLevel:     1, // INFO
		// ICA defaults
		icaLabelsToKeep:        "brain,other",
		plotSelected:           make(map[int]bool),
		featurePlotterSelected: make(map[string]bool),
		plotFormats:            []string{"png", "svg", "pdf"},
		plotFormatSelected: map[string]bool{
			"png": true,
			"svg": true,
		},
		plotDpiOptions:      []int{150, 300, 600},
		plotDpiIndex:        1,
		plotSavefigDpiIndex: 2,

		// Preprocessing defaults
		prepUsePyprep:           true,
		prepUseIcalabel:         true,
		prepNJobs:               1,
		prepMontage:             "easycap-M1",
		prepResample:            500,
		prepLFreq:               0.1,
		prepHFreq:               100.0,
		prepNotch:               60,
		prepICAMethod:           0,
		prepICAComp:             0.99,
		prepProbThresh:          0.8,
		prepEpochsTmin:          -5.0,
		prepEpochsTmax:          12.0,
		prepLineFreq:            60,
		prepEpochsBaselineStart: 0,
		prepEpochsBaselineEnd:   0,
		prepEpochsNoBaseline:    false,
		prepEpochsReject:        0,

		// Utilities defaults
		rawMontage:           "easycap-M1",
		rawLineFreq:          60,
		rawOverwrite:         false,
		rawZeroBaseOnsets:    false,
		rawTrimToFirstVolume: false,
		rawEventPrefixes:     "",
		rawKeepAnnotations:   false,
		mergeEventPrefixes:   "",
		mergeEventTypes:      "",
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
			types.StepReviewExecute,
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
			"trial_table":   true,
			"lag_features":  true,
			"pain_residual": true,

			// Core Analyses
			"correlations":     true,
			"regression":       false,
			"condition":        true,
			"temporal":         false,
			"pain_sensitivity": true,
			"cluster":          false,

			// Advanced/Causal Analyses
			"mediation":     false,
			"moderation":    false,
			"mixed_effects": false,

			// Quality & Validation
			"stability":  true,
			"validation": true,
			"report":     true,
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
			types.StepReviewExecute,
		}

	case types.PipelineML:
		m.modeOptions = []string{"regression", "timegen", "classify"}
		m.modeDescriptions = []string{
			"LOSO regression",
			"Time generalization",
			"Binary classification",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
			types.StepReviewExecute,
		}

	case types.PipelinePreprocessing:
		m.modeOptions = []string{"full", "bad-channels", "ica", "epochs"}
		m.modeDescriptions = []string{
			"Full preprocessing pipeline",
			"Bad channel detection only",
			"ICA fitting and labeling",
			"Epoch creation only",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectMode,
			types.StepAdvancedConfig,
			types.StepReviewExecute,
		}

	case types.PipelineMergePsychoPyData:
		m.modeOptions = []string{"merge-behavior"}
		m.modeDescriptions = []string{"Merge PsychoPy data into BIDS events files"}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
			types.StepReviewExecute,
		}

	case types.PipelineRawToBIDS:
		m.modeOptions = []string{"raw-to-bids"}
		m.modeDescriptions = []string{"Convert raw EEG data to BIDS format"}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepAdvancedConfig,
			types.StepReviewExecute,
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
			types.StepReviewExecute, // Review & Execute should be the last step
		}

	default:
		m.modeOptions = []string{styles.ModeCompute}
		m.modeDescriptions = []string{"Run computation"}
		m.steps = []types.WizardStep{types.StepSelectSubjects, types.StepReviewExecute}
	}

	if len(m.steps) > 0 {
		m.CurrentStep = m.steps[0]
	}

	return m
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

type tickMsg struct{}

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*100, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
		m.TickToast()
		return m, m.tick()

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
		case "left", "h":
			m.handleLeft()
		case "right", "l":
			m.handleRight()
		case " ":
			// Space to edit bands or ROIs, or handle default space behavior
			if m.CurrentStep == types.StepSelectBands {
				m.startBandEdit()
			} else if m.CurrentStep == types.StepSelectROIs {
				m.startROIEdit()
			} else {
				m.handleSpace()
			}
		case "enter":
			return m.handleEnter()
		case "tab":
			m.handleTab()
		case "a":
			isTimeRangeStep := m.CurrentStep == types.StepTimeRange
			isNotEditing := m.editingRangeIdx == noRangeEditing
			if isTimeRangeStep && isNotEditing {
				newName := fmt.Sprintf("range%d", len(m.TimeRanges)+1)
				m.TimeRanges = append(m.TimeRanges, types.TimeRange{Name: newName, Tmin: "", Tmax: ""})
				m.timeRangeCursor = len(m.TimeRanges) - 1
				m.editingRangeIdx = m.timeRangeCursor
				m.editingField = fieldName
			} else if m.CurrentStep == types.StepSelectBands {
				m.addNewBand()
			} else if m.CurrentStep == types.StepSelectROIs {
				m.addNewROI()
			} else {
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
			} else if m.CurrentStep == types.StepSelectBands {
				m.removeBand()
			} else if m.CurrentStep == types.StepSelectROIs {
				m.removeROI()
			}
		case "n":
			m.selectNone()

		case "e", "E":
			// Edit band frequencies or ROI channels
			if m.CurrentStep == types.StepSelectBands {
				m.startBandEdit()
			} else if m.CurrentStep == types.StepSelectROIs {
				m.startROIEdit()
			}

		case "+", "=":
			// Legacy support - map to add
			if m.CurrentStep == types.StepSelectBands {
				m.addNewBand()
			} else if m.CurrentStep == types.StepSelectROIs {
				m.addNewROI()
			}

		case "-", "_":
			// Legacy support - map to delete
			if m.CurrentStep == types.StepSelectBands {
				m.removeBand()
			} else if m.CurrentStep == types.StepSelectROIs {
				m.removeROI()
			}

		case "f5", "ctrl+r":
			// Signal to parent to refresh subjects
			m.subjectsLoading = true
			return m, func() tea.Msg { return messages.RefreshSubjectsMsg{} }
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.helpOverlay.Width = min(50, m.width-10)
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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

// SetAvailableMetadata stores runtime-derived metadata (e.g., discovered time
// windows / event columns) for use in UI hints and lightweight validation.
func (m *Model) SetAvailableMetadata(windows []string, eventColumns []string) {
	m.availableWindows = append([]string(nil), windows...)
	m.availableColumns = append([]string(nil), eventColumns...)
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

// SetColumnsDiscoveryError sets the error from column discovery
func (m *Model) SetColumnsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.columnDiscoveryError = err.Error()
	m.columnDiscoveryDone = true
}

// GetDiscoveredColumnValues returns the unique values for a column
func (m Model) GetDiscoveredColumnValues(column string) []string {
	if m.discoveredColumnValues == nil {
		return nil
	}
	return m.discoveredColumnValues[column]
}

// getExpandedListLength returns the length of the currently expanded list
func (m Model) getExpandedListLength() int {
	switch m.expandedOption {
	case expandedConnectivityMeasures:
		return len(connectivityMeasures)
	case expandedDirectedConnMeasures:
		return len(directedConnectivityMeasures)
	case expandedConditionCompareColumn, expandedTemporalConditionColumn, expandedClusterConditionColumn:
		return len(m.availableColumns)
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
		return len(m.discoveredColumns)
	case expandedPlotComparisonValues:
		if m.plotComparisonColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.plotComparisonColumn))
	case expandedConditionCompareWindows, expandedPlotComparisonWindows:
		return len(m.availableWindows)
	case expandedRunAdjustmentColumn:
		return len(m.availableColumns)
	case expandedCorrelationsTargetColumn:
		return len(m.availableColumns) + 1 // +1 for "(none)" option
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
		return m.availableColumns
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
		return m.discoveredColumns
	case expandedPlotComparisonValues:
		if m.plotComparisonColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.plotComparisonColumn)
	case expandedConditionCompareWindows, expandedPlotComparisonWindows:
		return m.availableWindows
	case expandedRunAdjustmentColumn:
		return m.availableColumns
	case expandedCorrelationsTargetColumn:
		return append([]string{"(none)"}, m.availableColumns...)
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
		selectedValues = m.plotComparisonValuesSpec
	case expandedConditionCompareWindows:
		selectedValues = m.conditionCompareWindows
	case expandedPlotComparisonWindows:
		selectedValues = m.plotComparisonWindowsSpec
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
		m.plotComparisonColumn = selectedItem
		m.plotComparisonValuesSpec = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedPlotComparisonValues:
		m.toggleSpaceValue(selectedItem, &m.plotComparisonValuesSpec)

	case expandedConditionCompareWindows:
		m.toggleSpaceValue(selectedItem, &m.conditionCompareWindows)

	case expandedPlotComparisonWindows:
		m.toggleSpaceValue(selectedItem, &m.plotComparisonWindowsSpec)

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
		return m.plotComparisonColumn == item
	case expandedRunAdjustmentColumn:
		return m.runAdjustmentColumn == item
	case expandedCorrelationsTargetColumn:
		if item == "(none)" {
			return m.correlationsTargetColumn == ""
		}
		return m.correlationsTargetColumn == item
	case expandedConditionCompareValues, expandedTemporalConditionValues, expandedClusterConditionValues, expandedPlotComparisonValues,
		expandedConditionCompareWindows, expandedPlotComparisonWindows:
		return m.isColumnValueSelected(item)
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
	case textFieldDerivRoot:
		return m.derivRoot
	case textFieldSourceRoot:
		return m.sourceRoot
	case textFieldRawMontage:
		return m.rawMontage
	case textFieldPrepMontage:
		return m.prepMontage
	case textFieldRawEventPrefixes:
		return m.rawEventPrefixes
	case textFieldMergeEventPrefixes:
		return m.mergeEventPrefixes
	case textFieldMergeEventTypes:
		return m.mergeEventTypes
	case textFieldTrialTableExtraEventColumns:
		return m.trialTableExtraEventCols
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
	case textFieldPACPairs:
		return m.pacPairsSpec
	case textFieldBurstBands:
		return m.burstBandsSpec
	case textFieldSpectralRatioPairs:
		return m.spectralRatioPairsSpec
	case textFieldAsymmetryChannelPairs:
		return m.asymmetryChannelPairsSpec
	case textFieldERPComponents:
		return m.erpComponentsSpec
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
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		return m.elasticNetAlphaGrid
	case textFieldElasticNetL1RatioGrid:
		return m.elasticNetL1RatioGrid
	case textFieldRfMaxDepthGrid:
		return m.rfMaxDepthGrid
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
	default:
		return ""
	}
}

func (m *Model) ApplyConfigKeys(values map[string]interface{}) {
	raw, ok := values["time_frequency_analysis.bands"]
	if !ok {
		return
	}
	bands := parseConfigBands(raw)
	if len(bands) == 0 {
		return
	}

	m.bands = bands
	m.bandSelected = make(map[int]bool)
	for i := range m.bands {
		m.bandSelected[i] = true
	}
	m.bandCursor = 0
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
	case textFieldDerivRoot:
		m.derivRoot = value
	case textFieldSourceRoot:
		m.sourceRoot = value
	case textFieldRawMontage:
		m.rawMontage = value
	case textFieldPrepMontage:
		m.prepMontage = value
	case textFieldRawEventPrefixes:
		m.rawEventPrefixes = value
	case textFieldMergeEventPrefixes:
		m.mergeEventPrefixes = value
	case textFieldMergeEventTypes:
		m.mergeEventTypes = value
	case textFieldTrialTableExtraEventColumns:
		m.trialTableExtraEventCols = value
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
	case textFieldPACPairs:
		m.pacPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldBurstBands:
		m.burstBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSpectralRatioPairs:
		m.spectralRatioPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldAsymmetryChannelPairs:
		m.asymmetryChannelPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldERPComponents:
		m.erpComponentsSpec = strings.Join(strings.Fields(value), "")
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
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		m.elasticNetAlphaGrid = strings.Join(strings.Fields(value), "")
	case textFieldElasticNetL1RatioGrid:
		m.elasticNetL1RatioGrid = strings.Join(strings.Fields(value), "")
	case textFieldRfMaxDepthGrid:
		m.rfMaxDepthGrid = strings.Join(strings.Fields(value), "")
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
		if len(m.availableWindows) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.ComparisonWindowsSpec), m.availableWindows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown window(s): "+strings.Join(unknown, ", "), "warning")
			}
		}
	case plotItemConfigFieldComparisonSegment:
		cfg.ComparisonSegment = strings.TrimSpace(value)
		if cfg.ComparisonSegment != "" && len(m.availableWindows) > 0 {
			unknown := unknownFromList([]string{cfg.ComparisonSegment}, m.availableWindows)
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
	optFeatGroupSpatialTransform
	optFeatGroupStorage
	optFeatGroupExecution
	optFeatGroupValidation
	optFeatGroupSourceLoc
	// Behavior section headers (expand/collapse)
	optBehaviorGroupGeneral
	optBehaviorGroupTrialTable
	optBehaviorGroupCorrelations
	optBehaviorGroupPainSens
	optBehaviorGroupConfounds
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
	optAperiodicRange
	optAperiodicPeakZ
	optAperiodicMinR2
	optAperiodicMinPoints
	optAperiodicPsdBandwidth
	optAperiodicMaxRms
	optAperiodicLineNoiseFreq
	optAperiodicLineNoiseWidthHz
	optAperiodicLineNoiseHarmonics
	optPEOrder
	optPEDelay
	optBurstThresholdMethod
	optBurstThresholdPercentile
	optBurstThreshold
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
	optSpectralEdge
	optSpectralRatioPairs
	optSpectralLineNoiseFreq
	optSpectralLineNoiseWidthHz
	optSpectralLineNoiseHarmonics
	optAsymmetryChannelPairs
	optConnOutputLevel
	optConnGraphMetrics
	optConnGraphProp
	optConnWindowLen
	optConnWindowStep
	optConnAECMode
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
	// Source localization options (LCMV, eLORETA)
	optSourceLocMethod
	optSourceLocSpacing
	optSourceLocParc
	optSourceLocReg
	optSourceLocSnr
	optSourceLocLoose
	optSourceLocDepth
	optSourceLocConnMethod

	optFailOnMissingWindows
	optFailOnMissingNamedWindow
	// Storage options
	optSaveSubjectLevelFeatures
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
	optTrialTableOnlyMode
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
	optConditionCompareColumn
	optConditionCompareWindows
	optConditionCompareValues
	optConditionWindowPrimaryUnit
	optConditionPermutationPrimary
	// Behavior options - Trial table / residual
	optTrialTableFormat
	optTrialTableIncludeFeatures
	optTrialTableIncludeCovars
	optTrialTableIncludeEvents
	optTrialTableAddLagFeatures
	optTrialTableExtraEventCols
	optTrialTableHighMissingFrac
	optFeatureSummariesEnabled
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
	// Behavior options - Confounds
	optConfoundsAddAsCovariates
	optConfoundsMaxCovariates
	optConfoundsQCColumnPatterns
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
	optCorrelationsPreferPainResidual
	optCorrelationsUseCrossfitPainResidual
	optCorrelationsPrimaryUnit
	optCorrelationsPermutationPrimary
	optCorrelationsTargetColumn
	// Behavior options - Pain sensitivity / temporal
	optTemporalResolutionMs
	optTemporalTimeMinMs
	optTemporalTimeMaxMs
	optTemporalSmoothMs
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
	optMLSkipTimeGen
	// Preprocessing options
	optPrepUsePyprep
	optPrepUseIcalabel
	optPrepNJobs
	optPrepMontage
	optPrepResample
	optPrepLFreq
	optPrepHFreq
	optPrepNotch
	optPrepICAMethod
	optPrepICAComp
	optPrepProbThresh
	optPrepEpochsTmin
	optPrepEpochsTmax
	optPrepLineFreq
	optPrepEpochsBaseline
	optPrepEpochsNoBaseline
	optPrepEpochsReject
	// Raw-to-BIDS options
	optRawMontage
	optRawLineFreq
	optRawOverwrite
	optRawZeroBaseOnsets
	optRawTrimToFirstVolume
	optRawEventPrefixes
	optRawKeepAnnotations
	// Merge-behavior options
	optMergeEventPrefixes
	optMergeEventTypes
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
	optPlotQualityWidthPerPlot
	optPlotQualityHeightPerPlot
	optPlotQualityDistributionNCols
	optPlotQualityDistributionMaxFeatures
	optPlotQualityOutlierZThreshold
	optPlotQualityOutlierMaxFeatures
	optPlotQualityOutlierMaxTrials
	optPlotQualitySnrThresholdDb
	optPlotComplexityWidthPerMeasure
	optPlotComplexityHeightPerSegment
	optPlotConnectivityWidthPerCircle
	optPlotConnectivityWidthPerBand
	optPlotConnectivityHeightPerMeasure
	optPlotConnectivityCircleTopFraction
	optPlotConnectivityCircleMinLines
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
	// TFR parameters (features pipeline)
	optFeatGroupTFR
	optTfrFreqMin
	optTfrFreqMax
	optTfrNFreqs
	optTfrMinCycles
	optTfrNCyclesFactor
	optTfrDecim
	optTfrWorkers
	// Machine Learning model hyperparameters
	optElasticNetAlphaGrid
	optElasticNetL1RatioGrid
	optRfNEstimators
	optRfMaxDepthGrid
	// ICA labels to keep
	optIcaLabelsToKeep
	// System/global settings
	optSystemNJobs
	optSystemStrictMode
	optLoggingLevel
)

const (
	expandedNone                     = -1
	expandedConnectivityMeasures     = 0
	expandedConditionCompareColumn   = 1
	expandedConditionCompareValues   = 2
	expandedConditionCompareWindows  = 3
	expandedTemporalConditionColumn  = 4
	expandedTemporalConditionValues  = 5
	expandedClusterConditionColumn   = 6
	expandedClusterConditionValues   = 7
	expandedPlotComparisonColumn     = 8
	expandedPlotComparisonValues     = 9
	expandedPlotComparisonWindows    = 10
	expandedDirectedConnMeasures     = 11
	expandedRunAdjustmentColumn      = 12
	expandedCorrelationsTargetColumn = 13
)

// getFeaturesOptions returns the active advanced options for the features pipeline
func (m Model) getFeaturesOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	if m.isCategorySelected("connectivity") {
		options = append(options, optFeatGroupConnectivity)
		if m.featGroupConnectivityExpanded {
			options = append(options, optConnectivity, optConnOutputLevel, optConnGraphMetrics, optConnGraphProp, optConnWindowLen, optConnWindowStep, optConnAECMode)
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
				optPACSource, optPACNormalize, optPACNSurrogates, optPACAllowHarmonicOverlap, optPACMaxHarmonic, optPACHarmonicToleranceHz, optPACRandomSeed)
		}
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optFeatGroupAperiodic)
		if m.featGroupAperiodicExpanded {
			options = append(options, optAperiodicRange, optAperiodicPeakZ, optAperiodicMinR2, optAperiodicMinPoints, optAperiodicPsdBandwidth, optAperiodicMaxRms, optAperiodicLineNoiseFreq, optAperiodicLineNoiseWidthHz, optAperiodicLineNoiseHarmonics)
		}
	}
	if m.isCategorySelected("complexity") {
		options = append(options, optFeatGroupComplexity)
		if m.featGroupComplexityExpanded {
			options = append(options, optPEOrder, optPEDelay)
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
			options = append(options, optBurstThresholdMethod, optBurstThresholdPercentile, optBurstThreshold, optBurstMinDuration, optBurstBands)
		}
	}
	if m.isCategorySelected("power") {
		options = append(options, optFeatGroupPower)
		if m.featGroupPowerExpanded {
			options = append(options, optPowerRequireBaseline, optPowerBaselineMode)
		}
	}
	if m.isCategorySelected("spectral") {
		options = append(options, optFeatGroupSpectral)
		if m.featGroupSpectralExpanded {
			options = append(options, optSpectralEdge)
		}
	}
	if m.isCategorySelected("ratios") {
		options = append(options, optFeatGroupRatios)
		if m.featGroupRatiosExpanded {
			options = append(options, optSpectralRatioPairs)
		}
	}
	if m.isCategorySelected("asymmetry") {
		options = append(options, optFeatGroupAsymmetry)
		if m.featGroupAsymmetryExpanded {
			options = append(options, optAsymmetryChannelPairs)
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
			options = append(options, optSourceLocMethod, optSourceLocSpacing, optSourceLocParc)
			// Show method-specific options based on selected method
			if m.sourceLocMethod == 0 { // LCMV
				options = append(options, optSourceLocReg)
			} else { // eLORETA
				options = append(options, optSourceLocSnr, optSourceLocLoose, optSourceLocDepth)
			}
			options = append(options, optSourceLocConnMethod)
		}
	}

	// TFR settings (always available for features that use time-frequency)
	options = append(options, optFeatGroupTFR)
	if m.featGroupTFRExpanded {
		options = append(options, optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrNCyclesFactor, optTfrDecim, optTfrWorkers)
	}

	options = append(options, optFeatGroupStorage)
	if m.featGroupStorageExpanded {
		options = append(options, optSaveSubjectLevelFeatures)
	}

	options = append(options, optFeatGroupExecution)
	if m.featGroupExecutionExpanded {
		options = append(options, optMinEpochs)
	}

	options = append(options, optFeatGroupValidation)
	if m.featGroupValidationExpanded {
		options = append(options, optFailOnMissingWindows, optFailOnMissingNamedWindow)
	}

	return options
}

// getPreprocessingOptions returns advanced options for preprocessing
func (m Model) getPreprocessingOptions() []optionType {
	mode := m.modeOptions[m.modeIndex]
	options := []optionType{optUseDefaults}

	if mode == "full" || mode == "bad-channels" {
		options = append(options, optPrepUsePyprep, optPrepNJobs, optPrepMontage, optPrepResample, optPrepLFreq, optPrepHFreq, optPrepNotch, optPrepLineFreq)
	}

	if mode == "full" || mode == "ica" {
		options = append(options, optPrepICAMethod, optPrepICAComp, optPrepUseIcalabel, optPrepProbThresh, optIcaLabelsToKeep)
	}

	if mode == "full" || mode == "epochs" {
		options = append(options, optPrepEpochsTmin, optPrepEpochsTmax, optPrepEpochsNoBaseline, optPrepEpochsBaseline, optPrepEpochsReject)
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
		optRawZeroBaseOnsets,
		optRawTrimToFirstVolume,
		optRawEventPrefixes,
		optRawKeepAnnotations,
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
	switch plot.Group {
	case "machine_learning":
		return false
	default:
		return true
	}
}

func (m Model) plotConfigFields(plot PlotItem) []plotItemConfigField {
	fields := make([]plotItemConfigField, 0, 8)
	if plot.Group == "tfr" {
		fields = append(fields, plotItemConfigFieldTfrDefaultBaselineWindow)
	}
	if m.plotSupportsComparisons(plot) {
		fields = append(fields,
			plotItemConfigFieldCompareWindows,
			plotItemConfigFieldComparisonWindows,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonSegment,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
			plotItemConfigFieldComparisonROIs,
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

	rows := make([]plottingAdvancedRow, 0, 256)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowOption, opt: optUseDefaults})

	rows = append(rows, plottingAdvancedRow{kind: plottingRowSection, label: "Selected Plots"})
	for _, plot := range m.selectedPlotItemsForConfig() {
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

	rows = append(rows, plottingAdvancedRow{kind: plottingRowSection, label: "Global Plot Settings"})
	for _, opt := range m.getPlottingOptions() {
		if opt == optUseDefaults {
			continue
		}
		// Deprecated global options moved into per-plot config.
		if opt == optPlotGroupTFRMisc || opt == optPlotTfrDefaultBaselineWindow {
			continue
		}
		rows = append(rows, plottingAdvancedRow{kind: plottingRowOption, opt: opt})
	}
	return rows
}

func (m Model) getPlottingOptions() []optionType {
	options := []optionType{optUseDefaults}

	options = append(options, optPlotGroupDefaults)
	if m.plotGroupDefaultsExpanded {
		options = append(options, optPlotBboxInches, optPlotPadInches)
	}

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

	options = append(options, optPlotGroupScatter)
	if m.plotGroupScatterExpanded {
		options = append(options,
			optPlotScatterMarkerSizeSmall,
			optPlotScatterMarkerSizeLarge,
			optPlotScatterMarkerSizeDefault,
			optPlotScatterAlpha,
			optPlotScatterEdgecolor,
			optPlotScatterEdgewidth,
		)
	}

	options = append(options, optPlotGroupBar)
	if m.plotGroupBarExpanded {
		options = append(options, optPlotBarAlpha, optPlotBarWidth, optPlotBarCapsize, optPlotBarCapsizeLarge)
	}

	options = append(options, optPlotGroupLine)
	if m.plotGroupLineExpanded {
		options = append(options,
			optPlotLineWidthThin,
			optPlotLineWidthStandard,
			optPlotLineWidthThick,
			optPlotLineWidthBold,
			optPlotLineAlphaStandard,
			optPlotLineAlphaDim,
			optPlotLineAlphaZeroLine,
			optPlotLineAlphaFitLine,
			optPlotLineAlphaDiagonal,
			optPlotLineAlphaReference,
			optPlotLineRegressionWidth,
			optPlotLineResidualWidth,
			optPlotLineQQWidth,
		)
	}

	options = append(options, optPlotGroupHistogram)
	if m.plotGroupHistogramExpanded {
		options = append(options,
			optPlotHistBins,
			optPlotHistBinsBehavioral,
			optPlotHistBinsResidual,
			optPlotHistBinsTFR,
			optPlotHistEdgecolor,
			optPlotHistEdgewidth,
			optPlotHistAlpha,
			optPlotHistAlphaResidual,
			optPlotHistAlphaTFR,
		)
	}

	options = append(options, optPlotGroupKDE)
	if m.plotGroupKDEExpanded {
		options = append(options, optPlotKdePoints, optPlotKdeColor, optPlotKdeLinewidth, optPlotKdeAlpha)
	}

	options = append(options, optPlotGroupErrorbar)
	if m.plotGroupErrorbarExpanded {
		options = append(options, optPlotErrorbarMarkersize, optPlotErrorbarCapsize, optPlotErrorbarCapsizeLarge)
	}

	options = append(options, optPlotGroupText)
	if m.plotGroupTextExpanded {
		options = append(options,
			optPlotTextStatsX,
			optPlotTextStatsY,
			optPlotTextPvalueX,
			optPlotTextPvalueY,
			optPlotTextBootstrapX,
			optPlotTextBootstrapY,
			optPlotTextChannelAnnotationX,
			optPlotTextChannelAnnotationY,
			optPlotTextTitleY,
			optPlotTextResidualQcTitleY,
		)
	}

	options = append(options, optPlotGroupValidation)
	if m.plotGroupValidationExpanded {
		options = append(options,
			optPlotValidationMinBinsForCalibration,
			optPlotValidationMaxBinsForCalibration,
			optPlotValidationSamplesPerBin,
			optPlotValidationMinRoisForFDR,
			optPlotValidationMinPvaluesForFDR,
		)
	}

	options = append(options, optPlotGroupTFRMisc)
	if m.plotGroupTFRMiscExpanded {
		options = append(options, optPlotTfrDefaultBaselineWindow)
	}

	options = append(options, optPlotGroupTopomap)
	if m.plotGroupTopomapExpanded {
		options = append(
			options,
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

	options = append(options, optPlotGroupTFR)
	if m.plotGroupTFRExpanded {
		options = append(options, optPlotTFRLogBase, optPlotTFRPercentageMultiplier)
	}

	options = append(options, optPlotGroupSizing)
	if m.plotGroupSizingExpanded {
		options = append(
			options,
			optPlotRoiWidthPerBand,
			optPlotRoiWidthPerMetric,
			optPlotRoiHeightPerRoi,
			optPlotPowerWidthPerBand,
			optPlotPowerHeightPerSegment,
			optPlotItpcWidthPerBin,
			optPlotItpcHeightPerBand,
			optPlotItpcWidthPerBandBox,
			optPlotItpcHeightBox,
			optPlotPacCmap,
			optPlotPacWidthPerRoi,
			optPlotPacHeightBox,
			optPlotAperiodicWidthPerColumn,
			optPlotAperiodicHeightPerRow,
			optPlotAperiodicNPerm,
			optPlotQualityWidthPerPlot,
			optPlotQualityHeightPerPlot,
			optPlotQualityDistributionNCols,
			optPlotQualityDistributionMaxFeatures,
			optPlotQualityOutlierZThreshold,
			optPlotQualityOutlierMaxFeatures,
			optPlotQualityOutlierMaxTrials,
			optPlotQualitySnrThresholdDb,
			optPlotComplexityWidthPerMeasure,
			optPlotComplexityHeightPerSegment,
			optPlotConnectivityWidthPerCircle,
			optPlotConnectivityWidthPerBand,
			optPlotConnectivityHeightPerMeasure,
			optPlotConnectivityCircleTopFraction,
			optPlotConnectivityCircleMinLines,
		)
	}

	options = append(options, optPlotGroupSelection)
	if m.plotGroupSelectionExpanded {
		options = append(
			options,
			optPlotPacPairs,
			optPlotConnectivityMeasures,
			optPlotSpectralMetrics,
			optPlotBurstsMetrics,
			optPlotAsymmetryStat,
			optPlotTemporalTimeBins,
			optPlotTemporalTimeLabels,
		)
	}

	options = append(options, optPlotGroupComparisons)
	if m.plotGroupComparisonsExpanded {
		options = append(
			options,
			optPlotCompareWindows,
			optPlotComparisonWindows,
			optPlotCompareColumns,
			optPlotComparisonSegment,
			optPlotComparisonColumn,
			optPlotComparisonValues,
			optPlotComparisonLabels,
			optPlotComparisonROIs,
		)
	}

	return options
}

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	// General section - always visible, but contents depend on selected computations
	options = append(options, optBehaviorGroupGeneral)
	if m.behaviorGroupGeneralExpanded {
		// Always show core statistical options
		options = append(options,
			optCorrMethod,
			optRobustCorrelation,
			optBootstrap,
			optRNGSeed,
			optBehaviorNJobs,
			optFDRAlpha,
		)

		// N Permutations - relevant for cluster, temporal, regression, or correlations with permutation enabled
		needsPermutations := m.isComputationSelected("cluster") ||
			m.isComputationSelected("temporal") ||
			m.isComputationSelected("regression") ||
			m.isComputationSelected("correlations")
		if needsPermutations {
			options = append(options, optNPerm)
		}

		// Covariate controls - relevant for regression, models, influence, correlations, stability
		needsCovariates := m.isComputationSelected("regression") ||
			m.isComputationSelected("models") ||
			m.isComputationSelected("influence") ||
			m.isComputationSelected("correlations") ||
			m.isComputationSelected("stability")
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

		// Trial table only mode - relevant when trial_table is selected
		if m.isComputationSelected("trial_table") {
			options = append(options, optTrialTableOnlyMode)
		}

		// Change scores, LOSO stability, Bayes factors - relevant for correlations
		if m.isComputationSelected("correlations") {
			options = append(options,
				optComputeChangeScores,
				optComputeLosoStability,
				optComputeBayesFactors,
			)
		}
	}

	// Trial table section - only show if trial_table computation is selected
	if m.isComputationSelected("trial_table") {
		options = append(options, optBehaviorGroupTrialTable)
		if m.behaviorGroupTrialTableExpanded {
			options = append(options,
				optTrialTableFormat,
				optTrialTableIncludeFeatures,
				optTrialTableIncludeCovars,
				optTrialTableIncludeEvents,
				optTrialTableAddLagFeatures,
				optTrialTableExtraEventCols,
				optTrialTableHighMissingFrac,
				optFeatureSummariesEnabled,
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
	if m.isComputationSelected("correlations") {
		options = append(options, optBehaviorGroupCorrelations)
		if m.behaviorGroupCorrelationsExpanded {
			options = append(options,
				optCorrelationsTargetRating,
				optCorrelationsTargetTemperature,
				optCorrelationsTargetPainResidual,
				optCorrelationsTargetColumn,
				optCorrelationsPreferPainResidual,
				optCorrelationsPrimaryUnit,
				optCorrelationsPermutationPrimary,
				optCorrelationsUseCrossfitPainResidual,
			)
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

	// Output section - always visible
	options = append(options, optBehaviorGroupOutput)
	if m.behaviorGroupOutputExpanded {
		options = append(options, optAlsoSaveCsv)
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
	}

	// Dynamic options based on selected plots/categories
	if m.IsPlotCategorySelected("tfr") || m.IsPlotCategorySelected("features") {
		// ITPC and PAC settings
		options = append(options, optPlotSharedColorbar)
	}

	return options
}

func (m Model) getMLOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optMLNPerm,
		optMLInnerSplits,
		optMLOuterJobs,
		optRNGSeed,
		optMLSkipTimeGen,
		optElasticNetAlphaGrid,
		optElasticNetL1RatioGrid,
		optRfNEstimators,
		optRfMaxDepthGrid,
	}
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
	default:
		return false
	}
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return false
	}
	return options[m.advancedCursor] == opt
}
