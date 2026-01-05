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
// Data Definitions
///////////////////////////////////////////////////////////////////

type Computation struct {
	Key         string
	Name        string
	Description string
}

type FeatureCategory struct {
	Key         string
	Name        string
	Description string
}

var behaviorComputations = []Computation{
	{"correlations", "Correlations", "EEG-rating correlations"},
	{"pain_sensitivity", "Pain Sensitivity", "Individual pain sensitivity analysis"},
	{"condition", "Condition Comparison", "Compare conditions (e.g., ramp vs active)"},
	{"temporal", "Temporal Correlations", "Time-resolved correlation analysis"},
	{"cluster", "Cluster Permutation", "Cluster-based permutation tests"},
	{"mediation", "Mediation Analysis", "Path analysis and mediation models"},
	{"mixed_effects", "Mixed Effects", "Mixed-effects modeling"},
}

var behaviorPostComputations = []Computation{
	{"confounds", "Confounds Audit", "Audit QC confounds vs targets"},
	{"regression", "Trialwise Regression", "Trialwise regression/moderation models"},
	{"models", "Model Families", "Sensitivity model families"},
	{"stability", "Stability (Run/Block)", "Within-subject stability diagnostics"},
	{"consistency", "Consistency Summary", "Effect direction consistency"},
	{"influence", "Influence Diagnostics", "Cook's distance and leverage"},
	{"report", "Subject Report", "Single-subject report"},
}

type FrequencyBand struct {
	Key         string
	Name        string
	Description string
}

type behaviorSection struct {
	Key     string
	Label   string
	Enabled bool
}

var frequencyBands = []FrequencyBand{
	{"delta", "Delta", "Delta band"},
	{"theta", "Theta", "Theta band"},
	{"alpha", "Alpha", "Alpha band"},
	{"beta", "Beta", "Beta band"},
	{"gamma", "Gamma", "Gamma band"},
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

// Feature file selection for behavior pipeline
type FeatureFile struct {
	Key         string
	Name        string
	Description string
}

var featureFileOptions = []FeatureFile{
	{"power", "Power Features", "EEG power spectral features"},
	{"connectivity", "Connectivity", "Functional connectivity features"},
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
	textFieldRawEventPrefixes
	textFieldMergeEventPrefixes
	textFieldMergeEventTypes
	// Behavior advanced config text fields
	textFieldTrialTableExtraEventColumns
	textFieldConfoundsQCColumnPatterns
	// Features advanced config text fields
	textFieldPACPairs
	textFieldBurstBands
	textFieldSpectralRatioPairs
	textFieldAsymmetryChannelPairs
	textFieldERPComponents
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
	textFieldPlotPacCmap
	textFieldPlotPacPairs
	textFieldPlotConnectivityMeasures
	textFieldPlotSpectralMetrics
	textFieldPlotBurstsMetrics
	textFieldPlotTemporalTimeBins
	textFieldPlotTemporalTimeLabels
	textFieldPlotAsymmetryStat
	// Decoding advanced config text fields
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
	{ID: "erp_contrast", Group: "erp", Name: "Contrast", Description: "ERP condition contrasts (Pain vs No-Pain)", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
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
	{ID: "behavior_temporal_topomaps", Group: "behavior", Name: "Temporal Topomaps", Description: "Temporal correlation topomaps", RequiredFiles: []string{"stats/temporal_correlations_by_pain*.npz"}, RequiresStats: true},
	{ID: "behavior_pain_clusters", Group: "behavior", Name: "Pain Clusters", Description: "Cluster-based temporal contrasts", RequiredFiles: []string{"stats/pain_nonpain_time_clusters_*.tsv"}, RequiresStats: true},
	{ID: "behavior_dose_response", Group: "behavior", Name: "Dose Response", Description: "Dose-response curves and contrasts", RequiredFiles: []string{"features_power*.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_top_predictors", Group: "behavior", Name: "Top Predictors", Description: "Top predictors summary", RequiredFiles: []string{"stats/correlations*.tsv"}, RequiresStats: true},
	{ID: "behavior_temperature_models", Group: "behavior", Name: "Temperature Models", Description: "Subject-level temperature→rating diagnostics", RequiredFiles: []string{"stats/trials*.tsv"}, RequiresStats: true},
	{ID: "behavior_stability_groupwise", Group: "behavior", Name: "Stability (Run/Block)", Description: "Within-subject stability of feature→outcome associations", RequiredFiles: []string{"stats/stability_groupwise*.tsv"}, RequiresStats: true},
	// Decoding
	{ID: "decoding_regression_plots", Group: "decoding", Name: "Regression Plots", Description: "LOSO regression diagnostics", RequiredFiles: []string{"decoding/regression/loso_predictions.tsv"}},
	{ID: "decoding_timegen_plots", Group: "decoding", Name: "Time-Generalization", Description: "Time-generalization matrices", RequiredFiles: []string{"decoding/time_generalization/time_generalization_regression.npz"}},
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
	{"decoding", "Decoding", "Decoding regression and time-generalization"},
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

	// Post computation selection (for behavior - computations requiring others)
	postComputations        []Computation
	postComputationSelected map[int]bool
	postComputationCursor   int
	postComputationOffset   int // Scroll offset for post computations list
	computationListFocus    int // 0 = primary computations, 1 = post computations

	// Category selection (for features pipeline)
	categories    []string
	categoryDescs []string
	categoryIndex int
	selected      map[int]bool

	// Band selection (for features and behavior pipeline)
	bands        []FrequencyBand
	bandSelected map[int]bool
	bandCursor   int

	// Spatial mode selection (for features pipeline)
	spatialSelected map[int]bool
	spatialCursor   int

	// Time range input (for features pipeline)
	TimeRanges      []types.TimeRange
	timeRangeCursor int // Which range is focused
	editingRangeIdx int // Which range is being edited (-1 for none)
	editingField    int // 0=name, 1=tmin, 2=tmax

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

	plotGroupTopomapExpanded   bool
	plotGroupTFRExpanded       bool
	plotGroupSizingExpanded    bool
	plotGroupSelectionExpanded bool

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

	plotValidationMinSamplesForPlot        int
	plotValidationMinSamplesForKDE         int
	plotValidationMinSamplesForFit         int
	plotValidationMinSamplesForCalibration int
	plotValidationMinBinsForCalibration    int
	plotValidationMaxBinsForCalibration    int
	plotValidationSamplesPerBin            int
	plotValidationMinRoisForFDR            int
	plotValidationMinPvaluesForFDR         int

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
	expandedOption int // -1 = none expanded
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
	featGroupConnectivityExpanded bool
	featGroupPACExpanded          bool
	featGroupAperiodicExpanded    bool
	featGroupComplexityExpanded   bool
	featGroupBurstsExpanded       bool
	featGroupPowerExpanded        bool
	featGroupSpectralExpanded     bool
	featGroupERPExpanded          bool
	featGroupRatiosExpanded       bool
	featGroupAsymmetryExpanded    bool
	featGroupStorageExpanded      bool
	featGroupExecutionExpanded    bool
	featGroupValidationExpanded   bool
	featGroupTFRExpanded          bool

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
	erpComponentsSpec     string // e.g. n1=0.10-0.20,n2=0.20-0.35,p2=0.35-0.50

	// Burst configuration
	burstThresholdZ  float64
	burstMinDuration int    // ms
	burstBandsSpec   string // e.g. beta,gamma

	// Power configuration
	powerBaselineMode    int // 0: logratio, 1: mean, 2: ratio, 3: zscore, 4: zlogratio
	powerRequireBaseline bool

	// Spectral configuration
	spectralEdgePercentile float64
	spectralRatioPairsSpec string // e.g. theta:beta,alpha:beta
	// Validation & Generic
	minEpochsForFeatures int

	failOnMissingWindows     bool
	failOnMissingNamedWindow bool

	// Asymmetry
	asymmetryChannelPairsSpec string // e.g. F3:F4,C3:C4

	// Connectivity configuration
	connOutputLevel  int // 0: full, 1: global_only
	connGraphMetrics bool
	connGraphProp    float64
	connWindowLen    float64
	connWindowStep   float64
	connAECMode      int // 0: orth, 1: none, 2: sym

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
	behaviorMinSamples    int     // default min samples

	behaviorComputeChangeScores  bool
	behaviorComputeBayesFactors  bool
	behaviorComputeLosoStability bool

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
	behaviorGroupMixedEffectsExpanded bool

	// Trial table / pain residual config (subject-level)
	trialTableFormat          int // 0=parquet, 1=tsv
	trialTableIncludeFeatures bool
	trialTableIncludeCovars   bool
	trialTableIncludeEvents   bool
	trialTableAddLagFeatures  bool
	trialTableExtraEventCols  string
	trialTableValidateEnabled bool
	trialTableRatingMin       float64
	trialTableRatingMax       float64
	trialTableTempMin         float64
	trialTableTempMax         float64
	trialTableHighMissingFrac float64

	painResidualEnabled                bool
	painResidualMethod                 int // 0=spline, 1=poly
	painResidualMinSamples             int
	painResidualPolyDegree             int
	painResidualModelCompareEnabled    bool
	painResidualModelCompareMinSamples int
	painResidualBreakpointEnabled      bool
	painResidualBreakpointMinSamples   int
	painResidualBreakpointCandidates   int
	painResidualBreakpointQlow         float64
	painResidualBreakpointQhigh        float64

	// Confounds
	confoundsAddAsCovariates  bool
	confoundsMaxCovariates    int
	confoundsQCColumnPatterns string

	// Regression
	regressionOutcome            int // 0=rating, 1=pain_residual, 2=temperature
	regressionIncludeTemperature bool
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
	regressionPermutations       int
	regressionMaxFeatures        int // 0 = no limit

	// Models
	modelsIncludeTemperature  bool
	modelsTempControl         int // 0=linear, 1=rating_hat, 2=spline
	modelsTempSplineKnots     int
	modelsTempSplineQlow      float64
	modelsTempSplineQhigh     float64
	modelsTempSplineMinN      int
	modelsIncludeTrialOrder   bool
	modelsIncludePrev         bool
	modelsIncludeRunBlock     bool
	modelsIncludeInteraction  bool
	modelsStandardize         bool
	modelsMinSamples          int
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
	stabilityMethod         int // 0=spearman, 1=pearson
	stabilityOutcome        int // 0=auto, 1=rating, 2=pain_residual
	stabilityGroupColumn    int // 0=auto, 1=run, 2=block
	stabilityPartialTemp    bool
	stabilityMinGroupTrials int
	stabilityMaxFeatures    int
	stabilityAlpha          float64

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
	correlationsTargetRating       bool
	correlationsTargetTemperature  bool
	correlationsTargetPainResidual bool

	// Pain sensitivity
	painSensitivityMinTrials int

	// Report
	reportTopN int

	// Temporal
	temporalResolutionMs int
	temporalSmoothMs     int
	temporalTimeMinMs    int
	temporalTimeMaxMs    int

	// Mixed effects (group-level; still configurable)
	mixedEffectsType int // 0=intercept, 1=intercept_slope

	// Mediation
	mediationMinEffect float64

	// Condition extras
	conditionMinTrials int
	conditionFailFast  bool
	// Cluster-specific
	clusterThreshold float64 // Forming threshold for clusters
	clusterMinSize   int     // Minimum cluster size
	clusterTail      int     // 0=two-tailed, 1=upper, -1=lower
	// Mediation-specific
	mediationBootstrap    int // Bootstrap iterations for mediation
	mediationMaxMediators int // Max mediators to test
	// Mixed effects-specific
	mixedMaxFeatures int // Max features for mixed effects
	// Condition-specific
	conditionEffectThreshold float64 // Min effect size to report

	// Decoding pipeline advanced config
	decodingNPerm int  // Permutations for significance test
	innerSplits   int  // CV inner splits
	skipTimeGen   bool // Skip time generalization
	// Decoding model hyperparameters
	decodingMinTrialsInner int    // min_trials_inner for CV
	elasticNetAlphaGrid    string // alpha grid as comma-separated values
	elasticNetL1RatioGrid  string // l1_ratio grid as comma-separated values
	rfNEstimators          int    // Random forest n_estimators
	rfMaxDepthGrid         string // max_depth grid as comma-separated values (use "null" for None)

	// TFR parameters (for features pipeline)
	tfrFreqMin       float64 // Min frequency for TFR
	tfrFreqMax       float64 // Max frequency for TFR
	tfrNFreqs        int     // Number of frequency bins
	tfrMinCycles     float64 // Minimum cycles for wavelet
	tfrNCyclesFactor float64 // Cycles factor
	tfrDecim         int     // Decimation factor
	tfrWorkers       int     // Workers for parallel TFR (-1 = all)

	// System/global settings
	systemNJobs      int  // Global n_jobs (-1 = all)
	systemStrictMode bool // Strict mode for validation
	loggingLevel     int  // 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

	// ICA labels to keep
	icaLabelsToKeep string // Comma-separated ICA labels (e.g., "brain,other")

	// Preprocessing pipeline advanced config
	prepUsePyprep   bool
	prepUseIcalabel bool
	prepNJobs       int
	prepResample    int
	prepLFreq       float64
	prepHFreq       float64
	prepNotch       int
	prepICAMethod   int // 0: fastica, 1: infomax, 2: picard
	prepICAComp     float64
	prepProbThresh  float64
	prepEpochsTmin  float64
	prepEpochsTmax  float64

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
		Pipeline:                pipeline,
		selected:                make(map[int]bool),
		subjectSelected:         make(map[string]bool),
		computationSelected:     make(map[int]bool),
		postComputationSelected: make(map[int]bool),
		bands:                   frequencyBands,
		bandSelected:            make(map[int]bool),
		spatialSelected:         make(map[int]bool),
		helpOverlay:             help,
		// Advanced config defaults (shared)
		useDefaultAdvanced:            true,
		expandedOption:                expandedNone, // No option expanded initially
		connectivityMeasures:          make(map[int]bool),
		featGroupConnectivityExpanded: false,
		featGroupPACExpanded:          false,
		featGroupAperiodicExpanded:    false,
		featGroupComplexityExpanded:   false,
		featGroupBurstsExpanded:       false,
		featGroupPowerExpanded:        false,
		featGroupSpectralExpanded:     false,
		featGroupERPExpanded:          false,
		featGroupRatiosExpanded:       false,
		featGroupAsymmetryExpanded:    false,
		featGroupStorageExpanded:      true,
		featGroupExecutionExpanded:    true,
		featGroupValidationExpanded:   true,
		plotItemConfigs:               make(map[string]PlotItemConfig),
		plotItemConfigExpanded:        make(map[string]bool),
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
		// Burst defaults
		burstThresholdZ:  2.0,
		burstMinDuration: 50,
		burstBandsSpec:   "beta,gamma",
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
		// Validation & Generic
		minEpochsForFeatures: 10,

		failOnMissingWindows:      false,
		failOnMissingNamedWindow:  true,
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
		behaviorMinSamples:    10,

		behaviorComputeChangeScores:  true,
		behaviorComputeBayesFactors:  false,
		behaviorComputeLosoStability: true,

		trialTableFormat:          1,
		trialTableIncludeFeatures: true,
		trialTableIncludeCovars:   true,
		trialTableIncludeEvents:   true,
		trialTableAddLagFeatures:  true,
		trialTableExtraEventCols:  "",
		trialTableValidateEnabled: true,
		trialTableRatingMin:       0.0,
		trialTableRatingMax:       10.0,
		trialTableTempMin:         25.0,
		trialTableTempMax:         55.0,
		trialTableHighMissingFrac: 0.5,

		painResidualEnabled:                true,
		painResidualMethod:                 0,
		painResidualMinSamples:             10,
		painResidualPolyDegree:             2,
		painResidualModelCompareEnabled:    true,
		painResidualModelCompareMinSamples: 10,
		painResidualBreakpointEnabled:      true,
		painResidualBreakpointMinSamples:   12,
		painResidualBreakpointCandidates:   15,
		painResidualBreakpointQlow:         0.15,
		painResidualBreakpointQhigh:        0.85,

		confoundsAddAsCovariates:  false,
		confoundsMaxCovariates:    3,
		confoundsQCColumnPatterns: "^quality_.*_global_,^quality_.*_ch_",

		regressionOutcome:            0,
		regressionIncludeTemperature: true,
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
		regressionPermutations:       0,
		regressionMaxFeatures:        0,

		modelsIncludeTemperature:  true,
		modelsTempControl:         0,
		modelsTempSplineKnots:     4,
		modelsTempSplineQlow:      0.05,
		modelsTempSplineQhigh:     0.95,
		modelsTempSplineMinN:      12,
		modelsIncludeTrialOrder:   true,
		modelsIncludePrev:         false,
		modelsIncludeRunBlock:     true,
		modelsIncludeInteraction:  true,
		modelsStandardize:         true,
		modelsMinSamples:          20,
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

		stabilityMethod:         0,
		stabilityOutcome:        0,
		stabilityGroupColumn:    0,
		stabilityPartialTemp:    true,
		stabilityMinGroupTrials: 8,
		stabilityMaxFeatures:    50,
		stabilityAlpha:          0.05,

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
		influenceTempSplineMinN:      12,
		influenceIncludeTrialOrder:   true,
		influenceIncludeRunBlock:     true,
		influenceIncludeInteraction:  false,
		influenceStandardize:         true,
		influenceCooksThreshold:      0.0,
		influenceLeverageThreshold:   0.0,

		correlationsTargetRating:       true,
		correlationsTargetTemperature:  true,
		correlationsTargetPainResidual: true,

		painSensitivityMinTrials: 10,
		reportTopN:               15,
		temporalResolutionMs:     50,
		temporalSmoothMs:         100,
		temporalTimeMinMs:        -200,
		temporalTimeMaxMs:        1000,
		mixedEffectsType:         0,
		mediationMinEffect:       0.05,
		// Cluster defaults
		clusterThreshold: 0.05,
		clusterMinSize:   2,
		clusterTail:      0,
		// Mediation defaults
		mediationBootstrap:    1000,
		mediationMaxMediators: 20,
		// Mixed effects defaults
		mixedMaxFeatures: 50,
		// Condition defaults
		conditionEffectThreshold: 0.5,
		conditionMinTrials:       10,
		conditionFailFast:        true,
		// Decoding defaults
		decodingNPerm:          0,
		innerSplits:            3,
		skipTimeGen:            false,
		decodingMinTrialsInner: 3,
		elasticNetAlphaGrid:    "0.001,0.01,0.1,1,10",
		elasticNetL1RatioGrid:  "0.2,0.5,0.8",
		rfNEstimators:          500,
		rfMaxDepthGrid:         "5,10,20,null",
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
		prepUsePyprep:   true,
		prepUseIcalabel: true,
		prepNJobs:       1,
		prepResample:    500,
		prepLFreq:       0.1,
		prepHFreq:       100.0,
		prepNotch:       60,
		prepICAMethod:   0,
		prepICAComp:     0.99,
		prepProbThresh:  0.8,
		prepEpochsTmin:  -5.0,
		prepEpochsTmax:  12.0,

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
	m.TimeRanges = []types.TimeRange{
		{Name: "baseline", Tmin: "", Tmax: ""},
		{Name: "active", Tmin: "", Tmax: ""},
	}
	m.editingRangeIdx = -1

	switch pipeline {
	case types.PipelineFeatures:
		m.modeOptions = []string{styles.ModeCompute}
		m.modeDescriptions = []string{
			"Extract EEG feature sets",
		}
		m.categories = []string{
			"power", "spectral", "aperiodic", "erp", "erds", "ratios", "asymmetry",
			"connectivity", "itpc", "pac",
			"complexity", "bursts", "quality",
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
			"Inter-trial phase coh.",
			"Phase-amplitude coupling",
			"Signal complexity",
			"Oscillatory burst dynamics",
			"Trial quality metrics",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepConfigureOptions, // Category selection
			types.StepSelectBands,
			types.StepSelectSpatial,
			types.StepTimeRange,
			types.StepAdvancedConfig,
			types.StepReviewExecute,
		}
		for i := range frequencyBands {
			m.bandSelected[i] = true
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
		m.postComputations = behaviorPostComputations

		// Default selections for primary computations
		defaultPrimaryComps := map[string]bool{
			"correlations":     true,
			"pain_sensitivity": true,
			"condition":        true,
			"temporal":         false,
			"cluster":          false,
			"mediation":        false,
			"mixed_effects":    false,
		}
		for i, c := range behaviorComputations {
			m.computationSelected[i] = defaultPrimaryComps[c.Key]
		}

		// Default selections for post computations
		defaultPostComps := map[string]bool{
			"confounds":   true,
			"regression":  false,
			"models":      false,
			"stability":   true,
			"consistency": true,
			"influence":   true,
			"report":      true,
		}
		for i, c := range behaviorPostComputations {
			m.postComputationSelected[i] = defaultPostComps[c.Key]
		}

		// Initialize feature file selection
		m.featureFiles = featureFileOptions
		m.featureFileSelected = make(map[string]bool)

		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectComputations,
			types.StepSelectFeatureFiles,
			types.StepSelectBands,
			types.StepAdvancedConfig,
			types.StepReviewExecute,
		}
		for i := range frequencyBands {
			m.bandSelected[i] = true
		}

	case types.PipelineDecoding:
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
				if len(m.subjectFilter) > 0 {
					m.subjectFilter = m.subjectFilter[:len(m.subjectFilter)-1]
				}
			default:
				if len(msg.String()) == 1 {
					m.subjectFilter += msg.String()
				}
			}
			return m, nil
		}

		if m.ConfirmingExecute {
			switch msg.String() {
			case "y", "Y", "enter":
				m.ConfirmingExecute = false
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
				if len(m.textBuffer) > 0 {
					m.textBuffer = m.textBuffer[:len(m.textBuffer)-1]
				}
			default:
				if len(msg.String()) == 1 {
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
				if len(m.numberBuffer) > 0 {
					m.numberBuffer = m.numberBuffer[:len(m.numberBuffer)-1]
				}
			default:
				// Accept digits, decimal point, and minus sign
				char := msg.String()
				if len(char) == 1 && (char >= "0" && char <= "9" || char == "." || char == "-") {
					m.numberBuffer += char
				}
			}
			return m, nil
		}

		// Handle time range input for tmin/tmax
		if m.editingRangeIdx >= 0 && m.editingRangeIdx < len(m.TimeRanges) {
			switch msg.String() {
			case "esc":
				m.editingRangeIdx = -1
			case "enter":
				// Commit and move to next field, or exit if at end
				if m.editingField < 2 {
					m.editingField++
				} else {
					m.editingRangeIdx = -1
					m.editingField = 0
				}
			case "tab":
				// Cycle through fields
				m.editingField = (m.editingField + 1) % 3
			case "backspace":
				ref := &m.TimeRanges[m.editingRangeIdx]
				if m.editingField == 0 && len(ref.Name) > 0 {
					ref.Name = ref.Name[:len(ref.Name)-1]
				} else if m.editingField == 1 && len(ref.Tmin) > 0 {
					ref.Tmin = ref.Tmin[:len(ref.Tmin)-1]
				} else if m.editingField == 2 && len(ref.Tmax) > 0 {
					ref.Tmax = ref.Tmax[:len(ref.Tmax)-1]
				}
			default:
				r := msg.String()
				if len(r) == 1 {
					ref := &m.TimeRanges[m.editingRangeIdx]
					if m.editingField == 0 {
						ref.Name += r
					} else {
						// For numeric fields, only accept digits, dot, minus
						if (r >= "0" && r <= "9") || r == "." || r == "-" {
							if m.editingField == 1 {
								ref.Tmin += r
							} else {
								ref.Tmax += r
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
			m.handleSpace()
		case "enter":
			return m.handleEnter()
		case "tab":
			m.handleTab()
		case "a":
			if m.CurrentStep == types.StepTimeRange && m.editingRangeIdx == -1 {
				newName := fmt.Sprintf("range%d", len(m.TimeRanges)+1)
				m.TimeRanges = append(m.TimeRanges, types.TimeRange{Name: newName, Tmin: "", Tmax: ""})
				m.timeRangeCursor = len(m.TimeRanges) - 1
				m.editingRangeIdx = m.timeRangeCursor
				m.editingField = 0 // Focus Name for new range
			} else {
				m.selectAll()
			}
		case "d", "x":
			if m.CurrentStep == types.StepTimeRange && m.editingRangeIdx == -1 {
				if len(m.TimeRanges) > 0 {
					idx := m.timeRangeCursor
					m.TimeRanges = append(m.TimeRanges[:idx], m.TimeRanges[idx+1:]...)
					if m.timeRangeCursor >= len(m.TimeRanges) {
						m.timeRangeCursor = len(m.TimeRanges) - 1
					}
					if m.timeRangeCursor < 0 {
						m.timeRangeCursor = 0
					}
				}
			}
		case "n":
			m.selectNone()

		case "Q":
			// Quick preset
			if m.CurrentStep == types.StepConfigureOptions && m.Pipeline == types.PipelineFeatures {
				m.ApplyFeaturePreset("quick")
			} else if m.CurrentStep == types.StepSelectComputations && m.Pipeline == types.PipelineBehavior {
				m.ApplyBehaviorPreset("quick")
			}

		case "F":
			// Full preset
			if m.CurrentStep == types.StepConfigureOptions && m.Pipeline == types.PipelineFeatures {
				m.ApplyFeaturePreset("full")
			} else if m.CurrentStep == types.StepSelectComputations && m.Pipeline == types.PipelineBehavior {
				m.ApplyBehaviorPreset("full")
			}

		case "C":
			// Connectivity preset (features only)
			if m.CurrentStep == types.StepConfigureOptions && m.Pipeline == types.PipelineFeatures {
				m.ApplyFeaturePreset("connectivity")
			}

		case "S":
			// Spectral preset (features only)
			if m.CurrentStep == types.StepConfigureOptions && m.Pipeline == types.PipelineFeatures {
				m.ApplyFeaturePreset("spectral")
			}

		case "R":
			// Regression preset (behavior only)
			if m.CurrentStep == types.StepSelectComputations && m.Pipeline == types.PipelineBehavior {
				m.ApplyBehaviorPreset("regression")
			}

		case "T":
			// Temporal preset (behavior only)
			if m.CurrentStep == types.StepSelectComputations && m.Pipeline == types.PipelineBehavior {
				m.ApplyBehaviorPreset("temporal")
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
	maxLines := m.height - 16
	if maxLines < 8 {
		maxLines = 8
	}

	total := len(m.computations)
	if total <= 0 {
		m.computationOffset = 0
		return
	}
	cursorLine := m.computationCursor
	if cursorLine < 0 {
		cursorLine = 0
	}
	if cursorLine >= total {
		cursorLine = total - 1
	}

	if cursorLine < m.computationOffset {
		m.computationOffset = cursorLine
	} else if cursorLine >= m.computationOffset+maxLines {
		m.computationOffset = cursorLine - maxLines + 1
	}

	if m.computationOffset < 0 {
		m.computationOffset = 0
	}
	if total > maxLines && m.computationOffset > total-maxLines {
		m.computationOffset = total - maxLines
	}
}

// UpdateAdvancedOffset calculates and updates the scrolling offset for advanced config lists.
func (m *Model) UpdateAdvancedOffset() {
	if m.height <= 0 {
		m.advancedOffset = 0
		return
	}

	// Total height minus overhead (approx; reduced for more visible content).
	// Must match values used in render_steps.go for consistency.
	maxLines := m.height - 12
	switch m.Pipeline {
	case types.PipelineFeatures:
		maxLines = m.height - 10
	case types.PipelineBehavior:
		maxLines = m.height - 12
	}
	if maxLines < 8 {
		maxLines = 8
	}

	totalLines := 0
	cursorLine := 0

	switch m.Pipeline {
	case types.PipelineBehavior:
		options := m.getBehaviorOptions()
		totalLines = len(options)
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
	if cursorLine < 0 {
		cursorLine = 0
	}
	if cursorLine >= totalLines {
		cursorLine = totalLines - 1
	}

	maxOffset := totalLines - maxLines
	if maxOffset < 0 {
		maxOffset = 0
	}

	if cursorLine < m.advancedOffset {
		m.advancedOffset = cursorLine
	} else if cursorLine >= m.advancedOffset+maxLines {
		m.advancedOffset = cursorLine - maxLines + 1
	}

	if m.advancedOffset < 0 {
		m.advancedOffset = 0
	}
	if m.advancedOffset > maxOffset {
		m.advancedOffset = maxOffset
	}
}

// UpdatePlotOffset calculates and updates the scrolling offset for the plots list
func (m *Model) UpdatePlotOffset() {
	// Total height minus overhead (approx 18-20 lines)
	maxLines := m.height - 18
	if maxLines < 10 {
		maxLines = 10 // Minimum window
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

	// Adjust offset
	if cursorLine < m.plotOffset {
		m.plotOffset = cursorLine
	} else if cursorLine >= m.plotOffset+maxLines {
		m.plotOffset = cursorLine - maxLines + 1
	}

	// Bound check
	if m.plotOffset < 0 {
		m.plotOffset = 0
	}
	if lineIdx > maxLines && m.plotOffset > lineIdx-maxLines {
		m.plotOffset = lineIdx - maxLines
	}
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
	maxLines := m.height - 16
	if maxLines < 10 {
		maxLines = 10
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

	if cursorLine < m.featurePlotterOffset {
		m.featurePlotterOffset = cursorLine
	} else if cursorLine >= m.featurePlotterOffset+maxLines {
		m.featurePlotterOffset = cursorLine - maxLines + 1
	}
	if m.featurePlotterOffset < 0 {
		m.featurePlotterOffset = 0
	}
	if lineIdx > maxLines && m.featurePlotterOffset > lineIdx-maxLines {
		m.featurePlotterOffset = lineIdx - maxLines
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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
		ok := true
		if plot.RequiresEpochs && !s.HasEpochs {
			missing["epochs"]++
			ok = false
		}
		if plot.RequiresFeatures && !s.HasFeatures {
			missing["features"]++
			ok = false
		}
		if plot.RequiresStats && !s.HasStats {
			missing["stats"]++
			ok = false
		}
		if ok {
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
		// Only consider selected subjects
		if !m.subjectSelected[s.ID] {
			continue
		}

		if s.FeatureAvailability == nil {
			continue
		}

		for cat, info := range s.FeatureAvailability.Features {
			if info.Available {
				m.featureAvailability[cat] = true
				if info.LastModified != "" {
					if existing, ok := m.featureLastModified[cat]; !ok || info.LastModified > existing {
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
		if !m.subjectSelected[s.ID] {
			continue
		}

		if s.FeatureAvailability == nil || s.FeatureAvailability.Computations == nil {
			continue
		}

		for comp, info := range s.FeatureAvailability.Computations {
			if info.Available {
				m.computationAvailability[comp] = true
				if info.LastModified != "" {
					if existing, ok := m.computationLastModified[comp]; !ok || info.LastModified > existing {
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
	case textFieldRawEventPrefixes:
		return m.rawEventPrefixes
	case textFieldMergeEventPrefixes:
		return m.mergeEventPrefixes
	case textFieldMergeEventTypes:
		return m.mergeEventTypes
	case textFieldTrialTableExtraEventColumns:
		return m.trialTableExtraEventCols
	case textFieldConfoundsQCColumnPatterns:
		return m.confoundsQCColumnPatterns
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
	// Decoding hyperparameter text fields
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
		low, okLow := entry[0].(float64)
		high, okHigh := entry[1].(float64)
		if !okLow || !okHigh {
			continue
		}
		bands = append(bands, FrequencyBand{
			Key:         name,
			Name:        titleCase(name),
			Description: fmt.Sprintf("%.1f-%.1f Hz", low, high),
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
		{Key: "confounds", Label: "Confounds", Enabled: m.isComputationSelected("confounds")},
		{Key: "regression", Label: "Regression", Enabled: m.isComputationSelected("regression")},
		{Key: "models", Label: "Models", Enabled: m.isComputationSelected("models")},
		{Key: "stability", Label: "Stability", Enabled: m.isComputationSelected("stability")},
		{Key: "consistency", Label: "Consistency", Enabled: m.isComputationSelected("consistency")},
		{Key: "influence", Label: "Influence", Enabled: m.isComputationSelected("influence")},
		{Key: "condition", Label: "Condition", Enabled: m.isComputationSelected("condition")},
		{Key: "temporal", Label: "Temporal", Enabled: m.isComputationSelected("temporal")},
		{Key: "cluster", Label: "Cluster", Enabled: m.isComputationSelected("cluster")},
		{Key: "mediation", Label: "Mediation", Enabled: m.isComputationSelected("mediation")},
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
	case textFieldRawEventPrefixes:
		m.rawEventPrefixes = value
	case textFieldMergeEventPrefixes:
		m.mergeEventPrefixes = value
	case textFieldMergeEventTypes:
		m.mergeEventTypes = value
	case textFieldTrialTableExtraEventColumns:
		m.trialTableExtraEventCols = value
	case textFieldConfoundsQCColumnPatterns:
		m.confoundsQCColumnPatterns = value
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
	// Decoding hyperparameter text fields
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
	optFeatGroupPAC
	optFeatGroupAperiodic
	optFeatGroupComplexity
	optFeatGroupBursts
	optFeatGroupPower
	optFeatGroupSpectral
	optFeatGroupERP
	optFeatGroupRatios
	optFeatGroupAsymmetry
	optFeatGroupStorage
	optFeatGroupExecution
	optFeatGroupValidation
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
	optBehaviorGroupMixedEffects
	optMicrostateStates
	optGroupTemplates
	optFixedTemplates
	optConnectivity
	optPACPhaseRange
	optPACAmpRange
	optPACMethod
	optPACMinEpochs
	optPACPairs
	optAperiodicRange
	optAperiodicPeakZ
	optAperiodicMinR2
	optAperiodicMinPoints
	optPEOrder
	optPEDelay
	optBurstPercentile
	optERPBaseline
	optERPAllowNoBaseline
	optERPComponents
	optBurstThreshold
	optBurstMinDuration
	optBurstBands
	optPowerBaselineMode
	optPowerRequireBaseline
	optSpectralEdge
	optSpectralRatioPairs
	optAsymmetryChannelPairs
	optConnOutputLevel
	optConnGraphMetrics
	optConnGraphProp
	optConnWindowLen
	optConnWindowStep
	optConnAECMode
	optMinEpochs

	optFailOnMissingWindows
	optFailOnMissingNamedWindow
	// Behavior options - General
	optCorrMethod
	optBootstrap
	optNPerm
	optRNGSeed
	optControlTemp
	optControlOrder
	optTrialTableOnlyMode
	optFDRAlpha
	// Behavior options - Cluster
	optClusterThreshold
	optClusterMinSize
	optClusterTail
	// Behavior options - Mediation
	optMediationBootstrap
	optMediationMaxMediators
	// Behavior options - Mixed Effects
	optMixedMaxFeatures
	// Behavior options - Condition
	optConditionEffectThreshold
	optConditionMinTrials
	optConditionFailFast
	// Behavior options - Trial table / residual
	optTrialTableFormat
	optTrialTableIncludeFeatures
	optTrialTableIncludeCovars
	optTrialTableIncludeEvents
	optTrialTableAddLagFeatures
	optTrialTableExtraEventCols
	optTrialTableValidate
	optTrialTableRatingMin
	optTrialTableRatingMax
	optTrialTableTempMin
	optTrialTableTempMax
	optTrialTableHighMissingFrac
	optFeatureSummariesEnabled
	optPainResidualEnabled
	optPainResidualMethod
	optPainResidualMinSamples
	optPainResidualPolyDegree
	optPainResidualModelCompare
	optPainResidualModelCompareMinSamples
	optPainResidualBreakpoint
	optPainResidualBreakpointMinSamples
	optPainResidualBreakpointCandidates
	optPainResidualBreakpointQlow
	optPainResidualBreakpointQhigh
	// Behavior options - General extra
	optRobustCorrelation
	optBehaviorNJobs
	optBehaviorMinSamples
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
	optRegressionTempSplineMinSamples
	optRegressionIncludeTrialOrder
	optRegressionIncludePrev
	optRegressionIncludeRunBlock
	optRegressionIncludeInteraction
	optRegressionStandardize
	optRegressionMinSamples
	optRegressionPermutations
	optRegressionMaxFeatures
	// Behavior options - Models
	optModelsIncludeTemperature
	optModelsTempControl
	optModelsTempSplineKnots
	optModelsTempSplineQlow
	optModelsTempSplineQhigh
	optModelsTempSplineMinSamples
	optModelsIncludeTrialOrder
	optModelsIncludePrev
	optModelsIncludeRunBlock
	optModelsIncludeInteraction
	optModelsStandardize
	optModelsMinSamples
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
	optStabilityMinGroupTrials
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
	optInfluenceTempSplineMinSamples
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
	// Behavior options - Pain sensitivity / temporal
	optPainSensitivityMinTrials
	optTemporalResolutionMs
	optTemporalTimeMinMs
	optTemporalTimeMaxMs
	optTemporalSmoothMs
	// Behavior options - Mixed effects / mediation
	optMixedEffectsType
	optMediationMinEffect
	// Plotting options
	optPlotPNG
	optPlotSVG
	optPlotPDF
	optPlotDPI
	optPlotSaveDPI
	optPlotSharedColorbar
	// Decoding options
	optDecodingNPerm
	optDecodingInnerSplits
	optDecodingSkipTimeGen
	// Preprocessing options
	optPrepUsePyprep
	optPrepUseIcalabel
	optPrepNJobs
	optPrepResample
	optPrepLFreq
	optPrepHFreq
	optPrepNotch
	optPrepICAMethod
	optPrepICAComp
	optPrepProbThresh
	optPrepEpochsTmin
	optPrepEpochsTmax
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
	optPlotValidationMinSamplesForPlot
	optPlotValidationMinSamplesForKDE
	optPlotValidationMinSamplesForFit
	optPlotValidationMinSamplesForCalibration
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
	// TFR parameters (features pipeline)
	optFeatGroupTFR
	optTfrFreqMin
	optTfrFreqMax
	optTfrNFreqs
	optTfrMinCycles
	optTfrNCyclesFactor
	optTfrDecim
	optTfrWorkers
	// Decoding model hyperparameters
	optDecodingMinTrialsInner
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
	expandedNone                 = -1
	expandedConnectivityMeasures = 0
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

	if m.isCategorySelected("pac") {
		options = append(options, optFeatGroupPAC)
		if m.featGroupPACExpanded {
			options = append(options, optPACPhaseRange, optPACAmpRange, optPACMethod, optPACMinEpochs, optPACPairs)
		}
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optFeatGroupAperiodic)
		if m.featGroupAperiodicExpanded {
			options = append(options, optAperiodicRange, optAperiodicPeakZ, optAperiodicMinR2, optAperiodicMinPoints)
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
			options = append(options, optERPBaseline, optERPAllowNoBaseline, optERPComponents)
		}
	}
	if m.isCategorySelected("bursts") {
		options = append(options, optFeatGroupBursts)
		if m.featGroupBurstsExpanded {
			options = append(options, optBurstThreshold, optBurstMinDuration, optBurstBands)
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

	// TFR settings (always available for features that use time-frequency)
	options = append(options, optFeatGroupTFR)
	if m.featGroupTFRExpanded {
		options = append(options, optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrNCyclesFactor, optTfrDecim, optTfrWorkers)
	}

	options = append(options, optFeatGroupStorage)
	if m.featGroupStorageExpanded {
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
		options = append(options, optPrepUsePyprep, optPrepNJobs, optPrepResample, optPrepLFreq, optPrepHFreq, optPrepNotch)
	}

	if mode == "full" || mode == "ica" {
		options = append(options, optPrepICAMethod, optPrepICAComp, optPrepUseIcalabel, optPrepProbThresh, optIcaLabelsToKeep)
	}

	if mode == "full" || mode == "epochs" {
		options = append(options, optPrepEpochsTmin, optPrepEpochsTmax)
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
	case "features", "behavior", "tfr", "erp":
		return true
	default:
		return false
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
			optPlotValidationMinSamplesForPlot,
			optPlotValidationMinSamplesForKDE,
			optPlotValidationMinSamplesForFit,
			optPlotValidationMinSamplesForCalibration,
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

	return options
}

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	// General section - always visible
	options = append(options, optBehaviorGroupGeneral)
	if m.behaviorGroupGeneralExpanded {
		options = append(options,
			optCorrMethod,
			optRobustCorrelation,
			optBootstrap,
			optNPerm,
			optRNGSeed,
			optBehaviorNJobs,
			optBehaviorMinSamples,
			optControlTemp,
			optControlOrder,
			optTrialTableOnlyMode,
			optFDRAlpha,
			optComputeChangeScores,
			optComputeLosoStability,
			optComputeBayesFactors,
		)
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
				optTrialTableValidate,
				optTrialTableRatingMin,
				optTrialTableRatingMax,
				optTrialTableTempMin,
				optTrialTableTempMax,
				optTrialTableHighMissingFrac,
				optFeatureSummariesEnabled,
				optPainResidualEnabled,
				optPainResidualMethod,
				optPainResidualMinSamples,
				optPainResidualPolyDegree,
				optPainResidualModelCompare,
				optPainResidualModelCompareMinSamples,
				optPainResidualBreakpoint,
				optPainResidualBreakpointMinSamples,
				optPainResidualBreakpointCandidates,
				optPainResidualBreakpointQlow,
				optPainResidualBreakpointQhigh,
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
			)
		}
	}

	// Confounds section
	if m.isComputationSelected("confounds") {
		options = append(options, optBehaviorGroupConfounds)
		if m.behaviorGroupConfoundsExpanded {
			options = append(options, optConfoundsAddAsCovariates, optConfoundsMaxCovariates, optConfoundsQCColumnPatterns)
		}
	}

	// Regression section
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
					optRegressionTempSplineMinSamples,
				)
			}
			options = append(options,
				optRegressionIncludeTrialOrder,
				optRegressionIncludePrev,
				optRegressionIncludeRunBlock,
				optRegressionIncludeInteraction,
				optRegressionStandardize,
				optRegressionMinSamples,
				optRegressionPermutations,
				optRegressionMaxFeatures,
			)
		}
	}

	// Models section
	if m.isComputationSelected("models") {
		options = append(options, optBehaviorGroupModels)
		if m.behaviorGroupModelsExpanded {
			options = append(options,
				optModelsIncludeTemperature,
				optModelsTempControl,
			)
			if m.modelsTempControl == 2 {
				options = append(options,
					optModelsTempSplineKnots,
					optModelsTempSplineQlow,
					optModelsTempSplineQhigh,
					optModelsTempSplineMinSamples,
				)
			}
			options = append(options,
				optModelsIncludeTrialOrder,
				optModelsIncludePrev,
				optModelsIncludeRunBlock,
				optModelsIncludeInteraction,
				optModelsStandardize,
				optModelsMinSamples,
				optModelsMaxFeatures,
				optModelsOutcomeRating,
				optModelsOutcomePainResidual,
				optModelsOutcomeTemperature,
				optModelsOutcomePainBinary,
				optModelsFamilyOLS,
				optModelsFamilyRobust,
				optModelsFamilyQuantile,
				optModelsFamilyLogit,
				optModelsBinaryOutcome,
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
				optStabilityMinGroupTrials,
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
					optInfluenceTempSplineMinSamples,
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
			options = append(options, optConditionFailFast, optConditionEffectThreshold, optConditionMinTrials)
		}
	}

	// Temporal section
	if m.isComputationSelected("temporal") {
		options = append(options, optBehaviorGroupTemporal)
		if m.behaviorGroupTemporalExpanded {
			options = append(options, optTemporalResolutionMs, optTemporalTimeMinMs, optTemporalTimeMaxMs, optTemporalSmoothMs)
		}
	}

	// Cluster section
	if m.isComputationSelected("cluster") {
		options = append(options, optBehaviorGroupCluster)
		if m.behaviorGroupClusterExpanded {
			options = append(options, optClusterThreshold, optClusterMinSize, optClusterTail)
		}
	}

	// Mediation section
	if m.isComputationSelected("mediation") {
		options = append(options, optBehaviorGroupMediation)
		if m.behaviorGroupMediationExpanded {
			options = append(options, optMediationBootstrap, optMediationMinEffect, optMediationMaxMediators)
		}
	}

	// Mixed effects section
	if m.isComputationSelected("mixed_effects") {
		options = append(options, optBehaviorGroupMixedEffects)
		if m.behaviorGroupMixedEffectsExpanded {
			options = append(options, optMixedEffectsType, optMixedMaxFeatures)
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
	}

	// Dynamic options based on selected plots/categories
	if m.IsPlotCategorySelected("tfr") || m.IsPlotCategorySelected("features") {
		// ITPC and PAC settings
		options = append(options, optPlotSharedColorbar)
	}

	return options
}

func (m Model) getDecodingOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optDecodingNPerm,
		optDecodingInnerSplits,
		optDecodingMinTrialsInner,
		optRNGSeed,
		optDecodingSkipTimeGen,
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
	case types.PipelineDecoding:
		options = m.getDecodingOptions()
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
