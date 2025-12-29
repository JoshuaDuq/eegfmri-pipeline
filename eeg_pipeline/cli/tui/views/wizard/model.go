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
	{"trial_table", "Trial Table", "Export canonical per-trial analysis table"},
	{"confounds", "Confounds Audit", "Audit QC confounds vs targets"},
	{"regression", "Trialwise Regression", "Trialwise regression/moderation models"},
	{"models", "Model Families", "Sensitivity model families (robust/quantile/logistic)"},
	{"stability", "Stability (Run/Block)", "Within-subject stability diagnostics (non-gating)"},
	{"consistency", "Consistency Summary", "Effect direction consistency across outcomes"},
	{"influence", "Influence Diagnostics", "Cook's distance and leverage summaries"},
	{"report", "Subject Report", "Single-subject report (reproducible summary)"},
	{"correlations", "Correlations", "EEG-rating correlations"},
	{"pain_sensitivity", "Pain Sensitivity", "Individual pain sensitivity analysis"},
	{"condition", "Condition Comparison", "Compare conditions (e.g., ramp vs active)"},
	{"temporal", "Temporal Correlations", "Time-resolved correlation analysis"},
	{"cluster", "Cluster Permutation", "Cluster-based permutation tests"},
	{"mediation", "Mediation Analysis", "Path analysis and mediation models"},
	{"mixed_effects", "Mixed Effects", "Mixed-effects modeling"},
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
	{"all", "All Combined", "All features combined (features_all.tsv)"},
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
)

var defaultPlotItems = []PlotItem{
	// Features
	{ID: "features_power", Group: "features", Name: "Power", Description: "Band power summaries and topomaps", RequiredFiles: []string{"features_power.tsv"}, RequiresFeatures: true},
	{ID: "features_connectivity", Group: "features", Name: "Connectivity", Description: "Connectivity heatmaps and networks", RequiredFiles: []string{"features_connectivity.parquet"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "features_aperiodic", Group: "features", Name: "Aperiodic", Description: "1/f spectral slope diagnostics", RequiredFiles: []string{"features_aperiodic.tsv"}, RequiresFeatures: true},
	{ID: "features_itpc", Group: "features", Name: "ITPC", Description: "Inter-trial phase coherence plots", RequiredFiles: []string{"features_itpc.tsv", "stats/itpc_data.npz"}, RequiresFeatures: true},
	{ID: "features_pac", Group: "features", Name: "PAC", Description: "Phase-amplitude coupling plots", RequiredFiles: []string{"features_pac*.tsv"}, RequiresFeatures: true},
	{ID: "features_erds", Group: "features", Name: "ERDS", Description: "Event-related desync/sync plots", RequiredFiles: []string{"features_erds.tsv"}, RequiresFeatures: true},
	{ID: "features_complexity", Group: "features", Name: "Complexity", Description: "Complexity distributions and condition comparisons", RequiredFiles: []string{"features_complexity.tsv"}, RequiresFeatures: true},
	{ID: "features_quality", Group: "features", Name: "Quality", Description: "Feature quality diagnostics and outlier views", RequiredFiles: []string{"features_quality.tsv"}, RequiresFeatures: true},
	{ID: "features_spectral", Group: "features", Name: "Spectral", Description: "Spectral peak and edge features", RequiredFiles: []string{"features_spectral.tsv"}, RequiresFeatures: true},
	{ID: "features_ratios", Group: "features", Name: "Ratios", Description: "Band power ratios", RequiredFiles: []string{"features_ratios.tsv"}, RequiresFeatures: true},
	{ID: "features_asymmetry", Group: "features", Name: "Asymmetry", Description: "Hemispheric asymmetry indices", RequiredFiles: []string{"features_asymmetry.tsv"}, RequiresFeatures: true},
	{ID: "features_bursts", Group: "features", Name: "Bursts", Description: "Oscillatory burst dynamics", RequiredFiles: []string{"features_bursts.tsv"}, RequiresFeatures: true},
	{ID: "features_erp", Group: "features", Name: "ERP", Description: "ERP visualizations from epochs", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Behavior
	{ID: "behavior_psychometrics", Group: "behavior", Name: "Psychometrics", Description: "Rating distributions and psychometrics", RequiredFiles: []string{"events.tsv"}},
	{ID: "behavior_power_scatter", Group: "behavior", Name: "Power ROI Scatter", Description: "Power vs behavior scatter plots", RequiredFiles: []string{"features_power.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true, Dependencies: []string{"features_power"}},
	{ID: "behavior_complexity_scatter", Group: "behavior", Name: "Complexity Scatter", Description: "Complexity vs behavior scatter plots", RequiredFiles: []string{"features_complexity.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true, Dependencies: []string{"features_complexity"}},
	{ID: "behavior_aperiodic_scatter", Group: "behavior", Name: "Aperiodic Scatter", Description: "Aperiodic vs behavior scatter plots", RequiredFiles: []string{"features_aperiodic.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true, Dependencies: []string{"features_aperiodic"}},
	{ID: "behavior_connectivity_scatter", Group: "behavior", Name: "Connectivity Scatter", Description: "Connectivity vs behavior scatter plots", RequiredFiles: []string{"features_connectivity.parquet", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true, Dependencies: []string{"features_connectivity"}},
	{ID: "behavior_itpc_scatter", Group: "behavior", Name: "ITPC Scatter", Description: "ITPC vs behavior scatter plots", RequiredFiles: []string{"features_itpc.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true, Dependencies: []string{"features_itpc"}},
	{ID: "behavior_temporal_topomaps", Group: "behavior", Name: "Temporal Topomaps", Description: "Temporal correlation topomaps", RequiredFiles: []string{"stats/temporal_correlations_by_pain*.npz"}, RequiresStats: true},
	{ID: "behavior_pain_clusters", Group: "behavior", Name: "Pain Clusters", Description: "Cluster-based temporal contrasts", RequiredFiles: []string{"stats/pain_nonpain_time_clusters_*.tsv"}, RequiresStats: true},
	{ID: "behavior_dose_response", Group: "behavior", Name: "Dose Response", Description: "Dose-response curves and contrasts", RequiredFiles: []string{"features_power.tsv", "epochs/*.fif"}, RequiresEpochs: true, RequiresFeatures: true},
	{ID: "behavior_top_predictors", Group: "behavior", Name: "Top Predictors", Description: "Top predictors summary", RequiredFiles: []string{"stats/correlations*.tsv"}, RequiresStats: true},
	// TFR
	{ID: "tfr_scalpmean", Group: "tfr", Name: "Scalp-Mean", Description: "Scalp-mean TFR plots", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_scalpmean_contrast", Group: "tfr", Name: "Scalp-Mean Contrast", Description: "Pain vs non-pain contrasts", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_channels", Group: "tfr", Name: "Channels", Description: "Channel-level TFR plots", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_channels_contrast", Group: "tfr", Name: "Channels Contrast", Description: "Channel-level contrast plots", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_rois", Group: "tfr", Name: "ROIs", Description: "ROI-level TFR plots", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "tfr_rois_contrast", Group: "tfr", Name: "ROI Contrast", Description: "ROI-level contrast plots", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_topomaps", Group: "tfr", Name: "Topomaps", Description: "Time-frequency topomaps", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "tfr_band_evolution", Group: "tfr", Name: "Band Evolution", Description: "Band evolution over time", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	// ERP
	{ID: "erp_butterfly", Group: "erp", Name: "Butterfly", Description: "Butterfly ERP plots (all channels)", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_roi", Group: "erp", Name: "ROI Waveforms", Description: "ROI-based ERP waveforms", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	{ID: "erp_contrast", Group: "erp", Name: "Contrast", Description: "ERP condition contrasts (Pain vs No-Pain)", RequiredFiles: []string{"epochs/*.fif", "events.tsv"}, RequiresEpochs: true},
	{ID: "erp_topomaps", Group: "erp", Name: "Topomaps", Description: "ERP spatial distributions", RequiredFiles: []string{"epochs/*.fif"}, RequiresEpochs: true},
	// Decoding
	{ID: "decoding_regression_plots", Group: "decoding", Name: "Regression Plots", Description: "LOSO regression diagnostics", RequiredFiles: []string{"decoding/regression/loso_predictions.tsv"}},
	{ID: "decoding_timegen_plots", Group: "decoding", Name: "Time-Generalization", Description: "Time-generalization matrices", RequiredFiles: []string{"decoding/time_generalization/time_generalization_regression.npz"}},
}

var defaultPlotCategories = []FeatureCategory{
	{"features", "Features", "Feature distribution, QC, and descriptive summaries"},
	{"behavior", "Behavior", "EEG-behavior correlations, temporal stats, and summaries"},
	{"tfr", "Time-Frequency", "Time-frequency representations, contrasts, and topomaps"},
	{"erp", "ERP", "Event-related potential waveforms and topographies"},
	{"decoding", "Decoding", "Decoding regression diagnostics and time-generalization"},
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
	plotCategories []FeatureCategory
	plotItems      []PlotItem
	plotSelected   map[int]bool
	plotCursor     int
	plotOffset     int // Scroll offset for plots

	// Plotting output configuration
	plotFormats         []string
	plotFormatSelected  map[string]bool
	plotDpiOptions      []int
	plotDpiIndex        int
	plotSavefigDpiIndex int
	plotSharedColorbar  bool
	plotConfigCursor    int

	// Subject selection
	subjects         []types.SubjectStatus
	subjectSelected  map[string]bool
	subjectCursor    int
	subjectsLoading  bool
	subjectFilter    string
	filteringSubject bool

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

	// Command preview overlay
	showCommandPreview bool

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
	minEpochsForFeatures     int
	exportAllFeatures        bool
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
	featureSummariesEnabled   bool

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
	regressionFeatureSet         int // 0=pain_summaries, 1=all
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
	modelsFeatureSet          int // 0=pain_summaries, 1=all
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
	stabilityFeatureSet     int // 0=pain_summaries, 1=all
	stabilityMethod         int // 0=spearman, 1=pearson
	stabilityOutcome        int // 0=auto, 1=rating, 2=pain_residual
	stabilityGroupColumn    int // 0=auto, 1=run, 2=block
	stabilityPartialTemp    bool
	stabilityMinGroupTrials int
	stabilityMaxFeatures    int
	stabilityAlpha          float64

	// Consistency & influence
	consistencyEnabled           bool
	influenceFeatureSet          int // 0=pain_summaries, 1=all
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
	correlationsFeatureSet         int // 0=pain_summaries, 1=all
	correlationsTargetRating       bool
	correlationsTargetTemperature  bool
	correlationsTargetPainResidual bool

	// Pain sensitivity
	painSensitivityMinTrials  int
	painSensitivityFeatureSet int // 0=pain_summaries, 1=all

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
		{Key: "P", Description: "Preview command"},
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
		bands:               frequencyBands,
		bandSelected:        make(map[int]bool),
		spatialSelected:     make(map[int]bool),
		helpOverlay:         help,
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
		minEpochsForFeatures:      10,
		exportAllFeatures:         false,
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

		trialTableFormat:          0,
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
		featureSummariesEnabled:   true,

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

		regressionFeatureSet:         0,
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

		modelsFeatureSet:          0,
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

		stabilityFeatureSet:     0,
		stabilityMethod:         0,
		stabilityOutcome:        0,
		stabilityGroupColumn:    0,
		stabilityPartialTemp:    true,
		stabilityMinGroupTrials: 8,
		stabilityMaxFeatures:    50,
		stabilityAlpha:          0.05,

		consistencyEnabled:           true,
		influenceFeatureSet:          0,
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

		correlationsFeatureSet:         0,
		correlationsTargetRating:       true,
		correlationsTargetTemperature:  true,
		correlationsTargetPainResidual: true,

		painSensitivityMinTrials:  10,
		painSensitivityFeatureSet: 0,
		reportTopN:                15,
		temporalResolutionMs:      50,
		temporalSmoothMs:          100,
		temporalTimeMinMs:         -200,
		temporalTimeMaxMs:         1000,
		mixedEffectsType:          0,
		mediationMinEffect:        0.05,
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
		decodingNPerm: 0,
		innerSplits:   3,
		skipTimeGen:   false,
		plotSelected:  make(map[int]bool),
		plotFormats:   []string{"png", "svg", "pdf"},
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
		defaultComps := map[string]bool{
			"trial_table":      true,
			"confounds":        true,
			"stability":        true,
			"consistency":      true,
			"influence":        true,
			"report":           true,
			"correlations":     true,
			"pain_sensitivity": true,
			"condition":        true,
			"temporal":         false,
		}
		for i, c := range behaviorComputations {
			m.computationSelected[i] = defaultComps[c.Key]
		}
		// Initialize feature file selection
		m.featureFiles = featureFileOptions
		m.featureFileSelected = make(map[string]bool)
		// Default: select "all" combined features
		m.featureFileSelected["all"] = true

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

	case types.PipelineCombineFeatures:
		m.modeOptions = []string{"combine-features"}
		m.modeDescriptions = []string{"Merge individual features into features_all.tsv"}

		// Filter out "all" from options
		m.featureFiles = []FeatureFile{}
		for _, f := range featureFileOptions {
			if f.Key != "all" {
				m.featureFiles = append(m.featureFiles, f)
			}
		}

		m.featureFileSelected = make(map[string]bool)
		// Default: all checked for aggregating
		for _, f := range m.featureFiles {
			m.featureFileSelected[f.Key] = true
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectFeatureFiles,
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
		// Handle command preview overlay first
		if m.showCommandPreview {
			if msg.String() == "p" || msg.String() == "P" || msg.String() == "esc" {
				m.showCommandPreview = false
			}
			return m, nil
		}

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
			case "enter":
				m.commitTextInput()
				m.editingText = false
				m.textBuffer = ""
				m.editingTextField = textFieldNone
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

		case "p", "P":
			// Toggle command preview overlay
			m.showCommandPreview = !m.showCommandPreview

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
	if m.Pipeline == types.PipelineFeatures && m.useDefaultAdvanced {
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
	m.editingText = true
}

func (m *Model) commitTextInput() {
	m.setTextFieldValue(m.editingTextField, m.textBuffer)
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
	}
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
	optExportAll
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
	optRegressionFeatureSet
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
	optModelsFeatureSet
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
	optStabilityFeatureSet
	optStabilityMethod
	optStabilityOutcome
	optStabilityGroupColumn
	optStabilityPartialTemp
	optStabilityMinGroupTrials
	optStabilityMaxFeatures
	optStabilityAlpha
	// Behavior options - Consistency / Influence
	optConsistencyEnabled
	optInfluenceFeatureSet
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
	optCorrelationsFeatureSet
	optCorrelationsTargetRating
	optCorrelationsTargetTemperature
	optCorrelationsTargetPainResidual
	// Behavior options - Pain sensitivity / temporal
	optPainSensitivityMinTrials
	optPainSensitivityFeatureSet
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

	options = append(options, optFeatGroupStorage)
	if m.featGroupStorageExpanded {
		options = append(options, optExportAll)
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
		options = append(options, optPrepICAMethod, optPrepICAComp, optPrepUseIcalabel, optPrepProbThresh)
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
				optCorrelationsFeatureSet,
				optCorrelationsTargetRating,
				optCorrelationsTargetTemperature,
				optCorrelationsTargetPainResidual,
			)
		}
	}

	// Pain sensitivity section
	if m.isComputationSelected("pain_sensitivity") {
		options = append(options, optBehaviorGroupPainSens)
		if m.behaviorGroupPainSensExpanded {
			options = append(options, optPainSensitivityFeatureSet, optPainSensitivityMinTrials)
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
				optRegressionFeatureSet,
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
				optModelsFeatureSet,
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
				optStabilityFeatureSet,
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
				optInfluenceFeatureSet,
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
	if m.isCategorySelected("TFR") || m.isCategorySelected("Features") {
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
		optRNGSeed,
		optDecodingSkipTimeGen,
	}
}

func (m Model) isCurrentlyEditing(opt optionType) bool {

	if !m.editingNumber {
		return false
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
