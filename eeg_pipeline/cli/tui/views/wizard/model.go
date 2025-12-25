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

type FrequencyBand struct {
	Key         string
	Name        string
	Description string
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
)

var defaultPlotItems = []PlotItem{
	// Features
	{"features_power", "features", "Power", "Band power summaries and topomaps", []string{"features_power.tsv"}, false, true, false},
	{"features_connectivity", "features", "Connectivity", "Connectivity heatmaps and networks", []string{"features_connectivity.parquet"}, true, true, false},
	{"features_aperiodic", "features", "Aperiodic", "1/f spectral slope diagnostics", []string{"features_aperiodic.tsv"}, false, true, false},
	{"features_itpc", "features", "ITPC", "Inter-trial phase coherence plots", []string{"features_itpc.tsv", "stats/itpc_data.npz"}, false, true, false},
	{"features_pac", "features", "PAC", "Phase-amplitude coupling plots", []string{"features_pac*.tsv"}, false, true, false},
	{"features_erds", "features", "ERDS", "Event-related desync/sync plots", []string{"features_erds.tsv"}, false, true, false},
	{"features_complexity", "features", "Complexity", "Complexity distributions and condition comparisons", []string{"features_complexity.tsv"}, false, true, false},
	{"features_quality", "features", "Quality", "Feature quality diagnostics and outlier views", []string{"features_quality.tsv"}, false, true, false},
	{"features_spectral", "features", "Spectral", "Spectral peak and edge features", []string{"features_spectral.tsv"}, false, true, false},
	{"features_ratios", "features", "Ratios", "Band power ratios", []string{"features_ratios.tsv"}, false, true, false},
	{"features_asymmetry", "features", "Asymmetry", "Hemispheric asymmetry indices", []string{"features_asymmetry.tsv"}, false, true, false},
	{"features_bursts", "features", "Bursts", "Oscillatory burst dynamics", []string{"features_bursts.tsv"}, false, true, false},
	{"features_erp", "features", "ERP", "ERP visualizations from epochs", []string{"epochs/*.fif"}, true, false, false},
	// Behavior
	{"behavior_psychometrics", "behavior", "Psychometrics", "Rating distributions and psychometrics", []string{"events.tsv"}, false, false, false},
	{"behavior_power_scatter", "behavior", "Power ROI Scatter", "Power vs behavior scatter plots", []string{"features_power.tsv", "epochs/*.fif"}, true, true, false},
	{"behavior_complexity_scatter", "behavior", "Complexity Scatter", "Complexity vs behavior scatter plots", []string{"features_complexity.tsv", "epochs/*.fif"}, true, true, false},
	{"behavior_aperiodic_scatter", "behavior", "Aperiodic Scatter", "Aperiodic vs behavior scatter plots", []string{"features_aperiodic.tsv", "epochs/*.fif"}, true, true, false},
	{"behavior_connectivity_scatter", "behavior", "Connectivity Scatter", "Connectivity vs behavior scatter plots", []string{"features_connectivity.parquet", "epochs/*.fif"}, true, true, false},
	{"behavior_itpc_scatter", "behavior", "ITPC Scatter", "ITPC vs behavior scatter plots", []string{"features_itpc.tsv", "epochs/*.fif"}, true, true, false},
	{"behavior_temporal_topomaps", "behavior", "Temporal Topomaps", "Temporal correlation topomaps", []string{"stats/temporal_correlations_by_pain*.npz"}, false, false, true},
	{"behavior_pain_clusters", "behavior", "Pain Clusters", "Cluster-based temporal contrasts", []string{"stats/pain_nonpain_time_clusters_*.tsv"}, false, false, true},
	{"behavior_dose_response", "behavior", "Dose Response", "Dose-response curves and contrasts", []string{"features_power.tsv", "epochs/*.fif"}, true, true, false},
	{"behavior_mediation", "behavior", "Mediation", "Mediation path diagrams", []string{"stats/mediation.tsv"}, false, false, true},
	{"behavior_top_predictors", "behavior", "Top Predictors", "Top predictors summary", []string{"stats/correlations*.tsv", "stats/corr_stats_all_features_vs_rating*.tsv"}, false, false, true},
	// TFR
	{"tfr_scalpmean", "tfr", "Scalp-Mean", "Scalp-mean TFR plots", []string{"epochs/*.fif"}, true, false, false},
	{"tfr_scalpmean_contrast", "tfr", "Scalp-Mean Contrast", "Pain vs non-pain contrasts", []string{"epochs/*.fif", "events.tsv"}, true, false, false},
	{"tfr_channels", "tfr", "Channels", "Channel-level TFR plots", []string{"epochs/*.fif"}, true, false, false},
	{"tfr_channels_contrast", "tfr", "Channels Contrast", "Channel-level contrast plots", []string{"epochs/*.fif", "events.tsv"}, true, false, false},
	{"tfr_rois", "tfr", "ROIs", "ROI-level TFR plots", []string{"epochs/*.fif"}, true, false, false},
	{"tfr_rois_contrast", "tfr", "ROI Contrast", "ROI-level contrast plots", []string{"epochs/*.fif", "events.tsv"}, true, false, false},
	{"tfr_topomaps", "tfr", "Topomaps", "Time-frequency topomaps", []string{"epochs/*.fif", "events.tsv"}, true, false, false},
	{"tfr_band_evolution", "tfr", "Band Evolution", "Band evolution over time", []string{"epochs/*.fif", "events.tsv"}, true, false, false},
	// ERP
	{"erp_butterfly", "erp", "Butterfly", "Butterfly ERP plots (all channels)", []string{"epochs/*.fif"}, true, false, false},
	{"erp_roi", "erp", "ROI Waveforms", "ROI-based ERP waveforms", []string{"epochs/*.fif"}, true, false, false},
	{"erp_contrast", "erp", "Contrast", "ERP condition contrasts (Pain vs No-Pain)", []string{"epochs/*.fif", "events.tsv"}, true, false, false},
	{"erp_topomaps", "erp", "Topomaps", "ERP spatial distributions", []string{"epochs/*.fif"}, true, false, false},
	// Decoding
	{"decoding_regression_plots", "decoding", "Regression Plots", "LOSO regression diagnostics", []string{"decoding/regression/loso_predictions.tsv"}, false, false, false},
	{"decoding_timegen_plots", "decoding", "Time-Generalization", "Time-generalization matrices", []string{"decoding/time_generalization/time_generalization_regression.npz"}, false, false, false},
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

	// Multi-select expansion state for advanced config
	expandedOption int // -1 = none expanded, 5 = connectivity
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

	// PAC/CFC configuration
	pacPhaseMin  float64 // Min phase frequency (Hz)
	pacPhaseMax  float64 // Max phase frequency (Hz)
	pacAmpMin    float64 // Min amplitude frequency (Hz)
	pacAmpMax    float64 // Max amplitude frequency (Hz)
	pacMethod    int     // 0: mvl, 1: kl, 2: tort, 3: ozkurt
	pacMinEpochs int

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

	// Burst configuration
	burstThresholdZ  float64
	burstMinDuration int // ms

	// Power configuration
	powerBaselineMode int // 0: logratio, 1: mean, 2: ratio, 3: zscore, 4: zlogratio

	// Spectral configuration
	spectralEdgePercentile float64
	// Validation & Generic
	minEpochsForFeatures int
	exportAllFeatures    bool

	// Connectivity configuration
	connOutputLevel  int // 0: full, 1: global_only
	connGraphMetrics bool
	connGraphProp    float64
	connWindowLen    float64
	connWindowStep   float64
	connAECMode      int // 0: orth, 1: none, 2: sym

	// Behavior pipeline advanced config
	correlationMethod  string  // "spearman" or "pearson"
	bootstrapSamples   int     // 0 = disabled, 1000+ recommended
	nPermutations      int     // For cluster tests
	rngSeed            int     // 0 = use project default
	controlTemperature bool    // Include temperature as covariate
	controlTrialOrder  bool    // Include trial order as covariate
	fdrAlpha           float64 // FDR correction threshold
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
}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New(pipeline types.Pipeline, repoRoot string) Model {
	help := components.NewHelpOverlay("Wizard Shortcuts", 50)
	help.AddSection("Navigation", []components.HelpItem{
		{Key: "↑/↓ or j/k", Description: "Move cursor"},
		{Key: "←/→ or h/l", Description: "Switch computation (behavior)"},
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
		useDefaultAdvanced:   true,
		expandedOption:       -1, // No option expanded initially
		connectivityMeasures: make(map[int]bool),
		// PAC/CFC defaults (from config)
		pacPhaseMin:  4.0,
		pacPhaseMax:  8.0,
		pacAmpMin:    30.0,
		pacAmpMax:    80.0,
		pacMethod:    0,
		pacMinEpochs: 2,
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
		// Burst defaults
		burstThresholdZ:  2.0,
		burstMinDuration: 50,
		// Power defaults
		powerBaselineMode: 0,
		// Spectral defaults
		spectralEdgePercentile: 0.95,
		// Connectivity defaults
		connOutputLevel:  0,
		connGraphMetrics: true,
		connGraphProp:    0.1,
		connWindowLen:    1.0,
		connWindowStep:   0.5,
		connAECMode:      0,
		// Validation & Generic
		minEpochsForFeatures: 10,
		exportAllFeatures:    false,
		// Behavior defaults
		correlationMethod:  "spearman",
		bootstrapSamples:   1000,
		nPermutations:      1000,
		rngSeed:            0,
		controlTemperature: true,
		controlTrialOrder:  false,
		fdrAlpha:           0.05,
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
		for i := range behaviorComputations {
			m.computationSelected[i] = true
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
	}

	// Always update plot offset if in that step to ensure it's in sync
	if m.CurrentStep == types.StepSelectPlots {
		m.UpdatePlotOffset()
	}

	return m, nil
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

	// Calculate feature availability based on all subjects initially
	// (all are selected by default)
	m.updateFeatureAvailability()
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
	}
}

///////////////////////////////////////////////////////////////////
// Advanced Options Helpers
///////////////////////////////////////////////////////////////////

type optionType int

const (
	// Feature Pipeline Advanced Options
	optUseDefaults optionType = iota
	optMicrostateStates
	optGroupTemplates
	optFixedTemplates
	optConnectivity
	optPACPhaseRange
	optPACAmpRange
	optPACMethod
	optPACMinEpochs
	optAperiodicRange
	optAperiodicPeakZ
	optAperiodicMinR2
	optAperiodicMinPoints
	optPEOrder
	optPEDelay
	optBurstPercentile
	optERPBaseline
	optBurstThreshold
	optBurstMinDuration
	optPowerBaselineMode
	optSpectralEdge
	optConnOutputLevel
	optConnGraphMetrics
	optConnGraphProp
	optConnWindowLen
	optConnWindowStep
	optConnAECMode
	optMinEpochs
	optExportAll
	// Behavior options - General
	optCorrMethod
	optBootstrap
	optNPerm
	optRNGSeed
	optControlTemp
	optControlOrder
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

// getFeaturesOptions returns the active advanced options for the features pipeline
func (m Model) getFeaturesOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	if m.isCategorySelected("connectivity") {
		options = append(options, optConnectivity, optConnOutputLevel, optConnGraphMetrics, optConnGraphProp, optConnWindowLen, optConnWindowStep, optConnAECMode)
	}

	if m.isCategorySelected("pac") {
		options = append(options, optPACPhaseRange, optPACAmpRange, optPACMethod, optPACMinEpochs)
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optAperiodicRange, optAperiodicPeakZ, optAperiodicMinR2, optAperiodicMinPoints)
	}
	if m.isCategorySelected("complexity") {
		options = append(options, optPEOrder, optPEDelay)
	}
	if m.isCategorySelected("erp") {
		options = append(options, optERPBaseline)
	}
	if m.isCategorySelected("bursts") {
		options = append(options, optBurstThreshold, optBurstMinDuration)
	}
	if m.isCategorySelected("power") {
		options = append(options, optPowerBaselineMode)
	}
	if m.isCategorySelected("spectral") {
		options = append(options, optSpectralEdge)
	}

	options = append(options,
		optExportAll,
		optMinEpochs,
	)

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
	options := []optionType{
		optUseDefaults,
		optCorrMethod,
		optBootstrap,
		optNPerm,
		optRNGSeed,
		optControlTemp,
		optControlOrder,
		optFDRAlpha,
	}

	// Add cluster options if cluster computation is selected
	if m.isComputationSelected("cluster") {
		options = append(options, optClusterThreshold, optClusterMinSize, optClusterTail)
	}

	// Add mediation options if mediation computation is selected
	if m.isComputationSelected("mediation") {
		options = append(options, optMediationBootstrap, optMediationMaxMediators)
	}

	// Add mixed effects options if mixed_effects computation is selected
	if m.isComputationSelected("mixed_effects") {
		options = append(options, optMixedMaxFeatures)
	}

	// Add condition options if condition computation is selected
	if m.isComputationSelected("condition") {
		options = append(options, optConditionEffectThreshold)
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
