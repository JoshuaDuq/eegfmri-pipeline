package wizard

import (
	"fmt"
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
	{"export", "Export Results", "Export analysis results"},
}

type FrequencyBand struct {
	Key         string
	Name        string
	Description string
}

var frequencyBands = []FrequencyBand{
	{"delta", "Delta", "1.0-3.9 Hz"},
	{"theta", "Theta", "4.0-7.9 Hz"},
	{"alpha", "Alpha", "8.0-12.9 Hz"},
	{"beta", "Beta", "13.0-30.0 Hz"},
	{"gamma", "Gamma", "30.1-80.0 Hz"},
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
	{"temporal", "Temporal", "Time-resolved (binned) features"},
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
}

var plotItems = []PlotItem{
	// Features
	{"features_power", "Features", "Power", "Band power summaries and topomaps", []string{"features_power.tsv"}, false, true},
	{"features_connectivity", "Features", "Connectivity", "Connectivity heatmaps and network views", []string{"features_connectivity.parquet"}, false, true},
	{"features_aperiodic", "Features", "Aperiodic", "1/f spectral slope diagnostics", []string{"features_aperiodic.tsv"}, false, true},
	{"features_itpc", "Features", "ITPC", "Inter-trial phase coherence plots", []string{"features_itpc.tsv"}, false, true},
	{"features_pac", "Features", "PAC", "Phase-amplitude coupling plots", []string{"features_pac_trials.tsv"}, false, true},
	{"features_erds", "Features", "ERDS", "Event-related desync/sync plots", []string{"features_erds.tsv"}, false, true},
	// Behavior
	{"behavior_psychometrics", "Behavior", "Psychometrics", "Rating distributions and psychometrics", []string{"events.tsv", "features_power.tsv"}, false, true},
	{"behavior_power_scatter", "Behavior", "Power ROI Scatter", "Power vs behavior scatter plots", []string{"features_power.tsv", "stats/corr_stats_power_*"}, false, true},
	{"behavior_complexity_scatter", "Behavior", "Complexity Scatter", "Complexity vs behavior scatter plots", []string{"features_complexity.tsv"}, false, true},
	{"behavior_aperiodic_scatter", "Behavior", "Aperiodic Scatter", "Aperiodic vs behavior scatter plots", []string{"features_aperiodic.tsv"}, false, true},
	{"behavior_connectivity_scatter", "Behavior", "Connectivity Scatter", "Connectivity vs behavior scatter plots", []string{"features_connectivity.parquet"}, false, true},
	{"behavior_itpc_scatter", "Behavior", "ITPC Scatter", "ITPC vs behavior scatter plots", []string{"features_itpc.tsv"}, false, true},
	{"behavior_temporal_topomaps", "Behavior", "Temporal Topomaps", "Temporal correlation topomaps", []string{"stats/temporal_*"}, false, true},
	{"behavior_pain_clusters", "Behavior", "Pain Clusters", "Cluster-based temporal contrasts", []string{"stats/cluster_*"}, false, true},
	{"behavior_dose_response", "Behavior", "Dose-Response", "Induced dose-response curves", []string{"stats/correlations.tsv", "events.tsv"}, false, true},
	{"behavior_mediation", "Behavior", "Mediation", "Mediation path diagrams", []string{"stats/mediation.tsv"}, false, true},
	{"behavior_top_predictors", "Behavior", "Top Predictors", "Summary of top behavioral predictors", []string{"stats/correlations.tsv"}, false, true},
	// TFR
	{"tfr_scalpmean", "TFR", "Scalp-Mean", "Scalp-mean TFR plots", []string{"epochs/*.fif"}, true, false},
	{"tfr_scalpmean_contrast", "TFR", "Scalp-Mean Contrast", "Pain vs non-pain contrasts", []string{"epochs/*.fif", "events.tsv"}, true, false},
	{"tfr_channels", "TFR", "Channels", "Channel-level TFR plots", []string{"epochs/*.fif"}, true, false},
	{"tfr_channels_contrast", "TFR", "Channels Contrast", "Channel-level contrast plots", []string{"epochs/*.fif", "events.tsv"}, true, false},
	{"tfr_rois", "TFR", "ROIs", "ROI-level TFR plots", []string{"epochs/*.fif"}, true, false},
	{"tfr_rois_contrast", "TFR", "ROI Contrast", "ROI-level contrast plots", []string{"epochs/*.fif", "events.tsv"}, true, false},
	{"tfr_topomaps", "TFR", "Topomaps", "Time-frequency topomaps", []string{"epochs/*.fif", "events.tsv"}, true, false},
	{"tfr_band_evolution", "TFR", "Band Evolution", "Band evolution over time", []string{"epochs/*.fif"}, true, false},
	// ERP
	{"erp_butterfly", "ERP", "Butterfly", "Butterfly ERP plots (all channels)", []string{"epochs/*.fif"}, true, false},
	{"erp_roi", "ERP", "ROI Waveforms", "ROI-based ERP waveforms", []string{"epochs/*.fif"}, true, false},
	{"erp_contrast", "ERP", "Contrast", "ERP condition contrasts (Pain vs No-Pain)", []string{"epochs/*.fif", "events.tsv"}, true, false},
	{"erp_topomaps", "ERP", "Topomaps", "ERP spatial distributions", []string{"epochs/*.fif"}, true, false},
	// Decoding
	{"decoding_regression_plots", "Decoding", "Regression Plots", "LOSO regression diagnostics", []string{"decoding/regression/loso_predictions.tsv"}, false, true},
	{"decoding_timegen_plots", "Decoding", "Time-Generalization", "Time-generalization matrices", []string{"decoding/time_generalization/time_generalization_regression.npz"}, false, true},
}

var plotCategories = []FeatureCategory{
	{"ERP", "Event-Related Potentials", "Waveforms, butterfly plots, and topo contrasts"},
	{"TFR", "Time-Frequency", "Time-frequency representations and topomaps"},
	{"Behavior", "EEG-Behavior", "Correlation scatter plots and psychometrics"},
	{"Decoding", "Decoding", "Regression diagnostics and time-generalization"},
	{"Features", "Feature Overviews", "General feature distribution and QC plots"},
}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	Pipeline    types.Pipeline
	CurrentStep types.WizardStep
	steps       []types.WizardStep
	stepIndex   int

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
	plotItems    []PlotItem
	plotSelected map[int]bool
	plotCursor   int
	plotOffset   int // Scroll offset for plots

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

	// Features pipeline advanced config
	connectivityMeasures map[int]bool // Selected connectivity measures

	// PAC/CFC configuration
	pacPhaseMin float64 // Min phase frequency (Hz)
	pacPhaseMax float64 // Max phase frequency (Hz)
	pacAmpMin   float64 // Min amplitude frequency (Hz)
	pacAmpMax   float64 // Max amplitude frequency (Hz)

	// Aperiodic configuration
	aperiodicFmin float64 // Min frequency for aperiodic fit
	aperiodicFmax float64 // Max frequency for aperiodic fit

	// Complexity configuration
	complexityPEOrder int // Permutation entropy order (3-7)

	// ERP configuration
	erpBaselineCorrection bool

	// Burst configuration
	burstThresholdZ float64

	// Power configuration
	powerBaselineMode int // 0: logratio, 1: mean, 2: ratio, 3: zscore, 4: zlogratio

	// Spectral configuration
	spectralEdgePercentile float64

	// Connectivity configuration
	connOutputLevel  int // 0: full, 1: global_only
	connGraphMetrics bool
	connAECMode      int // 0: orth, 1: none, 2: sym

	// Behavior pipeline advanced config
	correlationMethod  string  // "spearman" or "pearson"
	bootstrapSamples   int     // 0 = disabled, 1000+ recommended
	nPermutations      int     // For cluster tests
	rngSeed            int     // 0 = use project default
	controlTemperature bool    // Include temperature as covariate
	controlTrialOrder  bool    // Include trial order as covariate
	fdrAlpha           float64 // FDR correction threshold

	// Decoding pipeline advanced config
	decodingNPerm int  // Permutations for significance test
	innerSplits   int  // CV inner splits
	skipTimeGen   bool // Skip time generalization
}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New(pipeline types.Pipeline) Model {
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
		pacPhaseMin: 4.0,
		pacPhaseMax: 8.0,
		pacAmpMin:   30.0,
		pacAmpMax:   80.0,
		// Aperiodic defaults
		aperiodicFmin: 2.0,
		aperiodicFmax: 40.0,
		// Complexity defaults
		complexityPEOrder: 3,
		// ERP defaults
		erpBaselineCorrection: true,
		// Burst defaults
		burstThresholdZ: 2.0,
		// Power defaults
		powerBaselineMode: 0,
		// Spectral defaults
		spectralEdgePercentile: 0.95,
		// Connectivity defaults
		connOutputLevel:  0,
		connGraphMetrics: true,
		connAECMode:      0,
		// Behavior defaults
		correlationMethod:  "spearman",
		bootstrapSamples:   1000,
		nPermutations:      1000,
		rngSeed:            0,
		controlTemperature: true,
		controlTrialOrder:  false,
		fdrAlpha:           0.05,
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
	}

	// Time ranges
	m.TimeRanges = []types.TimeRange{
		{Name: "baseline", Tmin: "", Tmax: ""},
		{Name: "active", Tmin: "", Tmax: ""},
	}
	m.editingRangeIdx = -1

	switch pipeline {
	case types.PipelineFeatures:
		m.modeOptions = []string{styles.ModeCompute, styles.ModeVisualize}
		m.modeDescriptions = []string{
			"Extract features from epochs",
			"Generate visualizations",
		}
		m.categories = []string{
			"power", "spectral", "aperiodic", "erp", "erds", "ratios", "asymmetry",
			"connectivity", "itpc", "pac",
			"complexity", "bursts", "quality", "temporal",
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
			"Time-resolved (binned) features",
		}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,   // Moved up - subject selection first to assess data availability
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
		m.modeOptions = []string{styles.ModeCompute, styles.ModeVisualize}
		m.modeDescriptions = []string{
			"Compute EEG-behavior correlations",
			"Generate correlation plots",
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
			types.StepAdvancedConfig,
			types.StepSelectSubjects,
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
			types.StepReviewExecute,
		}

	case types.PipelineRawToBIDS:
		m.modeOptions = []string{"raw-to-bids"}
		m.modeDescriptions = []string{"Convert raw EEG data to BIDS format"}
		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepReviewExecute,
		}

	case types.PipelinePlotting:
		m.modeOptions = []string{styles.ModeVisualize}
		m.modeDescriptions = []string{"Generate selected visualization suites"}
		m.plotItems = plotItems
		for i := range m.plotItems {
			m.plotSelected[i] = true
		}
		m.plotSharedColorbar = true

		// Initialize categories for plotting
		m.categories = make([]string, len(plotCategories))
		m.categoryDescs = make([]string, len(plotCategories))
		for i, cat := range plotCategories {
			m.categories[i] = cat.Name
			m.categoryDescs[i] = cat.Description
			m.selected[i] = true // All selected by default
		}

		m.steps = []types.WizardStep{
			types.StepSelectSubjects,
			types.StepSelectPlotCategories,
			types.StepSelectPlots,
			types.StepPlotConfig,
			types.StepReviewExecute, // Review & Execute should be the last step
		}

	default:
		m.modeOptions = []string{styles.ModeCompute, styles.ModeVisualize}
		m.modeDescriptions = []string{"Run computation", "Generate visualizations"}
		m.steps = []types.WizardStep{types.StepSelectMode, types.StepSelectSubjects, types.StepReviewExecute}
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
				// Only accept digits
				if len(msg.String()) == 1 && msg.String() >= "0" && msg.String() <= "9" {
					m.numberBuffer += msg.String()
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
	optAperiodicRange
	optPEOrder
	optBurstPercentile
	optERPBaseline
	optBurstThreshold
	optPowerBaselineMode
	optSpectralEdge
	optConnOutputLevel
	optConnGraphMetrics
	optConnAECMode
	// Behavior options
	optCorrMethod
	optBootstrap
	optNPerm
	optRNGSeed
	optControlTemp
	optControlOrder
	optFDRAlpha
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
)

// getFeaturesOptions returns the active advanced options for the features pipeline
func (m Model) getFeaturesOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	if m.isCategorySelected("connectivity") {
		options = append(options, optConnectivity, optConnOutputLevel, optConnGraphMetrics, optConnAECMode)
	}

	if m.isCategorySelected("pac") {
		options = append(options, optPACPhaseRange, optPACAmpRange)
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optAperiodicRange)
	}
	if m.isCategorySelected("complexity") {
		options = append(options, optPEOrder)
	}
	if m.isCategorySelected("erp") {
		options = append(options, optERPBaseline)
	}
	if m.isCategorySelected("bursts") {
		options = append(options, optBurstThreshold)
	}
	if m.isCategorySelected("power") {
		options = append(options, optPowerBaselineMode)
	}
	if m.isCategorySelected("spectral") {
		options = append(options, optSpectralEdge)
	}

	return options
}

func (m Model) getBehaviorOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optCorrMethod,
		optBootstrap,
		optNPerm,
		optRNGSeed,
		optControlTemp,
		optControlOrder,
		optFDRAlpha,
	}
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
	default:
		return false
	}
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return false
	}
	return options[m.advancedCursor] == opt
}
