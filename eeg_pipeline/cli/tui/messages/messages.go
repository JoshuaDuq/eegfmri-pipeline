package messages

import "time"

// =========================================================================
// Application State Messages
// =========================================================================

// TickMsg is sent periodically for progress and elapsed-time updates.
type TickMsg struct{}

// =========================================================================
// Execution Messages
// =========================================================================

// CommandStartedMsg indicates a command has started execution
type CommandStartedMsg struct {
	Operation     string
	Subjects      []string
	TotalSubjects int
}

// CommandDoneMsg indicates command execution has completed
type CommandDoneMsg struct {
	ExitCode int
	Duration time.Duration
	Success  bool
	Error    error
}

// StreamOutputMsg contains a line of streaming output during execution
type StreamOutputMsg struct {
	Line string
}

// StreamDoneMsg indicates the output stream has closed
type StreamDoneMsg struct{}

// LogCopiedMsg indicates log was copied to clipboard
type LogCopiedMsg struct{}

// =========================================================================
// Subject Discovery Messages
// =========================================================================

// AvailabilityInfo holds availability status with last modified timestamp
type AvailabilityInfo struct {
	Available    bool    `json:"available"`
	LastModified *string `json:"last_modified,omitempty"`
}

// FeatureAvailability holds per-feature, per-band, and per-computation availability with timestamps
type FeatureAvailability struct {
	Features     map[string]AvailabilityInfo `json:"features,omitempty"`
	Bands        map[string]AvailabilityInfo `json:"bands,omitempty"`
	Computations map[string]AvailabilityInfo `json:"computations,omitempty"`
}

// SubjectInfo holds subject processing status
type SubjectInfo struct {
	ID                  string               `json:"id"`
	HasSourceData       bool                 `json:"has_source_data"`
	HasBids             bool                 `json:"has_bids"`
	HasDerivatives      bool                 `json:"has_derivatives"`
	HasEpochs           bool                 `json:"has_epochs"`        // Deprecated: kept for backward compatibility
	HasPreprocessing    bool                 `json:"has_preprocessing"` // Deprecated: kept for backward compatibility
	HasFeatures         bool                 `json:"has_features"`      // Deprecated: kept for backward compatibility
	HasStats            bool                 `json:"has_stats"`         // Deprecated: kept for backward compatibility
	AvailableBands      []string             `json:"available_bands,omitempty"`
	FeatureAvailability *FeatureAvailability `json:"feature_availability,omitempty"`
	EpochMetadata       map[string]float64   `json:"epoch_metadata,omitempty"`
}

// SubjectsLoadedMsg is sent when subject discovery completes
type SubjectsLoadedMsg struct {
	Subjects                 []SubjectInfo
	AvailableWindows         []string
	AvailableWindowsByFeature map[string][]string
	AvailableEventColumns    []string
	AvailableChannels        []string
	UnavailableChannels      []string
	Error                    error
}

// ColumnsDiscoveredMsg is sent when column discovery from events/trial tables completes
type ColumnsDiscoveredMsg struct {
	Columns []string            // Available column names
	Values  map[string][]string // Unique values for each column
	Windows []string             // Available windows (for condition effects discovery)
	Source  string              // "events", "trial_table", or "condition_effects"
	Error   error               // Error if discovery failed
}

// ROIsDiscoveredMsg is sent when ROI discovery from feature parquet files completes
type ROIsDiscoveredMsg struct {
	ROIs  []string // Available ROI names from feature data
	Error error    // Error if discovery failed
}

// PlotterInfo describes a single plotting option exposed by the backend
type PlotterInfo struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Name     string `json:"name"`
}

// PlottersLoadedMsg is sent when plotter discovery completes
type PlottersLoadedMsg struct {
	FeaturePlotters map[string][]PlotterInfo
	Error           error
}

// ConfigSummary holds key configuration values for the TUI
type ConfigSummary struct {
	Task               string `json:"task"`
	BidsRoot           string `json:"bids_root"`
	BidsFmriRoot       string `json:"bids_fmri_root"`
	DerivRoot          string `json:"deriv_root"`
	SourceRoot         string `json:"source_root"`
	PreprocessingNJobs int    `json:"preprocessing_n_jobs"`
}

// ConfigLoadedMsg is sent when config discovery completes
type ConfigLoadedMsg struct {
	Summary ConfigSummary
	Error   error
}

// ConfigKeysLoadedMsg provides specific config values for TUI forms
type ConfigKeysLoadedMsg struct {
	Values map[string]interface{}
	Error  error
}

// TaskUpdatedMsg notifies the app that the active task has changed
type TaskUpdatedMsg struct {
	Task string
}

// SubjectStartedMsg indicates processing of a subject has started
type SubjectStartedMsg struct {
	Subject string
	Current int
	Total   int
}

// SubjectDoneMsg indicates processing of a subject has completed
type SubjectDoneMsg struct {
	Subject string
	Success bool
}

// =========================================================================
// Progress Tracking Messages
// =========================================================================

// StepProgressMsg provides detailed progress for a specific step
type StepProgressMsg struct {
	Subject string
	Step    string
	Current int
	Total   int
	Pct     int
}

// =========================================================================
// Logging Messages
// =========================================================================

// LogMsg contains a log message from command execution
type LogMsg struct {
	Level   string
	Message string
	Subject string
}

// =========================================================================
// Cloud Messages (referenced from cloud package)
// =========================================================================

// Note: Cloud-related messages are defined in the cloud package to avoid
// circular dependencies. These include:
// - VMStatusMsg
// - SyncCompleteMsg
// - RunCompleteMsg
// - PullCompleteMsg

// RefreshSubjectsMsg requests a reload of the subject list
type RefreshSubjectsMsg struct{}

// ResourceUpdateMsg contains real-time CPU and memory usage
type ResourceUpdateMsg struct {
	CPUUsage      float64
	MemoryUsage   float64
	CPUCoreUsages []float64 // Per-core CPU usage percentages
	NumCPUCores   int       // Total number of CPU cores
}

// FmriConditionsDiscoveredMsg contains discovered fMRI trial_type conditions
type FmriConditionsDiscoveredMsg struct {
	Conditions []string
	Subject    string
	Task       string
	Error      error
}

// FmriColumnsDiscoveredMsg contains discovered fMRI event columns and values
type FmriColumnsDiscoveredMsg struct {
	Columns []string
	Values  map[string][]string
	Source  string
	Error   error
}

// MultigroupStatsDiscoveredMsg contains discovered multigroup comparison stats
type MultigroupStatsDiscoveredMsg struct {
	Available    bool     // Whether multigroup stats are available
	Groups       []string // Group labels from precomputed stats
	NFeatures    int      // Number of features with stats
	NSignificant int      // Number of FDR-significant comparisons
	File         string   // Path to stats file
	Error        error    // Error if discovery failed
}
