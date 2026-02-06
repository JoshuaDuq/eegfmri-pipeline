package app

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/eeg-pipeline/tui/views/wizard"
)

// Persistence and local state helpers.

func findRepoRoot() string {
	exePath, err := os.Executable()
	if err == nil {
		repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exePath))))
		if isValidRepoRoot(repoRoot) {
			return repoRoot
		}
	}

	// Fallback to current working directory
	cwd, err := os.Getwd()
	if err != nil {
		return cwd
	}

	return findRepoRootFromPath(cwd)
}

func findRepoRootFromPath(startPath string) string {
	checkDir := startPath
	for i := 0; i < maxNavDepth; i++ {
		if isValidRepoRoot(checkDir) {
			return checkDir
		}
		checkDir = filepath.Dir(checkDir)
	}
	return startPath
}

func isValidRepoRoot(path string) bool {
	_, err := os.Stat(filepath.Join(path, "eeg_pipeline"))
	return err == nil
}

func (m *Model) isValidPipelineIndex(index int) bool {
	return index >= 0 && index <= maxPipelineIndex
}

func (m *Model) getStatePath() string {
	return filepath.Join(m.repoRoot, "eeg_pipeline", "data", "derivatives", ".tui_state.json")
}

func (m *Model) loadState() {
	path := m.getStatePath()
	data, err := os.ReadFile(path)
	if err != nil {
		// State file doesn't exist or can't be read - use defaults
		return
	}

	var state TUIState
	if err := json.Unmarshal(data, &state); err != nil {
		// Invalid state file - use defaults
		return
	}

	m.persistentState = state
}

func (m *Model) saveState() {
	path := m.getStatePath()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		// Cannot create directory - state won't be saved
		return
	}

	data, err := json.MarshalIndent(m.persistentState, "", "  ")
	if err != nil {
		// Cannot serialize state - state won't be saved
		return
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		// Cannot write state file - state won't be saved
		return
	}
}

// restoreWizardConfig restores bands, ROIs, spatial selection, and pipeline config from persistent state
func (m *Model) restoreWizardConfig() {
	// Restore bands
	if len(m.persistentState.Bands) > 0 {
		bands := make([]wizard.FrequencyBand, len(m.persistentState.Bands))
		for i, b := range m.persistentState.Bands {
			bands[i] = wizard.FrequencyBand{
				Key:    b.Key,
				Name:   b.Name,
				LowHz:  b.LowHz,
				HighHz: b.HighHz,
			}
		}
		m.wizard.SetBands(bands, m.persistentState.BandSelected)
	}

	// Restore ROIs
	if len(m.persistentState.ROIs) > 0 {
		rois := make([]wizard.ROIDefinition, len(m.persistentState.ROIs))
		for i, r := range m.persistentState.ROIs {
			rois[i] = wizard.ROIDefinition{
				Key:      r.Key,
				Name:     r.Name,
				Channels: r.Channels,
			}
		}
		m.wizard.SetROIs(rois, m.persistentState.ROISelected)
	}

	// Restore spatial selection
	if len(m.persistentState.SpatialSelected) > 0 {
		m.wizard.SetSpatialSelected(m.persistentState.SpatialSelected)
	}

	// Restore pipeline-specific advanced config
	if m.persistentState.PipelineConfigs != nil {
		pipelineName := m.selectedPipeline.String()
		if cfg, ok := m.persistentState.PipelineConfigs[pipelineName]; ok {
			m.wizard.ImportConfig(cfg)
		}
	}
}

// saveWizardConfig copies bands, ROIs, spatial selection, and pipeline config to persistent state
func (m *Model) saveWizardConfig() {
	// Save bands
	bands := m.wizard.GetBands()
	m.persistentState.Bands = make([]BandState, len(bands))
	for i, b := range bands {
		m.persistentState.Bands[i] = BandState{
			Key:    b.Key,
			Name:   b.Name,
			LowHz:  b.LowHz,
			HighHz: b.HighHz,
		}
	}
	m.persistentState.BandSelected = m.wizard.GetBandSelected()

	// Save ROIs
	rois := m.wizard.GetROIs()
	m.persistentState.ROIs = make([]ROIState, len(rois))
	for i, r := range rois {
		m.persistentState.ROIs[i] = ROIState{
			Key:      r.Key,
			Name:     r.Name,
			Channels: r.Channels,
		}
	}
	m.persistentState.ROISelected = m.wizard.GetROISelected()

	// Save spatial selection
	m.persistentState.SpatialSelected = m.wizard.GetSpatialSelected()

	// Save pipeline-specific advanced config
	if m.persistentState.PipelineConfigs == nil {
		m.persistentState.PipelineConfigs = make(map[string]map[string]interface{})
	}
	pipelineName := m.selectedPipeline.String()
	m.persistentState.PipelineConfigs[pipelineName] = m.wizard.ExportConfig()
}
