package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/eeg-pipeline/tui/views/wizard"
)

var retiredBadROIChannels = map[string]bool{
	"AF4": true,
	"C4":  true,
	"CP1": true,
	"CPZ": true,
	"CZ":  true,
	"F4":  true,
	"FC1": true,
	"FC2": true,
	"FC3": true,
	"FP1": true,
	"FZ":  true,
}

const roiCacheVersionCurrent = 1

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

func sanitizeCachedROIs(rois []ROIState) ([]ROIState, bool) {
	if len(rois) == 0 {
		return rois, false
	}

	changed := false
	sanitized := make([]ROIState, 0, len(rois))
	for _, roi := range rois {
		parts := strings.Split(roi.Channels, ",")
		kept := make([]string, 0, len(parts))
		seen := make(map[string]bool)
		for _, raw := range parts {
			ch := strings.TrimSpace(raw)
			if ch == "" {
				continue
			}
			upper := strings.ToUpper(ch)
			if retiredBadROIChannels[upper] {
				changed = true
				continue
			}
			if seen[upper] {
				changed = true
				continue
			}
			seen[upper] = true
			kept = append(kept, ch)
		}
		if len(kept) == 0 {
			changed = true
			continue
		}
		newChannels := strings.Join(kept, ",")
		if newChannels != roi.Channels {
			changed = true
		}
		roi.Channels = newChannels
		sanitized = append(sanitized, roi)
	}
	return sanitized, changed
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
		if m.persistentState.ROICacheVersion < roiCacheVersionCurrent {
			currentROIs := m.wizard.GetROIs()
			m.persistentState.ROIs = make([]ROIState, len(currentROIs))
			for i, r := range currentROIs {
				m.persistentState.ROIs[i] = ROIState{
					Key:      r.Key,
					Name:     r.Name,
					Channels: r.Channels,
				}
			}
			m.persistentState.ROISelected = m.wizard.GetROISelected()
			m.persistentState.ROICacheVersion = roiCacheVersionCurrent
			m.saveState()
		}
		sanitizedROIs, changed := sanitizeCachedROIs(m.persistentState.ROIs)
		if len(sanitizedROIs) == 0 {
			currentROIs := m.wizard.GetROIs()
			m.persistentState.ROIs = make([]ROIState, len(currentROIs))
			for i, r := range currentROIs {
				m.persistentState.ROIs[i] = ROIState{
					Key:      r.Key,
					Name:     r.Name,
					Channels: r.Channels,
				}
			}
			m.persistentState.ROISelected = m.wizard.GetROISelected()
			m.persistentState.ROICacheVersion = roiCacheVersionCurrent
			m.saveState()
		} else if changed {
			m.persistentState.ROIs = sanitizedROIs
			if len(m.persistentState.ROISelected) > len(sanitizedROIs) {
				m.persistentState.ROISelected = m.persistentState.ROISelected[:len(sanitizedROIs)]
			}
			m.persistentState.ROICacheVersion = roiCacheVersionCurrent
			m.saveState()
		}

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

// restoreWizardConfigValues restores bands and pipeline configuration from
// persistent state without overwriting UI selections. Used after async config
// hydration (ApplyConfigKeys) to re-apply persisted parameter values that
// were overwritten by YAML defaults, while preserving any selection changes
// the user has already made in the current session.
func (m *Model) restoreWizardConfigValues() {
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

	if m.persistentState.PipelineConfigs != nil {
		pipelineName := m.selectedPipeline.String()
		if cfg, ok := m.persistentState.PipelineConfigs[pipelineName]; ok {
			m.wizard.ImportConfigValues(cfg)
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
	m.persistentState.ROICacheVersion = roiCacheVersionCurrent

	// Save spatial selection
	m.persistentState.SpatialSelected = m.wizard.GetSpatialSelected()

	// Save pipeline-specific advanced config
	if m.persistentState.PipelineConfigs == nil {
		m.persistentState.PipelineConfigs = make(map[string]map[string]interface{})
	}
	pipelineName := m.selectedPipeline.String()
	m.persistentState.PipelineConfigs[pipelineName] = m.wizard.ExportConfig()
	m.syncMainMenuSessionData()
}
