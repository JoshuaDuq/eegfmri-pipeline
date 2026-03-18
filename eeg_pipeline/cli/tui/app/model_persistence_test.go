package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/mainmenu"
	"github.com/eeg-pipeline/tui/views/wizard"
)

func TestGetStatePath(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{repoRoot: repoRoot}

	got := m.getStatePath()
	want := filepath.Join(repoRoot, "eeg_pipeline", "data", "derivatives", ".tui_state.json")
	if got != want {
		t.Fatalf("getStatePath() = %q, want %q", got, want)
	}
}

func TestLoadState_NoFileKeepsExistingState(t *testing.T) {
	m := Model{
		repoRoot: t.TempDir(),
		persistentState: TUIState{
			LastPipeline: 2,
		},
	}

	m.loadState()
	if m.persistentState.LastPipeline != 2 {
		t.Fatalf("expected LastPipeline to remain unchanged, got %d", m.persistentState.LastPipeline)
	}
}

func TestLoadState_InvalidJSONKeepsExistingState(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		repoRoot: repoRoot,
		persistentState: TUIState{
			LastPipeline: 3,
		},
	}

	statePath := m.getStatePath()
	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(statePath, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	m.loadState()
	if m.persistentState.LastPipeline != 3 {
		t.Fatalf("expected LastPipeline to remain unchanged, got %d", m.persistentState.LastPipeline)
	}
}

func TestLoadState_ValidJSONOverwritesState(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		repoRoot: repoRoot,
		persistentState: TUIState{
			LastPipeline: 1,
		},
	}

	wantState := TUIState{
		LastPipeline:    5,
		ROICacheVersion: 1,
		TimeRanges: []types.TimeRange{
			{Name: "win", Tmin: "-0.2", Tmax: "0.6"},
		},
	}

	statePath := m.getStatePath()
	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	data, err := json.Marshal(wantState)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(statePath, data, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	m.loadState()
	if m.persistentState.LastPipeline != 5 {
		t.Fatalf("LastPipeline = %d, want 5", m.persistentState.LastPipeline)
	}
	if len(m.persistentState.TimeRanges) != 1 || m.persistentState.TimeRanges[0].Name != "win" {
		t.Fatalf("unexpected time ranges: %+v", m.persistentState.TimeRanges)
	}
}

func TestSaveState_WritesFile(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		repoRoot: repoRoot,
		persistentState: TUIState{
			LastPipeline: 4,
			TimeRanges: []types.TimeRange{
				{Name: "baseline", Tmin: "-0.2", Tmax: "0.0"},
			},
		},
	}

	m.saveState()

	statePath := m.getStatePath()
	data, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatalf("expected state file to be written: %v", err)
	}

	var got TUIState
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.LastPipeline != 4 || len(got.TimeRanges) != 1 {
		t.Fatalf("unexpected saved state: %+v", got)
	}
}

func TestRestoreWizardConfigValues_RestoresBandsAndConfigValues(t *testing.T) {
	repoRoot := t.TempDir()

	m := Model{
		repoRoot:         repoRoot,
		selectedPipeline: types.PipelineFmri,
		mainMenu:         mainmenu.New(),
		wizard:           wizard.New(types.PipelineFmri, repoRoot),
		persistentState: TUIState{
			Bands: []BandState{
				{Key: "alpha", Name: "Alpha", LowHz: 8, HighHz: 12},
			},
			BandSelected: []bool{true},
			PipelineConfigs: map[string]map[string]interface{}{
				types.PipelineFmri.String(): {
					"fmriFmriprepImage": "persisted/image:latest",
				},
			},
		},
	}

	m.restoreWizardConfigValues()

	if got := m.wizard.GetBands(); len(got) != 1 || got[0].Key != "alpha" {
		t.Fatalf("expected restored bands, got %+v", got)
	}
	if got := m.wizard.GetBandSelected(); len(got) != 1 || !got[0] {
		t.Fatalf("expected restored band selection, got %+v", got)
	}
	if got := m.wizard.ExportConfig()["fmriFmriprepImage"]; got != "persisted/image:latest" {
		t.Fatalf("expected persisted config value to be restored, got %v", got)
	}
}

func TestSaveWizardConfig_PersistsBandsROIsSpatialAndPipelineConfig(t *testing.T) {
	repoRoot := t.TempDir()

	wiz := wizard.New(types.PipelineFeatures, repoRoot)
	wiz.SetBands([]wizard.FrequencyBand{
		{Key: "delta", Name: "Delta", LowHz: 1, HighHz: 4},
		{Key: "alpha", Name: "Alpha", LowHz: 8, HighHz: 12},
	}, []bool{true, false})
	wiz.SetROIs([]wizard.ROIDefinition{
		{Key: "Occ", Name: "Occipital", Channels: "O1,O2"},
	}, []bool{true})
	wiz.SetSpatialSelected([]bool{true, false, true})

	m := Model{
		repoRoot:         repoRoot,
		mainMenu:         mainmenu.New(),
		wizard:           wiz,
		selectedPipeline: types.PipelineFeatures,
		persistentState: TUIState{
			PipelineConfigs: make(map[string]map[string]interface{}),
		},
	}

	m.saveWizardConfig()

	if len(m.persistentState.Bands) != 2 {
		t.Fatalf("expected 2 persisted bands, got %d", len(m.persistentState.Bands))
	}
	if got := m.persistentState.Bands[1].Key; got != "alpha" {
		t.Fatalf("expected second band key %q, got %q", "alpha", got)
	}
	if got := m.persistentState.BandSelected; len(got) != 2 || !got[0] || got[1] {
		t.Fatalf("unexpected band selection: %+v", got)
	}

	if len(m.persistentState.ROIs) != 1 || m.persistentState.ROIs[0].Key != "Occ" {
		t.Fatalf("unexpected ROI persistence: %+v", m.persistentState.ROIs)
	}
	if len(m.persistentState.ROISelected) != 1 || !m.persistentState.ROISelected[0] {
		t.Fatalf("unexpected ROI selection: %+v", m.persistentState.ROISelected)
	}
	if m.persistentState.ROICacheVersion != roiCacheVersionCurrent {
		t.Fatalf("ROICacheVersion = %d, want %d", m.persistentState.ROICacheVersion, roiCacheVersionCurrent)
	}

	if got := m.persistentState.SpatialSelected; len(got) != 3 || !got[0] || got[1] || !got[2] {
		t.Fatalf("unexpected spatial selection: %+v", got)
	}

	cfg := m.persistentState.PipelineConfigs[types.PipelineFeatures.String()]
	if cfg == nil {
		t.Fatal("expected pipeline config to be persisted")
	}
}

