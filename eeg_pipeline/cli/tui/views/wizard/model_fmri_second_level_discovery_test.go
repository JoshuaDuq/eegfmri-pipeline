package wizard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestEnsureFmriSecondLevelContrastDiscovery_FindsSharedContrasts(t *testing.T) {
	root := t.TempDir()
	writeSecondLevelFirstLevelOutput(t, root, "01", "pain", "pain")
	writeSecondLevelFirstLevelOutput(t, root, "01", "pain", "cue")
	writeSecondLevelFirstLevelOutput(t, root, "02", "pain", "pain")

	m := New(types.PipelineFmriAnalysis, ".")
	m.task = "pain"
	m.derivRoot = root
	m.subjects = []types.SubjectStatus{{ID: "01"}, {ID: "02"}}
	m.subjectSelected = map[string]bool{"01": true, "02": true}

	if err := m.ensureFmriSecondLevelContrastDiscovery(); err != nil {
		t.Fatalf("expected shared contrast discovery to succeed, got %v", err)
	}

	got := m.GetFmriSecondLevelDiscoveredContrastNames()
	want := []string{"pain"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected discovered contrasts: got=%#v want=%#v", got, want)
	}
}

func TestEnsureFmriSecondLevelCovariatesDiscovery_ReadsColumnsAndValues(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "groups.tsv")
	content := "participant_id\tgroup\tage\nsub-01\tcontrol\t20\nsub-02\tpatient\t30\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write groups.tsv: %v", err)
	}

	m := New(types.PipelineFmriAnalysis, ".")
	m.fmriSecondLevelCovariatesFile = path

	if err := m.ensureFmriSecondLevelCovariatesDiscovery(); err != nil {
		t.Fatalf("expected covariates discovery to succeed, got %v", err)
	}

	columns, values, errText := m.currentFmriSecondLevelCovariatesDiscovery()
	if errText != "" {
		t.Fatalf("unexpected covariates discovery error: %s", errText)
	}
	wantColumns := []string{"participant_id", "group", "age"}
	if !reflect.DeepEqual(columns, wantColumns) {
		t.Fatalf("unexpected covariates columns: got=%#v want=%#v", columns, wantColumns)
	}
	wantGroups := []string{"control", "patient"}
	if got := values["group"]; !reflect.DeepEqual(got, wantGroups) {
		t.Fatalf("unexpected group values: got=%#v want=%#v", got, wantGroups)
	}
}

func TestToggleFmriSecondLevelContrastNames_OpensExpandedList(t *testing.T) {
	root := t.TempDir()
	writeSecondLevelFirstLevelOutput(t, root, "01", "pain", "pain")
	writeSecondLevelFirstLevelOutput(t, root, "02", "pain", "pain")

	m := New(types.PipelineFmriAnalysis, ".")
	m.modeIndex = 1
	m.task = "pain"
	m.derivRoot = root
	m.subjects = []types.SubjectStatus{{ID: "01"}, {ID: "02"}}
	m.subjectSelected = map[string]bool{"01": true, "02": true}
	m.advancedCursor = findOptionIndex(t, m.getFmriAnalysisOptions(), optFmriSecondLevelContrastNames)

	m.toggleFmriAnalysisAdvancedOption()

	if m.expandedOption != expandedFmriSecondLevelContrastNames {
		t.Fatalf("expected contrast selector to open, got expanded option %d", m.expandedOption)
	}
}

func TestToggleFmriSecondLevelGroupColumn_OpensExpandedList(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "groups.tsv")
	content := "participant_id\tgroup\tage\nsub-01\tcontrol\t20\nsub-02\tpatient\t30\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write groups.tsv: %v", err)
	}

	m := New(types.PipelineFmriAnalysis, ".")
	m.modeIndex = 1
	m.fmriSecondLevelModelIndex = 1
	m.fmriSecondLevelCovariatesFile = path
	m.advancedCursor = findOptionIndex(t, m.getFmriAnalysisOptions(), optFmriSecondLevelGroupColumn)

	m.toggleFmriAnalysisAdvancedOption()

	if m.expandedOption != expandedFmriSecondLevelGroupColumn {
		t.Fatalf("expected group column selector to open, got expanded option %d", m.expandedOption)
	}
}

func writeSecondLevelFirstLevelOutput(t *testing.T, root string, subject string, task string, contrastName string) {
	t.Helper()

	subjectLabel := normalizeFmriSecondLevelSubjectLabel(subject)
	dir := filepath.Join(
		root,
		subjectLabel,
		"fmri",
		"first_level",
		"task-"+task,
		"contrast-"+contrastName,
	)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}

	base := filepath.Join(dir, subjectLabel+"_task-"+task+"_contrast-"+contrastName+"_stat-effect_size_test")
	if err := os.WriteFile(base+".nii.gz", []byte("nifti"), 0o644); err != nil {
		t.Fatalf("write nifti: %v", err)
	}

	payload := map[string]any{
		"subject":            subjectLabel,
		"task":               task,
		"contrast_name":      contrastName,
		"output_type_actual": fmriSecondLevelRequiredOutputType,
		"contrast_cfg": map[string]any{
			"fmriprep_space": fmriSecondLevelRequiredSpace,
		},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal sidecar: %v", err)
	}
	if err := os.WriteFile(base+".json", encoded, 0o644); err != nil {
		t.Fatalf("write sidecar: %v", err)
	}
}

func findOptionIndex(t *testing.T, options []optionType, want optionType) int {
	t.Helper()
	for index, option := range options {
		if option == want {
			return index
		}
	}
	t.Fatalf("option %v not found in %#v", want, options)
	return -1
}
