package wizard

import (
	"regexp"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/eeg-pipeline/tui/types"
)

var wizardANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func wizardStripANSI(s string) string {
	return wizardANSIPattern.ReplaceAllString(s, "")
}

func wizardLineIndex(t *testing.T, view, needle string) int {
	t.Helper()

	lines := strings.Split(wizardStripANSI(view), "\n")
	for i, line := range lines {
		if strings.Contains(line, needle) {
			return i
		}
	}
	t.Fatalf("could not find %q in view:\n%s", needle, view)
	return -1
}

func TestUpdate_MouseWheelDownMovesComputationSelection(t *testing.T) {
	m := New(types.PipelineBehavior, "")
	m.CurrentStep = types.StepSelectComputations
	m.computationCursor = 0

	updated, _ := m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelDown})
	got := updated.(Model)

	if got.computationCursor != 1 {
		t.Fatalf("expected wheel down to move computation cursor to 1, got %d", got.computationCursor)
	}
}

func TestNew_DoesNotStartInEditingMode(t *testing.T) {
	m := New(types.PipelineBehavior, "")

	if m.IsEditing() {
		t.Fatalf("expected a fresh wizard model to start outside editing mode")
	}
	if m.editingRangeIdx != noRangeEditing {
		t.Fatalf("expected editingRangeIdx=%d, got %d", noRangeEditing, m.editingRangeIdx)
	}
}

func TestUpdate_MouseClickTogglesComputationSelection(t *testing.T) {
	m := New(types.PipelineBehavior, "")
	m.CurrentStep = types.StepSelectComputations
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "Regression")
	if got := m.matchComputationLine(m.viewLineAt(idx)); got < 0 {
		t.Fatalf("matcher did not recognize regression line:\n%s", wizardStripANSI(m.View()))
	}
	updated, _ := m.handleMouse(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if got.computationCursor != 3 {
		t.Fatalf("expected regression cursor 3, got %d", got.computationCursor)
	}
	if !got.computationSelected[3] {
		t.Fatalf("expected click to toggle regression on")
	}
}

func TestUpdate_MouseClickStartsFilteringEdit(t *testing.T) {
	m := New(types.PipelinePreprocessing, "")
	m.CurrentStep = types.StepPreprocessingFiltering
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "High-Pass Freq")
	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if !got.editingNumber {
		t.Fatalf("expected click to start numeric editing")
	}
	if got.advancedCursor != 1 {
		t.Fatalf("expected high-pass row cursor 1, got %d", got.advancedCursor)
	}
}

func TestUpdate_MouseMotionHighlightsMLScope(t *testing.T) {
	m := New(types.PipelineML, "")
	m.CurrentStep = types.StepSelectSubjects
	m.width = 140
	m.height = 40
	m.mlScope = MLCVScopeGroup

	line := wizardStripANSI(m.viewLineAt(wizardLineIndex(t, m.View(), "Scope:")))
	x := strings.Index(line, "Subject (within)")
	if x < 0 {
		t.Fatalf("could not find subject scope label in:\n%s", line)
	}

	updated, _ := m.Update(tea.MouseMsg{
		X:      x + 1,
		Y:      wizardLineIndex(t, m.View(), "Scope:"),
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionMotion,
	})
	got := updated.(Model)

	if got.mlScope != MLCVScopeSubject {
		t.Fatalf("expected hover to move ML scope to subject, got %v", got.mlScope)
	}
}

func TestUpdate_MouseClickStartsSubjectFiltering(t *testing.T) {
	m := New(types.PipelineBehavior, "")
	m.CurrentStep = types.StepSelectSubjects
	m.width = 140
	m.height = 40
	m.subjectFilter = "abc"

	idx := wizardLineIndex(t, m.View(), "Filter:")
	updated, _ := m.handleMouse(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if !got.filteringSubject {
		t.Fatalf("expected click to enter subject filtering mode")
	}
}

func TestUpdate_MouseMotionHighlightsDefaultAdvancedConfig(t *testing.T) {
	m := New(types.PipelineBehavior, "")
	m.CurrentStep = types.StepAdvancedConfig
	m.width = 140
	m.height = 40
	m.useDefaultAdvanced = true

	idx := wizardLineIndex(t, m.View(), "Configuration:")
	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionMotion,
	})
	got := updated.(Model)

	if got.advancedCursor != 0 {
		t.Fatalf("expected hover to keep default config cursor at 0, got %d", got.advancedCursor)
	}
}

func TestUpdate_MouseClickTogglesPreprocessingStage(t *testing.T) {
	m := New(types.PipelinePreprocessing, "")
	m.CurrentStep = types.StepSelectPreprocessingStages
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "Epochs")
	before := m.prepStageSelected[3]
	updated, _ := m.handleMouse(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if got.prepStageCursor != 3 {
		t.Fatalf("expected epochs cursor 3, got %d", got.prepStageCursor)
	}
	if got.prepStageSelected[3] == before {
		t.Fatalf("expected click to toggle epochs selection")
	}
}

func TestUpdate_MouseClickCyclesICAMethod(t *testing.T) {
	m := New(types.PipelinePreprocessing, "")
	m.CurrentStep = types.StepPreprocessingICA
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "ICA Method")
	updated, _ := m.handleMouse(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if got.prepICAAlgorithm != 1 {
		t.Fatalf("expected ICA method to cycle to 1, got %d", got.prepICAAlgorithm)
	}
}

func TestUpdate_MouseMotionHighlightsICAMethod(t *testing.T) {
	m := New(types.PipelinePreprocessing, "")
	m.CurrentStep = types.StepPreprocessingICA
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "ICA Method")
	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionMotion,
	})
	got := updated.(Model)

	if got.advancedCursor != 1 {
		t.Fatalf("expected hover to move ICA method cursor to 1, got %d", got.advancedCursor)
	}
}

func TestUpdate_MouseMotionHighlightsPreprocessingStage(t *testing.T) {
	m := New(types.PipelinePreprocessing, "")
	m.CurrentStep = types.StepSelectPreprocessingStages
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "Epochs")
	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionMotion,
	})
	got := updated.(Model)

	if got.prepStageCursor != 3 {
		t.Fatalf("expected hover to move epochs cursor to 3, got %d", got.prepStageCursor)
	}
}

func TestUpdate_MouseClickTogglesAdvancedDefaults(t *testing.T) {
	m := New(types.PipelineBehavior, "")
	m.CurrentStep = types.StepAdvancedConfig
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "Use Defaults")
	updated, _ := m.handleMouse(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionPress,
	})
	got := updated.(Model)

	if got.advancedCursor != 0 {
		t.Fatalf("expected advanced cursor 0, got %d", got.advancedCursor)
	}
	if !got.useDefaultAdvanced {
		t.Fatalf("expected click to toggle advanced defaults on")
	}
}

func TestUpdate_MouseMotionHighlightsComputationSelection(t *testing.T) {
	m := New(types.PipelineBehavior, "")
	m.CurrentStep = types.StepSelectComputations
	m.width = 140
	m.height = 40

	idx := wizardLineIndex(t, m.View(), "Regression")
	updated, _ := m.Update(tea.MouseMsg{
		X:      0,
		Y:      idx,
		Button: tea.MouseButtonLeft,
		Action: tea.MouseActionMotion,
	})
	got := updated.(Model)

	if got.computationCursor != 3 {
		t.Fatalf("expected hover to move regression cursor to 3, got %d", got.computationCursor)
	}
}
