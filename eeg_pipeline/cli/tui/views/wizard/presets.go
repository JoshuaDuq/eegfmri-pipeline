package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/types"
)

const (
	toastDurationTicks = 30 // 3 seconds at 100ms per tick
)

func (m *Model) ShowToast(message, toastType string) {
	m.toastMessage = message
	m.toastType = toastType
	m.toastTicker = toastDurationTicks
}

func (m *Model) TickToast() {
	if m.toastTicker == 0 {
		return
	}
	m.toastTicker--
	if m.toastTicker == 0 {
		m.toastMessage = ""
		m.toastType = ""
	}
}

func (m *Model) ApplyFeaturePreset(presetKey string) {
	// Feature presets have been removed
}

func (m *Model) ApplyBehaviorPreset(presetKey string) {
	m.computationSelected = make(map[int]bool)

	if presetKey == "full" {
		m.selectAllBehaviorComputations()
		m.notifyPresetApplied("Full Analysis")
		return
	}

	computationsToSelect := m.getBehaviorPresetComputations(presetKey)
	if computationsToSelect == nil {
		return
	}

	m.selectComputationsByKeys(computationsToSelect)
	presetName := m.getBehaviorPresetName(presetKey)
	m.notifyPresetApplied(presetName)
}

func (m *Model) ClearPreset() {
	m.activePreset = ""
}

func (m *Model) ToggleShowOnlyValid() {
	m.showOnlyValid = !m.showOnlyValid
	m.subjectCursor = 0
	m.subjectScrollTop = 0
}

func (m Model) GetFilteredSubjectCount() (filtered, total, valid int) {
	total = len(m.subjects)
	for _, subject := range m.subjects {
		if m.isSubjectFilteredOut(subject) {
			continue
		}

		subjectValid, _ := m.Pipeline.ValidateSubject(subject)
		if subjectValid {
			valid++
		}

		if m.showOnlyValid && !subjectValid {
			continue
		}

		filtered++
	}
	return filtered, total, valid
}

func (m *Model) getBehaviorPresetComputations(presetKey string) []string {
	switch presetKey {
	case "quick":
		return []string{"correlations", "report"}
	case "regression":
		return []string{"regression", "models", "stability", "influence"}
	case "temporal":
		return []string{"temporal", "cluster", "mediation"}
	case "full":
		return []string{}
	default:
		return nil
	}
}

func (m *Model) selectAllBehaviorComputations() {
	for i := range m.computations {
		m.computationSelected[i] = true
	}
}

func (m *Model) selectComputationsByKeys(targetKeys []string) {
	for i, computation := range m.computations {
		for _, targetKey := range targetKeys {
			if computation.Key == targetKey {
				m.computationSelected[i] = true
				break
			}
		}
	}
}

func (m *Model) getBehaviorPresetName(presetKey string) string {
	presetNames := map[string]string{
		"quick":      "Quick Analysis",
		"regression": "Regression Focus",
		"temporal":   "Temporal Focus",
	}
	return presetNames[presetKey]
}

func (m *Model) notifyPresetApplied(presetName string) {
	m.activePreset = presetName
	m.ShowToast("Applied: "+presetName+" preset", "success")
}

func (m *Model) isSubjectFilteredOut(subject types.SubjectStatus) bool {
	if m.subjectFilter == "" {
		return false
	}
	filterLower := strings.ToLower(m.subjectFilter)
	subjectIDLower := strings.ToLower(subject.ID)
	return !strings.Contains(subjectIDLower, filterLower)
}
