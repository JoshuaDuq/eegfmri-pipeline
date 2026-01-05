package wizard

import (
	"strings"
)

func (m *Model) ShowToast(message, toastType string) {
	m.toastMessage = message
	m.toastType = toastType
	m.toastTicker = 30 // Show for 3 seconds (30 ticks at 100ms)
}

func (m *Model) TickToast() {
	if m.toastTicker > 0 {
		m.toastTicker--
		if m.toastTicker == 0 {
			m.toastMessage = ""
			m.toastType = ""
		}
	}
}

func (m *Model) ApplyFeaturePreset(presetKey string) {
	m.selected = make(map[int]bool)

	var categoriesToSelect []string

	switch presetKey {
	case "quick":
		categoriesToSelect = []string{"power", "aperiodic", "complexity"}
	case "full":
		for i := range m.categories {
			m.selected[i] = true
		}
		m.activePreset = "Full Analysis"
		m.ShowToast("Applied: Full Analysis preset", "success")
		return
	case "connectivity":
		categoriesToSelect = []string{"power", "connectivity", "pac"}
	case "spectral":
		categoriesToSelect = []string{"power", "spectral", "aperiodic", "ratios"}
	default:
		return
	}

	for i, cat := range m.categories {
		for _, target := range categoriesToSelect {
			if strings.ToLower(cat) == target {
				m.selected[i] = true
				break
			}
		}
	}

	presetNames := map[string]string{
		"quick":        "Quick Run",
		"connectivity": "Connectivity Focus",
		"spectral":     "Spectral Focus",
	}

	m.activePreset = presetNames[presetKey]
	m.ShowToast("Applied: "+m.activePreset+" preset", "success")
}

func (m *Model) ApplyBehaviorPreset(presetKey string) {
	m.computationSelected = make(map[int]bool)
	m.postComputationSelected = make(map[int]bool)

	var computationsToSelect []string

	switch presetKey {
	case "quick":
		computationsToSelect = []string{"correlations", "report"}
	case "full":
		// Select all in both lists
		for i := range m.computations {
			m.computationSelected[i] = true
		}
		for i := range m.postComputations {
			m.postComputationSelected[i] = true
		}
		m.activePreset = "Full Analysis"
		m.ShowToast("Applied: Full Analysis preset", "success")
		return
	case "regression":
		computationsToSelect = []string{"regression", "models", "stability", "influence"}
	case "temporal":
		computationsToSelect = []string{"temporal", "cluster", "mediation"}
	default:
		return
	}

	// Match against primary computations
	for i, comp := range m.computations {
		for _, target := range computationsToSelect {
			if comp.Key == target {
				m.computationSelected[i] = true
				break
			}
		}
	}

	// Match against post computations
	for i, comp := range m.postComputations {
		for _, target := range computationsToSelect {
			if comp.Key == target {
				m.postComputationSelected[i] = true
				break
			}
		}
	}

	presetNames := map[string]string{
		"quick":      "Quick Analysis",
		"regression": "Regression Focus",
		"temporal":   "Temporal Focus",
	}

	m.activePreset = presetNames[presetKey]
	m.ShowToast("Applied: "+m.activePreset+" preset", "success")
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
	for _, s := range m.subjects {
		if m.subjectFilter != "" && !strings.Contains(strings.ToLower(s.ID), strings.ToLower(m.subjectFilter)) {
			continue
		}

		subjectValid, _ := m.Pipeline.ValidateSubject(s)
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
