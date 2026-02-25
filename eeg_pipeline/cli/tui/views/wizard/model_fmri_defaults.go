package wizard

import "strings"

func normalizeFmriColumnName(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func (m Model) fmriDiscoveredColumnByName(name string) string {
	target := normalizeFmriColumnName(name)
	if target == "" {
		return ""
	}
	for _, col := range m.fmriDiscoveredColumns {
		clean := strings.TrimSpace(col)
		if clean == "" {
			continue
		}
		if normalizeFmriColumnName(clean) == target {
			return clean
		}
	}
	return ""
}

func (m Model) resolveFmriColumnFromDiscovered(candidates []string, fallback string) string {
	for _, candidate := range candidates {
		if found := m.fmriDiscoveredColumnByName(candidate); found != "" {
			return found
		}
	}
	if found := m.fmriDiscoveredColumnByName(fallback); found != "" {
		return found
	}
	if len(m.fmriDiscoveredColumns) > 0 {
		first := strings.TrimSpace(m.fmriDiscoveredColumns[0])
		if first != "" {
			return first
		}
	}
	if strings.TrimSpace(fallback) != "" {
		return strings.TrimSpace(fallback)
	}
	for _, candidate := range candidates {
		clean := strings.TrimSpace(candidate)
		if clean != "" {
			return clean
		}
	}
	return ""
}

func (m Model) fmriConditionColumnCandidates() []string {
	candidates := splitCSVList(m.eventColCondition)
	if len(candidates) == 0 {
		candidates = []string{"condition", "trial_type"}
	}
	return appendUniqueStrings(candidates, "condition", "trial_type")
}

func (m Model) fmriDefaultConditionColumn() string {
	return m.resolveFmriColumnFromDiscovered(m.fmriConditionColumnCandidates(), "trial_type")
}

func (m Model) fmriDefaultPhaseColumn() string {
	return m.resolveFmriColumnFromDiscovered([]string{"stim_phase", "phase"}, "stim_phase")
}

func (m Model) resolveFmriConditionColumn(current string) string {
	clean := strings.TrimSpace(current)
	if clean != "" {
		if found := m.fmriDiscoveredColumnByName(clean); found != "" {
			return found
		}
		return clean
	}
	return m.fmriDefaultConditionColumn()
}

func (m Model) resolveFmriPhaseColumn(current string) string {
	clean := strings.TrimSpace(current)
	if clean != "" {
		if found := m.fmriDiscoveredColumnByName(clean); found != "" {
			return found
		}
		return clean
	}
	return m.fmriDefaultPhaseColumn()
}

func (m Model) FmriDiscoveryConditionColumn() string {
	return m.fmriDefaultConditionColumn()
}
