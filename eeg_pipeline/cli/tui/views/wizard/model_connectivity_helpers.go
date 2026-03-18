package wizard

import "strings"

func canonicalConnectivityMeasureKey(token string) (string, bool) {
	normalized := strings.ToLower(strings.TrimSpace(token))
	if normalized == "" {
		return "", false
	}

	for _, measure := range connectivityMeasures {
		if normalized == strings.ToLower(measure.Key) || normalized == strings.ToLower(measure.Name) {
			return measure.Key, true
		}
	}

	return "", false
}

func normalizeConnectivityMeasureTokens(tokens []string) []string {
	seen := make(map[string]bool)
	result := make([]string, 0, len(tokens))

	for _, token := range tokens {
		key, ok := canonicalConnectivityMeasureKey(token)
		if !ok || seen[key] {
			continue
		}
		seen[key] = true
		result = append(result, key)
	}

	return result
}

func normalizeConnectivityMeasureSpec(spec string) string {
	return strings.Join(normalizeConnectivityMeasureTokens(strings.Fields(spec)), " ")
}

func (m *Model) setSelectedConnectivityMeasures(tokens []string) {
	for i := range connectivityMeasures {
		m.connectivityMeasures[i] = false
	}

	selected := make(map[string]bool)
	for _, key := range normalizeConnectivityMeasureTokens(tokens) {
		selected[key] = true
	}

	for i, measure := range connectivityMeasures {
		if selected[measure.Key] {
			m.connectivityMeasures[i] = true
		}
	}
}

func connectivityDynamicMeasuresIndex(tokens []string) (int, bool) {
	selected := make(map[string]bool)
	for _, key := range normalizeConnectivityMeasureTokens(tokens) {
		selected[key] = true
	}

	switch {
	case selected["wpli"] && selected["aec"]:
		return 0, true
	case selected["wpli"]:
		return 1, true
	case selected["aec"]:
		return 2, true
	default:
		return 0, false
	}
}
