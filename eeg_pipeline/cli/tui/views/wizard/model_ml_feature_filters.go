package wizard

import "strings"

var mlFeatureScopes = []string{"global", "roi", "ch", "chpair"}

var mlFeatureStats = []string{
	"mean",
	"median",
	"std",
	"var",
	"max",
	"min",
	"wpli",
	"imcoh",
	"aec",
	"plv",
	"pli",
	"psi",
	"dtf",
	"pdc",
	"itpc",
	"mi",
	"sampen",
	"perm_entropy",
}

func appendUniqueStrings(base []string, values ...string) []string {
	seen := make(map[string]struct{}, len(base))
	for _, item := range base {
		seen[item] = struct{}{}
	}
	for _, raw := range values {
		value := strings.TrimSpace(raw)
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		base = append(base, value)
		seen[value] = struct{}{}
	}
	return base
}

func (m Model) mlFeatureFamiliesOptions() []string {
	options := []string{"(config default)"}
	useAvailability := len(m.featureAvailability) > 0
	for _, feature := range featureFileOptions {
		if useAvailability && !m.featureAvailability[feature.Key] {
			continue
		}
		options = appendUniqueStrings(options, feature.Key)
	}
	return appendUniqueStrings(options, splitLooseList(m.mlFeatureFamiliesSpec)...)
}

func (m Model) mlFeatureBandsOptions() []string {
	options := []string{"(none)"}
	if len(m.bands) > 0 {
		for _, band := range m.bands {
			options = appendUniqueStrings(options, band.Key)
		}
	} else {
		for _, band := range frequencyBands {
			options = appendUniqueStrings(options, band.Key)
		}
	}
	return appendUniqueStrings(options, splitLooseList(m.mlFeatureBandsSpec)...)
}

func (m Model) mlFeatureSegmentsOptions() []string {
	options := []string{"(none)"}
	options = appendUniqueStrings(options, m.availableWindows...)
	for _, window := range m.TimeRanges {
		options = appendUniqueStrings(options, window.Name)
	}
	return appendUniqueStrings(options, splitLooseList(m.mlFeatureSegmentsSpec)...)
}

func (m Model) mlFeatureScopesOptions() []string {
	options := []string{"(none)"}
	options = appendUniqueStrings(options, mlFeatureScopes...)
	return appendUniqueStrings(options, splitLooseList(m.mlFeatureScopesSpec)...)
}

func (m Model) mlFeatureStatsOptions() []string {
	options := []string{"(none)"}
	options = appendUniqueStrings(options, mlFeatureStats...)
	return appendUniqueStrings(options, splitLooseList(m.mlFeatureStatsSpec)...)
}
