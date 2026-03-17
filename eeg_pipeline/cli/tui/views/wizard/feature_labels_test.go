package wizard

import "testing"

func TestFeatureCategoryLabel(t *testing.T) {
	tests := map[string]string{
		"power":                "Power",
		"spectral":             "Spectral",
		"aperiodic":            "Aperiodic",
		"erp":                  "ERP",
		"erds":                 "ERDS",
		"directedconnectivity": "Directed Connectivity",
		"sourcelocalization":   "Source Localization",
		"itpc":                 "ITPC",
	}

	for input, want := range tests {
		if got := featureCategoryLabel(input); got != want {
			t.Fatalf("featureCategoryLabel(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestFeatureCategoryIsEventSpecific(t *testing.T) {
	tests := map[string]bool{
		"power":       false,
		"spectral":    false,
		"aperiodic":   false,
		"erp":         true,
		"erds":        true,
		"itpc":        true,
		"pac":         false,
		"microstates": false,
	}

	for input, want := range tests {
		if got := featureCategoryIsEventSpecific(input); got != want {
			t.Fatalf("featureCategoryIsEventSpecific(%q) = %t, want %t", input, got, want)
		}
	}
}
