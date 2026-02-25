package executor

import "testing"

func TestFlattenConfigValues_FlattensNestedSections(t *testing.T) {
	values := map[string]interface{}{
		"time_frequency_analysis": map[string]interface{}{
			"bands": map[string]interface{}{
				"alpha": []interface{}{8.0, 12.0},
			},
		},
		"fmri_preprocessing": map[string]interface{}{
			"fmriprep": map[string]interface{}{
				"nthreads":     float64(8),
				"omp_nthreads": float64(4),
			},
		},
		"machine_learning.targets.regression": "value",
	}

	got := flattenConfigValues(values)

	if _, ok := got["time_frequency_analysis.bands"]; !ok {
		t.Fatalf("expected flattened bands key to be present")
	}
	if got["fmri_preprocessing.fmriprep.nthreads"] != float64(8) {
		t.Fatalf("expected nthreads=8, got %v", got["fmri_preprocessing.fmriprep.nthreads"])
	}
	if got["fmri_preprocessing.fmriprep.omp_nthreads"] != float64(4) {
		t.Fatalf("expected omp_nthreads=4, got %v", got["fmri_preprocessing.fmriprep.omp_nthreads"])
	}
	if got["machine_learning.targets.regression"] != "value" {
		t.Fatalf("expected dotted key to be preserved, got %v", got["machine_learning.targets.regression"])
	}
}
