package wizard

import "testing"

func TestDoseResponseFeatureMetadataDiscovery(t *testing.T) {
	m := Model{}
	m.SetTrialTableColumns(
		[]string{
			"stimulus_temp",
			"power_active_alpha_global_mean",
			"power_active_beta_roi_Frontal_mean",
			"power_active_alpha_ch_Cz_logratio",
			"power_active_alpha_chpair_F3-F4_mean",
			"connectivity_active_alpha_global_geff",
		},
		nil,
	)

	bands := m.GetDoseResponseBands([]string{"power"})
	if len(bands) != 2 || bands[0] != "alpha" || bands[1] != "beta" {
		t.Fatalf("unexpected bands: %#v", bands)
	}

	scopes := m.GetDoseResponseScopes([]string{"power"})
	wantScopes := map[string]bool{"global": true, "roi": true, "ch": true, "chpair": true}
	if len(scopes) != 4 {
		t.Fatalf("unexpected scopes length: %#v", scopes)
	}
	for _, s := range scopes {
		if !wantScopes[s] {
			t.Fatalf("unexpected scope %q in %#v", s, scopes)
		}
	}

	rois := m.GetDoseResponseROIs([]string{"power"})
	if len(rois) != 1 || rois[0] != "Frontal" {
		t.Fatalf("unexpected rois: %#v", rois)
	}

	stats := m.GetDoseResponseStats([]string{"power"})
	if len(stats) != 2 || stats[0] != "logratio" || stats[1] != "mean" {
		t.Fatalf("unexpected stats: %#v", stats)
	}
}

func TestTrialTableFeatureCategoryDetectionIncludesMicrostates(t *testing.T) {
	m := Model{}
	m.SetTrialTableColumns(
		[]string{
			"microstates_active_alpha_global_mean",
			"power_active_alpha_global_mean",
		},
		nil,
	)

	cats := m.GetTrialTableFeatureCategories()
	if len(cats) == 0 {
		t.Fatalf("expected detected categories, got none")
	}

	foundMicrostates := false
	for _, c := range cats {
		if c == "microstates" {
			foundMicrostates = true
			break
		}
	}
	if !foundMicrostates {
		t.Fatalf("expected microstates in detected categories, got: %#v", cats)
	}
}
