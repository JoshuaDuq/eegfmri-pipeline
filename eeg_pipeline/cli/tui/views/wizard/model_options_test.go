package wizard

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func hasOption(opts []optionType, want optionType) bool {
	for _, opt := range opts {
		if opt == want {
			return true
		}
	}
	return false
}

func TestGetFeaturesOptions_HidesTFRForNonTimeFrequencySelections(t *testing.T) {
	m := Model{
		categories:           []string{"aperiodic", "power"},
		selected:             map[int]bool{0: true},
		featGroupTFRExpanded: true,
		iafEnabled:           true,
	}

	opts := m.getFeaturesOptions()

	if hasOption(opts, optFeatGroupTFR) {
		t.Fatalf("did not expect optFeatGroupTFR for non-TF category selection")
	}
	if hasOption(opts, optIAFEnabled) {
		t.Fatalf("did not expect IAF options for non-TF category selection")
	}
}

func TestGetFeaturesOptions_ShowsTFRForTimeFrequencySelections(t *testing.T) {
	m := Model{
		categories:           []string{"aperiodic", "power"},
		selected:             map[int]bool{1: true},
		featGroupTFRExpanded: true,
		iafEnabled:           true,
	}

	opts := m.getFeaturesOptions()

	if !hasOption(opts, optFeatGroupTFR) {
		t.Fatalf("expected optFeatGroupTFR when TF category is selected")
	}
	if !hasOption(opts, optIAFEnabled) {
		t.Fatalf("expected IAF controls when TF category is selected and TFR group expanded")
	}
}

func TestGetFeaturesOptions_NoUnrelatedFeatureGroupsLeak(t *testing.T) {
	categories := []string{
		"power", "connectivity", "directedconnectivity", "sourcelocalization",
		"aperiodic", "itpc", "pac", "complexity", "ratios", "asymmetry",
		"quality", "microstates", "erds", "spectral", "erp", "bursts",
	}

	groupForCategory := map[string]optionType{
		"power":                optFeatGroupPower,
		"connectivity":         optFeatGroupConnectivity,
		"directedconnectivity": optFeatGroupDirectedConnectivity,
		"sourcelocalization":   optFeatGroupSourceLoc,
		"aperiodic":            optFeatGroupAperiodic,
		"itpc":                 optFeatGroupITPC,
		"pac":                  optFeatGroupPAC,
		"complexity":           optFeatGroupComplexity,
		"ratios":               optFeatGroupRatios,
		"asymmetry":            optFeatGroupAsymmetry,
		"quality":              optFeatGroupQuality,
		"microstates":          optFeatGroupMicrostates,
		"erds":                 optFeatGroupERDS,
		"spectral":             optFeatGroupSpectral,
		"erp":                  optFeatGroupERP,
		"bursts":               optFeatGroupBursts,
	}

	allFeatureGroups := []optionType{
		optFeatGroupConnectivity,
		optFeatGroupDirectedConnectivity,
		optFeatGroupPAC,
		optFeatGroupAperiodic,
		optFeatGroupComplexity,
		optFeatGroupERP,
		optFeatGroupBursts,
		optFeatGroupPower,
		optFeatGroupSpectral,
		optFeatGroupRatios,
		optFeatGroupAsymmetry,
		optFeatGroupQuality,
		optFeatGroupMicrostates,
		optFeatGroupERDS,
		optFeatGroupSpatialTransform,
		optFeatGroupSourceLoc,
		optFeatGroupITPC,
		optFeatGroupTFR,
	}

	isTFCategory := map[string]bool{
		"power":                true,
		"connectivity":         true,
		"directedconnectivity": true,
		"itpc":                 true,
		"pac":                  true,
		"erds":                 true,
		"bursts":               true,
	}

	for idx, selectedCategory := range categories {
		m := Model{
			categories: categories,
			selected:   map[int]bool{idx: true},
		}
		opts := m.getFeaturesOptions()

		allowed := map[optionType]bool{
			optUseDefaults:        true,
			optFeatGroupStorage:   true,
			optFeatGroupExecution: true,
		}
		if grp, ok := groupForCategory[selectedCategory]; ok {
			allowed[grp] = true
		}
		if selectedCategory == "connectivity" {
			allowed[optFeatGroupSpatialTransform] = true
		}
		if isTFCategory[selectedCategory] {
			allowed[optFeatGroupTFR] = true
		}

		for _, grp := range allFeatureGroups {
			has := hasOption(opts, grp)
			if has && !allowed[grp] {
				t.Fatalf("unexpected feature group %v shown when only %q selected", grp, selectedCategory)
			}
		}
	}
}

func TestGetFeaturesOptions_ExecutionOptionsAreCategoryScoped(t *testing.T) {
	categories := []string{"aperiodic", "complexity", "connectivity", "power", "itpc"}

	newModel := func(selectedIdx int) Model {
		return Model{
			categories:                 categories,
			selected:                   map[int]bool{selectedIdx: true},
			featGroupExecutionExpanded: true,
		}
	}

	// aperiodic only
	opts := newModel(0).getFeaturesOptions()
	if !hasOption(opts, optFeatNJobsAperiodic) {
		t.Fatalf("expected aperiodic n_jobs option for aperiodic selection")
	}
	if hasOption(opts, optFeatNJobsConnectivity) || hasOption(opts, optFeatNJobsComplexity) {
		t.Fatalf("did not expect connectivity/complexity n_jobs for aperiodic selection")
	}

	// complexity only
	opts = newModel(1).getFeaturesOptions()
	if !hasOption(opts, optFeatNJobsComplexity) {
		t.Fatalf("expected complexity n_jobs option for complexity selection")
	}
	if hasOption(opts, optFeatNJobsAperiodic) || hasOption(opts, optFeatNJobsConnectivity) {
		t.Fatalf("did not expect aperiodic/connectivity n_jobs for complexity selection")
	}

	// connectivity only
	opts = newModel(2).getFeaturesOptions()
	if !hasOption(opts, optFeatNJobsConnectivity) {
		t.Fatalf("expected connectivity n_jobs option for connectivity selection")
	}
	if hasOption(opts, optFeatNJobsAperiodic) || hasOption(opts, optFeatNJobsComplexity) {
		t.Fatalf("did not expect aperiodic/complexity n_jobs for connectivity selection")
	}

	// power only => band n_jobs appears
	opts = newModel(3).getFeaturesOptions()
	if !hasOption(opts, optFeatNJobsBands) {
		t.Fatalf("expected bands n_jobs option for power selection")
	}
	if hasOption(opts, optFeatNJobsConnectivity) {
		t.Fatalf("did not expect connectivity n_jobs for power selection")
	}

	// itpc only => itpc n_jobs appears, but band/conn/aperiodic/complexity do not
	opts = newModel(4).getFeaturesOptions()
	if !hasOption(opts, optItpcNJobs) {
		t.Fatalf("expected itpc n_jobs option for itpc selection")
	}
	if hasOption(opts, optFeatNJobsBands) || hasOption(opts, optFeatNJobsConnectivity) || hasOption(opts, optFeatNJobsAperiodic) || hasOption(opts, optFeatNJobsComplexity) {
		t.Fatalf("did not expect unrelated execution n_jobs options for itpc selection")
	}
}

func TestGetBehaviorOptions_HidesGlobalStatsAndIOForNonStatOnlySelections(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "report"
	}

	opts := m.getBehaviorOptions()

	if hasOption(opts, optBehaviorStatsTempControl) {
		t.Fatalf("did not expect behavior stats options for report-only selection")
	}
	if hasOption(opts, optGlobalNBootstrap) {
		t.Fatalf("did not expect global stats options for report-only selection")
	}
	if hasOption(opts, optValidationMinEpochs) {
		t.Fatalf("did not expect validation options for report-only selection")
	}
	if hasOption(opts, optIOTemperatureRange) {
		t.Fatalf("did not expect system/io options for report-only selection")
	}
}

func TestGetBehaviorOptions_ShowsGlobalStatsAndIOForStatSelections(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "correlations"
	}

	opts := m.getBehaviorOptions()

	if !hasOption(opts, optBehaviorStatsTempControl) {
		t.Fatalf("expected behavior stats options for correlations selection")
	}
	if !hasOption(opts, optGlobalNBootstrap) {
		t.Fatalf("expected global stats options for correlations selection")
	}
	if !hasOption(opts, optValidationMinEpochs) {
		t.Fatalf("expected validation options for correlations selection")
	}
	if !hasOption(opts, optIOTemperatureRange) {
		t.Fatalf("expected system/io options for correlations selection")
	}
}

func TestGetBehaviorOptions_HidesGlobalStatsAndIOForTrialTableOnly(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "trial_table"
	}

	opts := m.getBehaviorOptions()

	if hasOption(opts, optBehaviorStatsTempControl) || hasOption(opts, optGlobalNBootstrap) || hasOption(opts, optIOTemperatureRange) {
		t.Fatalf("did not expect stats/global/io options for trial_table-only selection")
	}
}

func TestGetBehaviorOptions_HidesGlobalStatsAndIOForPainResidualOnly(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "pain_residual"
	}

	opts := m.getBehaviorOptions()

	if hasOption(opts, optBehaviorStatsTempControl) || hasOption(opts, optGlobalNBootstrap) || hasOption(opts, optIOTemperatureRange) {
		t.Fatalf("did not expect stats/global/io options for pain_residual-only selection")
	}
}
