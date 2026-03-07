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

func TestGetFeaturesOptions_SpatialTransformPerFamilyOptionsAreCategoryScoped(t *testing.T) {
	m := Model{
		categories: []string{"connectivity", "power", "itpc", "microstates"},
		selected: map[int]bool{
			0: true, // connectivity
		},
		featGroupSpatialTransformExpanded: true,
	}

	opts := m.getFeaturesOptions()

	if !hasOption(opts, optFeatGroupSpatialTransform) {
		t.Fatalf("expected spatial transform group when connectivity is selected")
	}
	if !hasOption(opts, optSpatialTransformPerFamilyConnectivity) {
		t.Fatalf("expected connectivity per-family spatial transform option")
	}
	if hasOption(opts, optSpatialTransformPerFamilyPower) {
		t.Fatalf("did not expect power per-family spatial transform option when power is not selected")
	}
	if hasOption(opts, optSpatialTransformPerFamilyItpc) {
		t.Fatalf("did not expect itpc per-family spatial transform option when itpc is not selected")
	}
	if hasOption(opts, optSpatialTransformPerFamilyMicrostates) {
		t.Fatalf("did not expect microstates per-family spatial transform option when microstates is not selected")
	}
}

func TestGetBehaviorOptions_HidesInferenceAndAdvancedForNonStatSelections(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "report"
	}

	opts := m.getBehaviorOptions()

	if hasOption(opts, optBehaviorGroupStats) || hasOption(opts, optBehaviorGroupAdvanced) {
		t.Fatalf("did not expect inference/advanced group headers for report-only selection")
	}
}

func TestGetBehaviorOptions_ShowsInferenceAndAdvancedForStatSelections(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "correlations"
	}

	// Group headers should appear when collapsed
	opts := m.getBehaviorOptions()
	if !hasOption(opts, optBehaviorGroupStats) {
		t.Fatalf("expected Inference & Shared Settings group header for correlations selection")
	}
	if !hasOption(opts, optBehaviorGroupAdvanced) {
		t.Fatalf("expected Advanced group header for correlations selection")
	}

	// Child options should appear when groups are expanded
	m.behaviorGroupStatsExpanded = true
	m.behaviorGroupAdvancedExpanded = true
	opts = m.getBehaviorOptions()

	if !hasOption(opts, optBehaviorStatsPredictorControl) {
		t.Fatalf("expected inference stats options for correlations selection")
	}
	if !hasOption(opts, optGlobalNBootstrap) {
		t.Fatalf("expected global stats options for correlations selection")
	}
	if !hasOption(opts, optValidationMinEpochs) {
		t.Fatalf("expected validation options for correlations selection")
	}
	if !hasOption(opts, optIOPredictorRange) {
		t.Fatalf("expected system/io options for correlations selection")
	}
}

func TestGetBehaviorOptions_HidesInferenceAndAdvancedForTrialTableOnly(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "trial_table"
	}

	opts := m.getBehaviorOptions()

	if hasOption(opts, optBehaviorGroupStats) || hasOption(opts, optBehaviorGroupAdvanced) {
		t.Fatalf("did not expect inference/advanced group headers for trial_table-only selection")
	}
}

func TestGetBehaviorOptions_ShowsInferenceAndAdvancedForPainResidualOnly(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "predictor_residual"
	}

	opts := m.getBehaviorOptions()

	if !hasOption(opts, optBehaviorGroupStats) {
		t.Fatalf("expected inference/shared settings group header for predictor_residual-only selection")
	}
	if !hasOption(opts, optBehaviorGroupAdvanced) {
		t.Fatalf("expected advanced group header for predictor_residual-only selection")
	}
}

func TestGetBehaviorOptions_CorrelationsOnlyHidesMultilevelControls(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "correlations"
	}
	m.behaviorGroupCorrelationsExpanded = true

	opts := m.getBehaviorOptions()

	if hasOption(opts, optBehaviorSubMultilevel) {
		t.Fatalf("did not expect multilevel subsection for correlations-only selection")
	}
	if hasOption(opts, optCorrelationsMultilevel) {
		t.Fatalf("did not expect multilevel computation toggle in correlations-only advanced config")
	}
	if hasOption(opts, optGroupLevelBlockPermutation) {
		t.Fatalf("did not expect group-level controls for correlations-only selection")
	}
}

func TestGetPreprocessingOptions_HidesAlignmentAndEventMappingRows(t *testing.T) {
	m := Model{modeIndex: 0}

	opts := m.getPreprocessingOptions()

	if !hasOption(opts, optConfigSetOverrides) {
		t.Fatalf("expected config overrides row to remain available")
	}
	if hasOption(opts, optAlignAllowMisalignedTrim) ||
		hasOption(opts, optAlignMinAlignmentSamples) ||
		hasOption(opts, optAlignTrimToFirstVolume) ||
		hasOption(opts, optAlignFmriOnsetReference) {
		t.Fatalf("did not expect alignment rows in preprocessing advanced options")
	}
	if hasOption(opts, optEventColPredictor) ||
		hasOption(opts, optEventColRating) ||
		hasOption(opts, optEventColBinaryOutcome) ||
		hasOption(opts, optEventColCondition) ||
		hasOption(opts, optEventColRequired) ||
		hasOption(opts, optConditionPreferredPrefixes) {
		t.Fatalf("did not expect event column mapping rows in preprocessing advanced options")
	}
}

func TestNonPreprocessingPipelines_DoNotShowPreprocessingAlignmentOrEventMappingRows(t *testing.T) {
	assertNoPrepLeak := func(t *testing.T, opts []optionType) {
		t.Helper()
		if hasOption(opts, optAlignAllowMisalignedTrim) ||
			hasOption(opts, optAlignMinAlignmentSamples) ||
			hasOption(opts, optAlignTrimToFirstVolume) ||
			hasOption(opts, optAlignFmriOnsetReference) ||
			hasOption(opts, optEventColPredictor) ||
			hasOption(opts, optEventColRating) ||
			hasOption(opts, optEventColBinaryOutcome) ||
			hasOption(opts, optEventColCondition) ||
			hasOption(opts, optEventColRequired) ||
			hasOption(opts, optConditionPreferredPrefixes) {
			t.Fatalf("unexpected preprocessing-only alignment/event mapping option in non-preprocessing pipeline")
		}
	}

	features := New(types.PipelineFeatures, ".")
	assertNoPrepLeak(t, features.getFeaturesOptions())

	behavior := New(types.PipelineBehavior, ".")
	assertNoPrepLeak(t, behavior.getBehaviorOptions())

	ml := New(types.PipelineML, ".")
	assertNoPrepLeak(t, ml.getMLOptions())

	fmriPrep := New(types.PipelineFmri, ".")
	assertNoPrepLeak(t, fmriPrep.getFmriPreprocessingOptions())

	fmriAnalysis := New(types.PipelineFmriAnalysis, ".")
	assertNoPrepLeak(t, fmriAnalysis.getFmriAnalysisOptions())
}
