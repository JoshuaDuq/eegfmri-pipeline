package wizard

import (
	"reflect"
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func containsString(items []string, want string) bool {
	for _, it := range items {
		if it == want {
			return true
		}
	}
	return false
}

func argValue(args []string, key string) (string, bool) {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == key {
			return args[i+1], true
		}
	}
	return "", false
}

func containsSubsequence(items []string, subseq []string) bool {
	if len(subseq) == 0 {
		return true
	}
	for i := 0; i <= len(items)-len(subseq); i++ {
		match := true
		for j := 0; j < len(subseq); j++ {
			if items[i+j] != subseq[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func TestParseConfigSetOverrides_ParsesAndFiltersEntries(t *testing.T) {
	got := parseConfigSetOverrides("project.task=rest; analysis.min_subjects_for_group=4\n--set ml.n_perm=100;invalid")
	want := []string{
		"project.task=rest",
		"analysis.min_subjects_for_group=4",
		"ml.n_perm=100",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected overrides: got=%#v want=%#v", got, want)
	}
}

func TestBuildCommand_AppendsConfigSetOverrides(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	m.configSetOverrides = "project.task=rest;analysis.min_subjects_for_group=4"

	cmd := m.BuildCommand()
	if !strings.Contains(cmd, "--set project.task=rest") {
		t.Fatalf("expected first --set override in command, got: %s", cmd)
	}
	if !strings.Contains(cmd, "--set analysis.min_subjects_for_group=4") {
		t.Fatalf("expected second --set override in command, got: %s", cmd)
	}
}

func TestBuildBehaviorAdvancedArgs_OmitsDeprecatedTfHeatmapFlags(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.tfHeatmapEnabled = false
	m.tfHeatmapFreqsSpec = "4,8,13"
	m.tfHeatmapTimeResMs = 80

	args := m.buildBehaviorAdvancedArgs()
	if containsString(args, "--no-tf-heatmap-enabled") {
		t.Fatalf("did not expect deprecated --no-tf-heatmap-enabled in args: %#v", args)
	}
	if containsString(args, "--tf-heatmap-freqs") {
		t.Fatalf("did not expect deprecated --tf-heatmap-freqs in args: %#v", args)
	}
	if containsString(args, "--tf-heatmap-time-resolution-ms") {
		t.Fatalf("did not expect deprecated --tf-heatmap-time-resolution-ms in args: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_UsesSingleExplicitCorrelationTargetColumn(t *testing.T) {
	m := New(types.PipelineBehavior, ".")

	// Isolate correlations behavior.
	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "correlations"
	}
	m.correlationsTargetColumn = "vas_custom"

	args := m.buildBehaviorAdvancedArgs()

	if containsString(args, "--correlations-targets") {
		t.Fatalf("did not expect legacy --correlations-targets in args: %#v", args)
	}
	v, ok := argValue(args, "--correlations-target-column")
	if !ok {
		t.Fatalf("expected --correlations-target-column in args, got: %#v", args)
	}
	if v != "vas_custom" {
		t.Fatalf("unexpected --correlations-target-column value %q", v)
	}
}

func TestBuildBehaviorAdvancedArgs_EmitsEmptyCorrelationTargetColumn(t *testing.T) {
	m := New(types.PipelineBehavior, ".")

	for i := range m.computations {
		m.computationSelected[i] = m.computations[i].Key == "correlations"
	}
	m.correlationsTargetColumn = ""

	args := m.buildBehaviorAdvancedArgs()
	v, ok := argValue(args, "--correlations-target-column")
	if !ok {
		t.Fatalf("expected --correlations-target-column in args, got: %#v", args)
	}
	if v != "" {
		t.Fatalf("expected empty --correlations-target-column value, got %q", v)
	}
}

func TestBuildBehaviorAdvancedArgs_EmitsCanonicalColumns(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.behaviorOutcomeColumn = "vas_custom"
	m.behaviorPredictorColumn = "stimulus_intensity"

	args := m.buildBehaviorAdvancedArgs()

	outcome, ok := argValue(args, "--outcome-column")
	if !ok || outcome != "vas_custom" {
		t.Fatalf("expected --outcome-column vas_custom, got args: %#v", args)
	}
	predictor, ok := argValue(args, "--predictor-column")
	if !ok || predictor != "stimulus_intensity" {
		t.Fatalf("expected --predictor-column stimulus_intensity, got args: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_UsesStatsPredictorControlFlag(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.behaviorStatsTempControl = 1 // linear

	args := m.buildBehaviorAdvancedArgs()
	if containsString(args, "--stats-temp-control") {
		t.Fatalf("did not expect legacy --stats-temp-control in args: %#v", args)
	}
	if !containsSubsequence(args, []string{"--stats-predictor-control", "linear"}) {
		t.Fatalf("expected --stats-predictor-control linear in args: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_StatsPredictorNoneDisablesPredictorControl(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.behaviorStatsTempControl = 2 // none
	m.controlPredictor = true

	args := m.buildBehaviorAdvancedArgs()
	if !containsString(args, "--no-predictor-control") {
		t.Fatalf("expected --no-predictor-control when stats predictor control is none: %#v", args)
	}
	if containsString(args, "--stats-predictor-control") {
		t.Fatalf("did not expect --stats-predictor-control when set to none: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_DisabledCarpetAndTSNRAddsFlags(t *testing.T) {
	m := Model{}
	m.fmriAnalysisPlotsEnabled = true

	m.fmriAnalysisPlotCarpetQC = false
	m.fmriAnalysisPlotTSNRQC = false

	args := m.buildFmriAnalysisAdvancedArgs()

	if !containsString(args, "--plot-no-carpet-qc") {
		t.Fatalf("expected --plot-no-carpet-qc in args, got: %#v", args)
	}
	if !containsString(args, "--plot-no-tsnr-qc") {
		t.Fatalf("expected --plot-no-tsnr-qc in args, got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_EnabledCarpetAndTSNRDoesNotAddFlags(t *testing.T) {
	m := Model{}
	m.fmriAnalysisPlotsEnabled = true

	m.fmriAnalysisPlotCarpetQC = true
	m.fmriAnalysisPlotTSNRQC = true

	args := m.buildFmriAnalysisAdvancedArgs()

	if containsString(args, "--plot-no-carpet-qc") {
		t.Fatalf("did not expect --plot-no-carpet-qc in args, got: %#v", args)
	}
	if containsString(args, "--plot-no-tsnr-qc") {
		t.Fatalf("did not expect --plot-no-tsnr-qc in args, got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_SmoothingFwhmEmitted(t *testing.T) {
	m := Model{}
	m.fmriAnalysisPlotsEnabled = true
	m.fmriAnalysisSmoothingFwhm = 5.0

	args := m.buildFmriAnalysisAdvancedArgs()

	v, ok := argValue(args, "--smoothing-fwhm")
	if !ok {
		t.Fatalf("expected --smoothing-fwhm in args, got: %#v", args)
	}
	if v != "5.0" && v != "5.00" && v != "5" {
		t.Fatalf("unexpected --smoothing-fwhm value %q in args: %#v", v, args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_DisabledSignaturesAddsFlag(t *testing.T) {
	m := Model{}
	m.fmriAnalysisPlotsEnabled = true
	m.fmriAnalysisPlotSignatures = false

	args := m.buildFmriAnalysisAdvancedArgs()
	if !containsString(args, "--plot-no-signatures") {
		t.Fatalf("expected --plot-no-signatures in args, got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_SignatureDirEmittedWhenEnabled(t *testing.T) {
	m := Model{}
	m.fmriAnalysisPlotsEnabled = true
	m.fmriAnalysisPlotSignatures = true
	m.fmriAnalysisSignatureDir = "/tmp/signatures"

	args := m.buildFmriAnalysisAdvancedArgs()
	if containsString(args, "--plot-no-signatures") {
		t.Fatalf("did not expect --plot-no-signatures in args, got: %#v", args)
	}
	v, ok := argValue(args, "--signature-dir")
	if !ok {
		t.Fatalf("expected --signature-dir in args, got: %#v", args)
	}
	if v != "/tmp/signatures" {
		t.Fatalf("unexpected --signature-dir value %q", v)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_EventsToModelEmittedForFirstLevel(t *testing.T) {
	m := Model{}
	m.fmriAnalysisEventsToModel = "stimulation,pain_question,vas_rating"

	args := m.buildFmriAnalysisAdvancedArgs()

	v, ok := argValue(args, "--events-to-model")
	if !ok {
		t.Fatalf("expected --events-to-model in args, got: %#v", args)
	}
	if v != "stimulation,pain_question,vas_rating" {
		t.Fatalf("unexpected --events-to-model value %q", v)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_EmitsConditionScopeColumn(t *testing.T) {
	m := Model{}
	m.fmriAnalysisScopeColumn = "event_group"
	m.fmriAnalysisScopeTrialTypes = "stim"

	args := m.buildFmriAnalysisAdvancedArgs()

	if !containsSubsequence(args, []string{"--condition-scope-column", "event_group"}) {
		t.Fatalf("expected --condition-scope-column event_group, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-scope-trial-types", "stim"}) {
		t.Fatalf("expected --condition-scope-trial-types stim, got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_EmitsPhaseColumnScopeFlags(t *testing.T) {
	m := Model{}
	m.fmriAnalysisPhaseColumn = "event_phase"
	m.fmriAnalysisPhaseScopeColumn = "event_group"
	m.fmriAnalysisPhaseScopeValue = "stimulation"
	m.fmriAnalysisStimPhasesToModel = "plateau,peak"

	args := m.buildFmriAnalysisAdvancedArgs()

	if !containsSubsequence(args, []string{"--phase-column", "event_phase"}) {
		t.Fatalf("expected --phase-column event_phase, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--phase-scope-column", "event_group"}) {
		t.Fatalf("expected --phase-scope-column event_group, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--phase-scope-value", "stimulation"}) {
		t.Fatalf("expected --phase-scope-value stimulation, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--stim-phases-to-model", "plateau,peak"}) {
		t.Fatalf("expected --stim-phases-to-model plateau,peak, got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_EmitsTrialSignatureScopeColumns(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"first-level", "trial-signatures"}
	m.modeIndex = 1
	m.fmriTrialSigScopeTrialTypeColumn = "event_group"
	m.fmriTrialSigScopePhaseColumn = "event_phase"

	args := m.buildFmriAnalysisAdvancedArgs()

	if !containsSubsequence(args, []string{"--signature-scope-trial-type-column", "event_group"}) {
		t.Fatalf("expected --signature-scope-trial-type-column event_group, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--signature-scope-phase-column", "event_phase"}) {
		t.Fatalf("expected --signature-scope-phase-column event_phase, got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_BetaSeriesEmitsTrialSignatureFlags(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"first-level", "trial-signatures"}
	m.modeIndex = 1 // trial-signatures (beta-series by default)

	m.fmriTrialSigIncludeOtherEvents = false
	m.fmriTrialSigMethodIndex = 0 // beta-series
	m.fmriTrialSigWriteTrialBetas = true
	m.fmriTrialSigWriteTrialVariances = false
	m.fmriTrialSigWriteConditionBetas = false

	args := m.buildFmriAnalysisAdvancedArgs()

	if !containsString(args, "--no-include-other-events") {
		t.Fatalf("expected --no-include-other-events in args, got: %#v", args)
	}
	if !containsString(args, "--write-trial-betas") {
		t.Fatalf("expected --write-trial-betas in args, got: %#v", args)
	}
	if containsString(args, "--no-write-trial-betas") {
		t.Fatalf("did not expect --no-write-trial-betas in args, got: %#v", args)
	}
	if !containsString(args, "--no-write-trial-variances") {
		t.Fatalf("expected --no-write-trial-variances in args, got: %#v", args)
	}
	if !containsString(args, "--no-write-condition-betas") {
		t.Fatalf("expected --no-write-condition-betas in args, got: %#v", args)
	}
	if containsString(args, "--signatures") {
		t.Fatalf("did not expect --signatures in args (config-driven selection), got: %#v", args)
	}
}

func TestBuildFmriAnalysisAdvancedArgs_BetaSeriesEmitsSignatureGroupingFlags(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"first-level", "trial-signatures"}
	m.modeIndex = 1 // trial-signatures

	m.fmriTrialSigGroupColumn = "temperature"
	m.fmriTrialSigGroupValuesSpec = "44.3 45.3 46.3"
	m.fmriTrialSigGroupScopeIndex = 1  // per-run
	m.fmriTrialSigMethodIndex = 0      // beta-series
	m.fmriTrialSigGroupExpanded = true // irrelevant for args but matches UI

	args := m.buildFmriAnalysisAdvancedArgs()

	if v, ok := argValue(args, "--signature-group-column"); !ok || v != "temperature" {
		t.Fatalf("expected --signature-group-column temperature, got: %#v", args)
	}
	if !containsString(args, "--signature-group-values") {
		t.Fatalf("expected --signature-group-values in args, got: %#v", args)
	}
	for _, want := range []string{"44.3", "45.3", "46.3"} {
		if !containsString(args, want) {
			t.Fatalf("expected %q in args, got: %#v", want, args)
		}
	}
	if v, ok := argValue(args, "--signature-group-scope"); !ok || v != "per-run" {
		t.Fatalf("expected --signature-group-scope per-run, got: %#v", args)
	}
}

func TestBuildMLAdvancedArgs_FmriSignatureTargetEmitsFlags(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"regression"}
	m.modeIndex = 0

	m.mlTarget = "fmri_signature"
	m.mlFmriSigMethodIndex = 1 // lss
	m.mlFmriSigContrastName = "contrast"
	m.mlFmriSigMetricIndex = 2 // pearson_r
	m.mlFmriSigNormalizationIndex = 1
	m.mlFmriSigRoundDecimals = 4

	args := m.buildMLAdvancedArgs()

	if v, ok := argValue(args, "--target"); !ok || v != "fmri_signature" {
		t.Fatalf("expected --target fmri_signature, got: %#v", args)
	}
	if v, ok := argValue(args, "--fmri-signature-method"); !ok || v != "lss" {
		t.Fatalf("expected --fmri-signature-method lss, got: %#v", args)
	}
	if containsString(args, "--fmri-signature-name") {
		t.Fatalf("did not expect --fmri-signature-name (config-driven selection), got: %#v", args)
	}
	if v, ok := argValue(args, "--fmri-signature-metric"); !ok || v != "pearson_r" {
		t.Fatalf("expected --fmri-signature-metric pearson_r, got: %#v", args)
	}
	if v, ok := argValue(args, "--fmri-signature-normalization"); !ok || v != "zscore_within_run" {
		t.Fatalf("expected --fmri-signature-normalization zscore_within_run, got: %#v", args)
	}
	if v, ok := argValue(args, "--fmri-signature-round-decimals"); !ok || v != "4" {
		t.Fatalf("expected --fmri-signature-round-decimals 4, got: %#v", args)
	}
}

func TestBuildMLAdvancedArgs_ClassifyEmitsCNNClassificationModel(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"classify"}
	m.modeIndex = 0
	m.mlClassificationModel = MLClassificationCNN

	args := m.buildMLAdvancedArgs()

	if v, ok := argValue(args, "--classification-model"); !ok || v != "cnn" {
		t.Fatalf("expected --classification-model cnn, got: %#v", args)
	}
}

func TestBuildMLAdvancedArgs_EmitsMLPlottingFlags(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"shap"}
	m.modeIndex = 0

	m.mlPlotsEnabled = false
	m.mlPlotFormatsSpec = "png pdf"
	m.mlPlotDPI = 450
	m.mlPlotTopNFeatures = 30
	m.mlPlotDiagnostics = false

	args := m.buildMLAdvancedArgs()

	if !containsString(args, "--no-ml-plots") {
		t.Fatalf("expected --no-ml-plots in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--ml-plot-formats", "png", "pdf"}) {
		t.Fatalf("expected --ml-plot-formats png pdf in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--ml-plot-dpi", "450"}) {
		t.Fatalf("expected --ml-plot-dpi 450 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--ml-plot-top-n-features", "30"}) {
		t.Fatalf("expected --ml-plot-top-n-features 30 in args, got: %#v", args)
	}
	if !containsString(args, "--ml-plot-no-diagnostics") {
		t.Fatalf("expected --ml-plot-no-diagnostics in args, got: %#v", args)
	}
}

func TestBuildMLAdvancedArgs_EmitsWithinSubjectWithinBlockPermutationScheme(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"shap"}
	m.modeIndex = 0
	m.mlCvPermutationScheme = 1

	args := m.buildMLAdvancedArgs()

	if !containsSubsequence(args, []string{"--cv-permutation-scheme", "within_subject_within_block"}) {
		t.Fatalf("expected --cv-permutation-scheme within_subject_within_block in args, got: %#v", args)
	}
}

func TestMLClassificationModelCycleIncludesCNNAndEnsemble(t *testing.T) {
	if got := MLClassificationCNN.CLIValue(); got != "cnn" {
		t.Fatalf("expected cnn CLI value, got %q", got)
	}
	if next := MLClassificationRF.Next(); next != MLClassificationCNN {
		t.Fatalf("expected rf.Next() to be cnn, got %v", next)
	}
	if next := MLClassificationCNN.Next(); next != MLClassificationEnsemble {
		t.Fatalf("expected cnn.Next() to be ensemble, got %v", next)
	}
	if next := MLClassificationEnsemble.Next(); next != MLClassificationDefault {
		t.Fatalf("expected ensemble.Next() to wrap to default, got %v", next)
	}
}

func TestApplyConfigKeys_HydratesMLSettingsIncludingCNN(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"classify"}
	m.modeIndex = 0

	m.ApplyConfigKeys(map[string]interface{}{
		"machine_learning.targets.classification":              "binary_outcome",
		"machine_learning.targets.binary_threshold":            30.0,
		"machine_learning.data.feature_families":               []interface{}{"power", "connectivity"},
		"machine_learning.data.feature_harmonization":          "intersection",
		"machine_learning.data.covariates":                     []interface{}{"temperature", "block"},
		"machine_learning.data.require_trial_ml_safe":          true,
		"machine_learning.plotting.enabled":                    false,
		"machine_learning.plotting.formats":                    []interface{}{"png", "pdf"},
		"machine_learning.plotting.dpi":                        450.0,
		"machine_learning.plotting.top_n_features":             30.0,
		"machine_learning.plotting.include_diagnostics":        false,
		"machine_learning.classification.model":                "cnn",
		"machine_learning.cv.inner_splits":                     4.0,
		"machine_learning.models.random_forest.max_depth_grid": []interface{}{5.0, 10.0, nil},
	})

	if m.mlTarget != "binary_outcome" {
		t.Fatalf("expected mlTarget binary_outcome, got %q", m.mlTarget)
	}
	if !m.mlBinaryThresholdEnabled || m.mlBinaryThreshold != 30.0 {
		t.Fatalf("expected binary threshold enabled at 30.0, got enabled=%v value=%v", m.mlBinaryThresholdEnabled, m.mlBinaryThreshold)
	}
	if m.mlClassificationModel != MLClassificationCNN {
		t.Fatalf("expected CNN classification model, got %v", m.mlClassificationModel)
	}
	if m.mlFeatureFamiliesSpec != "power connectivity" {
		t.Fatalf("unexpected feature families spec: %q", m.mlFeatureFamiliesSpec)
	}
	if m.mlFeatureHarmonization != MLFeatureHarmonizationIntersection {
		t.Fatalf("unexpected feature harmonization: %v", m.mlFeatureHarmonization)
	}
	if m.mlCovariatesSpec != "temperature block" {
		t.Fatalf("unexpected covariates spec: %q", m.mlCovariatesSpec)
	}
	if !m.mlRequireTrialMlSafe {
		t.Fatalf("expected require-trial-ml-safe=true")
	}
	if m.mlPlotsEnabled {
		t.Fatalf("expected mlPlotsEnabled=false")
	}
	if m.mlPlotFormatsSpec != "png pdf" {
		t.Fatalf("unexpected ML plot formats: %q", m.mlPlotFormatsSpec)
	}
	if m.mlPlotDPI != 450 {
		t.Fatalf("expected mlPlotDPI=450, got %d", m.mlPlotDPI)
	}
	if m.mlPlotTopNFeatures != 30 {
		t.Fatalf("expected mlPlotTopNFeatures=30, got %d", m.mlPlotTopNFeatures)
	}
	if m.mlPlotDiagnostics {
		t.Fatalf("expected mlPlotDiagnostics=false")
	}
	if m.innerSplits != 4 {
		t.Fatalf("expected innerSplits=4, got %d", m.innerSplits)
	}
	if m.rfMaxDepthGrid != "5,10" {
		t.Fatalf("unexpected RF max depth grid: %q", m.rfMaxDepthGrid)
	}
}

func TestApplyConfigKeys_HydratesBehaviorSettings(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.ApplyConfigKeys(map[string]interface{}{
		"behavior_analysis.statistics.correlation_method":                                 "pearson",
		"behavior_analysis.robust_correlation":                                            "winsorized",
		"behavior_analysis.bootstrap":                                                     1500.0,
		"behavior_analysis.statistics.default_n_bootstrap":                                2500.0,
		"behavior_analysis.statistics.predictor_control":                                  "linear",
		"behavior_analysis.permutation.scheme":                                            "circular_shift",
		"behavior_analysis.permutation.group_column_preference":                           []interface{}{"run_id", "block"},
		"behavior_analysis.features.exclude_non_trialwise_features":                       false,
		"behavior_analysis.predictor_models.model_comparison.enabled":                     false,
		"behavior_analysis.predictor_models.breakpoint_test.enabled":                      false,
		"behavior_analysis.correlations.min_runs":                                         6.0,
		"behavior_analysis.correlations.target_column":                                    "custom_rating",
		"behavior_analysis.correlations.prefer_predictor_residual":                        true,
		"behavior_analysis.correlations.permutation.n_permutations":                       77.0,
		"behavior_analysis.predictor_sensitivity.primary_unit":                            "run_mean",
		"behavior_analysis.predictor_sensitivity.n_permutations":                          300.0,
		"behavior_analysis.predictor_sensitivity.p_primary_mode":                          "asymptotic",
		"behavior_analysis.group_level.multilevel_correlations.allow_parametric_fallback": true,
		"behavior_analysis.condition.primary_unit":                                        "run_mean",
		"behavior_analysis.condition.compare_labels":                                      []interface{}{"low", "high"},
		"behavior_analysis.condition.window_comparison.min_samples":                       12.0,
		"behavior_analysis.mixed_effects.include_predictor":                               false,
		"behavior_analysis.mediation.p_primary_mode":                                      "asymptotic",
		"behavior_analysis.moderation.p_primary_mode":                                     "asymptotic",
		"behavior_analysis.regression.primary_unit":                                       "run_mean",
		"behavior_analysis.temporal.correction_method":                                    "cluster",
		"behavior_analysis.cluster_correction.enabled":                                    true,
		"behavior_analysis.cluster_correction.alpha":                                      0.01,
		"behavior_analysis.cluster_correction.min_cluster_size":                           4.0,
		"behavior_analysis.cluster_correction.tail":                                       -1.0,
		"validation.min_epochs":                                                           30.0,
		"validation.min_channels":                                                         16.0,
		"validation.max_amplitude_uv":                                                     400.0,
		"io.constants.predictor_range":                                                    []interface{}{35.0, 55.0},
		"io.constants.max_missing_channels_fraction":                                      0.2,
	})

	if m.correlationMethod != "pearson" {
		t.Fatalf("expected correlation method pearson, got %q", m.correlationMethod)
	}
	if m.robustCorrelation != 2 {
		t.Fatalf("expected robustCorrelation=2 (winsorized), got %d", m.robustCorrelation)
	}
	if m.bootstrapSamples != 1500 {
		t.Fatalf("expected bootstrapSamples=1500, got %d", m.bootstrapSamples)
	}
	if m.globalNBootstrap != 2500 {
		t.Fatalf("expected globalNBootstrap=2500, got %d", m.globalNBootstrap)
	}
	if m.behaviorStatsTempControl != 1 {
		t.Fatalf("expected behaviorStatsTempControl=1 (linear), got %d", m.behaviorStatsTempControl)
	}
	if m.behaviorPermScheme != 1 {
		t.Fatalf("expected behaviorPermScheme=1 (circular_shift), got %d", m.behaviorPermScheme)
	}
	if m.behaviorPermGroupColumnPreference != "run_id,block" {
		t.Fatalf("unexpected behaviorPermGroupColumnPreference: %q", m.behaviorPermGroupColumnPreference)
	}
	if m.behaviorExcludeNonTrialwiseFeatures {
		t.Fatalf("expected behaviorExcludeNonTrialwiseFeatures=false")
	}
	if m.predictorResidualModelCompareEnabled {
		t.Fatalf("expected predictorResidualModelCompareEnabled=false")
	}
	if m.predictorResidualBreakpointEnabled {
		t.Fatalf("expected predictorResidualBreakpointEnabled=false")
	}
	if m.correlationsMinRuns != 6 {
		t.Fatalf("expected correlationsMinRuns=6, got %d", m.correlationsMinRuns)
	}
	if !m.correlationsPreferPredictorResidual {
		t.Fatalf("expected correlationsPreferPredictorResidual=true")
	}
	if m.correlationsPermutations != 77 {
		t.Fatalf("expected correlationsPermutations=77, got %d", m.correlationsPermutations)
	}
	if m.correlationsTargetColumn != "custom_rating" {
		t.Fatalf("unexpected correlationsTargetColumn: %q", m.correlationsTargetColumn)
	}
	if m.predictorSensitivityPrimaryUnit != 1 {
		t.Fatalf("expected predictorSensitivityPrimaryUnit=1 (run_mean), got %d", m.predictorSensitivityPrimaryUnit)
	}
	if m.predictorSensitivityPermutations != 300 {
		t.Fatalf("expected predictorSensitivityPermutations=300, got %d", m.predictorSensitivityPermutations)
	}
	if m.predictorSensitivityPermutationPrimary {
		t.Fatalf("expected predictorSensitivityPermutationPrimary=false")
	}
	if !m.groupLevelAllowParametricFallback {
		t.Fatalf("expected groupLevelAllowParametricFallback=true")
	}
	if m.conditionPrimaryUnit != 1 {
		t.Fatalf("expected conditionPrimaryUnit=1 (run_mean), got %d", m.conditionPrimaryUnit)
	}
	if m.conditionCompareLabels != "low,high" {
		t.Fatalf("unexpected conditionCompareLabels: %q", m.conditionCompareLabels)
	}
	if m.conditionWindowMinSamples != 12 {
		t.Fatalf("expected conditionWindowMinSamples=12, got %d", m.conditionWindowMinSamples)
	}
	if m.mixedIncludePredictor {
		t.Fatalf("expected mixedIncludePredictor=false")
	}
	if m.mediationPermutationPrimary {
		t.Fatalf("expected mediationPermutationPrimary=false")
	}
	if m.moderationPermutationPrimary {
		t.Fatalf("expected moderationPermutationPrimary=false")
	}
	if m.regressionPrimaryUnit != 1 {
		t.Fatalf("expected regressionPrimaryUnit=1 (run_mean), got %d", m.regressionPrimaryUnit)
	}
	if m.temporalCorrectionMethod != 1 {
		t.Fatalf("expected temporalCorrectionMethod=1 (cluster), got %d", m.temporalCorrectionMethod)
	}
	if !m.clusterCorrectionEnabled {
		t.Fatalf("expected clusterCorrectionEnabled=true")
	}
	if m.clusterCorrectionAlpha != 0.01 {
		t.Fatalf("expected clusterCorrectionAlpha=0.01, got %v", m.clusterCorrectionAlpha)
	}
	if m.clusterCorrectionMinClusterSize != 4 {
		t.Fatalf("expected clusterCorrectionMinClusterSize=4, got %d", m.clusterCorrectionMinClusterSize)
	}
	if m.clusterCorrectionTailGlobal != 2 {
		t.Fatalf("expected clusterCorrectionTailGlobal=2 (-1), got %d", m.clusterCorrectionTailGlobal)
	}
	if m.validationMinEpochs != 30 {
		t.Fatalf("expected validationMinEpochs=30, got %d", m.validationMinEpochs)
	}
	if m.validationMinChannels != 16 {
		t.Fatalf("expected validationMinChannels=16, got %d", m.validationMinChannels)
	}
	if m.validationMaxAmplitudeUv != 400.0 {
		t.Fatalf("expected validationMaxAmplitudeUv=400.0, got %v", m.validationMaxAmplitudeUv)
	}
	if m.ioPredictorRange != "35,55" {
		t.Fatalf("unexpected ioPredictorRange: %q", m.ioPredictorRange)
	}
	if m.ioMaxMissingChannelsFraction != 0.2 {
		t.Fatalf("expected ioMaxMissingChannelsFraction=0.2, got %v", m.ioMaxMissingChannelsFraction)
	}
}

func TestBuildCommand_PlottingGroupScopeAddsAnalysisScopeFlag(t *testing.T) {
	m := Model{}
	m.Pipeline = types.PipelinePlotting
	m.modeOptions = []string{"visualize"}
	m.modeIndex = 0
	m.plottingScope = PlottingScopeGroup

	cmd := m.BuildCommand()
	if !strings.Contains(cmd, "--analysis-scope group") {
		t.Fatalf("expected --analysis-scope group in command, got: %s", cmd)
	}
}

func TestBuildFeaturesAdvancedArgs_IncludesERDSPainMarkerFlags(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "erds" {
			m.selected[i] = true
			break
		}
	}

	m.erdsOnsetThresholdSigma = 1.8
	m.erdsOnsetMinDurationMs = 45.0
	m.erdsReboundMinLatencyMs = 180.0
	m.erdsInferContralateral = false

	args := m.buildFeaturesAdvancedArgs()

	if !containsSubsequence(args, []string{"--erds-onset-threshold-sigma", "1.80"}) {
		t.Fatalf("expected --erds-onset-threshold-sigma 1.80 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--erds-onset-min-duration-ms", "45.0"}) {
		t.Fatalf("expected --erds-onset-min-duration-ms 45.0 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--erds-rebound-min-latency-ms", "180.0"}) {
		t.Fatalf("expected --erds-rebound-min-latency-ms 180.0 in args, got: %#v", args)
	}
	if !containsString(args, "--no-erds-infer-contralateral") {
		t.Fatalf("expected --no-erds-infer-contralateral in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_IncludesMicrostatesFlags(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "microstates" {
			m.selected[i] = true
			break
		}
	}

	m.microstatesNStates = 6
	m.microstatesMinPeakDistanceMs = 12.5
	m.microstatesMaxGfpPeaksPerEpoch = 300
	m.microstatesMinDurationMs = 25.0
	m.microstatesGfpPeakProminence = 0.15
	m.microstatesRandomState = 77

	args := m.buildFeaturesAdvancedArgs()

	if !containsSubsequence(args, []string{"--microstates-n-states", "6"}) {
		t.Fatalf("expected --microstates-n-states 6 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--microstates-min-peak-distance-ms", "12.5"}) {
		t.Fatalf("expected --microstates-min-peak-distance-ms 12.5 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--microstates-max-gfp-peaks-per-epoch", "300"}) {
		t.Fatalf("expected --microstates-max-gfp-peaks-per-epoch 300 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--microstates-min-duration-ms", "25.0"}) {
		t.Fatalf("expected --microstates-min-duration-ms 25.0 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--microstates-gfp-peak-prominence", "0.15"}) {
		t.Fatalf("expected --microstates-gfp-peak-prominence 0.15 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--microstates-random-state", "77"}) {
		t.Fatalf("expected --microstates-random-state 77 in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_IncludesConnectivityAdvancedFlags(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "connectivity" {
			m.selected[i] = true
			break
		}
	}

	m.connMode = 2 // fourier
	m.connAECAbsolute = false
	m.connEnableAEC = false
	m.connNFreqsPerBand = 12
	m.connNCycles = 5.5
	m.connDecim = 2
	m.connMinSegSamples = 80
	m.connSmallWorldNRand = 250

	args := m.buildFeaturesAdvancedArgs()

	if !containsSubsequence(args, []string{"--conn-mode", "fourier"}) {
		t.Fatalf("expected --conn-mode fourier in args, got: %#v", args)
	}
	if !containsString(args, "--no-conn-aec-absolute") {
		t.Fatalf("expected --no-conn-aec-absolute in args, got: %#v", args)
	}
	if !containsString(args, "--no-conn-enable-aec") {
		t.Fatalf("expected --no-conn-enable-aec in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--conn-n-freqs-per-band", "12"}) {
		t.Fatalf("expected --conn-n-freqs-per-band 12 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--conn-n-cycles", "5.50"}) {
		t.Fatalf("expected --conn-n-cycles 5.50 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--conn-decim", "2"}) {
		t.Fatalf("expected --conn-decim 2 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--conn-min-segment-samples", "80"}) {
		t.Fatalf("expected --conn-min-segment-samples 80 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--conn-small-world-n-rand", "250"}) {
		t.Fatalf("expected --conn-small-world-n-rand 250 in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_SpatialTransformPerFamilyFlagsAreCategoryScoped(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "connectivity" {
			m.selected[i] = true
		}
	}

	m.spatialTransformPerFamilyConnectivity = 1 // none
	m.spatialTransformPerFamilyPower = 2        // csd
	m.spatialTransformPerFamilyItpc = 3         // laplacian

	args := m.buildFeaturesAdvancedArgs()

	if !containsSubsequence(args, []string{"--spatial-transform-connectivity", "none"}) {
		t.Fatalf("expected connectivity per-family spatial transform flag in args, got: %#v", args)
	}
	if containsString(args, "--spatial-transform-power") {
		t.Fatalf("did not expect power per-family spatial transform flag when power is not selected: %#v", args)
	}
	if containsString(args, "--spatial-transform-itpc") {
		t.Fatalf("did not expect itpc per-family spatial transform flag when itpc is not selected: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_ConnectivityMeasuresIncludeImcoh(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "connectivity" {
			m.selected[i] = true
			break
		}
	}

	for i := range m.connectivityMeasures {
		m.connectivityMeasures[i] = false
	}
	for i, measure := range connectivityMeasures {
		if measure.Key == "imcoh" {
			m.connectivityMeasures[i] = true
		}
	}

	args := m.buildFeaturesAdvancedArgs()
	if !containsSubsequence(args, []string{"--connectivity-measures", "imcoh"}) {
		t.Fatalf("expected --connectivity-measures imcoh in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_IncludesSourceSubjectsDirFlag(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "sourcelocalization" {
			m.selected[i] = true
			break
		}
	}

	m.sourceLocMode = 1
	m.sourceLocSubjectsDir = "/tmp/freesurfer_subjects"

	args := m.buildFeaturesAdvancedArgs()
	if !containsSubsequence(args, []string{"--source-subjects-dir", "/tmp/freesurfer_subjects"}) {
		t.Fatalf("expected --source-subjects-dir /tmp/freesurfer_subjects in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_EmitsSourceFmriConditionScopeColumn(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "sourcelocalization" {
			m.selected[i] = true
			break
		}
	}
	m.sourceLocMode = 1
	m.sourceLocFmriEnabled = true
	m.sourceLocFmriContrastEnabled = true
	m.sourceLocFmriConditionScopeColumn = "event_group"
	m.sourceLocFmriConditionScopeTrialTypes = "stimulation"

	args := m.buildFeaturesAdvancedArgs()

	if !containsSubsequence(args, []string{"--source-fmri-condition-scope-column", "event_group"}) {
		t.Fatalf("expected --source-fmri-condition-scope-column event_group in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--source-fmri-condition-scope-trial-types", "stimulation"}) {
		t.Fatalf("expected --source-fmri-condition-scope-trial-types stimulation in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_EmitsSourceFmriPhaseScopeFlags(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "sourcelocalization" {
			m.selected[i] = true
			break
		}
	}
	m.sourceLocMode = 1
	m.sourceLocFmriEnabled = true
	m.sourceLocFmriContrastEnabled = true
	m.sourceLocFmriPhaseColumn = "event_phase"
	m.sourceLocFmriPhaseScopeColumn = "event_group"
	m.sourceLocFmriPhaseScopeValue = "stimulation"
	m.sourceLocFmriStimPhasesToModel = "plateau"

	args := m.buildFeaturesAdvancedArgs()

	if !containsSubsequence(args, []string{"--source-fmri-phase-column", "event_phase"}) {
		t.Fatalf("expected --source-fmri-phase-column event_phase in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--source-fmri-phase-scope-column", "event_group"}) {
		t.Fatalf("expected --source-fmri-phase-scope-column event_group in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--source-fmri-phase-scope-value", "stimulation"}) {
		t.Fatalf("expected --source-fmri-phase-scope-value stimulation in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--source-fmri-stim-phases-to-model", "plateau"}) {
		t.Fatalf("expected --source-fmri-stim-phases-to-model plateau in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_IncludesPACRandomSeedFlag(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	for i, cat := range m.categories {
		if cat == "pac" {
			m.selected[i] = true
			break
		}
	}

	m.pacRandomSeed = 123

	args := m.buildFeaturesAdvancedArgs()
	if !containsSubsequence(args, []string{"--pac-random-seed", "123"}) {
		t.Fatalf("expected --pac-random-seed 123 in args, got: %#v", args)
	}
}

func TestBuildPreprocessingAdvancedArgs_EmitsEventColCondition(t *testing.T) {
	m := New(types.PipelinePreprocessing, ".")
	m.eventColCondition = "condition_label,trial_kind"

	args := m.buildPreprocessingAdvancedArgs()
	if !containsSubsequence(args, []string{"--event-col-condition", "condition_label", "trial_kind"}) {
		t.Fatalf("expected --event-col-condition condition_label trial_kind in args, got: %#v", args)
	}
}

func TestBuildFeaturesAdvancedArgs_PACSurrogateMethodMapping(t *testing.T) {
	cases := []struct {
		methodIndex int
		want        string
	}{
		{1, "circular_shift"},
		{2, "swap_phase_amp"},
		{3, "time_shift"},
	}

	for _, tc := range cases {
		m := New(types.PipelineFeatures, ".")
		for i, cat := range m.categories {
			if cat == "pac" {
				m.selected[i] = true
				break
			}
		}
		m.pacSurrogateMethod = tc.methodIndex

		args := m.buildFeaturesAdvancedArgs()
		if !containsSubsequence(args, []string{"--pac-surrogate-method", tc.want}) {
			t.Fatalf("methodIndex=%d expected --pac-surrogate-method %s in args, got: %#v", tc.methodIndex, tc.want, args)
		}
	}
}

func TestBuildBehaviorAdvancedArgs_IncludesMinSampleFlags(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		switch comp.Key {
		case "trial_table", "predictor_residual", "regression", "models", "validation", "moderation", "stability", "predictor_sensitivity", "condition":
			m.computationSelected[i] = true
		}
	}

	m.predictorResidualMinSamples = 14
	m.predictorResidualModelCompareMinSamples = 11
	m.predictorResidualBreakpointMinSamples = 16

	m.regressionTempControl = 2
	m.regressionTempSplineMinN = 18
	m.regressionMinSamples = 22

	m.modelsTempControl = 2
	m.modelsTempSplineMinN = 17
	m.modelsMinSamples = 25

	m.influenceTempControl = 2
	m.influenceTempSplineMinN = 19

	m.behaviorMinSamples = 7
	m.stabilityMinGroupN = 4
	m.predictorSensitivityMinTrials = 9
	m.conditionMinTrials = 6
	m.moderationMinSamples = 21

	args := m.buildBehaviorAdvancedArgs()
	if !containsSubsequence(args, []string{"--min-samples", "7"}) {
		t.Fatalf("expected --min-samples 7 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-residual-min-samples", "14"}) {
		t.Fatalf("expected --predictor-residual-min-samples 14 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-residual-model-compare-min-samples", "11"}) {
		t.Fatalf("expected --predictor-residual-model-compare-min-samples 11 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-residual-breakpoint-min-samples", "16"}) {
		t.Fatalf("expected --predictor-residual-breakpoint-min-samples 16 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--regression-predictor-spline-min-samples", "18"}) {
		t.Fatalf("expected --regression-predictor-spline-min-samples 18 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--regression-min-samples", "22"}) {
		t.Fatalf("expected --regression-min-samples 22 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--models-predictor-spline-min-samples", "17"}) {
		t.Fatalf("expected --models-predictor-spline-min-samples 17 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--models-min-samples", "25"}) {
		t.Fatalf("expected --models-min-samples 25 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--influence-predictor-spline-min-samples", "19"}) {
		t.Fatalf("expected --influence-predictor-spline-min-samples 19 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--stability-min-group-trials", "4"}) {
		t.Fatalf("expected --stability-min-group-trials 4 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-sensitivity-min-trials", "9"}) {
		t.Fatalf("expected --predictor-sensitivity-min-trials 9 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-min-trials", "6"}) {
		t.Fatalf("expected --condition-min-trials 6 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--moderation-min-samples", "21"}) {
		t.Fatalf("expected --moderation-min-samples 21 in args, got: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_EmitsGroupLevelAndModelValidityFlags(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		switch comp.Key {
		case "models", "multilevel_correlations":
			m.computationSelected[i] = true
		}
	}

	m.groupLevelBlockPermutation = false
	m.groupLevelTarget = "predictor_residual"
	m.groupLevelControlPredictor = false
	m.groupLevelControlTrialOrder = false
	m.groupLevelControlRunEffects = false
	m.groupLevelMaxRunDummies = 12
	m.groupLevelAllowParametricFallback = true
	m.modelsPrimaryUnit = 1 // run_mean
	m.modelsForceTrialIIDAsymptotic = true

	args := m.buildBehaviorAdvancedArgs()

	if !containsSubsequence(args, []string{"--group-level-target", "predictor_residual"}) {
		t.Fatalf("expected --group-level-target predictor_residual in args, got: %#v", args)
	}
	if !containsString(args, "--no-group-level-control-predictor") {
		t.Fatalf("expected --no-group-level-control-predictor in args, got: %#v", args)
	}
	if !containsString(args, "--no-group-level-control-trial-order") {
		t.Fatalf("expected --no-group-level-control-trial-order in args, got: %#v", args)
	}
	if !containsString(args, "--no-group-level-control-run-effects") {
		t.Fatalf("expected --no-group-level-control-run-effects in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--group-level-max-run-dummies", "12"}) {
		t.Fatalf("expected --group-level-max-run-dummies 12 in args, got: %#v", args)
	}
	if !containsString(args, "--group-level-allow-parametric-fallback") {
		t.Fatalf("expected --group-level-allow-parametric-fallback in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--models-primary-unit", "run_mean"}) {
		t.Fatalf("expected --models-primary-unit run_mean in args, got: %#v", args)
	}
	if !containsString(args, "--models-force-trial-iid-asymptotic") {
		t.Fatalf("expected --models-force-trial-iid-asymptotic in args, got: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_EmitsNewBehaviorRuntimeCoverageFlags(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		switch comp.Key {
		case "correlations", "predictor_sensitivity", "condition", "temporal", "regression", "mediation", "moderation", "mixed_effects":
			m.computationSelected[i] = true
		}
	}

	m.correlationsMinRuns = 5
	m.predictorSensitivityPrimaryUnit = 1
	m.predictorSensitivityPermutations = 250
	m.predictorSensitivityPermutationPrimary = false
	m.conditionPrimaryUnit = 1
	m.conditionWindowMinSamples = 14
	m.conditionCompareLabels = "low,high"
	m.mixedIncludePredictor = false
	m.mediationPermutationPrimary = false
	m.moderationPermutationPrimary = false
	m.correlationsPreferPredictorResidual = true
	m.correlationsPermutations = 111
	m.correlationsPowerSegment = "stimulation"
	m.regressionPrimaryUnit = 1
	m.temporalCorrectionMethod = 1

	args := m.buildBehaviorAdvancedArgs()

	if !containsSubsequence(args, []string{"--correlations-min-runs", "5"}) {
		t.Fatalf("expected --correlations-min-runs 5 in args, got: %#v", args)
	}
	if !containsString(args, "--correlations-prefer-predictor-residual") {
		t.Fatalf("expected --correlations-prefer-predictor-residual in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--correlations-permutations", "111"}) {
		t.Fatalf("expected --correlations-permutations 111 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--correlations-power-segment", "stimulation"}) {
		t.Fatalf("expected --correlations-power-segment stimulation in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-sensitivity-primary-unit", "run_mean"}) {
		t.Fatalf("expected --predictor-sensitivity-primary-unit run_mean in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-sensitivity-permutations", "250"}) {
		t.Fatalf("expected --predictor-sensitivity-permutations 250 in args, got: %#v", args)
	}
	if !containsString(args, "--no-predictor-sensitivity-permutation-primary") {
		t.Fatalf("expected --no-predictor-sensitivity-permutation-primary in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-primary-unit", "run_mean"}) {
		t.Fatalf("expected --condition-primary-unit run_mean in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--regression-primary-unit", "run_mean"}) {
		t.Fatalf("expected --regression-primary-unit run_mean in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--temporal-correction-method", "cluster"}) {
		t.Fatalf("expected --temporal-correction-method cluster in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-window-min-samples", "14"}) {
		t.Fatalf("expected --condition-window-min-samples 14 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-compare-labels", "low", "high"}) {
		t.Fatalf("expected --condition-compare-labels low high in args, got: %#v", args)
	}
	if !containsString(args, "--no-mixed-include-predictor") {
		t.Fatalf("expected --no-mixed-include-predictor in args, got: %#v", args)
	}
	if !containsString(args, "--no-mediation-permutation-primary") {
		t.Fatalf("expected --no-mediation-permutation-primary in args, got: %#v", args)
	}
	if !containsString(args, "--no-moderation-permutation-primary") {
		t.Fatalf("expected --no-moderation-permutation-primary in args, got: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_EmitsExplicitBooleanDisableFlags(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		switch comp.Key {
		case "trial_table", "correlations", "regression", "models", "validation", "condition":
			m.computationSelected[i] = true
		}
	}

	m.runAdjustmentEnabled = false
	m.predictorResidualEnabled = true
	m.predictorResidualCrossfitEnabled = false
	m.regressionIncludePrev = false
	m.modelsIncludePrev = false
	m.modelsForceTrialIIDAsymptotic = false
	m.influenceIncludeInteraction = false
	m.correlationsPermutationPrimary = false
	m.correlationsUseCrossfitResidual = false
	m.conditionPermutationPrimary = false
	m.behaviorStatsAllowIIDTrials = false
	m.behaviorStatsHierarchicalFDR = false
	m.behaviorStatsComputeReliability = false
	m.behaviorExcludeNonTrialwiseFeatures = false
	m.clusterCorrectionEnabled = false

	args := m.buildBehaviorAdvancedArgs()

	expectedFlags := []string{
		"--no-run-adjustment",
		"--no-predictor-residual-crossfit",
		"--no-regression-include-prev-terms",
		"--no-models-include-prev-terms",
		"--no-models-force-trial-iid-asymptotic",
		"--no-influence-include-interaction",
		"--no-correlations-permutation-primary",
		"--no-correlations-use-crossfit-predictor-residual",
		"--no-condition-permutation-primary",
		"--no-stats-allow-iid-trials",
		"--no-stats-hierarchical-fdr",
		"--no-stats-compute-reliability",
		"--no-exclude-non-trialwise-features",
		"--no-cluster-correction-enabled",
	}
	for _, flag := range expectedFlags {
		if !containsString(args, flag) {
			t.Fatalf("expected %s in args, got: %#v", flag, args)
		}
	}
}

func TestBuildCommand_BehaviorComputeIncludesSelectedBands(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	m.modeOptions = []string{"compute", "visualize"}
	m.modeIndex = 0
	m.useDefaultAdvanced = true

	for i := range m.bandSelected {
		m.bandSelected[i] = false
	}
	for i, band := range m.bands {
		if band.Key == "alpha" {
			m.bandSelected[i] = true
		}
	}

	cmd := m.BuildCommand()
	if !strings.Contains(cmd, "--bands alpha") {
		t.Fatalf("expected --bands alpha in command, got: %s", cmd)
	}
}

func TestBuildBehaviorAdvancedArgs_EmitsValidateOnlyAndFeatureFilters(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		switch comp.Key {
		case "correlations", "predictor_sensitivity", "condition", "temporal", "cluster", "mediation", "moderation":
			m.computationSelected[i] = true
		}
	}

	m.behaviorValidateOnly = true
	m.correlationsFeaturesSpec = "power,connectivity"
	m.predictorSensitivityFeaturesSpec = "erp,itpc"
	m.conditionFeaturesSpec = "spectral"
	m.temporalFeaturesSpec = "power,erds"
	m.clusterFeaturesSpec = "connectivity"
	m.mediationFeaturesSpec = "quality"
	m.moderationFeaturesSpec = "aperiodic,complexity"

	args := m.buildBehaviorAdvancedArgs()
	if !containsString(args, "--validate-only") {
		t.Fatalf("expected --validate-only in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--correlations-features", "power", "connectivity"}) {
		t.Fatalf("expected --correlations-features power connectivity in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--predictor-sensitivity-features", "erp", "itpc"}) {
		t.Fatalf("expected --predictor-sensitivity-features erp itpc in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-features", "spectral"}) {
		t.Fatalf("expected --condition-features spectral in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--temporal-features", "power", "erds"}) {
		t.Fatalf("expected --temporal-features power erds in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--cluster-features", "connectivity"}) {
		t.Fatalf("expected --cluster-features connectivity in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--mediation-features", "quality"}) {
		t.Fatalf("expected --mediation-features quality in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--moderation-features", "aperiodic", "complexity"}) {
		t.Fatalf("expected --moderation-features aperiodic complexity in args, got: %#v", args)
	}
}

func TestBuildBehaviorAdvancedArgs_GroupLevelTargetUsesAvailableTargetList(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		m.computationSelected[i] = comp.Key == "multilevel_correlations"
	}
	m.discoveredColumns = []string{"temperature", "rating"}
	m.groupLevelTarget = "temperature"

	args := m.buildBehaviorAdvancedArgs()
	if !containsSubsequence(args, []string{"--group-level-target", "temperature"}) {
		t.Fatalf("expected --group-level-target temperature in args, got: %#v", args)
	}
}

func TestShouldSkipStep_PlottingRoiStepSkippedForBandPowerTopomapsOnly(t *testing.T) {
	m := Model{}
	m.Pipeline = types.PipelinePlotting
	m.plotItems = []PlotItem{{ID: "band_power_topomaps", Group: "power"}}
	m.plotSelected = map[int]bool{0: true}
	m.selected = map[int]bool{}

	if m.plottingNeedsROIs() {
		t.Fatalf("expected plottingNeedsROIs=false for band_power_topomaps only")
	}
	if !m.shouldSkipStep(types.StepSelectROIs) {
		t.Fatalf("expected StepSelectROIs to be skipped for band_power_topomaps only")
	}
}

func TestBuildCommand_PlottingBandPowerTopomapsDoesNotEmitRois(t *testing.T) {
	m := Model{}
	m.Pipeline = types.PipelinePlotting
	m.modeOptions = []string{"visualize"}
	m.modeIndex = 0
	m.plotItems = []PlotItem{{ID: "band_power_topomaps", Group: "power"}}
	m.plotSelected = map[int]bool{0: true}
	m.selected = map[int]bool{}

	// Force ROI definitions to be non-default so we'd emit --rois if the pipeline needed ROIs.
	m.rois = []ROIDefinition{{Key: "TestROI", Name: "TestROI", Channels: "Fp1,Fp2"}}
	m.roiSelected = map[int]bool{0: true}

	cmd := m.BuildCommand()
	if strings.Contains(cmd, "--rois") {
		t.Fatalf("did not expect --rois in command, got: %s", cmd)
	}
}

func TestBuildPlotItemConfigArgs_BandPowerTopomapsFallsBackToComparisonWindows(t *testing.T) {
	m := Model{}
	m.plotItems = []PlotItem{{ID: "band_power_topomaps", Group: "power"}}
	m.plotSelected = map[int]bool{0: true}
	m.plotItemConfigs = map[string]PlotItemConfig{
		"band_power_topomaps": {
			ComparisonWindowsSpec: "baseline plateau",
		},
	}

	args := m.buildPlotItemConfigArgs()
	want := []string{"--plot-item-config", "band_power_topomaps", "topomap_windows", "baseline", "plateau"}
	if !containsSubsequence(args, want) {
		t.Fatalf("expected topomap_windows to default to comparison windows; args=%#v", args)
	}
}

func TestBuildPlotItemConfigArgs_DoseResponseEmitsBandsRoisScopesAndStat(t *testing.T) {
	m := Model{}
	m.plotItems = []PlotItem{{ID: "behavior_dose_response", Group: "behavior"}}
	m.plotSelected = map[int]bool{0: true}
	m.plotItemConfigs = map[string]PlotItemConfig{
		"behavior_dose_response": {
			DoseResponseDoseColumn:     "stimulus_temp",
			DoseResponseResponseColumn: "power",
			DoseResponseSegment:        "active",
			DoseResponseBandsSpec:      "alpha beta",
			DoseResponseROIsSpec:       "Frontal Sensorimotor_Right",
			DoseResponseScopesSpec:     "global roi",
			DoseResponseStat:           "mean",
		},
	}

	args := m.buildPlotItemConfigArgs()
	if !containsSubsequence(args, []string{"--plot-item-config", "behavior_dose_response", "dose_response_bands", "alpha", "beta"}) {
		t.Fatalf("expected dose_response_bands args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--plot-item-config", "behavior_dose_response", "dose_response_rois", "Frontal", "Sensorimotor_Right"}) {
		t.Fatalf("expected dose_response_rois args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--plot-item-config", "behavior_dose_response", "dose_response_scopes", "global", "roi"}) {
		t.Fatalf("expected dose_response_scopes args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--plot-item-config", "behavior_dose_response", "dose_response_stat", "mean"}) {
		t.Fatalf("expected dose_response_stat args, got: %#v", args)
	}
}
