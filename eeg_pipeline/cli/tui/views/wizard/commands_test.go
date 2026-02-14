package wizard

import (
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

func TestBuildFmriAnalysisAdvancedArgs_BetaSeriesEmitsTrialSignatureFlags(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"first-level", "trial-signatures"}
	m.modeIndex = 1 // trial-signatures (beta-series by default)

	m.fmriTrialSigIncludeOtherEvents = false
	m.fmriTrialSigMethodIndex = 0 // beta-series
	m.fmriTrialSigWriteTrialBetas = true
	m.fmriTrialSigWriteTrialVariances = false
	m.fmriTrialSigWriteConditionBetas = false
	m.fmriTrialSigSignatureNPS = true
	m.fmriTrialSigSignatureSIIPS1 = false

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
	v, ok := argValue(args, "--signatures")
	if !ok {
		t.Fatalf("expected --signatures in args, got: %#v", args)
	}
	if v != "NPS" {
		t.Fatalf("unexpected --signatures value %q (expected NPS)", v)
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
	m.mlFmriSigContrastName = "pain_vs_nonpain"
	m.mlFmriSigSignatureIndex = 0 // NPS
	m.mlFmriSigMetricIndex = 2    // pearson_r
	m.mlFmriSigNormalizationIndex = 1
	m.mlFmriSigRoundDecimals = 4

	args := m.buildMLAdvancedArgs()

	if v, ok := argValue(args, "--target"); !ok || v != "fmri_signature" {
		t.Fatalf("expected --target fmri_signature, got: %#v", args)
	}
	if v, ok := argValue(args, "--fmri-signature-method"); !ok || v != "lss" {
		t.Fatalf("expected --fmri-signature-method lss, got: %#v", args)
	}
	if v, ok := argValue(args, "--fmri-signature-name"); !ok || v != "NPS" {
		t.Fatalf("expected --fmri-signature-name NPS, got: %#v", args)
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

func TestMLClassificationModelCycleIncludesCNN(t *testing.T) {
	if got := MLClassificationCNN.CLIValue(); got != "cnn" {
		t.Fatalf("expected cnn CLI value, got %q", got)
	}
	if next := MLClassificationRF.Next(); next != MLClassificationCNN {
		t.Fatalf("expected rf.Next() to be cnn, got %v", next)
	}
	if next := MLClassificationCNN.Next(); next != MLClassificationDefault {
		t.Fatalf("expected cnn.Next() to wrap to default, got %v", next)
	}
}

func TestApplyConfigKeys_HydratesMLSettingsIncludingCNN(t *testing.T) {
	m := Model{}
	m.modeOptions = []string{"classify"}
	m.modeIndex = 0

	m.ApplyConfigKeys(map[string]interface{}{
		"machine_learning.targets.classification":              "pain_binary",
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

	if m.mlTarget != "pain_binary" {
		t.Fatalf("expected mlTarget pain_binary, got %q", m.mlTarget)
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

func TestBuildBehaviorAdvancedArgs_IncludesMinSampleFlags(t *testing.T) {
	m := New(types.PipelineBehavior, ".")
	for i, comp := range m.computations {
		switch comp.Key {
		case "trial_table", "pain_residual", "regression", "models", "validation", "moderation", "stability", "pain_sensitivity", "condition":
			m.computationSelected[i] = true
		}
	}

	m.painResidualMinSamples = 14
	m.painResidualModelCompareMinSamples = 11
	m.painResidualBreakpointMinSamples = 16

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
	m.painSensitivityMinTrials = 9
	m.conditionMinTrials = 6
	m.moderationMinSamples = 21

	args := m.buildBehaviorAdvancedArgs()
	if !containsSubsequence(args, []string{"--min-samples", "7"}) {
		t.Fatalf("expected --min-samples 7 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--pain-residual-min-samples", "14"}) {
		t.Fatalf("expected --pain-residual-min-samples 14 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--pain-residual-model-compare-min-samples", "11"}) {
		t.Fatalf("expected --pain-residual-model-compare-min-samples 11 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--pain-residual-breakpoint-min-samples", "16"}) {
		t.Fatalf("expected --pain-residual-breakpoint-min-samples 16 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--regression-temperature-spline-min-samples", "18"}) {
		t.Fatalf("expected --regression-temperature-spline-min-samples 18 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--regression-min-samples", "22"}) {
		t.Fatalf("expected --regression-min-samples 22 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--models-temperature-spline-min-samples", "17"}) {
		t.Fatalf("expected --models-temperature-spline-min-samples 17 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--models-min-samples", "25"}) {
		t.Fatalf("expected --models-min-samples 25 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--influence-temperature-spline-min-samples", "19"}) {
		t.Fatalf("expected --influence-temperature-spline-min-samples 19 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--stability-min-group-trials", "4"}) {
		t.Fatalf("expected --stability-min-group-trials 4 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--pain-sensitivity-min-trials", "9"}) {
		t.Fatalf("expected --pain-sensitivity-min-trials 9 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--condition-min-trials", "6"}) {
		t.Fatalf("expected --condition-min-trials 6 in args, got: %#v", args)
	}
	if !containsSubsequence(args, []string{"--moderation-min-samples", "21"}) {
		t.Fatalf("expected --moderation-min-samples 21 in args, got: %#v", args)
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
