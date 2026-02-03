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
