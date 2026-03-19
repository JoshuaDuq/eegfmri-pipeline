package wizard

import (
	"fmt"
	"strings"
)

// fMRI/fMRI analysis pipeline advanced argument builders.

func (m Model) buildFmriAdvancedArgs() []string {
	ab := newArgBuilder()

	// Runtime
	engine := "docker"
	if m.fmriEngineIndex%2 == 1 {
		engine = "apptainer"
	}
	ab.args = append(ab.args, "--engine", engine)
	ab.addIfNonEmpty("--fmriprep-image", m.fmriFmriprepImage)
	ab.addIfNonEmpty("--fmriprep-output-dir", expandUserPath(m.fmriFmriprepOutputDir))
	ab.addIfNonEmpty("--fmriprep-work-dir", expandUserPath(m.fmriFmriprepWorkDir))
	ab.addIfNonEmpty("--fs-license-file", expandUserPath(m.fmriFreesurferLicenseFile))
	ab.addIfNonEmpty("--fs-subjects-dir", expandUserPath(m.fmriFreesurferSubjectsDir))

	// Output
	ab.addSpaceListFlag("--output-spaces", m.fmriOutputSpacesSpec)
	ab.addSpaceListFlag("--ignore", m.fmriIgnoreSpec)
	ab.addIfNonEmpty("--bids-filter-file", expandUserPath(m.fmriBidsFilterFile))

	levelOptions := []string{"full", "resampling", "minimal"}
	if m.fmriLevelIndex > 0 {
		ab.args = append(ab.args, "--level", levelOptions[m.fmriLevelIndex%3])
	}

	ciftiOptions := []string{"", "91k", "170k"}
	if m.fmriCiftiOutputIndex > 0 {
		ab.args = append(ab.args, "--cifti-output", ciftiOptions[m.fmriCiftiOutputIndex%3])
	}

	ab.addIfNonEmpty("--task-id", m.fmriTaskId)

	// Performance
	if m.fmriNThreads > 0 {
		ab.args = append(ab.args, "--nthreads", fmt.Sprintf("%d", m.fmriNThreads))
	}
	if m.fmriOmpNThreads > 0 {
		ab.args = append(ab.args, "--omp-nthreads", fmt.Sprintf("%d", m.fmriOmpNThreads))
	}
	if m.fmriMemMb > 0 {
		ab.args = append(ab.args, "--mem-mb", fmt.Sprintf("%d", m.fmriMemMb))
	}
	if m.fmriLowMem {
		ab.args = append(ab.args, "--low-mem")
	}

	// Anatomical
	if m.fmriSkipReconstruction {
		ab.args = append(ab.args, "--fs-no-reconall")
	}
	if m.fmriLongitudinal {
		ab.args = append(ab.args, "--longitudinal")
	}
	if strings.TrimSpace(m.fmriSkullStripTemplate) != "" && m.fmriSkullStripTemplate != "OASIS30ANTs" {
		ab.args = append(ab.args, "--skull-strip-template", m.fmriSkullStripTemplate)
	}
	if m.fmriSkullStripFixedSeed {
		ab.args = append(ab.args, "--skull-strip-fixed-seed")
	}

	// BOLD processing
	bold2t1wInitOptions := []string{"register", "header"}
	if m.fmriBold2T1wInitIndex == 1 {
		ab.args = append(ab.args, "--bold2t1w-init", bold2t1wInitOptions[1])
	}
	if m.fmriBold2T1wDof != 6 {
		ab.args = append(ab.args, "--bold2t1w-dof", fmt.Sprintf("%d", m.fmriBold2T1wDof))
	}
	if m.fmriSliceTimeRef != 0.5 {
		ab.args = append(ab.args, "--slice-time-ref", fmt.Sprintf("%.2f", m.fmriSliceTimeRef))
	}
	if m.fmriDummyScans > 0 {
		ab.args = append(ab.args, "--dummy-scans", fmt.Sprintf("%d", m.fmriDummyScans))
	}

	// Quality control
	if m.fmriFdSpikeThreshold != 0.5 {
		ab.args = append(ab.args, "--fd-spike-threshold", fmt.Sprintf("%.2f", m.fmriFdSpikeThreshold))
	}
	if m.fmriDvarsSpikeThreshold != 1.5 {
		ab.args = append(ab.args, "--dvars-spike-threshold", fmt.Sprintf("%.2f", m.fmriDvarsSpikeThreshold))
	}

	// Denoising
	if m.fmriUseAroma {
		ab.args = append(ab.args, "--use-aroma")
	}

	// Surface
	if m.fmriMedialSurfaceNan {
		ab.args = append(ab.args, "--medial-surface-nan")
	}
	if m.fmriNoMsm {
		ab.args = append(ab.args, "--no-msm")
	}

	// Multi-echo
	if m.fmriMeOutputEchos {
		ab.args = append(ab.args, "--me-output-echos")
	}

	// Reproducibility
	if m.fmriRandomSeed > 0 {
		ab.args = append(ab.args, "--random-seed", fmt.Sprintf("%d", m.fmriRandomSeed))
	}

	// Validation
	if m.fmriSkipBidsValidation {
		ab.args = append(ab.args, "--skip-bids-validation")
	}
	if m.fmriStopOnFirstCrash {
		ab.args = append(ab.args, "--stop-on-first-crash")
	}
	if !m.fmriCleanWorkdir {
		ab.args = append(ab.args, "--no-clean-workdir")
	}
	if m.fmriTaskIsRest {
		ab.args = append(ab.args, "--task-is-rest")
	} else {
		ab.args = append(ab.args, "--no-task-is-rest")
	}

	// Advanced
	ab.addIfNonEmpty("--fmriprep-extra-args", m.fmriExtraArgs)

	return ab.build()
}

func (m Model) buildFmriAnalysisAdvancedArgs() []string {
	ab := newArgBuilder()

	mode := "first-level"
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}
	if mode == "second-level" {
		modelOptions := []string{"one-sample", "two-sample", "paired", "repeated-measures"}
		ab.args = append(ab.args, "--group-model", modelOptions[m.fmriSecondLevelModelIndex%len(modelOptions)])
		ab.addIfNonEmpty("--group-input-root", expandUserPath(strings.TrimSpace(m.fmriSecondLevelInputRoot)))

		contrastNames := splitSpaceList(strings.TrimSpace(m.fmriSecondLevelContrastNames))
		if len(contrastNames) > 0 {
			ab.args = append(ab.args, "--group-contrast-names")
			ab.args = append(ab.args, contrastNames...)
		}

		conditionLabels := splitSpaceList(strings.TrimSpace(m.fmriSecondLevelConditionLabels))
		if len(conditionLabels) > 0 {
			ab.args = append(ab.args, "--group-condition-labels")
			ab.args = append(ab.args, conditionLabels...)
		}

		ab.addIfNonEmpty("--group-covariates-file", expandUserPath(strings.TrimSpace(m.fmriSecondLevelCovariatesFile)))
		ab.addIfNonEmpty("--group-subject-column", strings.TrimSpace(m.fmriSecondLevelSubjectColumn))

		covariateColumns := splitSpaceList(strings.TrimSpace(m.fmriSecondLevelCovariateColumns))
		if len(covariateColumns) > 0 {
			ab.args = append(ab.args, "--group-covariate-columns")
			ab.args = append(ab.args, covariateColumns...)
		}

		ab.addIfNonEmpty("--group-column", strings.TrimSpace(m.fmriSecondLevelGroupColumn))
		ab.addIfNonEmpty("--group-a-value", strings.TrimSpace(m.fmriSecondLevelGroupAValue))
		ab.addIfNonEmpty("--group-b-value", strings.TrimSpace(m.fmriSecondLevelGroupBValue))
		ab.addIfNonEmpty("--formula", strings.TrimSpace(m.fmriSecondLevelFormula))
		ab.addIfNonEmpty("--contrast-name", strings.TrimSpace(m.fmriSecondLevelOutputName))
		ab.addIfNonEmpty("--output-dir", expandUserPath(strings.TrimSpace(m.fmriSecondLevelOutputDir)))

		if m.fmriSecondLevelWriteDesignMatrix {
			ab.args = append(ab.args, "--write-design-matrix")
		} else {
			ab.args = append(ab.args, "--no-write-design-matrix")
		}
		if m.fmriSecondLevelPermutationEnabled {
			ab.args = append(
				ab.args,
				"--group-permutation-inference",
				"--group-n-permutations",
				fmt.Sprintf("%d", m.fmriSecondLevelPermutationCount),
			)
			if !m.fmriSecondLevelTwoSided {
				ab.args = append(ab.args, "--group-one-sided")
			}
		}

		return ab.build()
	}

	isFirstLevel := mode == "first-level"
	isRest := mode == "rest"
	isTrialSignatures := mode == "trial-signatures"
	trialMethod := "beta-series"
	if m.fmriTrialSigMethodIndex%2 == 1 {
		trialMethod = "lss"
	}

	inputSource := "fmriprep"
	if m.fmriAnalysisInputSourceIndex%2 == 1 {
		inputSource = "bids_raw"
	}
	ab.args = append(ab.args, "--input-source", inputSource)

	if strings.TrimSpace(m.fmriAnalysisFmriprepSpace) != "" && strings.TrimSpace(m.fmriAnalysisFmriprepSpace) != "T1w" {
		ab.args = append(ab.args, "--fmriprep-space", strings.TrimSpace(m.fmriAnalysisFmriprepSpace))
	}

	if !m.fmriAnalysisRequireFmriprep {
		ab.args = append(ab.args, "--no-require-fmriprep")
	}

	// Runs: accept space-separated ints (e.g. "1 2 3")
	runsSpec := strings.TrimSpace(m.fmriAnalysisRunsSpec)
	if runsSpec != "" {
		runs := splitSpaceList(runsSpec)
		if len(runs) > 0 {
			ab.args = append(ab.args, "--runs")
			ab.args = append(ab.args, runs...)
		}
	}
	if isRest {
		ab.args = append(ab.args, "--task-is-rest")
	} else {
		ab.args = append(ab.args, "--no-task-is-rest")
	}

	// Contrast
	if !isRest {
		ab.addIfNonEmpty("--contrast-name", strings.TrimSpace(m.fmriAnalysisContrastName))
	}

	if isFirstLevel && m.fmriAnalysisContrastType%2 == 1 {
		ab.args = append(ab.args, "--contrast-type", "custom")
		ab.addIfNonEmpty("--formula", strings.TrimSpace(m.fmriAnalysisFormula))
	} else if !isRest {
		ab.args = append(ab.args, "--contrast-type", "t-test")
		ab.addIfNonEmpty("--cond-a-column", m.resolveFmriConditionColumn(m.fmriAnalysisCondAColumn))
		ab.addIfNonEmpty("--cond-a-value", strings.TrimSpace(m.fmriAnalysisCondAValue))
		ab.addIfNonEmpty("--cond-b-column", m.resolveFmriConditionColumn(m.fmriAnalysisCondBColumn))
		ab.addIfNonEmpty("--cond-b-value", strings.TrimSpace(m.fmriAnalysisCondBValue))
	}

	// GLM
	if !isRest {
		hrfOptions := []string{"spm", "flobs", "fir"}
		ab.args = append(ab.args, "--hrf-model", hrfOptions[m.fmriAnalysisHrfModel%len(hrfOptions)])
		driftOptions := []string{"none", "cosine", "polynomial"}
		ab.args = append(ab.args, "--drift-model", driftOptions[m.fmriAnalysisDriftModel%len(driftOptions)])
	}

	ab.args = append(ab.args, "--high-pass-hz", fmt.Sprintf("%.6f", m.fmriAnalysisHighPassHz))
	ab.args = append(ab.args, "--low-pass-hz", fmt.Sprintf("%.6f", m.fmriAnalysisLowPassHz))
	ab.args = append(ab.args, "--smoothing-fwhm", fmt.Sprintf("%.1f", m.fmriAnalysisSmoothingFwhm))

	// Confounds / QC
	confoundsOptions := []string{
		"auto",
		"none",
		"motion6",
		"motion12",
		"motion24",
		"motion24+wmcsf",
		"motion24+wmcsf+fd",
	}
	ab.args = append(ab.args, "--confounds-strategy", confoundsOptions[m.fmriAnalysisConfoundsStrategy%len(confoundsOptions)])
	if isFirstLevel {
		ab.addIfNonEmpty("--events-to-model", strings.TrimSpace(m.fmriAnalysisEventsToModel))
		eventsToModelColumn := strings.TrimSpace(m.fmriAnalysisEventsToModelColumn)
		if eventsToModelColumn != "" && !strings.EqualFold(eventsToModelColumn, "trial_type") {
			ab.args = append(ab.args, "--events-to-model-column", eventsToModelColumn)
		}
		scopeColumn := m.resolveFmriConditionColumn(m.fmriAnalysisScopeColumn)
		if scopeColumn != "" && !strings.EqualFold(scopeColumn, "trial_type") {
			ab.args = append(ab.args, "--condition-scope-column", scopeColumn)
		}
		scopeTrialTypes := strings.TrimSpace(m.fmriAnalysisScopeTrialTypes)
		if scopeTrialTypes != "" {
			ab.args = append(ab.args, "--condition-scope-trial-types")
			ab.args = append(ab.args, splitSpaceList(scopeTrialTypes)...)
		}
		phaseColumn := m.resolveFmriPhaseColumn(m.fmriAnalysisPhaseColumn)
		if phaseColumn != "" && !strings.EqualFold(phaseColumn, "stim_phase") {
			ab.args = append(ab.args, "--phase-column", phaseColumn)
		}
		phaseScopeColumn := m.resolveFmriConditionColumn(m.fmriAnalysisPhaseScopeColumn)
		if phaseScopeColumn != "" && !strings.EqualFold(phaseScopeColumn, "trial_type") {
			ab.args = append(ab.args, "--phase-scope-column", phaseScopeColumn)
		}
		phaseScopeValue := strings.TrimSpace(m.fmriAnalysisPhaseScopeValue)
		if phaseScopeValue != "" {
			ab.args = append(ab.args, "--phase-scope-value", phaseScopeValue)
		}
		if strings.TrimSpace(m.fmriAnalysisStimPhasesToModel) != "" {
			ab.args = append(ab.args, "--stim-phases-to-model", strings.TrimSpace(m.fmriAnalysisStimPhasesToModel))
		}
	}
	if isFirstLevel && m.fmriAnalysisWriteDesignMatrix {
		ab.args = append(ab.args, "--write-design-matrix")
	}
	if isRest {
		connectivityKinds := []string{"correlation"}
		ab.addIfNonEmpty("--atlas-labels-img", expandUserPath(strings.TrimSpace(m.fmriAnalysisAtlasLabelsImg)))
		ab.addIfNonEmpty("--atlas-labels-tsv", expandUserPath(strings.TrimSpace(m.fmriAnalysisAtlasLabelsTsv)))
		ab.args = append(ab.args, "--connectivity-kind", connectivityKinds[m.fmriAnalysisConnectivityKind])
		if m.fmriAnalysisStandardize {
			ab.args = append(ab.args, "--standardize")
		} else {
			ab.args = append(ab.args, "--no-standardize")
		}
		if m.fmriAnalysisDetrend {
			ab.args = append(ab.args, "--detrend")
		} else {
			ab.args = append(ab.args, "--no-detrend")
		}
	}

	// Output
	ab.addIfNonEmpty("--output-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisOutputDir)))

	if isFirstLevel {
		outTypeOptions := []string{"z-score", "t-stat", "cope", "beta"}
		ab.args = append(ab.args, "--output-type", outTypeOptions[m.fmriAnalysisOutputType%len(outTypeOptions)])

		if m.fmriAnalysisResampleToFS {
			ab.args = append(ab.args, "--resample-to-freesurfer")
			ab.addIfNonEmpty("--freesurfer-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisFreesurferDir)))
		}
	}

	// Trial-wise signatures (beta-series / lss)
	if isTrialSignatures {
		if !m.fmriTrialSigIncludeOtherEvents {
			ab.args = append(ab.args, "--no-include-other-events")
		}
		if m.fmriTrialSigMaxTrialsPerRun > 0 {
			ab.args = append(ab.args, "--max-trials-per-run", fmt.Sprintf("%d", m.fmriTrialSigMaxTrialsPerRun))
		}
		weighting := []string{"variance", "mean"}
		if m.fmriTrialSigFixedEffectsWeighting%len(weighting) != 0 {
			ab.args = append(ab.args, "--fixed-effects-weighting", weighting[m.fmriTrialSigFixedEffectsWeighting%len(weighting)])
		}
		if m.fmriTrialSigWriteTrialBetas {
			ab.args = append(ab.args, "--write-trial-betas")
		} else {
			ab.args = append(ab.args, "--no-write-trial-betas")
		}
		if m.fmriTrialSigWriteTrialVariances {
			ab.args = append(ab.args, "--write-trial-variances")
		} else {
			ab.args = append(ab.args, "--no-write-trial-variances")
		}
		if !m.fmriTrialSigWriteConditionBetas {
			ab.args = append(ab.args, "--no-write-condition-betas")
		}

		// Optional: restrict which trial_type/stim_phase values are eligible for trial selection.
		trialTypeScope := strings.TrimSpace(m.fmriTrialSigScopeTrialTypes)
		if trialTypeScope != "" {
			ab.args = append(ab.args, "--signature-scope-trial-types")
			ab.args = append(ab.args, splitSpaceList(trialTypeScope)...)
		}

		// Optional: restrict which stim_phase values are eligible for trial selection.
		// Empty => omit flag (no stim_phase scoping).
		phaseSpec := strings.TrimSpace(m.fmriTrialSigScopeStimPhases)
		if phaseSpec != "" {
			ab.args = append(ab.args, "--signature-scope-stim-phases")
			ab.args = append(ab.args, splitSpaceList(phaseSpec)...)
		}
		trialTypeScopeColumn := m.resolveFmriConditionColumn(m.fmriTrialSigScopeTrialTypeColumn)
		if trialTypeScopeColumn != "" && !strings.EqualFold(trialTypeScopeColumn, "trial_type") {
			ab.args = append(ab.args, "--signature-scope-trial-type-column", trialTypeScopeColumn)
		}
		phaseScopeColumn := m.resolveFmriPhaseColumn(m.fmriTrialSigScopePhaseColumn)
		if phaseScopeColumn != "" && !strings.EqualFold(phaseScopeColumn, "stim_phase") {
			ab.args = append(ab.args, "--signature-scope-phase-column", phaseScopeColumn)
		}

		if trialMethod == "lss" && m.fmriTrialSigLssOtherRegressorsIndex%2 == 1 {
			ab.args = append(ab.args, "--lss-other-regressors", "all")
		}

		if strings.TrimSpace(m.fmriAnalysisSignatureDir) != "" {
			ab.args = append(ab.args, "--signature-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisSignatureDir)))
		}
		if strings.TrimSpace(m.fmriAnalysisSignatureMaps) != "" {
			ab.args = append(ab.args, "--signature-maps")
			ab.args = append(ab.args, strings.Fields(strings.TrimSpace(m.fmriAnalysisSignatureMaps))...)
		}

		// Signature grouping
		groupCol := strings.TrimSpace(m.fmriTrialSigGroupColumn)
		groupVals := splitSpaceList(strings.TrimSpace(m.fmriTrialSigGroupValuesSpec))
		if groupCol != "" && len(groupVals) > 0 {
			ab.args = append(ab.args, "--signature-group-column", groupCol)
			ab.args = append(ab.args, "--signature-group-values")
			ab.args = append(ab.args, groupVals...)
			if m.fmriTrialSigGroupScopeIndex%2 == 1 {
				ab.args = append(ab.args, "--signature-group-scope", "per-run")
			}
		}

		return ab.build()
	}

	// Plotting / Report (CLI defaults are off)
	if isFirstLevel && m.fmriAnalysisPlotsEnabled {
		ab.args = append(ab.args, "--plots")

		if m.fmriAnalysisPlotHTML {
			ab.args = append(ab.args, "--plot-html-report")
		}

		plotSpaceOptions := []string{"both", "native", "mni"}
		ab.args = append(ab.args, "--plot-space", plotSpaceOptions[m.fmriAnalysisPlotSpaceIndex%len(plotSpaceOptions)])

		thresholdModeOptions := []string{"z", "fdr", "none"}
		ab.args = append(ab.args, "--plot-threshold-mode", thresholdModeOptions[m.fmriAnalysisPlotThresholdModeIndex%len(thresholdModeOptions)])

		ab.args = append(ab.args, "--plot-z-threshold", fmt.Sprintf("%.2f", m.fmriAnalysisPlotZThreshold))
		if m.fmriAnalysisPlotThresholdModeIndex%3 == 1 { // fdr
			ab.args = append(ab.args, "--plot-fdr-q", fmt.Sprintf("%.3f", m.fmriAnalysisPlotFdrQ))
		}
		if m.fmriAnalysisPlotClusterMinVoxels > 0 {
			ab.args = append(ab.args, "--plot-cluster-min-voxels", fmt.Sprintf("%d", m.fmriAnalysisPlotClusterMinVoxels))
		}

		vmaxModeOptions := []string{"per-space-robust", "shared-robust", "manual"}
		ab.args = append(ab.args, "--plot-vmax-mode", vmaxModeOptions[m.fmriAnalysisPlotVmaxModeIndex%len(vmaxModeOptions)])
		if m.fmriAnalysisPlotVmaxModeIndex%3 == 2 { // manual
			ab.args = append(ab.args, "--plot-vmax", fmt.Sprintf("%.2f", m.fmriAnalysisPlotVmaxManual))
		}

		if !m.fmriAnalysisPlotIncludeUnthresholded {
			ab.args = append(ab.args, "--no-plot-include-unthresholded")
		}

		if !m.fmriAnalysisPlotEffectSize {
			ab.args = append(ab.args, "--plot-no-effect-size")
		}
		if !m.fmriAnalysisPlotStandardError {
			ab.args = append(ab.args, "--plot-no-standard-error")
		}
		if !m.fmriAnalysisPlotMotionQC {
			ab.args = append(ab.args, "--plot-no-motion-qc")
		}
		if !m.fmriAnalysisPlotCarpetQC {
			ab.args = append(ab.args, "--plot-no-carpet-qc")
		}
		if !m.fmriAnalysisPlotTSNRQC {
			ab.args = append(ab.args, "--plot-no-tsnr-qc")
		}
		if !m.fmriAnalysisPlotDesignQC {
			ab.args = append(ab.args, "--plot-no-design-qc")
		}
		if !m.fmriAnalysisPlotEmbedImages {
			ab.args = append(ab.args, "--plot-no-embed-images")
		}
		if !m.fmriAnalysisPlotSignatures {
			ab.args = append(ab.args, "--plot-no-signatures")
		} else {
			if strings.TrimSpace(m.fmriAnalysisSignatureDir) != "" {
				ab.args = append(ab.args, "--signature-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisSignatureDir)))
			}
			if strings.TrimSpace(m.fmriAnalysisSignatureMaps) != "" {
				ab.args = append(ab.args, "--signature-maps")
				ab.args = append(ab.args, strings.Fields(strings.TrimSpace(m.fmriAnalysisSignatureMaps))...)
			}
		}

		// Formats: require at least one
		var formats []string
		if m.fmriAnalysisPlotFormatPNG {
			formats = append(formats, "png")
		}
		if m.fmriAnalysisPlotFormatSVG {
			formats = append(formats, "svg")
		}
		if len(formats) > 0 {
			ab.args = append(ab.args, "--plot-formats")
			ab.args = append(ab.args, formats...)
		}

		// Plot types: require at least one
		var plotTypes []string
		if m.fmriAnalysisPlotTypeSlices {
			plotTypes = append(plotTypes, "slices")
		}
		if m.fmriAnalysisPlotTypeGlass {
			plotTypes = append(plotTypes, "glass")
		}
		if m.fmriAnalysisPlotTypeHist {
			plotTypes = append(plotTypes, "hist")
		}
		if m.fmriAnalysisPlotTypeClusters {
			plotTypes = append(plotTypes, "clusters")
		}
		if len(plotTypes) > 0 {
			ab.args = append(ab.args, "--plot-types")
			ab.args = append(ab.args, plotTypes...)
		}
	}

	return ab.build()
}
