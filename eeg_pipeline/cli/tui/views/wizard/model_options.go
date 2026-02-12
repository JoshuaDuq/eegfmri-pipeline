package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/types"
)

// Advanced-option list builders and active-edit checks.

func (m Model) getFeaturesOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	if m.isCategorySelected("connectivity") {
		options = append(options, optFeatGroupConnectivity)
		if m.featGroupConnectivityExpanded {
			options = append(options, optConnectivity, optConnOutputLevel, optConnGranularity)
			if m.connGranularity == 1 {
				options = append(options, optConnConditionColumn, optConnConditionValues)
			}
			options = append(
				options,
				optConnPhaseEstimator,
				optConnMinEpochsPerGroup,
				optConnMinCyclesPerBand,
				optConnMinSegmentSec,
				optConnMinSegmentSamples,
				optConnWarnNoSpatialTransform,
				optConnGraphMetrics,
				optConnGraphProp,
				optConnSmallWorldNRand,
				optConnWindowLen,
				optConnWindowStep,
				optConnMode,
				optConnAECMode,
				optConnAECAbsolute,
				optConnEnableAEC,
				optConnNFreqsPerBand,
				optConnNCycles,
				optConnDecim,
				optConnAECOutput,
				optConnForceWithinEpochML,
				optConnDynamicEnabled,
				optConnDynamicMeasures,
				optConnDynamicAutocorrLag,
				optConnDynamicMinWindows,
				optConnDynamicIncludeROIPairs,
				optConnDynamicStateEnabled,
				optConnDynamicStateNStates,
				optConnDynamicStateMinWindows,
				optConnDynamicStateRandomSeed,
			)
		}
	}

	if m.isCategorySelected("directedconnectivity") {
		options = append(options, optFeatGroupDirectedConnectivity)
		if m.featGroupDirectedConnExpanded {
			options = append(options, optDirectedConnMeasures, optDirectedConnOutputLevel, optDirectedConnMvarOrder, optDirectedConnNFreqs, optDirectedConnMinSegSamples)
		}
	}

	if m.isCategorySelected("pac") {
		options = append(options, optFeatGroupPAC)
		if m.featGroupPACExpanded {
			options = append(options, optPACPhaseRange, optPACAmpRange, optPACMethod, optPACMinEpochs, optPACPairs,
				optPACSource, optPACNormalize, optPACNSurrogates, optPACAllowHarmonicOverlap, optPACMaxHarmonic, optPACHarmonicToleranceHz, optPACRandomSeed, optPACComputeWaveformQC, optPACWaveformOffsetMs)
		}
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optFeatGroupAperiodic)
		if m.featGroupAperiodicExpanded {
			options = append(
				options,
				optAperiodicModel,
				optAperiodicPsdMethod,
				optAperiodicFmin,
				optAperiodicFmax,
				optAperiodicPsdBandwidth,
				optAperiodicMinSegmentSec,
				optAperiodicExcludeLineNoise,
				optAperiodicLineNoiseFreq,
				optAperiodicLineNoiseWidthHz,
				optAperiodicLineNoiseHarmonics,
				optAperiodicPeakZ,
				optAperiodicMinR2,
				optAperiodicMinPoints,
				optAperiodicMaxRms,
				optAperiodicSubtractEvoked,
			)
		}
	}
	if m.isCategorySelected("complexity") {
		options = append(options, optFeatGroupComplexity)
		if m.featGroupComplexityExpanded {
			options = append(
				options,
				optPEOrder,
				optPEDelay,
				optComplexitySampleEntropyOrder,
				optComplexitySampleEntropyR,
				optComplexityMSEScaleMin,
				optComplexityMSEScaleMax,
				optComplexitySignalBasis,
				optComplexityMinSegmentSec,
				optComplexityMinSamples,
				optComplexityZscore,
			)
		}
	}
	if m.isCategorySelected("erp") {
		options = append(options, optFeatGroupERP)
		if m.featGroupERPExpanded {
			options = append(options, optERPBaseline, optERPAllowNoBaseline, optERPComponents, optERPSmoothMs, optERPPeakProminenceUv, optERPLowpassHz)
		}
	}
	if m.isCategorySelected("bursts") {
		options = append(options, optFeatGroupBursts)
		if m.featGroupBurstsExpanded {
			options = append(
				options,
				optBurstThresholdMethod,
				optBurstThresholdPercentile,
				optBurstThreshold,
				optBurstThresholdReference,
				optBurstMinTrialsPerCondition,
				optBurstMinSegmentSec,
				optBurstSkipInvalidSegments,
				optBurstMinDuration,
				optBurstMinCycles,
				optBurstBands,
			)
		}
	}
	if m.isCategorySelected("power") {
		options = append(options, optFeatGroupPower)
		if m.featGroupPowerExpanded {
			options = append(
				options,
				optPowerRequireBaseline,
				optPowerBaselineMode,
				optPowerSubtractEvoked,
				optPowerMinTrialsPerCondition,
				optPowerExcludeLineNoise,
				optPowerLineNoiseFreq,
				optPowerLineNoiseWidthHz,
				optPowerLineNoiseHarmonics,
				optPowerEmitDb,
			)
		}
	}
	if m.isCategorySelected("spectral") {
		options = append(options, optFeatGroupSpectral)
		if m.featGroupSpectralExpanded {
			options = append(
				options,
				optSpectralIncludeLogRatios,
				optSpectralPsdMethod,
				optSpectralPsdAdaptive,
				optSpectralMultitaperAdaptive,
				optSpectralFmin,
				optSpectralFmax,
				optSpectralSegments,
				optSpectralMinSegmentSec,
				optSpectralMinCyclesAtFmin,
				optSpectralExcludeLineNoise,
				optSpectralLineNoiseFreq,
				optSpectralLineNoiseWidthHz,
				optSpectralLineNoiseHarmonics,
				optSpectralEdge,
			)
		}
	}
	if m.isCategorySelected("ratios") {
		options = append(options, optFeatGroupRatios)
		if m.featGroupRatiosExpanded {
			options = append(options, optSpectralRatioPairs, optRatiosMinSegmentSec, optRatiosMinCyclesAtFmin, optRatiosSkipInvalidSegments)
		}
	}
	if m.isCategorySelected("asymmetry") {
		options = append(options, optFeatGroupAsymmetry)
		if m.featGroupAsymmetryExpanded {
			options = append(
				options,
				optAsymmetryChannelPairs,
				optAsymmetryMinSegmentSec,
				optAsymmetryMinCyclesAtFmin,
				optAsymmetrySkipInvalidSegments,
				optAsymmetryEmitActivationConvention,
				optAsymmetryActivationBands,
			)
		}
	}
	if m.isCategorySelected("quality") {
		options = append(options, optFeatGroupQuality)
		if m.featGroupQualityExpanded {
			options = append(options, optQualityPsdMethod, optQualityFmin, optQualityFmax, optQualityNFft,
				optQualityExcludeLineNoise, optQualityLineNoiseFreq, optQualityLineNoiseWidthHz, optQualityLineNoiseHarmonics,
				optQualitySnrSignalBandMin, optQualitySnrSignalBandMax, optQualitySnrNoiseBandMin, optQualitySnrNoiseBandMax,
				optQualityMuscleBandMin, optQualityMuscleBandMax)
		}
	}
	if m.isCategorySelected("microstates") {
		options = append(options, optFeatGroupMicrostates)
		if m.featGroupMicrostatesExpanded {
			options = append(
				options,
				optMicrostatesNStates,
				optMicrostatesMinPeakDistanceMs,
				optMicrostatesMaxGfpPeaksPerEpoch,
				optMicrostatesMinDurationMs,
				optMicrostatesGfpPeakProminence,
				optMicrostatesRandomState,
			)
		}
	}
	if m.isCategorySelected("erds") {
		options = append(options, optFeatGroupERDS)
		if m.featGroupERDSExpanded {
			options = append(
				options,
				optERDSUseLogRatio,
				optERDSMinBaselinePower,
				optERDSMinActivePower,
				optERDSMinSegmentSec,
				optERDSBands,
				optERDSOnsetThresholdSigma,
				optERDSOnsetMinDurationMs,
				optERDSReboundMinLatencyMs,
				optERDSInferContralateral,
			)
		}
	}

	// Spatial transform (for volume conduction reduction) - useful when connectivity is selected
	if m.isCategorySelected("connectivity") {
		options = append(options, optFeatGroupSpatialTransform)
		if m.featGroupSpatialTransformExpanded {
			options = append(options, optSpatialTransform, optSpatialTransformLambda2, optSpatialTransformStiffness)
		}
	}

	// Source localization (LCMV, eLORETA)
	if m.isCategorySelected("sourcelocalization") {
		options = append(options, optFeatGroupSourceLoc)
		if m.featGroupSourceLocExpanded {
			// Mode selection: EEG-only vs fMRI-informed
			options = append(options, optSourceLocMode)
			options = append(options, optSourceLocMethod, optSourceLocSpacing, optSourceLocParc)
			// Show method-specific options based on selected method
			if m.sourceLocMethod == 0 { // LCMV
				options = append(options, optSourceLocReg)
			} else { // eLORETA
				options = append(options, optSourceLocSnr, optSourceLocLoose, optSourceLocDepth)
			}
			options = append(options, optSourceLocConnMethod)

			// fMRI-informed mode (mode == 1) requires additional paths
			if m.sourceLocMode == 1 {
				// BEM/Trans generation options (Docker-based)
				// Note: FS License is configured in global paths, subject is from step 1
				options = append(options, optSourceLocSubjectsDir)
				options = append(options, optSourceLocCreateTrans, optSourceLocCreateBemModel, optSourceLocCreateBemSolution)
				// If not auto-creating, user must provide paths
				if !m.sourceLocCreateTrans {
					options = append(options, optSourceLocTrans)
				}
				if !m.sourceLocCreateBemSolution {
					options = append(options, optSourceLocBem)
				}
				options = append(options, optSourceLocMindistMm)
				options = append(options, optSourceLocFmriEnabled)
				if m.sourceLocFmriEnabled || strings.TrimSpace(m.sourceLocFmriStatsMap) != "" {
					options = append(options,
						optSourceLocFmriStatsMap,
						optSourceLocFmriProvenance,
						optSourceLocFmriRequireProvenance,
						optSourceLocFmriThreshold,
						optSourceLocFmriTail,
						optSourceLocFmriMaxClusters,
						optSourceLocFmriMaxVoxPerClus,
						optSourceLocFmriMaxTotalVox,
						optSourceLocFmriRandomSeed,
					)
					options = append(options, optSourceLocFmriMinClusterMM3)
					if m.sourceLocFmriMinClusterMM3 <= 0 {
						options = append(options, optSourceLocFmriMinClusterVox)
					}
					options = append(options, optSourceLocFmriContrastEnabled)
					if m.sourceLocFmriContrastEnabled {
						options = append(options, optSourceLocFmriContrastType)
						// Show condition fields based on contrast type
						if m.sourceLocFmriContrastType == 3 { // custom formula
							options = append(options, optSourceLocFmriContrastFormula)
						} else {
							options = append(options, optSourceLocFmriCondAColumn, optSourceLocFmriCondAValue)
							options = append(options, optSourceLocFmriCondBColumn, optSourceLocFmriCondBValue)
						}
						options = append(options, optSourceLocFmriContrastName)
						options = append(options, optSourceLocFmriAutoDetectRuns)
						if !m.sourceLocFmriAutoDetectRuns {
							options = append(options, optSourceLocFmriRunsToInclude)
						}
						options = append(options, optSourceLocFmriHrfModel, optSourceLocFmriDriftModel, optSourceLocFmriConditionScopeTrialTypes, optSourceLocFmriStimPhasesToModel)
						options = append(options, optSourceLocFmriHighPassHz, optSourceLocFmriLowPassHz)
						options = append(options, optSourceLocFmriClusterCorrection)
						if m.sourceLocFmriClusterCorrection {
							options = append(options, optSourceLocFmriClusterPThreshold)
						}
						options = append(options, optSourceLocFmriOutputType, optSourceLocFmriResampleToFS)
						// fMRI-specific time windows
						options = append(options, optSourceLocFmriWindowAName, optSourceLocFmriWindowATmin, optSourceLocFmriWindowATmax)
						options = append(options, optSourceLocFmriWindowBName, optSourceLocFmriWindowBTmin, optSourceLocFmriWindowBTmax)
					}
				}
			}
		}
	}

	// ITPC options (condition-based ITPC for avoiding pseudo-replication)
	if m.isCategorySelected("itpc") {
		options = append(options, optFeatGroupITPC)
		if m.featGroupITPCExpanded {
			options = append(options, optItpcMethod, optItpcAllowUnsafeLoo, optItpcBaselineCorrection)
			// Show condition-based options only when method is "condition" (method index 3)
			if m.itpcMethod == 3 {
				options = append(options, optItpcConditionColumn, optItpcConditionValues, optItpcMinTrialsPerCondition)
			}
		}
	}

	// TFR settings only for feature families that use time-frequency representations.
	if m.hasTimeFrequencyFeatureSelection() {
		options = append(options, optFeatGroupTFR)
		if m.featGroupTFRExpanded {
			options = append(
				options,
				optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrNCyclesFactor, optTfrWorkers,
				optBandEnvelopePadSec, optBandEnvelopePadCycles,
				optIAFEnabled,
			)
			if m.iafEnabled {
				options = append(
					options,
					optIAFAlphaWidthHz,
					optIAFSearchRangeMin,
					optIAFSearchRangeMax,
					optIAFMinProminence,
					optIAFRois,
					optIAFMinCyclesAtFmin,
					optIAFMinBaselineSec,
					optIAFAllowFullFallback,
					optIAFAllowAllChannelsFallback,
				)
			}
		}
	}

	options = append(options, optFeatGroupStorage)
	if m.featGroupStorageExpanded {
		options = append(options, optSaveSubjectLevelFeatures, optFeatAlsoSaveCsv)
	}

	options = append(options, optFeatGroupExecution)
	if m.featGroupExecutionExpanded {
		options = append(
			options,
			optMinEpochs,
			optFeatAnalysisMode,
			optFeatComputeChangeScores,
			optFeatSaveTfrWithSidecar,
		)
		if m.hasBandFeatureSelection() {
			options = append(options, optFeatNJobsBands)
		}
		if m.hasConnectivityFeatureSelection() {
			options = append(options, optFeatNJobsConnectivity)
		}
		if m.isCategorySelected("aperiodic") {
			options = append(options, optFeatNJobsAperiodic)
		}
		if m.isCategorySelected("complexity") {
			options = append(options, optFeatNJobsComplexity)
		}
		if m.isCategorySelected("itpc") {
			options = append(options, optItpcNJobs)
		}
	}

	return options
}

func (m Model) hasTimeFrequencyFeatureSelection() bool {
	tfCategories := []string{
		"power",
		"connectivity",
		"directedconnectivity",
		"itpc",
		"pac",
		"erds",
		"bursts",
	}
	for _, cat := range tfCategories {
		if m.isCategorySelected(cat) {
			return true
		}
	}
	return false
}

func (m Model) hasBandFeatureSelection() bool {
	bandCategories := []string{
		"power",
		"ratios",
		"asymmetry",
		"spectral",
		"erds",
		"bursts",
		"quality",
	}
	for _, cat := range bandCategories {
		if m.isCategorySelected(cat) {
			return true
		}
	}
	return false
}

func (m Model) hasConnectivityFeatureSelection() bool {
	return m.isCategorySelected("connectivity") || m.isCategorySelected("directedconnectivity")
}

// getPreprocessingOptions returns advanced options for preprocessing with collapsible groups
func (m Model) getPreprocessingOptions() []optionType {
	isFull := m.modeIndex == 0 || m.modeOptions[m.modeIndex] == "full"
	options := []optionType{optUseDefaults}

	// Stage Selection group (only show if not in full mode)
	if !isFull {
		options = append(options, optPrepGroupStages)
		if m.prepGroupStagesExpanded {
			options = append(options,
				optPrepStageBadChannels,
				optPrepStageFiltering,
				optPrepStageICA,
				optPrepStageEpoching,
			)
		}
	}

	// General Settings group (montage, jobs, etc.)
	options = append(options, optPrepGroupGeneral)
	if m.prepGroupGeneralExpanded {
		options = append(options,
			optPrepMontage,
			optPrepChTypes,
			optPrepEegReference,
			optPrepEogChannels,
			optPrepRandomState,
			optPrepTaskIsRest,
			optPrepNJobs,
			optPrepUsePyprep,
			optPrepUseIcalabel,
		)
	}

	// Filtering group
	if isFull || m.prepStageSelected[1] {
		options = append(options, optPrepGroupFiltering)
		if m.prepGroupFilteringExpanded {
			options = append(options,
				optPrepResample,
				optPrepLFreq,
				optPrepHFreq,
				optPrepNotch,
				optPrepLineFreq,
				optPrepZaplineFline,
				optPrepFindBreaks,
			)
		}
	}

	// PyPREP Advanced group (part of bad channel detection if enabled)
	if (isFull || m.prepStageSelected[0]) && m.prepUsePyprep {
		options = append(options, optPrepGroupPyprep)
		if m.prepGroupPyprepExpanded {
			options = append(options,
				optPrepRansac,
				optPrepRepeats,
				optPrepAverageReref,
				optPrepFileExtension,
				optPrepConsiderPreviousBads,
				optPrepOverwriteChansTsv,
				optPrepDeleteBreaks,
				optPrepBreaksMinLength,
				optPrepTStartAfterPrevious,
				optPrepTStopBeforeNext,
				optPrepRenameAnotDict,
				optPrepCustomBadDict,
			)
		}
	}

	// ICA group
	if isFull || m.prepStageSelected[2] {
		options = append(options, optPrepGroupICA)
		if m.prepGroupICAExpanded {
			options = append(options,
				optPrepSpatialFilter,
				optPrepICAAlgorithm,
				optPrepICAComp,
				optPrepICALFreq,
				optPrepICARejThresh,
				optPrepProbThresh,
				optPrepKeepMnebidsBads,
				optIcaLabelsToKeep,
			)
		}
	}

	// Epoching group
	if isFull || m.prepStageSelected[3] {
		options = append(options, optPrepGroupEpoching)
		if m.prepGroupEpochingExpanded {
			options = append(options,
				optPrepConditions,
				optPrepEpochsTmin,
				optPrepEpochsTmax,
				optPrepEpochsNoBaseline,
				optPrepEpochsBaseline,
				optPrepEpochsReject,
				optPrepRejectMethod,
				optPrepRunSourceEstimation,
				optPrepWriteCleanEvents,
				optPrepOverwriteCleanEvents,
				optPrepCleanEventsStrict,
			)
		}
	}

	return options
}

func (m Model) getFmriPreprocessingOptions() []optionType {
	options := []optionType{optUseDefaults}

	// Runtime group
	options = append(options, optFmriGroupRuntime)
	if m.fmriGroupRuntimeExpanded {
		options = append(options, optFmriEngine, optFmriFmriprepImage)
	}

	// Output group
	options = append(options, optFmriGroupOutput)
	if m.fmriGroupOutputExpanded {
		options = append(options, optFmriOutputSpaces, optFmriIgnore, optFmriLevel, optFmriCiftiOutput, optFmriTaskId)
	}

	// Performance group
	options = append(options, optFmriGroupPerformance)
	if m.fmriGroupPerformanceExpanded {
		options = append(options, optFmriNThreads, optFmriOmpNThreads, optFmriMemMb, optFmriLowMem)
	}

	// Anatomical group
	options = append(options, optFmriGroupAnatomical)
	if m.fmriGroupAnatomicalExpanded {
		options = append(options, optFmriSkipReconstruction, optFmriLongitudinal, optFmriSkullStripTemplate, optFmriSkullStripFixedSeed)
	}

	// BOLD processing group
	options = append(options, optFmriGroupBold)
	if m.fmriGroupBoldExpanded {
		options = append(options, optFmriBold2T1wInit, optFmriBold2T1wDof, optFmriSliceTimeRef, optFmriDummyScans)
	}

	// Quality control group
	options = append(options, optFmriGroupQc)
	if m.fmriGroupQcExpanded {
		options = append(options, optFmriFdSpikeThreshold, optFmriDvarsSpikeThreshold)
	}

	// Denoising group
	options = append(options, optFmriGroupDenoising)
	if m.fmriGroupDenoisingExpanded {
		options = append(options, optFmriUseAroma)
	}

	// Surface group
	options = append(options, optFmriGroupSurface)
	if m.fmriGroupSurfaceExpanded {
		options = append(options, optFmriMedialSurfaceNan, optFmriNoMsm)
	}

	// Multi-echo group
	options = append(options, optFmriGroupMultiecho)
	if m.fmriGroupMultiechoExpanded {
		options = append(options, optFmriMeOutputEchos)
	}

	// Reproducibility group
	options = append(options, optFmriGroupRepro)
	if m.fmriGroupReproExpanded {
		options = append(options, optFmriRandomSeed)
	}

	// Validation group
	options = append(options, optFmriGroupValidation)
	if m.fmriGroupValidationExpanded {
		options = append(options, optFmriSkipBidsValidation, optFmriStopOnFirstCrash, optFmriCleanWorkdir)
	}

	// Advanced group
	options = append(options, optFmriGroupAdvanced)
	if m.fmriGroupAdvancedExpanded {
		options = append(options, optFmriExtraArgs)
	}

	return options
}

func (m Model) getFmriAnalysisOptions() []optionType {
	options := []optionType{optUseDefaults}

	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}
	isFirstLevel := mode == "" || mode == "first-level"

	options = append(options, optFmriAnalysisGroupInput)
	if m.fmriAnalysisGroupInputExpanded {
		options = append(options,
			optFmriAnalysisInputSource,
			optFmriAnalysisFmriprepSpace,
			optFmriAnalysisRequireFmriprep,
			optFmriAnalysisRuns,
		)
	}

	options = append(options, optFmriAnalysisGroupContrast)
	if m.fmriAnalysisGroupContrastExpanded {
		if isFirstLevel {
			options = append(options, optFmriAnalysisContrastType)
		}
		options = append(options,
			optFmriAnalysisCondAColumn,
			optFmriAnalysisCondAValue,
			optFmriAnalysisCondBColumn,
			optFmriAnalysisCondBValue,
			optFmriAnalysisContrastName,
		)
		if isFirstLevel && m.fmriAnalysisContrastType == 1 {
			options = append(options, optFmriAnalysisFormula)
		}
	}

	options = append(options, optFmriAnalysisGroupGLM)
	if m.fmriAnalysisGroupGLMExpanded {
		options = append(options,
			optFmriAnalysisHrfModel,
			optFmriAnalysisDriftModel,
			optFmriAnalysisHighPassHz,
			optFmriAnalysisLowPassHz,
			optFmriAnalysisSmoothingFwhm,
		)
	}

	options = append(options, optFmriAnalysisGroupConfounds)
	if m.fmriAnalysisGroupConfoundsExpanded {
		if isFirstLevel {
			options = append(options, optFmriAnalysisEventsToModel, optFmriAnalysisScopeTrialTypes, optFmriAnalysisStimPhasesToModel)
		}
		options = append(options, optFmriAnalysisConfoundsStrategy)
		if isFirstLevel {
			options = append(options, optFmriAnalysisWriteDesignMatrix)
		}
	}

	options = append(options, optFmriAnalysisGroupOutput)
	if m.fmriAnalysisGroupOutputExpanded {
		if isFirstLevel {
			options = append(options,
				optFmriAnalysisOutputType,
				optFmriAnalysisOutputDir,
				optFmriAnalysisResampleToFS,
			)
			if m.fmriAnalysisResampleToFS {
				options = append(options, optFmriAnalysisFreesurferDir)
			}
		} else {
			options = append(options, optFmriAnalysisOutputDir)
		}
	}

	if isFirstLevel {
		options = append(options, optFmriAnalysisGroupPlotting)
		if m.fmriAnalysisGroupPlottingExpanded {
			options = append(options, optFmriAnalysisPlotsEnabled, optFmriAnalysisPlotHTML, optFmriAnalysisPlotSpace)

			// Thresholding
			options = append(options, optFmriAnalysisPlotThresholdMode, optFmriAnalysisPlotZThreshold)
			if m.fmriAnalysisPlotThresholdModeIndex%3 == 1 { // fdr
				options = append(options, optFmriAnalysisPlotFdrQ)
			}
			options = append(options, optFmriAnalysisPlotClusterMinVoxels)

			// Scaling
			options = append(options, optFmriAnalysisPlotVmaxMode)
			if m.fmriAnalysisPlotVmaxModeIndex%3 == 2 { // manual
				options = append(options, optFmriAnalysisPlotVmaxManual)
			}

			// Content
			options = append(options,
				optFmriAnalysisPlotIncludeUnthresholded,
				optFmriAnalysisPlotFormatPNG,
				optFmriAnalysisPlotFormatSVG,
				optFmriAnalysisPlotTypeSlices,
				optFmriAnalysisPlotTypeGlass,
				optFmriAnalysisPlotTypeHist,
				optFmriAnalysisPlotTypeClusters,
				optFmriAnalysisPlotEffectSize,
				optFmriAnalysisPlotStandardError,
				optFmriAnalysisPlotMotionQC,
				optFmriAnalysisPlotCarpetQC,
				optFmriAnalysisPlotTSNRQC,
				optFmriAnalysisPlotDesignQC,
				optFmriAnalysisPlotEmbedImages,
				optFmriAnalysisPlotSignatures,
				optFmriAnalysisSignatureDir,
			)
		}
	} else {
		options = append(options, optFmriTrialSigGroup)
		if m.fmriTrialSigGroupExpanded {
			trialMethod := "beta-series"
			if m.fmriTrialSigMethodIndex%2 == 1 {
				trialMethod = "lss"
			}
			options = append(options,
				optFmriTrialSigMethod,
				optFmriTrialSigIncludeOtherEvents,
				optFmriTrialSigMaxTrialsPerRun,
				optFmriTrialSigFixedEffectsWeighting,
				optFmriTrialSigWriteConditionBetas,
				optFmriTrialSigWriteTrialBetas,
				optFmriTrialSigWriteTrialVariances,
				optFmriTrialSigSignatureNPS,
				optFmriTrialSigSignatureSIIPS1,
				optFmriAnalysisSignatureDir,
				optFmriTrialSigScopeTrialTypes,
				optFmriTrialSigScopeStimPhases,
				optFmriTrialSigGroupColumn,
				optFmriTrialSigGroupValues,
				optFmriTrialSigGroupScope,
			)
			if trialMethod == "lss" {
				options = append(options, optFmriTrialSigLssOtherRegressors)
			}
		}
	}

	return options
}

// getRawToBidsOptions returns advanced options for raw-to-bids
func (m Model) getRawToBidsOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optRawMontage,
		optRawLineFreq,
		optRawOverwrite,
		optRawTrimToFirstVolume,
		optRawEventPrefixes,
		optRawKeepAnnotations,
	}
}

func (m Model) getFmriRawToBidsOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optFmriRawSession,
		optFmriRawRestTask,
		optFmriRawIncludeRest,
		optFmriRawIncludeFieldmaps,
		optFmriRawDicomMode,
		optFmriRawOverwrite,
		optFmriRawCreateEvents,
		optFmriRawEventGranularity,
		optFmriRawOnsetReference,
		optFmriRawOnsetOffsetS,
		optFmriRawDcm2niixPath,
		optFmriRawDcm2niixArgs,
	}
}

// getMergeBehaviorOptions returns advanced options for merge-behavior
func (m Model) getMergeBehaviorOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optMergeEventPrefixes,
		optMergeEventTypes,
		optMergeQCColumns,
	}
}

type plottingAdvancedRowKind int

const (
	plottingRowOption plottingAdvancedRowKind = iota
	plottingRowSection
	plottingRowPlotHeader
	plottingRowPlotField
	plottingRowPlotInfo
)

type plottingAdvancedRow struct {
	kind plottingAdvancedRowKind

	// kind == plottingRowOption
	opt optionType

	// kind == plottingRowPlotHeader/plottingRowPlotField/plottingRowPlotInfo
	plotID string

	// kind == plottingRowPlotField
	plotField plotItemConfigField

	// kind == plottingRowSection/plottingRowPlotInfo
	label string
}

func (m Model) selectedPlotItemsForConfig() []PlotItem {
	items := make([]PlotItem, 0, len(m.plotItems))
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] {
			continue
		}
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		items = append(items, plot)
	}
	return items
}

func (m Model) plotSupportsComparisons(plot PlotItem) bool {
	// Only plots that actually use aligned_events/events_df for condition
	// comparisons should have comparison configs. Based on Python code analysis.
	switch plot.ID {
	// Aperiodic - uses aligned_events
	case "aperiodic_topomaps",
		"aperiodic_by_condition",
		// Connectivity - uses aligned_events
		"connectivity_by_condition",
		"connectivity_circle_condition",
		"connectivity_network",
		// ERDS - uses aligned_events
		"erds_by_condition",
		// Complexity - uses aligned_events
		"complexity_by_condition",
		// Spectral - uses aligned_events
		"spectral_by_condition",
		// Ratios - uses aligned_events
		"ratios_by_condition",
		// Asymmetry - uses aligned_events
		"asymmetry_by_condition",
		// Bursts - uses aligned_events
		"bursts_by_condition",
		// ITPC - uses aligned_events
		"itpc_by_condition",
		// PAC - uses aligned_events
		"pac_by_condition",
		// Power - uses aligned_events
		"power_by_condition",
		"power_spectral_density",
		// ERP - all use conditions
		"erp_butterfly",
		"erp_roi",
		"erp_contrast",
		// TFR contrast plots - use conditions (column comparisons only, no window comparisons)
		"tfr_scalpmean_contrast",
		"tfr_channels_contrast",
		"tfr_rois_contrast",
		"tfr_topomaps",
		"tfr_band_evolution":
		return true
	}
	return false
}

func (m Model) plotConfigFields(plot PlotItem) []plotItemConfigField {
	if plot.ID == "behavior_temporal_topomaps" {
		return []plotItemConfigField{plotItemConfigFieldBehaviorTemporalStatsFeatureFolder}
	}
	fields := make([]plotItemConfigField, 0, 8)
	if plot.ID == "behavior_dose_response" {
		fields = append(fields,
			plotItemConfigFieldDoseResponseDoseColumn,
			plotItemConfigFieldDoseResponseResponseColumn,
			plotItemConfigFieldDoseResponseSegment,
		)
	}
	if plot.ID == "behavior_pain_probability" {
		fields = append(fields,
			plotItemConfigFieldDoseResponseDoseColumn,
			plotItemConfigFieldDoseResponsePainColumn,
		)
	}
	if plot.ID == "band_power_topomaps" {
		fields = append(fields,
			plotItemConfigFieldTopomapWindow,
			plotItemConfigFieldCompareWindows,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonWindows,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	}
	if plot.ID == "connectivity_circle_condition" {
		fields = append(fields,
			plotItemConfigFieldConnectivityCircleTopFraction,
			plotItemConfigFieldConnectivityCircleMinLines,
			plotItemConfigFieldComparisonSegment,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "connectivity_network" {
		fields = append(fields,
			plotItemConfigFieldConnectivityNetworkTopFraction,
			plotItemConfigFieldComparisonSegment,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "erp_butterfly" {
		// erp_butterfly only uses column comparisons, not window/segment/ROI comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "erp_roi" {
		// erp_roi uses column comparisons and ROI filtering, but not window/segment comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
			plotItemConfigFieldComparisonROIs,
		)
	} else if plot.ID == "erp_contrast" {
		// erp_contrast only uses column comparisons, not window/segment/ROI comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "aperiodic_topomaps" {
		// aperiodic_topomaps only uses column comparisons, not window/segment comparisons
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if plot.ID == "power_spectral_density" {
		// power_spectral_density only uses column comparisons (supports 1+ conditions)
		// CompareColumns toggle not needed - column comparison is always required
		fields = append(fields,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)
	} else if m.plotSupportsComparisons(plot) {
		// TFR plots only use column comparisons, not window comparisons
		isTfrPlot := plot.Group == "tfr"

		if !isTfrPlot {
			// Feature plots can use window comparisons
			fields = append(fields,
				plotItemConfigFieldCompareWindows,
				plotItemConfigFieldComparisonWindows,
			)
			// Feature plots use comparison_segment to specify which time window to compare
			fields = append(fields, plotItemConfigFieldComparisonSegment)
		}

		// Column comparison fields (used by both TFR and feature plots)
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)

		// ROI field only for feature plots that use ROIs in comparisons
		// TFR topomaps are already spatial, so ROIs don't apply
		if !isTfrPlot && plot.ID != "tfr_rois" && plot.ID != "tfr_channels_contrast" && plot.ID != "tfr_scalpmean_contrast" {
			fields = append(fields, plotItemConfigFieldComparisonROIs)
		}
	}
	if plot.ID == "itpc_topomaps" {
		fields = append(fields, plotItemConfigFieldItpcSharedColorbar)
	}
	if plot.ID == "behavior_scatter" {
		fields = append(fields,
			plotItemConfigFieldBehaviorScatterFeatures,
			plotItemConfigFieldBehaviorScatterColumns,
			plotItemConfigFieldBehaviorScatterAggregationModes,
			plotItemConfigFieldBehaviorScatterSegment,
		)
	}
	if plot.ID == "tfr_topomaps" {
		fields = append(fields,
			plotItemConfigFieldTfrTopomapActiveWindow,
			plotItemConfigFieldTfrTopomapWindowSizeMs,
			plotItemConfigFieldTfrTopomapWindowCount,
			plotItemConfigFieldTfrTopomapLabelXPosition,
			plotItemConfigFieldTfrTopomapLabelYPositionBottom,
			plotItemConfigFieldTfrTopomapLabelYPosition,
			plotItemConfigFieldTfrTopomapTitleY,
			plotItemConfigFieldTfrTopomapTitlePad,
			plotItemConfigFieldTfrTopomapSubplotsRight,
			plotItemConfigFieldTfrTopomapTemporalHspace,
			plotItemConfigFieldTfrTopomapTemporalWspace,
		)
	}
	return fields
}

func (m Model) getPlottingAdvancedRows() []plottingAdvancedRow {
	// When using defaults, keep the list to a single actionable row so
	// navigation matches the minimal renderer.
	if m.useDefaultAdvanced {
		return []plottingAdvancedRow{{kind: plottingRowOption, opt: optUseDefaults}}
	}

	rows := make([]plottingAdvancedRow, 0, 128)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowOption, opt: optUseDefaults})

	// Only show per-plot configs for selected plots
	selectedPlots := m.selectedPlotItemsForConfig()
	if len(selectedPlots) == 0 {
		rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotInfo, label: "No plots selected."})
		return rows
	}

	// Per-plot configs only (global styling moved to categories page)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowSection, label: "Plot-Specific Settings"})
	for _, plot := range selectedPlots {
		rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotHeader, plotID: plot.ID})

		fields := m.plotConfigFields(plot)
		if len(fields) == 0 {
			if m.plotItemConfigExpanded[plot.ID] {
				rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotInfo, plotID: plot.ID, label: "No plot-specific settings."})
			}
			continue
		}
		if !m.plotItemConfigExpanded[plot.ID] {
			continue
		}
		for _, f := range fields {
			rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotField, plotID: plot.ID, plotField: f})
		}
	}

	return rows
}

func (m Model) getGlobalStylingOptions() []optionType {
	// Truly global styling options that apply to ALL plots
	options := []optionType{}

	// Defaults & Output
	options = append(options, optPlotGroupDefaults)
	if m.plotGroupDefaultsExpanded {
		options = append(options, optPlotBboxInches, optPlotPadInches)
	}

	// Fonts
	options = append(options, optPlotGroupFonts)
	if m.plotGroupFontsExpanded {
		options = append(options,
			optPlotFontFamily,
			optPlotFontWeight,
			optPlotFontSizeSmall,
			optPlotFontSizeMedium,
			optPlotFontSizeLarge,
			optPlotFontSizeTitle,
			optPlotFontSizeAnnotation,
			optPlotFontSizeLabel,
			optPlotFontSizeYLabel,
			optPlotFontSizeSuptitle,
			optPlotFontSizeFigureTitle,
		)
	}

	// Layout
	options = append(options, optPlotGroupLayout)
	if m.plotGroupLayoutExpanded {
		options = append(options,
			optPlotLayoutTightRect,
			optPlotLayoutTightRectMicrostate,
			optPlotGridSpecWidthRatios,
			optPlotGridSpecHeightRatios,
			optPlotGridSpecHspace,
			optPlotGridSpecWspace,
			optPlotGridSpecLeft,
			optPlotGridSpecRight,
			optPlotGridSpecTop,
			optPlotGridSpecBottom,
		)
	}

	// Figure Sizes
	options = append(options, optPlotGroupFigureSizes)
	if m.plotGroupFigureSizesExpanded {
		options = append(options,
			optPlotFigureSizeStandard,
			optPlotFigureSizeMedium,
			optPlotFigureSizeSmall,
			optPlotFigureSizeSquare,
			optPlotFigureSizeWide,
			optPlotFigureSizeTFR,
			optPlotFigureSizeTopomap,
		)
	}

	// Colors
	options = append(options, optPlotGroupColors)
	if m.plotGroupColorsExpanded {
		options = append(options,
			optPlotColorPain,
			optPlotColorNonpain,
			optPlotColorSignificant,
			optPlotColorNonsignificant,
			optPlotColorGray,
			optPlotColorLightGray,
			optPlotColorBlack,
			optPlotColorBlue,
			optPlotColorRed,
			optPlotColorNetworkNode,
		)
	}

	// Alpha
	options = append(options, optPlotGroupAlpha)
	if m.plotGroupAlphaExpanded {
		options = append(options,
			optPlotAlphaGrid,
			optPlotAlphaFill,
			optPlotAlphaCI,
			optPlotAlphaCILine,
			optPlotAlphaTextBox,
			optPlotAlphaViolinBody,
			optPlotAlphaRidgeFill,
		)
	}

	// Topomap
	options = append(options, optPlotGroupTopomap)
	if m.plotGroupTopomapExpanded {
		options = append(options,
			optPlotTopomapContours,
			optPlotTopomapColormap,
			optPlotTopomapColorbarFraction,
			optPlotTopomapColorbarPad,
			optPlotTopomapDiffAnnotation,
			optPlotTopomapAnnotateDescriptive,
			optPlotTopomapSigMaskMarker,
			optPlotTopomapSigMaskMarkerFaceColor,
			optPlotTopomapSigMaskMarkerEdgeColor,
			optPlotTopomapSigMaskLinewidth,
			optPlotTopomapSigMaskMarkersize,
		)
	}

	// TFR
	options = append(options, optPlotGroupTFR)
	if m.plotGroupTFRExpanded {
		options = append(options,
			optPlotTFRLogBase,
			optPlotTFRPercentageMultiplier,
			optPlotTFRTopomapWindowSizeMs,
			optPlotTFRTopomapWindowCount,
			optPlotTFRTopomapLabelXPosition,
			optPlotTFRTopomapLabelYPositionBottom,
			optPlotTFRTopomapLabelYPosition,
			optPlotTFRTopomapTitleY,
			optPlotTFRTopomapTitlePad,
			optPlotTFRTopomapSubplotsRight,
			optPlotTFRTopomapTemporalHspace,
			optPlotTFRTopomapTemporalWspace,
		)
	}

	return options
}

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	// Check if any computation is selected
	hasAnyComputation := len(m.SelectedComputations()) > 0

	// General section - only show if at least one computation is selected
	if hasAnyComputation {
		options = append(options, optBehaviorGroupGeneral)
		if m.behaviorGroupGeneralExpanded {
			// RNG Seed and N Jobs are always relevant
			options = append(options, optRNGSeed, optBehaviorNJobs, optBehaviorMinSamples)

			// Correlation method and robust correlation - only for correlations, stability, pain_sensitivity
			needsCorrelationMethod := m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability") ||
				m.isComputationSelected("pain_sensitivity")
			if needsCorrelationMethod {
				options = append(options, optCorrMethod, optRobustCorrelation)
			}

			// Bootstrap - relevant for correlations, stability
			needsBootstrap := m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability")
			if needsBootstrap {
				options = append(options, optBootstrap)
			}

			// FDR Alpha - relevant for correlations, condition, temporal, cluster, regression
			needsFDR := m.isComputationSelected("correlations") ||
				m.isComputationSelected("condition") ||
				m.isComputationSelected("temporal") ||
				m.isComputationSelected("cluster") ||
				m.isComputationSelected("regression")
			if needsFDR {
				options = append(options, optFDRAlpha)
			}

			// N Permutations - relevant for cluster, temporal, regression, correlations, mediation, moderation
			needsPermutations := m.isComputationSelected("cluster") ||
				m.isComputationSelected("temporal") ||
				m.isComputationSelected("regression") ||
				m.isComputationSelected("correlations") ||
				m.isComputationSelected("mediation") ||
				m.isComputationSelected("moderation")
			if needsPermutations {
				options = append(options, optNPerm)
			}

			// Covariate controls - relevant for regression, models, influence, correlations, stability, pain_sensitivity
			needsCovariates := m.isComputationSelected("regression") ||
				m.isComputationSelected("models") ||
				m.isComputationSelected("influence") ||
				m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability") ||
				m.isComputationSelected("pain_sensitivity")
			if needsCovariates {
				options = append(options, optControlTemp, optControlOrder)
			}

			// Run adjustment - relevant for trial_table, correlations
			needsRunAdjustment := m.isComputationSelected("trial_table") ||
				m.isComputationSelected("correlations")
			if needsRunAdjustment {
				options = append(options,
					optRunAdjustmentEnabled,
					optRunAdjustmentColumn,
					optRunAdjustmentIncludeInCorrelations,
					optRunAdjustmentMaxDummies,
				)
			}

			// Change scores, LOSO stability, Bayes factors - relevant for correlations
			if m.isComputationSelected("correlations") {
				options = append(options,
					optComputeChangeScores,
					optComputeLosoStability,
					optComputeBayesFactors,
				)
			}

			// Feature QC (optional gating) - relevant for correlations / multilevel correlations
			if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
				options = append(options,
					optFeatureQCEnabled,
					optFeatureQCMaxMissingPct,
					optFeatureQCMinVariance,
					optFeatureQCCheckWithinRunVariance,
				)
			}
		}
	}

	// Trial table section - only show if trial_table computation is selected
	if m.isComputationSelected("trial_table") {
		options = append(options, optBehaviorGroupTrialTable)
		if m.behaviorGroupTrialTableExpanded {
			options = append(options,
				optTrialTableFormat,
				optTrialTableAddLagFeatures,
				optTrialOrderMaxMissingFraction,
				optFeatureSummariesEnabled,
			)
		}
	}

	// Pain Residual section - only show if pain_residual computation is selected
	if m.isComputationSelected("pain_residual") {
		options = append(options, optBehaviorGroupPainResidual)
		if m.behaviorGroupPainResidualExpanded {
			options = append(options,
				optPainResidualEnabled,
				optPainResidualMethod,
				optPainResidualPolyDegree,
				optPainResidualSplineDfCandidates,
				optPainResidualMinSamples,
				optPainResidualModelCompare,
				optPainResidualModelComparePolyDegrees,
				optPainResidualModelCompareMinSamples,
				optPainResidualBreakpoint,
				optPainResidualBreakpointCandidates,
				optPainResidualBreakpointMinSamples,
				optPainResidualBreakpointQlow,
				optPainResidualBreakpointQhigh,
				optPainResidualCrossfitEnabled,
				optPainResidualCrossfitGroupColumn,
				optPainResidualCrossfitNSplits,
				optPainResidualCrossfitMethod,
				optPainResidualCrossfitSplineKnots,
			)
		}
	}

	// Correlations section
	if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
		options = append(options, optBehaviorGroupCorrelations)
		if m.behaviorGroupCorrelationsExpanded {
			if m.isComputationSelected("correlations") {
				options = append(options,
					optCorrelationsTargetRating,
					optCorrelationsTargetTemperature,
					optCorrelationsTargetPainResidual,
					optCorrelationsTargetColumn,
					optCorrelationsPreferPainResidual,
					optCorrelationsTypes,
					optCorrelationsPrimaryUnit,
					optCorrelationsPermutationPrimary,
					optCorrelationsUseCrossfitPainResidual,
				)
			}
			options = append(options, optCorrelationsMultilevel)
			if m.isComputationSelected("multilevel_correlations") {
				options = append(options, optGroupLevelBlockPermutation)
			}
		}
	}

	// Regression section (includes model sensitivity options)
	if m.isComputationSelected("regression") {
		options = append(options, optBehaviorGroupRegression)
		if m.behaviorGroupRegressionExpanded {
			options = append(options,
				optRegressionOutcome,
				optRegressionIncludeTemperature,
				optRegressionTempControl,
			)
			if m.regressionTempControl == 2 {
				options = append(options,
					optRegressionTempSplineKnots,
					optRegressionTempSplineQlow,
					optRegressionTempSplineQhigh,
					optRegressionTempSplineMinN,
				)
			}
			options = append(options,
				optRegressionIncludeTrialOrder,
				optRegressionIncludePrev,
				optRegressionIncludeRunBlock,
				optRegressionIncludeInteraction,
				optRegressionStandardize,
				optRegressionMinSamples,
				optRegressionPermutations,
				optRegressionMaxFeatures,
			)
			// Model sensitivity options (now part of regression)
			options = append(options,
				optModelsFamilyOLS,
				optModelsFamilyRobust,
				optModelsFamilyQuantile,
				optModelsFamilyLogit,
			)
		}
	}

	// Models section
	if m.isComputationSelected("models") {
		options = append(options, optBehaviorGroupModels)
		if m.behaviorGroupModelsExpanded {
			options = append(options,
				optModelsIncludeTemperature,
				optModelsTempControl,
			)
			if m.modelsTempControl == 2 {
				options = append(options,
					optModelsTempSplineKnots,
					optModelsTempSplineQlow,
					optModelsTempSplineQhigh,
					optModelsTempSplineMinN,
				)
			}
			options = append(options,
				optModelsIncludeTrialOrder,
				optModelsIncludePrev,
				optModelsIncludeRunBlock,
				optModelsIncludeInteraction,
				optModelsStandardize,
				optModelsMinSamples,
				optModelsMaxFeatures,
				optModelsOutcomeRating,
				optModelsOutcomePainResidual,
				optModelsOutcomeTemperature,
				optModelsOutcomePainBinary,
				optModelsFamilyOLS,
				optModelsFamilyRobust,
				optModelsFamilyQuantile,
				optModelsFamilyLogit,
				optModelsBinaryOutcome,
			)
		}
	}

	// Stability section
	if m.isComputationSelected("stability") {
		options = append(options, optBehaviorGroupStability)
		if m.behaviorGroupStabilityExpanded {
			options = append(options,
				optStabilityMethod,
				optStabilityOutcome,
				optStabilityGroupColumn,
				optStabilityPartialTemp,
				optStabilityMinGroupTrials,
				optStabilityMaxFeatures,
				optStabilityAlpha,
			)
		}
	}

	// Consistency section
	if m.isComputationSelected("consistency") {
		options = append(options, optBehaviorGroupConsistency)
		if m.behaviorGroupConsistencyExpanded {
			options = append(options, optConsistencyEnabled)
		}
	}

	// Influence section
	if m.isComputationSelected("influence") {
		options = append(options, optBehaviorGroupInfluence)
		if m.behaviorGroupInfluenceExpanded {
			options = append(options,
				optInfluenceOutcomeRating,
				optInfluenceOutcomePainResidual,
				optInfluenceOutcomeTemperature,
				optInfluenceMaxFeatures,
				optInfluenceIncludeTemperature,
				optInfluenceTempControl,
			)
			if m.influenceTempControl == 2 {
				options = append(options,
					optInfluenceTempSplineKnots,
					optInfluenceTempSplineQlow,
					optInfluenceTempSplineQhigh,
					optInfluenceTempSplineMinN,
				)
			}
			options = append(options,
				optInfluenceIncludeTrialOrder,
				optInfluenceIncludeRunBlock,
				optInfluenceIncludeInteraction,
				optInfluenceStandardize,
				optInfluenceCooksThreshold,
				optInfluenceLeverageThreshold,
			)
		}
	}

	// Report section
	if m.isComputationSelected("report") {
		options = append(options, optBehaviorGroupReport)
		if m.behaviorGroupReportExpanded {
			options = append(options, optReportTopN)
		}
	}

	// Pain sensitivity section
	if m.isComputationSelected("pain_sensitivity") {
		options = append(options, optBehaviorGroupPainSens)
		if m.behaviorGroupPainSensExpanded {
			options = append(options, optPainSensitivityMinTrials)
		}
	}

	// Condition section
	if m.isComputationSelected("condition") {
		options = append(options, optBehaviorGroupCondition)
		if m.behaviorGroupConditionExpanded {
			options = append(options,
				optConditionCompareColumn,
				optConditionCompareValues,
				optConditionCompareWindows,
				optConditionMinTrials,
				optConditionWindowPrimaryUnit,
				optConditionPermutationPrimary,
				optConditionFailFast,
				optConditionEffectThreshold,
				optConditionOverwrite,
			)
		}
	}

	// Temporal section
	if m.isComputationSelected("temporal") {
		options = append(options, optBehaviorGroupTemporal)
		if m.behaviorGroupTemporalExpanded {
			options = append(options,
				optTemporalResolutionMs,
				optTemporalTimeMinMs,
				optTemporalTimeMaxMs,
				optTemporalSmoothMs,
				optTemporalTargetColumn,
				optTemporalSplitByCondition,
				optTemporalConditionColumn,
				optTemporalConditionValues,
				optTemporalIncludeROIAverages,
				optTemporalIncludeTFGrid,
			)
			// Show ITPC-specific options when 'itpc' is selected in step 3 (feature selection)
			if m.featureFileSelected["itpc"] {
				options = append(options,
					optTemporalITPCBaselineCorrection,
					optTemporalITPCBaselineMin,
					optTemporalITPCBaselineMax,
				)
			}
			// Show ERDS-specific options when 'erds' is selected in step 3 (feature selection)
			if m.featureFileSelected["erds"] {
				options = append(options,
					optTemporalERDSBaselineMin,
					optTemporalERDSBaselineMax,
					optTemporalERDSMethod,
				)
			}
			// TF Heatmap options (always visible when temporal is expanded)
			options = append(options,
				optTemporalTfHeatmapEnabled,
			)
			if m.tfHeatmapEnabled {
				options = append(options,
					optTemporalTfHeatmapFreqs,
					optTemporalTfHeatmapTimeResMs,
				)
			}
		}
	}

	// Cluster section
	if m.isComputationSelected("cluster") {
		options = append(options, optBehaviorGroupCluster)
		if m.behaviorGroupClusterExpanded {
			options = append(
				options,
				optClusterThreshold,
				optClusterMinSize,
				optClusterTail,
				optClusterConditionColumn,
				optClusterConditionValues,
			)
		}
	}

	// Mediation section
	if m.isComputationSelected("mediation") {
		options = append(options, optBehaviorGroupMediation)
		if m.behaviorGroupMediationExpanded {
			options = append(options, optMediationBootstrap, optMediationPermutations, optMediationMinEffect, optMediationMaxMediatorsEnabled, optMediationMaxMediators)
		}
	}

	// Moderation section
	if m.isComputationSelected("moderation") {
		options = append(options, optBehaviorGroupModeration)
		if m.behaviorGroupModerationExpanded {
			options = append(options, optModerationMaxFeaturesEnabled, optModerationMaxFeatures, optModerationMinSamples, optModerationPermutations)
		}
	}

	// Mixed effects section
	if m.isComputationSelected("mixed_effects") {
		options = append(options, optBehaviorGroupMixedEffects)
		if m.behaviorGroupMixedEffectsExpanded {
			options = append(options, optMixedEffectsType, optMixedMaxFeatures)
		}
	}

	// Output section - only show if at least one computation is selected
	if hasAnyComputation {
		options = append(options, optBehaviorGroupOutput)
		if m.behaviorGroupOutputExpanded {
			options = append(options, optAlsoSaveCsv, optBehaviorOverwrite)
		}
	}

	return options
}

func (m Model) getPlotConfigOptions() []optionType {
	options := []optionType{
		optPlotPNG,
		optPlotSVG,
		optPlotPDF,
		optPlotDPI,
		optPlotSaveDPI,
		optPlotOverwrite,
	}

	// Dynamic options based on selected plots/categories
	if m.IsPlotCategorySelected("tfr") || m.IsPlotCategorySelected("features") {
		// ITPC and PAC settings
		options = append(options, optPlotSharedColorbar)
	}

	return options
}

func (m Model) getMLOptions() []optionType {
	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}

	opts := []optionType{
		optUseDefaults,
		optMLTarget,
		optMLFeatureFamilies,
		optMLFeatureBands,
		optMLFeatureSegments,
		optMLFeatureScopes,
		optMLFeatureStats,
		optMLFeatureHarmonization,
		optMLCovariates,
	}

	if mode == "incremental_validity" {
		opts = append(opts, optMLBaselinePredictors)
	}

	if strings.EqualFold(strings.TrimSpace(m.mlTarget), "fmri_signature") {
		opts = append(opts, optMLFmriSigGroup)
		if m.mlFmriSigGroupExpanded {
			opts = append(opts,
				optMLFmriSigMethod,
				optMLFmriSigContrastName,
				optMLFmriSigSignature,
				optMLFmriSigMetric,
				optMLFmriSigNormalization,
				optMLFmriSigRoundDecimals,
			)
		}
	}

	opts = append(opts, optMLRequireTrialMlSafe)

	if mode == "classify" {
		opts = append(opts, optMLClassificationModel, optMLBinaryThresholdEnabled)
		if m.mlBinaryThresholdEnabled {
			opts = append(opts, optMLBinaryThreshold)
		}
		opts = append(opts, optVarianceThresholdGrid)
	} else if mode != "timegen" && mode != "" {
		// Most non-classification stages use the regression model family (timegen is separate).
		if mode != "model_comparison" {
			opts = append(opts, optMLRegressionModel)
		}
		opts = append(
			opts,
			optElasticNetAlphaGrid,
			optElasticNetL1RatioGrid,
			optRidgeAlphaGrid,
			optRfNEstimators,
			optRfMaxDepthGrid,
			optVarianceThresholdGrid,
		)
	}

	opts = append(opts, optMLNPerm, optMLInnerSplits, optMLOuterJobs, optRNGSeed)

	if mode == "uncertainty" {
		opts = append(opts, optMLUncertaintyAlpha)
	}
	if mode == "permutation" {
		opts = append(opts, optMLPermNRepeats)
	}

	return opts
}

func (m Model) isCurrentlyEditing(opt optionType) bool {
	if !m.editingNumber {
		return false
	}

	// Plotting advanced config uses a mixed row model (per-plot + global options),
	// so the cursor no longer indexes directly into getPlottingOptions().
	if m.Pipeline == types.PipelinePlotting {
		rows := m.getPlottingAdvancedRows()
		if m.advancedCursor < 0 || m.advancedCursor >= len(rows) {
			return false
		}
		return rows[m.advancedCursor].kind == plottingRowOption && rows[m.advancedCursor].opt == opt
	}

	var options []optionType
	switch m.Pipeline {
	case types.PipelineFeatures:
		options = m.getFeaturesOptions()
	case types.PipelineBehavior:
		options = m.getBehaviorOptions()
	case types.PipelineML:
		options = m.getMLOptions()
	case types.PipelinePreprocessing:
		options = m.getPreprocessingOptions()
	case types.PipelineRawToBIDS:
		options = m.getRawToBidsOptions()
	case types.PipelineFmriRawToBIDS:
		options = m.getFmriRawToBidsOptions()
	case types.PipelineFmri:
		options = m.getFmriPreprocessingOptions()
	case types.PipelineFmriAnalysis:
		options = m.getFmriAnalysisOptions()
	default:
		return false
	}
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return false
	}
	return options[m.advancedCursor] == opt
}
