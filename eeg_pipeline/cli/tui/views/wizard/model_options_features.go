package wizard

import (
	"strings"
)

// Features advanced-option list builders.

func (m Model) getFeaturesOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)
	options = append(options, optConfigSetOverrides)

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
				optMicrostatesAssignFromGfpPeaks,
				optMicrostatesFixedTemplatesPath,
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
				optERDSOnsetThresholdSigma,
				optERDSOnsetMinDurationMs,
				optERDSReboundMinLatencyMs,
				optERDSInferContralateral,
				optERDSConditionMarkerBands,
				optERDSLateralityColumns,
				optERDSSomatosensoryLeftChannels,
				optERDSSomatosensoryRightChannels,
				optERDSOnsetMinThresholdPercent,
				optERDSReboundThresholdSigma,
				optERDSReboundMinThresholdPercent,
			)
		}
	}

	// Spatial transform (for volume conduction reduction) - useful when connectivity is selected
	if m.isCategorySelected("connectivity") {
		options = append(options, optFeatGroupSpatialTransform)
		if m.featGroupSpatialTransformExpanded {
			options = append(options, optSpatialTransform, optSpatialTransformLambda2, optSpatialTransformStiffness)
			// Per-family spatial transform overrides: only show selected families.
			if m.isCategorySelected("connectivity") {
				options = append(options, optSpatialTransformPerFamilyConnectivity)
			}
			if m.isCategorySelected("itpc") {
				options = append(options, optSpatialTransformPerFamilyItpc)
			}
			if m.isCategorySelected("pac") {
				options = append(options, optSpatialTransformPerFamilyPac)
			}
			if m.isCategorySelected("power") {
				options = append(options, optSpatialTransformPerFamilyPower)
			}
			if m.isCategorySelected("aperiodic") {
				options = append(options, optSpatialTransformPerFamilyAperiodic)
			}
			if m.isCategorySelected("bursts") {
				options = append(options, optSpatialTransformPerFamilyBursts)
			}
			if m.isCategorySelected("erds") {
				options = append(options, optSpatialTransformPerFamilyErds)
			}
			if m.isCategorySelected("complexity") {
				options = append(options, optSpatialTransformPerFamilyComplexity)
			}
			if m.isCategorySelected("ratios") {
				options = append(options, optSpatialTransformPerFamilyRatios)
			}
			if m.isCategorySelected("asymmetry") {
				options = append(options, optSpatialTransformPerFamilyAsymmetry)
			}
			if m.isCategorySelected("spectral") {
				options = append(options, optSpatialTransformPerFamilySpectral)
			}
			if m.isCategorySelected("erp") {
				options = append(options, optSpatialTransformPerFamilyErp)
			}
			if m.isCategorySelected("quality") {
				options = append(options, optSpatialTransformPerFamilyQuality)
			}
			if m.isCategorySelected("microstates") {
				options = append(options, optSpatialTransformPerFamilyMicrostates)
			}
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
						options = append(
							options,
							optSourceLocFmriHrfModel,
							optSourceLocFmriDriftModel,
							optSourceLocFmriConditionScopeColumn,
							optSourceLocFmriConditionScopeTrialTypes,
							optSourceLocFmriPhaseColumn,
							optSourceLocFmriPhaseScopeColumn,
							optSourceLocFmriPhaseScopeValue,
							optSourceLocFmriStimPhasesToModel,
						)
						options = append(options, optSourceLocFmriInputSource, optSourceLocFmriRequireFmriprep)
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
			options = append(options, optItpcMinSegmentSec, optItpcMinCyclesAtFmin)
		}
	}

	// PAC segment validity (shown when PAC is selected)
	if m.isCategorySelected("pac") {
		options = append(options, optPACMinSegmentSec, optPACMinCyclesAtFmin, optPACSurrogateMethod)
	}

	// TFR settings only for feature families that use time-frequency representations.
	if m.hasTimeFrequencyFeatureSelection() {
		options = append(options, optFeatGroupTFR)
		if m.featGroupTFRExpanded {
			options = append(
				options,
				optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrMaxCycles, optTfrNCyclesFactor, optTfrDecimPower, optTfrDecimPhase, optTfrWorkers,
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
		options = append(options, optFeatAlsoSaveCsv)
	}

	// Directed connectivity missing
	if m.isCategorySelected("directedconnectivity") {
		options = append(options, optDirectedConnMinSamplesPerMvarParam)
	}

	// Aperiodic missing
	if m.isCategorySelected("aperiodic") {
		options = append(options, optAperiodicMaxFreqResolutionHz, optAperiodicMultitaperAdaptive)
	}

	options = append(options, optFeatGroupExecution)
	if m.featGroupExecutionExpanded {
		options = append(
			options,
			optMinEpochs,
			optFeatAnalysisMode,
			optAggregationMethod,
			optFeatureTmin,
			optFeatureTmax,
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

	// Change scores config (shown when change scores are enabled)
	if m.featComputeChangeScores {
		options = append(options, optChangeScoresTransform, optChangeScoresWindowPairs)
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
