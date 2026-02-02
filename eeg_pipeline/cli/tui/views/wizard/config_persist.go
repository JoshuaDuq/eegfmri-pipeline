package wizard

import "encoding/json"

// ExportConfig exports all advanced configuration options to a map for persistence.
// This includes all pipeline-specific settings (features, behavior, preprocessing, fMRI, plotting, ML).
func (m Model) ExportConfig() map[string]interface{} {
	cfg := make(map[string]interface{})

	// UI State - Selections
	cfg["modeIndex"] = m.modeIndex
	cfg["computationSelected"] = mapToIntList(m.computationSelected)
	cfg["categorySelected"] = mapToIntList(m.selected)
	cfg["prepStageSelected"] = mapToIntList(m.prepStageSelected)
	cfg["featureFileSelected"] = stringsMapToBoolMap(m.featureFileSelected)
	cfg["plotSelected"] = mapToIntList(m.plotSelected)
	cfg["featurePlotterSelected"] = stringsMapToBoolMap(m.featurePlotterSelected)
	cfg["plotFormatSelected"] = stringsMapToBoolMap(m.plotFormatSelected)
	cfg["connectivityMeasures"] = mapToIntList(m.connectivityMeasures)
	cfg["directedConnMeasures"] = mapToIntList(m.directedConnMeasures)

	// UI State - Group Expansion
	cfg["featGroupConnectivityExpanded"] = m.featGroupConnectivityExpanded
	cfg["featGroupPACExpanded"] = m.featGroupPACExpanded
	cfg["featGroupAperiodicExpanded"] = m.featGroupAperiodicExpanded
	cfg["featGroupComplexityExpanded"] = m.featGroupComplexityExpanded
	cfg["featGroupBurstsExpanded"] = m.featGroupBurstsExpanded
	cfg["featGroupPowerExpanded"] = m.featGroupPowerExpanded
	cfg["featGroupSpectralExpanded"] = m.featGroupSpectralExpanded
	cfg["featGroupERPExpanded"] = m.featGroupERPExpanded
	cfg["featGroupRatiosExpanded"] = m.featGroupRatiosExpanded
	cfg["featGroupAsymmetryExpanded"] = m.featGroupAsymmetryExpanded
	cfg["featGroupQualityExpanded"] = m.featGroupQualityExpanded
	cfg["featGroupERDSExpanded"] = m.featGroupERDSExpanded
	cfg["featGroupStorageExpanded"] = m.featGroupStorageExpanded
	cfg["featGroupExecutionExpanded"] = m.featGroupExecutionExpanded
	cfg["featGroupDirectedConnExpanded"] = m.featGroupDirectedConnExpanded
	cfg["featGroupSourceLocExpanded"] = m.featGroupSourceLocExpanded
	cfg["featGroupSourceLocFmriExpanded"] = m.featGroupSourceLocFmriExpanded
	cfg["featGroupSourceLocContrastExpanded"] = m.featGroupSourceLocContrastExpanded
	cfg["featGroupSourceLocGLMExpanded"] = m.featGroupSourceLocGLMExpanded
	cfg["featGroupTFRExpanded"] = m.featGroupTFRExpanded

	// fMRI UI State
	cfg["fmriGroupRuntimeExpanded"] = m.fmriGroupRuntimeExpanded
	cfg["fmriGroupOutputExpanded"] = m.fmriGroupOutputExpanded
	cfg["fmriGroupPerformanceExpanded"] = m.fmriGroupPerformanceExpanded
	cfg["fmriGroupAnatomicalExpanded"] = m.fmriGroupAnatomicalExpanded
	cfg["fmriGroupBoldExpanded"] = m.fmriGroupBoldExpanded
	cfg["fmriGroupQcExpanded"] = m.fmriGroupQcExpanded
	cfg["fmriGroupDenoisingExpanded"] = m.fmriGroupDenoisingExpanded
	cfg["fmriGroupSurfaceExpanded"] = m.fmriGroupSurfaceExpanded
	cfg["fmriGroupMultiechoExpanded"] = m.fmriGroupMultiechoExpanded
	cfg["fmriGroupReproExpanded"] = m.fmriGroupReproExpanded
	cfg["fmriGroupValidationExpanded"] = m.fmriGroupValidationExpanded
	cfg["fmriGroupAdvancedExpanded"] = m.fmriGroupAdvancedExpanded

	// Plotting UI State
	cfg["plotGroupDefaultsExpanded"] = m.plotGroupDefaultsExpanded
	cfg["plotGroupFontsExpanded"] = m.plotGroupFontsExpanded
	cfg["plotGroupLayoutExpanded"] = m.plotGroupLayoutExpanded
	cfg["plotGroupFigureSizesExpanded"] = m.plotGroupFigureSizesExpanded
	cfg["plotGroupColorsExpanded"] = m.plotGroupColorsExpanded
	cfg["plotGroupAlphaExpanded"] = m.plotGroupAlphaExpanded
	cfg["plotGroupScatterExpanded"] = m.plotGroupScatterExpanded
	cfg["plotGroupBarExpanded"] = m.plotGroupBarExpanded
	cfg["plotGroupLineExpanded"] = m.plotGroupLineExpanded
	cfg["plotGroupHistogramExpanded"] = m.plotGroupHistogramExpanded
	cfg["plotGroupTopomapExpanded"] = m.plotGroupTopomapExpanded
	cfg["plotGroupTFRExpanded"] = m.plotGroupTFRExpanded
	cfg["plotGroupSizingExpanded"] = m.plotGroupSizingExpanded
	cfg["plotGroupSelectionExpanded"] = m.plotGroupSelectionExpanded
	cfg["plotGroupComparisonsExpanded"] = m.plotGroupComparisonsExpanded
	cfg["plotGroupKDEExpanded"] = m.plotGroupKDEExpanded
	cfg["plotGroupErrorbarExpanded"] = m.plotGroupErrorbarExpanded
	cfg["plotGroupTextExpanded"] = m.plotGroupTextExpanded
	cfg["plotGroupValidationExpanded"] = m.plotGroupValidationExpanded
	cfg["plotGroupTFRMiscExpanded"] = m.plotGroupTFRMiscExpanded
	// Plotting per-plot config + UI expansion state
	cfg["plotItemConfigs"] = m.plotItemConfigs
	cfg["plotItemConfigExpanded"] = stringsMapToBoolMap(m.plotItemConfigExpanded)

	// PAC/CFC configuration
	cfg["pacPhaseMin"] = m.pacPhaseMin
	cfg["pacPhaseMax"] = m.pacPhaseMax
	cfg["pacAmpMin"] = m.pacAmpMin
	cfg["pacAmpMax"] = m.pacAmpMax
	cfg["pacMethod"] = m.pacMethod
	cfg["pacMinEpochs"] = m.pacMinEpochs
	cfg["pacPairsSpec"] = m.pacPairsSpec
	cfg["pacSource"] = m.pacSource
	cfg["pacNormalize"] = m.pacNormalize
	cfg["pacNSurrogates"] = m.pacNSurrogates
	cfg["pacRandomSeed"] = m.pacRandomSeed
	cfg["pacAllowHarmonicOvrlap"] = m.pacAllowHarmonicOvrlap
	cfg["pacMaxHarmonic"] = m.pacMaxHarmonic
	cfg["pacHarmonicToleranceHz"] = m.pacHarmonicToleranceHz
	cfg["pacComputeWaveformQC"] = m.pacComputeWaveformQC
	cfg["pacWaveformOffsetMs"] = m.pacWaveformOffsetMs

	// Aperiodic configuration
	cfg["aperiodicFmin"] = m.aperiodicFmin
	cfg["aperiodicFmax"] = m.aperiodicFmax
	cfg["aperiodicPeakZ"] = m.aperiodicPeakZ
	cfg["aperiodicMinR2"] = m.aperiodicMinR2
	cfg["aperiodicMinPoints"] = m.aperiodicMinPoints
	cfg["aperiodicMinSegmentSec"] = m.aperiodicMinSegmentSec
	cfg["aperiodicModel"] = m.aperiodicModel
	cfg["aperiodicPsdMethod"] = m.aperiodicPsdMethod
	cfg["aperiodicSubtractEvoked"] = m.aperiodicSubtractEvoked
	cfg["aperiodicPsdBandwidth"] = m.aperiodicPsdBandwidth
	cfg["aperiodicMaxRms"] = m.aperiodicMaxRms
	cfg["aperiodicExcludeLineNoise"] = m.aperiodicExcludeLineNoise
	cfg["aperiodicLineNoiseFreq"] = m.aperiodicLineNoiseFreq
	cfg["aperiodicLineNoiseWidthHz"] = m.aperiodicLineNoiseWidthHz
	cfg["aperiodicLineNoiseHarmonics"] = m.aperiodicLineNoiseHarmonics

	// Complexity configuration
	cfg["complexityPEOrder"] = m.complexityPEOrder
	cfg["complexityPEDelay"] = m.complexityPEDelay
	cfg["complexitySignalBasis"] = m.complexitySignalBasis
	cfg["complexityMinSegmentSec"] = m.complexityMinSegmentSec
	cfg["complexityMinSamples"] = m.complexityMinSamples
	cfg["complexityZscore"] = m.complexityZscore

	// ERP configuration
	cfg["erpBaselineCorrection"] = m.erpBaselineCorrection
	cfg["erpAllowNoBaseline"] = m.erpAllowNoBaseline
	cfg["erpComponentsSpec"] = m.erpComponentsSpec
	cfg["erpSmoothMs"] = m.erpSmoothMs
	cfg["erpPeakProminenceUv"] = m.erpPeakProminenceUv
	cfg["erpLowpassHz"] = m.erpLowpassHz

	// Burst configuration
	cfg["burstThresholdZ"] = m.burstThresholdZ
	cfg["burstThresholdMethod"] = m.burstThresholdMethod
	cfg["burstThresholdPercentile"] = m.burstThresholdPercentile
	cfg["burstThresholdReference"] = m.burstThresholdReference
	cfg["burstMinTrialsPerCondition"] = m.burstMinTrialsPerCondition
	cfg["burstMinSegmentSec"] = m.burstMinSegmentSec
	cfg["burstSkipInvalidSegments"] = m.burstSkipInvalidSegments
	cfg["burstMinDuration"] = m.burstMinDuration
	cfg["burstMinCycles"] = m.burstMinCycles
	cfg["burstBandsSpec"] = m.burstBandsSpec

	// Power/Spectral configuration
	cfg["powerBaselineMode"] = m.powerBaselineMode
	cfg["powerRequireBaseline"] = m.powerRequireBaseline
	cfg["powerSubtractEvoked"] = m.powerSubtractEvoked
	cfg["powerMinTrialsPerCondition"] = m.powerMinTrialsPerCondition
	cfg["powerExcludeLineNoise"] = m.powerExcludeLineNoise
	cfg["powerLineNoiseFreq"] = m.powerLineNoiseFreq
	cfg["powerLineNoiseWidthHz"] = m.powerLineNoiseWidthHz
	cfg["powerLineNoiseHarmonics"] = m.powerLineNoiseHarmonics
	cfg["powerEmitDb"] = m.powerEmitDb
	cfg["spectralEdgePercentile"] = m.spectralEdgePercentile
	cfg["spectralRatioPairsSpec"] = m.spectralRatioPairsSpec
	cfg["spectralPsdAdaptive"] = m.spectralPsdAdaptive
	cfg["spectralMultitaperAdaptive"] = m.spectralMultitaperAdaptive
	cfg["spectralPsdMethod"] = m.spectralPsdMethod
	cfg["spectralFmin"] = m.spectralFmin
	cfg["spectralFmax"] = m.spectralFmax
	cfg["spectralIncludeLogRatios"] = m.spectralIncludeLogRatios
	cfg["spectralExcludeLineNoise"] = m.spectralExcludeLineNoise
	cfg["spectralLineNoiseFreq"] = m.spectralLineNoiseFreq
	cfg["spectralLineNoiseWidthHz"] = m.spectralLineNoiseWidthHz
	cfg["spectralLineNoiseHarmonics"] = m.spectralLineNoiseHarmonics
	cfg["spectralMinSegmentSec"] = m.spectralMinSegmentSec
	cfg["spectralMinCyclesAtFmin"] = m.spectralMinCyclesAtFmin

	// Connectivity configuration
	cfg["connOutputLevel"] = m.connOutputLevel
	cfg["connGraphMetrics"] = m.connGraphMetrics
	cfg["connGraphProp"] = m.connGraphProp
	cfg["connWindowLen"] = m.connWindowLen
	cfg["connWindowStep"] = m.connWindowStep
	cfg["connAECMode"] = m.connAECMode
	cfg["connGranularity"] = m.connGranularity
	cfg["connConditionColumn"] = m.connConditionColumn
	cfg["connConditionValues"] = m.connConditionValues
	cfg["connMinEpochsPerGroup"] = m.connMinEpochsPerGroup
	cfg["connMinCyclesPerBand"] = m.connMinCyclesPerBand
	cfg["connWarnNoSpatialTransform"] = m.connWarnNoSpatialTransform
	cfg["connPhaseEstimator"] = m.connPhaseEstimator
	cfg["connMinSegmentSec"] = m.connMinSegmentSec
	cfg["connAECOutput"] = m.connAECOutput
	cfg["connForceWithinEpochML"] = m.connForceWithinEpochML

	// ITPC configuration
	cfg["itpcMethod"] = m.itpcMethod
	cfg["itpcConditionColumn"] = m.itpcConditionColumn
	cfg["itpcConditionValues"] = m.itpcConditionValues
	cfg["itpcMinTrialsPerCondition"] = m.itpcMinTrialsPerCondition
	cfg["itpcAllowUnsafeLoo"] = m.itpcAllowUnsafeLoo
	cfg["itpcBaselineCorrection"] = m.itpcBaselineCorrection
	cfg["itpcNJobs"] = m.itpcNJobs

	// Source localization
	cfg["sourceLocEnabled"] = m.sourceLocEnabled
	cfg["sourceLocMode"] = m.sourceLocMode
	cfg["sourceLocMethod"] = m.sourceLocMethod
	cfg["sourceLocSpacing"] = m.sourceLocSpacing
	cfg["sourceLocParc"] = m.sourceLocParc
	cfg["sourceLocReg"] = m.sourceLocReg
	cfg["sourceLocSnr"] = m.sourceLocSnr
	cfg["sourceLocLoose"] = m.sourceLocLoose
	cfg["sourceLocDepth"] = m.sourceLocDepth
	cfg["sourceLocConnMethod"] = m.sourceLocConnMethod
	cfg["sourceLocSubject"] = m.sourceLocSubject
	cfg["sourceLocTrans"] = m.sourceLocTrans
	cfg["sourceLocBem"] = m.sourceLocBem
	cfg["sourceLocMindistMm"] = m.sourceLocMindistMm
	cfg["sourceLocFmriEnabled"] = m.sourceLocFmriEnabled
	cfg["sourceLocFmriStatsMap"] = m.sourceLocFmriStatsMap
	cfg["sourceLocFmriProvenance"] = m.sourceLocFmriProvenance
	cfg["sourceLocFmriRequireProv"] = m.sourceLocFmriRequireProv
	cfg["sourceLocFmriThreshold"] = m.sourceLocFmriThreshold
	cfg["sourceLocFmriTail"] = m.sourceLocFmriTail
	cfg["sourceLocFmriMinClusterVox"] = m.sourceLocFmriMinClusterVox
	cfg["sourceLocFmriMinClusterMM3"] = m.sourceLocFmriMinClusterMM3
	cfg["sourceLocFmriMaxClusters"] = m.sourceLocFmriMaxClusters
	cfg["sourceLocFmriMaxVoxPerClus"] = m.sourceLocFmriMaxVoxPerClus
	cfg["sourceLocFmriMaxTotalVox"] = m.sourceLocFmriMaxTotalVox
	cfg["sourceLocFmriRandomSeed"] = m.sourceLocFmriRandomSeed
	cfg["sourceLocCreateTrans"] = m.sourceLocCreateTrans
	cfg["sourceLocAllowIdentityTrans"] = m.sourceLocAllowIdentityTrans
	cfg["sourceLocCreateBemModel"] = m.sourceLocCreateBemModel
	cfg["sourceLocCreateBemSolution"] = m.sourceLocCreateBemSolution

	// fMRI contrast builder
	cfg["sourceLocFmriContrastEnabled"] = m.sourceLocFmriContrastEnabled
	cfg["sourceLocFmriContrastType"] = m.sourceLocFmriContrastType
	cfg["sourceLocFmriCondAColumn"] = m.sourceLocFmriCondAColumn
	cfg["sourceLocFmriCondAValue"] = m.sourceLocFmriCondAValue
	cfg["sourceLocFmriCondBColumn"] = m.sourceLocFmriCondBColumn
	cfg["sourceLocFmriCondBValue"] = m.sourceLocFmriCondBValue
	cfg["sourceLocFmriContrastFormula"] = m.sourceLocFmriContrastFormula
	cfg["sourceLocFmriContrastName"] = m.sourceLocFmriContrastName
	cfg["sourceLocFmriRunsToInclude"] = m.sourceLocFmriRunsToInclude
	cfg["sourceLocFmriAutoDetectRuns"] = m.sourceLocFmriAutoDetectRuns
	cfg["sourceLocFmriHrfModel"] = m.sourceLocFmriHrfModel
	cfg["sourceLocFmriDriftModel"] = m.sourceLocFmriDriftModel
	cfg["sourceLocFmriHighPassHz"] = m.sourceLocFmriHighPassHz
	cfg["sourceLocFmriLowPassHz"] = m.sourceLocFmriLowPassHz
	cfg["sourceLocFmriClusterCorrection"] = m.sourceLocFmriClusterCorrection
	cfg["sourceLocFmriClusterPThreshold"] = m.sourceLocFmriClusterPThreshold
	cfg["sourceLocFmriOutputType"] = m.sourceLocFmriOutputType
	cfg["sourceLocFmriResampleToFS"] = m.sourceLocFmriResampleToFS
	cfg["sourceLocFmriWindowAName"] = m.sourceLocFmriWindowAName
	cfg["sourceLocFmriWindowATmin"] = m.sourceLocFmriWindowATmin
	cfg["sourceLocFmriWindowATmax"] = m.sourceLocFmriWindowATmax
	cfg["sourceLocFmriWindowBName"] = m.sourceLocFmriWindowBName
	cfg["sourceLocFmriWindowBTmin"] = m.sourceLocFmriWindowBTmin
	cfg["sourceLocFmriWindowBTmax"] = m.sourceLocFmriWindowBTmax

	// Aggregation/storage
	cfg["aggregationMethod"] = m.aggregationMethod
	cfg["minEpochsForFeatures"] = m.minEpochsForFeatures
	cfg["featAnalysisMode"] = m.featAnalysisMode
	cfg["featComputeChangeScores"] = m.featComputeChangeScores
	cfg["featSaveTfrWithSidecar"] = m.featSaveTfrWithSidecar
	cfg["featNJobsBands"] = m.featNJobsBands
	cfg["featNJobsConnectivity"] = m.featNJobsConnectivity
	cfg["featNJobsAperiodic"] = m.featNJobsAperiodic
	cfg["featNJobsComplexity"] = m.featNJobsComplexity
	cfg["saveSubjectLevelFeatures"] = m.saveSubjectLevelFeatures
	cfg["featAlsoSaveCsv"] = m.featAlsoSaveCsv

	// Spatial transform
	cfg["spatialTransform"] = m.spatialTransform
	cfg["spatialTransformLambda2"] = m.spatialTransformLambda2
	cfg["spatialTransformStiffness"] = m.spatialTransformStiffness

	// TFR parameters
	cfg["tfrFreqMin"] = m.tfrFreqMin
	cfg["tfrFreqMax"] = m.tfrFreqMax
	cfg["tfrNFreqs"] = m.tfrNFreqs
	cfg["tfrMinCycles"] = m.tfrMinCycles
	cfg["tfrMaxCycles"] = m.tfrMaxCycles
	cfg["tfrNCyclesFactor"] = m.tfrNCyclesFactor
	cfg["tfrDecim"] = m.tfrDecim
	cfg["tfrDecimPower"] = m.tfrDecimPower
	cfg["tfrDecimPhase"] = m.tfrDecimPhase
	cfg["tfrWorkers"] = m.tfrWorkers

	// Quality features
	cfg["qualityPsdMethod"] = m.qualityPsdMethod
	cfg["qualityFmin"] = m.qualityFmin
	cfg["qualityFmax"] = m.qualityFmax
	cfg["qualityNfft"] = m.qualityNfft
	cfg["qualityExcludeLineNoise"] = m.qualityExcludeLineNoise
	cfg["qualityLineNoiseFreq"] = m.qualityLineNoiseFreq
	cfg["qualityLineNoiseWidthHz"] = m.qualityLineNoiseWidthHz
	cfg["qualityLineNoiseHarmonics"] = m.qualityLineNoiseHarmonics
	cfg["qualitySnrSignalBandMin"] = m.qualitySnrSignalBandMin
	cfg["qualitySnrSignalBandMax"] = m.qualitySnrSignalBandMax
	cfg["qualitySnrNoiseBandMin"] = m.qualitySnrNoiseBandMin
	cfg["qualitySnrNoiseBandMax"] = m.qualitySnrNoiseBandMax
	cfg["qualityMuscleBandMin"] = m.qualityMuscleBandMin
	cfg["qualityMuscleBandMax"] = m.qualityMuscleBandMax

	// ERDS configuration
	cfg["erdsUseLogRatio"] = m.erdsUseLogRatio
	cfg["erdsMinBaselinePower"] = m.erdsMinBaselinePower
	cfg["erdsMinActivePower"] = m.erdsMinActivePower
	cfg["erdsMinSegmentSec"] = m.erdsMinSegmentSec
	cfg["erdsBandsSpec"] = m.erdsBandsSpec

	// Asymmetry & Ratios
	cfg["asymmetryChannelPairsSpec"] = m.asymmetryChannelPairsSpec
	cfg["asymmetryMinSegmentSec"] = m.asymmetryMinSegmentSec
	cfg["asymmetryMinCyclesAtFmin"] = m.asymmetryMinCyclesAtFmin
	cfg["asymmetrySkipInvalidSegments"] = m.asymmetrySkipInvalidSegments
	cfg["asymmetryEmitActivationConvention"] = m.asymmetryEmitActivationConvention
	cfg["asymmetryActivationBandsSpec"] = m.asymmetryActivationBandsSpec
	cfg["ratiosMinSegmentSec"] = m.ratiosMinSegmentSec
	cfg["ratiosMinCyclesAtFmin"] = m.ratiosMinCyclesAtFmin
	cfg["ratiosSkipInvalidSegments"] = m.ratiosSkipInvalidSegments
	cfg["ratioSource"] = m.ratioSource

	// IAF
	cfg["iafEnabled"] = m.iafEnabled
	cfg["iafAlphaWidthHz"] = m.iafAlphaWidthHz
	cfg["iafSearchRangeMin"] = m.iafSearchRangeMin
	cfg["iafSearchRangeMax"] = m.iafSearchRangeMax
	cfg["iafMinProminence"] = m.iafMinProminence
	cfg["iafRoisSpec"] = m.iafRoisSpec
	cfg["iafMinCyclesAtFmin"] = m.iafMinCyclesAtFmin
	cfg["iafMinBaselineSec"] = m.iafMinBaselineSec
	cfg["iafAllowFullFallback"] = m.iafAllowFullFallback
	cfg["iafAllowAllChannelsFallback"] = m.iafAllowAllChannelsFallback

	// Directed connectivity
	cfg["directedConnEnabled"] = m.directedConnEnabled
	cfg["directedConnOutputLevel"] = m.directedConnOutputLevel
	cfg["directedConnMvarOrder"] = m.directedConnMvarOrder
	cfg["directedConnNFreqs"] = m.directedConnNFreqs
	cfg["directedConnMinSegSamples"] = m.directedConnMinSegSamples

	// Behavior pipeline
	cfg["correlationMethod"] = m.correlationMethod
	cfg["robustCorrelation"] = m.robustCorrelation
	cfg["bootstrapSamples"] = m.bootstrapSamples
	cfg["nPermutations"] = m.nPermutations
	cfg["rngSeed"] = m.rngSeed
	cfg["fdrAlpha"] = m.fdrAlpha
	cfg["behaviorNJobs"] = m.behaviorNJobs
	cfg["alsoSaveCsv"] = m.alsoSaveCsv
	cfg["behaviorOverwrite"] = m.behaviorOverwrite
	cfg["runAdjustmentColumn"] = m.runAdjustmentColumn
	cfg["runAdjustmentEnabled"] = m.runAdjustmentEnabled
	cfg["runAdjustmentIncludeInCorrelations"] = m.runAdjustmentIncludeInCorrelations
	cfg["runAdjustmentMaxDummies"] = m.runAdjustmentMaxDummies
	cfg["conditionCompareColumn"] = m.conditionCompareColumn
	cfg["conditionCompareValues"] = m.conditionCompareValues
	cfg["conditionCompareWindows"] = m.conditionCompareWindows
	cfg["conditionFailFast"] = m.conditionFailFast
	cfg["conditionEffectThreshold"] = m.conditionEffectThreshold
	cfg["conditionOverwrite"] = m.conditionOverwrite
	cfg["temporalTargetColumn"] = m.temporalTargetColumn
	cfg["temporalConditionColumn"] = m.temporalConditionColumn
	cfg["temporalConditionValues"] = m.temporalConditionValues
	cfg["temporalSplitByCondition"] = m.temporalSplitByCondition
	cfg["temporalIncludeROIAverages"] = m.temporalIncludeROIAverages
	cfg["temporalIncludeTFGrid"] = m.temporalIncludeTFGrid
	cfg["temporalFeaturePowerEnabled"] = m.temporalFeaturePowerEnabled
	cfg["temporalFeatureITPCEnabled"] = m.temporalFeatureITPCEnabled
	cfg["temporalFeatureERDSEnabled"] = m.temporalFeatureERDSEnabled
	cfg["temporalITPCBaselineCorrection"] = m.temporalITPCBaselineCorrection
	cfg["temporalITPCBaselineMin"] = m.temporalITPCBaselineMin
	cfg["temporalITPCBaselineMax"] = m.temporalITPCBaselineMax
	cfg["temporalERDSBaselineMin"] = m.temporalERDSBaselineMin
	cfg["temporalERDSBaselineMax"] = m.temporalERDSBaselineMax
	cfg["temporalERDSMethod"] = m.temporalERDSMethod
	cfg["controlTemperature"] = m.controlTemperature
	cfg["controlTrialOrder"] = m.controlTrialOrder
	cfg["behaviorComputeChangeScores"] = m.behaviorComputeChangeScores
	cfg["behaviorComputeBayesFactors"] = m.behaviorComputeBayesFactors
	cfg["behaviorComputeLosoStability"] = m.behaviorComputeLosoStability

	// Trial table
	cfg["trialTableFormat"] = m.trialTableFormat
	cfg["trialTableAddLagFeatures"] = m.trialTableAddLagFeatures
	cfg["trialOrderMaxMissingFraction"] = m.trialOrderMaxMissingFraction

	// Feature QC
	cfg["featureQCEnabled"] = m.featureQCEnabled
	cfg["featureQCMaxMissingPct"] = m.featureQCMaxMissingPct
	cfg["featureQCMinVariance"] = m.featureQCMinVariance
	cfg["featureQCCheckWithinRunVariance"] = m.featureQCCheckWithinRunVariance

	// Pain residual
	cfg["painResidualEnabled"] = m.painResidualEnabled
	cfg["painResidualMethod"] = m.painResidualMethod
	cfg["painResidualPolyDegree"] = m.painResidualPolyDegree
	cfg["painResidualSplineDfCandidates"] = m.painResidualSplineDfCandidates
	cfg["painResidualModelCompareEnabled"] = m.painResidualModelCompareEnabled
	cfg["painResidualModelComparePolyDegrees"] = m.painResidualModelComparePolyDegrees
	cfg["painResidualBreakpointEnabled"] = m.painResidualBreakpointEnabled
	cfg["painResidualBreakpointCandidates"] = m.painResidualBreakpointCandidates
	cfg["painResidualBreakpointQlow"] = m.painResidualBreakpointQlow
	cfg["painResidualBreakpointQhigh"] = m.painResidualBreakpointQhigh
	cfg["painResidualCrossfitEnabled"] = m.painResidualCrossfitEnabled
	cfg["painResidualCrossfitGroupColumn"] = m.painResidualCrossfitGroupColumn
	cfg["painResidualCrossfitNSplits"] = m.painResidualCrossfitNSplits
	cfg["painResidualCrossfitMethod"] = m.painResidualCrossfitMethod
	cfg["painResidualCrossfitSplineKnots"] = m.painResidualCrossfitSplineKnots

	// Regression
	cfg["regressionOutcome"] = m.regressionOutcome
	cfg["regressionIncludeTemperature"] = m.regressionIncludeTemperature
	cfg["regressionTempControl"] = m.regressionTempControl
	cfg["regressionTempSplineKnots"] = m.regressionTempSplineKnots
	cfg["regressionTempSplineQlow"] = m.regressionTempSplineQlow
	cfg["regressionTempSplineQhigh"] = m.regressionTempSplineQhigh
	cfg["regressionIncludeTrialOrder"] = m.regressionIncludeTrialOrder
	cfg["regressionIncludePrev"] = m.regressionIncludePrev
	cfg["regressionIncludeRunBlock"] = m.regressionIncludeRunBlock
	cfg["regressionIncludeInteraction"] = m.regressionIncludeInteraction
	cfg["regressionStandardize"] = m.regressionStandardize
	cfg["regressionPermutations"] = m.regressionPermutations
	cfg["regressionMaxFeatures"] = m.regressionMaxFeatures

	// Models
	cfg["modelsIncludeTemperature"] = m.modelsIncludeTemperature
	cfg["modelsTempControl"] = m.modelsTempControl
	cfg["modelsTempSplineKnots"] = m.modelsTempSplineKnots
	cfg["modelsTempSplineQlow"] = m.modelsTempSplineQlow
	cfg["modelsTempSplineQhigh"] = m.modelsTempSplineQhigh
	cfg["modelsIncludeTrialOrder"] = m.modelsIncludeTrialOrder
	cfg["modelsIncludePrev"] = m.modelsIncludePrev
	cfg["modelsIncludeRunBlock"] = m.modelsIncludeRunBlock
	cfg["modelsIncludeInteraction"] = m.modelsIncludeInteraction
	cfg["modelsStandardize"] = m.modelsStandardize
	cfg["modelsMaxFeatures"] = m.modelsMaxFeatures
	cfg["modelsOutcomeRating"] = m.modelsOutcomeRating
	cfg["modelsOutcomePainResidual"] = m.modelsOutcomePainResidual
	cfg["modelsOutcomeTemperature"] = m.modelsOutcomeTemperature
	cfg["modelsOutcomePainBinary"] = m.modelsOutcomePainBinary
	cfg["modelsFamilyOLS"] = m.modelsFamilyOLS
	cfg["modelsFamilyRobust"] = m.modelsFamilyRobust
	cfg["modelsFamilyQuantile"] = m.modelsFamilyQuantile
	cfg["modelsFamilyLogit"] = m.modelsFamilyLogit
	cfg["modelsBinaryOutcome"] = m.modelsBinaryOutcome

	// Stability
	cfg["stabilityMethod"] = m.stabilityMethod
	cfg["stabilityOutcome"] = m.stabilityOutcome
	cfg["stabilityGroupColumn"] = m.stabilityGroupColumn
	cfg["stabilityPartialTemp"] = m.stabilityPartialTemp
	cfg["stabilityMaxFeatures"] = m.stabilityMaxFeatures
	cfg["stabilityAlpha"] = m.stabilityAlpha

	// Consistency & influence
	cfg["consistencyEnabled"] = m.consistencyEnabled
	cfg["influenceOutcomeRating"] = m.influenceOutcomeRating
	cfg["influenceOutcomePainResidual"] = m.influenceOutcomePainResidual
	cfg["influenceOutcomeTemperature"] = m.influenceOutcomeTemperature
	cfg["influenceMaxFeatures"] = m.influenceMaxFeatures
	cfg["influenceIncludeTemperature"] = m.influenceIncludeTemperature
	cfg["influenceTempControl"] = m.influenceTempControl
	cfg["influenceTempSplineKnots"] = m.influenceTempSplineKnots
	cfg["influenceTempSplineQlow"] = m.influenceTempSplineQlow
	cfg["influenceTempSplineQhigh"] = m.influenceTempSplineQhigh
	cfg["influenceTempSplineMinN"] = m.influenceTempSplineMinN
	cfg["influenceIncludeTrialOrder"] = m.influenceIncludeTrialOrder
	cfg["influenceIncludeRunBlock"] = m.influenceIncludeRunBlock
	cfg["influenceIncludeInteraction"] = m.influenceIncludeInteraction
	cfg["influenceStandardize"] = m.influenceStandardize
	cfg["influenceCooksThreshold"] = m.influenceCooksThreshold
	cfg["influenceLeverageThreshold"] = m.influenceLeverageThreshold

	// Correlations
	cfg["correlationsTargetRating"] = m.correlationsTargetRating
	cfg["correlationsTargetTemperature"] = m.correlationsTargetTemperature
	cfg["correlationsTargetPainResidual"] = m.correlationsTargetPainResidual
	cfg["correlationsTypesSpec"] = m.correlationsTypesSpec
	cfg["correlationsTargetColumn"] = m.correlationsTargetColumn
	cfg["correlationsPreferPainResidual"] = m.correlationsPreferPainResidual
	cfg["correlationsUseCrossfitResidual"] = m.correlationsUseCrossfitResidual
	cfg["correlationsPrimaryUnit"] = m.correlationsPrimaryUnit
	cfg["correlationsPermutationPrimary"] = m.correlationsPermutationPrimary
	cfg["groupLevelBlockPermutation"] = m.groupLevelBlockPermutation

	// Mixed Effects & Mediation
	cfg["mixedEffectsType"] = m.mixedEffectsType
	cfg["mixedMaxFeatures"] = m.mixedMaxFeatures
	cfg["mediationMinEffect"] = m.mediationMinEffect
	cfg["mediationBootstrap"] = m.mediationBootstrap
	cfg["mediationMaxMediatorsEnabled"] = m.mediationMaxMediatorsEnabled
	cfg["mediationMaxMediators"] = m.mediationMaxMediators
	cfg["mediationPermutations"] = m.mediationPermutations
	cfg["moderationMaxFeaturesEnabled"] = m.moderationMaxFeaturesEnabled
	cfg["moderationMaxFeatures"] = m.moderationMaxFeatures
	cfg["moderationPermutations"] = m.moderationPermutations

	// Cluster tests
	cfg["clusterThreshold"] = m.clusterThreshold
	cfg["clusterMinSize"] = m.clusterMinSize
	cfg["clusterTail"] = m.clusterTail
	cfg["clusterConditionColumn"] = m.clusterConditionColumn
	cfg["clusterConditionValues"] = m.clusterConditionValues

	// Temporal
	cfg["temporalResolutionMs"] = m.temporalResolutionMs
	cfg["temporalSmoothMs"] = m.temporalSmoothMs
	cfg["temporalTimeMinMs"] = m.temporalTimeMinMs
	cfg["temporalTimeMaxMs"] = m.temporalTimeMaxMs

	// Preprocessing (EEG)
	cfg["prepUsePyprep"] = m.prepUsePyprep
	cfg["prepUseIcalabel"] = m.prepUseIcalabel
	cfg["prepNJobs"] = m.prepNJobs
	cfg["prepMontage"] = m.prepMontage
	cfg["prepResample"] = m.prepResample
	cfg["prepLFreq"] = m.prepLFreq
	cfg["prepHFreq"] = m.prepHFreq
	cfg["prepNotch"] = m.prepNotch
	cfg["prepLineFreq"] = m.prepLineFreq
	cfg["prepChTypes"] = m.prepChTypes
	cfg["prepEegReference"] = m.prepEegReference
	cfg["prepEogChannels"] = m.prepEogChannels
	cfg["prepRandomState"] = m.prepRandomState
	cfg["prepTaskIsRest"] = m.prepTaskIsRest
	cfg["prepZaplineFline"] = m.prepZaplineFline
	cfg["prepFindBreaks"] = m.prepFindBreaks
	cfg["prepRansac"] = m.prepRansac
	cfg["prepRepeats"] = m.prepRepeats
	cfg["prepAverageReref"] = m.prepAverageReref
	cfg["prepFileExtension"] = m.prepFileExtension
	cfg["prepConsiderPreviousBads"] = m.prepConsiderPreviousBads
	cfg["prepOverwriteChansTsv"] = m.prepOverwriteChansTsv
	cfg["prepDeleteBreaks"] = m.prepDeleteBreaks
	cfg["prepBreaksMinLength"] = m.prepBreaksMinLength
	cfg["prepTStartAfterPrevious"] = m.prepTStartAfterPrevious
	cfg["prepTStopBeforeNext"] = m.prepTStopBeforeNext
	cfg["prepSpatialFilter"] = m.prepSpatialFilter
	cfg["prepICAAlgorithm"] = m.prepICAAlgorithm
	cfg["prepICAComp"] = m.prepICAComp
	cfg["prepICALFreq"] = m.prepICALFreq
	cfg["prepICARejThresh"] = m.prepICARejThresh
	cfg["prepProbThresh"] = m.prepProbThresh
	cfg["prepKeepMnebidsBads"] = m.prepKeepMnebidsBads
	cfg["prepConditions"] = m.prepConditions
	cfg["prepEpochsTmin"] = m.prepEpochsTmin
	cfg["prepEpochsTmax"] = m.prepEpochsTmax
	cfg["prepEpochsBaselineStart"] = m.prepEpochsBaselineStart
	cfg["prepEpochsBaselineEnd"] = m.prepEpochsBaselineEnd
	cfg["prepEpochsNoBaseline"] = m.prepEpochsNoBaseline
	cfg["prepEpochsReject"] = m.prepEpochsReject
	cfg["prepWriteCleanEvents"] = m.prepWriteCleanEvents
	cfg["prepOverwriteCleanEvents"] = m.prepOverwriteCleanEvents
	cfg["prepCleanEventsStrict"] = m.prepCleanEventsStrict
	cfg["prepRejectMethod"] = m.prepRejectMethod
	cfg["prepRunSourceEstimation"] = m.prepRunSourceEstimation
	cfg["icaLabelsToKeep"] = m.icaLabelsToKeep

	// Preprocessing UI State (EEG)
	cfg["prepGroupStagesExpanded"] = m.prepGroupStagesExpanded
	cfg["prepGroupGeneralExpanded"] = m.prepGroupGeneralExpanded
	cfg["prepGroupFilteringExpanded"] = m.prepGroupFilteringExpanded
	cfg["prepGroupPyprepExpanded"] = m.prepGroupPyprepExpanded
	cfg["prepGroupICAExpanded"] = m.prepGroupICAExpanded
	cfg["prepGroupEpochingExpanded"] = m.prepGroupEpochingExpanded

	// fMRI Preprocessing
	cfg["fmriEngineIndex"] = m.fmriEngineIndex
	cfg["fmriFmriprepImage"] = m.fmriFmriprepImage
	cfg["fmriFmriprepOutputDir"] = m.fmriFmriprepOutputDir
	cfg["fmriFmriprepWorkDir"] = m.fmriFmriprepWorkDir
	cfg["fmriFreesurferLicenseFile"] = m.fmriFreesurferLicenseFile
	cfg["fmriFreesurferSubjectsDir"] = m.fmriFreesurferSubjectsDir
	cfg["fmriOutputSpacesSpec"] = m.fmriOutputSpacesSpec
	cfg["fmriIgnoreSpec"] = m.fmriIgnoreSpec
	cfg["fmriBidsFilterFile"] = m.fmriBidsFilterFile
	cfg["fmriExtraArgs"] = m.fmriExtraArgs
	cfg["fmriUseAroma"] = m.fmriUseAroma
	cfg["fmriSkipBidsValidation"] = m.fmriSkipBidsValidation
	cfg["fmriStopOnFirstCrash"] = m.fmriStopOnFirstCrash
	cfg["fmriCleanWorkdir"] = m.fmriCleanWorkdir
	cfg["fmriSkipReconstruction"] = m.fmriSkipReconstruction
	cfg["fmriMemMb"] = m.fmriMemMb
	cfg["fmriNThreads"] = m.fmriNThreads
	cfg["fmriOmpNThreads"] = m.fmriOmpNThreads
	cfg["fmriLowMem"] = m.fmriLowMem
	cfg["fmriLongitudinal"] = m.fmriLongitudinal
	cfg["fmriCiftiOutputIndex"] = m.fmriCiftiOutputIndex
	cfg["fmriSkullStripTemplate"] = m.fmriSkullStripTemplate
	cfg["fmriSkullStripFixedSeed"] = m.fmriSkullStripFixedSeed
	cfg["fmriRandomSeed"] = m.fmriRandomSeed
	cfg["fmriDummyScans"] = m.fmriDummyScans
	cfg["fmriBold2T1wInitIndex"] = m.fmriBold2T1wInitIndex
	cfg["fmriBold2T1wDof"] = m.fmriBold2T1wDof
	cfg["fmriSliceTimeRef"] = m.fmriSliceTimeRef
	cfg["fmriFdSpikeThreshold"] = m.fmriFdSpikeThreshold
	cfg["fmriDvarsSpikeThreshold"] = m.fmriDvarsSpikeThreshold
	cfg["fmriMeOutputEchos"] = m.fmriMeOutputEchos
	cfg["fmriMedialSurfaceNan"] = m.fmriMedialSurfaceNan
	cfg["fmriNoMsm"] = m.fmriNoMsm
	cfg["fmriLevelIndex"] = m.fmriLevelIndex
	cfg["fmriTaskId"] = m.fmriTaskId

	// fMRI Analysis
	cfg["fmriAnalysisInputSourceIndex"] = m.fmriAnalysisInputSourceIndex
	cfg["fmriAnalysisFmriprepSpace"] = m.fmriAnalysisFmriprepSpace
	cfg["fmriAnalysisRequireFmriprep"] = m.fmriAnalysisRequireFmriprep
	cfg["fmriAnalysisRunsSpec"] = m.fmriAnalysisRunsSpec
	cfg["fmriAnalysisContrastType"] = m.fmriAnalysisContrastType
	cfg["fmriAnalysisCondAColumn"] = m.fmriAnalysisCondAColumn
	cfg["fmriAnalysisCondAValue"] = m.fmriAnalysisCondAValue
	cfg["fmriAnalysisCondBColumn"] = m.fmriAnalysisCondBColumn
	cfg["fmriAnalysisCondBValue"] = m.fmriAnalysisCondBValue
	cfg["fmriAnalysisContrastName"] = m.fmriAnalysisContrastName
	cfg["fmriAnalysisFormula"] = m.fmriAnalysisFormula
	cfg["fmriAnalysisEventsToModel"] = m.fmriAnalysisEventsToModel
	cfg["fmriAnalysisHrfModel"] = m.fmriAnalysisHrfModel
	cfg["fmriAnalysisDriftModel"] = m.fmriAnalysisDriftModel
	cfg["fmriAnalysisHighPassHz"] = m.fmriAnalysisHighPassHz
	cfg["fmriAnalysisLowPassHz"] = m.fmriAnalysisLowPassHz
	cfg["fmriAnalysisSmoothingFwhm"] = m.fmriAnalysisSmoothingFwhm
	cfg["fmriAnalysisOutputType"] = m.fmriAnalysisOutputType
	cfg["fmriAnalysisOutputDir"] = m.fmriAnalysisOutputDir
	cfg["fmriAnalysisResampleToFS"] = m.fmriAnalysisResampleToFS
	cfg["fmriAnalysisFreesurferDir"] = m.fmriAnalysisFreesurferDir
	cfg["fmriAnalysisConfoundsStrategy"] = m.fmriAnalysisConfoundsStrategy
	cfg["fmriAnalysisWriteDesignMatrix"] = m.fmriAnalysisWriteDesignMatrix

	cfg["fmriAnalysisGroupInputExpanded"] = m.fmriAnalysisGroupInputExpanded
	cfg["fmriAnalysisGroupContrastExpanded"] = m.fmriAnalysisGroupContrastExpanded
	cfg["fmriAnalysisGroupGLMExpanded"] = m.fmriAnalysisGroupGLMExpanded
	cfg["fmriAnalysisGroupConfoundsExpanded"] = m.fmriAnalysisGroupConfoundsExpanded
	cfg["fmriAnalysisGroupOutputExpanded"] = m.fmriAnalysisGroupOutputExpanded
	cfg["fmriAnalysisGroupPlottingExpanded"] = m.fmriAnalysisGroupPlottingExpanded

	cfg["fmriAnalysisPlotsEnabled"] = m.fmriAnalysisPlotsEnabled
	cfg["fmriAnalysisPlotHTML"] = m.fmriAnalysisPlotHTML
	cfg["fmriAnalysisPlotSpaceIndex"] = m.fmriAnalysisPlotSpaceIndex
	cfg["fmriAnalysisPlotThresholdModeIndex"] = m.fmriAnalysisPlotThresholdModeIndex
	cfg["fmriAnalysisPlotZThreshold"] = m.fmriAnalysisPlotZThreshold
	cfg["fmriAnalysisPlotFdrQ"] = m.fmriAnalysisPlotFdrQ
	cfg["fmriAnalysisPlotClusterMinVoxels"] = m.fmriAnalysisPlotClusterMinVoxels
	cfg["fmriAnalysisPlotVmaxModeIndex"] = m.fmriAnalysisPlotVmaxModeIndex
	cfg["fmriAnalysisPlotVmaxManual"] = m.fmriAnalysisPlotVmaxManual
	cfg["fmriAnalysisPlotIncludeUnthresholded"] = m.fmriAnalysisPlotIncludeUnthresholded
	cfg["fmriAnalysisPlotFormatPNG"] = m.fmriAnalysisPlotFormatPNG
	cfg["fmriAnalysisPlotFormatSVG"] = m.fmriAnalysisPlotFormatSVG
	cfg["fmriAnalysisPlotTypeSlices"] = m.fmriAnalysisPlotTypeSlices
	cfg["fmriAnalysisPlotTypeGlass"] = m.fmriAnalysisPlotTypeGlass
	cfg["fmriAnalysisPlotTypeHist"] = m.fmriAnalysisPlotTypeHist
	cfg["fmriAnalysisPlotTypeClusters"] = m.fmriAnalysisPlotTypeClusters
	cfg["fmriAnalysisPlotEffectSize"] = m.fmriAnalysisPlotEffectSize
	cfg["fmriAnalysisPlotStandardError"] = m.fmriAnalysisPlotStandardError
	cfg["fmriAnalysisPlotMotionQC"] = m.fmriAnalysisPlotMotionQC
	cfg["fmriAnalysisPlotCarpetQC"] = m.fmriAnalysisPlotCarpetQC
	cfg["fmriAnalysisPlotTSNRQC"] = m.fmriAnalysisPlotTSNRQC
	cfg["fmriAnalysisPlotDesignQC"] = m.fmriAnalysisPlotDesignQC
	cfg["fmriAnalysisPlotEmbedImages"] = m.fmriAnalysisPlotEmbedImages
	cfg["fmriAnalysisPlotSignatures"] = m.fmriAnalysisPlotSignatures
	cfg["fmriAnalysisSignatureDir"] = m.fmriAnalysisSignatureDir
	cfg["fmriTrialSigGroupExpanded"] = m.fmriTrialSigGroupExpanded
	cfg["fmriTrialSigMethodIndex"] = m.fmriTrialSigMethodIndex
	cfg["fmriTrialSigIncludeOtherEvents"] = m.fmriTrialSigIncludeOtherEvents
	cfg["fmriTrialSigMaxTrialsPerRun"] = m.fmriTrialSigMaxTrialsPerRun
	cfg["fmriTrialSigFixedEffectsWeighting"] = m.fmriTrialSigFixedEffectsWeighting
	cfg["fmriTrialSigWriteTrialBetas"] = m.fmriTrialSigWriteTrialBetas
	cfg["fmriTrialSigWriteTrialVariances"] = m.fmriTrialSigWriteTrialVariances
	cfg["fmriTrialSigWriteConditionBetas"] = m.fmriTrialSigWriteConditionBetas
	cfg["fmriTrialSigSignatureNPS"] = m.fmriTrialSigSignatureNPS
	cfg["fmriTrialSigSignatureSIIPS1"] = m.fmriTrialSigSignatureSIIPS1
	cfg["fmriTrialSigLssOtherRegressorsIndex"] = m.fmriTrialSigLssOtherRegressorsIndex
	cfg["fmriTrialSigGroupColumn"] = m.fmriTrialSigGroupColumn
	cfg["fmriTrialSigGroupValuesSpec"] = m.fmriTrialSigGroupValuesSpec
	cfg["fmriTrialSigGroupScopeIndex"] = m.fmriTrialSigGroupScopeIndex
	cfg["fmriTrialSigScopeStimPhases"] = m.fmriTrialSigScopeStimPhases

	// ML pipeline
	cfg["mlNPerm"] = m.mlNPerm
	cfg["innerSplits"] = m.innerSplits
	cfg["outerJobs"] = m.outerJobs
	cfg["mlScope"] = int(m.mlScope)
	cfg["mlTarget"] = m.mlTarget
	cfg["mlFmriSigGroupExpanded"] = m.mlFmriSigGroupExpanded
	cfg["mlFmriSigMethodIndex"] = m.mlFmriSigMethodIndex
	cfg["mlFmriSigContrastName"] = m.mlFmriSigContrastName
	cfg["mlFmriSigSignatureIndex"] = m.mlFmriSigSignatureIndex
	cfg["mlFmriSigMetricIndex"] = m.mlFmriSigMetricIndex
	cfg["mlFmriSigNormalizationIndex"] = m.mlFmriSigNormalizationIndex
	cfg["mlFmriSigRoundDecimals"] = m.mlFmriSigRoundDecimals
	cfg["mlBinaryThresholdEnabled"] = m.mlBinaryThresholdEnabled
	cfg["mlBinaryThreshold"] = m.mlBinaryThreshold
	cfg["mlFeatureFamiliesSpec"] = m.mlFeatureFamiliesSpec
	cfg["mlFeatureBandsSpec"] = m.mlFeatureBandsSpec
	cfg["mlFeatureSegmentsSpec"] = m.mlFeatureSegmentsSpec
	cfg["mlFeatureScopesSpec"] = m.mlFeatureScopesSpec
	cfg["mlFeatureStatsSpec"] = m.mlFeatureStatsSpec
	cfg["mlFeatureHarmonization"] = int(m.mlFeatureHarmonization)
	cfg["mlCovariatesSpec"] = m.mlCovariatesSpec
	cfg["mlBaselinePredictorsSpec"] = m.mlBaselinePredictorsSpec
	cfg["mlRegressionModel"] = int(m.mlRegressionModel)
	cfg["mlClassificationModel"] = int(m.mlClassificationModel)
	cfg["mlRequireTrialMlSafe"] = m.mlRequireTrialMlSafe
	cfg["mlUncertaintyAlpha"] = m.mlUncertaintyAlpha
	cfg["mlPermNRepeats"] = m.mlPermNRepeats
	cfg["elasticNetAlphaGrid"] = m.elasticNetAlphaGrid
	cfg["elasticNetL1RatioGrid"] = m.elasticNetL1RatioGrid
	cfg["ridgeAlphaGrid"] = m.ridgeAlphaGrid
	cfg["rfNEstimators"] = m.rfNEstimators
	cfg["rfMaxDepthGrid"] = m.rfMaxDepthGrid
	cfg["varianceThresholdGrid"] = m.varianceThresholdGrid

	// Plotting Detailed Config
	cfg["plotDpiIndex"] = m.plotDpiIndex
	cfg["plotSavefigDpiIndex"] = m.plotSavefigDpiIndex
	cfg["plotSharedColorbar"] = m.plotSharedColorbar
	cfg["plotBboxInches"] = m.plotBboxInches
	cfg["plotPadInches"] = m.plotPadInches
	cfg["plotFontFamily"] = m.plotFontFamily
	cfg["plotFontWeight"] = m.plotFontWeight
	cfg["plotFontSizeSmall"] = m.plotFontSizeSmall
	cfg["plotFontSizeMedium"] = m.plotFontSizeMedium
	cfg["plotFontSizeLarge"] = m.plotFontSizeLarge
	cfg["plotFontSizeTitle"] = m.plotFontSizeTitle
	cfg["plotFontSizeAnnotation"] = m.plotFontSizeAnnotation
	cfg["plotFontSizeLabel"] = m.plotFontSizeLabel
	cfg["plotFontSizeYLabel"] = m.plotFontSizeYLabel
	cfg["plotFontSizeSuptitle"] = m.plotFontSizeSuptitle
	cfg["plotFontSizeFigureTitle"] = m.plotFontSizeFigureTitle
	cfg["plotLayoutTightRectSpec"] = m.plotLayoutTightRectSpec
	cfg["plotLayoutTightRectMicrostateSpec"] = m.plotLayoutTightRectMicrostateSpec
	cfg["plotGridSpecWidthRatiosSpec"] = m.plotGridSpecWidthRatiosSpec
	cfg["plotGridSpecHeightRatiosSpec"] = m.plotGridSpecHeightRatiosSpec
	cfg["plotGridSpecHspace"] = m.plotGridSpecHspace
	cfg["plotGridSpecWspace"] = m.plotGridSpecWspace
	cfg["plotGridSpecLeft"] = m.plotGridSpecLeft
	cfg["plotGridSpecRight"] = m.plotGridSpecRight
	cfg["plotGridSpecTop"] = m.plotGridSpecTop
	cfg["plotGridSpecBottom"] = m.plotGridSpecBottom
	cfg["plotFigureSizeStandardSpec"] = m.plotFigureSizeStandardSpec
	cfg["plotFigureSizeMediumSpec"] = m.plotFigureSizeMediumSpec
	cfg["plotFigureSizeSmallSpec"] = m.plotFigureSizeSmallSpec
	cfg["plotFigureSizeSquareSpec"] = m.plotFigureSizeSquareSpec
	cfg["plotFigureSizeWideSpec"] = m.plotFigureSizeWideSpec
	cfg["plotFigureSizeTFRSpec"] = m.plotFigureSizeTFRSpec
	cfg["plotFigureSizeTopomapSpec"] = m.plotFigureSizeTopomapSpec
	cfg["plotColorPain"] = m.plotColorPain
	cfg["plotColorNonpain"] = m.plotColorNonpain
	cfg["plotColorSignificant"] = m.plotColorSignificant
	cfg["plotColorNonsignificant"] = m.plotColorNonsignificant
	cfg["plotColorGray"] = m.plotColorGray
	cfg["plotColorLightGray"] = m.plotColorLightGray
	cfg["plotColorBlack"] = m.plotColorBlack
	cfg["plotColorBlue"] = m.plotColorBlue
	cfg["plotColorRed"] = m.plotColorRed
	cfg["plotColorNetworkNode"] = m.plotColorNetworkNode
	cfg["plotAlphaGrid"] = m.plotAlphaGrid
	cfg["plotAlphaFill"] = m.plotAlphaFill
	cfg["plotAlphaCI"] = m.plotAlphaCI
	cfg["plotAlphaCILine"] = m.plotAlphaCILine
	cfg["plotAlphaTextBox"] = m.plotAlphaTextBox
	cfg["plotAlphaViolinBody"] = m.plotAlphaViolinBody
	cfg["plotAlphaRidgeFill"] = m.plotAlphaRidgeFill
	cfg["plotScatterMarkerSizeSmall"] = m.plotScatterMarkerSizeSmall
	cfg["plotScatterMarkerSizeLarge"] = m.plotScatterMarkerSizeLarge
	cfg["plotScatterMarkerSizeDefault"] = m.plotScatterMarkerSizeDefault
	cfg["plotScatterAlpha"] = m.plotScatterAlpha
	cfg["plotScatterEdgeColor"] = m.plotScatterEdgeColor
	cfg["plotScatterEdgeWidth"] = m.plotScatterEdgeWidth
	cfg["plotBarAlpha"] = m.plotBarAlpha
	cfg["plotBarWidth"] = m.plotBarWidth
	cfg["plotBarCapsize"] = m.plotBarCapsize
	cfg["plotBarCapsizeLarge"] = m.plotBarCapsizeLarge
	cfg["plotLineWidthThin"] = m.plotLineWidthThin
	cfg["plotLineWidthStandard"] = m.plotLineWidthStandard
	cfg["plotLineWidthThick"] = m.plotLineWidthThick
	cfg["plotLineWidthBold"] = m.plotLineWidthBold
	cfg["plotLineAlphaStandard"] = m.plotLineAlphaStandard
	cfg["plotLineAlphaDim"] = m.plotLineAlphaDim
	cfg["plotLineAlphaZeroLine"] = m.plotLineAlphaZeroLine
	cfg["plotLineAlphaFitLine"] = m.plotLineAlphaFitLine
	cfg["plotLineAlphaDiagonal"] = m.plotLineAlphaDiagonal
	cfg["plotLineAlphaReference"] = m.plotLineAlphaReference
	cfg["plotLineRegressionWidth"] = m.plotLineRegressionWidth
	cfg["plotLineResidualWidth"] = m.plotLineResidualWidth
	cfg["plotLineQQWidth"] = m.plotLineQQWidth
	cfg["plotHistBins"] = m.plotHistBins
	cfg["plotHistBinsBehavioral"] = m.plotHistBinsBehavioral
	cfg["plotHistBinsResidual"] = m.plotHistBinsResidual
	cfg["plotHistBinsTFR"] = m.plotHistBinsTFR
	cfg["plotHistEdgeColor"] = m.plotHistEdgeColor
	cfg["plotHistEdgeWidth"] = m.plotHistEdgeWidth
	cfg["plotHistAlpha"] = m.plotHistAlpha
	cfg["plotHistAlphaResidual"] = m.plotHistAlphaResidual
	cfg["plotHistAlphaTFR"] = m.plotHistAlphaTFR
	cfg["plotKdePoints"] = m.plotKdePoints
	cfg["plotKdeColor"] = m.plotKdeColor
	cfg["plotKdeLinewidth"] = m.plotKdeLinewidth
	cfg["plotKdeAlpha"] = m.plotKdeAlpha
	cfg["plotErrorbarMarkerSize"] = m.plotErrorbarMarkerSize
	cfg["plotErrorbarCapsize"] = m.plotErrorbarCapsize
	cfg["plotErrorbarCapsizeLarge"] = m.plotErrorbarCapsizeLarge
	cfg["plotTextStatsX"] = m.plotTextStatsX
	cfg["plotTextStatsY"] = m.plotTextStatsY
	cfg["plotTextPvalueX"] = m.plotTextPvalueX
	cfg["plotTextPvalueY"] = m.plotTextPvalueY
	cfg["plotTextBootstrapX"] = m.plotTextBootstrapX
	cfg["plotTextBootstrapY"] = m.plotTextBootstrapY
	cfg["plotTextChannelAnnotationX"] = m.plotTextChannelAnnotationX
	cfg["plotTextChannelAnnotationY"] = m.plotTextChannelAnnotationY
	cfg["plotTextTitleY"] = m.plotTextTitleY
	cfg["plotTextResidualQcTitleY"] = m.plotTextResidualQcTitleY
	cfg["plotValidationMinBinsForCalibration"] = m.plotValidationMinBinsForCalibration
	cfg["plotValidationMaxBinsForCalibration"] = m.plotValidationMaxBinsForCalibration
	cfg["plotValidationSamplesPerBin"] = m.plotValidationSamplesPerBin
	cfg["plotValidationMinRoisForFDR"] = m.plotValidationMinRoisForFDR
	cfg["plotValidationMinPvaluesForFDR"] = m.plotValidationMinPvaluesForFDR
	cfg["plotTfrDefaultBaselineWindowSpec"] = m.plotTfrDefaultBaselineWindowSpec
	cfg["plotTopomapContours"] = m.plotTopomapContours
	cfg["plotTopomapColormap"] = m.plotTopomapColormap
	cfg["plotTopomapColorbarFraction"] = m.plotTopomapColorbarFraction
	cfg["plotTopomapColorbarPad"] = m.plotTopomapColorbarPad
	cfg["plotTopomapDiffAnnotation"] = m.plotTopomapDiffAnnotation
	cfg["plotTopomapAnnotateDesc"] = m.plotTopomapAnnotateDesc
	cfg["plotTopomapSigMaskMarker"] = m.plotTopomapSigMaskMarker
	cfg["plotTopomapSigMaskMarkerFaceColor"] = m.plotTopomapSigMaskMarkerFaceColor
	cfg["plotTopomapSigMaskMarkerEdgeColor"] = m.plotTopomapSigMaskMarkerEdgeColor
	cfg["plotTopomapSigMaskLinewidth"] = m.plotTopomapSigMaskLinewidth
	cfg["plotTopomapSigMaskMarkerSize"] = m.plotTopomapSigMaskMarkerSize
	cfg["plotTFRLogBase"] = m.plotTFRLogBase
	cfg["plotTFRPercentageMultiplier"] = m.plotTFRPercentageMultiplier
	cfg["plotTFRTopomapWindowSizeMs"] = m.plotTFRTopomapWindowSizeMs
	cfg["plotTFRTopomapWindowCount"] = m.plotTFRTopomapWindowCount
	cfg["plotTFRTopomapLabelXPosition"] = m.plotTFRTopomapLabelXPosition
	cfg["plotTFRTopomapLabelYPositionBottom"] = m.plotTFRTopomapLabelYPositionBottom
	cfg["plotTFRTopomapLabelYPosition"] = m.plotTFRTopomapLabelYPosition
	cfg["plotTFRTopomapTitleY"] = m.plotTFRTopomapTitleY
	cfg["plotTFRTopomapTitlePad"] = m.plotTFRTopomapTitlePad
	cfg["plotTFRTopomapSubplotsRight"] = m.plotTFRTopomapSubplotsRight
	cfg["plotTFRTopomapTemporalHspace"] = m.plotTFRTopomapTemporalHspace
	cfg["plotTFRTopomapTemporalWspace"] = m.plotTFRTopomapTemporalWspace
	cfg["plotRoiWidthPerBand"] = m.plotRoiWidthPerBand
	cfg["plotRoiWidthPerMetric"] = m.plotRoiWidthPerMetric
	cfg["plotRoiHeightPerRoi"] = m.plotRoiHeightPerRoi
	cfg["plotPowerWidthPerBand"] = m.plotPowerWidthPerBand
	cfg["plotPowerHeightPerSegment"] = m.plotPowerHeightPerSegment
	cfg["plotItpcWidthPerBin"] = m.plotItpcWidthPerBin
	cfg["plotItpcHeightPerBand"] = m.plotItpcHeightPerBand
	cfg["plotItpcWidthPerBandBox"] = m.plotItpcWidthPerBandBox
	cfg["plotItpcHeightBox"] = m.plotItpcHeightBox
	cfg["plotPacCmap"] = m.plotPacCmap
	cfg["plotPacWidthPerRoi"] = m.plotPacWidthPerRoi
	cfg["plotPacHeightBox"] = m.plotPacHeightBox
	cfg["plotAperiodicWidthPerColumn"] = m.plotAperiodicWidthPerColumn
	cfg["plotAperiodicHeightPerRow"] = m.plotAperiodicHeightPerRow
	cfg["plotAperiodicNPerm"] = m.plotAperiodicNPerm
	cfg["plotComplexityWidthPerMeasure"] = m.plotComplexityWidthPerMeasure
	cfg["plotComplexityHeightPerSegment"] = m.plotComplexityHeightPerSegment
	cfg["plotConnectivityWidthPerCircle"] = m.plotConnectivityWidthPerCircle
	cfg["plotConnectivityWidthPerBand"] = m.plotConnectivityWidthPerBand
	cfg["plotConnectivityHeightPerMeasure"] = m.plotConnectivityHeightPerMeasure
	cfg["plotConnectivityCircleTopFraction"] = m.plotConnectivityCircleTopFraction
	cfg["plotConnectivityCircleMinLines"] = m.plotConnectivityCircleMinLines
	cfg["plotConnectivityNetworkTopFraction"] = m.plotConnectivityNetworkTopFraction
	cfg["plotPacPairsSpec"] = m.plotPacPairsSpec
	cfg["plotSpectralMetricsSpec"] = m.plotSpectralMetricsSpec
	cfg["plotBurstsMetricsSpec"] = m.plotBurstsMetricsSpec
	cfg["plotTemporalTimeBinsSpec"] = m.plotTemporalTimeBinsSpec
	cfg["plotTemporalTimeLabelsSpec"] = m.plotTemporalTimeLabelsSpec
	cfg["plotAsymmetryStatSpec"] = m.plotAsymmetryStatSpec
	cfg["plotCompareWindows"] = m.plotCompareWindows
	cfg["plotComparisonWindowsSpec"] = m.plotComparisonWindowsSpec
	cfg["plotCompareColumns"] = m.plotCompareColumns
	cfg["plotComparisonSegment"] = m.plotComparisonSegment
	cfg["plotComparisonColumn"] = m.plotComparisonColumn
	cfg["plotComparisonValuesSpec"] = m.plotComparisonValuesSpec
	cfg["plotComparisonLabelsSpec"] = m.plotComparisonLabelsSpec
	cfg["plotComparisonROIsSpec"] = m.plotComparisonROIsSpec
	if m.plotOverwrite != nil {
		cfg["plotOverwrite"] = *m.plotOverwrite
	}

	// System
	cfg["systemNJobs"] = m.systemNJobs
	cfg["systemStrictMode"] = m.systemStrictMode
	cfg["loggingLevel"] = m.loggingLevel

	return cfg
}

// ImportConfig restores configuration from a persisted map.
func (m *Model) ImportConfig(cfg map[string]interface{}) {
	if cfg == nil {
		return
	}

	// Helper functions for type-safe extraction
	getFloat := func(key string, def float64) float64 {
		if v, ok := cfg[key]; ok {
			switch val := v.(type) {
			case float64:
				return val
			case int:
				return float64(val)
			}
		}
		return def
	}
	getInt := func(key string, def int) int {
		if v, ok := cfg[key]; ok {
			switch val := v.(type) {
			case float64:
				return int(val)
			case int:
				return val
			}
		}
		return def
	}
	getBool := func(key string, def bool) bool {
		if v, ok := cfg[key].(bool); ok {
			return v
		}
		return def
	}
	getString := func(key string, def string) string {
		if v, ok := cfg[key].(string); ok {
			return v
		}
		return def
	}

	// UI State - Selections
	m.modeIndex = getInt("modeIndex", m.modeIndex)
	if v, ok := cfg["computationSelected"]; ok {
		m.computationSelected = listToMap(v)
	}
	if v, ok := cfg["categorySelected"]; ok {
		m.selected = listToMap(v)
	}
	if v, ok := cfg["prepStageSelected"]; ok {
		m.prepStageSelected = listToMap(v)
	}
	if v, ok := cfg["featureFileSelected"]; ok {
		m.featureFileSelected = boolMapToStringsMap(v)
	}
	if v, ok := cfg["plotSelected"]; ok {
		m.plotSelected = listToMap(v)
	}
	if v, ok := cfg["featurePlotterSelected"]; ok {
		m.featurePlotterSelected = boolMapToStringsMap(v)
	}
	if v, ok := cfg["plotFormatSelected"]; ok {
		m.plotFormatSelected = boolMapToStringsMap(v)
	}
	if v, ok := cfg["connectivityMeasures"]; ok {
		m.connectivityMeasures = listToMap(v)
	}
	if v, ok := cfg["directedConnMeasures"]; ok {
		m.directedConnMeasures = listToMap(v)
	}

	// UI State - Group Expansion
	m.featGroupConnectivityExpanded = getBool("featGroupConnectivityExpanded", m.featGroupConnectivityExpanded)
	m.featGroupPACExpanded = getBool("featGroupPACExpanded", m.featGroupPACExpanded)
	m.featGroupAperiodicExpanded = getBool("featGroupAperiodicExpanded", m.featGroupAperiodicExpanded)
	m.featGroupComplexityExpanded = getBool("featGroupComplexityExpanded", m.featGroupComplexityExpanded)
	m.featGroupBurstsExpanded = getBool("featGroupBurstsExpanded", m.featGroupBurstsExpanded)
	m.featGroupPowerExpanded = getBool("featGroupPowerExpanded", m.featGroupPowerExpanded)
	m.featGroupSpectralExpanded = getBool("featGroupSpectralExpanded", m.featGroupSpectralExpanded)
	m.featGroupERPExpanded = getBool("featGroupERPExpanded", m.featGroupERPExpanded)
	m.featGroupRatiosExpanded = getBool("featGroupRatiosExpanded", m.featGroupRatiosExpanded)
	m.featGroupAsymmetryExpanded = getBool("featGroupAsymmetryExpanded", m.featGroupAsymmetryExpanded)
	m.featGroupQualityExpanded = getBool("featGroupQualityExpanded", m.featGroupQualityExpanded)
	m.featGroupERDSExpanded = getBool("featGroupERDSExpanded", m.featGroupERDSExpanded)
	m.featGroupStorageExpanded = getBool("featGroupStorageExpanded", m.featGroupStorageExpanded)
	m.featGroupExecutionExpanded = getBool("featGroupExecutionExpanded", m.featGroupExecutionExpanded)
	m.featGroupDirectedConnExpanded = getBool("featGroupDirectedConnExpanded", m.featGroupDirectedConnExpanded)
	m.featGroupSourceLocExpanded = getBool("featGroupSourceLocExpanded", m.featGroupSourceLocExpanded)
	m.featGroupSourceLocFmriExpanded = getBool("featGroupSourceLocFmriExpanded", m.featGroupSourceLocFmriExpanded)
	m.featGroupSourceLocContrastExpanded = getBool("featGroupSourceLocContrastExpanded", m.featGroupSourceLocContrastExpanded)
	m.featGroupSourceLocGLMExpanded = getBool("featGroupSourceLocGLMExpanded", m.featGroupSourceLocGLMExpanded)
	m.featGroupTFRExpanded = getBool("featGroupTFRExpanded", m.featGroupTFRExpanded)

	// fMRI UI State
	m.fmriGroupRuntimeExpanded = getBool("fmriGroupRuntimeExpanded", m.fmriGroupRuntimeExpanded)
	m.fmriGroupOutputExpanded = getBool("fmriGroupOutputExpanded", m.fmriGroupOutputExpanded)
	m.fmriGroupPerformanceExpanded = getBool("fmriGroupPerformanceExpanded", m.fmriGroupPerformanceExpanded)
	m.fmriGroupAnatomicalExpanded = getBool("fmriGroupAnatomicalExpanded", m.fmriGroupAnatomicalExpanded)
	m.fmriGroupBoldExpanded = getBool("fmriGroupBoldExpanded", m.fmriGroupBoldExpanded)
	m.fmriGroupQcExpanded = getBool("fmriGroupQcExpanded", m.fmriGroupQcExpanded)
	m.fmriGroupDenoisingExpanded = getBool("fmriGroupDenoisingExpanded", m.fmriGroupDenoisingExpanded)
	m.fmriGroupSurfaceExpanded = getBool("fmriGroupSurfaceExpanded", m.fmriGroupSurfaceExpanded)
	m.fmriGroupMultiechoExpanded = getBool("fmriGroupMultiechoExpanded", m.fmriGroupMultiechoExpanded)
	m.fmriGroupReproExpanded = getBool("fmriGroupReproExpanded", m.fmriGroupReproExpanded)
	m.fmriGroupValidationExpanded = getBool("fmriGroupValidationExpanded", m.fmriGroupValidationExpanded)
	m.fmriGroupAdvancedExpanded = getBool("fmriGroupAdvancedExpanded", m.fmriGroupAdvancedExpanded)

	// Plotting UI State
	m.plotGroupDefaultsExpanded = getBool("plotGroupDefaultsExpanded", m.plotGroupDefaultsExpanded)
	m.plotGroupFontsExpanded = getBool("plotGroupFontsExpanded", m.plotGroupFontsExpanded)
	m.plotGroupLayoutExpanded = getBool("plotGroupLayoutExpanded", m.plotGroupLayoutExpanded)
	m.plotGroupFigureSizesExpanded = getBool("plotGroupFigureSizesExpanded", m.plotGroupFigureSizesExpanded)
	m.plotGroupColorsExpanded = getBool("plotGroupColorsExpanded", m.plotGroupColorsExpanded)
	m.plotGroupAlphaExpanded = getBool("plotGroupAlphaExpanded", m.plotGroupAlphaExpanded)
	m.plotGroupScatterExpanded = getBool("plotGroupScatterExpanded", m.plotGroupScatterExpanded)
	m.plotGroupBarExpanded = getBool("plotGroupBarExpanded", m.plotGroupBarExpanded)
	m.plotGroupLineExpanded = getBool("plotGroupLineExpanded", m.plotGroupLineExpanded)
	m.plotGroupHistogramExpanded = getBool("plotGroupHistogramExpanded", m.plotGroupHistogramExpanded)
	m.plotGroupTopomapExpanded = getBool("plotGroupTopomapExpanded", m.plotGroupTopomapExpanded)
	m.plotGroupTFRExpanded = getBool("plotGroupTFRExpanded", m.plotGroupTFRExpanded)
	m.plotGroupSizingExpanded = getBool("plotGroupSizingExpanded", m.plotGroupSizingExpanded)
	m.plotGroupSelectionExpanded = getBool("plotGroupSelectionExpanded", m.plotGroupSelectionExpanded)
	m.plotGroupComparisonsExpanded = getBool("plotGroupComparisonsExpanded", m.plotGroupComparisonsExpanded)
	m.plotGroupKDEExpanded = getBool("plotGroupKDEExpanded", m.plotGroupKDEExpanded)
	m.plotGroupErrorbarExpanded = getBool("plotGroupErrorbarExpanded", m.plotGroupErrorbarExpanded)
	m.plotGroupTextExpanded = getBool("plotGroupTextExpanded", m.plotGroupTextExpanded)
	m.plotGroupValidationExpanded = getBool("plotGroupValidationExpanded", m.plotGroupValidationExpanded)
	m.plotGroupTFRMiscExpanded = getBool("plotGroupTFRMiscExpanded", m.plotGroupTFRMiscExpanded)

	// Plotting per-plot config + UI expansion state
	if v, ok := cfg["plotItemConfigExpanded"]; ok {
		m.plotItemConfigExpanded = boolMapToStringsMap(v)
	}
	if v, ok := cfg["plotItemConfigs"]; ok {
		raw, ok := v.(map[string]interface{})
		if ok {
			m.plotItemConfigs = make(map[string]PlotItemConfig, len(raw))
			for plotID, rawCfg := range raw {
				blob, err := json.Marshal(rawCfg)
				if err != nil {
					continue
				}
				var pc PlotItemConfig
				if err := json.Unmarshal(blob, &pc); err != nil {
					continue
				}
				m.plotItemConfigs[plotID] = pc
			}
		}
	}

	// PAC/CFC
	m.pacPhaseMin = getFloat("pacPhaseMin", m.pacPhaseMin)
	m.pacPhaseMax = getFloat("pacPhaseMax", m.pacPhaseMax)
	m.pacAmpMin = getFloat("pacAmpMin", m.pacAmpMin)
	m.pacAmpMax = getFloat("pacAmpMax", m.pacAmpMax)
	m.pacMethod = getInt("pacMethod", m.pacMethod)
	m.pacMinEpochs = getInt("pacMinEpochs", m.pacMinEpochs)
	m.pacPairsSpec = getString("pacPairsSpec", m.pacPairsSpec)
	m.pacSource = getInt("pacSource", m.pacSource)
	m.pacNormalize = getBool("pacNormalize", m.pacNormalize)
	m.pacNSurrogates = getInt("pacNSurrogates", m.pacNSurrogates)
	m.pacRandomSeed = getInt("pacRandomSeed", m.pacRandomSeed)
	m.pacAllowHarmonicOvrlap = getBool("pacAllowHarmonicOvrlap", m.pacAllowHarmonicOvrlap)
	m.pacMaxHarmonic = getInt("pacMaxHarmonic", m.pacMaxHarmonic)
	m.pacHarmonicToleranceHz = getFloat("pacHarmonicToleranceHz", m.pacHarmonicToleranceHz)
	m.pacComputeWaveformQC = getBool("pacComputeWaveformQC", m.pacComputeWaveformQC)
	m.pacWaveformOffsetMs = getFloat("pacWaveformOffsetMs", m.pacWaveformOffsetMs)

	// Aperiodic
	m.aperiodicFmin = getFloat("aperiodicFmin", m.aperiodicFmin)
	m.aperiodicFmax = getFloat("aperiodicFmax", m.aperiodicFmax)
	m.aperiodicPeakZ = getFloat("aperiodicPeakZ", m.aperiodicPeakZ)
	m.aperiodicMinR2 = getFloat("aperiodicMinR2", m.aperiodicMinR2)
	m.aperiodicMinPoints = getInt("aperiodicMinPoints", m.aperiodicMinPoints)
	m.aperiodicMinSegmentSec = getFloat("aperiodicMinSegmentSec", m.aperiodicMinSegmentSec)
	m.aperiodicModel = getInt("aperiodicModel", m.aperiodicModel)
	m.aperiodicPsdMethod = getInt("aperiodicPsdMethod", m.aperiodicPsdMethod)
	m.aperiodicSubtractEvoked = getBool("aperiodicSubtractEvoked", m.aperiodicSubtractEvoked)
	m.aperiodicPsdBandwidth = getFloat("aperiodicPsdBandwidth", m.aperiodicPsdBandwidth)
	m.aperiodicMaxRms = getFloat("aperiodicMaxRms", m.aperiodicMaxRms)
	m.aperiodicExcludeLineNoise = getBool("aperiodicExcludeLineNoise", m.aperiodicExcludeLineNoise)
	m.aperiodicLineNoiseFreq = getFloat("aperiodicLineNoiseFreq", m.aperiodicLineNoiseFreq)
	m.aperiodicLineNoiseWidthHz = getFloat("aperiodicLineNoiseWidthHz", m.aperiodicLineNoiseWidthHz)
	m.aperiodicLineNoiseHarmonics = getInt("aperiodicLineNoiseHarmonics", m.aperiodicLineNoiseHarmonics)

	// Complexity
	m.complexityPEOrder = getInt("complexityPEOrder", m.complexityPEOrder)
	m.complexityPEDelay = getInt("complexityPEDelay", m.complexityPEDelay)
	m.complexitySignalBasis = getInt("complexitySignalBasis", m.complexitySignalBasis)
	m.complexityMinSegmentSec = getFloat("complexityMinSegmentSec", m.complexityMinSegmentSec)
	m.complexityMinSamples = getInt("complexityMinSamples", m.complexityMinSamples)
	m.complexityZscore = getBool("complexityZscore", m.complexityZscore)

	// ERP
	m.erpBaselineCorrection = getBool("erpBaselineCorrection", m.erpBaselineCorrection)
	m.erpAllowNoBaseline = getBool("erpAllowNoBaseline", m.erpAllowNoBaseline)
	m.erpComponentsSpec = getString("erpComponentsSpec", m.erpComponentsSpec)
	m.erpSmoothMs = getFloat("erpSmoothMs", m.erpSmoothMs)
	m.erpPeakProminenceUv = getFloat("erpPeakProminenceUv", m.erpPeakProminenceUv)
	m.erpLowpassHz = getFloat("erpLowpassHz", m.erpLowpassHz)

	// Burst
	m.burstThresholdZ = getFloat("burstThresholdZ", m.burstThresholdZ)
	m.burstThresholdMethod = getInt("burstThresholdMethod", m.burstThresholdMethod)
	m.burstThresholdPercentile = getFloat("burstThresholdPercentile", m.burstThresholdPercentile)
	m.burstThresholdReference = getInt("burstThresholdReference", m.burstThresholdReference)
	m.burstMinTrialsPerCondition = getInt("burstMinTrialsPerCondition", m.burstMinTrialsPerCondition)
	m.burstMinSegmentSec = getFloat("burstMinSegmentSec", m.burstMinSegmentSec)
	m.burstSkipInvalidSegments = getBool("burstSkipInvalidSegments", m.burstSkipInvalidSegments)
	m.burstMinDuration = getInt("burstMinDuration", m.burstMinDuration)
	m.burstMinCycles = getFloat("burstMinCycles", m.burstMinCycles)
	m.burstBandsSpec = getString("burstBandsSpec", m.burstBandsSpec)

	// Power/Spectral
	m.powerBaselineMode = getInt("powerBaselineMode", m.powerBaselineMode)
	m.powerRequireBaseline = getBool("powerRequireBaseline", m.powerRequireBaseline)
	m.powerSubtractEvoked = getBool("powerSubtractEvoked", m.powerSubtractEvoked)
	m.powerMinTrialsPerCondition = getInt("powerMinTrialsPerCondition", m.powerMinTrialsPerCondition)
	m.powerExcludeLineNoise = getBool("powerExcludeLineNoise", m.powerExcludeLineNoise)
	m.powerLineNoiseFreq = getFloat("powerLineNoiseFreq", m.powerLineNoiseFreq)
	m.powerLineNoiseWidthHz = getFloat("powerLineNoiseWidthHz", m.powerLineNoiseWidthHz)
	m.powerLineNoiseHarmonics = getInt("powerLineNoiseHarmonics", m.powerLineNoiseHarmonics)
	m.powerEmitDb = getBool("powerEmitDb", m.powerEmitDb)
	m.spectralEdgePercentile = getFloat("spectralEdgePercentile", m.spectralEdgePercentile)
	m.spectralRatioPairsSpec = getString("spectralRatioPairsSpec", m.spectralRatioPairsSpec)
	m.spectralPsdAdaptive = getBool("spectralPsdAdaptive", m.spectralPsdAdaptive)
	m.spectralMultitaperAdaptive = getBool("spectralMultitaperAdaptive", m.spectralMultitaperAdaptive)
	m.spectralPsdMethod = getInt("spectralPsdMethod", m.spectralPsdMethod)
	m.spectralFmin = getFloat("spectralFmin", m.spectralFmin)
	m.spectralFmax = getFloat("spectralFmax", m.spectralFmax)
	m.spectralIncludeLogRatios = getBool("spectralIncludeLogRatios", m.spectralIncludeLogRatios)
	m.spectralExcludeLineNoise = getBool("spectralExcludeLineNoise", m.spectralExcludeLineNoise)
	m.spectralLineNoiseFreq = getFloat("spectralLineNoiseFreq", m.spectralLineNoiseFreq)
	m.spectralLineNoiseWidthHz = getFloat("spectralLineNoiseWidthHz", m.spectralLineNoiseWidthHz)
	m.spectralLineNoiseHarmonics = getInt("spectralLineNoiseHarmonics", m.spectralLineNoiseHarmonics)
	m.spectralMinSegmentSec = getFloat("spectralMinSegmentSec", m.spectralMinSegmentSec)
	m.spectralMinCyclesAtFmin = getFloat("spectralMinCyclesAtFmin", m.spectralMinCyclesAtFmin)

	// Connectivity
	m.connOutputLevel = getInt("connOutputLevel", m.connOutputLevel)
	m.connGraphMetrics = getBool("connGraphMetrics", m.connGraphMetrics)
	m.connGraphProp = getFloat("connGraphProp", m.connGraphProp)
	m.connWindowLen = getFloat("connWindowLen", m.connWindowLen)
	m.connWindowStep = getFloat("connWindowStep", m.connWindowStep)
	m.connAECMode = getInt("connAECMode", m.connAECMode)
	m.connGranularity = getInt("connGranularity", m.connGranularity)
	m.connConditionColumn = getString("connConditionColumn", m.connConditionColumn)
	m.connConditionValues = getString("connConditionValues", m.connConditionValues)
	m.connMinEpochsPerGroup = getInt("connMinEpochsPerGroup", m.connMinEpochsPerGroup)
	m.connMinCyclesPerBand = getFloat("connMinCyclesPerBand", m.connMinCyclesPerBand)
	m.connWarnNoSpatialTransform = getBool("connWarnNoSpatialTransform", m.connWarnNoSpatialTransform)
	m.connPhaseEstimator = getInt("connPhaseEstimator", m.connPhaseEstimator)
	m.connMinSegmentSec = getFloat("connMinSegmentSec", m.connMinSegmentSec)
	m.connAECOutput = getInt("connAECOutput", m.connAECOutput)
	m.connForceWithinEpochML = getBool("connForceWithinEpochML", m.connForceWithinEpochML)

	// ITPC
	m.itpcMethod = getInt("itpcMethod", m.itpcMethod)
	m.itpcConditionColumn = getString("itpcConditionColumn", m.itpcConditionColumn)
	m.itpcConditionValues = getString("itpcConditionValues", m.itpcConditionValues)
	m.itpcMinTrialsPerCondition = getInt("itpcMinTrialsPerCondition", m.itpcMinTrialsPerCondition)
	m.itpcAllowUnsafeLoo = getBool("itpcAllowUnsafeLoo", m.itpcAllowUnsafeLoo)
	m.itpcBaselineCorrection = getInt("itpcBaselineCorrection", m.itpcBaselineCorrection)
	m.itpcNJobs = getInt("itpcNJobs", m.itpcNJobs)

	// Source localization
	m.sourceLocEnabled = getBool("sourceLocEnabled", m.sourceLocEnabled)
	m.sourceLocMode = getInt("sourceLocMode", m.sourceLocMode)
	m.sourceLocMethod = getInt("sourceLocMethod", m.sourceLocMethod)
	m.sourceLocSpacing = getInt("sourceLocSpacing", m.sourceLocSpacing)
	m.sourceLocParc = getInt("sourceLocParc", m.sourceLocParc)
	m.sourceLocReg = getFloat("sourceLocReg", m.sourceLocReg)
	m.sourceLocSnr = getFloat("sourceLocSnr", m.sourceLocSnr)
	m.sourceLocLoose = getFloat("sourceLocLoose", m.sourceLocLoose)
	m.sourceLocDepth = getFloat("sourceLocDepth", m.sourceLocDepth)
	m.sourceLocConnMethod = getInt("sourceLocConnMethod", m.sourceLocConnMethod)
	m.sourceLocSubject = getString("sourceLocSubject", m.sourceLocSubject)
	m.sourceLocTrans = getString("sourceLocTrans", m.sourceLocTrans)
	m.sourceLocBem = getString("sourceLocBem", m.sourceLocBem)
	m.sourceLocMindistMm = getFloat("sourceLocMindistMm", m.sourceLocMindistMm)
	m.sourceLocFmriEnabled = getBool("sourceLocFmriEnabled", m.sourceLocFmriEnabled)
	m.sourceLocFmriStatsMap = getString("sourceLocFmriStatsMap", m.sourceLocFmriStatsMap)
	m.sourceLocFmriProvenance = getInt("sourceLocFmriProvenance", m.sourceLocFmriProvenance)
	m.sourceLocFmriRequireProv = getBool("sourceLocFmriRequireProv", m.sourceLocFmriRequireProv)
	m.sourceLocFmriThreshold = getFloat("sourceLocFmriThreshold", m.sourceLocFmriThreshold)
	m.sourceLocFmriTail = getInt("sourceLocFmriTail", m.sourceLocFmriTail)
	m.sourceLocFmriMinClusterVox = getInt("sourceLocFmriMinClusterVox", m.sourceLocFmriMinClusterVox)
	m.sourceLocFmriMinClusterMM3 = getFloat("sourceLocFmriMinClusterMM3", m.sourceLocFmriMinClusterMM3)
	m.sourceLocFmriMaxClusters = getInt("sourceLocFmriMaxClusters", m.sourceLocFmriMaxClusters)
	m.sourceLocFmriMaxVoxPerClus = getInt("sourceLocFmriMaxVoxPerClus", m.sourceLocFmriMaxVoxPerClus)
	m.sourceLocFmriMaxTotalVox = getInt("sourceLocFmriMaxTotalVox", m.sourceLocFmriMaxTotalVox)
	m.sourceLocFmriRandomSeed = getInt("sourceLocFmriRandomSeed", m.sourceLocFmriRandomSeed)
	m.sourceLocCreateTrans = getBool("sourceLocCreateTrans", m.sourceLocCreateTrans)
	m.sourceLocAllowIdentityTrans = getBool("sourceLocAllowIdentityTrans", m.sourceLocAllowIdentityTrans)
	m.sourceLocCreateBemModel = getBool("sourceLocCreateBemModel", m.sourceLocCreateBemModel)
	m.sourceLocCreateBemSolution = getBool("sourceLocCreateBemSolution", m.sourceLocCreateBemSolution)

	// fMRI contrast builder
	m.sourceLocFmriContrastEnabled = getBool("sourceLocFmriContrastEnabled", m.sourceLocFmriContrastEnabled)
	m.sourceLocFmriContrastType = getInt("sourceLocFmriContrastType", m.sourceLocFmriContrastType)
	m.sourceLocFmriCondAColumn = getString("sourceLocFmriCondAColumn", m.sourceLocFmriCondAColumn)
	m.sourceLocFmriCondAValue = getString("sourceLocFmriCondAValue", m.sourceLocFmriCondAValue)
	m.sourceLocFmriCondBColumn = getString("sourceLocFmriCondBColumn", m.sourceLocFmriCondBColumn)
	m.sourceLocFmriCondBValue = getString("sourceLocFmriCondBValue", m.sourceLocFmriCondBValue)
	m.sourceLocFmriContrastFormula = getString("sourceLocFmriContrastFormula", m.sourceLocFmriContrastFormula)
	m.sourceLocFmriContrastName = getString("sourceLocFmriContrastName", m.sourceLocFmriContrastName)
	m.sourceLocFmriRunsToInclude = getString("sourceLocFmriRunsToInclude", m.sourceLocFmriRunsToInclude)
	m.sourceLocFmriAutoDetectRuns = getBool("sourceLocFmriAutoDetectRuns", m.sourceLocFmriAutoDetectRuns)
	m.sourceLocFmriHrfModel = getInt("sourceLocFmriHrfModel", m.sourceLocFmriHrfModel)
	m.sourceLocFmriDriftModel = getInt("sourceLocFmriDriftModel", m.sourceLocFmriDriftModel)
	m.sourceLocFmriHighPassHz = getFloat("sourceLocFmriHighPassHz", m.sourceLocFmriHighPassHz)
	m.sourceLocFmriLowPassHz = getFloat("sourceLocFmriLowPassHz", m.sourceLocFmriLowPassHz)
	m.sourceLocFmriClusterCorrection = getBool("sourceLocFmriClusterCorrection", m.sourceLocFmriClusterCorrection)
	m.sourceLocFmriClusterPThreshold = getFloat("sourceLocFmriClusterPThreshold", m.sourceLocFmriClusterPThreshold)
	m.sourceLocFmriOutputType = getInt("sourceLocFmriOutputType", m.sourceLocFmriOutputType)
	m.sourceLocFmriResampleToFS = getBool("sourceLocFmriResampleToFS", m.sourceLocFmriResampleToFS)
	m.sourceLocFmriWindowAName = getString("sourceLocFmriWindowAName", m.sourceLocFmriWindowAName)
	m.sourceLocFmriWindowATmin = getFloat("sourceLocFmriWindowATmin", m.sourceLocFmriWindowATmin)
	m.sourceLocFmriWindowATmax = getFloat("sourceLocFmriWindowATmax", m.sourceLocFmriWindowATmax)
	m.sourceLocFmriWindowBName = getString("sourceLocFmriWindowBName", m.sourceLocFmriWindowBName)
	m.sourceLocFmriWindowBTmin = getFloat("sourceLocFmriWindowBTmin", m.sourceLocFmriWindowBTmin)
	m.sourceLocFmriWindowBTmax = getFloat("sourceLocFmriWindowBTmax", m.sourceLocFmriWindowBTmax)

	// Aggregation/storage
	m.aggregationMethod = getInt("aggregationMethod", m.aggregationMethod)
	m.minEpochsForFeatures = getInt("minEpochsForFeatures", m.minEpochsForFeatures)
	m.featAnalysisMode = getInt("featAnalysisMode", m.featAnalysisMode)
	m.featComputeChangeScores = getBool("featComputeChangeScores", m.featComputeChangeScores)
	m.featSaveTfrWithSidecar = getBool("featSaveTfrWithSidecar", m.featSaveTfrWithSidecar)
	m.featNJobsBands = getInt("featNJobsBands", m.featNJobsBands)
	m.featNJobsConnectivity = getInt("featNJobsConnectivity", m.featNJobsConnectivity)
	m.featNJobsAperiodic = getInt("featNJobsAperiodic", m.featNJobsAperiodic)
	m.featNJobsComplexity = getInt("featNJobsComplexity", m.featNJobsComplexity)
	m.saveSubjectLevelFeatures = getBool("saveSubjectLevelFeatures", m.saveSubjectLevelFeatures)
	m.featAlsoSaveCsv = getBool("featAlsoSaveCsv", m.featAlsoSaveCsv)

	// Spatial transform
	m.spatialTransform = getInt("spatialTransform", m.spatialTransform)
	m.spatialTransformLambda2 = getFloat("spatialTransformLambda2", m.spatialTransformLambda2)
	m.spatialTransformStiffness = getFloat("spatialTransformStiffness", m.spatialTransformStiffness)

	// TFR
	m.tfrFreqMin = getFloat("tfrFreqMin", m.tfrFreqMin)
	m.tfrFreqMax = getFloat("tfrFreqMax", m.tfrFreqMax)
	m.tfrNFreqs = getInt("tfrNFreqs", m.tfrNFreqs)
	m.tfrMinCycles = getFloat("tfrMinCycles", m.tfrMinCycles)
	m.tfrMaxCycles = getFloat("tfrMaxCycles", m.tfrMaxCycles)
	m.tfrNCyclesFactor = getFloat("tfrNCyclesFactor", m.tfrNCyclesFactor)
	m.tfrDecim = getInt("tfrDecim", m.tfrDecim)
	m.tfrDecimPower = getInt("tfrDecimPower", m.tfrDecimPower)
	m.tfrDecimPhase = getInt("tfrDecimPhase", m.tfrDecimPhase)
	m.tfrWorkers = getInt("tfrWorkers", m.tfrWorkers)

	// Quality features
	m.qualityPsdMethod = getInt("qualityPsdMethod", m.qualityPsdMethod)
	m.qualityFmin = getFloat("qualityFmin", m.qualityFmin)
	m.qualityFmax = getFloat("qualityFmax", m.qualityFmax)
	m.qualityNfft = getInt("qualityNfft", m.qualityNfft)
	m.qualityExcludeLineNoise = getBool("qualityExcludeLineNoise", m.qualityExcludeLineNoise)
	m.qualityLineNoiseFreq = getFloat("qualityLineNoiseFreq", m.qualityLineNoiseFreq)
	m.qualityLineNoiseWidthHz = getFloat("qualityLineNoiseWidthHz", m.qualityLineNoiseWidthHz)
	m.qualityLineNoiseHarmonics = getInt("qualityLineNoiseHarmonics", m.qualityLineNoiseHarmonics)
	m.qualitySnrSignalBandMin = getFloat("qualitySnrSignalBandMin", m.qualitySnrSignalBandMin)
	m.qualitySnrSignalBandMax = getFloat("qualitySnrSignalBandMax", m.qualitySnrSignalBandMax)
	m.qualitySnrNoiseBandMin = getFloat("qualitySnrNoiseBandMin", m.qualitySnrNoiseBandMin)
	m.qualitySnrNoiseBandMax = getFloat("qualitySnrNoiseBandMax", m.qualitySnrNoiseBandMax)
	m.qualityMuscleBandMin = getFloat("qualityMuscleBandMin", m.qualityMuscleBandMin)
	m.qualityMuscleBandMax = getFloat("qualityMuscleBandMax", m.qualityMuscleBandMax)

	// ERDS configuration
	m.erdsUseLogRatio = getBool("erdsUseLogRatio", m.erdsUseLogRatio)
	m.erdsMinBaselinePower = getFloat("erdsMinBaselinePower", m.erdsMinBaselinePower)
	m.erdsMinActivePower = getFloat("erdsMinActivePower", m.erdsMinActivePower)
	m.erdsMinSegmentSec = getFloat("erdsMinSegmentSec", m.erdsMinSegmentSec)
	m.erdsBandsSpec = getString("erdsBandsSpec", m.erdsBandsSpec)

	// Asymmetry & Ratios
	m.asymmetryChannelPairsSpec = getString("asymmetryChannelPairsSpec", m.asymmetryChannelPairsSpec)
	m.asymmetryMinSegmentSec = getFloat("asymmetryMinSegmentSec", m.asymmetryMinSegmentSec)
	m.asymmetryMinCyclesAtFmin = getFloat("asymmetryMinCyclesAtFmin", m.asymmetryMinCyclesAtFmin)
	m.asymmetrySkipInvalidSegments = getBool("asymmetrySkipInvalidSegments", m.asymmetrySkipInvalidSegments)
	m.asymmetryEmitActivationConvention = getBool("asymmetryEmitActivationConvention", m.asymmetryEmitActivationConvention)
	m.asymmetryActivationBandsSpec = getString("asymmetryActivationBandsSpec", m.asymmetryActivationBandsSpec)
	m.ratiosMinSegmentSec = getFloat("ratiosMinSegmentSec", m.ratiosMinSegmentSec)
	m.ratiosMinCyclesAtFmin = getFloat("ratiosMinCyclesAtFmin", m.ratiosMinCyclesAtFmin)
	m.ratiosSkipInvalidSegments = getBool("ratiosSkipInvalidSegments", m.ratiosSkipInvalidSegments)
	m.ratioSource = getInt("ratioSource", m.ratioSource)

	// IAF
	m.iafEnabled = getBool("iafEnabled", m.iafEnabled)
	m.iafAlphaWidthHz = getFloat("iafAlphaWidthHz", m.iafAlphaWidthHz)
	m.iafSearchRangeMin = getFloat("iafSearchRangeMin", m.iafSearchRangeMin)
	m.iafSearchRangeMax = getFloat("iafSearchRangeMax", m.iafSearchRangeMax)
	m.iafMinProminence = getFloat("iafMinProminence", m.iafMinProminence)
	m.iafRoisSpec = getString("iafRoisSpec", m.iafRoisSpec)
	m.iafMinCyclesAtFmin = getFloat("iafMinCyclesAtFmin", m.iafMinCyclesAtFmin)
	m.iafMinBaselineSec = getFloat("iafMinBaselineSec", m.iafMinBaselineSec)
	m.iafAllowFullFallback = getBool("iafAllowFullFallback", m.iafAllowFullFallback)
	m.iafAllowAllChannelsFallback = getBool("iafAllowAllChannelsFallback", m.iafAllowAllChannelsFallback)

	// Directed connectivity
	m.directedConnEnabled = getBool("directedConnEnabled", m.directedConnEnabled)
	m.directedConnOutputLevel = getInt("directedConnOutputLevel", m.directedConnOutputLevel)
	m.directedConnMvarOrder = getInt("directedConnMvarOrder", m.directedConnMvarOrder)
	m.directedConnNFreqs = getInt("directedConnNFreqs", m.directedConnNFreqs)
	m.directedConnMinSegSamples = getInt("directedConnMinSegSamples", m.directedConnMinSegSamples)

	// Behavior
	m.correlationMethod = getString("correlationMethod", m.correlationMethod)
	m.robustCorrelation = getInt("robustCorrelation", m.robustCorrelation)
	m.bootstrapSamples = getInt("bootstrapSamples", m.bootstrapSamples)
	m.nPermutations = getInt("nPermutations", m.nPermutations)
	m.rngSeed = getInt("rngSeed", m.rngSeed)
	m.fdrAlpha = getFloat("fdrAlpha", m.fdrAlpha)
	m.behaviorNJobs = getInt("behaviorNJobs", m.behaviorNJobs)
	m.alsoSaveCsv = getBool("alsoSaveCsv", m.alsoSaveCsv)
	m.behaviorOverwrite = getBool("behaviorOverwrite", m.behaviorOverwrite)
	m.runAdjustmentColumn = getString("runAdjustmentColumn", m.runAdjustmentColumn)
	m.runAdjustmentEnabled = getBool("runAdjustmentEnabled", m.runAdjustmentEnabled)
	m.runAdjustmentIncludeInCorrelations = getBool("runAdjustmentIncludeInCorrelations", m.runAdjustmentIncludeInCorrelations)
	m.runAdjustmentMaxDummies = getInt("runAdjustmentMaxDummies", m.runAdjustmentMaxDummies)
	m.conditionCompareColumn = getString("conditionCompareColumn", m.conditionCompareColumn)
	m.conditionCompareValues = getString("conditionCompareValues", m.conditionCompareValues)
	m.conditionCompareWindows = getString("conditionCompareWindows", m.conditionCompareWindows)
	m.conditionFailFast = getBool("conditionFailFast", m.conditionFailFast)
	m.conditionEffectThreshold = getFloat("conditionEffectThreshold", m.conditionEffectThreshold)
	m.conditionOverwrite = getBool("conditionOverwrite", m.conditionOverwrite)
	m.temporalTargetColumn = getString("temporalTargetColumn", m.temporalTargetColumn)
	m.temporalConditionColumn = getString("temporalConditionColumn", m.temporalConditionColumn)
	m.temporalConditionValues = getString("temporalConditionValues", m.temporalConditionValues)
	m.temporalSplitByCondition = getBool("temporalSplitByCondition", m.temporalSplitByCondition)
	m.temporalIncludeROIAverages = getBool("temporalIncludeROIAverages", m.temporalIncludeROIAverages)
	m.temporalIncludeTFGrid = getBool("temporalIncludeTFGrid", m.temporalIncludeTFGrid)
	m.temporalFeaturePowerEnabled = getBool("temporalFeaturePowerEnabled", m.temporalFeaturePowerEnabled)
	m.temporalFeatureITPCEnabled = getBool("temporalFeatureITPCEnabled", m.temporalFeatureITPCEnabled)
	m.temporalFeatureERDSEnabled = getBool("temporalFeatureERDSEnabled", m.temporalFeatureERDSEnabled)
	m.temporalITPCBaselineCorrection = getBool("temporalITPCBaselineCorrection", m.temporalITPCBaselineCorrection)
	m.temporalITPCBaselineMin = getFloat("temporalITPCBaselineMin", m.temporalITPCBaselineMin)
	m.temporalITPCBaselineMax = getFloat("temporalITPCBaselineMax", m.temporalITPCBaselineMax)
	m.temporalERDSBaselineMin = getFloat("temporalERDSBaselineMin", m.temporalERDSBaselineMin)
	m.temporalERDSBaselineMax = getFloat("temporalERDSBaselineMax", m.temporalERDSBaselineMax)
	m.temporalERDSMethod = getInt("temporalERDSMethod", m.temporalERDSMethod)
	m.controlTemperature = getBool("controlTemperature", m.controlTemperature)
	m.controlTrialOrder = getBool("controlTrialOrder", m.controlTrialOrder)
	m.behaviorComputeChangeScores = getBool("behaviorComputeChangeScores", m.behaviorComputeChangeScores)
	m.behaviorComputeBayesFactors = getBool("behaviorComputeBayesFactors", m.behaviorComputeBayesFactors)
	m.behaviorComputeLosoStability = getBool("behaviorComputeLosoStability", m.behaviorComputeLosoStability)

	// Trial table
	m.trialTableFormat = getInt("trialTableFormat", m.trialTableFormat)
	m.trialTableAddLagFeatures = getBool("trialTableAddLagFeatures", m.trialTableAddLagFeatures)
	m.trialOrderMaxMissingFraction = getFloat("trialOrderMaxMissingFraction", m.trialOrderMaxMissingFraction)

	// Feature QC
	m.featureQCEnabled = getBool("featureQCEnabled", m.featureQCEnabled)
	m.featureQCMaxMissingPct = getFloat("featureQCMaxMissingPct", m.featureQCMaxMissingPct)
	m.featureQCMinVariance = getFloat("featureQCMinVariance", m.featureQCMinVariance)
	m.featureQCCheckWithinRunVariance = getBool("featureQCCheckWithinRunVariance", m.featureQCCheckWithinRunVariance)
	// Backward-compat: we accept older persisted keys but ignore them.
	_ = getBool("trialTableIncludeFeatures", true)
	_ = getBool("trialTableIncludeCovars", true)
	_ = getBool("trialTableIncludeEvents", true)
	_ = getString("trialTableExtraEventCols", "")
	_ = getFloat("trialTableHighMissingFrac", 0.5)

	// Pain residual
	m.painResidualEnabled = getBool("painResidualEnabled", m.painResidualEnabled)
	m.painResidualMethod = getInt("painResidualMethod", m.painResidualMethod)
	m.painResidualPolyDegree = getInt("painResidualPolyDegree", m.painResidualPolyDegree)
	m.painResidualSplineDfCandidates = getString("painResidualSplineDfCandidates", m.painResidualSplineDfCandidates)
	m.painResidualModelCompareEnabled = getBool("painResidualModelCompareEnabled", m.painResidualModelCompareEnabled)
	m.painResidualModelComparePolyDegrees = getString("painResidualModelComparePolyDegrees", m.painResidualModelComparePolyDegrees)
	m.painResidualBreakpointEnabled = getBool("painResidualBreakpointEnabled", m.painResidualBreakpointEnabled)
	m.painResidualBreakpointCandidates = getInt("painResidualBreakpointCandidates", m.painResidualBreakpointCandidates)
	m.painResidualBreakpointQlow = getFloat("painResidualBreakpointQlow", m.painResidualBreakpointQlow)
	m.painResidualBreakpointQhigh = getFloat("painResidualBreakpointQhigh", m.painResidualBreakpointQhigh)
	m.painResidualCrossfitEnabled = getBool("painResidualCrossfitEnabled", m.painResidualCrossfitEnabled)
	m.painResidualCrossfitGroupColumn = getString("painResidualCrossfitGroupColumn", m.painResidualCrossfitGroupColumn)
	m.painResidualCrossfitNSplits = getInt("painResidualCrossfitNSplits", m.painResidualCrossfitNSplits)
	m.painResidualCrossfitMethod = getInt("painResidualCrossfitMethod", m.painResidualCrossfitMethod)
	m.painResidualCrossfitSplineKnots = getInt("painResidualCrossfitSplineKnots", m.painResidualCrossfitSplineKnots)

	// Regression
	m.regressionOutcome = getInt("regressionOutcome", m.regressionOutcome)
	m.regressionIncludeTemperature = getBool("regressionIncludeTemperature", m.regressionIncludeTemperature)
	m.regressionTempControl = getInt("regressionTempControl", m.regressionTempControl)
	m.regressionTempSplineKnots = getInt("regressionTempSplineKnots", m.regressionTempSplineKnots)
	m.regressionTempSplineQlow = getFloat("regressionTempSplineQlow", m.regressionTempSplineQlow)
	m.regressionTempSplineQhigh = getFloat("regressionTempSplineQhigh", m.regressionTempSplineQhigh)
	m.regressionIncludeTrialOrder = getBool("regressionIncludeTrialOrder", m.regressionIncludeTrialOrder)
	m.regressionIncludePrev = getBool("regressionIncludePrev", m.regressionIncludePrev)
	m.regressionIncludeRunBlock = getBool("regressionIncludeRunBlock", m.regressionIncludeRunBlock)
	m.regressionIncludeInteraction = getBool("regressionIncludeInteraction", m.regressionIncludeInteraction)
	m.regressionStandardize = getBool("regressionStandardize", m.regressionStandardize)
	m.regressionPermutations = getInt("regressionPermutations", m.regressionPermutations)
	m.regressionMaxFeatures = getInt("regressionMaxFeatures", m.regressionMaxFeatures)

	// Models
	m.modelsIncludeTemperature = getBool("modelsIncludeTemperature", m.modelsIncludeTemperature)
	m.modelsTempControl = getInt("modelsTempControl", m.modelsTempControl)
	m.modelsTempSplineKnots = getInt("modelsTempSplineKnots", m.modelsTempSplineKnots)
	m.modelsTempSplineQlow = getFloat("modelsTempSplineQlow", m.modelsTempSplineQlow)
	m.modelsTempSplineQhigh = getFloat("modelsTempSplineQhigh", m.modelsTempSplineQhigh)
	m.modelsIncludeTrialOrder = getBool("modelsIncludeTrialOrder", m.modelsIncludeTrialOrder)
	m.modelsIncludePrev = getBool("modelsIncludePrev", m.modelsIncludePrev)
	m.modelsIncludeRunBlock = getBool("modelsIncludeRunBlock", m.modelsIncludeRunBlock)
	m.modelsIncludeInteraction = getBool("modelsIncludeInteraction", m.modelsIncludeInteraction)
	m.modelsStandardize = getBool("modelsStandardize", m.modelsStandardize)
	m.modelsMaxFeatures = getInt("modelsMaxFeatures", m.modelsMaxFeatures)
	m.modelsOutcomeRating = getBool("modelsOutcomeRating", m.modelsOutcomeRating)
	m.modelsOutcomePainResidual = getBool("modelsOutcomePainResidual", m.modelsOutcomePainResidual)
	m.modelsOutcomeTemperature = getBool("modelsOutcomeTemperature", m.modelsOutcomeTemperature)
	m.modelsOutcomePainBinary = getBool("modelsOutcomePainBinary", m.modelsOutcomePainBinary)
	m.modelsFamilyOLS = getBool("modelsFamilyOLS", m.modelsFamilyOLS)
	m.modelsFamilyRobust = getBool("modelsFamilyRobust", m.modelsFamilyRobust)
	m.modelsFamilyQuantile = getBool("modelsFamilyQuantile", m.modelsFamilyQuantile)
	m.modelsFamilyLogit = getBool("modelsFamilyLogit", m.modelsFamilyLogit)
	m.modelsBinaryOutcome = getInt("modelsBinaryOutcome", m.modelsBinaryOutcome)

	// Stability
	m.stabilityMethod = getInt("stabilityMethod", m.stabilityMethod)
	m.stabilityOutcome = getInt("stabilityOutcome", m.stabilityOutcome)
	m.stabilityGroupColumn = getInt("stabilityGroupColumn", m.stabilityGroupColumn)
	m.stabilityPartialTemp = getBool("stabilityPartialTemp", m.stabilityPartialTemp)
	m.stabilityMaxFeatures = getInt("stabilityMaxFeatures", m.stabilityMaxFeatures)
	m.stabilityAlpha = getFloat("stabilityAlpha", m.stabilityAlpha)

	// Consistency & influence
	m.consistencyEnabled = getBool("consistencyEnabled", m.consistencyEnabled)
	m.influenceOutcomeRating = getBool("influenceOutcomeRating", m.influenceOutcomeRating)
	m.influenceOutcomePainResidual = getBool("influenceOutcomePainResidual", m.influenceOutcomePainResidual)
	m.influenceOutcomeTemperature = getBool("influenceOutcomeTemperature", m.influenceOutcomeTemperature)
	m.influenceMaxFeatures = getInt("influenceMaxFeatures", m.influenceMaxFeatures)
	m.influenceIncludeTemperature = getBool("influenceIncludeTemperature", m.influenceIncludeTemperature)
	m.influenceTempControl = getInt("influenceTempControl", m.influenceTempControl)
	m.influenceTempSplineKnots = getInt("influenceTempSplineKnots", m.influenceTempSplineKnots)
	m.influenceTempSplineQlow = getFloat("influenceTempSplineQlow", m.influenceTempSplineQlow)
	m.influenceTempSplineQhigh = getFloat("influenceTempSplineQhigh", m.influenceTempSplineQhigh)
	m.influenceTempSplineMinN = getInt("influenceTempSplineMinN", m.influenceTempSplineMinN)
	m.influenceIncludeTrialOrder = getBool("influenceIncludeTrialOrder", m.influenceIncludeTrialOrder)
	m.influenceIncludeRunBlock = getBool("influenceIncludeRunBlock", m.influenceIncludeRunBlock)
	m.influenceIncludeInteraction = getBool("influenceIncludeInteraction", m.influenceIncludeInteraction)
	m.influenceStandardize = getBool("influenceStandardize", m.influenceStandardize)
	m.influenceCooksThreshold = getFloat("influenceCooksThreshold", m.influenceCooksThreshold)
	m.influenceLeverageThreshold = getFloat("influenceLeverageThreshold", m.influenceLeverageThreshold)

	// Correlations
	m.correlationsTargetRating = getBool("correlationsTargetRating", m.correlationsTargetRating)
	m.correlationsTargetTemperature = getBool("correlationsTargetTemperature", m.correlationsTargetTemperature)
	m.correlationsTargetPainResidual = getBool("correlationsTargetPainResidual", m.correlationsTargetPainResidual)
	m.correlationsTypesSpec = getString("correlationsTypesSpec", m.correlationsTypesSpec)
	m.correlationsTargetColumn = getString("correlationsTargetColumn", m.correlationsTargetColumn)
	m.correlationsPreferPainResidual = getBool("correlationsPreferPainResidual", m.correlationsPreferPainResidual)
	m.correlationsUseCrossfitResidual = getBool("correlationsUseCrossfitResidual", m.correlationsUseCrossfitResidual)
	m.correlationsPrimaryUnit = getInt("correlationsPrimaryUnit", m.correlationsPrimaryUnit)
	m.correlationsPermutationPrimary = getBool("correlationsPermutationPrimary", m.correlationsPermutationPrimary)
	m.groupLevelBlockPermutation = getBool("groupLevelBlockPermutation", m.groupLevelBlockPermutation)

	// Mixed Effects & Mediation
	m.mixedEffectsType = getInt("mixedEffectsType", m.mixedEffectsType)
	m.mixedMaxFeatures = getInt("mixedMaxFeatures", m.mixedMaxFeatures)
	m.mediationMinEffect = getFloat("mediationMinEffect", m.mediationMinEffect)
	m.mediationBootstrap = getInt("mediationBootstrap", m.mediationBootstrap)
	m.mediationMaxMediatorsEnabled = getBool("mediationMaxMediatorsEnabled", m.mediationMaxMediatorsEnabled)
	m.mediationMaxMediators = getInt("mediationMaxMediators", m.mediationMaxMediators)
	m.mediationPermutations = getInt("mediationPermutations", m.mediationPermutations)
	m.moderationMaxFeaturesEnabled = getBool("moderationMaxFeaturesEnabled", m.moderationMaxFeaturesEnabled)
	m.moderationMaxFeatures = getInt("moderationMaxFeatures", m.moderationMaxFeatures)
	m.moderationPermutations = getInt("moderationPermutations", m.moderationPermutations)

	// Cluster tests
	m.clusterThreshold = getFloat("clusterThreshold", m.clusterThreshold)
	m.clusterMinSize = getInt("clusterMinSize", m.clusterMinSize)
	m.clusterTail = getInt("clusterTail", m.clusterTail)
	m.clusterConditionColumn = getString("clusterConditionColumn", m.clusterConditionColumn)
	m.clusterConditionValues = getString("clusterConditionValues", m.clusterConditionValues)

	// Temporal
	m.temporalResolutionMs = getInt("temporalResolutionMs", m.temporalResolutionMs)
	m.temporalSmoothMs = getInt("temporalSmoothMs", m.temporalSmoothMs)
	m.temporalTimeMinMs = getInt("temporalTimeMinMs", m.temporalTimeMinMs)
	m.temporalTimeMaxMs = getInt("temporalTimeMaxMs", m.temporalTimeMaxMs)

	// Preprocessing (EEG)
	m.prepUsePyprep = getBool("prepUsePyprep", m.prepUsePyprep)
	m.prepUseIcalabel = getBool("prepUseIcalabel", m.prepUseIcalabel)
	m.prepNJobs = getInt("prepNJobs", m.prepNJobs)
	m.prepMontage = getString("prepMontage", m.prepMontage)
	m.prepResample = getInt("prepResample", m.prepResample)
	m.prepLFreq = getFloat("prepLFreq", m.prepLFreq)
	m.prepHFreq = getFloat("prepHFreq", m.prepHFreq)
	m.prepNotch = getInt("prepNotch", m.prepNotch)
	m.prepLineFreq = getInt("prepLineFreq", m.prepLineFreq)
	m.prepChTypes = getString("prepChTypes", m.prepChTypes)
	m.prepEegReference = getString("prepEegReference", m.prepEegReference)
	m.prepEogChannels = getString("prepEogChannels", m.prepEogChannels)
	m.prepRandomState = getInt("prepRandomState", m.prepRandomState)
	m.prepTaskIsRest = getBool("prepTaskIsRest", m.prepTaskIsRest)
	m.prepZaplineFline = getFloat("prepZaplineFline", m.prepZaplineFline)
	m.prepFindBreaks = getBool("prepFindBreaks", m.prepFindBreaks)
	m.prepRansac = getBool("prepRansac", m.prepRansac)
	m.prepRepeats = getInt("prepRepeats", m.prepRepeats)
	m.prepAverageReref = getBool("prepAverageReref", m.prepAverageReref)
	m.prepFileExtension = getString("prepFileExtension", m.prepFileExtension)
	m.prepConsiderPreviousBads = getBool("prepConsiderPreviousBads", m.prepConsiderPreviousBads)
	m.prepOverwriteChansTsv = getBool("prepOverwriteChansTsv", m.prepOverwriteChansTsv)
	m.prepDeleteBreaks = getBool("prepDeleteBreaks", m.prepDeleteBreaks)
	m.prepBreaksMinLength = getInt("prepBreaksMinLength", m.prepBreaksMinLength)
	m.prepTStartAfterPrevious = getInt("prepTStartAfterPrevious", m.prepTStartAfterPrevious)
	m.prepTStopBeforeNext = getInt("prepTStopBeforeNext", m.prepTStopBeforeNext)
	m.prepSpatialFilter = getInt("prepSpatialFilter", m.prepSpatialFilter)
	m.prepICAAlgorithm = getInt("prepICAAlgorithm", m.prepICAAlgorithm)
	m.prepICAComp = getFloat("prepICAComp", m.prepICAComp)
	m.prepICALFreq = getFloat("prepICALFreq", m.prepICALFreq)
	m.prepICARejThresh = getFloat("prepICARejThresh", m.prepICARejThresh)
	m.prepProbThresh = getFloat("prepProbThresh", m.prepProbThresh)
	m.prepKeepMnebidsBads = getBool("prepKeepMnebidsBads", m.prepKeepMnebidsBads)
	m.prepConditions = getString("prepConditions", m.prepConditions)
	m.prepEpochsTmin = getFloat("prepEpochsTmin", m.prepEpochsTmin)
	m.prepEpochsTmax = getFloat("prepEpochsTmax", m.prepEpochsTmax)
	m.prepEpochsBaselineStart = getFloat("prepEpochsBaselineStart", m.prepEpochsBaselineStart)
	m.prepEpochsBaselineEnd = getFloat("prepEpochsBaselineEnd", m.prepEpochsBaselineEnd)
	m.prepEpochsNoBaseline = getBool("prepEpochsNoBaseline", m.prepEpochsNoBaseline)
	m.prepEpochsReject = getFloat("prepEpochsReject", m.prepEpochsReject)
	m.prepWriteCleanEvents = getBool("prepWriteCleanEvents", m.prepWriteCleanEvents)
	m.prepOverwriteCleanEvents = getBool("prepOverwriteCleanEvents", m.prepOverwriteCleanEvents)
	m.prepCleanEventsStrict = getBool("prepCleanEventsStrict", m.prepCleanEventsStrict)
	m.prepRejectMethod = getInt("prepRejectMethod", m.prepRejectMethod)
	m.prepRunSourceEstimation = getBool("prepRunSourceEstimation", m.prepRunSourceEstimation)
	m.icaLabelsToKeep = getString("icaLabelsToKeep", m.icaLabelsToKeep)

	// Preprocessing UI State (EEG)
	m.prepGroupStagesExpanded = getBool("prepGroupStagesExpanded", m.prepGroupStagesExpanded)
	m.prepGroupGeneralExpanded = getBool("prepGroupGeneralExpanded", m.prepGroupGeneralExpanded)
	m.prepGroupFilteringExpanded = getBool("prepGroupFilteringExpanded", m.prepGroupFilteringExpanded)
	m.prepGroupPyprepExpanded = getBool("prepGroupPyprepExpanded", m.prepGroupPyprepExpanded)
	m.prepGroupICAExpanded = getBool("prepGroupICAExpanded", m.prepGroupICAExpanded)
	m.prepGroupEpochingExpanded = getBool("prepGroupEpochingExpanded", m.prepGroupEpochingExpanded)

	// fMRI Preprocessing
	m.fmriEngineIndex = getInt("fmriEngineIndex", m.fmriEngineIndex)
	m.fmriFmriprepImage = getString("fmriFmriprepImage", m.fmriFmriprepImage)
	m.fmriFmriprepOutputDir = getString("fmriFmriprepOutputDir", m.fmriFmriprepOutputDir)
	m.fmriFmriprepWorkDir = getString("fmriFmriprepWorkDir", m.fmriFmriprepWorkDir)
	m.fmriFreesurferLicenseFile = getString("fmriFreesurferLicenseFile", m.fmriFreesurferLicenseFile)
	m.fmriFreesurferSubjectsDir = getString("fmriFreesurferSubjectsDir", m.fmriFreesurferSubjectsDir)
	m.fmriOutputSpacesSpec = getString("fmriOutputSpacesSpec", m.fmriOutputSpacesSpec)
	m.fmriIgnoreSpec = getString("fmriIgnoreSpec", m.fmriIgnoreSpec)
	m.fmriBidsFilterFile = getString("fmriBidsFilterFile", m.fmriBidsFilterFile)
	m.fmriExtraArgs = getString("fmriExtraArgs", m.fmriExtraArgs)
	m.fmriUseAroma = getBool("fmriUseAroma", m.fmriUseAroma)
	m.fmriSkipBidsValidation = getBool("fmriSkipBidsValidation", m.fmriSkipBidsValidation)
	m.fmriStopOnFirstCrash = getBool("fmriStopOnFirstCrash", m.fmriStopOnFirstCrash)
	m.fmriCleanWorkdir = getBool("fmriCleanWorkdir", m.fmriCleanWorkdir)
	m.fmriSkipReconstruction = getBool("fmriSkipReconstruction", m.fmriSkipReconstruction)
	m.fmriMemMb = getInt("fmriMemMb", m.fmriMemMb)
	m.fmriNThreads = getInt("fmriNThreads", m.fmriNThreads)
	m.fmriOmpNThreads = getInt("fmriOmpNThreads", m.fmriOmpNThreads)
	m.fmriLowMem = getBool("fmriLowMem", m.fmriLowMem)
	m.fmriLongitudinal = getBool("fmriLongitudinal", m.fmriLongitudinal)
	m.fmriCiftiOutputIndex = getInt("fmriCiftiOutputIndex", m.fmriCiftiOutputIndex)
	m.fmriSkullStripTemplate = getString("fmriSkullStripTemplate", m.fmriSkullStripTemplate)
	m.fmriSkullStripFixedSeed = getBool("fmriSkullStripFixedSeed", m.fmriSkullStripFixedSeed)
	m.fmriRandomSeed = getInt("fmriRandomSeed", m.fmriRandomSeed)
	m.fmriDummyScans = getInt("fmriDummyScans", m.fmriDummyScans)
	m.fmriBold2T1wInitIndex = getInt("fmriBold2T1wInitIndex", m.fmriBold2T1wInitIndex)
	m.fmriBold2T1wDof = getInt("fmriBold2T1wDof", m.fmriBold2T1wDof)
	m.fmriSliceTimeRef = getFloat("fmriSliceTimeRef", m.fmriSliceTimeRef)
	m.fmriFdSpikeThreshold = getFloat("fmriFdSpikeThreshold", m.fmriFdSpikeThreshold)
	m.fmriDvarsSpikeThreshold = getFloat("fmriDvarsSpikeThreshold", m.fmriDvarsSpikeThreshold)
	m.fmriMeOutputEchos = getBool("fmriMeOutputEchos", m.fmriMeOutputEchos)
	m.fmriMedialSurfaceNan = getBool("fmriMedialSurfaceNan", m.fmriMedialSurfaceNan)
	m.fmriNoMsm = getBool("fmriNoMsm", m.fmriNoMsm)
	m.fmriLevelIndex = getInt("fmriLevelIndex", m.fmriLevelIndex)
	m.fmriTaskId = getString("fmriTaskId", m.fmriTaskId)

	// fMRI Analysis
	m.fmriAnalysisInputSourceIndex = getInt("fmriAnalysisInputSourceIndex", m.fmriAnalysisInputSourceIndex)
	m.fmriAnalysisFmriprepSpace = getString("fmriAnalysisFmriprepSpace", m.fmriAnalysisFmriprepSpace)
	m.fmriAnalysisRequireFmriprep = getBool("fmriAnalysisRequireFmriprep", m.fmriAnalysisRequireFmriprep)
	m.fmriAnalysisRunsSpec = getString("fmriAnalysisRunsSpec", m.fmriAnalysisRunsSpec)
	m.fmriAnalysisContrastType = getInt("fmriAnalysisContrastType", m.fmriAnalysisContrastType)
	m.fmriAnalysisCondAColumn = getString("fmriAnalysisCondAColumn", "trial_type")
	m.fmriAnalysisCondAValue = getString("fmriAnalysisCondAValue", m.fmriAnalysisCondAValue)
	m.fmriAnalysisCondBColumn = getString("fmriAnalysisCondBColumn", "trial_type")
	m.fmriAnalysisCondBValue = getString("fmriAnalysisCondBValue", m.fmriAnalysisCondBValue)
	m.fmriAnalysisContrastName = getString("fmriAnalysisContrastName", m.fmriAnalysisContrastName)
	m.fmriAnalysisFormula = getString("fmriAnalysisFormula", m.fmriAnalysisFormula)
	m.fmriAnalysisEventsToModel = getString("fmriAnalysisEventsToModel", m.fmriAnalysisEventsToModel)
	m.fmriAnalysisHrfModel = getInt("fmriAnalysisHrfModel", m.fmriAnalysisHrfModel)
	m.fmriAnalysisDriftModel = getInt("fmriAnalysisDriftModel", m.fmriAnalysisDriftModel)
	m.fmriAnalysisHighPassHz = getFloat("fmriAnalysisHighPassHz", m.fmriAnalysisHighPassHz)
	m.fmriAnalysisLowPassHz = getFloat("fmriAnalysisLowPassHz", m.fmriAnalysisLowPassHz)
	m.fmriAnalysisSmoothingFwhm = getFloat("fmriAnalysisSmoothingFwhm", m.fmriAnalysisSmoothingFwhm)
	m.fmriAnalysisOutputType = getInt("fmriAnalysisOutputType", m.fmriAnalysisOutputType)
	m.fmriAnalysisOutputDir = getString("fmriAnalysisOutputDir", m.fmriAnalysisOutputDir)
	m.fmriAnalysisResampleToFS = getBool("fmriAnalysisResampleToFS", m.fmriAnalysisResampleToFS)
	m.fmriAnalysisFreesurferDir = getString("fmriAnalysisFreesurferDir", m.fmriAnalysisFreesurferDir)
	m.fmriAnalysisConfoundsStrategy = getInt("fmriAnalysisConfoundsStrategy", m.fmriAnalysisConfoundsStrategy)
	m.fmriAnalysisWriteDesignMatrix = getBool("fmriAnalysisWriteDesignMatrix", m.fmriAnalysisWriteDesignMatrix)

	m.fmriAnalysisGroupInputExpanded = getBool("fmriAnalysisGroupInputExpanded", m.fmriAnalysisGroupInputExpanded)
	m.fmriAnalysisGroupContrastExpanded = getBool("fmriAnalysisGroupContrastExpanded", m.fmriAnalysisGroupContrastExpanded)
	m.fmriAnalysisGroupGLMExpanded = getBool("fmriAnalysisGroupGLMExpanded", m.fmriAnalysisGroupGLMExpanded)
	m.fmriAnalysisGroupConfoundsExpanded = getBool("fmriAnalysisGroupConfoundsExpanded", m.fmriAnalysisGroupConfoundsExpanded)
	m.fmriAnalysisGroupOutputExpanded = getBool("fmriAnalysisGroupOutputExpanded", m.fmriAnalysisGroupOutputExpanded)
	m.fmriAnalysisGroupPlottingExpanded = getBool("fmriAnalysisGroupPlottingExpanded", m.fmriAnalysisGroupPlottingExpanded)

	m.fmriAnalysisPlotsEnabled = getBool("fmriAnalysisPlotsEnabled", m.fmriAnalysisPlotsEnabled)
	m.fmriAnalysisPlotHTML = getBool("fmriAnalysisPlotHTML", m.fmriAnalysisPlotHTML)
	m.fmriAnalysisPlotSpaceIndex = getInt("fmriAnalysisPlotSpaceIndex", m.fmriAnalysisPlotSpaceIndex)
	m.fmriAnalysisPlotThresholdModeIndex = getInt("fmriAnalysisPlotThresholdModeIndex", m.fmriAnalysisPlotThresholdModeIndex)
	m.fmriAnalysisPlotZThreshold = getFloat("fmriAnalysisPlotZThreshold", m.fmriAnalysisPlotZThreshold)
	m.fmriAnalysisPlotFdrQ = getFloat("fmriAnalysisPlotFdrQ", m.fmriAnalysisPlotFdrQ)
	m.fmriAnalysisPlotClusterMinVoxels = getInt("fmriAnalysisPlotClusterMinVoxels", m.fmriAnalysisPlotClusterMinVoxels)
	m.fmriAnalysisPlotVmaxModeIndex = getInt("fmriAnalysisPlotVmaxModeIndex", m.fmriAnalysisPlotVmaxModeIndex)
	m.fmriAnalysisPlotVmaxManual = getFloat("fmriAnalysisPlotVmaxManual", m.fmriAnalysisPlotVmaxManual)
	m.fmriAnalysisPlotIncludeUnthresholded = getBool("fmriAnalysisPlotIncludeUnthresholded", m.fmriAnalysisPlotIncludeUnthresholded)
	m.fmriAnalysisPlotFormatPNG = getBool("fmriAnalysisPlotFormatPNG", m.fmriAnalysisPlotFormatPNG)
	m.fmriAnalysisPlotFormatSVG = getBool("fmriAnalysisPlotFormatSVG", m.fmriAnalysisPlotFormatSVG)
	m.fmriAnalysisPlotTypeSlices = getBool("fmriAnalysisPlotTypeSlices", m.fmriAnalysisPlotTypeSlices)
	m.fmriAnalysisPlotTypeGlass = getBool("fmriAnalysisPlotTypeGlass", m.fmriAnalysisPlotTypeGlass)
	m.fmriAnalysisPlotTypeHist = getBool("fmriAnalysisPlotTypeHist", m.fmriAnalysisPlotTypeHist)
	m.fmriAnalysisPlotTypeClusters = getBool("fmriAnalysisPlotTypeClusters", m.fmriAnalysisPlotTypeClusters)
	m.fmriAnalysisPlotEffectSize = getBool("fmriAnalysisPlotEffectSize", m.fmriAnalysisPlotEffectSize)
	m.fmriAnalysisPlotStandardError = getBool("fmriAnalysisPlotStandardError", m.fmriAnalysisPlotStandardError)
	m.fmriAnalysisPlotMotionQC = getBool("fmriAnalysisPlotMotionQC", m.fmriAnalysisPlotMotionQC)
	m.fmriAnalysisPlotCarpetQC = getBool("fmriAnalysisPlotCarpetQC", m.fmriAnalysisPlotCarpetQC)
	m.fmriAnalysisPlotTSNRQC = getBool("fmriAnalysisPlotTSNRQC", m.fmriAnalysisPlotTSNRQC)
	m.fmriAnalysisPlotDesignQC = getBool("fmriAnalysisPlotDesignQC", m.fmriAnalysisPlotDesignQC)
	m.fmriAnalysisPlotEmbedImages = getBool("fmriAnalysisPlotEmbedImages", m.fmriAnalysisPlotEmbedImages)
	m.fmriAnalysisPlotSignatures = getBool("fmriAnalysisPlotSignatures", m.fmriAnalysisPlotSignatures)
	m.fmriAnalysisSignatureDir = getString("fmriAnalysisSignatureDir", m.fmriAnalysisSignatureDir)
	m.fmriTrialSigGroupExpanded = getBool("fmriTrialSigGroupExpanded", m.fmriTrialSigGroupExpanded)
	m.fmriTrialSigMethodIndex = getInt("fmriTrialSigMethodIndex", m.fmriTrialSigMethodIndex)
	m.fmriTrialSigIncludeOtherEvents = getBool("fmriTrialSigIncludeOtherEvents", m.fmriTrialSigIncludeOtherEvents)
	m.fmriTrialSigMaxTrialsPerRun = getInt("fmriTrialSigMaxTrialsPerRun", m.fmriTrialSigMaxTrialsPerRun)
	m.fmriTrialSigFixedEffectsWeighting = getInt("fmriTrialSigFixedEffectsWeighting", m.fmriTrialSigFixedEffectsWeighting)
	m.fmriTrialSigWriteTrialBetas = getBool("fmriTrialSigWriteTrialBetas", m.fmriTrialSigWriteTrialBetas)
	m.fmriTrialSigWriteTrialVariances = getBool("fmriTrialSigWriteTrialVariances", m.fmriTrialSigWriteTrialVariances)
	m.fmriTrialSigWriteConditionBetas = getBool("fmriTrialSigWriteConditionBetas", m.fmriTrialSigWriteConditionBetas)
	m.fmriTrialSigSignatureNPS = getBool("fmriTrialSigSignatureNPS", m.fmriTrialSigSignatureNPS)
	m.fmriTrialSigSignatureSIIPS1 = getBool("fmriTrialSigSignatureSIIPS1", m.fmriTrialSigSignatureSIIPS1)
	m.fmriTrialSigLssOtherRegressorsIndex = getInt("fmriTrialSigLssOtherRegressorsIndex", m.fmriTrialSigLssOtherRegressorsIndex)
	m.fmriTrialSigGroupColumn = getString("fmriTrialSigGroupColumn", m.fmriTrialSigGroupColumn)
	m.fmriTrialSigGroupValuesSpec = getString("fmriTrialSigGroupValuesSpec", m.fmriTrialSigGroupValuesSpec)
	m.fmriTrialSigGroupScopeIndex = getInt("fmriTrialSigGroupScopeIndex", m.fmriTrialSigGroupScopeIndex)
	m.fmriTrialSigScopeStimPhases = getString("fmriTrialSigScopeStimPhases", m.fmriTrialSigScopeStimPhases)

	// ML pipeline
	m.mlNPerm = getInt("mlNPerm", m.mlNPerm)
	m.innerSplits = getInt("innerSplits", m.innerSplits)
	m.outerJobs = getInt("outerJobs", m.outerJobs)
	m.mlScope = MLCVScope(getInt("mlScope", int(m.mlScope)))
	m.mlTarget = getString("mlTarget", m.mlTarget)
	m.mlFmriSigGroupExpanded = getBool("mlFmriSigGroupExpanded", m.mlFmriSigGroupExpanded)
	m.mlFmriSigMethodIndex = getInt("mlFmriSigMethodIndex", m.mlFmriSigMethodIndex)
	m.mlFmriSigContrastName = getString("mlFmriSigContrastName", m.mlFmriSigContrastName)
	m.mlFmriSigSignatureIndex = getInt("mlFmriSigSignatureIndex", m.mlFmriSigSignatureIndex)
	m.mlFmriSigMetricIndex = getInt("mlFmriSigMetricIndex", m.mlFmriSigMetricIndex)
	m.mlFmriSigNormalizationIndex = getInt("mlFmriSigNormalizationIndex", m.mlFmriSigNormalizationIndex)
	m.mlFmriSigRoundDecimals = getInt("mlFmriSigRoundDecimals", m.mlFmriSigRoundDecimals)
	m.mlBinaryThresholdEnabled = getBool("mlBinaryThresholdEnabled", m.mlBinaryThresholdEnabled)
	m.mlBinaryThreshold = getFloat("mlBinaryThreshold", m.mlBinaryThreshold)
	m.mlFeatureFamiliesSpec = getString("mlFeatureFamiliesSpec", m.mlFeatureFamiliesSpec)
	m.mlFeatureBandsSpec = getString("mlFeatureBandsSpec", m.mlFeatureBandsSpec)
	m.mlFeatureSegmentsSpec = getString("mlFeatureSegmentsSpec", m.mlFeatureSegmentsSpec)
	m.mlFeatureScopesSpec = getString("mlFeatureScopesSpec", m.mlFeatureScopesSpec)
	m.mlFeatureStatsSpec = getString("mlFeatureStatsSpec", m.mlFeatureStatsSpec)
	m.mlFeatureHarmonization = MLFeatureHarmonization(getInt("mlFeatureHarmonization", int(m.mlFeatureHarmonization)))
	m.mlCovariatesSpec = getString("mlCovariatesSpec", m.mlCovariatesSpec)
	m.mlBaselinePredictorsSpec = getString("mlBaselinePredictorsSpec", m.mlBaselinePredictorsSpec)
	m.mlRegressionModel = MLRegressionModel(getInt("mlRegressionModel", int(m.mlRegressionModel)))
	m.mlClassificationModel = MLClassificationModel(getInt("mlClassificationModel", int(m.mlClassificationModel)))
	m.mlRequireTrialMlSafe = getBool("mlRequireTrialMlSafe", m.mlRequireTrialMlSafe)
	m.mlUncertaintyAlpha = getFloat("mlUncertaintyAlpha", m.mlUncertaintyAlpha)
	m.mlPermNRepeats = getInt("mlPermNRepeats", m.mlPermNRepeats)
	m.elasticNetAlphaGrid = getString("elasticNetAlphaGrid", m.elasticNetAlphaGrid)
	m.elasticNetL1RatioGrid = getString("elasticNetL1RatioGrid", m.elasticNetL1RatioGrid)
	m.ridgeAlphaGrid = getString("ridgeAlphaGrid", m.ridgeAlphaGrid)
	m.rfNEstimators = getInt("rfNEstimators", m.rfNEstimators)
	m.rfMaxDepthGrid = getString("rfMaxDepthGrid", m.rfMaxDepthGrid)
	m.varianceThresholdGrid = getString("varianceThresholdGrid", m.varianceThresholdGrid)

	// Plotting
	m.plotDpiIndex = getInt("plotDpiIndex", m.plotDpiIndex)
	m.plotSavefigDpiIndex = getInt("plotSavefigDpiIndex", m.plotSavefigDpiIndex)
	m.plotSharedColorbar = getBool("plotSharedColorbar", m.plotSharedColorbar)
	m.plotBboxInches = getString("plotBboxInches", m.plotBboxInches)
	m.plotPadInches = getFloat("plotPadInches", m.plotPadInches)
	m.plotFontFamily = getString("plotFontFamily", m.plotFontFamily)
	m.plotFontWeight = getString("plotFontWeight", m.plotFontWeight)
	m.plotFontSizeSmall = getInt("plotFontSizeSmall", m.plotFontSizeSmall)
	m.plotFontSizeMedium = getInt("plotFontSizeMedium", m.plotFontSizeMedium)
	m.plotFontSizeLarge = getInt("plotFontSizeLarge", m.plotFontSizeLarge)
	m.plotFontSizeTitle = getInt("plotFontSizeTitle", m.plotFontSizeTitle)
	m.plotFontSizeAnnotation = getInt("plotFontSizeAnnotation", m.plotFontSizeAnnotation)
	m.plotFontSizeLabel = getInt("plotFontSizeLabel", m.plotFontSizeLabel)
	m.plotFontSizeYLabel = getInt("plotFontSizeYLabel", m.plotFontSizeYLabel)
	m.plotFontSizeSuptitle = getInt("plotFontSizeSuptitle", m.plotFontSizeSuptitle)
	m.plotFontSizeFigureTitle = getInt("plotFontSizeFigureTitle", m.plotFontSizeFigureTitle)
	m.plotLayoutTightRectSpec = getString("plotLayoutTightRectSpec", m.plotLayoutTightRectSpec)
	m.plotLayoutTightRectMicrostateSpec = getString("plotLayoutTightRectMicrostateSpec", m.plotLayoutTightRectMicrostateSpec)
	m.plotGridSpecWidthRatiosSpec = getString("plotGridSpecWidthRatiosSpec", m.plotGridSpecWidthRatiosSpec)
	m.plotGridSpecHeightRatiosSpec = getString("plotGridSpecHeightRatiosSpec", m.plotGridSpecHeightRatiosSpec)
	m.plotGridSpecHspace = getFloat("plotGridSpecHspace", m.plotGridSpecHspace)
	m.plotGridSpecWspace = getFloat("plotGridSpecWspace", m.plotGridSpecWspace)
	m.plotGridSpecLeft = getFloat("plotGridSpecLeft", m.plotGridSpecLeft)
	m.plotGridSpecRight = getFloat("plotGridSpecRight", m.plotGridSpecRight)
	m.plotGridSpecTop = getFloat("plotGridSpecTop", m.plotGridSpecTop)
	m.plotGridSpecBottom = getFloat("plotGridSpecBottom", m.plotGridSpecBottom)
	m.plotFigureSizeStandardSpec = getString("plotFigureSizeStandardSpec", m.plotFigureSizeStandardSpec)
	m.plotFigureSizeMediumSpec = getString("plotFigureSizeMediumSpec", m.plotFigureSizeMediumSpec)
	m.plotFigureSizeSmallSpec = getString("plotFigureSizeSmallSpec", m.plotFigureSizeSmallSpec)
	m.plotFigureSizeSquareSpec = getString("plotFigureSizeSquareSpec", m.plotFigureSizeSquareSpec)
	m.plotFigureSizeWideSpec = getString("plotFigureSizeWideSpec", m.plotFigureSizeWideSpec)
	m.plotFigureSizeTFRSpec = getString("plotFigureSizeTFRSpec", m.plotFigureSizeTFRSpec)
	m.plotFigureSizeTopomapSpec = getString("plotFigureSizeTopomapSpec", m.plotFigureSizeTopomapSpec)
	m.plotColorPain = getString("plotColorPain", m.plotColorPain)
	m.plotColorNonpain = getString("plotColorNonpain", m.plotColorNonpain)
	m.plotColorSignificant = getString("plotColorSignificant", m.plotColorSignificant)
	m.plotColorNonsignificant = getString("plotColorNonsignificant", m.plotColorNonsignificant)
	m.plotColorGray = getString("plotColorGray", m.plotColorGray)
	m.plotColorLightGray = getString("plotColorLightGray", m.plotColorLightGray)
	m.plotColorBlack = getString("plotColorBlack", m.plotColorBlack)
	m.plotColorBlue = getString("plotColorBlue", m.plotColorBlue)
	m.plotColorRed = getString("plotColorRed", m.plotColorRed)
	m.plotColorNetworkNode = getString("plotColorNetworkNode", m.plotColorNetworkNode)
	m.plotAlphaGrid = getFloat("plotAlphaGrid", m.plotAlphaGrid)
	m.plotAlphaFill = getFloat("plotAlphaFill", m.plotAlphaFill)
	m.plotAlphaCI = getFloat("plotAlphaCI", m.plotAlphaCI)
	m.plotAlphaCILine = getFloat("plotAlphaCILine", m.plotAlphaCILine)
	m.plotAlphaTextBox = getFloat("plotAlphaTextBox", m.plotAlphaTextBox)
	m.plotAlphaViolinBody = getFloat("plotAlphaViolinBody", m.plotAlphaViolinBody)
	m.plotAlphaRidgeFill = getFloat("plotAlphaRidgeFill", m.plotAlphaRidgeFill)
	m.plotScatterMarkerSizeSmall = getInt("plotScatterMarkerSizeSmall", m.plotScatterMarkerSizeSmall)
	m.plotScatterMarkerSizeLarge = getInt("plotScatterMarkerSizeLarge", m.plotScatterMarkerSizeLarge)
	m.plotScatterMarkerSizeDefault = getInt("plotScatterMarkerSizeDefault", m.plotScatterMarkerSizeDefault)
	m.plotScatterAlpha = getFloat("plotScatterAlpha", m.plotScatterAlpha)
	m.plotScatterEdgeColor = getString("plotScatterEdgeColor", m.plotScatterEdgeColor)
	m.plotScatterEdgeWidth = getFloat("plotScatterEdgeWidth", m.plotScatterEdgeWidth)
	m.plotBarAlpha = getFloat("plotBarAlpha", m.plotBarAlpha)
	m.plotBarWidth = getFloat("plotBarWidth", m.plotBarWidth)
	m.plotBarCapsize = getInt("plotBarCapsize", m.plotBarCapsize)
	m.plotBarCapsizeLarge = getInt("plotBarCapsizeLarge", m.plotBarCapsizeLarge)
	m.plotLineWidthThin = getFloat("plotLineWidthThin", m.plotLineWidthThin)
	m.plotLineWidthStandard = getFloat("plotLineWidthStandard", m.plotLineWidthStandard)
	m.plotLineWidthThick = getFloat("plotLineWidthThick", m.plotLineWidthThick)
	m.plotLineWidthBold = getFloat("plotLineWidthBold", m.plotLineWidthBold)
	m.plotLineAlphaStandard = getFloat("plotLineAlphaStandard", m.plotLineAlphaStandard)
	m.plotLineAlphaDim = getFloat("plotLineAlphaDim", m.plotLineAlphaDim)
	m.plotLineAlphaZeroLine = getFloat("plotLineAlphaZeroLine", m.plotLineAlphaZeroLine)
	m.plotLineAlphaFitLine = getFloat("plotLineAlphaFitLine", m.plotLineAlphaFitLine)
	m.plotLineAlphaDiagonal = getFloat("plotLineAlphaDiagonal", m.plotLineAlphaDiagonal)
	m.plotLineAlphaReference = getFloat("plotLineAlphaReference", m.plotLineAlphaReference)
	m.plotLineRegressionWidth = getFloat("plotLineRegressionWidth", m.plotLineRegressionWidth)
	m.plotLineResidualWidth = getFloat("plotLineResidualWidth", m.plotLineResidualWidth)
	m.plotLineQQWidth = getFloat("plotLineQQWidth", m.plotLineQQWidth)
	m.plotHistBins = getInt("plotHistBins", m.plotHistBins)
	m.plotHistBinsBehavioral = getInt("plotHistBinsBehavioral", m.plotHistBinsBehavioral)
	m.plotHistBinsResidual = getInt("plotHistBinsResidual", m.plotHistBinsResidual)
	m.plotHistBinsTFR = getInt("plotHistBinsTFR", m.plotHistBinsTFR)
	m.plotHistEdgeColor = getString("plotHistEdgeColor", m.plotHistEdgeColor)
	m.plotHistEdgeWidth = getFloat("plotHistEdgeWidth", m.plotHistEdgeWidth)
	m.plotHistAlpha = getFloat("plotHistAlpha", m.plotHistAlpha)
	m.plotHistAlphaResidual = getFloat("plotHistAlphaResidual", m.plotHistAlphaResidual)
	m.plotHistAlphaTFR = getFloat("plotHistAlphaTFR", m.plotHistAlphaTFR)
	m.plotKdePoints = getInt("plotKdePoints", m.plotKdePoints)
	m.plotKdeColor = getString("plotKdeColor", m.plotKdeColor)
	m.plotKdeLinewidth = getFloat("plotKdeLinewidth", m.plotKdeLinewidth)
	m.plotKdeAlpha = getFloat("plotKdeAlpha", m.plotKdeAlpha)
	m.plotErrorbarMarkerSize = getInt("plotErrorbarMarkerSize", m.plotErrorbarMarkerSize)
	m.plotErrorbarCapsize = getInt("plotErrorbarCapsize", m.plotErrorbarCapsize)
	m.plotErrorbarCapsizeLarge = getInt("plotErrorbarCapsizeLarge", m.plotErrorbarCapsizeLarge)
	m.plotTextStatsX = getFloat("plotTextStatsX", m.plotTextStatsX)
	m.plotTextStatsY = getFloat("plotTextStatsY", m.plotTextStatsY)
	m.plotTextPvalueX = getFloat("plotTextPvalueX", m.plotTextPvalueX)
	m.plotTextPvalueY = getFloat("plotTextPvalueY", m.plotTextPvalueY)
	m.plotTextBootstrapX = getFloat("plotTextBootstrapX", m.plotTextBootstrapX)
	m.plotTextBootstrapY = getFloat("plotTextBootstrapY", m.plotTextBootstrapY)
	m.plotTextChannelAnnotationX = getFloat("plotTextChannelAnnotationX", m.plotTextChannelAnnotationX)
	m.plotTextChannelAnnotationY = getFloat("plotTextChannelAnnotationY", m.plotTextChannelAnnotationY)
	m.plotTextTitleY = getFloat("plotTextTitleY", m.plotTextTitleY)
	m.plotTextResidualQcTitleY = getFloat("plotTextResidualQcTitleY", m.plotTextResidualQcTitleY)
	m.plotValidationMinBinsForCalibration = getInt("plotValidationMinBinsForCalibration", m.plotValidationMinBinsForCalibration)
	m.plotValidationMaxBinsForCalibration = getInt("plotValidationMaxBinsForCalibration", m.plotValidationMaxBinsForCalibration)
	m.plotValidationSamplesPerBin = getInt("plotValidationSamplesPerBin", m.plotValidationSamplesPerBin)
	m.plotValidationMinRoisForFDR = getInt("plotValidationMinRoisForFDR", m.plotValidationMinRoisForFDR)
	m.plotValidationMinPvaluesForFDR = getInt("plotValidationMinPvaluesForFDR", m.plotValidationMinPvaluesForFDR)
	m.plotTfrDefaultBaselineWindowSpec = getString("plotTfrDefaultBaselineWindowSpec", m.plotTfrDefaultBaselineWindowSpec)
	m.plotTopomapContours = getInt("plotTopomapContours", m.plotTopomapContours)
	m.plotTopomapColormap = getString("plotTopomapColormap", m.plotTopomapColormap)
	m.plotTopomapColorbarFraction = getFloat("plotTopomapColorbarFraction", m.plotTopomapColorbarFraction)
	m.plotTopomapColorbarPad = getFloat("plotTopomapColorbarPad", m.plotTopomapColorbarPad)
	if v, ok := cfg["plotTopomapDiffAnnotation"].(bool); ok {
		m.plotTopomapDiffAnnotation = &v
	}
	if v, ok := cfg["plotTopomapAnnotateDesc"].(bool); ok {
		m.plotTopomapAnnotateDesc = &v
	}
	m.plotTopomapSigMaskMarker = getString("plotTopomapSigMaskMarker", m.plotTopomapSigMaskMarker)
	m.plotTopomapSigMaskMarkerFaceColor = getString("plotTopomapSigMaskMarkerFaceColor", m.plotTopomapSigMaskMarkerFaceColor)
	m.plotTopomapSigMaskMarkerEdgeColor = getString("plotTopomapSigMaskMarkerEdgeColor", m.plotTopomapSigMaskMarkerEdgeColor)
	m.plotTopomapSigMaskLinewidth = getFloat("plotTopomapSigMaskLinewidth", m.plotTopomapSigMaskLinewidth)
	m.plotTopomapSigMaskMarkerSize = getFloat("plotTopomapSigMaskMarkerSize", m.plotTopomapSigMaskMarkerSize)
	m.plotTFRLogBase = getFloat("plotTFRLogBase", m.plotTFRLogBase)
	m.plotTFRPercentageMultiplier = getFloat("plotTFRPercentageMultiplier", m.plotTFRPercentageMultiplier)
	m.plotTFRTopomapWindowSizeMs = getFloat("plotTFRTopomapWindowSizeMs", m.plotTFRTopomapWindowSizeMs)
	m.plotTFRTopomapWindowCount = getInt("plotTFRTopomapWindowCount", m.plotTFRTopomapWindowCount)
	m.plotTFRTopomapLabelXPosition = getFloat("plotTFRTopomapLabelXPosition", m.plotTFRTopomapLabelXPosition)
	m.plotTFRTopomapLabelYPositionBottom = getFloat("plotTFRTopomapLabelYPositionBottom", m.plotTFRTopomapLabelYPositionBottom)
	m.plotTFRTopomapLabelYPosition = getFloat("plotTFRTopomapLabelYPosition", m.plotTFRTopomapLabelYPosition)
	m.plotTFRTopomapTitleY = getFloat("plotTFRTopomapTitleY", m.plotTFRTopomapTitleY)
	m.plotTFRTopomapTitlePad = getInt("plotTFRTopomapTitlePad", m.plotTFRTopomapTitlePad)
	m.plotTFRTopomapSubplotsRight = getFloat("plotTFRTopomapSubplotsRight", m.plotTFRTopomapSubplotsRight)
	m.plotTFRTopomapTemporalHspace = getFloat("plotTFRTopomapTemporalHspace", m.plotTFRTopomapTemporalHspace)
	m.plotTFRTopomapTemporalWspace = getFloat("plotTFRTopomapTemporalWspace", m.plotTFRTopomapTemporalWspace)
	m.plotRoiWidthPerBand = getFloat("plotRoiWidthPerBand", m.plotRoiWidthPerBand)
	m.plotRoiWidthPerMetric = getFloat("plotRoiWidthPerMetric", m.plotRoiWidthPerMetric)
	m.plotRoiHeightPerRoi = getFloat("plotRoiHeightPerRoi", m.plotRoiHeightPerRoi)
	m.plotPowerWidthPerBand = getFloat("plotPowerWidthPerBand", m.plotPowerWidthPerBand)
	m.plotPowerHeightPerSegment = getFloat("plotPowerHeightPerSegment", m.plotPowerHeightPerSegment)
	m.plotItpcWidthPerBin = getFloat("plotItpcWidthPerBin", m.plotItpcWidthPerBin)
	m.plotItpcHeightPerBand = getFloat("plotItpcHeightPerBand", m.plotItpcHeightPerBand)
	m.plotItpcWidthPerBandBox = getFloat("plotItpcWidthPerBandBox", m.plotItpcWidthPerBandBox)
	m.plotItpcHeightBox = getFloat("plotItpcHeightBox", m.plotItpcHeightBox)
	m.plotPacCmap = getString("plotPacCmap", m.plotPacCmap)
	m.plotPacWidthPerRoi = getFloat("plotPacWidthPerRoi", m.plotPacWidthPerRoi)
	m.plotPacHeightBox = getFloat("plotPacHeightBox", m.plotPacHeightBox)
	m.plotAperiodicWidthPerColumn = getFloat("plotAperiodicWidthPerColumn", m.plotAperiodicWidthPerColumn)
	m.plotAperiodicHeightPerRow = getFloat("plotAperiodicHeightPerRow", m.plotAperiodicHeightPerRow)
	m.plotAperiodicNPerm = getInt("plotAperiodicNPerm", m.plotAperiodicNPerm)
	m.plotComplexityWidthPerMeasure = getFloat("plotComplexityWidthPerMeasure", m.plotComplexityWidthPerMeasure)
	m.plotComplexityHeightPerSegment = getFloat("plotComplexityHeightPerSegment", m.plotComplexityHeightPerSegment)
	m.plotConnectivityWidthPerCircle = getFloat("plotConnectivityWidthPerCircle", m.plotConnectivityWidthPerCircle)
	m.plotConnectivityWidthPerBand = getFloat("plotConnectivityWidthPerBand", m.plotConnectivityWidthPerBand)
	m.plotConnectivityHeightPerMeasure = getFloat("plotConnectivityHeightPerMeasure", m.plotConnectivityHeightPerMeasure)
	m.plotConnectivityCircleTopFraction = getFloat("plotConnectivityCircleTopFraction", m.plotConnectivityCircleTopFraction)
	m.plotConnectivityCircleMinLines = getInt("plotConnectivityCircleMinLines", m.plotConnectivityCircleMinLines)
	m.plotConnectivityNetworkTopFraction = getFloat("plotConnectivityNetworkTopFraction", m.plotConnectivityNetworkTopFraction)
	m.plotPacPairsSpec = getString("plotPacPairsSpec", m.plotPacPairsSpec)
	m.plotSpectralMetricsSpec = getString("plotSpectralMetricsSpec", m.plotSpectralMetricsSpec)
	m.plotBurstsMetricsSpec = getString("plotBurstsMetricsSpec", m.plotBurstsMetricsSpec)
	m.plotTemporalTimeBinsSpec = getString("plotTemporalTimeBinsSpec", m.plotTemporalTimeBinsSpec)
	m.plotTemporalTimeLabelsSpec = getString("plotTemporalTimeLabelsSpec", m.plotTemporalTimeLabelsSpec)
	m.plotAsymmetryStatSpec = getString("plotAsymmetryStatSpec", m.plotAsymmetryStatSpec)
	if v, ok := cfg["plotCompareWindows"].(bool); ok {
		m.plotCompareWindows = &v
	}
	m.plotComparisonWindowsSpec = getString("plotComparisonWindowsSpec", m.plotComparisonWindowsSpec)
	if v, ok := cfg["plotCompareColumns"].(bool); ok {
		m.plotCompareColumns = &v
	}
	m.plotComparisonSegment = getString("plotComparisonSegment", m.plotComparisonSegment)
	m.plotComparisonColumn = getString("plotComparisonColumn", m.plotComparisonColumn)
	m.plotComparisonValuesSpec = getString("plotComparisonValuesSpec", m.plotComparisonValuesSpec)
	m.plotComparisonLabelsSpec = getString("plotComparisonLabelsSpec", m.plotComparisonLabelsSpec)
	m.plotComparisonROIsSpec = getString("plotComparisonROIsSpec", m.plotComparisonROIsSpec)
	if v, ok := cfg["plotOverwrite"].(bool); ok {
		m.plotOverwrite = &v
	}

	// System
	m.systemNJobs = getInt("systemNJobs", m.systemNJobs)
	m.systemStrictMode = getBool("systemStrictMode", m.systemStrictMode)
	m.loggingLevel = getInt("loggingLevel", m.loggingLevel)
}

// mapToIntList converts a map[int]bool to a slice of ints (only selected keys).
func mapToIntList(m map[int]bool) []int {
	list := make([]int, 0)
	for k, v := range m {
		if v {
			list = append(list, k)
		}
	}
	return list
}

// listToMap converts a slice of ints to a map[int]bool.
func listToMap(v interface{}) map[int]bool {
	m := make(map[int]bool)
	if list, ok := v.([]interface{}); ok {
		for _, item := range list {
			if f, ok := item.(float64); ok {
				m[int(f)] = true
			} else if i, ok := item.(int); ok {
				m[i] = true
			}
		}
	}
	return m
}

// stringsMapToBoolMap converts a map[string]bool to a slice of strings (only selected keys).
func stringsMapToBoolMap(m map[string]bool) []string {
	list := make([]string, 0)
	for k, v := range m {
		if v {
			list = append(list, k)
		}
	}
	return list
}

// boolMapToStringsMap converts a slice of strings to a map[string]bool.
func boolMapToStringsMap(v interface{}) map[string]bool {
	m := make(map[string]bool)
	if list, ok := v.([]interface{}); ok {
		for _, item := range list {
			if s, ok := item.(string); ok {
				m[s] = true
			}
		}
	}
	return m
}
