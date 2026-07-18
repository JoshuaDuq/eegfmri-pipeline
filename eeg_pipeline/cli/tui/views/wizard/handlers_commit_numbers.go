package wizard

import (
	"strconv"

	"github.com/eeg-pipeline/tui/types"
)

// Numeric input commit handlers by pipeline.

// commitNumberInput parses the number buffer and applies it to the current field
func (m *Model) commitNumberInput() {
	if m.numberBuffer == "" {
		return
	}

	val, err := strconv.ParseFloat(m.numberBuffer, 64)
	if err != nil {
		return // Invalid number, ignore
	}

	switch m.Pipeline {
	case types.PipelineFeatures:
		m.commitFeaturesNumber(val)
	case types.PipelineBehavior:
		m.commitBehaviorNumber(val)
	case types.PipelineML:
		m.commitMLNumber(val)
	case types.PipelinePreprocessing:
		m.commitPreprocessingNumber(val)
	case types.PipelineFmri:
		m.commitFmriNumber(val)
	case types.PipelineFmriAnalysis:
		m.commitFmriAnalysisNumber(val)
	}
	m.useDefaultAdvanced = false
}

func (m *Model) commitFeaturesNumber(val float64) {
	// Re-build the same options slice as toggleFeaturesAdvancedOption to find current opt
	options := m.getFeaturesOptions()

	if m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optAperiodicFmin:
		if val > 0 && val < m.aperiodicFmax {
			m.aperiodicFmin = val
		}
	case optAperiodicFmax:
		if val > 0 && val > m.aperiodicFmin {
			m.aperiodicFmax = val
		}
	case optAperiodicPeakZ:
		m.aperiodicPeakZ = val
	case optAperiodicMinR2:
		m.aperiodicMinR2 = val
	case optAperiodicMinPoints:
		m.aperiodicMinPoints = int(val)
	case optAperiodicPsdBandwidth:
		if val >= 0 {
			m.aperiodicPsdBandwidth = val
		}
	case optAperiodicMaxRms:
		if val >= 0 {
			m.aperiodicMaxRms = val
		}
	case optAperiodicMinSegmentSec:
		if val > 0 {
			m.aperiodicMinSegmentSec = val
		}
	case optAperiodicLineNoiseFreq:
		if val > 0 {
			m.aperiodicLineNoiseFreq = val
		}
	case optAperiodicLineNoiseWidthHz:
		if val > 0 {
			m.aperiodicLineNoiseWidthHz = val
		}
	case optAperiodicLineNoiseHarmonics:
		if val >= 0 {
			m.aperiodicLineNoiseHarmonics = int(val)
		}
	case optPACMinEpochs:
		m.pacMinEpochs = int(val)
	case optPACNSurrogates:
		m.pacNSurrogates = int(val)
	case optPACMaxHarmonic:
		m.pacMaxHarmonic = int(val)
	case optPACHarmonicToleranceHz:
		m.pacHarmonicToleranceHz = val
	case optPACRandomSeed:
		m.pacRandomSeed = int(val)
	case optPACWaveformOffsetMs:
		m.pacWaveformOffsetMs = val
	case optPEDelay:
		m.complexityPEDelay = int(val)
	case optComplexitySampleEntropyOrder:
		if val >= 1 {
			m.complexitySampEnOrder = int(val)
		}
	case optComplexitySampleEntropyR:
		if val > 0 {
			m.complexitySampEnR = val
		}
	case optComplexityMSEScaleMin:
		if val >= 1 {
			m.complexityMSEScaleMin = int(val)
			if m.complexityMSEScaleMax < m.complexityMSEScaleMin {
				m.complexityMSEScaleMax = m.complexityMSEScaleMin
			}
		}
	case optComplexityMSEScaleMax:
		if val >= 1 {
			m.complexityMSEScaleMax = int(val)
			if m.complexityMSEScaleMax < m.complexityMSEScaleMin {
				m.complexityMSEScaleMin = m.complexityMSEScaleMax
			}
		}
	case optComplexityMinSegmentSec:
		if val > 0 {
			m.complexityMinSegmentSec = val
		}
	case optComplexityMinSamples:
		if val >= 0 {
			m.complexityMinSamples = int(val)
		}
	case optBurstThresholdPercentile:
		if val >= 0 && val <= 100 {
			m.burstThresholdPercentile = val
		}
	case optBurstMinTrialsPerCondition:
		if val >= 1 {
			m.burstMinTrialsPerCondition = int(val)
		}
	case optBurstMinSegmentSec:
		if val >= 0 {
			m.burstMinSegmentSec = val
		}
	case optBurstMinDuration:
		m.burstMinDuration = int(val)
	case optBurstMinCycles:
		if val > 0 {
			m.burstMinCycles = val
		}
	case optERPSmoothMs:
		if val >= 0 {
			m.erpSmoothMs = val
		}
	case optERPPeakProminenceUv:
		if val >= 0 {
			m.erpPeakProminenceUv = val
		}
	case optERPLowpassHz:
		if val > 0 {
			m.erpLowpassHz = val
		}
	case optPowerMinTrialsPerCondition:
		if val >= 0 {
			m.powerMinTrialsPerCondition = int(val)
		}
	case optPowerLineNoiseFreq:
		if val > 0 {
			m.powerLineNoiseFreq = val
		}
	case optPowerLineNoiseWidthHz:
		if val > 0 {
			m.powerLineNoiseWidthHz = val
		}
	case optPowerLineNoiseHarmonics:
		if val >= 1 {
			m.powerLineNoiseHarmonics = int(val)
		}
	case optMinEpochs:
		m.minEpochsForFeatures = int(val)
	case optFeatNJobsBands:
		m.featNJobsBands = int(val)
	case optFeatNJobsConnectivity:
		m.featNJobsConnectivity = int(val)
	case optFeatNJobsAperiodic:
		m.featNJobsAperiodic = int(val)
	case optFeatNJobsComplexity:
		m.featNJobsComplexity = int(val)
	case optConnGraphProp:
		m.connGraphProp = val
	case optConnWindowLen:
		m.connWindowLen = val
	case optConnWindowStep:
		m.connWindowStep = val
	case optConnNFreqsPerBand:
		if val >= 1 {
			m.connNFreqsPerBand = int(val)
		}
	case optConnNCycles:
		if val >= 0 {
			m.connNCycles = val
		}
	case optConnDecim:
		if val >= 1 {
			m.connDecim = int(val)
		}
	case optConnMinSegmentSamples:
		if val >= 1 {
			m.connMinSegSamples = int(val)
		}
	case optConnSmallWorldNRand:
		if val >= 0 {
			m.connSmallWorldNRand = int(val)
		}
	case optConnMinEpochsPerGroup:
		if val >= 1 {
			m.connMinEpochsPerGroup = int(val)
		}
	case optConnMinCyclesPerBand:
		if val > 0 {
			m.connMinCyclesPerBand = val
		}
	case optConnMinSegmentSec:
		if val >= 0 {
			m.connMinSegmentSec = val
		}
	case optConnDynamicAutocorrLag:
		if val >= 1 {
			m.connDynamicAutocorrLag = int(val)
		}
	case optConnDynamicMinWindows:
		if val >= 2 {
			m.connDynamicMinWindows = int(val)
		}
	case optConnDynamicStateNStates:
		if val >= 2 {
			m.connDynamicStateNStates = int(val)
		}
	case optConnDynamicStateMinWindows:
		if val >= 3 {
			m.connDynamicStateMinWindows = int(val)
		}
	case optConnDynamicStateRandomSeed:
		if val >= 0 {
			m.connDynamicStateRandomSeed = int(val)
		} else if int(val) == -1 {
			m.connDynamicStateRandomSeed = -1
		}

	// Source localization numeric options
	case optSourceLocReg:
		if val >= 0 {
			m.sourceLocReg = val
		}
	case optSourceLocSnr:
		if val > 0 {
			m.sourceLocSnr = val
		}
	case optSourceLocLoose:
		if val >= 0 && val <= 1 {
			m.sourceLocLoose = val
		}
	case optSourceLocDepth:
		if val >= 0 && val <= 1 {
			m.sourceLocDepth = val
		}
	case optSourceLocMindistMm:
		if val >= 0 {
			m.sourceLocMindistMm = val
		}
	case optSourceLocContrastMinTrials:
		if val >= 1 {
			m.sourceLocContrastMinTrials = int(val)
		}

	case optSourceLocFmriThreshold:
		if val > 0 {
			m.sourceLocFmriThreshold = val
		}
	case optSourceLocFmriFdrQ:
		if val > 0 && val < 1 {
			m.sourceLocFmriFdrQ = val
		}
	case optSourceLocFmriMinClusterMM3:
		if val >= 0 {
			m.sourceLocFmriMinClusterMM3 = val
		}
	case optSourceLocFmriMinClusterVox:
		if val >= 0 {
			m.sourceLocFmriMinClusterVox = int(val)
		}
	case optSourceLocFmriMaxClusters:
		if val >= 1 {
			m.sourceLocFmriMaxClusters = int(val)
		}
	case optSourceLocFmriMaxVoxPerClus:
		if val >= 0 {
			m.sourceLocFmriMaxVoxPerClus = int(val)
		}
	case optSourceLocFmriMaxTotalVox:
		if val >= 0 {
			m.sourceLocFmriMaxTotalVox = int(val)
		}
	case optSourceLocFmriRandomSeed:
		if val >= 0 {
			m.sourceLocFmriRandomSeed = int(val)
		}
	case optSourceLocFmriHighPassHz:
		if val >= 0 {
			m.sourceLocFmriHighPassHz = val
		}
	case optSourceLocFmriLowPassHz:
		if val >= 0 {
			m.sourceLocFmriLowPassHz = val
		}
	// ITPC options
	case optItpcMinTrialsPerCondition:
		if val >= 1 {
			m.itpcMinTrialsPerCondition = int(val)
		}
	// Spatial transform options
	case optSpatialTransformLambda2:
		if val > 0 {
			m.spatialTransformLambda2 = val
		}
	case optSpatialTransformStiffness:
		if val >= 0 {
			m.spatialTransformStiffness = val
		}
	// TFR options
	case optTfrFreqMin:
		if val >= 0 {
			m.tfrFreqMin = val
		}
	case optTfrFreqMax:
		if val > 0 {
			m.tfrFreqMax = val
		}
	case optTfrNFreqs:
		if val >= 1 {
			m.tfrNFreqs = int(val)
		}
	case optTfrMinCycles:
		if val >= 1 {
			m.tfrMinCycles = val
		}
	case optTfrMaxCycles:
		if val >= 1 {
			m.tfrMaxCycles = val
		}
	case optTfrNCyclesFactor:
		if val >= 0.5 {
			m.tfrNCyclesFactor = val
		}
	case optTfrDecimPower:
		if val >= 1 {
			m.tfrDecimPower = int(val)
		}
	case optTfrDecimPhase:
		if val >= 1 {
			m.tfrDecimPhase = int(val)
		}
	case optTfrWorkers:
		m.tfrWorkers = int(val)
	case optItpcNJobs:
		m.itpcNJobs = int(val)
	// Asymmetry options
	case optAsymmetryMinSegmentSec:
		if val >= 0 {
			m.asymmetryMinSegmentSec = val
		}
	case optAsymmetryMinCyclesAtFmin:
		if val >= 0 {
			m.asymmetryMinCyclesAtFmin = val
		}
	// Ratios options
	case optRatiosMinSegmentSec:
		if val >= 0 {
			m.ratiosMinSegmentSec = val
		}
	case optRatiosMinCyclesAtFmin:
		if val >= 0 {
			m.ratiosMinCyclesAtFmin = val
		}
	// Spectral options
	case optSpectralFmin:
		if val >= 0 {
			m.spectralFmin = val
		}
	case optSpectralFmax:
		if val > 0 {
			m.spectralFmax = val
		}
	case optSpectralLineNoiseFreq:
		if val > 0 {
			m.spectralLineNoiseFreq = val
		}
	case optSpectralLineNoiseWidthHz:
		if val >= 0 {
			m.spectralLineNoiseWidthHz = val
		}
	case optSpectralLineNoiseHarmonics:
		if val >= 0 {
			m.spectralLineNoiseHarmonics = int(val)
		}
	case optSpectralMinSegmentSec:
		if val >= 0 {
			m.spectralMinSegmentSec = val
		}
	case optSpectralMinCyclesAtFmin:
		if val >= 0 {
			m.spectralMinCyclesAtFmin = val
		}
	// Band envelope / IAF
	case optBandEnvelopePadSec:
		if val >= 0 {
			m.bandEnvelopePadSec = val
		}
	case optBandEnvelopePadCycles:
		if val >= 0 {
			m.bandEnvelopePadCycles = val
		}
	case optIAFAlphaWidthHz:
		if val > 0 {
			m.iafAlphaWidthHz = val
		}
	case optIAFSearchRangeMin:
		m.iafSearchRangeMin = val
	case optIAFSearchRangeMax:
		m.iafSearchRangeMax = val
	case optIAFMinProminence:
		if val >= 0 {
			m.iafMinProminence = val
		}
	case optIAFMinCyclesAtFmin:
		if val >= 0 {
			m.iafMinCyclesAtFmin = val
		}
	case optIAFMinBaselineSec:
		if val >= 0 {
			m.iafMinBaselineSec = val
		}
	// Quality options
	case optQualityFmin:
		if val >= 0 {
			m.qualityFmin = val
		}
	case optQualityFmax:
		if val > 0 {
			m.qualityFmax = val
		}
	case optQualityNFft:
		if val >= 1 {
			m.qualityNfft = int(val)
		}
	case optQualityLineNoiseFreq:
		if val > 0 {
			m.qualityLineNoiseFreq = val
		}
	case optQualityLineNoiseWidthHz:
		if val >= 0 {
			m.qualityLineNoiseWidthHz = val
		}
	case optQualityLineNoiseHarmonics:
		if val >= 0 {
			m.qualityLineNoiseHarmonics = int(val)
		}
	case optQualitySnrSignalBandMin:
		if val >= 0 {
			m.qualitySnrSignalBandMin = val
		}
	case optQualitySnrSignalBandMax:
		if val > 0 {
			m.qualitySnrSignalBandMax = val
		}
	case optQualitySnrNoiseBandMin:
		if val >= 0 {
			m.qualitySnrNoiseBandMin = val
		}
	case optQualitySnrNoiseBandMax:
		if val > 0 {
			m.qualitySnrNoiseBandMax = val
		}
	case optQualityMuscleBandMin:
		if val >= 0 {
			m.qualityMuscleBandMin = val
		}
	case optQualityMuscleBandMax:
		if val > 0 {
			m.qualityMuscleBandMax = val
		}
	// Microstates options
	case optMicrostatesNStates:
		if val >= 2 {
			m.microstatesNStates = int(val)
		}
	case optMicrostatesMinPeakDistanceMs:
		if val >= 0 {
			m.microstatesMinPeakDistanceMs = val
		}
	case optMicrostatesMaxGfpPeaksPerEpoch:
		if val >= 1 {
			m.microstatesMaxGfpPeaksPerEpoch = int(val)
		}
	case optMicrostatesMinDurationMs:
		if val >= 0 {
			m.microstatesMinDurationMs = val
		}
	case optMicrostatesGfpPeakProminence:
		if val >= 0 {
			m.microstatesGfpPeakProminence = val
		}
	case optMicrostatesRandomState:
		if val >= 0 {
			m.microstatesRandomState = int(val)
		}
	// ERDS options
	case optERDSMinBaselinePower:
		if val > 0 {
			m.erdsMinBaselinePower = val
		}
	case optERDSMinActivePower:
		if val > 0 {
			m.erdsMinActivePower = val
		}
	case optERDSOnsetThresholdSigma:
		if val >= 0 {
			m.erdsOnsetThresholdSigma = val
		}
	case optERDSOnsetMinDurationMs:
		if val >= 0 {
			m.erdsOnsetMinDurationMs = val
		}
	case optERDSReboundMinLatencyMs:
		if val >= 0 {
			m.erdsReboundMinLatencyMs = val
		}
	// ITPC/PAC segment validity
	case optItpcMinSegmentSec:
		if val > 0 {
			m.itpcMinSegmentSec = val
		}
	case optItpcMinCyclesAtFmin:
		if val > 0 {
			m.itpcMinCyclesAtFmin = val
		}
	case optPACMinSegmentSec:
		if val > 0 {
			m.pacMinSegmentSec = val
		}
	case optPACMinCyclesAtFmin:
		if val > 0 {
			m.pacMinCyclesAtFmin = val
		}
	// Aperiodic missing
	case optAperiodicMaxFreqResolutionHz:
		if val > 0 {
			m.aperiodicMaxFreqResolutionHz = val
		}
	// Directed connectivity missing
	case optDirectedConnMinSamplesPerMvarParam:
		if val >= 1 {
			m.directedConnMinSamplesPerMvarParam = int(val)
		}
	// ERDS condition markers
	case optERDSOnsetMinThresholdPercent:
		if val >= 0 {
			m.erdsOnsetMinThresholdPercent = val
		}
	case optERDSReboundThresholdSigma:
		if val >= 0 {
			m.erdsReboundThresholdSigma = val
		}
	case optERDSReboundMinThresholdPercent:
		if val >= 0 {
			m.erdsReboundMinThresholdPercent = val
		}
	}
}

func (m *Model) commitBehaviorNumber(val float64) {
	options := m.getBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optBootstrap:
		if val >= 0 {
			m.bootstrapSamples = int(val)
		}
	case optNPerm:
		if val >= 0 {
			m.nPermutations = int(val)
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = int(val)
		}
	case optBehaviorNJobs:
		m.behaviorNJobs = int(val)
	case optFDRAlpha:
		if val > 0 && val <= 1 {
			m.fdrAlpha = val
		}
	case optRunAdjustmentMaxDummies:
		if val >= 1 {
			m.runAdjustmentMaxDummies = int(val)
		}
	case optGroupLevelMaxRunDummies:
		if val >= 1 {
			m.groupLevelMaxRunDummies = int(val)
		}
	case optCorrelationsMinRuns:
		if val >= 1 {
			m.correlationsMinRuns = int(val)
		}
	case optCorrelationsPermutations:
		if val >= 0 {
			m.correlationsPermutations = int(val)
		}
	case optBehaviorMinSamples:
		if val >= 0 {
			m.behaviorMinSamples = int(val)
		}
	case optTrialOrderMaxMissingFraction:
		if val >= 0 && val <= 1 {
			m.trialOrderMaxMissingFraction = val
		}
	// Predictor residual + diagnostics
	case optPredictorResidualPolyDegree:
		if val >= 1 {
			m.predictorResidualPolyDegree = int(val)
		}
	case optPredictorResidualCrossfitNSplits:
		if val >= 2 {
			m.predictorResidualCrossfitNSplits = int(val)
		}
	case optPredictorResidualCrossfitSplineKnots:
		if val >= 3 {
			m.predictorResidualCrossfitSplineKnots = int(val)
		}
	case optPredictorResidualMinSamples:
		if val >= 1 {
			m.predictorResidualMinSamples = int(val)
		}

	// Regression
	case optRegressionPredictorSplineKnots:
		if val >= 4 {
			m.regressionPredictorSplineKnots = int(val)
		}
	case optRegressionPredictorSplineQlow:
		if val > 0 && val < 1 {
			m.regressionPredictorSplineQlow = val
		}
	case optRegressionPredictorSplineQhigh:
		if val > 0 && val < 1 {
			m.regressionPredictorSplineQhigh = val
		}
	case optRegressionPredictorSplineMinN:
		if val >= 1 {
			m.regressionPredictorSplineMinN = int(val)
		}
	case optRegressionPermutations:
		if val >= 0 {
			m.regressionPermutations = int(val)
		}
	case optRegressionMinSamples:
		if val >= 1 {
			m.regressionMinSamples = int(val)
		}
	case optRegressionMaxFeatures:
		if val >= 0 {
			m.regressionMaxFeatures = int(val)
		}
	// Report / temporal
	case optTemporalResolutionMs:
		if val >= 1 {
			m.temporalResolutionMs = int(val)
		}
	case optTemporalTimeMinMs:
		m.temporalTimeMinMs = int(val)
	case optTemporalTimeMaxMs:
		m.temporalTimeMaxMs = int(val)
	case optTemporalSmoothMs:
		if val >= 0 {
			m.temporalSmoothMs = int(val)
		}
	case optTemporalTopomapWindowMs:
		if val >= 1 {
			m.temporalTopomapWindowMs = int(val)
		}
	// ITPC temporal options
	case optTemporalITPCBaselineMin:
		m.temporalITPCBaselineMin = val
	case optTemporalITPCBaselineMax:
		m.temporalITPCBaselineMax = val
	// ERDS temporal options
	case optTemporalERDSBaselineMin:
		m.temporalERDSBaselineMin = val
	case optTemporalERDSBaselineMax:
		m.temporalERDSBaselineMax = val
	// Temporal cluster correction options
	case optClusterCorrectionNPermutations:
		if val >= 1 {
			m.clusterCorrectionNPermutations = int(val)
		}
	case optClusterCorrectionAlpha:
		if val > 0 && val <= 1 {
			m.clusterCorrectionAlpha = val
		}
	case optClusterCorrectionFormingThreshold:
		if val > 0 && val <= 1 {
			m.clusterCorrectionFormingThreshold = val
		}
	case optClusterCorrectionMinTimepoints:
		if val >= 1 {
			m.clusterCorrectionMinTimepoints = int(val)
		}
	case optClusterCorrectionMinChannels:
		if val >= 1 {
			m.clusterCorrectionMinChannels = int(val)
		}
	case optClusterCorrectionMinClusterSize:
		if val >= 1 {
			m.clusterCorrectionMinClusterSize = int(val)
		}
	// TF Heatmap options
	case optTemporalTfHeatmapTimeResMs:
		if val >= 1 {
			m.tfHeatmapTimeResMs = int(val)
		}
	case optConditionMinTrials:
		if val >= 0 {
			m.conditionMinTrials = int(val)
		}
	case optClusterMinSize:
		if val >= 1 {
			m.clusterMinSize = int(val)
		}
	case optClusterThreshold:
		if val > 0 && val <= 1 {
			m.clusterThreshold = val
		}
	case optConditionEffectThreshold:
		m.conditionEffectThreshold = val
	case optStatisticsAlpha:
		if val > 0 && val <= 1 {
			m.statisticsAlpha = val
		}
	}
}

func (m *Model) commitMLNumber(val float64) {
	options := m.getMLOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optMLNPerm:
		if val >= 0 {
			m.mlNPerm = int(val)
		}
	case optMLInnerSplits:
		if val >= 2 {
			m.innerSplits = int(val)
		}
	case optMLOuterJobs:
		if val >= 1 {
			m.outerJobs = int(val)
		}
	case optMLBinaryThreshold:
		m.mlBinaryThreshold = val
	case optMLUncertaintyAlpha:
		if val > 0 && val < 1 {
			m.mlUncertaintyAlpha = val
		}
	case optMLPermNRepeats:
		if val >= 1 {
			m.mlPermNRepeats = int(val)
		}
	case optMLPlotDPI:
		if val >= 72 {
			m.mlPlotDPI = int(val)
		}
	case optMLPlotTopNFeatures:
		if val >= 1 {
			m.mlPlotTopNFeatures = int(val)
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = int(val)
		}
	case optRfNEstimators:
		if val >= 1 {
			m.rfNEstimators = int(val)
		}
	case optMLFmriSigRoundDecimals:
		if val >= 0 {
			m.mlFmriSigRoundDecimals = int(val)
		}
	// ML Preprocessing
	case optMLPCANComponents:
		if val > 0 {
			m.mlPCANComponents = val
		}
	case optMLFeatureSelectionPercentile:
		if val > 0 && val <= 100 {
			m.mlFeatureSelectionPercentile = val
		}
	case optMLPCARngSeed:
		if val >= 0 {
			m.mlPCARngSeed = int(val)
		}
	case optMLClassificationResamplerSeed:
		if val >= 0 {
			m.mlClassificationResamplerSeed = int(val)
		}
	// ML LR
	case optMLLrMaxIter:
		if val >= 1 {
			m.mlLrMaxIter = int(val)
		}
	// ML CNN
	case optMLCnnFilters1:
		if val >= 1 {
			m.mlCnnFilters1 = int(val)
		}
	case optMLCnnFilters2:
		if val >= 1 {
			m.mlCnnFilters2 = int(val)
		}
	case optMLCnnKernelSize1:
		if val >= 1 {
			m.mlCnnKernelSize1 = int(val)
		}
	case optMLCnnKernelSize2:
		if val >= 1 {
			m.mlCnnKernelSize2 = int(val)
		}
	case optMLCnnDropoutDense:
		if val >= 0 && val < 1 {
			m.mlCnnDropoutDense = val
		}
	case optMLCnnBatchSize:
		if val >= 1 {
			m.mlCnnBatchSize = int(val)
		}
	case optMLCnnEpochs:
		if val >= 1 {
			m.mlCnnEpochs = int(val)
		}
	case optMLCnnLearningRate:
		if val > 0 {
			m.mlCnnLearningRate = val
		}
	case optMLCnnPatience:
		if val >= 1 {
			m.mlCnnPatience = int(val)
		}
	case optMLCnnL2Lambda:
		if val >= 0 {
			m.mlCnnL2Lambda = val
		}
	case optMLCnnRandomSeed:
		if val >= 0 {
			m.mlCnnRandomSeed = int(val)
		}
	// ML CV / Evaluation / Analysis
	case optMLCvMinValidPermFraction:
		if val > 0 && val <= 1 {
			m.mlCvMinValidPermFraction = val
		}
	case optMLCvDefaultNBins:
		if val >= 2 {
			m.mlCvDefaultNBins = int(val)
		}
	case optMLEvalBootstrapIterations:
		if val >= 1 {
			m.mlEvalBootstrapIterations = int(val)
		}
	case optMLDataMaxExcludedSubjectFraction:
		if val >= 0 && val <= 1 {
			m.mlDataMaxExcludedSubjectFraction = val
		}
	case optMLIncrementalBaselineAlpha:
		if val > 0 && val < 1 {
			m.mlIncrementalBaselineAlpha = val
		}
	case optMLTimeGenMinSubjects:
		if val >= 1 {
			m.mlTimeGenMinSubjects = int(val)
		}
	case optMLTimeGenMinValidPermFraction:
		if val > 0 && val <= 1 {
			m.mlTimeGenMinValidPermFraction = val
		}
	case optMLClassMinSubjectsForAUC:
		if val >= 1 {
			m.mlClassMinSubjectsForAUC = int(val)
		}
	case optMLClassMaxFailedFoldFraction:
		if val >= 0 && val <= 1 {
			m.mlClassMaxFailedFoldFraction = val
		}
	}
}

func (m *Model) commitPreprocessingNumber(val float64) {
	options := m.getPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optPrepNJobs:
		if val >= -1 {
			m.prepNJobs = int(val)
		}
	case optPrepResample:
		if val > 0 {
			m.prepResample = int(val)
		}
	case optPrepLFreq:
		m.prepLFreq = val
	case optPrepHFreq:
		m.prepHFreq = val
	case optPrepNotch:
		if val > 0 {
			m.prepNotch = int(val)
		}
	case optPrepLineFreq:
		if val > 0 {
			m.prepLineFreq = int(val)
		}
	case optPrepZaplineFline:
		if val > 0 {
			m.prepZaplineFline = val
		}
	case optPrepICAComp:
		if val > 0 {
			m.prepICAComp = val
		}
	case optPrepICALFreq:
		if val > 0 {
			m.prepICALFreq = val
		}
	case optPrepICARejThresh:
		if val >= 0 {
			m.prepICARejThresh = val
		}
	case optPrepProbThresh:
		if val >= 0 && val <= 1 {
			m.prepProbThresh = val
		}
	case optPrepRepeats:
		if val >= 1 {
			m.prepRepeats = int(val)
		}
	case optPrepBreaksMinLength:
		if val > 0 {
			m.prepBreaksMinLength = int(val)
		}
	case optPrepTStartAfterPrevious:
		if val >= 0 {
			m.prepTStartAfterPrevious = int(val)
		}
	case optPrepTStopBeforeNext:
		if val >= 0 {
			m.prepTStopBeforeNext = int(val)
		}
	case optPrepEpochsTmin:
		m.prepEpochsTmin = val
	case optPrepEpochsTmax:
		m.prepEpochsTmax = val
	case optPrepEpochsBaseline:
		// For baseline, user enters a single number (start), and we assume end = 0
		// A more sophisticated approach would use text field for "start end" format
		m.prepEpochsBaselineStart = val
		m.prepEpochsBaselineEnd = 0
	case optPrepEpochsReject:
		if val >= 0 {
			m.prepEpochsReject = val
		}
	// Alignment
	case optAlignMinAlignmentSamples:
		if val >= 0 {
			m.alignMinAlignmentSamples = int(val)
		}
	}
}

func (m *Model) commitFmriNumber(val float64) {
	options := m.getFmriPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optFmriNThreads:
		if val >= 0 {
			m.fmriNThreads = int(val)
		}
	case optFmriOmpNThreads:
		if val >= 0 {
			m.fmriOmpNThreads = int(val)
		}
	case optFmriMemMb:
		if val >= 0 {
			m.fmriMemMb = int(val)
		}
	case optFmriBold2T1wDof:
		if val >= 0 {
			m.fmriBold2T1wDof = int(val)
		}
	case optFmriSliceTimeRef:
		if val >= 0 && val <= 1 {
			m.fmriSliceTimeRef = val
		}
	case optFmriDummyScans:
		if val >= 0 {
			m.fmriDummyScans = int(val)
		}
	case optFmriFdSpikeThreshold:
		if val >= 0 {
			m.fmriFdSpikeThreshold = val
		}
	case optFmriDvarsSpikeThreshold:
		if val >= 0 {
			m.fmriDvarsSpikeThreshold = val
		}
	case optFmriRandomSeed:
		if val >= 0 {
			m.fmriRandomSeed = int(val)
		}
	}
}

func (m *Model) commitFmriAnalysisNumber(val float64) {
	options := m.getFmriAnalysisOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optFmriAnalysisHighPassHz:
		if val >= 0 {
			m.fmriAnalysisHighPassHz = val
		}
	case optFmriAnalysisLowPassHz:
		if val >= 0 {
			m.fmriAnalysisLowPassHz = val
		}
	case optFmriAnalysisSmoothingFwhm:
		if val >= 0 {
			m.fmriAnalysisSmoothingFwhm = val
		}
	case optFmriAnalysisPlotZThreshold:
		if val > 0 {
			m.fmriAnalysisPlotZThreshold = val
		}
	case optFmriAnalysisPlotFdrQ:
		if val > 0 && val <= 1 {
			m.fmriAnalysisPlotFdrQ = val
		}
	case optFmriAnalysisPlotClusterMinVoxels:
		if val >= 0 {
			m.fmriAnalysisPlotClusterMinVoxels = int(val)
		}
	case optFmriAnalysisPlotVmaxManual:
		if val > 0 {
			m.fmriAnalysisPlotVmaxManual = val
		}
	case optFmriTrialSigMaxTrialsPerRun:
		if val >= 0 {
			m.fmriTrialSigMaxTrialsPerRun = int(val)
		}
	case optFmriSecondLevelPermutationCount:
		if val > 0 {
			m.fmriSecondLevelPermutationCount = int(val)
		}
	}
}

// findNextVisiblePlot finds the next plot index in a visible category
