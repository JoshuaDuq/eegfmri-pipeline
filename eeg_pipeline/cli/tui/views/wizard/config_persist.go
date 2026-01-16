package wizard

// ExportConfig exports all advanced configuration options to a map for persistence.
// This includes all pipeline-specific settings (features, behavior, preprocessing, fMRI, plotting, ML).
func (m Model) ExportConfig() map[string]interface{} {
	cfg := make(map[string]interface{})

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

	// Aperiodic configuration
	cfg["aperiodicFmin"] = m.aperiodicFmin
	cfg["aperiodicFmax"] = m.aperiodicFmax
	cfg["aperiodicPeakZ"] = m.aperiodicPeakZ
	cfg["aperiodicMinR2"] = m.aperiodicMinR2
	cfg["aperiodicMinPoints"] = m.aperiodicMinPoints
	cfg["aperiodicMinSegmentSec"] = m.aperiodicMinSegmentSec
	cfg["aperiodicModel"] = m.aperiodicModel
	cfg["aperiodicPsdMethod"] = m.aperiodicPsdMethod

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
	cfg["erpLowpassHz"] = m.erpLowpassHz

	// Burst configuration
	cfg["burstThresholdZ"] = m.burstThresholdZ
	cfg["burstThresholdMethod"] = m.burstThresholdMethod
	cfg["burstThresholdPercentile"] = m.burstThresholdPercentile
	cfg["burstMinDuration"] = m.burstMinDuration
	cfg["burstMinCycles"] = m.burstMinCycles
	cfg["burstBandsSpec"] = m.burstBandsSpec

	// Power/Spectral configuration
	cfg["powerBaselineMode"] = m.powerBaselineMode
	cfg["powerRequireBaseline"] = m.powerRequireBaseline
	cfg["spectralEdgePercentile"] = m.spectralEdgePercentile
	cfg["spectralRatioPairsSpec"] = m.spectralRatioPairsSpec
	cfg["spectralPsdMethod"] = m.spectralPsdMethod
	cfg["spectralFmin"] = m.spectralFmin
	cfg["spectralFmax"] = m.spectralFmax

	// Connectivity configuration
	cfg["connOutputLevel"] = m.connOutputLevel
	cfg["connGraphMetrics"] = m.connGraphMetrics
	cfg["connGraphProp"] = m.connGraphProp
	cfg["connWindowLen"] = m.connWindowLen
	cfg["connWindowStep"] = m.connWindowStep
	cfg["connAECMode"] = m.connAECMode
	cfg["connGranularity"] = m.connGranularity
	cfg["connMinEpochsPerGroup"] = m.connMinEpochsPerGroup

	// ITPC configuration
	cfg["itpcMethod"] = m.itpcMethod
	cfg["itpcConditionColumn"] = m.itpcConditionColumn
	cfg["itpcConditionValues"] = m.itpcConditionValues
	cfg["itpcMinTrialsPerCondition"] = m.itpcMinTrialsPerCondition

	// Source localization
	cfg["sourceLocEnabled"] = m.sourceLocEnabled
	cfg["sourceLocMode"] = m.sourceLocMode
	cfg["sourceLocMethod"] = m.sourceLocMethod
	cfg["sourceLocSpacing"] = m.sourceLocSpacing
	cfg["sourceLocParc"] = m.sourceLocParc
	cfg["sourceLocReg"] = m.sourceLocReg
	cfg["sourceLocSnr"] = m.sourceLocSnr

	// Aggregation/storage
	cfg["aggregationMethod"] = m.aggregationMethod
	cfg["minEpochsForFeatures"] = m.minEpochsForFeatures
	cfg["saveSubjectLevelFeatures"] = m.saveSubjectLevelFeatures

	// Spatial transform
	cfg["spatialTransform"] = m.spatialTransform
	cfg["spatialTransformLambda2"] = m.spatialTransformLambda2

	// TFR parameters
	cfg["tfrFreqMin"] = m.tfrFreqMin
	cfg["tfrFreqMax"] = m.tfrFreqMax
	cfg["tfrNFreqs"] = m.tfrNFreqs
	cfg["tfrMinCycles"] = m.tfrMinCycles
	cfg["tfrDecim"] = m.tfrDecim

	// Behavior pipeline
	cfg["correlationMethod"] = m.correlationMethod
	cfg["robustCorrelation"] = m.robustCorrelation
	cfg["bootstrapSamples"] = m.bootstrapSamples
	cfg["nPermutations"] = m.nPermutations
	cfg["rngSeed"] = m.rngSeed
	cfg["fdrAlpha"] = m.fdrAlpha
	cfg["behaviorNJobs"] = m.behaviorNJobs
	cfg["alsoSaveCsv"] = m.alsoSaveCsv

	// Trial table
	cfg["trialTableFormat"] = m.trialTableFormat
	cfg["trialTableIncludeFeatures"] = m.trialTableIncludeFeatures
	cfg["trialTableIncludeCovars"] = m.trialTableIncludeCovars
	cfg["trialTableIncludeEvents"] = m.trialTableIncludeEvents

	// Pain residual
	cfg["painResidualEnabled"] = m.painResidualEnabled
	cfg["painResidualMethod"] = m.painResidualMethod
	cfg["painResidualPolyDegree"] = m.painResidualPolyDegree

	// Regression
	cfg["regressionOutcome"] = m.regressionOutcome
	cfg["regressionIncludeTemperature"] = m.regressionIncludeTemperature
	cfg["regressionTempControl"] = m.regressionTempControl
	cfg["regressionPermutations"] = m.regressionPermutations

	// Correlations
	cfg["correlationsTargetRating"] = m.correlationsTargetRating
	cfg["correlationsTargetTemperature"] = m.correlationsTargetTemperature
	cfg["correlationsTargetPainResidual"] = m.correlationsTargetPainResidual

	// Temporal
	cfg["temporalResolutionMs"] = m.temporalResolutionMs
	cfg["temporalSmoothMs"] = m.temporalSmoothMs
	cfg["temporalTimeMinMs"] = m.temporalTimeMinMs
	cfg["temporalTimeMaxMs"] = m.temporalTimeMaxMs

	// Preprocessing
	cfg["prepUsePyprep"] = m.prepUsePyprep
	cfg["prepUseIcalabel"] = m.prepUseIcalabel
	cfg["prepNJobs"] = m.prepNJobs
	cfg["prepMontage"] = m.prepMontage
	cfg["prepResample"] = m.prepResample
	cfg["prepLFreq"] = m.prepLFreq
	cfg["prepHFreq"] = m.prepHFreq
	cfg["prepNotch"] = m.prepNotch
	cfg["prepLineFreq"] = m.prepLineFreq
	cfg["prepEpochsTmin"] = m.prepEpochsTmin
	cfg["prepEpochsTmax"] = m.prepEpochsTmax
	cfg["prepEpochsReject"] = m.prepEpochsReject
	cfg["prepICAAlgorithm"] = m.prepICAAlgorithm
	cfg["prepICAComp"] = m.prepICAComp
	cfg["prepProbThresh"] = m.prepProbThresh
	cfg["icaLabelsToKeep"] = m.icaLabelsToKeep

	// ML pipeline
	cfg["mlNPerm"] = m.mlNPerm
	cfg["innerSplits"] = m.innerSplits
	cfg["outerJobs"] = m.outerJobs
	cfg["skipTimeGen"] = m.skipTimeGen

	// System
	cfg["systemNJobs"] = m.systemNJobs
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

	// Aperiodic
	m.aperiodicFmin = getFloat("aperiodicFmin", m.aperiodicFmin)
	m.aperiodicFmax = getFloat("aperiodicFmax", m.aperiodicFmax)
	m.aperiodicPeakZ = getFloat("aperiodicPeakZ", m.aperiodicPeakZ)
	m.aperiodicMinR2 = getFloat("aperiodicMinR2", m.aperiodicMinR2)
	m.aperiodicMinPoints = getInt("aperiodicMinPoints", m.aperiodicMinPoints)
	m.aperiodicMinSegmentSec = getFloat("aperiodicMinSegmentSec", m.aperiodicMinSegmentSec)
	m.aperiodicModel = getInt("aperiodicModel", m.aperiodicModel)
	m.aperiodicPsdMethod = getInt("aperiodicPsdMethod", m.aperiodicPsdMethod)

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
	m.erpLowpassHz = getFloat("erpLowpassHz", m.erpLowpassHz)

	// Burst
	m.burstThresholdZ = getFloat("burstThresholdZ", m.burstThresholdZ)
	m.burstThresholdMethod = getInt("burstThresholdMethod", m.burstThresholdMethod)
	m.burstThresholdPercentile = getFloat("burstThresholdPercentile", m.burstThresholdPercentile)
	m.burstMinDuration = getInt("burstMinDuration", m.burstMinDuration)
	m.burstMinCycles = getFloat("burstMinCycles", m.burstMinCycles)
	m.burstBandsSpec = getString("burstBandsSpec", m.burstBandsSpec)

	// Power/Spectral
	m.powerBaselineMode = getInt("powerBaselineMode", m.powerBaselineMode)
	m.powerRequireBaseline = getBool("powerRequireBaseline", m.powerRequireBaseline)
	m.spectralEdgePercentile = getFloat("spectralEdgePercentile", m.spectralEdgePercentile)
	m.spectralRatioPairsSpec = getString("spectralRatioPairsSpec", m.spectralRatioPairsSpec)
	m.spectralPsdMethod = getInt("spectralPsdMethod", m.spectralPsdMethod)
	m.spectralFmin = getFloat("spectralFmin", m.spectralFmin)
	m.spectralFmax = getFloat("spectralFmax", m.spectralFmax)

	// Connectivity
	m.connOutputLevel = getInt("connOutputLevel", m.connOutputLevel)
	m.connGraphMetrics = getBool("connGraphMetrics", m.connGraphMetrics)
	m.connGraphProp = getFloat("connGraphProp", m.connGraphProp)
	m.connWindowLen = getFloat("connWindowLen", m.connWindowLen)
	m.connWindowStep = getFloat("connWindowStep", m.connWindowStep)
	m.connAECMode = getInt("connAECMode", m.connAECMode)
	m.connGranularity = getInt("connGranularity", m.connGranularity)
	m.connMinEpochsPerGroup = getInt("connMinEpochsPerGroup", m.connMinEpochsPerGroup)

	// ITPC
	m.itpcMethod = getInt("itpcMethod", m.itpcMethod)
	m.itpcConditionColumn = getString("itpcConditionColumn", m.itpcConditionColumn)
	m.itpcConditionValues = getString("itpcConditionValues", m.itpcConditionValues)
	m.itpcMinTrialsPerCondition = getInt("itpcMinTrialsPerCondition", m.itpcMinTrialsPerCondition)

	// Source localization
	m.sourceLocEnabled = getBool("sourceLocEnabled", m.sourceLocEnabled)
	m.sourceLocMode = getInt("sourceLocMode", m.sourceLocMode)
	m.sourceLocMethod = getInt("sourceLocMethod", m.sourceLocMethod)
	m.sourceLocSpacing = getInt("sourceLocSpacing", m.sourceLocSpacing)
	m.sourceLocParc = getInt("sourceLocParc", m.sourceLocParc)
	m.sourceLocReg = getFloat("sourceLocReg", m.sourceLocReg)
	m.sourceLocSnr = getFloat("sourceLocSnr", m.sourceLocSnr)

	// Aggregation/storage
	m.aggregationMethod = getInt("aggregationMethod", m.aggregationMethod)
	m.minEpochsForFeatures = getInt("minEpochsForFeatures", m.minEpochsForFeatures)
	m.saveSubjectLevelFeatures = getBool("saveSubjectLevelFeatures", m.saveSubjectLevelFeatures)

	// Spatial transform
	m.spatialTransform = getInt("spatialTransform", m.spatialTransform)
	m.spatialTransformLambda2 = getFloat("spatialTransformLambda2", m.spatialTransformLambda2)

	// TFR
	m.tfrFreqMin = getFloat("tfrFreqMin", m.tfrFreqMin)
	m.tfrFreqMax = getFloat("tfrFreqMax", m.tfrFreqMax)
	m.tfrNFreqs = getInt("tfrNFreqs", m.tfrNFreqs)
	m.tfrMinCycles = getFloat("tfrMinCycles", m.tfrMinCycles)
	m.tfrDecim = getInt("tfrDecim", m.tfrDecim)

	// Behavior
	m.correlationMethod = getString("correlationMethod", m.correlationMethod)
	m.robustCorrelation = getInt("robustCorrelation", m.robustCorrelation)
	m.bootstrapSamples = getInt("bootstrapSamples", m.bootstrapSamples)
	m.nPermutations = getInt("nPermutations", m.nPermutations)
	m.rngSeed = getInt("rngSeed", m.rngSeed)
	m.fdrAlpha = getFloat("fdrAlpha", m.fdrAlpha)
	m.behaviorNJobs = getInt("behaviorNJobs", m.behaviorNJobs)
	m.alsoSaveCsv = getBool("alsoSaveCsv", m.alsoSaveCsv)

	// Trial table
	m.trialTableFormat = getInt("trialTableFormat", m.trialTableFormat)
	m.trialTableIncludeFeatures = getBool("trialTableIncludeFeatures", m.trialTableIncludeFeatures)
	m.trialTableIncludeCovars = getBool("trialTableIncludeCovars", m.trialTableIncludeCovars)
	m.trialTableIncludeEvents = getBool("trialTableIncludeEvents", m.trialTableIncludeEvents)

	// Pain residual
	m.painResidualEnabled = getBool("painResidualEnabled", m.painResidualEnabled)
	m.painResidualMethod = getInt("painResidualMethod", m.painResidualMethod)
	m.painResidualPolyDegree = getInt("painResidualPolyDegree", m.painResidualPolyDegree)

	// Regression
	m.regressionOutcome = getInt("regressionOutcome", m.regressionOutcome)
	m.regressionIncludeTemperature = getBool("regressionIncludeTemperature", m.regressionIncludeTemperature)
	m.regressionTempControl = getInt("regressionTempControl", m.regressionTempControl)
	m.regressionPermutations = getInt("regressionPermutations", m.regressionPermutations)

	// Correlations
	m.correlationsTargetRating = getBool("correlationsTargetRating", m.correlationsTargetRating)
	m.correlationsTargetTemperature = getBool("correlationsTargetTemperature", m.correlationsTargetTemperature)
	m.correlationsTargetPainResidual = getBool("correlationsTargetPainResidual", m.correlationsTargetPainResidual)

	// Temporal
	m.temporalResolutionMs = getInt("temporalResolutionMs", m.temporalResolutionMs)
	m.temporalSmoothMs = getInt("temporalSmoothMs", m.temporalSmoothMs)
	m.temporalTimeMinMs = getInt("temporalTimeMinMs", m.temporalTimeMinMs)
	m.temporalTimeMaxMs = getInt("temporalTimeMaxMs", m.temporalTimeMaxMs)

	// Preprocessing
	m.prepUsePyprep = getBool("prepUsePyprep", m.prepUsePyprep)
	m.prepUseIcalabel = getBool("prepUseIcalabel", m.prepUseIcalabel)
	m.prepNJobs = getInt("prepNJobs", m.prepNJobs)
	m.prepMontage = getString("prepMontage", m.prepMontage)
	m.prepResample = getInt("prepResample", m.prepResample)
	m.prepLFreq = getFloat("prepLFreq", m.prepLFreq)
	m.prepHFreq = getFloat("prepHFreq", m.prepHFreq)
	m.prepNotch = getInt("prepNotch", m.prepNotch)
	m.prepLineFreq = getInt("prepLineFreq", m.prepLineFreq)
	m.prepEpochsTmin = getFloat("prepEpochsTmin", m.prepEpochsTmin)
	m.prepEpochsTmax = getFloat("prepEpochsTmax", m.prepEpochsTmax)
	m.prepEpochsReject = getFloat("prepEpochsReject", m.prepEpochsReject)
	m.prepICAAlgorithm = getInt("prepICAAlgorithm", m.prepICAAlgorithm)
	m.prepICAComp = getFloat("prepICAComp", m.prepICAComp)
	m.prepProbThresh = getFloat("prepProbThresh", m.prepProbThresh)
	m.icaLabelsToKeep = getString("icaLabelsToKeep", m.icaLabelsToKeep)

	// ML
	m.mlNPerm = getInt("mlNPerm", m.mlNPerm)
	m.innerSplits = getInt("innerSplits", m.innerSplits)
	m.outerJobs = getInt("outerJobs", m.outerJobs)
	m.skipTimeGen = getBool("skipTimeGen", m.skipTimeGen)

	// System
	m.systemNJobs = getInt("systemNJobs", m.systemNJobs)
	m.loggingLevel = getInt("loggingLevel", m.loggingLevel)
}
