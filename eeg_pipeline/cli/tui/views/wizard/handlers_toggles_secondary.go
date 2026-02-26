package wizard

import (
	"strings"
)

// Secondary advanced-toggle handlers (ML/preprocessing/fMRI analysis).

func (m *Model) toggleMLAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

	options := m.getMLOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optConfigSetOverrides:
		m.startTextEdit(textFieldConfigSetOverrides)
		m.useDefaultAdvanced = false
	case optMLGroupData:
		m.mlGroupDataExpanded = !m.mlGroupDataExpanded
	case optMLGroupModel:
		m.mlGroupModelExpanded = !m.mlGroupModelExpanded
	case optMLGroupTraining:
		m.mlGroupTrainingExpanded = !m.mlGroupTrainingExpanded
	case optMLGroupOutput:
		m.mlGroupOutputExpanded = !m.mlGroupOutputExpanded
	case optMLNPerm, optMLInnerSplits, optMLOuterJobs, optRNGSeed, optRfNEstimators, optMLBinaryThreshold, optMLUncertaintyAlpha, optMLPermNRepeats, optMLPlotDPI, optMLPlotTopNFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLTarget:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedMLTargetColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldMLTarget)
		}
		m.useDefaultAdvanced = false
	case optMLFmriSigGroup:
		m.mlFmriSigGroupExpanded = !m.mlFmriSigGroupExpanded
		m.useDefaultAdvanced = false
	case optMLFmriSigMethod:
		m.mlFmriSigMethodIndex = (m.mlFmriSigMethodIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optMLFmriSigContrastName:
		m.startTextEdit(textFieldMLFmriSigContrastName)
		m.useDefaultAdvanced = false
	case optMLFmriSigSignature:
		m.mlFmriSigSignatureIndex = (m.mlFmriSigSignatureIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optMLFmriSigMetric:
		m.mlFmriSigMetricIndex = (m.mlFmriSigMetricIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optMLFmriSigNormalization:
		m.mlFmriSigNormalizationIndex = (m.mlFmriSigNormalizationIndex + 1) % 5
		m.useDefaultAdvanced = false
	case optMLFmriSigRoundDecimals:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLRegressionModel:
		m.mlRegressionModel = m.mlRegressionModel.Next()
		m.useDefaultAdvanced = false
	case optMLClassificationModel:
		m.mlClassificationModel = m.mlClassificationModel.Next()
		m.useDefaultAdvanced = false
	case optMLBinaryThresholdEnabled:
		m.mlBinaryThresholdEnabled = !m.mlBinaryThresholdEnabled
		m.useDefaultAdvanced = false
	case optMLFeatureFamilies:
		m.expandedOption = expandedMLFeatureFamilies
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optMLFeatureBands:
		m.expandedOption = expandedMLFeatureBands
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optMLFeatureSegments:
		m.expandedOption = expandedMLFeatureSegments
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optMLFeatureScopes:
		m.expandedOption = expandedMLFeatureScopes
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optMLFeatureStats:
		m.expandedOption = expandedMLFeatureStats
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optMLFeatureHarmonization:
		m.mlFeatureHarmonization = m.mlFeatureHarmonization.Next()
		m.useDefaultAdvanced = false
	case optMLCovariates:
		m.startTextEdit(textFieldMLCovariates)
		m.useDefaultAdvanced = false
	case optMLBaselinePredictors:
		m.startTextEdit(textFieldMLBaselinePredictors)
		m.useDefaultAdvanced = false
	case optMLRequireTrialMlSafe:
		m.mlRequireTrialMlSafe = !m.mlRequireTrialMlSafe
		m.useDefaultAdvanced = false
	case optMLPlotsEnabled:
		m.mlPlotsEnabled = !m.mlPlotsEnabled
		m.useDefaultAdvanced = false
	case optMLPlotFormats:
		m.startTextEdit(textFieldMLPlotFormats)
		m.useDefaultAdvanced = false
	case optMLPlotDiagnostics:
		m.mlPlotDiagnostics = !m.mlPlotDiagnostics
		m.useDefaultAdvanced = false
	case optElasticNetAlphaGrid:
		m.startTextEdit(textFieldElasticNetAlphaGrid)
		m.useDefaultAdvanced = false
	case optElasticNetL1RatioGrid:
		m.startTextEdit(textFieldElasticNetL1RatioGrid)
		m.useDefaultAdvanced = false
	case optRidgeAlphaGrid:
		m.startTextEdit(textFieldRidgeAlphaGrid)
		m.useDefaultAdvanced = false
	case optRfMaxDepthGrid:
		m.startTextEdit(textFieldRfMaxDepthGrid)
		m.useDefaultAdvanced = false
	case optVarianceThresholdGrid:
		m.startTextEdit(textFieldVarianceThresholdGrid)
		m.useDefaultAdvanced = false
	// ML Preprocessing group
	case optMLGroupPreprocessing:
		m.mlGroupPreprocessingExpanded = !m.mlGroupPreprocessingExpanded
		m.useDefaultAdvanced = false
	case optMLImputer:
		m.mlImputer = (m.mlImputer + 1) % 3
		m.useDefaultAdvanced = false
	case optMLPowerTransformerMethod:
		m.mlPowerTransformerMethod = (m.mlPowerTransformerMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optMLPowerTransformerStandardize:
		m.mlPowerTransformerStandardize = !m.mlPowerTransformerStandardize
		m.useDefaultAdvanced = false
	case optMLDeconfound:
		m.mlDeconfound = !m.mlDeconfound
		m.useDefaultAdvanced = false
	case optMLFeatureSelectionPercentile:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLSpatialRegionsAllowed:
		m.startTextEdit(textFieldMLSpatialRegionsAllowed)
		m.useDefaultAdvanced = false
	case optMLClassificationResampler:
		m.mlClassificationResampler = (m.mlClassificationResampler + 1) % 3
		m.useDefaultAdvanced = false
	case optMLClassificationResamplerSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLPCAEnabled:
		m.mlPCAEnabled = !m.mlPCAEnabled
		m.useDefaultAdvanced = false
	case optMLPCANComponents, optMLPCARngSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLPCAWhiten:
		m.mlPCAWhiten = !m.mlPCAWhiten
		m.useDefaultAdvanced = false
	case optMLPCASvdSolver:
		m.mlPCASvdSolver = (m.mlPCASvdSolver + 1) % 3
		m.useDefaultAdvanced = false
	// SVM
	case optMLSvmKernel:
		m.mlSvmKernel = (m.mlSvmKernel + 1) % 3
		m.useDefaultAdvanced = false
	case optMLSvmCGrid:
		m.startTextEdit(textFieldMLSvmCGrid)
		m.useDefaultAdvanced = false
	case optMLSvmGammaGrid:
		m.startTextEdit(textFieldMLSvmGammaGrid)
		m.useDefaultAdvanced = false
	case optMLSvmClassWeight:
		m.mlSvmClassWeight = (m.mlSvmClassWeight + 1) % 2
		m.useDefaultAdvanced = false
	// Logistic Regression
	case optMLLrPenalty:
		m.mlLrPenalty = (m.mlLrPenalty + 1) % 3
		m.useDefaultAdvanced = false
	case optMLLrCGrid:
		m.startTextEdit(textFieldMLLrCGrid)
	case optMLLrL1RatioGrid:
		m.startTextEdit(textFieldMLLrL1RatioGrid)
		m.useDefaultAdvanced = false
	case optMLLrMaxIter:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLLrClassWeight:
		m.mlLrClassWeight = (m.mlLrClassWeight + 1) % 2
		m.useDefaultAdvanced = false
	// Random Forest extras
	case optMLRfMinSamplesSplitGrid:
		m.startTextEdit(textFieldMLRfMinSamplesSplitGrid)
		m.useDefaultAdvanced = false
	case optMLRfMinSamplesLeafGrid:
		m.startTextEdit(textFieldMLRfMinSamplesLeafGrid)
		m.useDefaultAdvanced = false
	case optMLRfBootstrap:
		m.mlRfBootstrap = !m.mlRfBootstrap
		m.useDefaultAdvanced = false
	case optMLRfClassWeight:
		m.mlRfClassWeight = (m.mlRfClassWeight + 1) % 3
		m.useDefaultAdvanced = false
	// Ensemble extras
	case optMLEnsembleCalibrate:
		m.mlEnsembleCalibrate = !m.mlEnsembleCalibrate
		m.useDefaultAdvanced = false
	// CNN group
	case optMLGroupCNN:
		m.mlGroupCNNExpanded = !m.mlGroupCNNExpanded
		m.useDefaultAdvanced = false
	case optMLCnnFilters1, optMLCnnFilters2, optMLCnnKernelSize1, optMLCnnKernelSize2,
		optMLCnnPoolSize, optMLCnnDenseUnits, optMLCnnBatchSize, optMLCnnEpochs,
		optMLCnnPatience, optMLCnnRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLCnnDropoutConv, optMLCnnDropoutDense, optMLCnnLearningRate,
		optMLCnnMinDelta, optMLCnnL2Lambda:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// CV / Evaluation / Analysis
	case optMLCvHygieneEnabled:
		m.mlCvHygieneEnabled = !m.mlCvHygieneEnabled
		m.useDefaultAdvanced = false
	case optMLCvPermutationScheme:
		m.mlCvPermutationScheme = (m.mlCvPermutationScheme + 1) % 2
		m.useDefaultAdvanced = false
	case optMLCvMinValidPermFraction, optMLCvDefaultNBins, optMLEvalBootstrapIterations,
		optMLDataMaxExcludedSubjectFraction, optMLIncrementalBaselineAlpha,
		optMLTimeGenMinSubjects, optMLTimeGenMinValidPermFraction,
		optMLClassMinSubjectsForAUC, optMLClassMaxFailedFoldFraction:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLEvalCIMethod:
		m.mlEvalCIMethod = (m.mlEvalCIMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optMLEvalSubjectWeighting:
		m.mlEvalSubjectWeighting = (m.mlEvalSubjectWeighting + 1) % 2
		m.useDefaultAdvanced = false
	case optMLDataCovariatesStrict:
		m.mlDataCovariatesStrict = !m.mlDataCovariatesStrict
		m.useDefaultAdvanced = false
	case optMLTargetsStrictRegressionContinuous:
		m.mlTargetsStrictRegressionCont = !m.mlTargetsStrictRegressionCont
		m.useDefaultAdvanced = false
	case optMLInterpretabilityGroupedOutputs:
		m.mlInterpretabilityGroupedOutputs = !m.mlInterpretabilityGroupedOutputs
		m.useDefaultAdvanced = false
	case optMLIncrementalRequireBaselinePredictors:
		m.mlIncrementalRequireBaselinePred = !m.mlIncrementalRequireBaselinePred
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes
	options = m.getMLOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) togglePreprocessingAdvancedOption() {
	options := m.getPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optConfigSetOverrides:
		m.startTextEdit(textFieldConfigSetOverrides)
		m.useDefaultAdvanced = false
	// Group expansion toggles
	case optPrepGroupStages:
		m.prepGroupStagesExpanded = !m.prepGroupStagesExpanded
	case optPrepGroupGeneral:
		m.prepGroupGeneralExpanded = !m.prepGroupGeneralExpanded
	case optPrepGroupFiltering:
		m.prepGroupFilteringExpanded = !m.prepGroupFilteringExpanded
	case optPrepGroupPyprep:
		m.prepGroupPyprepExpanded = !m.prepGroupPyprepExpanded
	case optPrepGroupICA:
		m.prepGroupICAExpanded = !m.prepGroupICAExpanded
	case optPrepGroupEpoching:
		m.prepGroupEpochingExpanded = !m.prepGroupEpochingExpanded
	// Stage toggles
	case optPrepStageBadChannels:
		m.prepStageSelected[0] = !m.prepStageSelected[0]
		m.useDefaultAdvanced = false
	case optPrepStageFiltering:
		m.prepStageSelected[1] = !m.prepStageSelected[1]
		m.useDefaultAdvanced = false
	case optPrepStageICA:
		m.prepStageSelected[2] = !m.prepStageSelected[2]
		m.useDefaultAdvanced = false
	case optPrepStageEpoching:
		m.prepStageSelected[3] = !m.prepStageSelected[3]
		m.useDefaultAdvanced = false
	case optPrepUsePyprep:
		m.prepUsePyprep = !m.prepUsePyprep
		m.useDefaultAdvanced = false
	case optPrepUseIcalabel:
		m.prepUseIcalabel = !m.prepUseIcalabel
		m.useDefaultAdvanced = false
	case optPrepMontage:
		m.startTextEdit(textFieldPrepMontage)
		m.useDefaultAdvanced = false
	case optPrepChTypes:
		m.startTextEdit(textFieldPrepChTypes)
		m.useDefaultAdvanced = false
	case optPrepEegReference:
		m.startTextEdit(textFieldPrepEegReference)
		m.useDefaultAdvanced = false
	case optPrepEogChannels:
		m.startTextEdit(textFieldPrepEogChannels)
		m.useDefaultAdvanced = false
	case optPrepRandomState:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepTaskIsRest:
		m.prepTaskIsRest = !m.prepTaskIsRest
		m.useDefaultAdvanced = false
	case optPrepNJobs, optPrepResample, optPrepLFreq, optPrepHFreq, optPrepNotch, optPrepLineFreq, optPrepZaplineFline, optPrepICAComp, optPrepICALFreq, optPrepICARejThresh, optPrepProbThresh, optPrepEpochsTmin, optPrepEpochsTmax, optPrepEpochsBaseline, optPrepEpochsReject:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepFindBreaks:
		m.prepFindBreaks = !m.prepFindBreaks
		m.useDefaultAdvanced = false
	case optPrepRansac:
		m.prepRansac = !m.prepRansac
		m.useDefaultAdvanced = false
	case optPrepRepeats, optPrepBreaksMinLength, optPrepTStartAfterPrevious, optPrepTStopBeforeNext:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepAverageReref:
		m.prepAverageReref = !m.prepAverageReref
		m.useDefaultAdvanced = false
	case optPrepFileExtension:
		m.startTextEdit(textFieldPrepFileExtension)
		m.useDefaultAdvanced = false
	case optPrepConsiderPreviousBads:
		m.prepConsiderPreviousBads = !m.prepConsiderPreviousBads
		m.useDefaultAdvanced = false
	case optPrepOverwriteChansTsv:
		m.prepOverwriteChansTsv = !m.prepOverwriteChansTsv
		m.useDefaultAdvanced = false
	case optPrepDeleteBreaks:
		m.prepDeleteBreaks = !m.prepDeleteBreaks
		m.useDefaultAdvanced = false
	case optPrepRenameAnotDict:
		m.startTextEdit(textFieldPrepRenameAnotDict)
		m.useDefaultAdvanced = false
	case optPrepCustomBadDict:
		m.startTextEdit(textFieldPrepCustomBadDict)
		m.useDefaultAdvanced = false
	case optPrepSpatialFilter:
		m.prepSpatialFilter = (m.prepSpatialFilter + 1) % 2
		m.useDefaultAdvanced = false
	case optPrepICAAlgorithm:
		m.prepICAAlgorithm = (m.prepICAAlgorithm + 1) % 4
		m.useDefaultAdvanced = false
	case optPrepKeepMnebidsBads:
		m.prepKeepMnebidsBads = !m.prepKeepMnebidsBads
		m.useDefaultAdvanced = false
	case optIcaLabelsToKeep:
		m.startTextEdit(textFieldIcaLabelsToKeep)
		m.useDefaultAdvanced = false
	case optPrepEpochsNoBaseline:
		m.prepEpochsNoBaseline = !m.prepEpochsNoBaseline
		m.useDefaultAdvanced = false
	case optPrepConditions:
		m.startTextEdit(textFieldPrepConditions)
		m.useDefaultAdvanced = false
	case optPrepRejectMethod:
		m.prepRejectMethod = (m.prepRejectMethod + 1) % 3
		m.useDefaultAdvanced = false
	case optPrepRunSourceEstimation:
		m.prepRunSourceEstimation = !m.prepRunSourceEstimation
	case optPrepWriteCleanEvents:
		m.prepWriteCleanEvents = !m.prepWriteCleanEvents
	case optPrepOverwriteCleanEvents:
		m.prepOverwriteCleanEvents = !m.prepOverwriteCleanEvents
	case optPrepCleanEventsStrict:
		m.prepCleanEventsStrict = !m.prepCleanEventsStrict
		m.useDefaultAdvanced = false
	// ECG channels
	case optPrepEcgChannels:
		m.startTextEdit(textFieldPrepEcgChannels)
		m.useDefaultAdvanced = false
	// Autoreject
	case optPrepAutorejectNInterpolate:
		m.startTextEdit(textFieldPrepAutorejectNInterpolate)
		m.useDefaultAdvanced = false
	// Alignment
	case optAlignAllowMisalignedTrim:
		m.alignAllowMisalignedTrim = !m.alignAllowMisalignedTrim
		m.useDefaultAdvanced = false
	case optAlignMinAlignmentSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAlignTrimToFirstVolume:
		m.alignTrimToFirstVolume = !m.alignTrimToFirstVolume
		m.useDefaultAdvanced = false
	case optAlignFmriOnsetReference:
		m.alignFmriOnsetReference = (m.alignFmriOnsetReference + 1) % 3
		m.useDefaultAdvanced = false
	// Event Column Mapping
	case optEventColPredictor:
		m.startTextEdit(textFieldEventColPredictor)
		m.useDefaultAdvanced = false
	case optEventColRating:
		m.startTextEdit(textFieldEventColOutcome)
		m.useDefaultAdvanced = false
	case optEventColBinaryOutcome:
		m.startTextEdit(textFieldEventColBinaryOutcome)
		m.useDefaultAdvanced = false
	case optEventColCondition:
		m.startTextEdit(textFieldEventColCondition)
		m.useDefaultAdvanced = false
	case optEventColRequired:
		m.startTextEdit(textFieldEventColRequired)
		m.useDefaultAdvanced = false
	case optConditionPreferredPrefixes:
		m.startTextEdit(textFieldConditionPreferredPrefixes)
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes
	options = m.getPreprocessingOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) toggleFmriAdvancedOption() {
	options := m.getFmriPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optConfigSetOverrides:
		m.startTextEdit(textFieldConfigSetOverrides)
		m.useDefaultAdvanced = false

	// Group expansion toggles
	case optFmriGroupRuntime:
		m.fmriGroupRuntimeExpanded = !m.fmriGroupRuntimeExpanded
	case optFmriGroupOutput:
		m.fmriGroupOutputExpanded = !m.fmriGroupOutputExpanded
	case optFmriGroupPerformance:
		m.fmriGroupPerformanceExpanded = !m.fmriGroupPerformanceExpanded
	case optFmriGroupAnatomical:
		m.fmriGroupAnatomicalExpanded = !m.fmriGroupAnatomicalExpanded
	case optFmriGroupBold:
		m.fmriGroupBoldExpanded = !m.fmriGroupBoldExpanded
	case optFmriGroupQc:
		m.fmriGroupQcExpanded = !m.fmriGroupQcExpanded
	case optFmriGroupDenoising:
		m.fmriGroupDenoisingExpanded = !m.fmriGroupDenoisingExpanded
	case optFmriGroupSurface:
		m.fmriGroupSurfaceExpanded = !m.fmriGroupSurfaceExpanded
	case optFmriGroupMultiecho:
		m.fmriGroupMultiechoExpanded = !m.fmriGroupMultiechoExpanded
	case optFmriGroupRepro:
		m.fmriGroupReproExpanded = !m.fmriGroupReproExpanded
	case optFmriGroupValidation:
		m.fmriGroupValidationExpanded = !m.fmriGroupValidationExpanded
	case optFmriGroupAdvanced:
		m.fmriGroupAdvancedExpanded = !m.fmriGroupAdvancedExpanded

	// Runtime
	case optFmriEngine:
		m.fmriEngineIndex = (m.fmriEngineIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriFmriprepImage:
		m.startTextEdit(textFieldFmriFmriprepImage)
		m.useDefaultAdvanced = false

	// Output
	case optFmriOutputSpaces:
		m.startTextEdit(textFieldFmriOutputSpaces)
		m.useDefaultAdvanced = false
	case optFmriIgnore:
		m.startTextEdit(textFieldFmriIgnore)
		m.useDefaultAdvanced = false
	case optFmriLevel:
		m.fmriLevelIndex = (m.fmriLevelIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriCiftiOutput:
		m.fmriCiftiOutputIndex = (m.fmriCiftiOutputIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriTaskId:
		m.startTextEdit(textFieldFmriTaskId)
		m.useDefaultAdvanced = false

	// Performance
	case optFmriNThreads:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriOmpNThreads:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriMemMb:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriLowMem:
		m.fmriLowMem = !m.fmriLowMem
		m.useDefaultAdvanced = false

	// Anatomical
	case optFmriSkipReconstruction:
		m.fmriSkipReconstruction = !m.fmriSkipReconstruction
		m.useDefaultAdvanced = false
	case optFmriLongitudinal:
		m.fmriLongitudinal = !m.fmriLongitudinal
		m.useDefaultAdvanced = false
	case optFmriSkullStripTemplate:
		m.startTextEdit(textFieldFmriSkullStripTemplate)
		m.useDefaultAdvanced = false
	case optFmriSkullStripFixedSeed:
		m.fmriSkullStripFixedSeed = !m.fmriSkullStripFixedSeed
		m.useDefaultAdvanced = false

	// BOLD processing
	case optFmriBold2T1wInit:
		m.fmriBold2T1wInitIndex = (m.fmriBold2T1wInitIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriBold2T1wDof:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriSliceTimeRef:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriDummyScans:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Quality control
	case optFmriFdSpikeThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriDvarsSpikeThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Denoising
	case optFmriUseAroma:
		m.fmriUseAroma = !m.fmriUseAroma
		m.useDefaultAdvanced = false

	// Surface
	case optFmriMedialSurfaceNan:
		m.fmriMedialSurfaceNan = !m.fmriMedialSurfaceNan
		m.useDefaultAdvanced = false
	case optFmriNoMsm:
		m.fmriNoMsm = !m.fmriNoMsm
		m.useDefaultAdvanced = false

	// Multi-echo
	case optFmriMeOutputEchos:
		m.fmriMeOutputEchos = !m.fmriMeOutputEchos
		m.useDefaultAdvanced = false

	// Reproducibility
	case optFmriRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Validation
	case optFmriSkipBidsValidation:
		m.fmriSkipBidsValidation = !m.fmriSkipBidsValidation
		m.useDefaultAdvanced = false
	case optFmriStopOnFirstCrash:
		m.fmriStopOnFirstCrash = !m.fmriStopOnFirstCrash
		m.useDefaultAdvanced = false
	case optFmriCleanWorkdir:
		m.fmriCleanWorkdir = !m.fmriCleanWorkdir
		m.useDefaultAdvanced = false

	// Advanced
	case optFmriExtraArgs:
		m.startTextEdit(textFieldFmriExtraArgs)
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes
	options = m.getFmriPreprocessingOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) toggleFmriAnalysisAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

	options := m.getFmriAnalysisOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optConfigSetOverrides:
		m.startTextEdit(textFieldConfigSetOverrides)
		m.useDefaultAdvanced = false

	// Group headers
	case optFmriAnalysisGroupInput:
		m.fmriAnalysisGroupInputExpanded = !m.fmriAnalysisGroupInputExpanded
	case optFmriAnalysisGroupContrast:
		m.fmriAnalysisGroupContrastExpanded = !m.fmriAnalysisGroupContrastExpanded
	case optFmriAnalysisGroupGLM:
		m.fmriAnalysisGroupGLMExpanded = !m.fmriAnalysisGroupGLMExpanded
	case optFmriAnalysisGroupConfounds:
		m.fmriAnalysisGroupConfoundsExpanded = !m.fmriAnalysisGroupConfoundsExpanded
	case optFmriAnalysisGroupOutput:
		m.fmriAnalysisGroupOutputExpanded = !m.fmriAnalysisGroupOutputExpanded
	case optFmriAnalysisGroupPlotting:
		m.fmriAnalysisGroupPlottingExpanded = !m.fmriAnalysisGroupPlottingExpanded
	case optFmriTrialSigGroup:
		m.fmriTrialSigGroupExpanded = !m.fmriTrialSigGroupExpanded

	// Input
	case optFmriAnalysisInputSource:
		m.fmriAnalysisInputSourceIndex = (m.fmriAnalysisInputSourceIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriAnalysisFmriprepSpace:
		m.startTextEdit(textFieldFmriAnalysisFmriprepSpace)
		m.useDefaultAdvanced = false
	case optFmriAnalysisRequireFmriprep:
		m.fmriAnalysisRequireFmriprep = !m.fmriAnalysisRequireFmriprep
		m.useDefaultAdvanced = false
	case optFmriAnalysisRuns:
		m.startTextEdit(textFieldFmriAnalysisRuns)
		m.useDefaultAdvanced = false

	// Contrast
	case optFmriAnalysisContrastType:
		m.fmriAnalysisContrastType = (m.fmriAnalysisContrastType + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriAnalysisCondAColumn:
		m.expandedOption = expandedFmriAnalysisCondAColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisCondAValue:
		if m.fmriAnalysisCondAColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		m.expandedOption = expandedFmriAnalysisCondAValue
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisCondBColumn:
		m.expandedOption = expandedFmriAnalysisCondBColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisCondBValue:
		if m.fmriAnalysisCondBColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		m.expandedOption = expandedFmriAnalysisCondBValue
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisContrastName:
		m.startTextEdit(textFieldFmriAnalysisContrastName)
		m.useDefaultAdvanced = false
	case optFmriAnalysisFormula:
		m.startTextEdit(textFieldFmriAnalysisFormula)
		m.useDefaultAdvanced = false

	// GLM
	case optFmriAnalysisHrfModel:
		m.fmriAnalysisHrfModel = (m.fmriAnalysisHrfModel + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriAnalysisDriftModel:
		m.fmriAnalysisDriftModel = (m.fmriAnalysisDriftModel + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriAnalysisHighPassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriAnalysisLowPassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriAnalysisSmoothingFwhm:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Confounds / QC
	case optFmriAnalysisEventsToModel:
		m.startTextEdit(textFieldFmriAnalysisEventsToModel)
		m.useDefaultAdvanced = false
	case optFmriAnalysisScopeColumn:
		m.expandedOption = expandedFmriAnalysisScopeColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisScopeTrialTypes:
		m.expandedOption = expandedFmriAnalysisScopeTrialTypes
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisPhaseColumn:
		m.expandedOption = expandedFmriAnalysisPhaseColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisPhaseScopeColumn:
		m.expandedOption = expandedFmriAnalysisPhaseScopeColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisPhaseScopeValue:
		m.expandedOption = expandedFmriAnalysisPhaseScopeValue
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisStimPhasesToModel:
		m.expandedOption = expandedFmriAnalysisStimPhases
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriAnalysisConfoundsStrategy:
		m.fmriAnalysisConfoundsStrategy = (m.fmriAnalysisConfoundsStrategy + 1) % 7
		m.useDefaultAdvanced = false
	case optFmriAnalysisWriteDesignMatrix:
		m.fmriAnalysisWriteDesignMatrix = !m.fmriAnalysisWriteDesignMatrix
		m.useDefaultAdvanced = false

	// Output
	case optFmriAnalysisOutputType:
		m.fmriAnalysisOutputType = (m.fmriAnalysisOutputType + 1) % 4
		m.useDefaultAdvanced = false
	case optFmriAnalysisOutputDir:
		m.startTextEdit(textFieldFmriAnalysisOutputDir)
		m.useDefaultAdvanced = false
	case optFmriAnalysisResampleToFS:
		m.fmriAnalysisResampleToFS = !m.fmriAnalysisResampleToFS
		m.useDefaultAdvanced = false
	case optFmriAnalysisFreesurferDir:
		m.startTextEdit(textFieldFmriAnalysisFreesurferDir)
		m.useDefaultAdvanced = false

	// Plotting / Report
	case optFmriAnalysisPlotsEnabled:
		m.fmriAnalysisPlotsEnabled = !m.fmriAnalysisPlotsEnabled
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotHTML:
		m.fmriAnalysisPlotHTML = !m.fmriAnalysisPlotHTML
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotSpace:
		m.fmriAnalysisPlotSpaceIndex = (m.fmriAnalysisPlotSpaceIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotThresholdMode:
		m.fmriAnalysisPlotThresholdModeIndex = (m.fmriAnalysisPlotThresholdModeIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotZThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotFdrQ:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotClusterMinVoxels:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotVmaxMode:
		m.fmriAnalysisPlotVmaxModeIndex = (m.fmriAnalysisPlotVmaxModeIndex + 1) % 3
		if m.fmriAnalysisPlotVmaxModeIndex%3 == 2 && m.fmriAnalysisPlotVmaxManual <= 0 {
			m.fmriAnalysisPlotVmaxManual = 5.0
		}
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotVmaxManual:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotIncludeUnthresholded:
		m.fmriAnalysisPlotIncludeUnthresholded = !m.fmriAnalysisPlotIncludeUnthresholded
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotFormatPNG:
		next := !m.fmriAnalysisPlotFormatPNG
		if !next && !m.fmriAnalysisPlotFormatSVG {
			m.ShowToast("At least one format is required (PNG/SVG)", "warning")
			return
		}
		m.fmriAnalysisPlotFormatPNG = next
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotFormatSVG:
		next := !m.fmriAnalysisPlotFormatSVG
		if !next && !m.fmriAnalysisPlotFormatPNG {
			m.ShowToast("At least one format is required (PNG/SVG)", "warning")
			return
		}
		m.fmriAnalysisPlotFormatSVG = next
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotTypeSlices:
		next := !m.fmriAnalysisPlotTypeSlices
		if !next && !(m.fmriAnalysisPlotTypeGlass || m.fmriAnalysisPlotTypeHist || m.fmriAnalysisPlotTypeClusters) {
			m.ShowToast("Select at least one plot type", "warning")
			return
		}
		m.fmriAnalysisPlotTypeSlices = next
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotTypeGlass:
		next := !m.fmriAnalysisPlotTypeGlass
		if !next && !(m.fmriAnalysisPlotTypeSlices || m.fmriAnalysisPlotTypeHist || m.fmriAnalysisPlotTypeClusters) {
			m.ShowToast("Select at least one plot type", "warning")
			return
		}
		m.fmriAnalysisPlotTypeGlass = next
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotTypeHist:
		next := !m.fmriAnalysisPlotTypeHist
		if !next && !(m.fmriAnalysisPlotTypeSlices || m.fmriAnalysisPlotTypeGlass || m.fmriAnalysisPlotTypeClusters) {
			m.ShowToast("Select at least one plot type", "warning")
			return
		}
		m.fmriAnalysisPlotTypeHist = next
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotTypeClusters:
		next := !m.fmriAnalysisPlotTypeClusters
		if !next && !(m.fmriAnalysisPlotTypeSlices || m.fmriAnalysisPlotTypeGlass || m.fmriAnalysisPlotTypeHist) {
			m.ShowToast("Select at least one plot type", "warning")
			return
		}
		m.fmriAnalysisPlotTypeClusters = next
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotEffectSize:
		m.fmriAnalysisPlotEffectSize = !m.fmriAnalysisPlotEffectSize
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotStandardError:
		m.fmriAnalysisPlotStandardError = !m.fmriAnalysisPlotStandardError
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotMotionQC:
		m.fmriAnalysisPlotMotionQC = !m.fmriAnalysisPlotMotionQC
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotCarpetQC:
		m.fmriAnalysisPlotCarpetQC = !m.fmriAnalysisPlotCarpetQC
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotTSNRQC:
		m.fmriAnalysisPlotTSNRQC = !m.fmriAnalysisPlotTSNRQC
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotDesignQC:
		m.fmriAnalysisPlotDesignQC = !m.fmriAnalysisPlotDesignQC
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotEmbedImages:
		m.fmriAnalysisPlotEmbedImages = !m.fmriAnalysisPlotEmbedImages
		m.useDefaultAdvanced = false
	case optFmriAnalysisPlotSignatures:
		m.fmriAnalysisPlotSignatures = !m.fmriAnalysisPlotSignatures
		m.useDefaultAdvanced = false
	case optFmriAnalysisSignatureDir:
		m.startTextEdit(textFieldFmriAnalysisSignatureDir)
	case optFmriAnalysisSignatureMaps:
		m.startTextEdit(textFieldFmriAnalysisSignatureMaps)
	case optFmriTrialSigScopeTrialTypeColumn:
		m.expandedOption = expandedFmriTrialSigScopeTrialTypeColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriTrialSigScopePhaseColumn:
		m.expandedOption = expandedFmriTrialSigScopePhaseColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriTrialSigScopeStimPhases:
		m.expandedOption = expandedFmriTrialSigStimPhases
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriTrialSigScopeTrialTypes:
		m.expandedOption = expandedFmriTrialSigScopeTrialTypes
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriTrialSigGroupColumn:
		if len(m.fmriDiscoveredColumns) > 0 {
			m.expandedOption = expandedFmriTrialSigGroupColumn
		} else {
			m.startTextEdit(textFieldFmriTrialSigGroupColumn)
		}
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optFmriTrialSigGroupValues:
		if strings.TrimSpace(m.fmriTrialSigGroupColumn) == "" {
			m.ShowToast("Select a group column first", "warning")
			return
		}
		if vals := m.GetFmriDiscoveredColumnValues(m.fmriTrialSigGroupColumn); len(vals) > 0 {
			m.expandedOption = expandedFmriTrialSigGroupValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldFmriTrialSigGroupValues)
		}
		m.useDefaultAdvanced = false
	case optFmriTrialSigGroupScope:
		m.fmriTrialSigGroupScopeIndex = (m.fmriTrialSigGroupScopeIndex + 1) % 2
		m.useDefaultAdvanced = false

	// Trial-wise signatures
	case optFmriTrialSigMethod:
		m.fmriTrialSigMethodIndex = (m.fmriTrialSigMethodIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriTrialSigIncludeOtherEvents:
		m.fmriTrialSigIncludeOtherEvents = !m.fmriTrialSigIncludeOtherEvents
		m.useDefaultAdvanced = false
	case optFmriTrialSigMaxTrialsPerRun:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriTrialSigFixedEffectsWeighting:
		m.fmriTrialSigFixedEffectsWeighting = (m.fmriTrialSigFixedEffectsWeighting + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriTrialSigWriteTrialBetas:
		m.fmriTrialSigWriteTrialBetas = !m.fmriTrialSigWriteTrialBetas
		m.useDefaultAdvanced = false
	case optFmriTrialSigWriteTrialVariances:
		m.fmriTrialSigWriteTrialVariances = !m.fmriTrialSigWriteTrialVariances
		m.useDefaultAdvanced = false
	case optFmriTrialSigWriteConditionBetas:
		m.fmriTrialSigWriteConditionBetas = !m.fmriTrialSigWriteConditionBetas
		m.useDefaultAdvanced = false
	case optFmriTrialSigSignatureOption1:
		next := !m.fmriTrialSigSignatureOption1
		if !next && !m.fmriTrialSigSignatureOption2 {
			m.ShowToast("Select at least one signature", "warning")
			return
		}
		m.fmriTrialSigSignatureOption1 = next
		m.useDefaultAdvanced = false
	case optFmriTrialSigSignatureOption2:
		next := !m.fmriTrialSigSignatureOption2
		if !next && !m.fmriTrialSigSignatureOption1 {
			m.ShowToast("Select at least one signature", "warning")
			return
		}
		m.fmriTrialSigSignatureOption2 = next
		m.useDefaultAdvanced = false
	case optFmriTrialSigLssOtherRegressors:
		m.fmriTrialSigLssOtherRegressorsIndex = (m.fmriTrialSigLssOtherRegressorsIndex + 1) % 2
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes
	options = m.getFmriAnalysisOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}
