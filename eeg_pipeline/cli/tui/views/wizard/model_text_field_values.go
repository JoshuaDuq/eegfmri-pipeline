package wizard

import "strings"

// Central text-field getter/setter mappings for wizard config editing.

func (m Model) getTextFieldValue(field textField) string {
	switch field {
	case textFieldTask:
		return m.task
	case textFieldBidsRoot:
		return m.bidsRoot
	case textFieldBidsFmriRoot:
		return m.bidsFmriRoot
	case textFieldDerivRoot:
		return m.derivRoot
	case textFieldSourceRoot:
		return m.sourceRoot
	case textFieldConfigSetOverrides:
		return m.configSetOverrides
	case textFieldFmriFmriprepImage:
		return m.fmriFmriprepImage
	case textFieldFmriFmriprepOutputDir:
		return m.fmriFmriprepOutputDir
	case textFieldFmriFmriprepWorkDir:
		return m.fmriFmriprepWorkDir
	case textFieldFmriFreesurferLicenseFile:
		return m.fmriFreesurferLicenseFile
	case textFieldFmriFreesurferSubjectsDir:
		return m.fmriFreesurferSubjectsDir
	case textFieldFmriOutputSpaces:
		return m.fmriOutputSpacesSpec
	case textFieldFmriIgnore:
		return m.fmriIgnoreSpec
	case textFieldFmriBidsFilterFile:
		return m.fmriBidsFilterFile
	case textFieldFmriExtraArgs:
		return m.fmriExtraArgs
	case textFieldFmriSkullStripTemplate:
		return m.fmriSkullStripTemplate
	case textFieldFmriTaskId:
		return m.fmriTaskId
	case textFieldFmriAnalysisFmriprepSpace:
		return m.fmriAnalysisFmriprepSpace
	case textFieldFmriAnalysisRuns:
		return m.fmriAnalysisRunsSpec
	case textFieldFmriAnalysisCondAColumn:
		return m.fmriAnalysisCondAColumn
	case textFieldFmriAnalysisCondAValue:
		return m.fmriAnalysisCondAValue
	case textFieldFmriAnalysisCondBColumn:
		return m.fmriAnalysisCondBColumn
	case textFieldFmriAnalysisCondBValue:
		return m.fmriAnalysisCondBValue
	case textFieldFmriAnalysisContrastName:
		return m.fmriAnalysisContrastName
	case textFieldFmriAnalysisFormula:
		return m.fmriAnalysisFormula
	case textFieldFmriAnalysisEventsToModel:
		return m.fmriAnalysisEventsToModel
	case textFieldFmriAnalysisEventsToModelColumn:
		return m.fmriAnalysisEventsToModelColumn
	case textFieldFmriAnalysisScopeColumn:
		return m.fmriAnalysisScopeColumn
	case textFieldFmriAnalysisScopeTrialTypes:
		return m.fmriAnalysisScopeTrialTypes
	case textFieldFmriAnalysisPhaseColumn:
		return m.fmriAnalysisPhaseColumn
	case textFieldFmriAnalysisPhaseScopeColumn:
		return m.fmriAnalysisPhaseScopeColumn
	case textFieldFmriAnalysisPhaseScopeValue:
		return m.fmriAnalysisPhaseScopeValue
	case textFieldFmriAnalysisStimPhasesToModel:
		return m.fmriAnalysisStimPhasesToModel
	case textFieldFmriAnalysisOutputDir:
		return m.fmriAnalysisOutputDir
	case textFieldFmriAnalysisFreesurferDir:
		return m.fmriAnalysisFreesurferDir
	case textFieldFmriAnalysisSignatureDir:
		return m.fmriAnalysisSignatureDir
	case textFieldFmriAnalysisSignatureMaps:
		return m.fmriAnalysisSignatureMaps
	case textFieldFmriSecondLevelInputRoot:
		return m.fmriSecondLevelInputRoot
	case textFieldFmriSecondLevelContrastNames:
		return m.fmriSecondLevelContrastNames
	case textFieldFmriSecondLevelConditionLabels:
		return m.fmriSecondLevelConditionLabels
	case textFieldFmriSecondLevelCovariatesFile:
		return m.fmriSecondLevelCovariatesFile
	case textFieldFmriSecondLevelSubjectColumn:
		return m.fmriSecondLevelSubjectColumn
	case textFieldFmriSecondLevelCovariateColumns:
		return m.fmriSecondLevelCovariateColumns
	case textFieldFmriSecondLevelGroupColumn:
		return m.fmriSecondLevelGroupColumn
	case textFieldFmriSecondLevelGroupAValue:
		return m.fmriSecondLevelGroupAValue
	case textFieldFmriSecondLevelGroupBValue:
		return m.fmriSecondLevelGroupBValue
	case textFieldFmriSecondLevelFormula:
		return m.fmriSecondLevelFormula
	case textFieldFmriSecondLevelOutputName:
		return m.fmriSecondLevelOutputName
	case textFieldFmriSecondLevelOutputDir:
		return m.fmriSecondLevelOutputDir
	case textFieldFmriTrialSigGroupColumn:
		return m.fmriTrialSigGroupColumn
	case textFieldFmriTrialSigGroupValues:
		return m.fmriTrialSigGroupValuesSpec
	case textFieldFmriTrialSigScopeTrialTypeColumn:
		return m.fmriTrialSigScopeTrialTypeColumn
	case textFieldFmriTrialSigScopePhaseColumn:
		return m.fmriTrialSigScopePhaseColumn
	case textFieldFmriTrialSigScopeTrialTypes:
		return m.fmriTrialSigScopeTrialTypes
	case textFieldFmriTrialSigScopeStimPhases:
		return m.fmriTrialSigScopeStimPhases
	case textFieldPrepMontage:
		return m.prepMontage
	case textFieldPrepChTypes:
		return m.prepChTypes
	case textFieldPrepEegReference:
		return m.prepEegReference
	case textFieldPrepEogChannels:
		return m.prepEogChannels
	case textFieldPrepConditions:
		return m.prepConditions
	case textFieldPrepFileExtension:
		return m.prepFileExtension
	case textFieldPrepRenameAnotDict:
		return m.prepRenameAnotDict
	case textFieldPrepCustomBadDict:
		return m.prepCustomBadDict
	case textFieldConditionCompareColumn:
		return m.conditionCompareColumn
	case textFieldConditionCompareValues:
		return m.conditionCompareValues
	case textFieldConditionCompareLabels:
		return m.conditionCompareLabels
	case textFieldTemporalConditionColumn:
		return m.temporalConditionColumn
	case textFieldTemporalConditionValues:
		return m.temporalConditionValues
	case textFieldTemporalTargetColumn:
		return m.temporalTargetColumn
	case textFieldTfHeatmapFreqs:
		return m.tfHeatmapFreqsSpec
	case textFieldRunAdjustmentColumn:
		return m.runAdjustmentColumn
	case textFieldBehaviorOutcomeColumn:
		return m.behaviorOutcomeColumn
	case textFieldBehaviorPredictorColumn:
		return m.behaviorPredictorColumn
	case textFieldPredictorResidualCrossfitGroupColumn:
		return m.predictorResidualCrossfitGroupColumn
	case textFieldPredictorResidualSplineDfCandidates:
		return m.predictorResidualSplineDfCandidates
	case textFieldClusterConditionColumn:
		return m.clusterConditionColumn
	case textFieldClusterConditionValues:
		return m.clusterConditionValues
	case textFieldCorrelationsTargetColumn:
		return m.correlationsTargetColumn
	case textFieldCorrelationsPowerSegment:
		return m.correlationsPowerSegment
	case textFieldGroupLevelTarget:
		return m.groupLevelTarget
	case textFieldCorrelationsTypes:
		return m.correlationsTypesSpec
	case textFieldCorrelationsFeatures:
		return m.correlationsFeaturesSpec
	case textFieldConditionFeatures:
		return m.conditionFeaturesSpec
	case textFieldTemporalFeatures:
		return m.temporalFeaturesSpec
	case textFieldClusterFeatures:
		return m.clusterFeaturesSpec
	case textFieldItpcConditionColumn:
		return m.itpcConditionColumn
	case textFieldItpcConditionValues:
		return m.itpcConditionValues
	case textFieldConnConditionColumn:
		return m.connConditionColumn
	case textFieldConnConditionValues:
		return m.connConditionValues
	case textFieldPACPairs:
		return m.pacPairsSpec
	case textFieldBurstBands:
		return m.burstBandsSpec
	case textFieldMicrostatesFixedTemplatesPath:
		return m.microstatesFixedTemplatesPath
	case textFieldSpectralRatioPairs:
		return m.spectralRatioPairsSpec
	case textFieldSpectralSegments:
		return m.spectralSegmentsSpec
	case textFieldAsymmetryChannelPairs:
		return m.asymmetryChannelPairsSpec
	case textFieldAsymmetryActivationBands:
		return m.asymmetryActivationBandsSpec
	case textFieldIAFRois:
		return m.iafRoisSpec
	case textFieldERPComponents:
		return m.erpComponentsSpec
	case textFieldSourceLocSubject:
		return m.sourceLocSubject
	case textFieldSourceLocSubjectsDir:
		return m.sourceLocSubjectsDir
	case textFieldSourceLocTrans:
		return m.sourceLocTrans
	case textFieldSourceLocBem:
		return m.sourceLocBem
	case textFieldSourceLocFmriStatsMap:
		return m.sourceLocFmriStatsMap
	case textFieldSourceLocContrastConditionColumn:
		return m.sourceLocContrastCondition
	case textFieldSourceLocContrastConditionA:
		return m.sourceLocContrastA
	case textFieldSourceLocContrastConditionB:
		return m.sourceLocContrastB
	case textFieldSourceLocFmriCondAColumn:
		return m.sourceLocFmriCondAColumn
	case textFieldSourceLocFmriCondAValue:
		return m.sourceLocFmriCondAValue
	case textFieldSourceLocFmriCondBColumn:
		return m.sourceLocFmriCondBColumn
	case textFieldSourceLocFmriCondBValue:
		return m.sourceLocFmriCondBValue
	case textFieldSourceLocFmriContrastFormula:
		return m.sourceLocFmriContrastFormula
	case textFieldSourceLocFmriContrastName:
		return m.sourceLocFmriContrastName
	case textFieldSourceLocFmriRunsToInclude:
		return m.sourceLocFmriRunsToInclude
	case textFieldSourceLocFmriEventsToModel:
		return m.sourceLocFmriEventsToModel
	case textFieldSourceLocFmriEventsToModelColumn:
		return m.sourceLocFmriEventsToModelColumn
	case textFieldSourceLocFmriConditionScopeColumn:
		return m.sourceLocFmriConditionScopeColumn
	case textFieldSourceLocFmriConditionScopeTrialTypes:
		return m.sourceLocFmriConditionScopeTrialTypes
	case textFieldSourceLocFmriPhaseColumn:
		return m.sourceLocFmriPhaseColumn
	case textFieldSourceLocFmriPhaseScopeColumn:
		return m.sourceLocFmriPhaseScopeColumn
	case textFieldSourceLocFmriPhaseScopeValue:
		return m.sourceLocFmriPhaseScopeValue
	case textFieldSourceLocFmriStimPhasesToModel:
		return m.sourceLocFmriStimPhasesToModel
	case textFieldPlotBboxInches:
		return m.plotBboxInches
	case textFieldPlotFontFamily:
		return m.plotFontFamily
	case textFieldPlotFontWeight:
		return m.plotFontWeight
	case textFieldPlotLayoutTightRect:
		return m.plotLayoutTightRectSpec
	case textFieldPlotLayoutTightRectMicrostate:
		return m.plotLayoutTightRectMicrostateSpec
	case textFieldPlotGridSpecWidthRatios:
		return m.plotGridSpecWidthRatiosSpec
	case textFieldPlotGridSpecHeightRatios:
		return m.plotGridSpecHeightRatiosSpec
	case textFieldPlotFigureSizeStandard:
		return m.plotFigureSizeStandardSpec
	case textFieldPlotFigureSizeMedium:
		return m.plotFigureSizeMediumSpec
	case textFieldPlotFigureSizeSmall:
		return m.plotFigureSizeSmallSpec
	case textFieldPlotFigureSizeSquare:
		return m.plotFigureSizeSquareSpec
	case textFieldPlotFigureSizeWide:
		return m.plotFigureSizeWideSpec
	case textFieldPlotFigureSizeTFR:
		return m.plotFigureSizeTFRSpec
	case textFieldPlotFigureSizeTopomap:
		return m.plotFigureSizeTopomapSpec
	case textFieldPlotColorCondB:
		return m.plotColorCondB
	case textFieldPlotColorCondA:
		return m.plotColorCondA
	case textFieldPlotColorSignificant:
		return m.plotColorSignificant
	case textFieldPlotColorNonsignificant:
		return m.plotColorNonsignificant
	case textFieldPlotColorGray:
		return m.plotColorGray
	case textFieldPlotColorLightGray:
		return m.plotColorLightGray
	case textFieldPlotColorBlack:
		return m.plotColorBlack
	case textFieldPlotColorBlue:
		return m.plotColorBlue
	case textFieldPlotColorRed:
		return m.plotColorRed
	case textFieldPlotColorNetworkNode:
		return m.plotColorNetworkNode
	case textFieldPlotScatterEdgecolor:
		return m.plotScatterEdgeColor
	case textFieldPlotHistEdgecolor:
		return m.plotHistEdgeColor
	case textFieldPlotKdeColor:
		return m.plotKdeColor
	case textFieldPlotTopomapColormap:
		return m.plotTopomapColormap
	case textFieldPlotTopomapSigMaskMarker:
		return m.plotTopomapSigMaskMarker
	case textFieldPlotTopomapSigMaskMarkerFaceColor:
		return m.plotTopomapSigMaskMarkerFaceColor
	case textFieldPlotTopomapSigMaskMarkerEdgeColor:
		return m.plotTopomapSigMaskMarkerEdgeColor
	case textFieldPlotTfrDefaultBaselineWindow:
		return m.plotTfrDefaultBaselineWindowSpec
	case textFieldPlotPacCmap:
		return m.plotPacCmap
	case textFieldPlotPacPairs:
		return m.plotPacPairsSpec
	case textFieldPlotConnectivityMeasures:
		return m.plotConnectivityMeasuresSpec
	case textFieldPlotSpectralMetrics:
		return m.plotSpectralMetricsSpec
	case textFieldPlotBurstsMetrics:
		return m.plotBurstsMetricsSpec
	case textFieldPlotTemporalTimeBins:
		return m.plotTemporalTimeBinsSpec
	case textFieldPlotTemporalTimeLabels:
		return m.plotTemporalTimeLabelsSpec
	case textFieldPlotAsymmetryStat:
		return m.plotAsymmetryStatSpec
	case textFieldPlotSourceHemi:
		return m.plotSourceHemi
	case textFieldPlotSourceViews:
		return m.plotSourceViews
	case textFieldPlotSourceCortex:
		return m.plotSourceCortex
	case textFieldPlotSourceSubjectsDir:
		return m.plotSourceSubjectsDir
	case textFieldPlotComparisonWindows:
		return m.plotComparisonWindowsSpec
	case textFieldPlotComparisonSegment:
		return m.plotComparisonSegment
	case textFieldPlotComparisonColumn:
		return m.plotComparisonColumn
	case textFieldPlotComparisonValues:
		return m.plotComparisonValuesSpec
	case textFieldPlotComparisonLabels:
		return m.plotComparisonLabelsSpec
	case textFieldPlotComparisonROIs:
		return m.plotComparisonROIsSpec
	// Machine Learning advanced config text fields
	case textFieldMLTarget:
		return m.mlTarget
	case textFieldMLFmriSigContrastName:
		return m.mlFmriSigContrastName
	case textFieldMLFeatureFamilies:
		return m.mlFeatureFamiliesSpec
	case textFieldMLFeatureBands:
		return m.mlFeatureBandsSpec
	case textFieldMLFeatureSegments:
		return m.mlFeatureSegmentsSpec
	case textFieldMLFeatureScopes:
		return m.mlFeatureScopesSpec
	case textFieldMLFeatureStats:
		return m.mlFeatureStatsSpec
	case textFieldMLCovariates:
		return m.mlCovariatesSpec
	case textFieldMLSpatialRegionsAllowed:
		return m.mlSpatialRegionsAllowed
	case textFieldMLBaselinePredictors:
		return m.mlBaselinePredictorsSpec
	case textFieldMLPlotFormats:
		return m.mlPlotFormatsSpec
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		return m.elasticNetAlphaGrid
	case textFieldElasticNetL1RatioGrid:
		return m.elasticNetL1RatioGrid
	case textFieldRidgeAlphaGrid:
		return m.ridgeAlphaGrid
	case textFieldRfMaxDepthGrid:
		return m.rfMaxDepthGrid
	case textFieldVarianceThresholdGrid:
		return m.varianceThresholdGrid
	// ML new text fields
	case textFieldMLSvmCGrid:
		return m.mlSvmCGrid
	case textFieldMLSvmGammaGrid:
		return m.mlSvmGammaGrid
	case textFieldMLLrCGrid:
		return m.mlLrCGrid
	case textFieldMLLrL1RatioGrid:
		return m.mlLrL1RatioGrid
	case textFieldMLRfMinSamplesSplitGrid:
		return m.mlRfMinSamplesSplitGrid
	case textFieldMLRfMinSamplesLeafGrid:
		return m.mlRfMinSamplesLeafGrid
	// EEG Preprocessing new text fields
	case textFieldPrepEcgChannels:
		return m.prepEcgChannels
	case textFieldPrepAutorejectNInterpolate:
		return m.prepAutorejectNInterpolate
	case textFieldPrepCleanEventsQCEcgVarianceOutputColumn:
		return m.prepCleanEventsQCEcgVarianceOutputColumn
	case textFieldPrepCleanEventsQCEcgVarianceChannels:
		return m.prepCleanEventsQCEcgVarianceChannels
	case textFieldPrepCleanEventsQCEcgVarianceWindow:
		return m.prepCleanEventsQCEcgVarianceWindow
	case textFieldPrepCleanEventsQCPeripheralLowGammaOutputColumn:
		return m.prepCleanEventsQCPeripheralLowGammaOutputColumn
	case textFieldPrepCleanEventsQCPeripheralLowGammaChannels:
		return m.prepCleanEventsQCPeripheralLowGammaChannels
	case textFieldPrepCleanEventsQCPeripheralLowGammaBand:
		return m.prepCleanEventsQCPeripheralLowGammaBand
	case textFieldPrepCleanEventsQCPeripheralLowGammaWindow:
		return m.prepCleanEventsQCPeripheralLowGammaWindow
	// Event Column Mapping text fields
	case textFieldEventColPredictor:
		return m.eventColPredictor
	case textFieldEventColOutcome:
		return m.eventColOutcome
	case textFieldEventColBinaryOutcome:
		return m.eventColBinaryOutcome
	case textFieldEventColCondition:
		return m.eventColCondition
	case textFieldEventColRequired:
		return m.eventColRequired
	case textFieldConditionPreferredPrefixes:
		return m.conditionPreferredPrefixes
	// Change Scores text fields
	case textFieldChangeScoresWindowPairs:
		return m.changeScoresWindowPairs
	// ERDS Condition Markers text fields
	case textFieldERDSConditionMarkerBands:
		return m.erdsConditionMarkerBands
	case textFieldERDSLateralityColumns:
		return m.erdsLateralityColumns
	case textFieldERDSSomatosensoryLeftChannels:
		return m.erdsSomatosensoryLeftChannels
	case textFieldERDSSomatosensoryRightChannels:
		return m.erdsSomatosensoryRightChannels
	// Behavior Statistics text fields
	case textFieldBehaviorPermGroupColumnPreference:
		return m.behaviorPermGroupColumnPreference
	case textFieldICCUnitColumns:
		return m.iccUnitColumns
	case textFieldBehaviorFeatureRegistryFilesJSON:
		return m.behaviorFeatureRegistryFilesJSON
	case textFieldBehaviorFeatureRegistrySourceToTypeJSON:
		return m.behaviorFeatureRegistrySourceJSON
	case textFieldBehaviorFeatureRegistryTypeHierarchyJSON:
		return m.behaviorFeatureRegistryHierarchyJSON
	case textFieldBehaviorFeatureRegistryPatternsJSON:
		return m.behaviorFeatureRegistryPatternsJSON
	case textFieldBehaviorFeatureRegistryClassifiersJSON:
		return m.behaviorFeatureRegistryClassifiersJSON
	// Preprocessing text fields
	case textFieldIcaLabelsToKeep:
		return m.icaLabelsToKeep
	default:
		return ""
	}
}

func (m *Model) setTextFieldValue(field textField, value string) {
	value = strings.TrimSpace(value)
	switch field {
	case textFieldTask:
		m.task = value
	case textFieldBidsRoot:
		m.bidsRoot = value
	case textFieldBidsFmriRoot:
		m.bidsFmriRoot = value
	case textFieldDerivRoot:
		m.derivRoot = value
	case textFieldSourceRoot:
		m.sourceRoot = value
	case textFieldConfigSetOverrides:
		m.configSetOverrides = value
	case textFieldFmriFmriprepImage:
		m.fmriFmriprepImage = value
	case textFieldFmriFmriprepOutputDir:
		m.fmriFmriprepOutputDir = value
	case textFieldFmriFmriprepWorkDir:
		m.fmriFmriprepWorkDir = value
	case textFieldFmriFreesurferLicenseFile:
		m.fmriFreesurferLicenseFile = value
	case textFieldFmriFreesurferSubjectsDir:
		m.fmriFreesurferSubjectsDir = value
	case textFieldFmriOutputSpaces:
		m.fmriOutputSpacesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriIgnore:
		m.fmriIgnoreSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriBidsFilterFile:
		m.fmriBidsFilterFile = value
	case textFieldFmriExtraArgs:
		m.fmriExtraArgs = value
	case textFieldFmriSkullStripTemplate:
		m.fmriSkullStripTemplate = strings.TrimSpace(value)
	case textFieldFmriTaskId:
		m.fmriTaskId = strings.TrimSpace(value)
	case textFieldFmriAnalysisFmriprepSpace:
		m.fmriAnalysisFmriprepSpace = strings.TrimSpace(value)
	case textFieldFmriAnalysisRuns:
		m.fmriAnalysisRunsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriAnalysisCondAColumn:
		m.fmriAnalysisCondAColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondAValue:
		m.fmriAnalysisCondAValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondBColumn:
		m.fmriAnalysisCondBColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondBValue:
		m.fmriAnalysisCondBValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisContrastName:
		m.fmriAnalysisContrastName = strings.TrimSpace(value)
	case textFieldFmriAnalysisFormula:
		m.fmriAnalysisFormula = strings.TrimSpace(value)
	case textFieldFmriAnalysisEventsToModel:
		m.fmriAnalysisEventsToModel = strings.TrimSpace(value)
	case textFieldFmriAnalysisEventsToModelColumn:
		m.fmriAnalysisEventsToModelColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisScopeColumn:
		m.fmriAnalysisScopeColumn = strings.TrimSpace(value)
		m.fmriAnalysisScopeTrialTypes = ""
	case textFieldFmriAnalysisScopeTrialTypes:
		m.fmriAnalysisScopeTrialTypes = strings.Join(strings.Fields(value), " ")
	case textFieldFmriAnalysisPhaseColumn:
		m.fmriAnalysisPhaseColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisPhaseScopeColumn:
		m.fmriAnalysisPhaseScopeColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisPhaseScopeValue:
		m.fmriAnalysisPhaseScopeValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisStimPhasesToModel:
		m.fmriAnalysisStimPhasesToModel = strings.TrimSpace(value)
	case textFieldFmriAnalysisOutputDir:
		m.fmriAnalysisOutputDir = strings.TrimSpace(value)
	case textFieldFmriAnalysisFreesurferDir:
		m.fmriAnalysisFreesurferDir = strings.TrimSpace(value)
	case textFieldFmriAnalysisSignatureDir:
		m.fmriAnalysisSignatureDir = strings.TrimSpace(value)
	case textFieldFmriAnalysisSignatureMaps:
		m.fmriAnalysisSignatureMaps = strings.TrimSpace(value)
	case textFieldFmriSecondLevelInputRoot:
		m.fmriSecondLevelInputRoot = strings.TrimSpace(value)
		m.fmriSecondLevelContrastDiscoveryKey = ""
	case textFieldFmriSecondLevelContrastNames:
		m.fmriSecondLevelContrastNames = strings.Join(strings.Fields(value), " ")
	case textFieldFmriSecondLevelConditionLabels:
		m.fmriSecondLevelConditionLabels = strings.Join(strings.Fields(value), " ")
	case textFieldFmriSecondLevelCovariatesFile:
		m.fmriSecondLevelCovariatesFile = strings.TrimSpace(value)
		m.fmriSecondLevelCovariatesDiscoveryKey = ""
	case textFieldFmriSecondLevelSubjectColumn:
		m.fmriSecondLevelSubjectColumn = strings.TrimSpace(value)
	case textFieldFmriSecondLevelCovariateColumns:
		m.fmriSecondLevelCovariateColumns = strings.Join(strings.Fields(value), " ")
	case textFieldFmriSecondLevelGroupColumn:
		m.fmriSecondLevelGroupColumn = strings.TrimSpace(value)
		m.fmriSecondLevelGroupAValue = ""
		m.fmriSecondLevelGroupBValue = ""
	case textFieldFmriSecondLevelGroupAValue:
		m.fmriSecondLevelGroupAValue = strings.TrimSpace(value)
	case textFieldFmriSecondLevelGroupBValue:
		m.fmriSecondLevelGroupBValue = strings.TrimSpace(value)
	case textFieldFmriSecondLevelFormula:
		m.fmriSecondLevelFormula = strings.TrimSpace(value)
	case textFieldFmriSecondLevelOutputName:
		m.fmriSecondLevelOutputName = strings.TrimSpace(value)
	case textFieldFmriSecondLevelOutputDir:
		m.fmriSecondLevelOutputDir = strings.TrimSpace(value)
	case textFieldFmriTrialSigGroupColumn:
		m.fmriTrialSigGroupColumn = strings.TrimSpace(value)
		m.fmriTrialSigGroupValuesSpec = "" // Reset values when column changes
	case textFieldFmriTrialSigGroupValues:
		m.fmriTrialSigGroupValuesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriTrialSigScopeTrialTypeColumn:
		m.fmriTrialSigScopeTrialTypeColumn = strings.TrimSpace(value)
	case textFieldFmriTrialSigScopePhaseColumn:
		m.fmriTrialSigScopePhaseColumn = strings.TrimSpace(value)
	case textFieldFmriTrialSigScopeTrialTypes:
		m.fmriTrialSigScopeTrialTypes = strings.Join(strings.Fields(value), " ")
	case textFieldFmriTrialSigScopeStimPhases:
		m.fmriTrialSigScopeStimPhases = strings.TrimSpace(value)
	case textFieldPrepMontage:
		m.prepMontage = value
	case textFieldPrepChTypes:
		m.prepChTypes = value
	case textFieldPrepEegReference:
		m.prepEegReference = value
	case textFieldPrepEogChannels:
		m.prepEogChannels = value
	case textFieldPrepConditions:
		m.prepConditions = value
	case textFieldPrepFileExtension:
		m.prepFileExtension = strings.TrimSpace(value)
	case textFieldPrepRenameAnotDict:
		m.prepRenameAnotDict = value
	case textFieldPrepCustomBadDict:
		m.prepCustomBadDict = value
	case textFieldConditionCompareColumn:
		m.conditionCompareColumn = strings.TrimSpace(value)
	case textFieldConditionCompareValues:
		m.conditionCompareValues = strings.TrimSpace(value)
	case textFieldConditionCompareLabels:
		m.conditionCompareLabels = strings.TrimSpace(value)
	case textFieldTemporalConditionColumn:
		m.temporalConditionColumn = strings.TrimSpace(value)
	case textFieldTemporalConditionValues:
		m.temporalConditionValues = strings.TrimSpace(value)
	case textFieldTemporalTargetColumn:
		m.temporalTargetColumn = strings.TrimSpace(value)
	case textFieldTfHeatmapFreqs:
		m.tfHeatmapFreqsSpec = strings.Join(strings.Fields(value), "")
	case textFieldRunAdjustmentColumn:
		m.runAdjustmentColumn = strings.TrimSpace(value)
	case textFieldBehaviorOutcomeColumn:
		m.behaviorOutcomeColumn = strings.TrimSpace(value)
	case textFieldBehaviorPredictorColumn:
		m.behaviorPredictorColumn = strings.TrimSpace(value)
	case textFieldPredictorResidualCrossfitGroupColumn:
		m.predictorResidualCrossfitGroupColumn = strings.TrimSpace(value)
	case textFieldPredictorResidualSplineDfCandidates:
		m.predictorResidualSplineDfCandidates = strings.Join(strings.Fields(value), "")
	case textFieldClusterConditionColumn:
		m.clusterConditionColumn = strings.TrimSpace(value)
	case textFieldClusterConditionValues:
		m.clusterConditionValues = strings.TrimSpace(value)
	case textFieldCorrelationsTargetColumn:
		m.correlationsTargetColumn = strings.TrimSpace(value)
	case textFieldCorrelationsPowerSegment:
		m.correlationsPowerSegment = strings.TrimSpace(value)
	case textFieldGroupLevelTarget:
		m.groupLevelTarget = strings.TrimSpace(value)
	case textFieldCorrelationsTypes:
		m.correlationsTypesSpec = strings.Join(strings.Fields(value), "")
	case textFieldCorrelationsFeatures:
		m.correlationsFeaturesSpec = strings.Join(strings.Fields(value), "")
	case textFieldConditionFeatures:
		m.conditionFeaturesSpec = strings.Join(strings.Fields(value), "")
	case textFieldTemporalFeatures:
		m.temporalFeaturesSpec = strings.Join(strings.Fields(value), "")
	case textFieldClusterFeatures:
		m.clusterFeaturesSpec = strings.Join(strings.Fields(value), "")
	case textFieldItpcConditionColumn:
		m.itpcConditionColumn = strings.TrimSpace(value)
	case textFieldItpcConditionValues:
		m.itpcConditionValues = strings.TrimSpace(value)
	case textFieldConnConditionColumn:
		m.connConditionColumn = strings.TrimSpace(value)
	case textFieldConnConditionValues:
		m.connConditionValues = strings.TrimSpace(value)
	case textFieldPACPairs:
		m.pacPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldBurstBands:
		m.burstBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldMicrostatesFixedTemplatesPath:
		m.microstatesFixedTemplatesPath = strings.TrimSpace(value)
	case textFieldSpectralRatioPairs:
		m.spectralRatioPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSpectralSegments:
		m.spectralSegmentsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldAsymmetryChannelPairs:
		m.asymmetryChannelPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldAsymmetryActivationBands:
		m.asymmetryActivationBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldIAFRois:
		m.iafRoisSpec = strings.Join(strings.Fields(value), "")
	case textFieldERPComponents:
		m.erpComponentsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSourceLocSubject:
		m.sourceLocSubject = value
	case textFieldSourceLocSubjectsDir:
		m.sourceLocSubjectsDir = value
	case textFieldSourceLocTrans:
		m.sourceLocTrans = value
	case textFieldSourceLocBem:
		m.sourceLocBem = value
	case textFieldSourceLocFmriStatsMap:
		m.sourceLocFmriStatsMap = value
	case textFieldSourceLocContrastConditionColumn:
		m.sourceLocContrastCondition = strings.TrimSpace(value)
	case textFieldSourceLocContrastConditionA:
		m.sourceLocContrastA = strings.TrimSpace(value)
	case textFieldSourceLocContrastConditionB:
		m.sourceLocContrastB = strings.TrimSpace(value)
	case textFieldSourceLocFmriCondAColumn:
		m.sourceLocFmriCondAColumn = value
	case textFieldSourceLocFmriCondAValue:
		m.sourceLocFmriCondAValue = value
	case textFieldSourceLocFmriCondBColumn:
		m.sourceLocFmriCondBColumn = value
	case textFieldSourceLocFmriCondBValue:
		m.sourceLocFmriCondBValue = value
	case textFieldSourceLocFmriContrastFormula:
		m.sourceLocFmriContrastFormula = value
	case textFieldSourceLocFmriContrastName:
		m.sourceLocFmriContrastName = value
	case textFieldSourceLocFmriRunsToInclude:
		m.sourceLocFmriRunsToInclude = value
	case textFieldSourceLocFmriEventsToModel:
		m.sourceLocFmriEventsToModel = strings.Join(strings.Fields(value), " ")
	case textFieldSourceLocFmriEventsToModelColumn:
		m.sourceLocFmriEventsToModelColumn = strings.TrimSpace(value)
	case textFieldSourceLocFmriConditionScopeColumn:
		m.sourceLocFmriConditionScopeColumn = strings.TrimSpace(value)
		m.sourceLocFmriConditionScopeTrialTypes = ""
	case textFieldSourceLocFmriConditionScopeTrialTypes:
		m.sourceLocFmriConditionScopeTrialTypes = strings.Join(strings.Fields(value), " ")
	case textFieldSourceLocFmriPhaseColumn:
		m.sourceLocFmriPhaseColumn = strings.TrimSpace(value)
		m.sourceLocFmriStimPhasesToModel = ""
	case textFieldSourceLocFmriPhaseScopeColumn:
		m.sourceLocFmriPhaseScopeColumn = strings.TrimSpace(value)
		m.sourceLocFmriPhaseScopeValue = ""
	case textFieldSourceLocFmriPhaseScopeValue:
		m.sourceLocFmriPhaseScopeValue = strings.TrimSpace(value)
	case textFieldSourceLocFmriStimPhasesToModel:
		m.sourceLocFmriStimPhasesToModel = strings.TrimSpace(value)
	case textFieldPlotBboxInches:
		m.plotBboxInches = value
	case textFieldPlotFontFamily:
		m.plotFontFamily = value
	case textFieldPlotFontWeight:
		m.plotFontWeight = value
	case textFieldPlotLayoutTightRect:
		m.plotLayoutTightRectSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotLayoutTightRectMicrostate:
		m.plotLayoutTightRectMicrostateSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotGridSpecWidthRatios:
		m.plotGridSpecWidthRatiosSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotGridSpecHeightRatios:
		m.plotGridSpecHeightRatiosSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeStandard:
		m.plotFigureSizeStandardSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeMedium:
		m.plotFigureSizeMediumSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeSmall:
		m.plotFigureSizeSmallSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeSquare:
		m.plotFigureSizeSquareSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeWide:
		m.plotFigureSizeWideSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeTFR:
		m.plotFigureSizeTFRSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeTopomap:
		m.plotFigureSizeTopomapSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotColorCondB:
		m.plotColorCondB = value
	case textFieldPlotColorCondA:
		m.plotColorCondA = value
	case textFieldPlotColorSignificant:
		m.plotColorSignificant = value
	case textFieldPlotColorNonsignificant:
		m.plotColorNonsignificant = value
	case textFieldPlotColorGray:
		m.plotColorGray = value
	case textFieldPlotColorLightGray:
		m.plotColorLightGray = value
	case textFieldPlotColorBlack:
		m.plotColorBlack = value
	case textFieldPlotColorBlue:
		m.plotColorBlue = value
	case textFieldPlotColorRed:
		m.plotColorRed = value
	case textFieldPlotColorNetworkNode:
		m.plotColorNetworkNode = value
	case textFieldPlotScatterEdgecolor:
		m.plotScatterEdgeColor = value
	case textFieldPlotHistEdgecolor:
		m.plotHistEdgeColor = value
	case textFieldPlotKdeColor:
		m.plotKdeColor = value
	case textFieldPlotTopomapColormap:
		m.plotTopomapColormap = value
	case textFieldPlotTopomapSigMaskMarker:
		m.plotTopomapSigMaskMarker = value
	case textFieldPlotTopomapSigMaskMarkerFaceColor:
		m.plotTopomapSigMaskMarkerFaceColor = value
	case textFieldPlotTopomapSigMaskMarkerEdgeColor:
		m.plotTopomapSigMaskMarkerEdgeColor = value
	case textFieldPlotTfrDefaultBaselineWindow:
		m.plotTfrDefaultBaselineWindowSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotPacCmap:
		m.plotPacCmap = value
	case textFieldPlotPacPairs:
		m.plotPacPairsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotConnectivityMeasures:
		m.plotConnectivityMeasuresSpec = normalizeConnectivityMeasureSpec(value)
	case textFieldPlotSpectralMetrics:
		m.plotSpectralMetricsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotBurstsMetrics:
		m.plotBurstsMetricsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotTemporalTimeBins:
		m.plotTemporalTimeBinsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotTemporalTimeLabels:
		m.plotTemporalTimeLabelsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotAsymmetryStat:
		m.plotAsymmetryStatSpec = value
	case textFieldPlotSourceHemi:
		m.plotSourceHemi = value
	case textFieldPlotSourceViews:
		m.plotSourceViews = strings.Join(strings.Fields(value), " ")
	case textFieldPlotSourceCortex:
		m.plotSourceCortex = value
	case textFieldPlotSourceSubjectsDir:
		m.plotSourceSubjectsDir = strings.TrimSpace(value)
	case textFieldPlotComparisonWindows:
		m.plotComparisonWindowsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotComparisonSegment:
		m.plotComparisonSegment = strings.TrimSpace(value)
	case textFieldPlotComparisonColumn:
		m.plotComparisonColumn = strings.TrimSpace(value)
	case textFieldPlotComparisonValues:
		m.plotComparisonValuesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotComparisonLabels:
		m.plotComparisonLabelsSpec = strings.TrimSpace(value)
	case textFieldPlotComparisonROIs:
		m.plotComparisonROIsSpec = strings.Join(strings.Fields(value), " ")
	// Machine Learning advanced config text fields
	case textFieldMLTarget:
		m.mlTarget = strings.TrimSpace(value)
	case textFieldMLFmriSigContrastName:
		m.mlFmriSigContrastName = strings.TrimSpace(value)
	case textFieldMLFeatureFamilies:
		m.mlFeatureFamiliesSpec = strings.TrimSpace(value)
	case textFieldMLFeatureBands:
		m.mlFeatureBandsSpec = strings.TrimSpace(value)
	case textFieldMLFeatureSegments:
		m.mlFeatureSegmentsSpec = strings.TrimSpace(value)
	case textFieldMLFeatureScopes:
		m.mlFeatureScopesSpec = strings.TrimSpace(value)
	case textFieldMLFeatureStats:
		m.mlFeatureStatsSpec = strings.TrimSpace(value)
	case textFieldMLCovariates:
		m.mlCovariatesSpec = strings.TrimSpace(value)
	case textFieldMLSpatialRegionsAllowed:
		m.mlSpatialRegionsAllowed = strings.TrimSpace(value)
	case textFieldMLBaselinePredictors:
		m.mlBaselinePredictorsSpec = strings.TrimSpace(value)
	case textFieldMLPlotFormats:
		m.mlPlotFormatsSpec = strings.Join(splitLooseList(value), " ")
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		m.elasticNetAlphaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldElasticNetL1RatioGrid:
		m.elasticNetL1RatioGrid = strings.Join(splitLooseList(value), ",")
	case textFieldRidgeAlphaGrid:
		m.ridgeAlphaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldRfMaxDepthGrid:
		m.rfMaxDepthGrid = strings.Join(splitLooseList(value), ",")
	case textFieldVarianceThresholdGrid:
		m.varianceThresholdGrid = strings.Join(splitLooseList(value), ",")
	// ML new text fields
	case textFieldMLSvmCGrid:
		m.mlSvmCGrid = strings.Join(splitLooseList(value), ",")
	case textFieldMLSvmGammaGrid:
		m.mlSvmGammaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldMLLrCGrid:
		m.mlLrCGrid = strings.Join(splitLooseList(value), ",")
	case textFieldMLLrL1RatioGrid:
		m.mlLrL1RatioGrid = strings.Join(splitLooseList(value), ",")
	case textFieldMLRfMinSamplesSplitGrid:
		m.mlRfMinSamplesSplitGrid = strings.Join(splitLooseList(value), ",")
	case textFieldMLRfMinSamplesLeafGrid:
		m.mlRfMinSamplesLeafGrid = strings.Join(splitLooseList(value), ",")
	// EEG Preprocessing new text fields
	case textFieldPrepEcgChannels:
		m.prepEcgChannels = value
	case textFieldPrepAutorejectNInterpolate:
		m.prepAutorejectNInterpolate = strings.Join(splitLooseList(value), ",")
	case textFieldPrepCleanEventsQCEcgVarianceOutputColumn:
		m.prepCleanEventsQCEcgVarianceOutputColumn = value
	case textFieldPrepCleanEventsQCEcgVarianceChannels:
		m.prepCleanEventsQCEcgVarianceChannels = value
	case textFieldPrepCleanEventsQCEcgVarianceWindow:
		m.prepCleanEventsQCEcgVarianceWindow = value
	case textFieldPrepCleanEventsQCPeripheralLowGammaOutputColumn:
		m.prepCleanEventsQCPeripheralLowGammaOutputColumn = value
	case textFieldPrepCleanEventsQCPeripheralLowGammaChannels:
		m.prepCleanEventsQCPeripheralLowGammaChannels = value
	case textFieldPrepCleanEventsQCPeripheralLowGammaBand:
		m.prepCleanEventsQCPeripheralLowGammaBand = value
	case textFieldPrepCleanEventsQCPeripheralLowGammaWindow:
		m.prepCleanEventsQCPeripheralLowGammaWindow = value
	// Event Column Mapping text fields
	case textFieldEventColPredictor:
		m.eventColPredictor = strings.Join(splitLooseList(value), ",")
	case textFieldEventColOutcome:
		m.eventColOutcome = strings.Join(splitLooseList(value), ",")
	case textFieldEventColBinaryOutcome:
		m.eventColBinaryOutcome = strings.Join(splitLooseList(value), ",")
	case textFieldEventColCondition:
		m.eventColCondition = strings.Join(splitLooseList(value), ",")
	case textFieldEventColRequired:
		m.eventColRequired = strings.Join(splitLooseList(value), ",")
	case textFieldConditionPreferredPrefixes:
		m.conditionPreferredPrefixes = strings.Join(splitLooseList(value), ",")
	// Change Scores text fields
	case textFieldChangeScoresWindowPairs:
		m.changeScoresWindowPairs = strings.TrimSpace(value)
	// ERDS Condition Markers text fields
	case textFieldERDSConditionMarkerBands:
		m.erdsConditionMarkerBands = strings.Join(splitLooseList(value), ",")
	case textFieldERDSLateralityColumns:
		m.erdsLateralityColumns = strings.Join(splitLooseList(value), ",")
	case textFieldERDSSomatosensoryLeftChannels:
		m.erdsSomatosensoryLeftChannels = strings.Join(splitLooseList(value), ",")
	case textFieldERDSSomatosensoryRightChannels:
		m.erdsSomatosensoryRightChannels = strings.Join(splitLooseList(value), ",")
	// Behavior Statistics text fields
	case textFieldBehaviorPermGroupColumnPreference:
		m.behaviorPermGroupColumnPreference = strings.Join(splitLooseList(value), ",")
	case textFieldICCUnitColumns:
		m.iccUnitColumns = strings.Join(splitLooseList(value), ",")
	case textFieldBehaviorFeatureRegistryFilesJSON:
		m.behaviorFeatureRegistryFilesJSON = strings.TrimSpace(value)
	case textFieldBehaviorFeatureRegistrySourceToTypeJSON:
		m.behaviorFeatureRegistrySourceJSON = strings.TrimSpace(value)
	case textFieldBehaviorFeatureRegistryTypeHierarchyJSON:
		m.behaviorFeatureRegistryHierarchyJSON = strings.TrimSpace(value)
	case textFieldBehaviorFeatureRegistryPatternsJSON:
		m.behaviorFeatureRegistryPatternsJSON = strings.TrimSpace(value)
	case textFieldBehaviorFeatureRegistryClassifiersJSON:
		m.behaviorFeatureRegistryClassifiersJSON = strings.TrimSpace(value)
	// Preprocessing text fields
	case textFieldIcaLabelsToKeep:
		m.icaLabelsToKeep = strings.Join(strings.Fields(value), "")
	}
}
