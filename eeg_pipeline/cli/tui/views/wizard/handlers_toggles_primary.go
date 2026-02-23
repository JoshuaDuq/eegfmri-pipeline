package wizard

import (
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/types"
)

// Primary advanced-toggle handlers (features/behavior/plotting/fMRI groups).

func (m *Model) toggleAdvancedOption() {
	switch m.Pipeline {
	case types.PipelineFeatures:
		m.toggleFeaturesAdvancedOption()
	case types.PipelineBehavior:
		m.toggleBehaviorAdvancedOption()
	case types.PipelinePlotting:
		m.togglePlottingAdvancedOption()
	case types.PipelineML:
		m.toggleMLAdvancedOption()
	case types.PipelinePreprocessing:
		m.togglePreprocessingAdvancedOption()
	case types.PipelineFmri:
		m.toggleFmriAdvancedOption()
	case types.PipelineFmriAnalysis:
		m.toggleFmriAnalysisAdvancedOption()
	}
}

func (m *Model) toggleFeaturesAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

	options := m.getFeaturesOptions()

	if m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optFeatGroupConnectivity:
		m.featGroupConnectivityExpanded = !m.featGroupConnectivityExpanded
		if !m.featGroupConnectivityExpanded && m.expandedOption == expandedConnectivityMeasures {
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
		m.useDefaultAdvanced = false
	case optFeatGroupDirectedConnectivity:
		m.featGroupDirectedConnExpanded = !m.featGroupDirectedConnExpanded
		if !m.featGroupDirectedConnExpanded && m.expandedOption == expandedDirectedConnMeasures {
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
		m.useDefaultAdvanced = false
	case optFeatGroupPAC:
		m.featGroupPACExpanded = !m.featGroupPACExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupAperiodic:
		m.featGroupAperiodicExpanded = !m.featGroupAperiodicExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupComplexity:
		m.featGroupComplexityExpanded = !m.featGroupComplexityExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupBursts:
		m.featGroupBurstsExpanded = !m.featGroupBurstsExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupPower:
		m.featGroupPowerExpanded = !m.featGroupPowerExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupSpectral:
		m.featGroupSpectralExpanded = !m.featGroupSpectralExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupERP:
		m.featGroupERPExpanded = !m.featGroupERPExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupRatios:
		m.featGroupRatiosExpanded = !m.featGroupRatiosExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupAsymmetry:
		m.featGroupAsymmetryExpanded = !m.featGroupAsymmetryExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupQuality:
		m.featGroupQualityExpanded = !m.featGroupQualityExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupMicrostates:
		m.featGroupMicrostatesExpanded = !m.featGroupMicrostatesExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupERDS:
		m.featGroupERDSExpanded = !m.featGroupERDSExpanded
		m.useDefaultAdvanced = false
	// Asymmetry advanced options
	case optAsymmetryMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAsymmetryMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAsymmetrySkipInvalidSegments:
		m.asymmetrySkipInvalidSegments = !m.asymmetrySkipInvalidSegments
		m.useDefaultAdvanced = false
	// Ratios advanced options
	case optRatiosMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRatiosMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRatiosSkipInvalidSegments:
		m.ratiosSkipInvalidSegments = !m.ratiosSkipInvalidSegments
		m.useDefaultAdvanced = false
	// Spectral advanced options
	case optSpectralPsdMethod:
		m.spectralPsdMethod = (m.spectralPsdMethod + 1) % 2 // 0: multitaper, 1: welch
		m.useDefaultAdvanced = false
	case optSpectralFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralFmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Quality advanced options
	case optQualityPsdMethod:
		m.qualityPsdMethod = (m.qualityPsdMethod + 1) % 2 // 0: welch, 1: multitaper
		m.useDefaultAdvanced = false
	case optQualityFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityFmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityNFft:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityExcludeLineNoise:
		m.qualityExcludeLineNoise = !m.qualityExcludeLineNoise
		m.useDefaultAdvanced = false
	case optQualityLineNoiseFreq:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityLineNoiseWidthHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityLineNoiseHarmonics:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrSignalBandMin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrSignalBandMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrNoiseBandMin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrNoiseBandMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityMuscleBandMin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityMuscleBandMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Microstates advanced options
	case optMicrostatesNStates:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMicrostatesMinPeakDistanceMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMicrostatesMaxGfpPeaksPerEpoch:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMicrostatesMinDurationMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMicrostatesGfpPeakProminence:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMicrostatesRandomState:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ERDS advanced options
	case optERDSUseLogRatio:
		m.erdsUseLogRatio = !m.erdsUseLogRatio
		m.useDefaultAdvanced = false
	case optERDSMinBaselinePower:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSMinActivePower:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSBands:
		m.startTextEdit(textFieldERDSBands)
		m.useDefaultAdvanced = false
	case optERDSOnsetThresholdSigma:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSOnsetMinDurationMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSReboundMinLatencyMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSInferContralateral:
		m.erdsInferContralateral = !m.erdsInferContralateral
		m.useDefaultAdvanced = false
	case optFeatGroupStorage:
		m.featGroupStorageExpanded = !m.featGroupStorageExpanded
	case optSaveSubjectLevelFeatures:
		m.saveSubjectLevelFeatures = !m.saveSubjectLevelFeatures
		m.useDefaultAdvanced = false
	case optFeatAlsoSaveCsv:
		m.featAlsoSaveCsv = !m.featAlsoSaveCsv
		m.useDefaultAdvanced = false
	case optFeatGroupExecution:
		m.featGroupExecutionExpanded = !m.featGroupExecutionExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupSourceLoc:
		m.featGroupSourceLocExpanded = !m.featGroupSourceLocExpanded
		m.useDefaultAdvanced = false
	case optConnectivity:
		m.expandedOption = expandedConnectivityMeasures
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optDirectedConnMeasures:
		m.expandedOption = expandedDirectedConnMeasures
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optDirectedConnOutputLevel:
		m.directedConnOutputLevel = (m.directedConnOutputLevel + 1) % 2 // 0: full, 1: global_only
		m.useDefaultAdvanced = false
	case optDirectedConnMvarOrder:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optDirectedConnNFreqs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optDirectedConnMinSegSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Source localization options
	case optSourceLocMode:
		m.sourceLocMode = (m.sourceLocMode + 1) % 2 // 0: EEG-only, 1: fMRI-informed
		m.useDefaultAdvanced = false
	case optSourceLocMethod:
		m.sourceLocMethod = (m.sourceLocMethod + 1) % 2 // 0: lcmv, 1: eloreta
		m.useDefaultAdvanced = false
	case optSourceLocSpacing:
		m.sourceLocSpacing = (m.sourceLocSpacing + 1) % 4 // 0: oct5, 1: oct6, 2: ico4, 3: ico5
		m.useDefaultAdvanced = false
	case optSourceLocParc:
		m.sourceLocParc = (m.sourceLocParc + 1) % 3 // 0: aparc, 1: aparc.a2009s, 2: HCPMMP1
		m.useDefaultAdvanced = false
	case optSourceLocReg:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocSnr:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocLoose:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocDepth:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocConnMethod:
		m.sourceLocConnMethod = (m.sourceLocConnMethod + 1) % 3 // 0: aec, 1: wpli, 2: plv
		m.useDefaultAdvanced = false
	case optSourceLocSubject:
		m.startTextEdit(textFieldSourceLocSubject)
		m.useDefaultAdvanced = false
	case optSourceLocSubjectsDir:
		m.startTextEdit(textFieldSourceLocSubjectsDir)
		m.useDefaultAdvanced = false
	case optSourceLocTrans:
		m.browsingField = "sourceLocTrans"
		m.pendingFileCmd = m.browseForFile("Select coregistration transform file", "sourceLocTrans", "FIF files", "fif")
		m.useDefaultAdvanced = false
	case optSourceLocBem:
		m.browsingField = "sourceLocBem"
		m.pendingFileCmd = m.browseForFile("Select BEM solution file", "sourceLocBem", "FIF files", "fif")
		m.useDefaultAdvanced = false
	case optSourceLocMindistMm:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriEnabled:
		m.sourceLocFmriEnabled = !m.sourceLocFmriEnabled
		m.useDefaultAdvanced = false
	case optSourceLocFmriStatsMap:
		m.browsingField = "sourceLocFmriStatsMap"
		m.pendingFileCmd = m.browseForFile("Select fMRI statistical map", "sourceLocFmriStatsMap", "NIfTI files", "nii,nii.gz")
		m.useDefaultAdvanced = false
	case optSourceLocFmriProvenance:
		m.sourceLocFmriProvenance = (m.sourceLocFmriProvenance + 1) % 2 // 0: independent, 1: same_dataset
		m.useDefaultAdvanced = false
	case optSourceLocFmriRequireProvenance:
		m.sourceLocFmriRequireProv = !m.sourceLocFmriRequireProv
		m.useDefaultAdvanced = false
	case optSourceLocFmriThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriTail:
		m.sourceLocFmriTail = (m.sourceLocFmriTail + 1) % 2 // 0: pos, 1: abs
		m.useDefaultAdvanced = false
	case optSourceLocFmriMinClusterMM3:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMinClusterVox:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMaxClusters:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMaxVoxPerClus:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMaxTotalVox:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// BEM/Trans generation options (Docker-based)
	case optSourceLocCreateTrans:
		m.sourceLocCreateTrans = !m.sourceLocCreateTrans
		m.useDefaultAdvanced = false
	case optSourceLocCreateBemModel:
		m.sourceLocCreateBemModel = !m.sourceLocCreateBemModel
		m.useDefaultAdvanced = false
	case optSourceLocCreateBemSolution:
		m.sourceLocCreateBemSolution = !m.sourceLocCreateBemSolution
		m.useDefaultAdvanced = false
	// fMRI contrast builder options
	case optSourceLocFmriContrastEnabled:
		m.sourceLocFmriContrastEnabled = !m.sourceLocFmriContrastEnabled
		// Trigger condition discovery when enabling contrast builder
		if m.sourceLocFmriContrastEnabled && len(m.sourceLocFmriConditions) == 0 {
			subject := ""
			for _, s := range m.subjects {
				if m.subjectSelected[s.ID] {
					subject = s.ID
					break
				}
			}
			m.pendingFmriConditionsCmd = executor.DiscoverFmriConditions(m.repoRoot, subject, m.task)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriContrastType:
		m.sourceLocFmriContrastType = (m.sourceLocFmriContrastType + 1) % 4 // 0: t-test, 1: paired, 2: F-test, 3: custom
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondAColumn:
		if len(m.fmriDiscoveredColumns) > 0 {
			m.expandedOption = expandedFmriCondAColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondAColumn)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondAValue:
		colVals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn)
		if len(colVals) > 0 {
			m.expandedOption = expandedFmriCondAValue
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondAValue)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondBColumn:
		if len(m.fmriDiscoveredColumns) > 0 {
			m.expandedOption = expandedFmriCondBColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondBColumn)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondBValue:
		colVals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn)
		if len(colVals) > 0 {
			m.expandedOption = expandedFmriCondBValue
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondBValue)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriContrastFormula:
		m.startTextEdit(textFieldSourceLocFmriContrastFormula)
		m.useDefaultAdvanced = false
	case optSourceLocFmriContrastName:
		m.startTextEdit(textFieldSourceLocFmriContrastName)
		m.useDefaultAdvanced = false
	case optSourceLocFmriRunsToInclude:
		m.startTextEdit(textFieldSourceLocFmriRunsToInclude)
		m.useDefaultAdvanced = false
	case optSourceLocFmriAutoDetectRuns:
		m.sourceLocFmriAutoDetectRuns = !m.sourceLocFmriAutoDetectRuns
		m.useDefaultAdvanced = false
	case optSourceLocFmriHrfModel:
		m.sourceLocFmriHrfModel = (m.sourceLocFmriHrfModel + 1) % 3 // 0: SPM, 1: FLOBS, 2: FIR
		m.useDefaultAdvanced = false
	case optSourceLocFmriDriftModel:
		m.sourceLocFmriDriftModel = (m.sourceLocFmriDriftModel + 1) % 3 // 0: none, 1: cosine, 2: polynomial
		m.useDefaultAdvanced = false
	case optSourceLocFmriHighPassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriLowPassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriConditionScopeTrialTypes:
		m.expandedOption = expandedSourceLocFmriScopeTrialTypes
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optSourceLocFmriStimPhasesToModel:
		m.expandedOption = expandedSourceLocFmriStimPhases
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optSourceLocFmriClusterCorrection:
		m.sourceLocFmriClusterCorrection = !m.sourceLocFmriClusterCorrection
		m.useDefaultAdvanced = false
	case optSourceLocFmriClusterPThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriOutputType:
		m.sourceLocFmriOutputType = (m.sourceLocFmriOutputType + 1) % 4 // 0: z-score, 1: t-stat, 2: cope, 3: beta
		m.useDefaultAdvanced = false
	case optSourceLocFmriResampleToFS:
		m.sourceLocFmriResampleToFS = !m.sourceLocFmriResampleToFS
		m.useDefaultAdvanced = false
	case optSourceLocFmriInputSource:
		m.sourceLocFmriInputSource = (m.sourceLocFmriInputSource + 1) % 2 // 0: fmriprep, 1: bids_raw
		m.useDefaultAdvanced = false
	case optSourceLocFmriRequireFmriprep:
		m.sourceLocFmriRequireFmriprep = !m.sourceLocFmriRequireFmriprep
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowAName:
		m.startTextEdit(textFieldSourceLocFmriWindowAName)
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowATmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowATmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowBName:
		m.startTextEdit(textFieldSourceLocFmriWindowBName)
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowBTmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowBTmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ITPC options
	case optFeatGroupITPC:
		m.featGroupITPCExpanded = !m.featGroupITPCExpanded
		m.useDefaultAdvanced = false
	case optItpcMethod:
		m.itpcMethod = (m.itpcMethod + 1) % 4 // 0: global, 1: fold_global, 2: loo, 3: condition
		m.useDefaultAdvanced = false
	case optItpcAllowUnsafeLoo:
		m.itpcAllowUnsafeLoo = !m.itpcAllowUnsafeLoo
		m.useDefaultAdvanced = false
	case optItpcBaselineCorrection:
		m.itpcBaselineCorrection = (m.itpcBaselineCorrection + 1) % 2 // 0: none, 1: subtract
		m.useDefaultAdvanced = false
	case optItpcConditionColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedItpcConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldItpcConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optItpcConditionValues:
		if m.itpcConditionColumn == "" {
			m.ShowToast("Select a condition column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.itpcConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedItpcConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldItpcConditionValues)
		}
		m.useDefaultAdvanced = false
	case optItpcMinTrialsPerCondition:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optItpcNJobs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACPhaseRange:
		if m.pacPhaseMin == 4.0 && m.pacPhaseMax == 8.0 {
			m.pacPhaseMin, m.pacPhaseMax = 2.0, 4.0 // delta
		} else if m.pacPhaseMin == 2.0 && m.pacPhaseMax == 4.0 {
			m.pacPhaseMin, m.pacPhaseMax = 8.0, 13.0 // alpha
		} else {
			m.pacPhaseMin, m.pacPhaseMax = 4.0, 8.0 // theta (default)
		}
		m.useDefaultAdvanced = false
	case optPACAmpRange:
		if m.pacAmpMin == 30.0 && m.pacAmpMax == 80.0 {
			m.pacAmpMin, m.pacAmpMax = 40.0, 100.0 // broader gamma
		} else if m.pacAmpMin == 40.0 && m.pacAmpMax == 100.0 {
			m.pacAmpMin, m.pacAmpMax = 60.0, 120.0 // high gamma
		} else {
			m.pacAmpMin, m.pacAmpMax = 30.0, 80.0 // default gamma
		}
		m.useDefaultAdvanced = false
	case optPACMethod:
		m.pacMethod = (m.pacMethod + 1) % 4
		m.useDefaultAdvanced = false
	case optPACMinEpochs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACPairs:
		m.startTextEdit(textFieldPACPairs)
		m.useDefaultAdvanced = false
	case optPACSource:
		m.pacSource = (m.pacSource + 1) % 2 // 0: precomputed, 1: tfr
		m.useDefaultAdvanced = false
	case optPACNormalize:
		m.pacNormalize = !m.pacNormalize
		m.useDefaultAdvanced = false
	case optPACNSurrogates:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACAllowHarmonicOverlap:
		m.pacAllowHarmonicOvrlap = !m.pacAllowHarmonicOvrlap
		m.useDefaultAdvanced = false
	case optPACMaxHarmonic:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACHarmonicToleranceHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACComputeWaveformQC:
		m.pacComputeWaveformQC = !m.pacComputeWaveformQC
		m.useDefaultAdvanced = false
	case optPACWaveformOffsetMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicFmin, optAperiodicFmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicPeakZ, optAperiodicMinR2, optAperiodicMinPoints, optAperiodicPsdBandwidth, optAperiodicMaxRms, optAperiodicLineNoiseFreq, optAperiodicLineNoiseWidthHz, optAperiodicLineNoiseHarmonics:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicModel:
		m.aperiodicModel = (m.aperiodicModel + 1) % 2 // 0: fixed, 1: knee
		m.useDefaultAdvanced = false
	case optAperiodicPsdMethod:
		m.aperiodicPsdMethod = (m.aperiodicPsdMethod + 1) % 2 // 0: multitaper, 1: welch
		m.useDefaultAdvanced = false
	case optAperiodicExcludeLineNoise:
		m.aperiodicExcludeLineNoise = !m.aperiodicExcludeLineNoise
		m.useDefaultAdvanced = false
	case optPEOrder:
		m.complexityPEOrder++
		if m.complexityPEOrder > 7 {
			m.complexityPEOrder = 3
		}
		m.useDefaultAdvanced = false
	case optPEDelay:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optComplexitySampleEntropyOrder, optComplexitySampleEntropyR, optComplexityMSEScaleMin, optComplexityMSEScaleMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optComplexitySignalBasis:
		m.complexitySignalBasis++
		if m.complexitySignalBasis > 1 {
			m.complexitySignalBasis = 0
		}
		m.useDefaultAdvanced = false
	case optComplexityMinSegmentSec, optComplexityMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optComplexityZscore:
		m.complexityZscore = !m.complexityZscore
		m.useDefaultAdvanced = false
	case optERPBaseline:
		m.erpBaselineCorrection = !m.erpBaselineCorrection
		m.useDefaultAdvanced = false
	case optERPAllowNoBaseline:
		m.erpAllowNoBaseline = !m.erpAllowNoBaseline
		m.useDefaultAdvanced = false
	case optERPComponents:
		m.startTextEdit(textFieldERPComponents)
		m.useDefaultAdvanced = false
	case optERPSmoothMs, optERPPeakProminenceUv, optERPLowpassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstThresholdMethod:
		m.burstThresholdMethod++
		if m.burstThresholdMethod > 2 {
			m.burstThresholdMethod = 0
		}
		m.useDefaultAdvanced = false
	case optBurstThresholdPercentile:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstThreshold:
		switch m.burstThresholdZ {
		case 1.5:
			m.burstThresholdZ = 2.0
		case 2.0:
			m.burstThresholdZ = 2.5
		case 2.5:
			m.burstThresholdZ = 3.0
		default:
			m.burstThresholdZ = 1.5
		}
		m.useDefaultAdvanced = false
	case optBurstThresholdReference:
		m.burstThresholdReference = (m.burstThresholdReference + 1) % 3 // 0: trial, 1: subject, 2: condition
		m.useDefaultAdvanced = false
	case optBurstMinTrialsPerCondition, optBurstMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstSkipInvalidSegments:
		m.burstSkipInvalidSegments = !m.burstSkipInvalidSegments
		m.useDefaultAdvanced = false
	case optBurstMinDuration:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstMinCycles:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstBands:
		m.startTextEdit(textFieldBurstBands)
		m.useDefaultAdvanced = false
	case optPowerBaselineMode:
		m.powerBaselineMode = (m.powerBaselineMode + 1) % 5
		m.useDefaultAdvanced = false
	case optPowerRequireBaseline:
		m.powerRequireBaseline = !m.powerRequireBaseline
		m.useDefaultAdvanced = false
	case optPowerSubtractEvoked:
		m.powerSubtractEvoked = !m.powerSubtractEvoked
		m.useDefaultAdvanced = false
	case optPowerMinTrialsPerCondition:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPowerExcludeLineNoise:
		m.powerExcludeLineNoise = !m.powerExcludeLineNoise
		m.useDefaultAdvanced = false
	case optPowerLineNoiseFreq, optPowerLineNoiseWidthHz, optPowerLineNoiseHarmonics:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPowerEmitDb:
		m.powerEmitDb = !m.powerEmitDb
		m.useDefaultAdvanced = false
	case optSpectralEdge:
		switch m.spectralEdgePercentile {
		case 0.90:
			m.spectralEdgePercentile = 0.95
		case 0.95:
			m.spectralEdgePercentile = 0.99
		default:
			m.spectralEdgePercentile = 0.90
		}
		m.useDefaultAdvanced = false
	case optSpectralIncludeLogRatios:
		m.spectralIncludeLogRatios = !m.spectralIncludeLogRatios
		m.useDefaultAdvanced = false
	case optSpectralPsdAdaptive:
		m.spectralPsdAdaptive = !m.spectralPsdAdaptive
		m.useDefaultAdvanced = false
	case optSpectralMultitaperAdaptive:
		m.spectralMultitaperAdaptive = !m.spectralMultitaperAdaptive
		m.useDefaultAdvanced = false
	case optSpectralExcludeLineNoise:
		m.spectralExcludeLineNoise = !m.spectralExcludeLineNoise
		m.useDefaultAdvanced = false
	case optSpectralLineNoiseFreq, optSpectralLineNoiseWidthHz, optSpectralLineNoiseHarmonics:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConnOutputLevel:
		m.connOutputLevel = (m.connOutputLevel + 1) % 2
		m.useDefaultAdvanced = false
	case optConnGranularity:
		m.connGranularity = (m.connGranularity + 1) % 3 // 0: trial, 1: condition, 2: subject
		m.useDefaultAdvanced = false
	case optConnConditionColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedConnConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConnConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optConnConditionValues:
		if m.connConditionColumn == "" {
			m.ShowToast("Select a condition column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.connConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedConnConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConnConditionValues)
		}
		m.useDefaultAdvanced = false
	case optConnPhaseEstimator:
		m.connPhaseEstimator = (m.connPhaseEstimator + 1) % 2 // 0: within_epoch, 1: across_epochs
		m.useDefaultAdvanced = false
	case optConnMinEpochsPerGroup, optConnMinCyclesPerBand, optConnMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConnWarnNoSpatialTransform:
		m.connWarnNoSpatialTransform = !m.connWarnNoSpatialTransform
		m.useDefaultAdvanced = false
	case optConnGraphMetrics:
		m.connGraphMetrics = !m.connGraphMetrics
		m.useDefaultAdvanced = false
	case optConnGraphProp, optConnWindowLen, optConnWindowStep:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConnAECMode:
		m.connAECMode = (m.connAECMode + 1) % 3
		m.useDefaultAdvanced = false
	case optConnMode:
		m.connMode = (m.connMode + 1) % 3
		m.useDefaultAdvanced = false
	case optConnAECAbsolute:
		m.connAECAbsolute = !m.connAECAbsolute
		m.useDefaultAdvanced = false
	case optConnEnableAEC:
		m.connEnableAEC = !m.connEnableAEC
		m.useDefaultAdvanced = false
	case optConnNFreqsPerBand, optConnNCycles, optConnDecim, optConnMinSegmentSamples, optConnSmallWorldNRand:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConnAECOutput:
		m.connAECOutput = (m.connAECOutput + 1) % 3 // 0: r, 1: z, 2: r+z
		m.useDefaultAdvanced = false
	case optConnForceWithinEpochML:
		m.connForceWithinEpochML = !m.connForceWithinEpochML
		m.useDefaultAdvanced = false
	case optConnDynamicEnabled:
		m.connDynamicEnabled = !m.connDynamicEnabled
		m.useDefaultAdvanced = false
	case optConnDynamicMeasures:
		m.connDynamicMeasures = (m.connDynamicMeasures + 1) % 3 // 0: wpli+aec, 1: wpli, 2: aec
		m.useDefaultAdvanced = false
	case optConnDynamicAutocorrLag, optConnDynamicMinWindows, optConnDynamicStateNStates, optConnDynamicStateMinWindows, optConnDynamicStateRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConnDynamicIncludeROIPairs:
		m.connDynamicIncludeROIPairs = !m.connDynamicIncludeROIPairs
		m.useDefaultAdvanced = false
	case optConnDynamicStateEnabled:
		m.connDynamicStateEnabled = !m.connDynamicStateEnabled
		m.useDefaultAdvanced = false

	case optMinEpochs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFeatAnalysisMode:
		m.featAnalysisMode = (m.featAnalysisMode + 1) % 2 // 0: group_stats, 1: trial_ml_safe
		m.useDefaultAdvanced = false
	case optFeatComputeChangeScores:
		m.featComputeChangeScores = !m.featComputeChangeScores
		m.useDefaultAdvanced = false
	case optFeatSaveTfrWithSidecar:
		m.featSaveTfrWithSidecar = !m.featSaveTfrWithSidecar
		m.useDefaultAdvanced = false
	case optFeatNJobsBands, optFeatNJobsConnectivity, optFeatNJobsAperiodic, optFeatNJobsComplexity:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralRatioPairs:
		m.startTextEdit(textFieldSpectralRatioPairs)
		m.useDefaultAdvanced = false
	case optSpectralSegments:
		m.startTextEdit(textFieldSpectralSegments)
		m.useDefaultAdvanced = false
	case optAperiodicSubtractEvoked:
		m.aperiodicSubtractEvoked = !m.aperiodicSubtractEvoked
		m.useDefaultAdvanced = false
	case optAsymmetryChannelPairs:
		m.startTextEdit(textFieldAsymmetryChannelPairs)
		m.useDefaultAdvanced = false
	case optAsymmetryEmitActivationConvention:
		m.asymmetryEmitActivationConvention = !m.asymmetryEmitActivationConvention
		m.useDefaultAdvanced = false
	case optAsymmetryActivationBands:
		m.startTextEdit(textFieldAsymmetryActivationBands)
		m.useDefaultAdvanced = false
	// Spatial transform section
	case optFeatGroupSpatialTransform:
		m.featGroupSpatialTransformExpanded = !m.featGroupSpatialTransformExpanded
		m.useDefaultAdvanced = false
	case optSpatialTransform:
		m.spatialTransform = (m.spatialTransform + 1) % 3 // 0=none, 1=csd, 2=laplacian
		m.useDefaultAdvanced = false
	case optSpatialTransformLambda2, optSpatialTransformStiffness:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Per-family spatial transform overrides (cycle through: inherit/none/csd/laplacian)
	case optSpatialTransformPerFamilyConnectivity:
		m.spatialTransformPerFamilyConnectivity = (m.spatialTransformPerFamilyConnectivity + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyItpc:
		m.spatialTransformPerFamilyItpc = (m.spatialTransformPerFamilyItpc + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyPac:
		m.spatialTransformPerFamilyPac = (m.spatialTransformPerFamilyPac + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyPower:
		m.spatialTransformPerFamilyPower = (m.spatialTransformPerFamilyPower + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyAperiodic:
		m.spatialTransformPerFamilyAperiodic = (m.spatialTransformPerFamilyAperiodic + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyBursts:
		m.spatialTransformPerFamilyBursts = (m.spatialTransformPerFamilyBursts + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyErds:
		m.spatialTransformPerFamilyErds = (m.spatialTransformPerFamilyErds + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyComplexity:
		m.spatialTransformPerFamilyComplexity = (m.spatialTransformPerFamilyComplexity + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyRatios:
		m.spatialTransformPerFamilyRatios = (m.spatialTransformPerFamilyRatios + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyAsymmetry:
		m.spatialTransformPerFamilyAsymmetry = (m.spatialTransformPerFamilyAsymmetry + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilySpectral:
		m.spatialTransformPerFamilySpectral = (m.spatialTransformPerFamilySpectral + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyErp:
		m.spatialTransformPerFamilyErp = (m.spatialTransformPerFamilyErp + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyQuality:
		m.spatialTransformPerFamilyQuality = (m.spatialTransformPerFamilyQuality + 1) % 4
		m.useDefaultAdvanced = false
	case optSpatialTransformPerFamilyMicrostates:
		m.spatialTransformPerFamilyMicrostates = (m.spatialTransformPerFamilyMicrostates + 1) % 4
		m.useDefaultAdvanced = false
	// ITPC/PAC segment validity
	case optItpcMinSegmentSec, optItpcMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACMinSegmentSec, optPACMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACSurrogateMethod:
		m.pacSurrogateMethod = (m.pacSurrogateMethod + 1) % 4
		m.useDefaultAdvanced = false
	// Aperiodic missing
	case optAperiodicMaxFreqResolutionHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicMultitaperAdaptive:
		m.aperiodicMultitaperAdaptive = !m.aperiodicMultitaperAdaptive
		m.useDefaultAdvanced = false
	// Directed connectivity missing
	case optDirectedConnMinSamplesPerMvarParam:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ERDS pain markers
	case optERDSPainMarkerBands:
		m.startTextEdit(textFieldERDSPainMarkerBands)
		m.useDefaultAdvanced = false
	case optERDSLateralityColumns:
		m.startTextEdit(textFieldERDSLateralityColumns)
		m.useDefaultAdvanced = false
	case optERDSSomatosensoryLeftChannels:
		m.startTextEdit(textFieldERDSSomatosensoryLeftChannels)
		m.useDefaultAdvanced = false
	case optERDSSomatosensoryRightChannels:
		m.startTextEdit(textFieldERDSSomatosensoryRightChannels)
		m.useDefaultAdvanced = false
	case optERDSOnsetMinThresholdPercent, optERDSReboundThresholdSigma, optERDSReboundMinThresholdPercent:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Microstates missing
	case optMicrostatesAssignFromGfpPeaks:
		m.microstatesAssignFromGfpPeaks = !m.microstatesAssignFromGfpPeaks
		m.useDefaultAdvanced = false
	case optMicrostatesFixedTemplatesPath:
		m.startTextEdit(textFieldMicrostatesFixedTemplatesPath)
		m.useDefaultAdvanced = false
	// Change scores
	case optChangeScoresTransform:
		m.changeScoresTransform = (m.changeScoresTransform + 1) % 3
		m.useDefaultAdvanced = false
	case optChangeScoresWindowPairs:
		m.startTextEdit(textFieldChangeScoresWindowPairs)
		m.useDefaultAdvanced = false
	// TFR section
	case optFeatGroupTFR:
		m.featGroupTFRExpanded = !m.featGroupTFRExpanded
	case optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrMaxCycles, optTfrNCyclesFactor, optTfrDecim, optTfrDecimPower, optTfrDecimPhase, optTfrWorkers:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAggregationMethod:
		m.aggregationMethod = (m.aggregationMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optFeatureTmin, optFeatureTmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBandEnvelopePadSec, optBandEnvelopePadCycles, optIAFAlphaWidthHz, optIAFSearchRangeMin, optIAFSearchRangeMax, optIAFMinProminence, optIAFMinCyclesAtFmin, optIAFMinBaselineSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optIAFEnabled:
		m.iafEnabled = !m.iafEnabled
		m.useDefaultAdvanced = false
	case optIAFRois:
		m.expandedOption = expandedIAFRois
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optIAFAllowFullFallback:
		m.iafAllowFullFallback = !m.iafAllowFullFallback
		m.useDefaultAdvanced = false
	case optIAFAllowAllChannelsFallback:
		m.iafAllowAllChannelsFallback = !m.iafAllowAllChannelsFallback
		m.useDefaultAdvanced = false
	}

	options = m.getFeaturesOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) togglePlottingAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

	rows := m.getPlottingAdvancedRows()
	if m.advancedCursor < 0 || m.advancedCursor >= len(rows) {
		return
	}

	cycleTriState := func(v *bool) *bool {
		if v == nil {
			t := true
			return &t
		}
		if *v {
			f := false
			return &f
		}
		return nil
	}

	row := rows[m.advancedCursor]
	switch row.kind {
	case plottingRowPlotHeader:
		m.plotItemConfigExpanded[row.plotID] = !m.plotItemConfigExpanded[row.plotID]
		m.UpdateAdvancedOffset()
		return
	case plottingRowPlotField:
		cfg := m.ensurePlotItemConfig(row.plotID)
		switch row.plotField {
		case plotItemConfigFieldCompareWindows:
			cfg.CompareWindows = cycleTriState(cfg.CompareWindows)
			m.plotItemConfigs[row.plotID] = cfg
			m.useDefaultAdvanced = false
		case plotItemConfigFieldCompareColumns:
			cfg.CompareColumns = cycleTriState(cfg.CompareColumns)
			m.plotItemConfigs[row.plotID] = cfg
			m.useDefaultAdvanced = false
		case plotItemConfigFieldItpcSharedColorbar:
			cfg.ItpcSharedColorbar = cycleTriState(cfg.ItpcSharedColorbar)
			m.plotItemConfigs[row.plotID] = cfg
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonWindows, plotItemConfigFieldComparisonSegment, plotItemConfigFieldTopomapWindow:
			// Open dropdown if windows available for this feature, otherwise text edit
			featureGroup := m.getFeatureGroupForPlot(row.plotID)
			windows := m.GetPlottingComparisonWindows(featureGroup)
			if len(windows) > 0 {
				m.expandedOption = expandedPlotComparisonWindows
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonColumn:
			// Open dropdown if columns available, otherwise text edit
			plotCols := m.GetPlottingComparisonColumns()
			if len(plotCols) > 0 {
				m.expandedOption = expandedPlotComparisonColumn
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldConnectivityCircleTopFraction, plotItemConfigFieldConnectivityCircleMinLines, plotItemConfigFieldConnectivityNetworkTopFraction:
			m.startPlotTextEdit(row.plotID, row.plotField)
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseDoseColumn, plotItemConfigFieldDoseResponsePainColumn:
			// Dose/pain columns come from aligned events / trial metadata.
			plotCols := m.GetPlottingComparisonColumns()
			if len(plotCols) > 0 {
				m.expandedOption = expandedPlotComparisonColumn
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseResponseColumn:
			// Response dropdown shows detected feature categories only.
			categories := m.GetTrialTableFeatureCategories()
			if len(categories) > 0 {
				m.expandedOption = expandedPlotComparisonColumn
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				// No fallbacks: do not show events columns.
				if len(m.trialTableColumns) == 0 {
					m.ShowToast("No trial table columns discovered (run trial_table first).", "warning")
				} else {
					m.ShowToast("No feature categories discovered in trial table.", "warning")
				}
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseSegment:
			// Segment is a window label (from available windows). Use the same dropdown list as comparison windows.
			windows := m.GetPlottingComparisonWindows()
			if len(windows) > 0 {
				m.expandedOption = expandedPlotComparisonWindows
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseBands:
			bands := m.GetDoseResponseBands(m.getDoseResponseCategoriesForEditingPlot())
			if len(bands) > 0 {
				m.expandedOption = expandedDoseResponseBands
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseROIs:
			rois := m.GetDoseResponseROIs(m.getDoseResponseCategoriesForEditingPlot())
			if len(rois) > 0 {
				m.expandedOption = expandedDoseResponseROIs
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseScopes:
			scopes := m.GetDoseResponseScopes(m.getDoseResponseCategoriesForEditingPlot())
			if len(scopes) > 0 {
				m.expandedOption = expandedDoseResponseScopes
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldDoseResponseStat:
			stats := m.GetDoseResponseStats(m.getDoseResponseCategoriesForEditingPlot())
			if len(stats) > 0 {
				m.expandedOption = expandedDoseResponseStat
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldTfrTopomapActiveWindow, plotItemConfigFieldTfrTopomapWindowSizeMs, plotItemConfigFieldTfrTopomapWindowCount,
			plotItemConfigFieldTfrTopomapLabelXPosition, plotItemConfigFieldTfrTopomapLabelYPositionBottom,
			plotItemConfigFieldTfrTopomapLabelYPosition, plotItemConfigFieldTfrTopomapTitleY,
			plotItemConfigFieldTfrTopomapTitlePad, plotItemConfigFieldTfrTopomapSubplotsRight,
			plotItemConfigFieldTfrTopomapTemporalHspace, plotItemConfigFieldTfrTopomapTemporalWspace:
			m.startPlotTextEdit(row.plotID, row.plotField)
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonValues:
			// Always open dropdown if column selected (values may be discovered later)
			cfg := m.plotItemConfigs[row.plotID]
			col := cfg.ComparisonColumn
			if col == "" {
				col = m.plotComparisonColumn // fallback to global
			}
			if col != "" {
				// Column is selected - open dropdown (even if values not discovered yet)
				m.expandedOption = expandedPlotComparisonValues
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				// No column selected - use text edit
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonROIs:
			// Open dropdown if ROIs available, otherwise text edit
			if len(m.discoveredROIs) > 0 {
				m.expandedOption = expandedPlotComparisonROIs
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonLabels:
			m.startPlotTextEdit(row.plotID, row.plotField)
			m.useDefaultAdvanced = false
		case plotItemConfigFieldBehaviorScatterFeatures:
			// Open dropdown for feature types
			m.expandedOption = expandedBehaviorScatterFeatures
			m.subCursor = 0
			m.editingPlotID = row.plotID
			m.editingPlotField = row.plotField
			m.useDefaultAdvanced = false
		case plotItemConfigFieldBehaviorScatterColumns:
			// Open dropdown if columns available, otherwise text edit
			plotCols := m.GetPlottingComparisonColumns()
			if len(plotCols) > 0 {
				m.expandedOption = expandedBehaviorScatterColumns
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldBehaviorScatterAggregationModes:
			// Open dropdown for aggregation modes
			m.expandedOption = expandedBehaviorScatterAggregation
			m.subCursor = 0
			m.editingPlotID = row.plotID
			m.editingPlotField = row.plotField
			m.useDefaultAdvanced = false
		case plotItemConfigFieldBehaviorScatterSegment:
			// Open dropdown if windows available, otherwise text edit
			windows := m.GetPlottingComparisonWindows()
			if len(windows) > 0 {
				m.expandedOption = expandedBehaviorScatterSegment
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldBehaviorTemporalStatsFeatureFolder:
			folders, err := m.discoverTemporalTopomapsStatsFeatureFolders()
			m.temporalTopomapsStatsFeatureFolders = folders
			if err != nil {
				m.temporalTopomapsStatsFeatureFoldersError = err.Error()
			} else {
				m.temporalTopomapsStatsFeatureFoldersError = ""
			}

			if len(folders) > 0 {
				m.expandedOption = expandedTemporalTopomapsFeatureDir
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		}
		m.UpdateAdvancedOffset()
		return
	case plottingRowSection, plottingRowPlotInfo:
		return
	}

	opt := row.opt
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
		if m.useDefaultAdvanced {
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
		m.UpdateAdvancedOffset()
		return

	case optPlotGroupDefaults, optPlotGroupFonts, optPlotGroupLayout, optPlotGroupFigureSizes,
		optPlotGroupColors, optPlotGroupAlpha, optPlotGroupScatter, optPlotGroupBar,
		optPlotGroupLine, optPlotGroupHistogram, optPlotGroupKDE, optPlotGroupErrorbar,
		optPlotGroupText, optPlotGroupValidation, optPlotGroupTopomap, optPlotGroupTFR,
		optPlotGroupSizing, optPlotGroupSelection, optPlotGroupComparisons, optPlotGroupTFRMisc:
		m.togglePlotGroupExpansion(opt)
		m.useDefaultAdvanced = false

	case optPlotBboxInches:
		m.startTextEdit(textFieldPlotBboxInches)
		m.useDefaultAdvanced = false
	case optPlotFontFamily:
		m.startTextEdit(textFieldPlotFontFamily)
		m.useDefaultAdvanced = false
	case optPlotFontWeight:
		m.startTextEdit(textFieldPlotFontWeight)
		m.useDefaultAdvanced = false
	case optPlotLayoutTightRect:
		m.startTextEdit(textFieldPlotLayoutTightRect)
		m.useDefaultAdvanced = false
	case optPlotLayoutTightRectMicrostate:
		m.startTextEdit(textFieldPlotLayoutTightRectMicrostate)
		m.useDefaultAdvanced = false
	case optPlotGridSpecWidthRatios:
		m.startTextEdit(textFieldPlotGridSpecWidthRatios)
		m.useDefaultAdvanced = false
	case optPlotGridSpecHeightRatios:
		m.startTextEdit(textFieldPlotGridSpecHeightRatios)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeStandard:
		m.startTextEdit(textFieldPlotFigureSizeStandard)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeMedium:
		m.startTextEdit(textFieldPlotFigureSizeMedium)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeSmall:
		m.startTextEdit(textFieldPlotFigureSizeSmall)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeSquare:
		m.startTextEdit(textFieldPlotFigureSizeSquare)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeWide:
		m.startTextEdit(textFieldPlotFigureSizeWide)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeTFR:
		m.startTextEdit(textFieldPlotFigureSizeTFR)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeTopomap:
		m.startTextEdit(textFieldPlotFigureSizeTopomap)
		m.useDefaultAdvanced = false

	case optPlotColorPain:
		m.startTextEdit(textFieldPlotColorPain)
		m.useDefaultAdvanced = false
	case optPlotColorNonpain:
		m.startTextEdit(textFieldPlotColorNonpain)
		m.useDefaultAdvanced = false
	case optPlotColorSignificant:
		m.startTextEdit(textFieldPlotColorSignificant)
		m.useDefaultAdvanced = false
	case optPlotColorNonsignificant:
		m.startTextEdit(textFieldPlotColorNonsignificant)
		m.useDefaultAdvanced = false
	case optPlotColorGray:
		m.startTextEdit(textFieldPlotColorGray)
		m.useDefaultAdvanced = false
	case optPlotColorLightGray:
		m.startTextEdit(textFieldPlotColorLightGray)
		m.useDefaultAdvanced = false
	case optPlotColorBlack:
		m.startTextEdit(textFieldPlotColorBlack)
		m.useDefaultAdvanced = false
	case optPlotColorBlue:
		m.startTextEdit(textFieldPlotColorBlue)
		m.useDefaultAdvanced = false
	case optPlotColorRed:
		m.startTextEdit(textFieldPlotColorRed)
		m.useDefaultAdvanced = false
	case optPlotColorNetworkNode:
		m.startTextEdit(textFieldPlotColorNetworkNode)
		m.useDefaultAdvanced = false

	case optPlotScatterEdgecolor:
		m.startTextEdit(textFieldPlotScatterEdgecolor)
		m.useDefaultAdvanced = false
	case optPlotHistEdgecolor:
		m.startTextEdit(textFieldPlotHistEdgecolor)
		m.useDefaultAdvanced = false
	case optPlotKdeColor:
		m.startTextEdit(textFieldPlotKdeColor)
		m.useDefaultAdvanced = false

	case optPlotTopomapContours,
		optPlotPadInches,
		optPlotFontSizeSmall,
		optPlotFontSizeMedium,
		optPlotFontSizeLarge,
		optPlotFontSizeTitle,
		optPlotFontSizeAnnotation,
		optPlotFontSizeLabel,
		optPlotFontSizeYLabel,
		optPlotFontSizeSuptitle,
		optPlotFontSizeFigureTitle,
		optPlotGridSpecHspace,
		optPlotGridSpecWspace,
		optPlotGridSpecLeft,
		optPlotGridSpecRight,
		optPlotGridSpecTop,
		optPlotGridSpecBottom,
		optPlotAlphaGrid,
		optPlotAlphaFill,
		optPlotAlphaCI,
		optPlotAlphaCILine,
		optPlotAlphaTextBox,
		optPlotAlphaViolinBody,
		optPlotAlphaRidgeFill,
		optPlotScatterMarkerSizeSmall,
		optPlotScatterMarkerSizeLarge,
		optPlotScatterMarkerSizeDefault,
		optPlotScatterAlpha,
		optPlotScatterEdgewidth,
		optPlotBarAlpha,
		optPlotBarWidth,
		optPlotBarCapsize,
		optPlotBarCapsizeLarge,
		optPlotLineWidthThin,
		optPlotLineWidthStandard,
		optPlotLineWidthThick,
		optPlotLineWidthBold,
		optPlotLineAlphaStandard,
		optPlotLineAlphaDim,
		optPlotLineAlphaZeroLine,
		optPlotLineAlphaFitLine,
		optPlotLineAlphaDiagonal,
		optPlotLineAlphaReference,
		optPlotLineRegressionWidth,
		optPlotLineResidualWidth,
		optPlotLineQQWidth,
		optPlotHistBins,
		optPlotHistBinsBehavioral,
		optPlotHistBinsResidual,
		optPlotHistBinsTFR,
		optPlotHistEdgewidth,
		optPlotHistAlpha,
		optPlotHistAlphaResidual,
		optPlotHistAlphaTFR,
		optPlotKdePoints,
		optPlotKdeLinewidth,
		optPlotKdeAlpha,
		optPlotErrorbarMarkersize,
		optPlotErrorbarCapsize,
		optPlotErrorbarCapsizeLarge,
		optPlotTextStatsX,
		optPlotTextStatsY,
		optPlotTextPvalueX,
		optPlotTextPvalueY,
		optPlotTextBootstrapX,
		optPlotTextBootstrapY,
		optPlotTextChannelAnnotationX,
		optPlotTextChannelAnnotationY,
		optPlotTextTitleY,
		optPlotTextResidualQcTitleY,
		optPlotValidationMinBinsForCalibration,
		optPlotValidationMaxBinsForCalibration,
		optPlotValidationSamplesPerBin,
		optPlotValidationMinRoisForFDR,
		optPlotValidationMinPvaluesForFDR,
		optPlotTopomapColorbarFraction,
		optPlotTopomapColorbarPad,
		optPlotTopomapSigMaskLinewidth,
		optPlotTopomapSigMaskMarkersize,
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
		optPlotRoiWidthPerBand,
		optPlotRoiWidthPerMetric,
		optPlotRoiHeightPerRoi,
		optPlotPowerWidthPerBand,
		optPlotPowerHeightPerSegment,
		optPlotItpcWidthPerBin,
		optPlotItpcHeightPerBand,
		optPlotItpcWidthPerBandBox,
		optPlotItpcHeightBox,
		optPlotPacWidthPerRoi,
		optPlotPacHeightBox,
		optPlotAperiodicWidthPerColumn,
		optPlotAperiodicHeightPerRow,
		optPlotAperiodicNPerm,
		optPlotComplexityWidthPerMeasure,
		optPlotComplexityHeightPerSegment,
		optPlotConnectivityWidthPerCircle,
		optPlotConnectivityWidthPerBand,
		optPlotConnectivityHeightPerMeasure,
		optPlotConnectivityCircleTopFraction,
		optPlotConnectivityCircleMinLines,
		optPlotConnectivityNetworkTopFraction:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	case optPlotTopomapColormap:
		m.startTextEdit(textFieldPlotTopomapColormap)
		m.useDefaultAdvanced = false
	case optPlotTopomapSigMaskMarker:
		m.startTextEdit(textFieldPlotTopomapSigMaskMarker)
		m.useDefaultAdvanced = false
	case optPlotTopomapSigMaskMarkerFaceColor:
		m.startTextEdit(textFieldPlotTopomapSigMaskMarkerFaceColor)
		m.useDefaultAdvanced = false
	case optPlotTopomapSigMaskMarkerEdgeColor:
		m.startTextEdit(textFieldPlotTopomapSigMaskMarkerEdgeColor)
		m.useDefaultAdvanced = false
	case optPlotPacCmap:
		m.startTextEdit(textFieldPlotPacCmap)
		m.useDefaultAdvanced = false

	case optPlotTopomapDiffAnnotation:
		m.plotTopomapDiffAnnotation = cycleTriState(m.plotTopomapDiffAnnotation)
		m.useDefaultAdvanced = false
	case optPlotTopomapAnnotateDescriptive:
		m.plotTopomapAnnotateDesc = cycleTriState(m.plotTopomapAnnotateDesc)
		m.useDefaultAdvanced = false

	case optPlotPacPairs:
		m.startTextEdit(textFieldPlotPacPairs)
		m.useDefaultAdvanced = false
	case optPlotConnectivityMeasures:
		m.startTextEdit(textFieldPlotConnectivityMeasures)
		m.useDefaultAdvanced = false
	case optPlotSpectralMetrics:
		m.startTextEdit(textFieldPlotSpectralMetrics)
		m.useDefaultAdvanced = false
	case optPlotBurstsMetrics:
		m.startTextEdit(textFieldPlotBurstsMetrics)
		m.useDefaultAdvanced = false
	case optPlotAsymmetryStat:
		m.startTextEdit(textFieldPlotAsymmetryStat)
		m.useDefaultAdvanced = false
	case optPlotTemporalTimeBins:
		m.startTextEdit(textFieldPlotTemporalTimeBins)
		m.useDefaultAdvanced = false
	case optPlotTemporalTimeLabels:
		m.startTextEdit(textFieldPlotTemporalTimeLabels)
		m.useDefaultAdvanced = false

	case optPlotCompareWindows:
		m.plotCompareWindows = cycleTriState(m.plotCompareWindows)
		m.useDefaultAdvanced = false
	case optPlotComparisonWindows:
		if len(m.availableWindows) > 0 {
			m.expandedOption = expandedPlotComparisonWindows
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPlotComparisonWindows)
		}
		m.useDefaultAdvanced = false
	case optPlotCompareColumns:
		m.plotCompareColumns = cycleTriState(m.plotCompareColumns)
		m.useDefaultAdvanced = false
	case optPlotComparisonSegment:
		m.startTextEdit(textFieldPlotComparisonSegment)
		m.useDefaultAdvanced = false
	case optPlotComparisonColumn:
		if len(m.discoveredColumns) > 0 {
			m.expandedOption = expandedPlotComparisonColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPlotComparisonColumn)
		}
		m.useDefaultAdvanced = false
	case optPlotComparisonValues:
		if m.plotComparisonColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetPlottingComparisonColumnValues(m.plotComparisonColumn); len(vals) > 0 {
			m.expandedOption = expandedPlotComparisonValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPlotComparisonValues)
		}
		m.useDefaultAdvanced = false
	case optPlotComparisonLabels:
		m.startTextEdit(textFieldPlotComparisonLabels)
		m.useDefaultAdvanced = false
	case optPlotComparisonROIs:
		m.startTextEdit(textFieldPlotComparisonROIs)
		m.useDefaultAdvanced = false
	case optPlotOverwrite:
		if m.plotOverwrite == nil {
			val := true
			m.plotOverwrite = &val
		} else {
			val := !*m.plotOverwrite
			m.plotOverwrite = &val
		}
		m.useDefaultAdvanced = false
	}

	rows = m.getPlottingAdvancedRows()
	if len(rows) == 0 {
		m.advancedCursor = 0
		m.UpdateAdvancedOffset()
		return
	}
	m.advancedCursor = clampCursor(m.advancedCursor, len(rows)-1)
	if rows[m.advancedCursor].kind == plottingRowSection || rows[m.advancedCursor].kind == plottingRowPlotInfo {
		m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, 1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) toggleBehaviorAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

	options := m.getBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	sections := m.behaviorSections()
	sectionEnabled := true
	if len(sections) > 0 {
		idx := m.behaviorConfigSection
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sections) {
			idx = len(sections) - 1
		}
		sectionEnabled = sections[idx].Enabled
	}

	opt := options[m.advancedCursor]
	if opt != optUseDefaults && !sectionEnabled {
		return
	}

	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	// Behavior sub-section headers are non-interactive visual separators
	case optBehaviorSubCorrelationSettings, optBehaviorSubCovariates, optBehaviorSubRunAdjustment,
		optBehaviorSubCorrelationsExtra, optBehaviorSubFeatureQC, optBehaviorSubOutcome,
		optBehaviorSubOutcomes, optBehaviorSubModelFamilies, optBehaviorSubInference,
		optBehaviorSubDiagnostics, optBehaviorSubFitting, optBehaviorSubCrossfit,
		optBehaviorSubTimeWindow, optBehaviorSubFeatures, optBehaviorSubITPC,
		optBehaviorSubERDS, optBehaviorSubMultilevel:
		// no-op: sub-headers are visual separators only
	// Behavior section header toggles
	case optBehaviorGroupGeneral:
		m.behaviorGroupGeneralExpanded = !m.behaviorGroupGeneralExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupTrialTable:
		m.behaviorGroupTrialTableExpanded = !m.behaviorGroupTrialTableExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupPainResidual:
		m.behaviorGroupPainResidualExpanded = !m.behaviorGroupPainResidualExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCorrelations:
		m.behaviorGroupCorrelationsExpanded = !m.behaviorGroupCorrelationsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupPainSens:
		m.behaviorGroupPainSensExpanded = !m.behaviorGroupPainSensExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupRegression:
		m.behaviorGroupRegressionExpanded = !m.behaviorGroupRegressionExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupModels:
		m.behaviorGroupModelsExpanded = !m.behaviorGroupModelsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupStability:
		m.behaviorGroupStabilityExpanded = !m.behaviorGroupStabilityExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupConsistency:
		m.behaviorGroupConsistencyExpanded = !m.behaviorGroupConsistencyExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupInfluence:
		m.behaviorGroupInfluenceExpanded = !m.behaviorGroupInfluenceExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupReport:
		m.behaviorGroupReportExpanded = !m.behaviorGroupReportExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCondition:
		m.behaviorGroupConditionExpanded = !m.behaviorGroupConditionExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupTemporal:
		m.behaviorGroupTemporalExpanded = !m.behaviorGroupTemporalExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCluster:
		m.behaviorGroupClusterExpanded = !m.behaviorGroupClusterExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupMediation:
		m.behaviorGroupMediationExpanded = !m.behaviorGroupMediationExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupModeration:
		m.behaviorGroupModerationExpanded = !m.behaviorGroupModerationExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupMixedEffects:
		m.behaviorGroupMixedEffectsExpanded = !m.behaviorGroupMixedEffectsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupStats:
		m.behaviorGroupStatsExpanded = !m.behaviorGroupStatsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupAnalyses:
		m.behaviorGroupAnalysesExpanded = !m.behaviorGroupAnalysesExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupAdvanced:
		m.behaviorGroupAdvancedExpanded = !m.behaviorGroupAdvancedExpanded
		m.useDefaultAdvanced = false
	case optCorrMethod:
		if m.correlationMethod == "spearman" {
			m.correlationMethod = "pearson"
		} else {
			m.correlationMethod = "spearman"
		}
		m.useDefaultAdvanced = false
	case optRobustCorrelation:
		m.robustCorrelation = (m.robustCorrelation + 1) % 4
		m.useDefaultAdvanced = false
	case optBootstrap:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optNPerm:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRNGSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBehaviorNJobs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optControlTemp:
		m.controlTemperature = !m.controlTemperature
		m.useDefaultAdvanced = false
	case optControlOrder:
		m.controlTrialOrder = !m.controlTrialOrder
		m.useDefaultAdvanced = false
	case optRunAdjustmentEnabled:
		m.runAdjustmentEnabled = !m.runAdjustmentEnabled
		m.useDefaultAdvanced = false
	case optRunAdjustmentColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedRunAdjustmentColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldRunAdjustmentColumn)
		}
		m.useDefaultAdvanced = false
	case optBehaviorOutcomeColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedBehaviorOutcomeColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldBehaviorOutcomeColumn)
		}
		m.useDefaultAdvanced = false
	case optBehaviorPredictorColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedBehaviorPredictorColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldBehaviorPredictorColumn)
		}
		m.useDefaultAdvanced = false
	case optRunAdjustmentIncludeInCorrelations:
		m.runAdjustmentIncludeInCorrelations = !m.runAdjustmentIncludeInCorrelations
		m.useDefaultAdvanced = false
	case optRunAdjustmentMaxDummies:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBehaviorMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFDRAlpha:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	case optComputeChangeScores:
		m.behaviorComputeChangeScores = !m.behaviorComputeChangeScores
		m.useDefaultAdvanced = false
	case optComputeLosoStability:
		m.behaviorComputeLosoStability = !m.behaviorComputeLosoStability
		m.useDefaultAdvanced = false
	case optComputeBayesFactors:
		m.behaviorComputeBayesFactors = !m.behaviorComputeBayesFactors
		m.useDefaultAdvanced = false
	case optBehaviorValidateOnly:
		m.behaviorValidateOnly = !m.behaviorValidateOnly
		m.useDefaultAdvanced = false

	// Trial table / residual options
	case optTrialTableFormat:
		m.trialTableFormat = (m.trialTableFormat + 1) % 2
		m.useDefaultAdvanced = false
	case optTrialTableAddLagFeatures:
		m.trialTableAddLagFeatures = !m.trialTableAddLagFeatures
		m.useDefaultAdvanced = false
	case optTrialOrderMaxMissingFraction:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFeatureSummariesEnabled:
		m.featureSummariesEnabled = !m.featureSummariesEnabled
		m.useDefaultAdvanced = false
	case optFeatureQCEnabled:
		m.featureQCEnabled = !m.featureQCEnabled
		m.useDefaultAdvanced = false
	case optFeatureQCMaxMissingPct, optFeatureQCMinVariance:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFeatureQCCheckWithinRunVariance:
		m.featureQCCheckWithinRunVariance = !m.featureQCCheckWithinRunVariance
		m.useDefaultAdvanced = false
	case optPainResidualEnabled:
		m.painResidualEnabled = !m.painResidualEnabled
		m.useDefaultAdvanced = false
	case optPainResidualMethod:
		m.painResidualMethod = (m.painResidualMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optPainResidualPolyDegree:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualSplineDfCandidates:
		m.startTextEdit(textFieldPainResidualSplineDfCandidates)
		m.useDefaultAdvanced = false
	case optPainResidualModelCompare:
		m.painResidualModelCompareEnabled = !m.painResidualModelCompareEnabled
		m.useDefaultAdvanced = false
	case optPainResidualModelComparePolyDegrees:
		m.startTextEdit(textFieldPainResidualModelComparePolyDegrees)
		m.useDefaultAdvanced = false
	case optPainResidualMinSamples, optPainResidualModelCompareMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualBreakpoint:
		m.painResidualBreakpointEnabled = !m.painResidualBreakpointEnabled
		m.useDefaultAdvanced = false
	case optPainResidualBreakpointCandidates, optPainResidualBreakpointMinSamples, optPainResidualBreakpointQlow, optPainResidualBreakpointQhigh:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitEnabled:
		m.painResidualCrossfitEnabled = !m.painResidualCrossfitEnabled
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitGroupColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedPainResidualCrossfitGroupColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPainResidualCrossfitGroupColumn)
		}
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitNSplits:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitMethod:
		m.painResidualCrossfitMethod = (m.painResidualCrossfitMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitSplineKnots:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Regression
	case optRegressionOutcome:
		m.regressionOutcome = (m.regressionOutcome + 1) % 3
		m.useDefaultAdvanced = false
	case optRegressionIncludeTemperature:
		m.regressionIncludeTemperature = !m.regressionIncludeTemperature
		m.useDefaultAdvanced = false
	case optRegressionTempControl:
		m.regressionTempControl = (m.regressionTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optRegressionTempSplineKnots, optRegressionTempSplineQlow, optRegressionTempSplineQhigh, optRegressionTempSplineMinN:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRegressionIncludeTrialOrder:
		m.regressionIncludeTrialOrder = !m.regressionIncludeTrialOrder
		m.useDefaultAdvanced = false
	case optRegressionIncludePrev:
		m.regressionIncludePrev = !m.regressionIncludePrev
		m.useDefaultAdvanced = false
	case optRegressionIncludeRunBlock:
		m.regressionIncludeRunBlock = !m.regressionIncludeRunBlock
		m.useDefaultAdvanced = false
	case optRegressionIncludeInteraction:
		m.regressionIncludeInteraction = !m.regressionIncludeInteraction
		m.useDefaultAdvanced = false
	case optRegressionStandardize:
		m.regressionStandardize = !m.regressionStandardize
		m.useDefaultAdvanced = false
	case optRegressionMinSamples, optRegressionPermutations, optRegressionMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRegressionPrimaryUnit:
		m.regressionPrimaryUnit = (m.regressionPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false

	// Models
	case optModelsIncludeTemperature:
		m.modelsIncludeTemperature = !m.modelsIncludeTemperature
		m.useDefaultAdvanced = false
	case optModelsTempControl:
		m.modelsTempControl = (m.modelsTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optModelsTempSplineKnots, optModelsTempSplineQlow, optModelsTempSplineQhigh, optModelsTempSplineMinN:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optModelsIncludeTrialOrder:
		m.modelsIncludeTrialOrder = !m.modelsIncludeTrialOrder
		m.useDefaultAdvanced = false
	case optModelsIncludePrev:
		m.modelsIncludePrev = !m.modelsIncludePrev
		m.useDefaultAdvanced = false
	case optModelsIncludeRunBlock:
		m.modelsIncludeRunBlock = !m.modelsIncludeRunBlock
		m.useDefaultAdvanced = false
	case optModelsIncludeInteraction:
		m.modelsIncludeInteraction = !m.modelsIncludeInteraction
		m.useDefaultAdvanced = false
	case optModelsStandardize:
		m.modelsStandardize = !m.modelsStandardize
		m.useDefaultAdvanced = false
	case optModelsMinSamples, optModelsMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optModelsOutcomeRating:
		m.modelsOutcomeRating = !m.modelsOutcomeRating
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomeRating = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomePainResidual:
		m.modelsOutcomePainResidual = !m.modelsOutcomePainResidual
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomePainResidual = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomeTemperature:
		m.modelsOutcomeTemperature = !m.modelsOutcomeTemperature
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomeTemperature = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomePainBinary:
		m.modelsOutcomePainBinary = !m.modelsOutcomePainBinary
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomePainBinary = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyOLS:
		m.modelsFamilyOLS = !m.modelsFamilyOLS
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyOLS = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyRobust:
		m.modelsFamilyRobust = !m.modelsFamilyRobust
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyRobust = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyQuantile:
		m.modelsFamilyQuantile = !m.modelsFamilyQuantile
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyQuantile = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyLogit:
		m.modelsFamilyLogit = !m.modelsFamilyLogit
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyLogit = true
		}
		m.useDefaultAdvanced = false
	case optModelsBinaryOutcome:
		m.modelsBinaryOutcome = (m.modelsBinaryOutcome + 1) % 2
		m.useDefaultAdvanced = false
	case optModelsPrimaryUnit:
		m.modelsPrimaryUnit = (m.modelsPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optModelsForceTrialIIDAsymptotic:
		m.modelsForceTrialIIDAsymptotic = !m.modelsForceTrialIIDAsymptotic
		m.useDefaultAdvanced = false

	// Stability
	case optStabilityMethod:
		m.stabilityMethod = (m.stabilityMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optStabilityOutcome:
		m.stabilityOutcome = (m.stabilityOutcome + 1) % 3
		m.useDefaultAdvanced = false
	case optStabilityGroupColumn:
		m.expandedOption = expandedStabilityGroupColumn
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optStabilityPartialTemp:
		m.stabilityPartialTemp = !m.stabilityPartialTemp
		m.useDefaultAdvanced = false
	case optStabilityMinGroupTrials, optStabilityMaxFeatures, optStabilityAlpha:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Consistency
	case optConsistencyEnabled:
		m.consistencyEnabled = !m.consistencyEnabled
		m.useDefaultAdvanced = false

	// Influence
	case optInfluenceOutcomeRating:
		m.influenceOutcomeRating = !m.influenceOutcomeRating
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual && !m.influenceOutcomeTemperature {
			m.influenceOutcomeRating = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceOutcomePainResidual:
		m.influenceOutcomePainResidual = !m.influenceOutcomePainResidual
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual && !m.influenceOutcomeTemperature {
			m.influenceOutcomePainResidual = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceOutcomeTemperature:
		m.influenceOutcomeTemperature = !m.influenceOutcomeTemperature
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual && !m.influenceOutcomeTemperature {
			m.influenceOutcomeTemperature = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optInfluenceIncludeTemperature:
		m.influenceIncludeTemperature = !m.influenceIncludeTemperature
		m.useDefaultAdvanced = false
	case optInfluenceTempControl:
		m.influenceTempControl = (m.influenceTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optInfluenceTempSplineKnots, optInfluenceTempSplineQlow, optInfluenceTempSplineQhigh, optInfluenceTempSplineMinN:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optInfluenceIncludeTrialOrder:
		m.influenceIncludeTrialOrder = !m.influenceIncludeTrialOrder
		m.useDefaultAdvanced = false
	case optInfluenceIncludeRunBlock:
		m.influenceIncludeRunBlock = !m.influenceIncludeRunBlock
		m.useDefaultAdvanced = false
	case optInfluenceIncludeInteraction:
		m.influenceIncludeInteraction = !m.influenceIncludeInteraction
		m.useDefaultAdvanced = false
	case optInfluenceStandardize:
		m.influenceStandardize = !m.influenceStandardize
		m.useDefaultAdvanced = false
	case optInfluenceCooksThreshold, optInfluenceLeverageThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Report
	case optReportTopN:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Correlations
	case optCorrelationsUseCrossfitPainResidual:
		m.correlationsUseCrossfitResidual = !m.correlationsUseCrossfitResidual
		m.useDefaultAdvanced = false
	case optCorrelationsPrimaryUnit:
		m.correlationsPrimaryUnit = (m.correlationsPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optCorrelationsMinRuns:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optCorrelationsPreferPainResidual:
		m.correlationsPreferPainResidual = !m.correlationsPreferPainResidual
		m.useDefaultAdvanced = false
	case optCorrelationsPermutations:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optCorrelationsPermutationPrimary:
		m.correlationsPermutationPrimary = !m.correlationsPermutationPrimary
		m.useDefaultAdvanced = false
	case optCorrelationsTargetColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedCorrelationsTargetColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldCorrelationsTargetColumn)
		}
		m.useDefaultAdvanced = false
	case optCorrelationsTypes:
		m.startTextEdit(textFieldCorrelationsTypes)
		m.useDefaultAdvanced = false
	case optCorrelationsFeatures:
		m.startTextEdit(textFieldCorrelationsFeatures)
		m.useDefaultAdvanced = false
	case optCorrelationsMultilevel:
		// Toggle multilevel_correlations computation
		for i, comp := range m.computations {
			if comp.Key == "multilevel_correlations" {
				m.computationSelected[i] = !m.computationSelected[i]
				break
			}
		}
		m.useDefaultAdvanced = false
	case optGroupLevelBlockPermutation:
		m.groupLevelBlockPermutation = !m.groupLevelBlockPermutation
		m.useDefaultAdvanced = false
	case optGroupLevelTarget:
		if len(m.availableGroupLevelTargets()) > 0 {
			m.expandedOption = expandedGroupLevelTarget
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldGroupLevelTarget)
		}
		m.useDefaultAdvanced = false
	case optGroupLevelControlTemperature:
		m.groupLevelControlTemperature = !m.groupLevelControlTemperature
		m.useDefaultAdvanced = false
	case optGroupLevelControlTrialOrder:
		m.groupLevelControlTrialOrder = !m.groupLevelControlTrialOrder
		m.useDefaultAdvanced = false
	case optGroupLevelControlRunEffects:
		m.groupLevelControlRunEffects = !m.groupLevelControlRunEffects
		m.useDefaultAdvanced = false
	case optGroupLevelMaxRunDummies:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optGroupLevelAllowParametricFallback:
		m.groupLevelAllowParametricFallback = !m.groupLevelAllowParametricFallback
		m.useDefaultAdvanced = false

	// Temporal
	case optTemporalResolutionMs, optTemporalTimeMinMs, optTemporalTimeMaxMs, optTemporalSmoothMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optTemporalCorrectionMethod:
		m.temporalCorrectionMethod = (m.temporalCorrectionMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optTemporalTargetColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedTemporalTargetColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldTemporalTargetColumn)
		}
		m.useDefaultAdvanced = false
	case optTemporalFeatures:
		m.startTextEdit(textFieldTemporalFeatures)
		m.useDefaultAdvanced = false
	case optTemporalSplitByCondition:
		m.temporalSplitByCondition = !m.temporalSplitByCondition
		m.useDefaultAdvanced = false
	case optTemporalConditionColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedTemporalConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldTemporalConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optTemporalConditionValues:
		if m.temporalConditionColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.temporalConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedTemporalConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldTemporalConditionValues)
		}
		m.useDefaultAdvanced = false
	case optTemporalIncludeROIAverages:
		m.temporalIncludeROIAverages = !m.temporalIncludeROIAverages
		m.useDefaultAdvanced = false
	case optTemporalIncludeTFGrid:
		m.temporalIncludeTFGrid = !m.temporalIncludeTFGrid
		m.useDefaultAdvanced = false
	// Temporal feature selection
	case optTemporalFeaturePower:
		m.temporalFeaturePowerEnabled = !m.temporalFeaturePowerEnabled
		if !m.temporalFeaturePowerEnabled && !m.temporalFeatureITPCEnabled && !m.temporalFeatureERDSEnabled {
			m.temporalFeaturePowerEnabled = true
		}
		m.useDefaultAdvanced = false
	case optTemporalFeatureITPC:
		m.temporalFeatureITPCEnabled = !m.temporalFeatureITPCEnabled
		if !m.temporalFeaturePowerEnabled && !m.temporalFeatureITPCEnabled && !m.temporalFeatureERDSEnabled {
			m.temporalFeatureITPCEnabled = true
		}
		m.useDefaultAdvanced = false
	case optTemporalFeatureERDS:
		m.temporalFeatureERDSEnabled = !m.temporalFeatureERDSEnabled
		if !m.temporalFeaturePowerEnabled && !m.temporalFeatureITPCEnabled && !m.temporalFeatureERDSEnabled {
			m.temporalFeatureERDSEnabled = true
		}
		m.useDefaultAdvanced = false
	// ITPC-specific options
	case optTemporalITPCBaselineCorrection:
		m.temporalITPCBaselineCorrection = !m.temporalITPCBaselineCorrection
		m.useDefaultAdvanced = false
	case optTemporalITPCBaselineMin, optTemporalITPCBaselineMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ERDS-specific options
	case optTemporalERDSBaselineMin, optTemporalERDSBaselineMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optTemporalERDSMethod:
		m.temporalERDSMethod = (m.temporalERDSMethod + 1) % 2 // Toggle between percent and zscore
		m.useDefaultAdvanced = false
	// TF Heatmap options
	case optTemporalTfHeatmapEnabled:
		m.tfHeatmapEnabled = !m.tfHeatmapEnabled
		m.useDefaultAdvanced = false
	case optTemporalTfHeatmapFreqs:
		m.startTextEdit(textFieldTfHeatmapFreqs)
		m.useDefaultAdvanced = false
	case optTemporalTfHeatmapTimeResMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	case optClusterThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optClusterMinSize:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optClusterTail:
		switch m.clusterTail {
		case 0:
			m.clusterTail = 1
		case 1:
			m.clusterTail = -1
		case -1:
			m.clusterTail = 0
		default:
			m.clusterTail = 0
		}
		m.useDefaultAdvanced = false
	case optClusterFeatures:
		m.startTextEdit(textFieldClusterFeatures)
		m.useDefaultAdvanced = false
	case optClusterConditionColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedClusterConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldClusterConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optClusterConditionValues:
		if m.clusterConditionColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.clusterConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedClusterConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldClusterConditionValues)
		}
		m.useDefaultAdvanced = false
	// Mediation options
	case optMediationBootstrap, optMediationMinEffect, optMediationPermutations:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMediationPermutationPrimary:
		m.mediationPermutationPrimary = !m.mediationPermutationPrimary
		m.useDefaultAdvanced = false
	case optMediationFeatures:
		m.startTextEdit(textFieldMediationFeatures)
		m.useDefaultAdvanced = false
	case optMediationMaxMediatorsEnabled:
		m.mediationMaxMediatorsEnabled = !m.mediationMaxMediatorsEnabled
		m.useDefaultAdvanced = false
	case optMediationMaxMediators:
		if m.mediationMaxMediatorsEnabled {
			m.startNumberEdit()
			m.useDefaultAdvanced = false
		}
	// Moderation options
	case optModerationMaxFeaturesEnabled:
		m.moderationMaxFeaturesEnabled = !m.moderationMaxFeaturesEnabled
		m.useDefaultAdvanced = false
	case optModerationMaxFeatures:
		if m.moderationMaxFeaturesEnabled {
			m.startNumberEdit()
			m.useDefaultAdvanced = false
		}
	case optModerationMinSamples, optModerationPermutations:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optModerationPermutationPrimary:
		m.moderationPermutationPrimary = !m.moderationPermutationPrimary
		m.useDefaultAdvanced = false
	case optModerationFeatures:
		m.startTextEdit(textFieldModerationFeatures)
		m.useDefaultAdvanced = false
	// Mixed effects options
	case optMixedMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMixedIncludeTemperature:
		m.mixedIncludeTemperature = !m.mixedIncludeTemperature
		m.useDefaultAdvanced = false
	case optMixedEffectsType:
		m.mixedEffectsType = (m.mixedEffectsType + 1) % 2
		m.useDefaultAdvanced = false
	// Condition options
	case optConditionCompareColumn:
		if len(m.GetAvailableColumns()) > 0 {
			m.expandedOption = expandedConditionCompareColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConditionCompareColumn)
		}
		m.useDefaultAdvanced = false
	case optConditionCompareWindows:
		if len(m.availableWindows) > 0 {
			m.expandedOption = expandedConditionCompareWindows
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConditionCompareWindows)
		}
		m.useDefaultAdvanced = false
	case optConditionFeatures:
		m.startTextEdit(textFieldConditionFeatures)
		m.useDefaultAdvanced = false
	case optConditionCompareLabels:
		m.startTextEdit(textFieldConditionCompareLabels)
		m.useDefaultAdvanced = false
	case optConditionCompareValues:
		if m.conditionCompareColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.conditionCompareColumn); len(vals) > 0 {
			m.expandedOption = expandedConditionCompareValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConditionCompareValues)
		}
		m.useDefaultAdvanced = false
	case optConditionMinTrials:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainSensitivityMinTrials:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainSensitivityPrimaryUnit:
		m.painSensitivityPrimaryUnit = (m.painSensitivityPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optPainSensitivityPermutations:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainSensitivityPermutationPrimary:
		m.painSensitivityPermutationPrimary = !m.painSensitivityPermutationPrimary
		m.useDefaultAdvanced = false
	case optPainSensitivityFeatures:
		m.startTextEdit(textFieldPainSensitivityFeatures)
		m.useDefaultAdvanced = false
	case optConditionPrimaryUnit:
		m.conditionPrimaryUnit = (m.conditionPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optConditionWindowPrimaryUnit:
		m.conditionWindowPrimaryUnit = (m.conditionWindowPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optConditionWindowMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConditionPermutationPrimary:
		m.conditionPermutationPrimary = !m.conditionPermutationPrimary
		m.useDefaultAdvanced = false
	case optConditionEffectThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConditionFailFast:
		m.conditionFailFast = !m.conditionFailFast
		m.useDefaultAdvanced = false
	case optConditionOverwrite:
		m.conditionOverwrite = !m.conditionOverwrite
		m.useDefaultAdvanced = false

	// Output section
	case optBehaviorGroupOutput:
		m.behaviorGroupOutputExpanded = !m.behaviorGroupOutputExpanded
		m.useDefaultAdvanced = false
	case optAlsoSaveCsv:
		m.alsoSaveCsv = !m.alsoSaveCsv
		m.useDefaultAdvanced = false
	case optBehaviorOverwrite:
		m.behaviorOverwrite = !m.behaviorOverwrite
		m.useDefaultAdvanced = false

	// Behavior Statistics
	case optBehaviorStatsTempControl:
		m.behaviorStatsTempControl = (m.behaviorStatsTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optBehaviorStatsAllowIIDTrials:
		m.behaviorStatsAllowIIDTrials = !m.behaviorStatsAllowIIDTrials
		m.useDefaultAdvanced = false
	case optBehaviorStatsHierarchicalFDR:
		m.behaviorStatsHierarchicalFDR = !m.behaviorStatsHierarchicalFDR
		m.useDefaultAdvanced = false
	case optBehaviorStatsComputeReliability:
		m.behaviorStatsComputeReliability = !m.behaviorStatsComputeReliability
		m.useDefaultAdvanced = false
	case optBehaviorPermScheme:
		m.behaviorPermScheme = (m.behaviorPermScheme + 1) % 2
		m.useDefaultAdvanced = false
	case optBehaviorPermGroupColumnPreference:
		m.startTextEdit(textFieldBehaviorPermGroupColumnPreference)
		m.useDefaultAdvanced = false
	case optBehaviorExcludeNonTrialwiseFeatures:
		m.behaviorExcludeNonTrialwiseFeatures = !m.behaviorExcludeNonTrialwiseFeatures
		m.useDefaultAdvanced = false
	// Global Statistics & Validation
	case optGlobalNBootstrap:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optClusterCorrectionEnabled:
		m.clusterCorrectionEnabled = !m.clusterCorrectionEnabled
		m.useDefaultAdvanced = false
	case optClusterCorrectionAlpha, optClusterCorrectionMinClusterSize:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optClusterCorrectionTail:
		m.clusterCorrectionTailGlobal = (m.clusterCorrectionTailGlobal + 1) % 3
		m.useDefaultAdvanced = false
	case optValidationMinEpochs, optValidationMinChannels, optValidationMaxAmplitudeUv:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// System / IO
	case optIOTemperatureRange:
		m.startTextEdit(textFieldIOTemperatureRange)
		m.useDefaultAdvanced = false
	case optIOMaxMissingChannelsFraction:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	}

	options = m.getBehaviorOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
}
