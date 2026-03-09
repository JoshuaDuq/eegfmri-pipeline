package wizard

// fMRI preprocessing and analysis advanced option builders.

func (m Model) getFmriPreprocessingOptions() []optionType {
	options := []optionType{optUseDefaults, optConfigSetOverrides}

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
	options := []optionType{optUseDefaults, optConfigSetOverrides}

	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}
	if mode == "second-level" {
		options = append(options, optFmriSecondLevelGroupInput)
		if m.fmriSecondLevelGroupInputExpanded {
			options = append(options,
				optFmriSecondLevelInputRoot,
				optFmriSecondLevelContrastNames,
				optFmriSecondLevelConditionLabels,
			)
		}

		options = append(options, optFmriSecondLevelGroupDesign)
		if m.fmriSecondLevelGroupDesignExpanded {
			options = append(options,
				optFmriSecondLevelModel,
				optFmriSecondLevelFormula,
				optFmriSecondLevelCovariatesFile,
				optFmriSecondLevelSubjectColumn,
			)
			switch m.fmriSecondLevelModelIndex % 4 {
			case 1: // two-sample
				options = append(options,
					optFmriSecondLevelCovariateColumns,
					optFmriSecondLevelGroupColumn,
					optFmriSecondLevelGroupAValue,
					optFmriSecondLevelGroupBValue,
				)
			case 2: // paired
				options = append(options, optFmriSecondLevelCovariateColumns)
			case 3: // repeated-measures
				// repeated-measures currently omits subject-level covariates
			default: // one-sample
				options = append(options, optFmriSecondLevelCovariateColumns)
			}
		}

		options = append(options, optFmriSecondLevelGroupInference)
		if m.fmriSecondLevelGroupInferExpanded {
			options = append(options,
				optFmriSecondLevelWriteDesignMatrix,
				optFmriSecondLevelPermutationEnabled,
			)
			if m.fmriSecondLevelPermutationEnabled {
				options = append(options,
					optFmriSecondLevelPermutationCount,
					optFmriSecondLevelTwoSided,
				)
			}
		}

		options = append(options, optFmriSecondLevelGroupOutput)
		if m.fmriSecondLevelGroupOutputExpanded {
			options = append(options,
				optFmriSecondLevelOutputName,
				optFmriSecondLevelOutputDir,
			)
		}

		return options
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
			options = append(
				options,
				optFmriAnalysisEventsToModel,
				optFmriAnalysisEventsToModelColumn,
				optFmriAnalysisScopeColumn,
				optFmriAnalysisScopeTrialTypes,
				optFmriAnalysisPhaseColumn,
				optFmriAnalysisPhaseScopeColumn,
				optFmriAnalysisPhaseScopeValue,
				optFmriAnalysisStimPhasesToModel,
			)
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
				optFmriAnalysisSignatureMaps,
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
				optFmriTrialSigSignatureOption1,
				optFmriTrialSigSignatureOption2,
				optFmriAnalysisSignatureDir,
				optFmriAnalysisSignatureMaps,
				optFmriTrialSigScopeTrialTypeColumn,
				optFmriTrialSigScopePhaseColumn,
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
