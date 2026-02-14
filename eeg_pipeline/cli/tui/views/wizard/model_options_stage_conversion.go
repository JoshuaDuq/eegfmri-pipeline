package wizard

// Raw/BIDS conversion and merge utility advanced option builders.

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

// getMergePsychopyOptions returns advanced options for merge-psychopy
func (m Model) getMergePsychopyOptions() []optionType {
	return []optionType{
		optUseDefaults,
		optMergeEventPrefixes,
		optMergeEventTypes,
		optMergeQCColumns,
	}
}
