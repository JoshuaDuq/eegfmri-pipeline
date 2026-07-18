package wizard

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestApplyConfigKeys_HydratesFeatureRestFlagFromFeatureConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"preprocessing.task_is_rest":       false,
		"feature_engineering.task_is_rest": true,
	})

	if !m.prepTaskIsRest {
		t.Fatalf("expected feature pipeline to hydrate task_is_rest from feature_engineering")
	}
}

func TestApplyConfigKeys_HydratesFeatureEngineeringDefaultsFromConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"feature_engineering.analysis_mode":                     "trial_ml_safe",
		"feature_engineering.constants.min_epochs_for_features": float64(14),
		"feature_engineering.compute_change_scores":             false,
		"feature_engineering.save_tfr_with_sidecar":             true,
		"feature_engineering.output.also_save_csv":              true,
		"feature_engineering.feature_categories":                []interface{}{"power", "connectivity"},
		"feature_engineering.spatial_modes":                     []interface{}{"roi", "channels"},
	})

	if m.featAnalysisMode != 1 {
		t.Fatalf("expected featAnalysisMode=1 (trial_ml_safe), got %d", m.featAnalysisMode)
	}
	if m.minEpochsForFeatures != 14 {
		t.Fatalf("expected minEpochsForFeatures=14, got %d", m.minEpochsForFeatures)
	}
	if m.featComputeChangeScores {
		t.Fatalf("expected featComputeChangeScores=false")
	}
	if !m.featSaveTfrWithSidecar {
		t.Fatalf("expected featSaveTfrWithSidecar=true")
	}
	if !m.featAlsoSaveCsv {
		t.Fatalf("expected featAlsoSaveCsv=true")
	}

	if got := m.SelectedCategories(); len(got) != 2 || got[0] != "connectivity" || got[1] != "power" {
		t.Fatalf("expected selected feature categories [connectivity power], got %#v", got)
	}
	if got := m.SelectedSpatialModes(); len(got) != 2 || got[0] != "roi" || got[1] != "channels" {
		t.Fatalf("expected selected spatial modes [roi channels], got %#v", got)
	}
}

func TestApplyConfigKeys_HydratesPreprocessingRestFlagFromPreprocessingConfig(t *testing.T) {
	m := New(types.PipelinePreprocessing, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"preprocessing.task_is_rest":       true,
		"feature_engineering.task_is_rest": false,
	})

	if !m.prepTaskIsRest {
		t.Fatalf("expected preprocessing pipeline to hydrate task_is_rest from preprocessing")
	}
}

func TestApplyConfigKeys_HydratesBidsRestRoot(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"paths.bids_rest_root":  "/data/bids/rest",
		"paths.deriv_rest_root": "/data/derivatives/rest",
	})

	if m.bidsRestRoot != "/data/bids/rest" {
		t.Fatalf("expected bidsRestRoot='/data/bids/rest', got %q", m.bidsRestRoot)
	}
	if m.derivRestRoot != "/data/derivatives/rest" {
		t.Fatalf("expected derivRestRoot='/data/derivatives/rest', got %q", m.derivRestRoot)
	}
}

func TestApplyConfigKeys_HydratesSourceLocFmriContrastConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	values := map[string]interface{}{
		"feature_engineering.sourcelocalization.mode":                                 "fmri_informed",
		"feature_engineering.sourcelocalization.fmri.enabled":                         true,
		"feature_engineering.sourcelocalization.fmri.contrast.type":                   "custom",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_a.column":     "event_group",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_b.column":     "group_label",
		"feature_engineering.sourcelocalization.fmri.contrast.events_to_model":        []interface{}{"stim", "rating"},
		"feature_engineering.sourcelocalization.fmri.contrast.events_to_model_column": "event_class",
		"feature_engineering.sourcelocalization.fmri.contrast.phase_scope_column":     "phase_scope_col",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_scope_column": "scope_col",
		"feature_engineering.sourcelocalization.fmri.contrast.stim_phases_to_model":   []interface{}{"ramp", "plateau"},
		"feature_engineering.sourcelocalization.fmri.contrast.condition_scope_trial_types": []interface{}{
			"stim",
			"outcome",
		},
	}

	m.ApplyConfigKeys(values)

	if m.sourceLocMode != 1 {
		t.Fatalf("expected sourceLocMode=1 (fmri_informed), got %d", m.sourceLocMode)
	}
	if !m.sourceLocFmriEnabled {
		t.Fatalf("expected sourceLocFmriEnabled=true")
	}
	if m.sourceLocFmriContrastType != 1 {
		t.Fatalf("expected sourceLocFmriContrastType=1 (custom), got %d", m.sourceLocFmriContrastType)
	}
	if m.sourceLocFmriCondAColumn != "event_group" {
		t.Fatalf("expected condition A column event_group, got %q", m.sourceLocFmriCondAColumn)
	}
	if m.sourceLocFmriCondBColumn != "group_label" {
		t.Fatalf("expected condition B column group_label, got %q", m.sourceLocFmriCondBColumn)
	}
	if m.sourceLocFmriPhaseScopeColumn != "phase_scope_col" {
		t.Fatalf("expected phase scope column phase_scope_col, got %q", m.sourceLocFmriPhaseScopeColumn)
	}
	if m.sourceLocFmriConditionScopeColumn != "scope_col" {
		t.Fatalf("expected condition scope column scope_col, got %q", m.sourceLocFmriConditionScopeColumn)
	}
	if m.sourceLocFmriEventsToModel != "stim rating" {
		t.Fatalf("expected events_to_model 'stim rating', got %q", m.sourceLocFmriEventsToModel)
	}
	if m.sourceLocFmriEventsToModelColumn != "event_class" {
		t.Fatalf("expected events_to_model_column event_class, got %q", m.sourceLocFmriEventsToModelColumn)
	}
	if m.sourceLocFmriStimPhasesToModel != "ramp plateau" {
		t.Fatalf("expected stim phases 'ramp plateau', got %q", m.sourceLocFmriStimPhasesToModel)
	}
	if m.sourceLocFmriConditionScopeTrialTypes != "stim outcome" {
		t.Fatalf(
			"expected condition scope values 'stim outcome', got %q",
			m.sourceLocFmriConditionScopeTrialTypes,
		)
	}
}

func TestApplyConfigKeys_HydratesSourceLocFmriThresholdingConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"feature_engineering.sourcelocalization.fmri.thresholding.mode":  "fdr",
		"feature_engineering.sourcelocalization.fmri.thresholding.fdr_q": 0.025,
	})

	if m.sourceLocFmriThresholdMode != 1 {
		t.Fatalf("expected sourceLocFmriThresholdMode=1 (fdr), got %d", m.sourceLocFmriThresholdMode)
	}
	if m.sourceLocFmriFdrQ != 0.025 {
		t.Fatalf("expected sourceLocFmriFdrQ=0.025, got %v", m.sourceLocFmriFdrQ)
	}
}

func TestApplyConfigKeys_HydratesBehaviorClusterCorrectionConfig(t *testing.T) {
	m := New(types.PipelineBehavior, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"behavior_analysis.cluster_correction.n_permutations":            2500,
		"behavior_analysis.cluster_correction.alpha":                     0.01,
		"behavior_analysis.cluster_correction.cluster_forming_threshold": 0.025,
		"behavior_analysis.cluster_correction.min_timepoints":            4,
		"behavior_analysis.cluster_correction.min_channels":              2,
		"behavior_analysis.cluster_correction.min_cluster_size":          3,
		"behavior_analysis.cluster_correction.tail":                      -1,
	})

	if m.clusterCorrectionNPermutations != 2500 {
		t.Fatalf("expected clusterCorrectionNPermutations=2500, got %d", m.clusterCorrectionNPermutations)
	}
	if m.clusterCorrectionAlpha != 0.01 {
		t.Fatalf("expected clusterCorrectionAlpha=0.01, got %v", m.clusterCorrectionAlpha)
	}
	if m.clusterCorrectionFormingThreshold != 0.025 {
		t.Fatalf("expected clusterCorrectionFormingThreshold=0.025, got %v", m.clusterCorrectionFormingThreshold)
	}
	if m.clusterCorrectionMinTimepoints != 4 {
		t.Fatalf("expected clusterCorrectionMinTimepoints=4, got %d", m.clusterCorrectionMinTimepoints)
	}
	if m.clusterCorrectionMinChannels != 2 {
		t.Fatalf("expected clusterCorrectionMinChannels=2, got %d", m.clusterCorrectionMinChannels)
	}
	if m.clusterCorrectionMinClusterSize != 3 {
		t.Fatalf("expected clusterCorrectionMinClusterSize=3, got %d", m.clusterCorrectionMinClusterSize)
	}
	if m.clusterCorrectionTail != -1 {
		t.Fatalf("expected clusterCorrectionTail=-1, got %d", m.clusterCorrectionTail)
	}
}

func TestApplyConfigKeys_HydratesFmriThreadConfig(t *testing.T) {
	m := New(types.PipelineFmri, ".")
	values := map[string]interface{}{
		"fmri_preprocessing.fmriprep.nthreads":     12,
		"fmri_preprocessing.fmriprep.omp_nthreads": float64(6),
	}

	m.ApplyConfigKeys(values)

	if m.fmriNThreads != 12 {
		t.Fatalf("expected fmriNThreads=12, got %d", m.fmriNThreads)
	}
	if m.fmriOmpNThreads != 6 {
		t.Fatalf("expected fmriOmpNThreads=6, got %d", m.fmriOmpNThreads)
	}
}

func TestApplyConfigKeys_HydratesFmriPathsAndPreprocessingConfig(t *testing.T) {
	m := New(types.PipelineFmri, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"paths.signature_dir": "/external/signatures",
		"paths.signature_maps": []interface{}{
			map[string]interface{}{"name": "SIG_A", "path": "maps/sig_a.nii.gz"},
			map[string]interface{}{"name": "SIG_B", "path": "maps/sig_b.nii.gz"},
		},
		"fmri_preprocessing.fmriprep.level":                 "resampling",
		"fmri_preprocessing.fmriprep.cifti_output":          "170k",
		"fmri_preprocessing.fmriprep.task_id":               "pain",
		"fmri_preprocessing.fmriprep.low_mem":               true,
		"fmri_preprocessing.fmriprep.longitudinal":          true,
		"fmri_preprocessing.fmriprep.skull_strip_template":  "NKI",
		"fmri_preprocessing.fmriprep.bold2t1w_init":         "header",
		"fmri_preprocessing.fmriprep.bold2t1w_dof":          9,
		"fmri_preprocessing.fmriprep.slice_time_ref":        0.25,
		"fmri_preprocessing.fmriprep.fd_spike_threshold":    0.6,
		"fmri_preprocessing.fmriprep.dvars_spike_threshold": 1.7,
		"fmri_preprocessing.fmriprep.medial_surface_nan":    true,
		"fmri_preprocessing.fmriprep.no_msm":                true,
		"fmri_preprocessing.fmriprep.me_output_echos":       true,
		"fmri_preprocessing.fmriprep.random_seed":           123,
	})

	if m.fmriAnalysisSignatureDir != "/external/signatures" {
		t.Fatalf("expected signature dir to hydrate, got %q", m.fmriAnalysisSignatureDir)
	}
	if m.fmriAnalysisSignatureMaps != "SIG_A:maps/sig_a.nii.gz SIG_B:maps/sig_b.nii.gz" {
		t.Fatalf("expected signature maps to hydrate, got %q", m.fmriAnalysisSignatureMaps)
	}
	if m.fmriLevelIndex != 1 {
		t.Fatalf("expected fmriLevelIndex=1 (resampling), got %d", m.fmriLevelIndex)
	}
	if m.fmriCiftiOutputIndex != 2 {
		t.Fatalf("expected fmriCiftiOutputIndex=2 (170k), got %d", m.fmriCiftiOutputIndex)
	}
	if m.fmriTaskId != "pain" {
		t.Fatalf("expected fmriTaskId='pain', got %q", m.fmriTaskId)
	}
	if !m.fmriLowMem {
		t.Fatalf("expected fmriLowMem=true")
	}
	if !m.fmriLongitudinal {
		t.Fatalf("expected fmriLongitudinal=true")
	}
	if m.fmriSkullStripTemplate != "NKI" {
		t.Fatalf("expected fmriSkullStripTemplate='NKI', got %q", m.fmriSkullStripTemplate)
	}
	if m.fmriBold2T1wInitIndex != 1 {
		t.Fatalf("expected fmriBold2T1wInitIndex=1 (header), got %d", m.fmriBold2T1wInitIndex)
	}
	if m.fmriBold2T1wDof != 9 {
		t.Fatalf("expected fmriBold2T1wDof=9, got %d", m.fmriBold2T1wDof)
	}
	if m.fmriSliceTimeRef != 0.25 {
		t.Fatalf("expected fmriSliceTimeRef=0.25, got %v", m.fmriSliceTimeRef)
	}
	if m.fmriFdSpikeThreshold != 0.6 {
		t.Fatalf("expected fmriFdSpikeThreshold=0.6, got %v", m.fmriFdSpikeThreshold)
	}
	if m.fmriDvarsSpikeThreshold != 1.7 {
		t.Fatalf("expected fmriDvarsSpikeThreshold=1.7, got %v", m.fmriDvarsSpikeThreshold)
	}
	if !m.fmriMedialSurfaceNan {
		t.Fatalf("expected fmriMedialSurfaceNan=true")
	}
	if !m.fmriNoMsm {
		t.Fatalf("expected fmriNoMsm=true")
	}
	if !m.fmriMeOutputEchos {
		t.Fatalf("expected fmriMeOutputEchos=true")
	}
	if m.fmriRandomSeed != 123 {
		t.Fatalf("expected fmriRandomSeed=123, got %d", m.fmriRandomSeed)
	}
}

func TestApplyConfigKeys_HydratesFmriAnalysisAndGroupConfig(t *testing.T) {
	m := New(types.PipelineFmri, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"fmri_contrast.input_source":                  "bids_raw",
		"fmri_contrast.fmriprep_space":                "MNI152NLin2009cAsym",
		"fmri_contrast.require_fmriprep":              false,
		"fmri_contrast.type":                          "custom",
		"fmri_contrast.condition_a.column":            "event_class",
		"fmri_contrast.condition_a.value":             "pain",
		"fmri_contrast.condition_b.column":            "event_class",
		"fmri_contrast.condition_b.value":             "rest",
		"fmri_contrast.condition_scope_trial_types":   []interface{}{"stimulation"},
		"fmri_contrast.condition_scope_column":        "trial_type",
		"fmri_contrast.events_to_model":               []interface{}{"stimulation", "rating"},
		"fmri_contrast.events_to_model_column":        "event_class",
		"fmri_contrast.phase_column":                  "stim_phase",
		"fmri_contrast.phase_scope_column":            "trial_type",
		"fmri_contrast.phase_scope_value":             "stimulation",
		"fmri_contrast.stim_phases_to_model":          []interface{}{"plateau"},
		"fmri_contrast.formula":                       "A - B",
		"fmri_contrast.name":                          "pain_vs_rest",
		"fmri_contrast.runs":                          []interface{}{1, 2},
		"fmri_contrast.hrf_model":                     "fir",
		"fmri_contrast.drift_model":                   "none",
		"fmri_contrast.high_pass_hz":                  0.01,
		"fmri_contrast.low_pass_hz":                   0.2,
		"fmri_contrast.smoothing_fwhm":                6.0,
		"fmri_contrast.confounds_strategy":            "motion24+wmcsf+fd",
		"fmri_contrast.write_design_matrix":           true,
		"fmri_contrast.output_type":                   "beta",
		"fmri_contrast.resample_to_freesurfer":        true,
		"fmri_contrast.output_dir":                    "/tmp/out",
		"fmri_contrast.freesurfer_dir":                "/tmp/fs",
		"fmri_group_level.model":                      "paired",
		"fmri_group_level.contrast_names":             []interface{}{"pain_low", "pain_high"},
		"fmri_group_level.condition_labels":           []interface{}{"Low", "High"},
		"fmri_group_level.input_root":                 "/tmp/input",
		"fmri_group_level.formula":                    "High - Low",
		"fmri_group_level.output_name":                "group",
		"fmri_group_level.output_dir":                 "/tmp/group",
		"fmri_group_level.write_design_matrix":        false,
		"fmri_group_level.covariates_file":            "/tmp/cov.tsv",
		"fmri_group_level.subject_column":             "sub",
		"fmri_group_level.covariate_columns":          []interface{}{"age", "sex"},
		"fmri_group_level.group_column":               "group",
		"fmri_group_level.group_a_value":              "control",
		"fmri_group_level.group_b_value":              "patient",
		"fmri_group_level.permutation.enabled":        true,
		"fmri_group_level.permutation.n_permutations": 1234,
		"fmri_group_level.permutation.two_sided":      false,
	})

	if m.fmriAnalysisInputSourceIndex != 1 {
		t.Fatalf("expected fmriAnalysisInputSourceIndex=1 (bids_raw), got %d", m.fmriAnalysisInputSourceIndex)
	}
	if m.fmriAnalysisFmriprepSpace != "MNI152NLin2009cAsym" {
		t.Fatalf("expected fmriAnalysisFmriprepSpace to hydrate, got %q", m.fmriAnalysisFmriprepSpace)
	}
	if m.fmriAnalysisRequireFmriprep {
		t.Fatalf("expected fmriAnalysisRequireFmriprep=false")
	}
	if m.fmriAnalysisContrastType != 1 {
		t.Fatalf("expected fmriAnalysisContrastType=1 (custom), got %d", m.fmriAnalysisContrastType)
	}
	if m.fmriAnalysisCondAColumn != "event_class" || m.fmriAnalysisCondAValue != "pain" {
		t.Fatalf("expected cond_a to hydrate, got %q / %q", m.fmriAnalysisCondAColumn, m.fmriAnalysisCondAValue)
	}
	if m.fmriAnalysisCondBColumn != "event_class" || m.fmriAnalysisCondBValue != "rest" {
		t.Fatalf("expected cond_b to hydrate, got %q / %q", m.fmriAnalysisCondBColumn, m.fmriAnalysisCondBValue)
	}
	if m.fmriAnalysisRunsSpec != "1 2" {
		t.Fatalf("expected runs spec to hydrate, got %q", m.fmriAnalysisRunsSpec)
	}
	if m.fmriAnalysisHrfModel != 2 {
		t.Fatalf("expected fmriAnalysisHrfModel=2 (fir), got %d", m.fmriAnalysisHrfModel)
	}
	if m.fmriAnalysisDriftModel != 0 {
		t.Fatalf("expected fmriAnalysisDriftModel=0 (none), got %d", m.fmriAnalysisDriftModel)
	}
	if m.fmriAnalysisSmoothingFwhm != 6.0 {
		t.Fatalf("expected fmriAnalysisSmoothingFwhm=6.0, got %v", m.fmriAnalysisSmoothingFwhm)
	}
	if m.fmriAnalysisConfoundsStrategy != 6 {
		t.Fatalf("expected fmriAnalysisConfoundsStrategy=6, got %d", m.fmriAnalysisConfoundsStrategy)
	}
	if !m.fmriAnalysisWriteDesignMatrix {
		t.Fatalf("expected fmriAnalysisWriteDesignMatrix=true")
	}
	if m.fmriAnalysisOutputType != 3 {
		t.Fatalf("expected fmriAnalysisOutputType=3 (beta), got %d", m.fmriAnalysisOutputType)
	}
	if !m.fmriAnalysisResampleToFS {
		t.Fatalf("expected fmriAnalysisResampleToFS=true")
	}
	if m.fmriAnalysisOutputDir != "/tmp/out" {
		t.Fatalf("expected fmriAnalysisOutputDir='/tmp/out', got %q", m.fmriAnalysisOutputDir)
	}
	if m.fmriAnalysisFreesurferDir != "/tmp/fs" {
		t.Fatalf("expected fmriAnalysisFreesurferDir='/tmp/fs', got %q", m.fmriAnalysisFreesurferDir)
	}

	if m.fmriSecondLevelModelIndex != 2 {
		t.Fatalf("expected fmriSecondLevelModelIndex=2 (paired), got %d", m.fmriSecondLevelModelIndex)
	}
	if m.fmriSecondLevelContrastNames != "pain_low pain_high" {
		t.Fatalf("expected fmriSecondLevelContrastNames to hydrate, got %q", m.fmriSecondLevelContrastNames)
	}
	if m.fmriSecondLevelConditionLabels != "Low High" {
		t.Fatalf("expected fmriSecondLevelConditionLabels to hydrate, got %q", m.fmriSecondLevelConditionLabels)
	}
	if m.fmriSecondLevelInputRoot != "/tmp/input" {
		t.Fatalf("expected fmriSecondLevelInputRoot='/tmp/input', got %q", m.fmriSecondLevelInputRoot)
	}
	if m.fmriSecondLevelFormula != "High - Low" {
		t.Fatalf("expected fmriSecondLevelFormula to hydrate, got %q", m.fmriSecondLevelFormula)
	}
	if m.fmriSecondLevelOutputName != "group" {
		t.Fatalf("expected fmriSecondLevelOutputName='group', got %q", m.fmriSecondLevelOutputName)
	}
	if m.fmriSecondLevelOutputDir != "/tmp/group" {
		t.Fatalf("expected fmriSecondLevelOutputDir='/tmp/group', got %q", m.fmriSecondLevelOutputDir)
	}
	if m.fmriSecondLevelWriteDesignMatrix {
		t.Fatalf("expected fmriSecondLevelWriteDesignMatrix=false")
	}
	if m.fmriSecondLevelCovariatesFile != "/tmp/cov.tsv" {
		t.Fatalf("expected fmriSecondLevelCovariatesFile='/tmp/cov.tsv', got %q", m.fmriSecondLevelCovariatesFile)
	}
	if m.fmriSecondLevelSubjectColumn != "sub" {
		t.Fatalf("expected fmriSecondLevelSubjectColumn='sub', got %q", m.fmriSecondLevelSubjectColumn)
	}
	if m.fmriSecondLevelCovariateColumns != "age sex" {
		t.Fatalf("expected fmriSecondLevelCovariateColumns='age sex', got %q", m.fmriSecondLevelCovariateColumns)
	}
	if m.fmriSecondLevelGroupColumn != "group" || m.fmriSecondLevelGroupAValue != "control" || m.fmriSecondLevelGroupBValue != "patient" {
		t.Fatalf(
			"expected two-sample group values to hydrate, got %q / %q / %q",
			m.fmriSecondLevelGroupColumn,
			m.fmriSecondLevelGroupAValue,
			m.fmriSecondLevelGroupBValue,
		)
	}
	if !m.fmriSecondLevelPermutationEnabled {
		t.Fatalf("expected fmriSecondLevelPermutationEnabled=true")
	}
	if m.fmriSecondLevelPermutationCount != 1234 {
		t.Fatalf("expected fmriSecondLevelPermutationCount=1234, got %d", m.fmriSecondLevelPermutationCount)
	}
	if m.fmriSecondLevelTwoSided {
		t.Fatalf("expected fmriSecondLevelTwoSided=false")
	}
}

func TestApplyConfigKeys_HydratesExtendedMLConfig(t *testing.T) {
	m := New(types.PipelineML, ".")
	values := map[string]interface{}{
		"machine_learning.preprocessing.imputer_strategy":                     "most_frequent",
		"machine_learning.preprocessing.pca.enabled":                          true,
		"machine_learning.preprocessing.pca.svd_solver":                       "randomized",
		"machine_learning.preprocessing.pca.n_components":                     0.9,
		"machine_learning.classification.resampler":                           "smote",
		"machine_learning.models.svm.kernel":                                  "poly",
		"machine_learning.models.svm.C_grid":                                  []interface{}{0.1, 1.0, 10.0},
		"machine_learning.models.random_forest.class_weight":                  "none",
		"machine_learning.analysis.time_generalization.min_subjects_per_cell": float64(7),
		"machine_learning.targets.strict_regression_target_continuous":        true,
		"machine_learning.interpretability.grouped_outputs":                   false,
		"machine_learning.classification.max_failed_fold_fraction":            0.33,
		"machine_learning.evaluation.bootstrap_iterations":                    float64(2500),
	}

	m.ApplyConfigKeys(values)

	if m.mlImputer != 2 {
		t.Fatalf("expected mlImputer=2 (most_frequent), got %d", m.mlImputer)
	}
	if !m.mlPCAEnabled {
		t.Fatalf("expected mlPCAEnabled=true")
	}
	if m.mlPCASvdSolver != 2 {
		t.Fatalf("expected mlPCASvdSolver=2 (randomized), got %d", m.mlPCASvdSolver)
	}
	if m.mlPCANComponents != 0.9 {
		t.Fatalf("expected mlPCANComponents=0.9, got %v", m.mlPCANComponents)
	}
	if m.mlClassificationResampler != 2 {
		t.Fatalf("expected mlClassificationResampler=2 (smote), got %d", m.mlClassificationResampler)
	}
	if m.mlSvmKernel != 2 {
		t.Fatalf("expected mlSvmKernel=2 (poly), got %d", m.mlSvmKernel)
	}
	if m.mlSvmCGrid != "0.1,1,10" {
		t.Fatalf("expected mlSvmCGrid='0.1,1,10', got %q", m.mlSvmCGrid)
	}
	if m.mlRfClassWeight != 2 {
		t.Fatalf("expected mlRfClassWeight=2 (none), got %d", m.mlRfClassWeight)
	}
	if m.mlTimeGenMinSubjects != 7 {
		t.Fatalf("expected mlTimeGenMinSubjects=7, got %d", m.mlTimeGenMinSubjects)
	}
	if !m.mlTargetsStrictRegressionCont {
		t.Fatalf("expected mlTargetsStrictRegressionCont=true")
	}
	if m.mlInterpretabilityGroupedOutputs {
		t.Fatalf("expected mlInterpretabilityGroupedOutputs=false")
	}
	if m.mlClassMaxFailedFoldFraction != 0.33 {
		t.Fatalf("expected mlClassMaxFailedFoldFraction=0.33, got %v", m.mlClassMaxFailedFoldFraction)
	}
	if m.mlEvalBootstrapIterations != 2500 {
		t.Fatalf("expected mlEvalBootstrapIterations=2500, got %d", m.mlEvalBootstrapIterations)
	}
}

func TestApplyConfigKeys_HydratesAdditionalFeatureAndPreprocessingConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	values := map[string]interface{}{
		"alignment.allow_misaligned_trim":                                         true,
		"alignment.min_alignment_samples":                                         float64(11),
		"alignment.fmri_onset_reference":                                          "first_iti_start",
		"eeg.ecg_channels":                                                        []interface{}{"ECG", "EKG"},
		"epochs.autoreject_n_interpolate":                                         []interface{}{4, 8, 16},
		"preprocessing.clean_events_qc.enabled":                                   false,
		"preprocessing.clean_events_qc.ecg_coupling.enabled":                      false,
		"preprocessing.clean_events_qc.ecg_coupling.output_column":                "ecg_coupling_qc",
		"preprocessing.clean_events_qc.ecg_coupling.channels":                     []interface{}{"ECG", "EKG"},
		"preprocessing.clean_events_qc.ecg_coupling.window":                       []interface{}{-1.5, 0.25},
		"preprocessing.clean_events_qc.peripheral_low_gamma.enabled":              false,
		"preprocessing.clean_events_qc.peripheral_low_gamma.output_column":        "gamma_qc",
		"preprocessing.clean_events_qc.peripheral_low_gamma.channels":             []interface{}{"Fp1", "Fp2"},
		"preprocessing.clean_events_qc.peripheral_low_gamma.band":                 []interface{}{28.0, 44.0},
		"preprocessing.clean_events_qc.peripheral_low_gamma.window":               []interface{}{-1.0, 0.5},
		"feature_engineering.spatial_transform_per_family.connectivity":           "laplacian",
		"feature_engineering.change_scores.transform":                             "log_ratio",
		"feature_engineering.change_scores.window_pairs":                          []interface{}{[]interface{}{"baseline", "active"}},
		"feature_engineering.pac.surrogate_method":                                "time_shift",
		"feature_engineering.aperiodic.max_freq_resolution_hz":                    0.5,
		"feature_engineering.directedconnectivity.min_samples_per_mvar_parameter": float64(25),
		"feature_engineering.erds.laterality_columns":                             []interface{}{"stim_side", "hand"},
		"feature_engineering.microstates.assign_from_gfp_peaks":                   false,
	}

	m.ApplyConfigKeys(values)

	if !m.alignAllowMisalignedTrim {
		t.Fatalf("expected alignAllowMisalignedTrim=true")
	}
	if m.alignMinAlignmentSamples != 11 {
		t.Fatalf("expected alignMinAlignmentSamples=11, got %d", m.alignMinAlignmentSamples)
	}
	if m.alignFmriOnsetReference != 1 {
		t.Fatalf("expected alignFmriOnsetReference=1 (first_volume/first_iti_start), got %d", m.alignFmriOnsetReference)
	}
	if m.prepEcgChannels != "ECG,EKG" {
		t.Fatalf("expected prepEcgChannels='ECG,EKG', got %q", m.prepEcgChannels)
	}
	if m.prepAutorejectNInterpolate != "4,8,16" {
		t.Fatalf("expected prepAutorejectNInterpolate='4,8,16', got %q", m.prepAutorejectNInterpolate)
	}
	if m.prepCleanEventsQCEnabled {
		t.Fatalf("expected prepCleanEventsQCEnabled=false")
	}
	if m.prepCleanEventsQCEcgVarianceEnabled {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceEnabled=false")
	}
	if m.prepCleanEventsQCEcgVarianceOutputColumn != "ecg_coupling_qc" {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceOutputColumn='ecg_coupling_qc', got %q", m.prepCleanEventsQCEcgVarianceOutputColumn)
	}
	if m.prepCleanEventsQCEcgVarianceChannels != "[\"ECG\",\"EKG\"]" {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceChannels JSON array, got %q", m.prepCleanEventsQCEcgVarianceChannels)
	}
	if m.prepCleanEventsQCEcgVarianceWindow != "[-1.5,0.25]" {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceWindow JSON array, got %q", m.prepCleanEventsQCEcgVarianceWindow)
	}
	if m.prepCleanEventsQCPeripheralLowGammaEnabled {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaEnabled=false")
	}
	if m.prepCleanEventsQCPeripheralLowGammaOutputColumn != "gamma_qc" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaOutputColumn='gamma_qc', got %q", m.prepCleanEventsQCPeripheralLowGammaOutputColumn)
	}
	if m.prepCleanEventsQCPeripheralLowGammaChannels != "[\"Fp1\",\"Fp2\"]" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaChannels JSON array, got %q", m.prepCleanEventsQCPeripheralLowGammaChannels)
	}
	if m.prepCleanEventsQCPeripheralLowGammaBand != "[28,44]" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaBand JSON array, got %q", m.prepCleanEventsQCPeripheralLowGammaBand)
	}
	if m.prepCleanEventsQCPeripheralLowGammaWindow != "[-1,0.5]" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaWindow JSON array, got %q", m.prepCleanEventsQCPeripheralLowGammaWindow)
	}
	if m.spatialTransformPerFamilyConnectivity != 3 {
		t.Fatalf("expected spatialTransformPerFamilyConnectivity=3 (laplacian), got %d", m.spatialTransformPerFamilyConnectivity)
	}
	if m.changeScoresTransform != 2 {
		t.Fatalf("expected changeScoresTransform=2 (log_ratio), got %d", m.changeScoresTransform)
	}
	if m.changeScoresWindowPairs != "baseline:active" {
		t.Fatalf("expected changeScoresWindowPairs='baseline:active', got %q", m.changeScoresWindowPairs)
	}
	if m.pacSurrogateMethod != 3 {
		t.Fatalf("expected pacSurrogateMethod=3 (time_shift), got %d", m.pacSurrogateMethod)
	}
	if m.aperiodicMaxFreqResolutionHz != 0.5 {
		t.Fatalf("expected aperiodicMaxFreqResolutionHz=0.5, got %v", m.aperiodicMaxFreqResolutionHz)
	}
	if m.directedConnMinSamplesPerMvarParam != 25 {
		t.Fatalf("expected directedConnMinSamplesPerMvarParam=25, got %d", m.directedConnMinSamplesPerMvarParam)
	}
	if m.erdsLateralityColumns != "stim_side,hand" {
		t.Fatalf("expected erdsLateralityColumns='stim_side,hand', got %q", m.erdsLateralityColumns)
	}
	if m.microstatesAssignFromGfpPeaks {
		t.Fatalf("expected microstatesAssignFromGfpPeaks=false")
	}
}

func TestApplyConfigKeys_HydratesConnectivityFeatureConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	values := map[string]interface{}{
		"feature_engineering.connectivity.measures":                     []interface{}{"imcoh", "plv"},
		"feature_engineering.connectivity.output_level":                 "global_only",
		"feature_engineering.connectivity.enable_graph_metrics":         true,
		"feature_engineering.connectivity.graph_top_prop":               0.25,
		"feature_engineering.connectivity.sliding_window_len":           1.5,
		"feature_engineering.connectivity.sliding_window_step":          0.75,
		"feature_engineering.connectivity.aec_mode":                     "sym",
		"feature_engineering.connectivity.mode":                         "multitaper",
		"feature_engineering.connectivity.aec_absolute":                 false,
		"feature_engineering.connectivity.enable_aec":                   false,
		"feature_engineering.connectivity.n_freqs_per_band":             float64(11),
		"feature_engineering.connectivity.n_cycles":                     6.5,
		"feature_engineering.connectivity.decim":                        float64(3),
		"feature_engineering.connectivity.min_segment_samples":          float64(80),
		"feature_engineering.connectivity.small_world_n_rand":           float64(150),
		"feature_engineering.connectivity.aec_output":                   []interface{}{"r", "z"},
		"feature_engineering.connectivity.force_within_epoch_for_ml":    false,
		"feature_engineering.connectivity.granularity":                  "condition",
		"feature_engineering.connectivity.condition_column":             "trial_type",
		"feature_engineering.connectivity.condition_values":             []interface{}{"pain", "nonpain"},
		"feature_engineering.connectivity.min_epochs_per_group":         float64(7),
		"feature_engineering.connectivity.min_cycles_per_band":          4.5,
		"feature_engineering.connectivity.warn_if_no_spatial_transform": false,
		"feature_engineering.connectivity.phase_estimator":              "across_epochs",
		"feature_engineering.connectivity.min_segment_sec":              2.5,
		"feature_engineering.connectivity.dynamic_enabled":              true,
		"feature_engineering.connectivity.dynamic_measures":             []interface{}{"aec"},
		"feature_engineering.connectivity.dynamic_autocorr_lag":         float64(2),
		"feature_engineering.connectivity.dynamic_min_windows":          float64(5),
		"feature_engineering.connectivity.dynamic_include_roi_pairs":    false,
		"feature_engineering.connectivity.dynamic_state_enabled":        false,
		"feature_engineering.connectivity.dynamic_state_n_states":       float64(4),
		"feature_engineering.connectivity.dynamic_state_min_windows":    float64(10),
		"feature_engineering.connectivity.dynamic_state_random_state":   float64(17),
	}

	m.ApplyConfigKeys(values)

	if got := m.selectedConnectivityMeasures(); len(got) != 2 || got[0] != "imcoh" || got[1] != "plv" {
		t.Fatalf("expected connectivity measures [imcoh plv], got %#v", got)
	}
	if m.connOutputLevel != 1 {
		t.Fatalf("expected connOutputLevel=1 (global_only), got %d", m.connOutputLevel)
	}
	if !m.connGraphMetrics {
		t.Fatalf("expected connGraphMetrics=true")
	}
	if m.connGraphProp != 0.25 {
		t.Fatalf("expected connGraphProp=0.25, got %v", m.connGraphProp)
	}
	if m.connWindowLen != 1.5 {
		t.Fatalf("expected connWindowLen=1.5, got %v", m.connWindowLen)
	}
	if m.connWindowStep != 0.75 {
		t.Fatalf("expected connWindowStep=0.75, got %v", m.connWindowStep)
	}
	if m.connAECMode != 2 {
		t.Fatalf("expected connAECMode=2 (sym), got %d", m.connAECMode)
	}
	if m.connMode != 1 {
		t.Fatalf("expected connMode=1 (multitaper), got %d", m.connMode)
	}
	if m.connAECAbsolute {
		t.Fatalf("expected connAECAbsolute=false")
	}
	if m.connEnableAEC {
		t.Fatalf("expected connEnableAEC=false")
	}
	if m.connNFreqsPerBand != 11 {
		t.Fatalf("expected connNFreqsPerBand=11, got %d", m.connNFreqsPerBand)
	}
	if m.connNCycles != 6.5 {
		t.Fatalf("expected connNCycles=6.5, got %v", m.connNCycles)
	}
	if m.connDecim != 3 {
		t.Fatalf("expected connDecim=3, got %d", m.connDecim)
	}
	if m.connMinSegSamples != 80 {
		t.Fatalf("expected connMinSegSamples=80, got %d", m.connMinSegSamples)
	}
	if m.connSmallWorldNRand != 150 {
		t.Fatalf("expected connSmallWorldNRand=150, got %d", m.connSmallWorldNRand)
	}
	if m.connAECOutput != 2 {
		t.Fatalf("expected connAECOutput=2 (r+z), got %d", m.connAECOutput)
	}
	if m.connForceWithinEpochML {
		t.Fatalf("expected connForceWithinEpochML=false")
	}
	if m.connGranularity != 1 {
		t.Fatalf("expected connGranularity=1 (condition), got %d", m.connGranularity)
	}
	if m.connConditionColumn != "trial_type" {
		t.Fatalf("expected connConditionColumn='trial_type', got %q", m.connConditionColumn)
	}
	if m.connConditionValues != "pain nonpain" {
		t.Fatalf("expected connConditionValues='pain nonpain', got %q", m.connConditionValues)
	}
	if m.connMinEpochsPerGroup != 7 {
		t.Fatalf("expected connMinEpochsPerGroup=7, got %d", m.connMinEpochsPerGroup)
	}
	if m.connMinCyclesPerBand != 4.5 {
		t.Fatalf("expected connMinCyclesPerBand=4.5, got %v", m.connMinCyclesPerBand)
	}
	if m.connWarnNoSpatialTransform {
		t.Fatalf("expected connWarnNoSpatialTransform=false")
	}
	if m.connPhaseEstimator != 1 {
		t.Fatalf("expected connPhaseEstimator=1 (across_epochs), got %d", m.connPhaseEstimator)
	}
	if m.connMinSegmentSec != 2.5 {
		t.Fatalf("expected connMinSegmentSec=2.5, got %v", m.connMinSegmentSec)
	}
	if !m.connDynamicEnabled {
		t.Fatalf("expected connDynamicEnabled=true")
	}
	if m.connDynamicMeasures != 2 {
		t.Fatalf("expected connDynamicMeasures=2 (aec), got %d", m.connDynamicMeasures)
	}
	if m.connDynamicAutocorrLag != 2 {
		t.Fatalf("expected connDynamicAutocorrLag=2, got %d", m.connDynamicAutocorrLag)
	}
	if m.connDynamicMinWindows != 5 {
		t.Fatalf("expected connDynamicMinWindows=5, got %d", m.connDynamicMinWindows)
	}
	if m.connDynamicIncludeROIPairs {
		t.Fatalf("expected connDynamicIncludeROIPairs=false")
	}
	if m.connDynamicStateEnabled {
		t.Fatalf("expected connDynamicStateEnabled=false")
	}
	if m.connDynamicStateNStates != 4 {
		t.Fatalf("expected connDynamicStateNStates=4, got %d", m.connDynamicStateNStates)
	}
	if m.connDynamicStateMinWindows != 10 {
		t.Fatalf("expected connDynamicStateMinWindows=10, got %d", m.connDynamicStateMinWindows)
	}
	if m.connDynamicStateRandomSeed != 17 {
		t.Fatalf("expected connDynamicStateRandomSeed=17, got %d", m.connDynamicStateRandomSeed)
	}
}
