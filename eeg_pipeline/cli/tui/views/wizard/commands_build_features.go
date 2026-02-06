package wizard

import (
	"fmt"
	"strings"
)

func (m Model) buildFeaturesAdvancedArgs() []string {
	var args []string

	// Connectivity options
	if m.isCategorySelected("connectivity") {
		measures := m.selectedConnectivityMeasures()
		if len(measures) > 0 {
			args = append(args, "--connectivity-measures")
			args = append(args, measures...)
		}
		if m.connOutputLevel == 1 {
			args = append(args, "--conn-output-level", "global_only")
		} else {
			args = append(args, "--conn-output-level", "full")
		}

		if m.connGraphMetrics {
			args = append(args, "--conn-graph-metrics")
		} else {
			args = append(args, "--no-conn-graph-metrics")
		}

		args = append(args, "--conn-graph-prop", fmt.Sprintf("%.2f", m.connGraphProp))
		args = append(args, "--conn-window-len", fmt.Sprintf("%.2f", m.connWindowLen))
		args = append(args, "--conn-window-step", fmt.Sprintf("%.2f", m.connWindowStep))

		aecModes := []string{"orth", "none", "sym"}
		if m.connAECMode < len(aecModes) {
			args = append(args, "--conn-aec-mode", aecModes[m.connAECMode])
		}
	}

	// PAC options
	if m.isCategorySelected("pac") {
		args = append(args, "--pac-phase-range", fmt.Sprintf("%.1f", m.pacPhaseMin), fmt.Sprintf("%.1f", m.pacPhaseMax))
		args = append(args, "--pac-amp-range", fmt.Sprintf("%.1f", m.pacAmpMin), fmt.Sprintf("%.1f", m.pacAmpMax))
		pacMethods := []string{"mvl", "kl", "tort", "ozkurt"}
		if m.pacMethod < len(pacMethods) {
			args = append(args, "--pac-method", pacMethods[m.pacMethod])
		}
		args = append(args, "--pac-min-epochs", fmt.Sprintf("%d", m.pacMinEpochs))
		if strings.TrimSpace(m.pacPairsSpec) != "" {
			args = append(args, "--pac-pairs")
			args = append(args, splitCSVList(m.pacPairsSpec)...)
		}
	}

	// Aperiodic options
	if m.isCategorySelected("aperiodic") {
		args = append(args, "--aperiodic-range", fmt.Sprintf("%.1f", m.aperiodicFmin), fmt.Sprintf("%.1f", m.aperiodicFmax))
		args = append(args, "--aperiodic-peak-z", fmt.Sprintf("%.2f", m.aperiodicPeakZ))
		args = append(args, "--aperiodic-min-r2", fmt.Sprintf("%.3f", m.aperiodicMinR2))
		args = append(args, "--aperiodic-min-points", fmt.Sprintf("%d", m.aperiodicMinPoints))
		if m.aperiodicPsdBandwidth > 0 {
			args = append(args, "--aperiodic-psd-bandwidth", fmt.Sprintf("%.1f", m.aperiodicPsdBandwidth))
		}
		if m.aperiodicMaxRms > 0 {
			args = append(args, "--aperiodic-max-rms", fmt.Sprintf("%.3f", m.aperiodicMaxRms))
		}
		if m.aperiodicMinSegmentSec != 2.0 {
			args = append(args, "--aperiodic-min-segment-sec", fmt.Sprintf("%.1f", m.aperiodicMinSegmentSec))
		}
		if m.aperiodicSubtractEvoked {
			args = append(args, "--aperiodic-subtract-evoked")
		}
	}

	if m.isCategorySelected("itpc") {
		itpcMethods := []string{"global", "fold_global", "loo", "condition"}
		if m.itpcMethod >= 0 && m.itpcMethod < len(itpcMethods) && m.itpcMethod != 0 {
			args = append(args, "--itpc-method", itpcMethods[m.itpcMethod])
		}
		if m.itpcMethod == 3 && strings.TrimSpace(m.itpcConditionColumn) != "" {
			args = append(args, "--itpc-condition-column", strings.TrimSpace(m.itpcConditionColumn))
			if strings.TrimSpace(m.itpcConditionValues) != "" {
				spec := strings.ReplaceAll(m.itpcConditionValues, ",", " ")
				vals := strings.Fields(spec)
				if len(vals) > 0 {
					args = append(args, "--itpc-condition-values")
					args = append(args, vals...)
				}
			}
		}
		if m.itpcMinTrialsPerCondition > 0 && m.itpcMinTrialsPerCondition != 10 {
			args = append(args, "--itpc-min-trials-per-condition", fmt.Sprintf("%d", m.itpcMinTrialsPerCondition))
		}
		if m.itpcNJobs != -1 {
			args = append(args, "--itpc-n-jobs", fmt.Sprintf("%d", m.itpcNJobs))
		}
	}

	if m.spatialTransform != 0 {
		transforms := []string{"none", "csd", "laplacian"}
		args = append(args, "--spatial-transform", transforms[m.spatialTransform])
		if m.spatialTransformLambda2 != 1e-5 {
			args = append(args, "--spatial-transform-lambda2", fmt.Sprintf("%.2e", m.spatialTransformLambda2))
		}
		if m.spatialTransformStiffness != 4.0 {
			args = append(args, "--spatial-transform-stiffness", fmt.Sprintf("%.1f", m.spatialTransformStiffness))
		}
	}

	// Additional connectivity scientific validity options
	if m.isCategorySelected("connectivity") {
		// AEC output format
		switch m.connAECOutput {
		case 1:
			args = append(args, "--aec-output", "z")
		case 2:
			args = append(args, "--aec-output", "r", "z")
		}
		// Force within_epoch for machine learning
		if !m.connForceWithinEpochML {
			args = append(args, "--no-conn-force-within-epoch-for-ml")
		}
	}

	if m.isCategorySelected("directedconnectivity") || m.directedConnEnabled {
		directedMeasures := m.selectedDirectedConnectivityMeasures()
		if len(directedMeasures) > 0 {
			args = append(args, "--directed-connectivity-measures")
			args = append(args, directedMeasures...)
		}
		if m.directedConnOutputLevel == 1 {
			args = append(args, "--directed-conn-output-level", "global_only")
		}
		if m.directedConnMvarOrder != 10 {
			args = append(args, "--directed-conn-mvar-order", fmt.Sprintf("%d", m.directedConnMvarOrder))
		}
		if m.directedConnNFreqs != 16 {
			args = append(args, "--directed-conn-n-freqs", fmt.Sprintf("%d", m.directedConnNFreqs))
		}
		if m.directedConnMinSegSamples != 100 {
			args = append(args, "--directed-conn-min-segment-samples", fmt.Sprintf("%d", m.directedConnMinSegSamples))
		}
	}

	// Source localization options (LCMV, eLORETA)
	if m.isCategorySelected("sourcelocalization") || m.sourceLocEnabled {
		methods := []string{"lcmv", "eloreta"}
		args = append(args, "--source-method", methods[m.sourceLocMethod])

		spacings := []string{"oct5", "oct6", "ico4", "ico5"}
		if m.sourceLocSpacing != 1 { // 1 is oct6 default
			args = append(args, "--source-spacing", spacings[m.sourceLocSpacing])
		}

		parcs := []string{"aparc", "aparc.a2009s", "HCPMMP1"}
		if m.sourceLocParc != 0 {
			args = append(args, "--source-parc", parcs[m.sourceLocParc])
		}

		if m.sourceLocMethod == 0 { // LCMV
			if m.sourceLocReg != 0.05 {
				args = append(args, "--source-reg", fmt.Sprintf("%.3f", m.sourceLocReg))
			}
		} else { // eLORETA
			if m.sourceLocSnr != 3.0 {
				args = append(args, "--source-snr", fmt.Sprintf("%.1f", m.sourceLocSnr))
			}
			if m.sourceLocLoose != 0.2 {
				args = append(args, "--source-loose", fmt.Sprintf("%.2f", m.sourceLocLoose))
			}
			if m.sourceLocDepth != 0.8 {
				args = append(args, "--source-depth", fmt.Sprintf("%.2f", m.sourceLocDepth))
			}
		}

		connMethods := []string{"aec", "wpli", "plv"}
		if m.sourceLocConnMethod != 0 {
			args = append(args, "--source-connectivity-method", connMethods[m.sourceLocConnMethod])
		}

		if m.sourceLocMode == 1 {
			if strings.TrimSpace(m.sourceLocSubject) != "" {
				args = append(args, "--source-subject", strings.TrimSpace(m.sourceLocSubject))
			}
			if m.sourceLocCreateTrans {
				args = append(args, "--source-create-trans")
				if m.sourceLocAllowIdentityTrans {
					args = append(args, "--source-allow-identity-trans")
				}
			}
			if m.sourceLocCreateBemModel {
				args = append(args, "--source-create-bem-model")
			}
			if m.sourceLocCreateBemSolution {
				args = append(args, "--source-create-bem-solution")
			}
			// If not auto-creating, user must provide paths
			if !m.sourceLocCreateTrans && strings.TrimSpace(m.sourceLocTrans) != "" {
				args = append(args, "--source-trans", expandUserPath(strings.TrimSpace(m.sourceLocTrans)))
			}
			if !m.sourceLocCreateBemSolution && strings.TrimSpace(m.sourceLocBem) != "" {
				args = append(args, "--source-bem", expandUserPath(strings.TrimSpace(m.sourceLocBem)))
			}
			if m.sourceLocMindistMm != 5.0 {
				args = append(args, "--source-mindist-mm", fmt.Sprintf("%.1f", m.sourceLocMindistMm))
			}

			fmriEnabled := m.sourceLocFmriEnabled || strings.TrimSpace(m.sourceLocFmriStatsMap) != ""
			if fmriEnabled {
				args = append(args, "--source-fmri")
				if strings.TrimSpace(m.sourceLocFmriStatsMap) != "" {
					args = append(args, "--source-fmri-stats-map", expandUserPath(strings.TrimSpace(m.sourceLocFmriStatsMap)))
				}
				provenances := []string{"independent", "same_dataset"}
				if m.sourceLocFmriProvenance >= 0 && m.sourceLocFmriProvenance < len(provenances) && m.sourceLocFmriProvenance != 0 {
					args = append(args, "--source-fmri-provenance", provenances[m.sourceLocFmriProvenance])
				}
				if !m.sourceLocFmriRequireProv {
					args = append(args, "--no-source-fmri-require-provenance")
				}
				if m.sourceLocFmriThreshold != 3.1 {
					args = append(args, "--source-fmri-threshold", fmt.Sprintf("%.2f", m.sourceLocFmriThreshold))
				}
				if m.sourceLocFmriTail == 1 {
					args = append(args, "--source-fmri-tail", "abs")
				}
				if m.sourceLocFmriMinClusterMM3 > 0 {
					args = append(args, "--source-fmri-cluster-min-mm3", fmt.Sprintf("%.0f", m.sourceLocFmriMinClusterMM3))
				} else if m.sourceLocFmriMinClusterVox != 50 {
					args = append(args, "--source-fmri-cluster-min-voxels", fmt.Sprintf("%d", m.sourceLocFmriMinClusterVox))
				}
				if m.sourceLocFmriMaxClusters != 20 {
					args = append(args, "--source-fmri-max-clusters", fmt.Sprintf("%d", m.sourceLocFmriMaxClusters))
				}
				if m.sourceLocFmriMaxVoxPerClus != 2000 {
					args = append(args, "--source-fmri-max-voxels-per-cluster", fmt.Sprintf("%d", m.sourceLocFmriMaxVoxPerClus))
				}
				if m.sourceLocFmriMaxTotalVox != 20000 {
					args = append(args, "--source-fmri-max-total-voxels", fmt.Sprintf("%d", m.sourceLocFmriMaxTotalVox))
				}
				if m.sourceLocFmriRandomSeed != 0 {
					args = append(args, "--source-fmri-random-seed", fmt.Sprintf("%d", m.sourceLocFmriRandomSeed))
				}

				if strings.TrimSpace(m.sourceLocFmriWindowAName) != "" {
					args = append(args, "--source-fmri-window-a-name", strings.TrimSpace(m.sourceLocFmriWindowAName))
					args = append(args, "--source-fmri-window-a-tmin", fmt.Sprintf("%.3f", m.sourceLocFmriWindowATmin))
					args = append(args, "--source-fmri-window-a-tmax", fmt.Sprintf("%.3f", m.sourceLocFmriWindowATmax))
				}
				if strings.TrimSpace(m.sourceLocFmriWindowBName) != "" {
					args = append(args, "--source-fmri-window-b-name", strings.TrimSpace(m.sourceLocFmriWindowBName))
					args = append(args, "--source-fmri-window-b-tmin", fmt.Sprintf("%.3f", m.sourceLocFmriWindowBTmin))
					args = append(args, "--source-fmri-window-b-tmax", fmt.Sprintf("%.3f", m.sourceLocFmriWindowBTmax))
				}

				// fMRI contrast builder options
				if m.sourceLocFmriContrastEnabled {
					args = append(args, "--source-fmri-contrast-enabled")
					contrastTypes := []string{"t-test", "paired-t-test", "f-test", "custom"}
					args = append(args, "--source-fmri-contrast-type", contrastTypes[m.sourceLocFmriContrastType])
					if m.sourceLocFmriContrastType == 3 { // custom formula
						if strings.TrimSpace(m.sourceLocFmriContrastFormula) != "" {
							args = append(args, "--source-fmri-contrast-formula", strings.TrimSpace(m.sourceLocFmriContrastFormula))
						}
					} else {
						if strings.TrimSpace(m.sourceLocFmriCondAColumn) != "" {
							args = append(args, "--source-fmri-cond-a-column", strings.TrimSpace(m.sourceLocFmriCondAColumn))
						}
						if strings.TrimSpace(m.sourceLocFmriCondAValue) != "" {
							args = append(args, "--source-fmri-cond-a-value", strings.TrimSpace(m.sourceLocFmriCondAValue))
						}
						if strings.TrimSpace(m.sourceLocFmriCondBColumn) != "" {
							args = append(args, "--source-fmri-cond-b-column", strings.TrimSpace(m.sourceLocFmriCondBColumn))
						}
						if strings.TrimSpace(m.sourceLocFmriCondBValue) != "" {
							args = append(args, "--source-fmri-cond-b-value", strings.TrimSpace(m.sourceLocFmriCondBValue))
						}
					}
					if strings.TrimSpace(m.sourceLocFmriContrastName) != "" && m.sourceLocFmriContrastName != "pain_vs_baseline" {
						args = append(args, "--source-fmri-contrast-name", strings.TrimSpace(m.sourceLocFmriContrastName))
					}
					if !m.sourceLocFmriAutoDetectRuns && strings.TrimSpace(m.sourceLocFmriRunsToInclude) != "" {
						args = append(args, "--source-fmri-runs", strings.TrimSpace(m.sourceLocFmriRunsToInclude))
					}
					hrfModels := []string{"spm", "flobs", "fir"}
					if m.sourceLocFmriHrfModel != 0 {
						args = append(args, "--source-fmri-hrf-model", hrfModels[m.sourceLocFmriHrfModel])
					}
					driftModels := []string{"none", "cosine", "polynomial"}
					if m.sourceLocFmriDriftModel != 1 { // cosine is default
						args = append(args, "--source-fmri-drift-model", driftModels[m.sourceLocFmriDriftModel])
					}
					if strings.TrimSpace(m.sourceLocFmriStimPhasesToModel) != "" {
						args = append(args, "--source-fmri-stim-phases-to-model", strings.TrimSpace(m.sourceLocFmriStimPhasesToModel))
					}
					if m.sourceLocFmriHighPassHz != 0.008 {
						args = append(args, "--source-fmri-high-pass", fmt.Sprintf("%.4f", m.sourceLocFmriHighPassHz))
					}
					if m.sourceLocFmriLowPassHz > 0 {
						args = append(args, "--source-fmri-low-pass", fmt.Sprintf("%.2f", m.sourceLocFmriLowPassHz))
					}
					if m.sourceLocFmriClusterCorrection {
						args = append(args, "--source-fmri-cluster-correction")
						if m.sourceLocFmriClusterPThreshold != 0.001 {
							args = append(args, "--source-fmri-cluster-p-threshold", fmt.Sprintf("%.4f", m.sourceLocFmriClusterPThreshold))
						}
					}
					outputTypes := []string{"z-score", "t-stat", "cope", "beta"}
					if m.sourceLocFmriOutputType != 0 {
						args = append(args, "--source-fmri-output-type", outputTypes[m.sourceLocFmriOutputType])
					}
					if !m.sourceLocFmriResampleToFS {
						args = append(args, "--no-source-fmri-resample-to-fs")
					}
				}
			}
		}
	}

	// Ratios options (scientific validity)
	if m.isCategorySelected("ratios") {
		if m.ratioSource == 1 {
			args = append(args, "--ratio-source", "powcorr")
		}
	}

	// Complexity options
	if m.isCategorySelected("complexity") {
		args = append(args, "--pe-order", fmt.Sprintf("%d", m.complexityPEOrder))
		args = append(args, "--pe-delay", fmt.Sprintf("%d", m.complexityPEDelay))
	}

	// ERP options
	if m.isCategorySelected("erp") {
		if m.erpBaselineCorrection {
			args = append(args, "--erp-baseline")
		} else {
			args = append(args, "--no-erp-baseline")
		}
		if m.erpAllowNoBaseline {
			args = append(args, "--erp-allow-no-baseline")
		} else {
			args = append(args, "--no-erp-allow-no-baseline")
		}
		if strings.TrimSpace(m.erpComponentsSpec) != "" {
			args = append(args, "--erp-components")
			args = append(args, splitCSVList(m.erpComponentsSpec)...)
		}
		if m.erpSmoothMs > 0 {
			args = append(args, "--erp-smooth-ms", fmt.Sprintf("%.1f", m.erpSmoothMs))
		}
		if m.erpPeakProminenceUv > 0 {
			args = append(args, "--erp-peak-prominence-uv", fmt.Sprintf("%.1f", m.erpPeakProminenceUv))
		}
		if m.erpLowpassHz > 0 {
			args = append(args, "--erp-lowpass-hz", fmt.Sprintf("%.1f", m.erpLowpassHz))
		}
	}

	// Burst options
	if m.isCategorySelected("bursts") {
		methods := []string{"percentile", "zscore", "mad"}
		if m.burstThresholdMethod >= 0 && m.burstThresholdMethod < len(methods) {
			args = append(args, "--burst-threshold-method", methods[m.burstThresholdMethod])
		}
		refs := []string{"trial", "subject", "condition"}
		if m.burstThresholdReference >= 0 && m.burstThresholdReference < len(refs) {
			args = append(args, "--burst-threshold-reference", refs[m.burstThresholdReference])
		}
		if m.burstMinTrialsPerCondition != 10 {
			args = append(args, "--burst-min-trials-per-condition", fmt.Sprintf("%d", m.burstMinTrialsPerCondition))
		}
		if m.burstMinSegmentSec != 2.0 {
			args = append(args, "--burst-min-segment-sec", fmt.Sprintf("%.2f", m.burstMinSegmentSec))
		}
		if m.burstSkipInvalidSegments {
			args = append(args, "--burst-skip-invalid-segments")
		} else {
			args = append(args, "--no-burst-skip-invalid-segments")
		}
		if m.burstThresholdMethod == 0 && m.burstThresholdPercentile > 0 {
			args = append(args, "--burst-threshold-percentile", fmt.Sprintf("%.1f", m.burstThresholdPercentile))
		}
		if m.burstThresholdMethod != 0 {
			args = append(args, "--burst-threshold", fmt.Sprintf("%.2f", m.burstThresholdZ))
		}
		args = append(args, "--burst-min-duration", fmt.Sprintf("%d", m.burstMinDuration))
		if m.burstMinCycles != 3.0 {
			args = append(args, "--burst-min-cycles", fmt.Sprintf("%.1f", m.burstMinCycles))
		}
		if strings.TrimSpace(m.burstBandsSpec) != "" {
			args = append(args, "--burst-bands")
			args = append(args, splitCSVList(m.burstBandsSpec)...)
		}
	}

	// Power options
	if m.isCategorySelected("power") {
		if m.powerRequireBaseline {
			args = append(args, "--power-require-baseline")
		} else {
			args = append(args, "--no-power-require-baseline")
		}
		if m.powerSubtractEvoked {
			args = append(args, "--power-subtract-evoked")
		} else {
			args = append(args, "--no-power-subtract-evoked")
		}
		if m.powerMinTrialsPerCondition != 2 {
			args = append(args, "--power-min-trials-per-condition", fmt.Sprintf("%d", m.powerMinTrialsPerCondition))
		}
		if m.powerExcludeLineNoise {
			args = append(args, "--power-exclude-line-noise")
		} else {
			args = append(args, "--no-power-exclude-line-noise")
		}
		if m.powerLineNoiseFreq != 60.0 {
			args = append(args, "--power-line-noise-freq", fmt.Sprintf("%.0f", m.powerLineNoiseFreq))
		}
		if m.powerLineNoiseWidthHz != 1.0 {
			args = append(args, "--power-line-noise-width-hz", fmt.Sprintf("%.1f", m.powerLineNoiseWidthHz))
		}
		if m.powerLineNoiseHarmonics != 3 {
			args = append(args, "--power-line-noise-harmonics", fmt.Sprintf("%d", m.powerLineNoiseHarmonics))
		}
		if m.powerEmitDb {
			args = append(args, "--power-emit-db")
		} else {
			args = append(args, "--no-power-emit-db")
		}
		modes := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}
		if m.powerBaselineMode < len(modes) {
			args = append(args, "--power-baseline-mode", modes[m.powerBaselineMode])
		}
	}

	// Spectral options
	if m.isCategorySelected("spectral") {
		args = append(args, "--spectral-edge-percentile", fmt.Sprintf("%.2f", m.spectralEdgePercentile))
	}

	// Ratios options
	if m.isCategorySelected("ratios") && strings.TrimSpace(m.spectralRatioPairsSpec) != "" {
		args = append(args, "--ratio-pairs")
		args = append(args, splitCSVList(m.spectralRatioPairsSpec)...)
	}

	// Asymmetry options
	if m.isCategorySelected("asymmetry") && strings.TrimSpace(m.asymmetryChannelPairsSpec) != "" {
		args = append(args, "--asymmetry-channel-pairs")
		args = append(args, splitCSVList(m.asymmetryChannelPairsSpec)...)
	}
	if m.isCategorySelected("asymmetry") && strings.TrimSpace(m.asymmetryActivationBandsSpec) != "" {
		args = append(args, "--asymmetry-activation-bands")
		args = append(args, splitCSVList(m.asymmetryActivationBandsSpec)...)
	}
	if m.isCategorySelected("asymmetry") {
		if m.asymmetryEmitActivationConvention {
			args = append(args, "--asymmetry-emit-activation-convention")
		} else {
			args = append(args, "--no-asymmetry-emit-activation-convention")
		}
	}

	// TFR parameters
	if m.tfrFreqMin != 1.0 {
		args = append(args, "--tfr-freq-min", fmt.Sprintf("%.1f", m.tfrFreqMin))
	}
	if m.tfrFreqMax != 100.0 {
		args = append(args, "--tfr-freq-max", fmt.Sprintf("%.1f", m.tfrFreqMax))
	}
	if m.tfrNFreqs != 40 {
		args = append(args, "--tfr-n-freqs", fmt.Sprintf("%d", m.tfrNFreqs))
	}
	if m.tfrMinCycles != 3.0 {
		args = append(args, "--tfr-min-cycles", fmt.Sprintf("%.1f", m.tfrMinCycles))
	}
	if m.tfrNCyclesFactor != 2.0 {
		args = append(args, "--tfr-n-cycles-factor", fmt.Sprintf("%.1f", m.tfrNCyclesFactor))
	}
	if m.tfrWorkers != -1 {
		args = append(args, "--tfr-workers", fmt.Sprintf("%d", m.tfrWorkers))
	}
	// TFR advanced
	if m.tfrMaxCycles != 15.0 {
		args = append(args, "--tfr-max-cycles", fmt.Sprintf("%.1f", m.tfrMaxCycles))
	}
	if m.tfrDecimPower != 4 {
		args = append(args, "--tfr-decim-power", fmt.Sprintf("%d", m.tfrDecimPower))
	}
	if m.tfrDecimPhase != 1 {
		args = append(args, "--tfr-decim-phase", fmt.Sprintf("%d", m.tfrDecimPhase))
	}

	// ITPC additional options
	if m.itpcAllowUnsafeLoo {
		args = append(args, "--itpc-allow-unsafe-loo")
	}
	if m.itpcBaselineCorrection != 0 {
		modes := []string{"none", "subtract"}
		args = append(args, "--itpc-baseline-correction", modes[m.itpcBaselineCorrection])
	}

	// Spectral advanced options
	if m.isCategorySelected("spectral") || m.isCategorySelected("ratios") {
		if !m.spectralIncludeLogRatios {
			args = append(args, "--no-spectral-include-log-ratios")
		}
		if m.spectralPsdMethod != 0 {
			args = append(args, "--spectral-psd-method", "welch")
		}
		if m.spectralPsdAdaptive {
			args = append(args, "--spectral-psd-adaptive")
		} else {
			args = append(args, "--no-spectral-psd-adaptive")
		}
		if m.spectralMultitaperAdaptive {
			args = append(args, "--spectral-multitaper-adaptive")
		} else {
			args = append(args, "--no-spectral-multitaper-adaptive")
		}
		if m.spectralFmin != 1.0 {
			args = append(args, "--spectral-fmin", fmt.Sprintf("%.1f", m.spectralFmin))
		}
		if m.spectralFmax != 80.0 {
			args = append(args, "--spectral-fmax", fmt.Sprintf("%.1f", m.spectralFmax))
		}
		if !m.spectralExcludeLineNoise {
			args = append(args, "--no-spectral-exclude-line-noise")
		}
		if m.spectralLineNoiseFreq != 60.0 {
			args = append(args, "--spectral-line-noise-freq", fmt.Sprintf("%.0f", m.spectralLineNoiseFreq))
		}
		if m.spectralLineNoiseWidthHz != 1.0 {
			args = append(args, "--spectral-line-noise-width-hz", fmt.Sprintf("%.1f", m.spectralLineNoiseWidthHz))
		}
		if m.spectralLineNoiseHarmonics != 3 {
			args = append(args, "--spectral-line-noise-harmonics", fmt.Sprintf("%d", m.spectralLineNoiseHarmonics))
		}
		if m.spectralMinSegmentSec != 2.0 {
			args = append(args, "--spectral-min-segment-sec", fmt.Sprintf("%.1f", m.spectralMinSegmentSec))
		}
		if m.spectralMinCyclesAtFmin != 3.0 {
			args = append(args, "--spectral-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.spectralMinCyclesAtFmin))
		}
	}

	// Band envelope options
	if m.bandEnvelopePadSec != 0.5 {
		args = append(args, "--band-envelope-pad-sec", fmt.Sprintf("%.2f", m.bandEnvelopePadSec))
	}
	if m.bandEnvelopePadCycles != 3.0 {
		args = append(args, "--band-envelope-pad-cycles", fmt.Sprintf("%.1f", m.bandEnvelopePadCycles))
	}

	// IAF options
	if m.iafEnabled {
		args = append(args, "--iaf-enabled")
		if m.iafAlphaWidthHz != 2.0 {
			args = append(args, "--iaf-alpha-width-hz", fmt.Sprintf("%.1f", m.iafAlphaWidthHz))
		}
		if m.iafSearchRangeMin != 7.0 || m.iafSearchRangeMax != 13.0 {
			args = append(args, "--iaf-search-range", fmt.Sprintf("%.1f", m.iafSearchRangeMin), fmt.Sprintf("%.1f", m.iafSearchRangeMax))
		}
		if m.iafMinProminence != 0.05 {
			args = append(args, "--iaf-min-prominence", fmt.Sprintf("%.3f", m.iafMinProminence))
		}
		if m.iafMinCyclesAtFmin != 5.0 {
			args = append(args, "--iaf-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.iafMinCyclesAtFmin))
		}
		if m.iafMinBaselineSec != 0.0 {
			args = append(args, "--iaf-min-baseline-sec", fmt.Sprintf("%.2f", m.iafMinBaselineSec))
		}
		if m.iafAllowFullFallback {
			args = append(args, "--iaf-allow-full-fallback")
		} else {
			args = append(args, "--no-iaf-allow-full-fallback")
		}
		if m.iafAllowAllChannelsFallback {
			args = append(args, "--iaf-allow-all-channels-fallback")
		} else {
			args = append(args, "--no-iaf-allow-all-channels-fallback")
		}
		rois := m.SelectedROIs()
		if len(rois) > 0 {
			args = append(args, "--iaf-rois")
			args = append(args, rois...)
		}
	}

	// Aperiodic advanced options
	if m.isCategorySelected("aperiodic") {
		if m.aperiodicModel != 0 {
			args = append(args, "--aperiodic-model", "knee")
		}
		if m.aperiodicPsdMethod != 0 {
			args = append(args, "--aperiodic-psd-method", "welch")
		}
		if !m.aperiodicExcludeLineNoise {
			args = append(args, "--no-aperiodic-exclude-line-noise")
		}
		if m.aperiodicLineNoiseFreq != 60.0 {
			args = append(args, "--aperiodic-line-noise-freq", fmt.Sprintf("%.0f", m.aperiodicLineNoiseFreq))
		}
		if m.aperiodicLineNoiseWidthHz != 1.0 {
			args = append(args, "--aperiodic-line-noise-width-hz", fmt.Sprintf("%.1f", m.aperiodicLineNoiseWidthHz))
		}
		if m.aperiodicLineNoiseHarmonics != 3 {
			args = append(args, "--aperiodic-line-noise-harmonics", fmt.Sprintf("%d", m.aperiodicLineNoiseHarmonics))
		}
	}

	// Connectivity advanced options
	if m.isCategorySelected("connectivity") {
		if m.connGranularity != 0 {
			granularities := []string{"trial", "condition", "subject"}
			args = append(args, "--conn-granularity", granularities[m.connGranularity])
		}
		if m.connGranularity == 1 && strings.TrimSpace(m.connConditionColumn) != "" {
			args = append(args, "--conn-condition-column", strings.TrimSpace(m.connConditionColumn))
			if strings.TrimSpace(m.connConditionValues) != "" {
				spec := strings.ReplaceAll(m.connConditionValues, ",", " ")
				vals := strings.Fields(spec)
				if len(vals) > 0 {
					args = append(args, "--conn-condition-values")
					args = append(args, vals...)
				}
			}
		}
		if m.connMinEpochsPerGroup != 5 {
			args = append(args, "--conn-min-epochs-per-group", fmt.Sprintf("%d", m.connMinEpochsPerGroup))
		}
		if m.connMinCyclesPerBand != 3.0 {
			args = append(args, "--conn-min-cycles-per-band", fmt.Sprintf("%.1f", m.connMinCyclesPerBand))
		}
		if !m.connWarnNoSpatialTransform {
			args = append(args, "--no-conn-warn-no-spatial-transform")
		}
		if m.connPhaseEstimator != 0 {
			args = append(args, "--conn-phase-estimator", "across_epochs")
		}
		if m.connMinSegmentSec != 1.0 {
			args = append(args, "--conn-min-segment-sec", fmt.Sprintf("%.1f", m.connMinSegmentSec))
		}
	}

	// PAC advanced options
	if m.isCategorySelected("pac") {
		if m.pacSource != 0 {
			args = append(args, "--pac-source", "tfr")
		}
		if !m.pacNormalize {
			args = append(args, "--no-pac-normalize")
		}
		if m.pacNSurrogates != 0 {
			args = append(args, "--pac-n-surrogates", fmt.Sprintf("%d", m.pacNSurrogates))
		}
		if m.pacAllowHarmonicOvrlap {
			args = append(args, "--pac-allow-harmonic-overlap")
		}
		if m.pacMaxHarmonic != 6 {
			args = append(args, "--pac-max-harmonic", fmt.Sprintf("%d", m.pacMaxHarmonic))
		}
		if m.pacHarmonicToleranceHz != 1.0 {
			args = append(args, "--pac-harmonic-tolerance-hz", fmt.Sprintf("%.1f", m.pacHarmonicToleranceHz))
		}
		if m.pacComputeWaveformQC {
			args = append(args, "--pac-compute-waveform-qc")
		} else {
			args = append(args, "--no-pac-compute-waveform-qc")
		}
		if m.pacWaveformOffsetMs != 5.0 {
			args = append(args, "--pac-waveform-offset-ms", fmt.Sprintf("%.1f", m.pacWaveformOffsetMs))
		}
	}

	// Complexity advanced options
	if m.isCategorySelected("complexity") {
		bases := []string{"filtered", "envelope"}
		basis := "filtered"
		if m.complexitySignalBasis >= 0 && m.complexitySignalBasis < len(bases) {
			basis = bases[m.complexitySignalBasis]
		}
		if basis != "filtered" {
			args = append(args, "--complexity-signal-basis", basis)
		}
		if m.complexityMinSegmentSec != 2.0 {
			args = append(args, "--complexity-min-segment-sec", fmt.Sprintf("%.2f", m.complexityMinSegmentSec))
		}
		if m.complexityMinSamples != 200 {
			args = append(args, "--complexity-min-samples", fmt.Sprintf("%d", m.complexityMinSamples))
		}
		if !m.complexityZscore {
			args = append(args, "--no-complexity-zscore")
		}
	}

	// Ratios advanced options
	if m.isCategorySelected("ratios") {
		if m.ratiosMinSegmentSec != 1.0 {
			args = append(args, "--ratios-min-segment-sec", fmt.Sprintf("%.2f", m.ratiosMinSegmentSec))
		}
		if m.ratiosMinCyclesAtFmin != 3.0 {
			args = append(args, "--ratios-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.ratiosMinCyclesAtFmin))
		}
		if !m.ratiosSkipInvalidSegments {
			args = append(args, "--no-ratios-skip-invalid-segments")
		}
	}

	// Asymmetry advanced options
	if m.isCategorySelected("asymmetry") {
		if m.asymmetryMinSegmentSec != 1.0 {
			args = append(args, "--asymmetry-min-segment-sec", fmt.Sprintf("%.2f", m.asymmetryMinSegmentSec))
		}
		if m.asymmetryMinCyclesAtFmin != 3.0 {
			args = append(args, "--asymmetry-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.asymmetryMinCyclesAtFmin))
		}
		if !m.asymmetrySkipInvalidSegments {
			args = append(args, "--no-asymmetry-skip-invalid-segments")
		}
	}

	// Quality options
	if m.isCategorySelected("quality") {
		if m.qualityPsdMethod != 0 {
			args = append(args, "--quality-psd-method", "multitaper")
		}
		if m.qualityFmin != 1.0 {
			args = append(args, "--quality-fmin", fmt.Sprintf("%.1f", m.qualityFmin))
		}
		if m.qualityFmax != 100.0 {
			args = append(args, "--quality-fmax", fmt.Sprintf("%.1f", m.qualityFmax))
		}
		if m.qualityNfft != 256 {
			args = append(args, "--quality-n-fft", fmt.Sprintf("%d", m.qualityNfft))
		}
		if !m.qualityExcludeLineNoise {
			args = append(args, "--no-quality-exclude-line-noise")
		}
		if m.qualityLineNoiseFreq != 60.0 {
			args = append(args, "--quality-line-noise-freq", fmt.Sprintf("%.0f", m.qualityLineNoiseFreq))
		}
		if m.qualityLineNoiseWidthHz != 1.0 {
			args = append(args, "--quality-line-noise-width-hz", fmt.Sprintf("%.1f", m.qualityLineNoiseWidthHz))
		}
		if m.qualityLineNoiseHarmonics != 3 {
			args = append(args, "--quality-line-noise-harmonics", fmt.Sprintf("%d", m.qualityLineNoiseHarmonics))
		}
		if m.qualitySnrSignalBandMin != 1.0 || m.qualitySnrSignalBandMax != 30.0 {
			args = append(args, "--quality-snr-signal-band", fmt.Sprintf("%.1f", m.qualitySnrSignalBandMin), fmt.Sprintf("%.1f", m.qualitySnrSignalBandMax))
		}
		if m.qualitySnrNoiseBandMin != 40.0 || m.qualitySnrNoiseBandMax != 80.0 {
			args = append(args, "--quality-snr-noise-band", fmt.Sprintf("%.1f", m.qualitySnrNoiseBandMin), fmt.Sprintf("%.1f", m.qualitySnrNoiseBandMax))
		}
		if m.qualityMuscleBandMin != 30.0 || m.qualityMuscleBandMax != 80.0 {
			args = append(args, "--quality-muscle-band", fmt.Sprintf("%.1f", m.qualityMuscleBandMin), fmt.Sprintf("%.1f", m.qualityMuscleBandMax))
		}
	}

	// ERDS options
	if m.isCategorySelected("erds") {
		if m.erdsUseLogRatio {
			args = append(args, "--erds-use-log-ratio")
		}
		if m.erdsMinBaselinePower != 1.0e-12 {
			args = append(args, "--erds-min-baseline-power", fmt.Sprintf("%.2e", m.erdsMinBaselinePower))
		}
		if m.erdsMinActivePower != 1.0e-12 {
			args = append(args, "--erds-min-active-power", fmt.Sprintf("%.2e", m.erdsMinActivePower))
		}
		if m.erdsMinSegmentSec != 0.5 {
			args = append(args, "--erds-min-segment-sec", fmt.Sprintf("%.2f", m.erdsMinSegmentSec))
		}
		if strings.TrimSpace(m.erdsBandsSpec) != "" && m.erdsBandsSpec != "alpha,beta" {
			args = append(args, "--erds-bands")
			args = append(args, splitCSVList(m.erdsBandsSpec)...)
		}
	}

	// Generic & Validation

	args = append(args, "--min-epochs", fmt.Sprintf("%d", m.minEpochsForFeatures))
	analysisModes := []string{"group_stats", "trial_ml_safe"}
	args = append(args, "--analysis-mode", analysisModes[m.featAnalysisMode])

	// Execution options
	if m.featComputeChangeScores {
		args = append(args, "--compute-change-scores")
	} else {
		args = append(args, "--no-compute-change-scores")
	}
	if m.featSaveTfrWithSidecar {
		args = append(args, "--save-tfr-with-sidecar")
	} else {
		args = append(args, "--no-save-tfr-with-sidecar")
	}
	if m.featNJobsBands != -1 {
		args = append(args, "--n-jobs-bands", fmt.Sprintf("%d", m.featNJobsBands))
	}
	if m.featNJobsConnectivity != -1 {
		args = append(args, "--n-jobs-connectivity", fmt.Sprintf("%d", m.featNJobsConnectivity))
	}
	if m.featNJobsAperiodic != -1 {
		args = append(args, "--n-jobs-aperiodic", fmt.Sprintf("%d", m.featNJobsAperiodic))
	}
	if m.featNJobsComplexity != -1 {
		args = append(args, "--n-jobs-complexity", fmt.Sprintf("%d", m.featNJobsComplexity))
	}

	// Storage options
	if m.saveSubjectLevelFeatures {
		args = append(args, "--save-subject-level-features")
	} else {
		args = append(args, "--no-save-subject-level-features")
	}
	if m.featAlsoSaveCsv {
		args = append(args, "--also-save-csv")
	} else {
		args = append(args, "--no-also-save-csv")
	}

	return args
}

// buildBehaviorAdvancedArgs returns CLI args for behavior pipeline advanced options
