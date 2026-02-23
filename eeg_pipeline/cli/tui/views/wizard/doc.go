// Package wizard implements the TUI wizard flow: steps, selection, and advanced config per pipeline.
//
// File layout:
//
// Core
//   - model.go          — Model state, constants, data types (Computation, ROIDefinition, etc.)
//   - render.go         — View entry: layout, header, footer, renderStepContent dispatch
//   - handlers.go       — Input handling and step navigation
//   - commands.go       — Command building for execution
//   - config_persist.go — Save/load wizard config
//   - presets.go        — Behavior presets and selection helpers
//   - utils.go          — Helpers (e.g. formatRelativeTime)
//
// Step rendering (shared and per-step UI)
//   - render_steps_common.go       — Shared: accent, scroll window, default config view, output paths, parseFloat, display helpers
//   - render_steps_selection.go    — Confirmation, mode, computation, category, band, ROI, spatial, feature file, subject
//   - render_steps_plot.go         — Plot selection, plot config, time range, feature plotter
//   - render_steps_preprocessing.go — Preprocessing stages, filtering, ICA, epochs (step UIs)
//   - render_steps_advanced.go     — Advanced config dispatcher + default fallback
//
// Advanced configuration (one file per pipeline)
//   - render_advanced_features.go       — Features pipeline
//   - render_advanced_behavior.go        — Behavior pipeline
//   - render_advanced_ml.go              — Machine learning pipeline
//   - render_advanced_preprocessing.go    — Preprocessing pipeline
//   - render_advanced_fmri.go            — fMRI preprocessing pipeline
//   - render_advanced_fmri_analysis.go   — fMRI analysis pipeline
//
// Plotting
//   - plotting_advanced.go — Plot categories and global styling (renderPlottingAdvancedConfigV2)
package wizard
