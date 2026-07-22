# Changelog

## 0.4.0 - Native 2D Image Support

This release adds first-class native 2D image support while preserving the
existing 3D z-stack workflow and feature names.

Highlights:

- Adds dimensionality-aware image/label I/O for native 2D TIFFs and 2D ground
  truth coordinates.
- Adds standard axis-aligned `2D Gaussian` fitting and covariance-enabled
  `Distorted 2D Gaussian` fitting, matching the naming convention used by the
  3D fit modes.
- Adds dimensional feature resolution so 2D models use area, boundary,
  circularity, and 2D sigma-product names rather than 3D volume/surface names.
- Omits z-derived features in 2D mode instead of filling them with placeholder
  values.
- Adds explicit `rolling_box_2d` background correction alongside the existing
  `rolling_ball_2d`, while retaining the 3D background correction methods.
- Expands tests for 2D preprocessing, fitting, feature packs, optimization
  recipes, CLI behavior, and detection output schemas.

## 0.3.0 - Publication-Ready GitHub Release Candidate

This release candidate contains the SNAPpy optimizer and model-output design
used for the final publication benchmark rerun.

Highlights:

- Adds physical-unit parameters for preprocessing, LoG scale, local-maximum
  suppression, and fit-derived features.
- Keeps fixed-split optimization as the supported public training mode:
  user-supplied `train/` images fit Stage 2 SVMs, and user-supplied `val/`
  images screen Stage 1 recipes, tune thresholds, and select the final model.
- Requires `pipeline_defaults.xy_spacing_nm` and
  `pipeline_defaults.z_spacing_nm` when optimizing with a dataset root.
- Uses compact official optimizer output: `model.joblib`,
  `model_config.json`, `model_summary.md`, and `optimization_splits.csv`.
- Removes official export flags for expanded candidate-feature and per-recipe
  audit tables. Benchmark-specific wrappers should write benchmark-specific
  performance, localization, and resource outputs.
- Replaces older Stage 2 feature-pack names with the current curated packs:
  `core_fit`, `core_contrast`, `core_morphology`, and `full_interpretable`.
- Uses simplified SVM class weighting with `class_weighting: "on"` or `"off"`.
- Keeps Stage 1 ranking simple: configured guardrails first, recall-band
  eligibility, higher mean Stage 1 F1, then deterministic recipe ID.
- Uses mean per-image validation F1 for Stage 2 threshold tuning and final model
  selection, with a small near-tie band that prefers simpler adequate models.
- Ensures official detection and lower-level split prediction helpers use the
  optimized model's saved decision threshold by default unless the caller
  explicitly overrides it.

This release is intended for GitHub source publication first. PyPI publication
should be a separate explicit decision.
