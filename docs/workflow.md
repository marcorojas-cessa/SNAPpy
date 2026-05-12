# SNAPpy Workflow

SNAPpy detects puncta in a 3D fluorescent z-stack with a two-stage workflow.

## 1. Read the Image

SNAPpy reads the TIFF stack into a floating-point 3D array. Coordinates are handled internally as `z, y, x` because that is the natural array order for image stacks.

## 2. Process the Stack

The detection recipe controls Stage 1 image processing. SNAPpy applies these operations in a fixed order: optional background correction, mandatory global robust z-score normalization, then optional 3D Gaussian smoothing. Background correction is applied first so uneven background does not define the normalization statistics. Smoothing is applied last as detector preparation, suppressing pixel-scale noise immediately before LoG or h-max candidate generation.

## 3. Generate Candidate Puncta

SNAPpy then searches the processed 3D image for local maxima. Most optimized recipes use a Laplacian-of-Gaussian response, which highlights compact bright objects. SNAPpy multiplies the LoG response by `sigma^2` and robust-z-normalizes the 3D LoG response before thresholding, so LoG thresholds are expressed in robust response-scale units. h-max recipes instead identify local maxima with sufficient intensity prominence above their surroundings. A local maximum is kept as a candidate if it passes the recipe threshold and neighborhood rules.

This stage is intentionally broad. It is better for the first stage to keep extra candidates than to miss true puncta before classification.

The built-in optimizer exposes two Stage 1 detector sets through `stage1_detector_set`: `log` and `hmax`. The default is `log`. Both detector sets use the same processing grid: 3D Gaussian smoothing sigmas of `0.5`, `1.0`, and `2.0` voxels, and background correction off or SciPy 3D box morphological-opening radii of `5` and `10` voxels. The LoG detector evaluates LoG sigmas of `1`, `2`, and `3` voxels, LoG robust-z thresholds of `0.1`, `0.25`, `0.5`, `0.75`, `1`, `1.5`, `2`, and `3`, and local-maxima neighborhoods of `1` and `2` voxels. The h-max detector evaluates `h_max_sigma_multiplier` values of `0.1`, `0.25`, `0.5`, `1`, `1.5`, `2`, and `3`, and local-maxima neighborhoods of `1` and `2` voxels. h-max uses `h_max_sigma_mode: robust` by default, where the h-max prominence threshold is the multiplier times the robust sigma of the processed 3D image; `std` is also available for custom recipes.

SNAPpy supports four public background-correction methods. `rolling_box_3d` is the default optimizer method and uses `scipy.ndimage.grey_opening` with a cubic 3D structuring element of side length `2 * radius + 1`. `slice_opening_2d` applies 2D morphological opening to each z-slice independently. `rolling_ball_2d` applies scikit-image rolling-ball background estimation to each z-slice independently. `rolling_ball_3d` calls scikit-image's n-dimensional rolling-ball implementation and is available for custom recipes, but it is substantially slower on large 3D z-stacks.

## 4. Fit and Measure Each Candidate

For each candidate, SNAPpy extracts a small 3D window around the candidate center from the processed detector volume used by Stage 1. In the default configuration, this means features are measured after optional background correction, global robust z-score normalization, and 3D Gaussian smoothing, not from raw voxel intensities. This preserves the historical SNAPpy feature definition and makes classifier features consistent with the candidate-generation image. Consequently, intensity and sigma features should be interpreted as model features in processed-image units rather than direct raw physical measurements.

The default Stage 2 fit is a 2D XY plus 1D Z Gaussian with `fit_window=7`. SNAPpy also keeps 3D Gaussian and distorted 3D Gaussian fitting implementations available for custom recipes, but they are not part of the built-in default optimization preset.

The fitted candidate is converted into numerical features. These include intensity, background, signal-to-noise ratio, model fit quality, Gaussian width estimates, anisotropy-related shape ratios, detector score, and derived quality features. Feature packs are explicit lists of feature columns; there is no separate expression layer.

## 5. Classify Candidates

The trained model receives the candidate feature table and scores each candidate. Candidates with scores above the selected decision threshold are retained as puncta. Candidates below the threshold are rejected as likely background, noise, merged objects, or fitting artifacts.

## 6. Write Results

Final detections are written as a CSV with `detection_id`, `x`, `y`, `z`, and `score` columns. The score is the classifier decision score, so higher values generally indicate stronger support for a candidate being a real punctum.

## Optimization Logic

SNAPpy optimization uses a simple validation-guided train/validate workflow. The `train/` split is used to fit the Stage 2 SVM. The `val/` split is used to choose Stage 1 candidate-generation settings, Stage 2 feature/SVM settings, and the final decision threshold. A held-out `test/` split, if used for benchmarking, should be handled outside SNAPpy after optimization.

First, SNAPpy screens every Stage 1 recipe on a small validation subset. A Stage 1 recipe is only allowed into Stage 2 if it finds enough ground-truth puncta and does not produce an unreasonable number of candidates. This keeps Stage 1 broad enough to protect recall, but prevents obviously impractical recipes from slowing or confusing classifier training.

Second, SNAPpy ranks the Stage 1 candidate-generation configurations that passed screening. The ranking mainly rewards recall, gives a smaller reward for precision, and penalizes excessive candidates per ground-truth punctum. The top three passing Stage 1 configurations enter Stage 2 by default.

Third, Stage 2 trains and validates a classifier for every configured feature pack attached to each shortlisted Stage 1 configuration. For each SVM hyperparameter setting, SNAPpy fits the SVM on all candidates generated from `train/`, applies that fitted SVM to all `val/` images, tunes the decision threshold on `val/`, and records validation F1, precision, and recall. The default feature-pack sweep includes `base_only`, `curated_balanced`, `shape_localization`, `distortion`, `intensity_quality`, `model_evidence`, and `full`. The final recipe is selected from these Stage 2 validation results using validation F1. If recipes are nearly tied, SNAPpy favors the recipe with stronger Stage 1 evidence and better validation precision.

By default, SNAPpy writes a concise model export. Add `--export-optimize-report` to write the optimizer audit trail into `export_optimize_report/`. `export_optimize_report/stage1_recipes.csv` defines every Stage 1 recipe ID. `export_optimize_report/stage1_by_image.csv` records each Stage 1 recipe on each validation image, including candidate count, ground-truth count, true positives, false positives, false negatives, precision, recall, F1, guardrail status, and preflight score. `export_optimize_report/stage1_summary.csv` aggregates those measurements and records whether each recipe passed screening, where it ranked, and whether it entered Stage 2. `export_optimize_report/stage2_recipes.csv` defines every Stage 2 recipe tested after Stage 1 shortlisting. `export_optimize_report/stage2_summary.csv` records full Stage 1 plus Stage 2 validation performance for each tested Stage 2 recipe. `export_optimize_report/selection_decision.md` gives a compact human-readable explanation of the decision, while `export_optimize_report/selection_decision.json` stores the same information in machine-readable form.

Add `--export-candidate-features` during optimization to write `export_candidate_features/val_candidates.csv` for the winning pipeline. This file contains validation candidates from the winning Stage 1 recipe and winning Stage 2 feature pack only. It includes subpixel coordinates, `maxima_score`, SVM decision score, model score, model acceptance status, accepted detection ID, ground-truth label, matched/nearest ground-truth information, and the selected feature columns. The same flag is available during detection, where it writes one unlabeled candidate-feature table per input image plus `export_candidate_features/candidate_features_manifest.json`.

Default Stage 1 pass/fail parameters are configured in the `preflight` block:

```json
{
  "preflight": {
    "stage1_n_val_images": 4,
    "min_stage1_recall": 0.25,
    "max_stage1_candidates_mean": 2500,
    "max_stage1_candidates_single": 4000,
    "max_stage1_candidates_per_label_mean": null,
    "auto_candidate_ratio_cap_enabled": true,
    "auto_candidate_ratio_caps": {"sparse": 130.0, "moderate": 100.0, "dense": 70.0},
    "auto_candidate_ratio_low_contrast_multiplier": 1.25,
    "auto_candidate_ratio_high_contrast_multiplier": 0.85
  }
}
```

If `max_stage1_candidates_per_label_mean` is set to a number, that value is used directly. If it is `null` and dataset profiling is enabled, SNAPpy automatically assigns the candidate/label cap from the density regime: `130` for sparse datasets, `100` for moderate-density datasets, and `70` for dense datasets. Low-contrast datasets multiply that cap by `1.25`; high-contrast datasets multiply it by `0.85`. Set `auto_candidate_ratio_cap_enabled` to `false` to leave this guardrail disabled when `max_stage1_candidates_per_label_mean` is `null`.

## Practical Notes

Use `detect` with a `model.joblib` file created by `mrsnappy optimize`. For new channels, imaging systems, or biological conditions, optimize and validate a model on representative labeled images before using detections for final quantitative work.

Basic command:

```bash
mrsnappy detect --model /path/to/model/model.joblib --input image.tif --output detections.csv
```

The output file can be opened in spreadsheet software or read back into Python/R. Each row is one detected punctum.
