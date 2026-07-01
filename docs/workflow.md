# SNAPpy Workflow

SNAPpy detects bright 3D puncta with a two-stage workflow. Stage 1 is deliberately broad and finds candidate peaks. Stage 2 measures each candidate and uses an SVM to reject false positives.

## Image Order

SNAPpy reads each TIFF stack as a 3D array in `z, y, x` order. Public output coordinates are written as `x, y, z`.

For Stage 1, each recipe processes the image in this order:

1. Optional background correction.
2. Optional global normalization, usually robust z-score.
3. Optional 3D Gaussian smoothing.
4. LoG or h-max local-maximum detection.
5. Optional score-ordered physical non-maximum suppression when `maxima_min_distance_nm` is used.

The feature volume used for fitting and Stage 2 features is the image after background correction and normalization but before detector-only smoothing. This avoids measuring features from unnecessarily blurred image data.

## Stage 1 Candidate Generation

SNAPpy supports two candidate detectors:

| Detector | What it does |
|---|---|
| `log` | Applies a Laplacian-of-Gaussian filter, robust-z-normalizes the response, thresholds the response, and keeps local maxima. |
| `h_max` | Finds local maxima with enough intensity prominence above their surroundings. The threshold is `h_max_sigma_multiplier * image_noise`. |

Voxel local-maxima spacing uses `maxima_neighborhood`. Physical local-maxima spacing uses `maxima_min_distance_nm`, which is applied after initial peak finding.

The shipped default optimizer config follows the publication benchmark starting
point: h-max detection, physical-unit local-maxima spacing, physical-unit
smoothing/background sweeps, and a 300 nm candidate-to-label match radius. Users
should replace the default `xy_spacing_nm` and `z_spacing_nm` values when their
microscope geometry differs.

Supported background methods:

| Method | Meaning |
|---|---|
| `rolling_box_3d` | Fast 3D grayscale-opening approximation using a box footprint. This is the default optimizer method. |
| `slice_opening_2d` | 2D morphological opening applied separately to each z-slice. |
| `rolling_ball_2d` | scikit-image rolling-ball background estimation applied separately to each z-slice. |
| `rolling_ball_3d` | scikit-image n-dimensional rolling-ball background estimation. This is slower on large 3D stacks. |

## Candidate Fitting

SNAPpy fits a local window around each candidate. The default fitting mode is:

```text
2D (XY) + 1D (Z) Gaussian
```

Other supported fitting modes are:

```text
3D Gaussian
Distorted 3D Gaussian
```

The local fit subtracts a mean perimeter background from the fitting window. `fit_background_width` controls the perimeter width. `fit_max_iterations` and `fit_tolerance` control nonlinear fit convergence.

## Stage 2 Features

Stage 2 feature packs are interpretable scalar measurements from each fitted candidate:

| Pack | Contents |
|---|---|
| `core_fit` | Signal intensity, local background/noise, fitted sigmas in nanometers, and fit-quality features. |
| `core_contrast` | `core_fit` plus local core/shell contrast and half-space contrast. |
| `core_morphology` | `core_fit` plus thresholded-object morphology and distorted-Gaussian covariance features when available. |
| `full_interpretable` | All implemented interpretable scalar features. |

Feature packs are resolved by fitting mode. Features that do not apply to a fitting mode are omitted automatically.

## Optimization

SNAPpy currently implements `fixed_split` optimization. The user supplies `train/` and `val/` folders.

| Split | Used for |
|---|---|
| `train/` | SVM training data. |
| `val/` | Stage 1 screening, Stage 1 ranking, Stage 2 feature/SVM selection, and final threshold tuning. |

Held-out test images should be scored outside SNAPpy after optimization.

### Stage 1 Screening

SNAPpy evaluates Stage 1 recipes on `preflight.stage1_n_val_images` validation images. Guardrails are optional except for the validation-image count. A recipe may be rejected for low mean recall on labeled images, too many candidates per image, too many candidates in one image, or too high a candidate/GT ratio.

Recall and Stage 1 F1 are averaged only over preflight images with at least one ground-truth label, because recall is undefined when an image has no ground truth. Empty-GT images still count for candidate burden through `max_stage1_candidates_mean` and `max_stage1_candidates_single`.

Candidate/GT ratio is computed per labeled image:

```text
n_candidates / n_ground_truth
```

Then SNAPpy averages that ratio across labeled preflight validation images.

### Stage 1 Ranking

Passing Stage 1 recipes are ranked as follows:

1. Find the best passing mean recall on labeled validation images.
2. Keep passing recipes within `stage1_ranking.recall_tolerance` of that recall.
3. Rank those eligible recipes by higher mean Stage 1 F1.
4. Break exact ties by recipe ID.
5. Send the top `optimizer.shortlist_top_k` Stage 1 recipes to Stage 2.

### Stage 2 Training and Selection

For each shortlisted Stage 1 recipe, SNAPpy trains SVM models for the configured feature packs and SVM hyperparameters.

Training labels are assigned one-to-one. Candidate/ground-truth pairs within the configured match radius are sorted by distance, then candidate ID. Each ground-truth point and each candidate can be matched only once.

For every SVM setting:

1. Fit the SVM on candidates from `train/`.
2. Score candidates from all `val/` images.
3. Tune one global decision threshold on `val/`.
4. Record mean per-image F1 and precision across all validation images, plus mean recall across labeled validation images.

Thresholds tested are:

```text
0.0
min(score) - 1e-6
score quantiles: 1%, 5%, 10%, 15%, ..., 95%, 99%
```

The threshold with highest mean per-image validation F1 wins. Exact threshold ties choose the threshold closest to `0.0`.

The final Stage 2 model is selected from recipes within `optimizer.stage2_f1_tolerance` of the best mean per-image validation F1. Within that near-tie band, SNAPpy chooses the simplest adequate model: simpler feature pack, Stage 1 pass-through before SVM when applicable, simpler SVM settings, better Stage 1 rank, then recipe ID.

If a shortlisted Stage 1 recipe produces only positive training candidates, SNAPpy evaluates it as `stage1_pass_through`. This is not a feature pack. It uses preprocessing, local-maximum detection, fitting, and no SVM.

## Detection

`mrsnappy detect` loads `model.joblib`, applies the selected Stage 1 recipe, extracts the selected Stage 2 features if the model uses an SVM, applies the saved decision threshold, and writes final detections.

Detection output contains:

```text
detection_id,x,y,z,score
```

## Caching

SNAPpy caches intermediate data in memory only. It does not write persistent cache files.

The optimizer caches:

- processed image volumes shared across Stage 1 recipes,
- candidate coordinates for current Stage 1 leaders,
- Gaussian fits and feature tables reused across Stage 2 feature/SVM sweeps.

Caching affects runtime only. It does not change model results.
