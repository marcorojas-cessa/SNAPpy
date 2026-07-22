# CLI, API, and Config Reference

SNAPpy has three user-facing commands:

```bash
mrsnappy init-config
mrsnappy optimize
mrsnappy detect
```

The matching Python API functions are:

```python
from mrsnappy import init_config, optimize, optimize_dry_run, detect
```

## 1. Write a Config

CLI:

```bash
mrsnappy init-config --output config.yaml
```

Python:

```python
from mrsnappy import init_config

init_config("config.yaml")
```

This writes an editable YAML config. It does not read images, optimize a model, or run detection.

Before optimizing, set the image dimensionality and physical spacing. For 3D
z-stacks:

```yaml
pipeline_defaults:
  image_dimensionality: 3
  xy_spacing_nm: 128.866
  z_spacing_nm: 300.0
```

For native 2D images:

```yaml
pipeline_defaults:
  image_dimensionality: 2
  xy_spacing_nm: 100.0
  z_spacing_nm: null
  fit_method: 2D Gaussian
```

Use the real pixel spacing and z-step spacing for your microscope data.

## 2. Optimize a Model

CLI:

```bash
mrsnappy optimize \
  --dataset-root /path/to/labeled_dataset \
  --out-dir /path/to/model \
  --config config.yaml
```

Python:

```python
from mrsnappy import optimize

optimize(
    dataset_root="/path/to/labeled_dataset",
    out_dir="/path/to/model",
    config="config.yaml",
)
```

Dry run:

```bash
mrsnappy optimize \
  --dataset-root /path/to/labeled_dataset \
  --out-dir /path/to/model \
  --config config.yaml \
  --dry-run
```

`--dry-run` writes an optimizer plan without fitting a model.

### Optimize Inputs

The current implemented optimization mode is `fixed_split`. The dataset root must contain user-defined `train/` and `val/` folders:

```text
labeled_dataset/
  train/
    image_001.tif
    image_001.csv
  val/
    image_101.tif
    image_101.csv
```

Each image must have a same-stem CSV label file. In 3D mode, TIFF images are
read as `z,y,x`, and labels must contain `x`, `y`, and `z` voxel-coordinate
columns. In native 2D mode, TIFF images are read as `y,x`, and labels may
contain `x,y`, `y,x`, or `axis-0,axis-1` style columns.

### Optimize Outputs

```text
model/
  model.joblib
  model_config.json
  model_summary.md
  optimization_splits.csv
```

| File | Contents |
|---|---|
| `model.joblib` | Trained native SNAPpy model used by `mrsnappy detect`. |
| `model_config.json` | Machine-readable effective config, dataset profile, optimizer plan, train/validation split summary, final model parameters, selected features, SVM settings when applicable, validation metrics, Stage 1 shortlist, and near-tie Stage 2 finalists. |
| `model_summary.md` | Human-readable summary of the optimized model. |
| `optimization_splits.csv` | Exact image and label paths used as `train/` and `val/`. |

Official SNAPpy does not export full candidate-feature tables, per-image benchmark tables, localization-offset tables, or resource metrics. Those belong in benchmark wrapper code.

## 3. Detect Puncta

One image:

```bash
mrsnappy detect \
  --model /path/to/model/model.joblib \
  --input /path/to/image.tif \
  --output /path/to/detections.csv
```

Folder:

```bash
mrsnappy detect \
  --model /path/to/model/model.joblib \
  --input /path/to/images \
  --output /path/to/detections
```

Image list:

```bash
mrsnappy detect \
  --model /path/to/model/model.joblib \
  --input-list image_paths.txt \
  --output /path/to/detections
```

Python:

```python
from mrsnappy import detect

detect(
    model="/path/to/model/model.joblib",
    input_path="/path/to/images",
    output="/path/to/detections",
)
```

For multiple images, output CSV names follow the input image stem. For example, `cell_A_003.tif` becomes `cell_A_003.csv`.

Detection CSV columns:

| Column | Meaning |
|---|---|
| `detection_id` | 1-based detection ID within the image. |
| `x`, `y` | Subpixel voxel coordinates. |
| `z` | Stack-axis coordinate. Present only for 3D models. |
| `score` | SVM decision score, or Stage 1 score for Stage 1 pass-through models. |

## CLI Options

### `init-config`

| Option | Meaning |
|---|---|
| `--output` | YAML config path to write. |

### `optimize`

| Option | Meaning |
|---|---|
| `--dataset-root` | Labeled dataset root. In `fixed_split` mode it contains `train/` and `val/`. |
| `--out-dir` | Output folder for `model.joblib` and optimization records. |
| `--config` | `default` or a JSON/YAML config path. Use an edited config for real optimization. |
| `--dataset-name` | Optional name written to metadata. |
| `--dry-run` | Write and print the optimizer plan without training. |

### `detect`

| Option | Meaning |
|---|---|
| `--model` | Path to a native `model.joblib` created by `mrsnappy optimize`. |
| `--input` | One TIFF image or a folder of TIFF images. |
| `--input-list` | Text file with one TIFF path per line. |
| `--output` | Output CSV for one image, or output folder for multiple images. |
| `--config` | Optional pipeline override. Most users should omit this because optimized models embed their recipe. |
| `--score-threshold` | Optional override for the model decision threshold. |

## Config Reference

The default config is the recommended starting point and matches the SNAPpy
publication benchmark geometry unless edited. It uses h-max candidate detection,
physical-unit smoothing/background sweeps, `match_distance_nm: 300.0`, and
default spacing values `xy_spacing_nm: 128.866` and `z_spacing_nm: 300.0`.
Replace the spacing and match radius when your microscope geometry or annotation
tolerance differs. Matrix fields are literal: if you provide a custom matrix
list, SNAPpy sweeps only the values listed in your config.

### Top-Level Settings

| Key | Expected value | Meaning |
|---|---|---|
| `dataset_name` | string | Short name written to metadata. |
| `dataset_root` | path or `null` | Usually supplied by `--dataset-root`; fixed-split root containing `train/` and `val/`. |
| `optimization_mode` | `fixed_split` | Current implemented mode. Cross-validation is planned but not implemented. |
| `match_distance` | positive number or `null` | Candidate-to-ground-truth matching radius in voxel units. Do not use with `match_distance_nm`. |
| `match_distance_nm` | positive number or `null` | Candidate-to-ground-truth matching radius in nanometers. This is the default mode. Requires `xy_spacing_nm`, and also `z_spacing_nm` for 3D. Do not use with `match_distance`. |
| `stage1_detector_set` | `hmax` or `log` | Built-in Stage 1 detector family to sweep. |
| `stage1_recipes` | list of recipe mappings | Optional explicit Stage 1 recipes. If supplied, these replace the built-in detector-set matrix. |
| `stage2_feature_packs` | list | Feature packs swept in Stage 2. Defaults to `core_fit`, `core_contrast`, `core_morphology`, and `full_interpretable`. |

Exactly one of `match_distance` or `match_distance_nm` must be set.

### Stage 1 Sweep Lists

| Key | Expected value | Meaning |
|---|---|---|
| `stage1_log_sigmas` | list of positive numbers | LoG Gaussian sigmas in voxel units. |
| `stage1_log_sigmas_nm` | list of positive numbers | LoG Gaussian sigmas in nanometers. |
| `stage1_log_thresholds` | list of numbers | Thresholds applied to robust-z-normalized LoG response. |
| `stage1_maxima_neighborhoods` | list of positive integers | Voxel-unit local-maxima spacing for LoG or h-max recipes. |
| `stage1_maxima_min_distances_nm` | list of positive numbers | Physical non-maximum-suppression distances in nanometers. These are swept as alternatives to voxel-unit `stage1_maxima_neighborhoods`, not paired with them. |
| `stage1_hmax_multipliers` | list of positive numbers | h-max prominence threshold as multiplier times image noise estimate. |
| `stage1_hmax_sigma_mode` | `robust` or `std` | Noise estimate used for h-max thresholding. |
| `stage1_smoothing_sigmas` | list of positive numbers plus optional `off` | Gaussian smoothing sigmas in voxel units. |
| `stage1_smoothing_sigmas_nm` | list of positive numbers plus optional `off` | Gaussian smoothing sigmas in nanometers. These are swept as alternatives to voxel-unit smoothing sigmas. |
| `stage1_background_method` | `rolling_box_2d`, `rolling_box_3d`, `slice_opening_2d`, `rolling_ball_2d`, or `rolling_ball_3d` | Background method swept when background radii are provided. |
| `stage1_background_radii` | list of positive numbers plus optional `off` | Background radii in voxel units. |
| `stage1_background_radii_nm` | list of positive numbers plus optional `off` | Background radii in nanometers. These are swept as alternatives to voxel-unit background radii. |

Physical-unit fields use `pipeline_defaults.xy_spacing_nm` and
`pipeline_defaults.z_spacing_nm` to convert nanometers to axis-specific voxel
values. The shipped default config uses physical-unit Stage 1 fields; voxel-unit
lists can still be used in custom configs.

### Preflight Guardrails

Guardrails are optional except `stage1_n_val_images`. If a guardrail value is `null` or absent, that guardrail is not used.

| Key | Expected value | Meaning |
|---|---|---|
| `preflight.stage1_n_val_images` | positive integer or `all` | Number of validation images used for the fast Stage 1 screen. Stage 2 still uses the full `val/` split. |
| `preflight.min_stage1_recall_mean` | number in `(0, 1]` or `null` | Minimum mean Stage 1 recall across labeled preflight images. Empty-GT images are excluded because recall is undefined there. |
| `preflight.max_stage1_candidates_mean` | positive number or `null` | Maximum mean candidates per preflight validation image. |
| `preflight.max_stage1_candidates_single` | positive number or `null` | Maximum candidates allowed on any one preflight image. Also used as the preflight candidate-generation cap when set. |
| `preflight.max_candidate_ratio_cap_mean` | positive number or `null` | Maximum mean candidate/GT ratio, computed as `n_candidates / n_ground_truth` only on labeled preflight images. Empty-GT images still count for candidate mean and single-image caps. |

### Stage 1 Ranking

| Key | Expected value | Meaning |
|---|---|---|
| `stage1_ranking.recall_tolerance` | number from `0` to `1` | Passing recipes within this labeled-image recall distance of the best passing recipe remain eligible. Eligible recipes are ranked by higher mean Stage 1 F1, then recipe ID. |
| `optimizer.shortlist_top_k` | positive integer | Number of Stage 1 recipes sent into Stage 2. |

### Stage 2 Selection

| Key | Expected value | Meaning |
|---|---|---|
| `optimizer.stage2_f1_tolerance` | number from `0` to `1` | Stage 2 recipes within this mean per-image validation F1 of the best recipe are considered near-ties. |
| `optimizer.max_stage1_preflight_configs` | positive integer or `null` | Safety cap for total Stage 1 recipes. |
| `optimizer.max_stage2_recipes_after_shortlist` | positive integer or `null` | Safety cap for Stage 2 feature-pack recipes after Stage 1 shortlisting. |

Within the Stage 2 F1 near-tie band, SNAPpy chooses the simplest adequate model: simpler feature pack, Stage 1 pass-through before SVM when applicable, simpler SVM settings, better Stage 1 rank, then recipe ID.

### Pipeline Defaults and Recipe Fields

These fields can appear in `pipeline_defaults` and, when needed, inside explicit `stage1_recipes`.

| Key | Expected value | Meaning |
|---|---|---|
| `image_dimensionality` | `2` or `3` | Native image dimensionality. Use `2` for `y,x` images and `3` for `z,y,x` stacks. |
| `xy_spacing_nm` | positive number | Physical xy pixel spacing. Required for optimization. |
| `z_spacing_nm` | positive number or `null` | Physical z-step spacing. Required for 3D optimization; omit or set `null` for native 2D. |
| `preproc_enabled` | `true` or `false` | Enables Stage 1 smoothing. |
| `preproc_method` | `gaussian` or `none` | Smoothing method. |
| `preproc_sigma` | positive number or `null` | Gaussian smoothing sigma in voxel units. |
| `preproc_sigma_nm` | positive number or `null` | Gaussian smoothing sigma in nanometers. |
| `norm_enabled` | `true` or `false` | Enables global intensity normalization. |
| `norm_method` | `robust_z_score` or `none` | Normalization method. |
| `background_enabled` | `true` or `false` | Enables background correction. |
| `background_method` | supported background method | Background method for this recipe. |
| `background_param` | positive number or `null` | Background radius in voxel units. |
| `background_param_nm` | positive number or `null` | Background radius in nanometers. |
| `background_clip` | `true` or `false` | Clip background-corrected negative values to zero. |
| `maxima_method` | `log` or `h_max` | Candidate detector. |
| `maxima_neighborhood` | positive integer or `null` | Voxel local-maximum spacing. |
| `maxima_min_distance_nm` | positive number or `null` | Physical non-maximum-suppression distance. |
| `sigma_value` | positive number or `null` | LoG sigma in voxel units. |
| `sigma_nm` | positive number or `null` | LoG sigma in nanometers. |
| `threshold_value` | number or `null` | LoG response threshold. |
| `h_max_sigma_multiplier` | positive number or `null` | h-max prominence multiplier. |
| `h_max_sigma_mode` | `robust` or `std` | h-max noise estimate. |
| `fit_method` | `2D Gaussian`, `Distorted 2D Gaussian`, `2D (XY) + 1D (Z) Gaussian`, `3D Gaussian`, or `Distorted 3D Gaussian` | Candidate fitting mode. |
| `fit_window` | positive odd integer | Square 2D or cubic 3D fitting window size in pixels/voxels. |
| `fit_background_width` | non-negative integer | Perimeter width used for local mean-background subtraction before fitting. |
| `fit_max_iterations` | positive integer | Maximum nonlinear fit iterations. |
| `fit_tolerance` | positive number | Nonlinear fit convergence tolerance. |
| `selected_features` | list | Advanced explicit feature list. Most users should use `stage2_feature_packs` instead. |

The default fitting mode is `2D (XY) + 1D (Z) Gaussian` for 3D configs.
Explicit recipes can use aliases `gaussian_2d`, `distorted_gaussian_2d`,
`xy_z_gaussian`, `gaussian_3d`, and `distorted_gaussian_3d`.

### Stage 2 Feature Packs

| Feature pack | Meaning |
|---|---|
| `core_fit` | Signal intensity, fitted sigma, and fit-quality features. |
| `core_contrast` | `core_fit` plus local core/shell and half-space contrast features. |
| `core_morphology` | `core_fit` plus object morphology and distorted-Gaussian covariance features when compatible. |
| `full_interpretable` | Complete scalar interpretable feature set. |

Feature packs are resolved by fitting mode. For example, distorted-Gaussian
covariance features are used only with distorted 2D or distorted 3D Gaussian
fitting. Z-only features are omitted for native 2D models.

### SVM Sweep

| Key | Expected value | Meaning |
|---|---|---|
| `svm_sweep.kernels` | list containing `linear`, `rbf`, and/or `polynomial` | SVM kernels to test. |
| `svm_sweep.box_constraints` | list of positive numbers | SVM `C` values. |
| `svm_sweep.kernel_scales` | list containing `auto`, `scale`, or positive numbers | RBF/poly gamma values. `auto` means `1 / n_features`; `scale` means `1 / (n_features * feature_variance)`. |
| `svm_sweep.polynomial_orders` | list of positive integers | Polynomial degrees. Used only for polynomial kernels. |
| `svm_sweep.standardize` | `true` or `false` | Standardize features before SVM training. |
| `svm_sweep.class_weighting` | `on` or `off` | `on` uses scikit-learn balanced class weights. |

For every SVM setting, SNAPpy trains on `train/`, scores `val/`, and tunes one global decision threshold. Thresholds tested are `0.0`, `min(score) - 1e-6`, and validation-score quantiles at 1%, 5%, 10%, 15%, ..., 95%, and 99%. The threshold with highest mean per-image validation F1 wins; exact ties choose the threshold closest to `0.0`.

### Runtime Cache

These settings affect runtime only. They do not change model results.

| Key | Expected value | Meaning |
|---|---|---|
| `runtime_cache.image_volume_cache_entries` | non-negative integer | Number of processed image volumes retained in memory. |
| `runtime_cache.stage1_cache_enabled` | `true` or `false` | Enable Stage 1 candidate-coordinate cache. |
| `runtime_cache.stage1_cache_entries` | non-negative integer | Candidate-coordinate cache size. During optimization, this cache is pruned to current Stage 1 leaders. |
| `runtime_cache.fit_cache_enabled` | `true` or `false` | Enable Gaussian-fit/feature-table cache during Stage 2. |
| `runtime_cache.fit_cache_entries` | non-negative integer | Fit/feature cache size. |

SNAPpy caches in memory only. It does not create persistent cache files.

### Advanced Dataset Profiling

SNAPpy records a lightweight dataset profile in `model_config.json` and `model_summary.md`. Most users should keep the defaults.

| Key | Expected value | Meaning |
|---|---|---|
| `profiling.enabled` | `true` or `false` | Enable the dataset profile summary. |
| `profiling.train_image_count` | positive integer | Number of training images sampled for the profile. |
| `profiling.val_image_count` | positive integer | Number of validation images sampled for the profile. |
| `profiling.gt_intensity_radius` | non-negative integer | Radius used when sampling image intensity near ground-truth coordinates. |
| `profiling.sparse_label_mean_max` | positive number or `null` | Mean labels/image at or below this value are marked sparse. `null` uses dimensionality-aware defaults: 2D uses 16; 3D uses 64. |
| `profiling.dense_label_mean_min` | positive number or `null` | Mean labels/image at or above this value are marked dense. `null` uses dimensionality-aware defaults: 2D uses 64; 3D uses 256. |
| `profiling.stage1_augmentation_enabled` | `true` or `false` | Add profile-guided Stage 1 recipes. Default is `false`. |
| `profiling.apply_runtime_pruning_to_explicit_recipes` | `true` or `false` | Apply backend candidate-limit guidance to explicit recipes. Default is `false`. |

## Public Python API

```python
init_config(output)
optimize(config="default", dataset_root=None, out_dir=None, dataset_name=None)
optimize_dry_run(config="default", dataset_root=None, out_dir=None, dataset_name=None)
detect(model, input_path=None, input_list=None, output=None, config=None, score_threshold=None)
```

Lower-level modules are available for development and testing, but they are not the recommended user workflow.
