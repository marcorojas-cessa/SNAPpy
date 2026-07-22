# SNAPpy

SNAPpy detects fluorescent puncta in 3D microscopy z-stacks and native 2D microscopy images. It uses a two-stage workflow:

1. Stage 1 finds candidate puncta with LoG or h-max local-maximum detection.
2. Stage 2 fits each candidate, measures interpretable local features, and uses an SVM to remove false candidates.

SNAPpy does not ship with trained model files. First optimize a model from labeled images, then use that `model.joblib` file to detect puncta in new images.

## Names

| Name | Meaning |
|---|---|
| `SNAPpy` | Project and repository name |
| `mrsnappy` | PyPI package, Python import, and terminal command |

The PyPI name `snappy` is already used by an unrelated project, so this package uses `mrsnappy`.

## Install

Install the current GitHub version:

```bash
python -m pip install "git+https://github.com/marcorojas-cessa/SNAPpy.git"
```

Or install the latest PyPI release:

```bash
python -m pip install mrsnappy
```

Check the command:

```bash
mrsnappy --help
```

For development:

```bash
git clone https://github.com/marcorojas-cessa/SNAPpy.git
cd SNAPpy
python -m pip install -e ".[dev]"
pytest
```

SNAPpy is CPU-based and does not require a GPU.

## Quick Start

Create an editable config:

```bash
mrsnappy init-config --output config.yaml
```

The default config is the benchmark-tested starting point used for the SNAPpy
publication work: h-max candidate detection, physical-unit smoothing/background
sweeps, a 300 nm candidate-to-label matching radius, and interpretable SVM
feature packs. Before using it on other microscope data, confirm or replace the
physical spacing values:

```yaml
pipeline_defaults:
  image_dimensionality: 3
  xy_spacing_nm: 128.866  # replace if your xy pixel spacing differs
  z_spacing_nm: 300.0     # replace if your z-step spacing differs
```

For native 2D images, set `image_dimensionality: 2`, set `xy_spacing_nm`,
and use `fit_method: 2D Gaussian`. In 2D mode `z_spacing_nm` is not required
and z-only Stage 2 features are omitted rather than filled with fake values.

Optimize a model from labeled training and validation images:

```bash
mrsnappy optimize \
  --dataset-root /path/to/labeled_dataset \
  --out-dir /path/to/model \
  --config config.yaml
```

Detect puncta in one new image:

```bash
mrsnappy detect \
  --model /path/to/model/model.joblib \
  --input /path/to/image.tif \
  --output /path/to/detections.csv
```

Detect puncta in a folder:

```bash
mrsnappy detect \
  --model /path/to/model/model.joblib \
  --input /path/to/images \
  --output /path/to/detections
```

For folder input, SNAPpy writes one CSV per image using the image stem. For example, `cell_A_003.tif` becomes `cell_A_003.csv`.

## Labeled Dataset Layout

`mrsnappy optimize` currently uses fixed-split optimization. The dataset root must contain `train/` and `val/` folders:

```text
labeled_dataset/
  train/
    image_001.tif
    image_001.csv
    image_002.tif
    image_002.csv
  val/
    image_101.tif
    image_101.csv
```

Each image must be either a 3D TIFF or, with `pipeline_defaults.image_dimensionality: 2`, a native 2D TIFF. For 3D images, each same-stem CSV must contain `x`, `y`, and `z` columns in voxel coordinates. For 2D images, CSV labels may contain `x,y`, `y,x`, or `axis-0,axis-1` style columns; internally SNAPpy stores 2D coordinates in image-axis order `(y,x)`.

Detection CSVs are dimensionality-aware: 3D output uses `detection_id,x,y,z,score`, while 2D output uses `detection_id,x,y,score`.

SNAPpy uses `train/` to fit the Stage 2 SVM. It uses `val/` to choose Stage 1 settings, Stage 2 feature/SVM settings, and the final SVM decision threshold. Held-out test scoring should be done outside SNAPpy by running `mrsnappy detect` and comparing detections to test labels.

## Optimize Outputs

Optimization writes:

```text
model/
  model.joblib
  model_config.json
  model_summary.md
  optimization_splits.csv
```

- `model.joblib`: trained model used by `mrsnappy detect`.
- `model_config.json`: machine-readable record of the exact config, selected model, selected features, validation metrics, Stage 1 shortlist, and near-tie Stage 2 finalists.
- `model_summary.md`: human-readable summary of the optimized model.
- `optimization_splits.csv`: exact train/validation image and label paths used during optimization.

Official SNAPpy intentionally keeps outputs compact. Benchmark-specific files such as per-image test metrics, resource metrics, localization offsets, or candidate-level audit tables should be produced by benchmark wrapper code, not by the core SNAPpy optimizer.

## Detect Output

Detection CSV files contain:

```csv
detection_id,x,y,z,score
1,42.3,88.1,12.0,1.74
2,51.9,91.5,13.2,0.62
```

Coordinates are voxel coordinates. `z` is the stack axis.
For native 2D models, the same output omits the `z` column.

## How SNAPpy Works

The short version:

1. Read the TIFF image or z-stack.
2. Optionally subtract background.
3. Normalize the image, usually by robust z-score.
4. Optionally smooth the image for Stage 1 detection.
5. Find candidate puncta with LoG or h-max local maxima.
6. Fit local Gaussian-style models around candidates.
7. Measure intensity, sigma, fit-quality, contrast, morphology, and distortion features.
8. Use the optimized SVM and threshold to keep likely true puncta.
9. Write final detections.

See [docs/workflow.md](docs/workflow.md) for the full method explanation and [docs/cli_api.md](docs/cli_api.md) for the complete command and config reference.

## Documentation

- [Installation](docs/installation.md)
- [CLI/API and config reference](docs/cli_api.md)
- [Workflow](docs/workflow.md)
- [2D/3D fit and feature reference](docs/2d_3d_fit_feature_review.md)
- [Model files](docs/models.md)
- [Hardware guidance](docs/hardware.md)
- [Release notes](CHANGELOG.md)

## Repository Scope

This repository contains the installable SNAPpy package, documentation, examples, and tests. Raw microscopy data, trained models, benchmark result trees, cluster logs, and manuscript files are intentionally excluded.

## Citation

If you use SNAPpy in a publication, cite the associated manuscript once available.
