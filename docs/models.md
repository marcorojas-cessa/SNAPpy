# Model Files

SNAPpy model files are created by `mrsnappy optimize`. The package does not include trained model artifacts.

## Create a Model

Install first:

```bash
python -m pip install mrsnappy
mrsnappy init-config --output config.yaml
mrsnappy optimize --train-dir /path/to/labeled_dataset --out-dir /path/to/model --config config.yaml
```

`/path/to/model/model.joblib` is the file passed to `mrsnappy detect`.

## Use a Model

Run one image:

```bash
mrsnappy detect --model /path/to/model/model.joblib --input image.tif --output detections.csv
```

Run every TIFF in a folder:

```bash
mrsnappy detect --model /path/to/model/model.joblib --input images/ --output detections/
```

## Model Contents

`model.joblib` contains the trained classifier or Stage 1 pass-through model, the selected feature list, the selected Stage 1 recipe, the selected Stage 2 settings, and the decision threshold.

`model_manifest.json`, written next to `model.joblib`, records the same model-selection information in a human-readable form, including validation metrics and the selected optimization settings.

SNAPpy only loads model files that declare the current native model format and schema. If a file does not match the expected schema, detection stops with an unsupported-model error.
