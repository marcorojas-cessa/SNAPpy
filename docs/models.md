# Model Files

SNAPpy model files are created by `mrsnappy optimize`. The package does not include trained model artifacts.

## Create a Model

Install first:

```bash
python -m pip install mrsnappy
mrsnappy init-config --output config.yaml
# Edit config.yaml and set pipeline_defaults.xy_spacing_nm and z_spacing_nm.
mrsnappy optimize --dataset-root /path/to/labeled_dataset --out-dir /path/to/model --config config.yaml
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

`model.joblib` contains the trained classifier or Stage 1 pass-through model, the selected feature list, the selected Stage 1 recipe, and the selected Stage 2 settings. SVM models include SVM parameters and a decision threshold. Stage 1 pass-through models instead record `model_type: stage1_pass_through`, `feature_pack_name: not_applicable`, `selected_features: []`, and `svm: null`.

`model_config.json`, written next to `model.joblib`, records the exact effective config, dataset profile, optimizer plan, train/validation split summary, winning Stage 1 + Stage 2 parameters, selected features, SVM parameters when applicable, validation metrics, Stage 1 score, and any nonwinning Stage 2 finalists inside the configured Stage 2 F1 tolerance. `model_summary.md` is the human-readable model report. `optimization_splits.csv` records which images were used for training and validation in the optimization run.

SNAPpy only loads model files that declare the current native model format and schema. If a file does not match the expected schema, detection stops with an unsupported-model error.
