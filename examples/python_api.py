from mrsnappy import detect, init_config, optimize, optimize_dry_run

init_config("config.yaml")
# Edit config.yaml and set pipeline_defaults.xy_spacing_nm and z_spacing_nm
# before running optimization.
optimize_dry_run(dataset_root="labeled_dataset", out_dir="model", config="config.yaml")
optimize(dataset_root="labeled_dataset", out_dir="model", config="config.yaml")

detect(model="model/model.joblib", input_path="image.tif", output="detections.csv")
