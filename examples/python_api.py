from mrsnappy import detect, init_config, optimize, optimize_dry_run

detect(model="ch1", input_path="image.tif", output="detections.csv")

init_config("config.yaml")
optimize_dry_run(train_dir="labeled_dataset", out_dir="model", config="config.yaml")
optimize(train_dir="labeled_dataset", out_dir="model", config="config.yaml")
