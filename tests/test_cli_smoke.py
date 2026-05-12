import json
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile


def _write_tiny_labeled_dataset(root):
    for split in ("train", "val"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        for idx in range(2):
            z, y, x = np.indices((7, 15, 15), dtype=np.float32)
            z0 = 3.0
            y0 = 6.0 + idx
            x0 = 7.0
            spot = np.exp(-(((z - z0) ** 2) / 2.0 + ((y - y0) ** 2) / 2.0 + ((x - x0) ** 2) / 2.0))
            volume = (10.0 + 1000.0 * spot).astype(np.float32)
            image_path = split_dir / f"{split}_{idx}.tif"
            csv_path = split_dir / f"{split}_{idx}.csv"
            tifffile.imwrite(image_path, volume)
            pd.DataFrame({"x": [x0], "y": [y0], "z": [z0]}).to_csv(csv_path, index=False)


def _write_native_accept_all_model(path):
    from mrsnappy.model import AcceptAllCandidatesModel, TrainedModel, save_model

    recipe = {
        "preproc_enabled": False,
        "background_enabled": False,
        "maxima_method": "simple_regional",
        "maxima_neighborhood": 1,
        "threshold_value": 0.1,
        "fit_method": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "selected_features": ["integrated_intensity"],
    }
    save_model(
        path,
        TrainedModel(
            model=AcceptAllCandidatesModel(),
            selected_features=["integrated_intensity"],
            best_params={"model_type": "stage1_pass_through"},
            decision_threshold=0.0,
            recipe=recipe,
        ),
    )


def test_cli_init_config_writes_editable_default(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    proc = subprocess.run(
        [sys.executable, "-m", "mrsnappy.cli", "init-config", "--output", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == str(config_path)
    text = config_path.read_text()
    assert "stage1_detector_set: log" in text
    assert "shortlist_top_k: 3" in text
    assert "runtime_cache:" in text
    assert "pipeline_defaults:" not in text


def test_public_api_matches_documented_frontend() -> None:
    import mrsnappy

    assert set(mrsnappy.__all__) == {
        "__version__",
        "detect",
        "init_config",
        "load_config",
        "optimize",
        "optimize_dry_run",
    }


def test_load_config_rejects_unsupported_svm_keys(tmp_path) -> None:
    from mrsnappy.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset_root: null
svm_sweep:
  selection_strategy: cross_validation
  cv_folds: 5
"""
    )

    try:
        load_config(config_path)
    except ValueError as exc:
        message = str(exc)
        assert "Unsupported svm_sweep key(s)" in message
        assert "cv_folds" in message
        assert "selection_strategy" in message
    else:
        raise AssertionError("Expected unsupported SVM selection keys to be rejected.")


def test_cli_optimize_dry_run_reports_safe_plan(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    out_dir = tmp_path / "run"
    config_path.write_text(json.dumps({"dataset_root": str(tmp_path), "profiling": {"enabled": False}}))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mrsnappy.cli",
            "optimize",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["unique_stage1_preflight_configs"] == 432
    assert payload["max_stage2_recipe_entries_after_shortlist"] == 21
    assert (out_dir / "optimizer_plan.dry_run.json").exists()


def test_api_optimize_exports_report_and_candidate_features(tmp_path) -> None:
    from mrsnappy import optimize

    dataset_root = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    config_path = tmp_path / "optimize.json"
    _write_tiny_labeled_dataset(dataset_root)
    config_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "profiling": {"enabled": False},
                "preflight": {
                    "stage1_n_val_images": 1,
                    "min_stage1_recall": 0.0,
                    "max_stage1_candidates_mean": 100,
                    "max_stage1_candidates_single": 100,
                    "max_stage1_candidates_per_label_mean": None,
                    "auto_candidate_ratio_cap_enabled": False,
                },
                "optimizer": {
                    "shortlist_top_k": 1,
                    "selection_margin": 0.0,
                    "max_stage1_preflight_configs": 10,
                    "max_stage2_recipes_after_shortlist": 2,
                },
                "stage1_smoothing_sigmas": [1.0],
                "stage1_background_params": [],
                "stage1_recipes": [{"maxima_method": "log", "sigma_value": 1.0, "threshold_value": 0.1, "maxima_neighborhood": 1}],
                "stage2_feature_packs": ["base_only"],
                "stage2_fit_variants": ["xy_z_gaussian"],
                "svm_sweep": {
                    "kernels": ["linear"],
                    "box_constraints": [1.0],
                    "kernel_scales": ["auto"],
                    "polynomial_orders": [2],
                    "standardize": True,
                    "class_weight_mode": "none",
                },
            }
        )
    )

    optimize(
        config=config_path,
        out_dir=run_dir,
        export_optimize_report=True,
        export_candidate_features=True,
    )

    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "model_manifest.json").exists()
    assert (run_dir / "export_optimize_report" / "stage1_recipes.csv").exists()
    assert (run_dir / "export_optimize_report" / "stage1_by_image.csv").exists()
    assert (run_dir / "export_optimize_report" / "stage1_summary.csv").exists()
    assert (run_dir / "export_optimize_report" / "stage2_recipes.csv").exists()
    assert (run_dir / "export_optimize_report" / "stage2_summary.csv").exists()
    assert not (run_dir / "stage2").exists()
    assert not (run_dir / "selection_decision.json").exists()
    stage1_by_image = pd.read_csv(run_dir / "export_optimize_report" / "stage1_by_image.csv")
    assert {"guardrail_pass", "preflight_score", "f1"} <= set(stage1_by_image.columns)
    features = pd.read_csv(run_dir / "export_candidate_features" / "val_candidates.csv")
    assert "maxima_score" in features.columns
    assert "stage1_score" not in features.columns
    assert {"svm_score", "model_score", "accepted_by_model", "accepted_detection_id"} <= set(features.columns)
    assert {"matched_gt_id", "matched_gt_x", "nearest_gt_distance"} <= set(features.columns)


def test_cli_help_exposes_only_public_commands() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mrsnappy.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "optimize" in proc.stdout
    assert "detect" in proc.stdout
    assert "predict" not in proc.stdout
    assert "--image" not in proc.stdout
    assert "models" not in proc.stdout
    assert "optimize-native" not in proc.stdout
    assert "predict-lab" not in proc.stdout
    assert "predict-split" not in proc.stdout


def test_cli_detect_with_native_model_writes_detection_csv(tmp_path) -> None:
    z, y, x = np.indices((9, 21, 21), dtype=np.float32)
    spot = np.exp(-(((z - 4.0) ** 2) / 2.0 + ((y - 10.0) ** 2) / 4.0 + ((x - 10.0) ** 2) / 4.0))
    volume = (50.0 + 500.0 * spot).astype(np.float32)
    image_path = tmp_path / "synthetic_spot.tif"
    output_path = tmp_path / "detections.csv"
    model_path = tmp_path / "model.joblib"
    tifffile.imwrite(image_path, volume)
    _write_native_accept_all_model(model_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mrsnappy.cli",
            "detect",
            "--model",
            str(model_path),
            "--input",
            str(image_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    df = pd.read_csv(output_path)
    assert list(df.columns) == ["detection_id", "x", "y", "z", "score"]


def test_cli_detect_requires_model_path(tmp_path) -> None:
    image_path = tmp_path / "synthetic_spot.tif"
    output_path = tmp_path / "detections.csv"
    tifffile.imwrite(image_path, np.zeros((9, 21, 21), dtype=np.float32))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mrsnappy.cli",
            "detect",
            "--model",
            "ch1",
            "--input",
            str(image_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "SNAPpy model not found: ch1" in proc.stderr


def test_cli_detect_batches_tiff_folder_and_exports_candidate_features(tmp_path) -> None:
    z, y, x = np.indices((9, 21, 21), dtype=np.float32)
    spot = np.exp(-(((z - 4.0) ** 2) / 2.0 + ((y - 10.0) ** 2) / 4.0 + ((x - 10.0) ** 2) / 4.0))
    volume = (50.0 + 500.0 * spot).astype(np.float32)
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "detections"
    model_path = tmp_path / "model.joblib"
    image_dir.mkdir()
    tifffile.imwrite(image_dir / "a.tif", volume)
    tifffile.imwrite(image_dir / "b.tiff", volume)
    _write_native_accept_all_model(model_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mrsnappy.cli",
            "detect",
            "--model",
            str(model_path),
            "--input",
            str(image_dir),
            "--output",
            str(output_dir),
            "--export-candidate-features",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert sorted(path.name for path in output_dir.glob("*.csv")) == ["a.csv", "b.csv"]
    feature_dir = output_dir / "export_candidate_features"
    assert sorted(path.name for path in feature_dir.glob("*_candidate_features.csv")) == ["a_candidate_features.csv", "b_candidate_features.csv"]
    assert (feature_dir / "candidate_features_manifest.json").exists()
    features = pd.read_csv(feature_dir / "a_candidate_features.csv")
    assert {
        "image_id",
        "candidate_id",
        "x",
        "y",
        "z",
        "maxima_score",
        "model_score",
        "accepted_by_model",
        "accepted_detection_id",
    } <= set(features.columns)
