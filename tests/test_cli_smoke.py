import json
import inspect
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
        "maxima_method": "h_max",
        "maxima_min_distance_nm": 1.0,
        "h_max_sigma_multiplier": 0.1,
        "h_max_sigma_mode": "std",
        "fit_method": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "xy_spacing_nm": 100.0,
        "z_spacing_nm": 300.0,
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
    assert "stage1_detector_set: hmax" in text
    assert "shortlist_top_k: 5" in text
    assert "runtime_cache:" in text
    assert 'stage1_smoothing_sigmas_nm: ["off", 64.433, 128.866, 257.732]' in text
    assert "match_distance_nm: 300.0" in text
    assert "pipeline_defaults:" in text
    assert "xy_spacing_nm:" in text
    assert "z_spacing_nm:" in text
    assert "exports:" not in text
    assert "export_optimize_report" not in text
    assert "export_candidate_features" not in text


def test_public_api_matches_documented_frontend() -> None:
    import mrsnappy
    from mrsnappy import detect, optimize, optimize_dry_run

    assert set(mrsnappy.__all__) == {
        "__version__",
        "detect",
        "init_config",
        "load_config",
        "optimize",
        "optimize_dry_run",
    }
    assert "match_distance" not in inspect.signature(optimize).parameters
    assert "match_distance" not in inspect.signature(optimize_dry_run).parameters
    assert "export_optimize_report" not in inspect.signature(optimize).parameters
    assert "export_candidate_features" not in inspect.signature(optimize).parameters
    assert "export_candidate_features" not in inspect.signature(detect).parameters


def test_cli_optimize_exposes_only_current_public_options() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mrsnappy.cli", "optimize", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--match-distance" not in proc.stdout
    assert "--export-optimize-report" not in proc.stdout
    assert "--export-candidate-features" not in proc.stdout


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


def test_load_config_resolves_packaged_default_template() -> None:
    from mrsnappy.config import load_config

    cfg = load_config("default")

    assert cfg["dataset_name"] == "mrsnappy_dataset"
    assert cfg["pipeline_defaults"]["xy_spacing_nm"] == 128.866
    assert cfg["pipeline_defaults"]["z_spacing_nm"] == 300.0
    assert cfg["match_distance"] is None
    assert cfg["match_distance_nm"] == 300.0
    assert cfg["svm_sweep"]["class_weighting"] == "on"


def test_cli_optimize_dry_run_reports_safe_plan(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    dataset_root = tmp_path / "dataset"
    out_dir = tmp_path / "run"
    _write_tiny_labeled_dataset(dataset_root)
    config_path.write_text(
        json.dumps(
            {
                "profiling": {"enabled": False},
                "pipeline_defaults": {"xy_spacing_nm": 100.0, "z_spacing_nm": 300.0},
            }
        )
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mrsnappy.cli",
            "optimize",
            "--config",
            str(config_path),
            "--dataset-root",
            str(dataset_root),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["unique_stage1_preflight_configs"] == 144
    assert payload["max_stage2_recipe_entries_after_shortlist"] == 20
    assert payload["optimization_mode"] == "fixed_split"
    assert (out_dir / "optimizer_plan.dry_run.json").exists()


def test_api_optimize_writes_default_model_package(tmp_path) -> None:
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
                    "pipeline_defaults": {"xy_spacing_nm": 100.0, "z_spacing_nm": 100.0},
                    "preflight": {
                    "stage1_n_val_images": "all",
                    "min_stage1_recall_mean": 0.01,
                    "max_stage1_candidates_mean": 100,
                    "max_stage1_candidates_single": 100,
                    "max_candidate_ratio_cap_mean": None,
                },
                "optimizer": {
                    "shortlist_top_k": 1,
                    "stage2_f1_tolerance": 0.0,
                    "max_stage1_preflight_configs": 10,
                    "max_stage2_recipes_after_shortlist": 2,
                },
                "stage1_smoothing_sigmas": [],
                "stage1_smoothing_sigmas_nm": [100.0],
                "stage1_background_params": [],
                "stage1_background_params_nm": [],
                "stage1_recipes": [
                    {
                        "maxima_method": "log",
                        "sigma_nm": 100.0,
                        "threshold_value": 0.1,
                        "maxima_min_distance_nm": 100.0,
                    }
                ],
                "stage2_feature_packs": ["core_fit"],
                "svm_sweep": {
                    "kernels": ["linear"],
                    "box_constraints": [1.0],
                    "kernel_scales": ["auto"],
                    "polynomial_orders": [2],
                    "standardize": True,
                    "class_weighting": "off",
                },
            }
        )
    )

    optimize(config=config_path, out_dir=run_dir)

    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "model_config.json").exists()
    assert (run_dir / "model_summary.md").exists()
    assert (run_dir / "optimization_splits.csv").exists()
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "model.joblib",
        "model_config.json",
        "model_summary.md",
        "optimization_splits.csv",
    ]
    assert not (run_dir / "model_manifest.json").exists()
    assert not (run_dir / "optimized_model_config.json").exists()
    assert not (run_dir / "effective_config.yaml").exists()
    assert not (run_dir / "optimization_split_summary.json").exists()
    assert not (run_dir / "optimizer_plan.json").exists()
    assert not (run_dir / "dataset_profile.json").exists()
    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "summary.md").exists()
    assert not (run_dir / "stage2").exists()
    assert not (run_dir / "preflight_progress.jsonl").exists()
    assert not (run_dir / "stage2_progress.jsonl").exists()
    assert not (run_dir / "selection_decision.json").exists()
    assert not (run_dir / "selection_decision.md").exists()
    assert not (run_dir / "export_optimize_report").exists()
    assert not (run_dir / "export_candidate_features").exists()
    model_config_text = (run_dir / "model_config.json").read_text()
    assert "NaN" not in model_config_text
    assert "Infinity" not in model_config_text
    model_config = json.loads(model_config_text)
    split_summary = model_config["optimization_split_summary"]
    assert split_summary["optimization_mode"] == "fixed_split"
    assert split_summary["folds"][0]["train_image_count"] == 2
    assert split_summary["folds"][0]["val_image_count"] == 2
    assert model_config["schema"] == "mrsnappy_model_config_v1"
    assert model_config["optimization_mode"] == "fixed_split"
    assert model_config["output_files"] == {
        "model": "model.joblib",
        "model_config": "model_config.json",
        "model_summary": "model_summary.md",
        "optimization_splits": "optimization_splits.csv",
    }
    assert model_config["effective_config"]["dataset_root"] == str(dataset_root)
    assert model_config["dataset_profile"]["enabled"] is False
    assert model_config["optimizer_plan"]["optimization_mode"] == "fixed_split"
    assert model_config["stage1_selection"]["guardrails"]["stage1_n_val_images_used"] == 2
    assert model_config["stage1_selection"]["shortlisted_stage1_recipes"][0]["stage1_rank_passed"] == 1
    assert model_config["stage1_recipe_id"]
    assert model_config["stage2_recipe_id"]
    assert model_config["winner_scores"]["stage2_validation"]["selection_metric"] == "mean per-image F1"
    summary_md = (run_dir / "model_summary.md").read_text()
    assert "## Stage 1 Guardrails" in summary_md
    assert "## Stage 1 Winning Recipes" in summary_md
    assert "## Stage 2 Winner" in summary_md
    assert model_config["stage1_parameters"]
    assert model_config["stage1_selection"]["shortlisted_stage1_recipes"]
    assert model_config["stage2_parameters"]["selected_features"]
    assert model_config["selection"]["ranking_order"] == [
        "keep recipes within stage2_f1_tolerance of the best mean per-image validation F1",
        "prefer simpler feature pack",
        "prefer simpler model and SVM hyperparameters",
        "then prefer better Stage 1 rank",
        "deterministic recipe_id",
    ]
    assert model_config["selection"]["finalists_within_stage2_f1_tolerance"]


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


def test_cli_detect_batches_tiff_folder(tmp_path) -> None:
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert sorted(path.name for path in output_dir.glob("*.csv")) == ["a.csv", "b.csv"]
    assert not (output_dir / "export_candidate_features").exists()
