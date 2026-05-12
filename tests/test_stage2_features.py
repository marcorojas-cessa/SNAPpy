from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import mrsnappy.pipeline as pipeline
from mrsnappy.model import AcceptAllCandidatesModel, TrainedModel, load_model


def test_stage2_fitting_and_features_use_processed_detector_volume(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    processed = np.full((5, 5, 5), 7.0, dtype=np.float32)

    def fake_read_volume(path):
        assert path == image_path
        return raw

    def fake_apply_processing_base(volume, cfg):
        assert volume is raw
        return processed

    def fake_apply_smoothing(volume, cfg):
        assert volume is processed
        return processed

    def fake_detect_candidates(volume, cfg):
        assert volume is processed
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([3.0], dtype=np.float32),
        )

    def fake_refine_candidates(volume, coords, scores, window_radius, fit_method, fit_cfg, full_fit_limit):
        assert volume is processed
        assert window_radius == 3
        assert fit_method == "2D (XY) + 1D (Z) Gaussian"

        return SimpleNamespace(
            coords=coords,
            table=[
                {
                    "z": 2.0,
                    "y": 2.0,
                    "x": 2.0,
                    "score_raw": float(scores[0]),
                    "amplitude": 7.0,
                    "amplitude_x": 7.0,
                    "amplitude_y": 7.0,
                    "amplitude_z": 7.0,
                    "amplitude_xy": 7.0,
                    "background": 0.0,
                    "integrated_intensity": 7.0,
                    "snr": 7.0,
                    "sigma_x": 1.0,
                    "sigma_y": 1.0,
                    "sigma_z": 1.0,
                    "rho_xy": 0.0,
                    "rho_xz": 0.0,
                    "rho_yz": 0.0,
                    "r_squared": 1.0,
                }
            ],
        )

    monkeypatch.setattr(pipeline, "read_volume", fake_read_volume)
    monkeypatch.setattr(pipeline, "apply_processing_base", fake_apply_processing_base)
    monkeypatch.setattr(pipeline, "apply_smoothing", fake_apply_smoothing)
    monkeypatch.setattr(pipeline, "detect_candidates", fake_detect_candidates)
    monkeypatch.setattr(pipeline, "refine_candidates", fake_refine_candidates)

    coords, scores, features = pipeline.detect_image(
        image_path,
        {"fit_method": "2D (XY) + 1D (Z) Gaussian", "fit_window": 7},
    )

    assert coords.tolist() == [[2.0, 2.0, 2.0]]
    assert scores.tolist() == [3.0]
    assert features.loc[0, "amplitude"] == 7.0
    assert "quality_weighted_snr" in features.columns
    assert "quality_vs_size_penalty" in features.columns
    assert "distortion_energy" in features.columns
    assert "log_integrated_intensity" in features.columns
    pipeline.clear_pipeline_caches()


def test_load_model_requires_native_schema(tmp_path) -> None:
    model_path = tmp_path / "unsupported_model.joblib"
    joblib.dump(
        {
            "model": AcceptAllCandidatesModel(),
            "selected_features": ["amplitude"],
            "decision_threshold": 0.0,
            "recipe": {"maxima_method": "log"},
        },
        model_path,
    )

    with pytest.raises(ValueError, match="Regenerate the model with the current `mrsnappy optimize` command"):
        load_model(model_path)


def test_train_native_model_accepts_all_candidates_when_stage1_labels_are_all_positive(monkeypatch, tmp_path) -> None:
    features = pd.DataFrame({"amplitude": [1.0, 2.0]})
    coords = np.asarray([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32)

    def fake_build_training_matrices(dataset_root, split, recipe, match_distance, image_limit=None):
        y = np.ones(2, dtype=np.int32)
        metas = [
            {
                "image_path": str(tmp_path / f"{split}_img.tif"),
                "coords": coords,
                "scores": np.ones(2, dtype=np.float32),
                "features": features,
                "labels": y,
                "gt": coords,
            }
        ]
        return features.copy(), y, metas

    monkeypatch.setattr(pipeline, "build_training_matrices", fake_build_training_matrices)

    trained = pipeline.train_native_model(
        tmp_path,
        {"recipe_id": "all_positive_stage1", "selected_features": ["amplitude"]},
        {"kernels": ["linear"], "box_constraints": [1.0]},
        tmp_path / "model.joblib",
        match_distance=2.0,
    )

    assert isinstance(trained.model, AcceptAllCandidatesModel)
    assert trained.best_params["model_type"] == "stage1_pass_through"
    assert trained.decision_threshold == 0.0
    assert (tmp_path / "model.joblib").exists()


def test_train_native_model_errors_when_stage1_labels_have_no_true_positives(monkeypatch, tmp_path) -> None:
    features = pd.DataFrame({"amplitude": [1.0, 2.0]})
    coords = np.asarray([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32)

    def fake_build_training_matrices(dataset_root, split, recipe, match_distance, image_limit=None):
        y = np.zeros(2, dtype=np.int32)
        metas = [
            {
                "image_path": str(tmp_path / f"{split}_img.tif"),
                "coords": coords,
                "scores": np.ones(2, dtype=np.float32),
                "features": features,
                "labels": y,
                "gt": np.empty((0, 3), dtype=np.float32),
            }
        ]
        return features.copy(), y, metas

    monkeypatch.setattr(pipeline, "build_training_matrices", fake_build_training_matrices)

    with pytest.raises(ValueError, match="generated no true-positive training candidates"):
        pipeline.train_native_model(
            tmp_path,
            {"recipe_id": "all_negative_stage1", "selected_features": ["amplitude"]},
            {"kernels": ["linear"], "box_constraints": [1.0]},
            tmp_path / "model.joblib",
            match_distance=2.0,
        )


def test_train_native_model_errors_when_stage1_generates_no_training_candidates(monkeypatch, tmp_path) -> None:
    features = pd.DataFrame({"amplitude": []})
    coords = np.empty((0, 3), dtype=np.float32)

    def fake_build_training_matrices(dataset_root, split, recipe, match_distance, image_limit=None):
        y = np.empty((0,), dtype=np.int32)
        metas = [
            {
                "image_path": str(tmp_path / f"{split}_img.tif"),
                "coords": coords,
                "scores": np.empty((0,), dtype=np.float32),
                "features": features,
                "labels": y,
                "gt": coords,
            }
        ]
        return features.copy(), y, metas

    monkeypatch.setattr(pipeline, "build_training_matrices", fake_build_training_matrices)

    with pytest.raises(ValueError, match="generated no training candidates"):
        pipeline.train_native_model(
            tmp_path,
            {"recipe_id": "empty_stage1", "selected_features": ["amplitude"]},
            {"kernels": ["linear"], "box_constraints": [1.0]},
            tmp_path / "model.joblib",
            match_distance=2.0,
        )


def test_detect_image_reuses_fit_cache_across_feature_packs(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    processed = np.full((5, 5, 5), 7.0, dtype=np.float32)
    calls = {"stage1": 0, "fit": 0}

    monkeypatch.setattr(pipeline, "read_volume", lambda path: raw)
    monkeypatch.setattr(pipeline, "apply_processing_base", lambda volume, cfg: processed)
    monkeypatch.setattr(pipeline, "apply_smoothing", lambda volume, cfg: processed)

    def fake_detect_candidates(volume, cfg):
        calls["stage1"] += 1
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([3.0], dtype=np.float32),
        )

    def fake_refine_candidates(volume, coords, scores, window_radius, fit_method, fit_cfg, full_fit_limit):
        calls["fit"] += 1
        return SimpleNamespace(
            coords=coords,
            table=[
                {
                    "z": 2.0,
                    "y": 2.0,
                    "x": 2.0,
                    "score_raw": float(scores[0]),
                    "amplitude": 7.0,
                    "amplitude_x": 7.0,
                    "amplitude_y": 7.0,
                    "amplitude_z": 7.0,
                    "amplitude_xy": 7.0,
                    "background": 1.0,
                    "integrated_intensity": 7.0,
                    "snr": 7.0,
                    "sigma_x": 1.0,
                    "sigma_y": 1.0,
                    "sigma_z": 1.0,
                    "rho_xy": 0.0,
                    "rho_xz": 0.0,
                    "rho_yz": 0.0,
                    "r_squared": 1.0,
                }
            ],
        )

    monkeypatch.setattr(pipeline, "detect_candidates", fake_detect_candidates)
    monkeypatch.setattr(pipeline, "refine_candidates", fake_refine_candidates)

    base_cfg = {
        "fit_method": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "fit_cache_enabled": True,
        "stage1_cache_enabled": True,
    }
    _, _, amplitude_features = pipeline.detect_image(
        image_path,
        {**base_cfg, "selected_features": ["amplitude"]},
    )
    _, _, quality_features = pipeline.detect_image(
        image_path,
        {**base_cfg, "selected_features": ["quality_weighted_snr"]},
    )

    assert calls == {"stage1": 1, "fit": 1}
    assert list(amplitude_features.columns) == ["amplitude"]
    assert list(quality_features.columns) == ["quality_weighted_snr"]
    assert quality_features.loc[0, "quality_weighted_snr"] != 0.0
    pipeline.clear_pipeline_caches()


def test_stage1_preflight_reuses_preprocessed_volume_across_detector_settings(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "image.csv"
    label_path.write_text("x,y,z\n2,2,2\n")
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    processed = np.ones((5, 5, 5), dtype=np.float32)
    calls = {"base": 0, "smooth": 0, "detect": 0}

    monkeypatch.setattr(pipeline, "read_volume", lambda path: raw)

    def fake_apply_processing_base(volume, cfg):
        calls["base"] += 1
        assert volume is raw
        return processed

    def fake_apply_smoothing(volume, cfg):
        calls["smooth"] += 1
        return processed

    def fake_detect_candidates(volume, cfg):
        calls["detect"] += 1
        assert volume is processed
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([float(cfg["threshold_value"])], dtype=np.float32),
        )

    monkeypatch.setattr(pipeline, "apply_processing_base", fake_apply_processing_base)
    monkeypatch.setattr(pipeline, "apply_smoothing", fake_apply_smoothing)
    monkeypatch.setattr(pipeline, "detect_candidates", fake_detect_candidates)

    base_cfg = {
        "stage1_cache_enabled": True,
        "preprocess_cache_entries": 4,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "preproc_sigma": 1.0,
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": True,
        "background_method": "slice_opening_2d",
        "background_param": 5.0,
        "maxima_method": "log",
        "sigma_value": 1.0,
        "maxima_neighborhood": 1,
    }

    pipeline.preflight_image(image_path, label_path, {**base_cfg, "threshold_value": 0.1}, match_distance=4.0)
    pipeline.preflight_image(image_path, label_path, {**base_cfg, "threshold_value": 1.0}, match_distance=4.0)

    assert calls == {"base": 1, "smooth": 1, "detect": 2}
    pipeline.clear_pipeline_caches()


def test_stage1_preflight_reuses_processing_base_across_smoothing_settings(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "image.csv"
    label_path.write_text("x,y,z\n2,2,2\n")
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    base = np.ones((5, 5, 5), dtype=np.float32)
    calls = {"base": 0, "smooth": 0, "detect": 0}

    monkeypatch.setattr(pipeline, "read_volume", lambda path: raw)

    def fake_apply_processing_base(volume, cfg):
        calls["base"] += 1
        assert volume is raw
        return base

    def fake_apply_smoothing(volume, cfg):
        calls["smooth"] += 1
        assert volume is base
        return np.full((5, 5, 5), float(cfg["preproc_sigma"]), dtype=np.float32)

    def fake_detect_candidates(volume, cfg):
        calls["detect"] += 1
        assert np.all(volume == float(cfg["preproc_sigma"]))
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([float(cfg["threshold_value"])], dtype=np.float32),
        )

    monkeypatch.setattr(pipeline, "apply_processing_base", fake_apply_processing_base)
    monkeypatch.setattr(pipeline, "apply_smoothing", fake_apply_smoothing)
    monkeypatch.setattr(pipeline, "detect_candidates", fake_detect_candidates)

    base_cfg = {
        "stage1_cache_enabled": True,
        "preprocess_cache_entries": 4,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": True,
        "background_method": "slice_opening_2d",
        "background_param": 5.0,
        "maxima_method": "log",
        "sigma_value": 1.0,
        "maxima_neighborhood": 1,
        "threshold_value": 1.0,
    }

    pipeline.preflight_image(image_path, label_path, {**base_cfg, "preproc_sigma": 0.5}, match_distance=4.0)
    pipeline.preflight_image(image_path, label_path, {**base_cfg, "preproc_sigma": 1.0}, match_distance=4.0)

    assert calls == {"base": 1, "smooth": 2, "detect": 2}
    pipeline.clear_pipeline_caches()


def test_predict_image_uses_recipe_embedded_in_optimized_model(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "image.tif"
    model_path = tmp_path / "model.joblib"
    embedded_recipe = {"recipe_id": "winner_recipe", "selected_features": ["amplitude"]}

    class DummyModel:
        def decision_function(self, features):
            assert features.tolist() == [[7.0]]
            return np.asarray([2.0], dtype=np.float32)

    def fake_load_model(path):
        assert path == model_path
        return TrainedModel(
            model=DummyModel(),
            selected_features=["amplitude"],
            best_params={},
            decision_threshold=0.0,
            recipe=embedded_recipe,
        )

    def fake_detect_image(path, recipe, label_count=None):
        assert path == image_path
        assert recipe == embedded_recipe
        return (
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.asarray([5.0], dtype=np.float32),
            __import__("pandas").DataFrame({"amplitude": [7.0]}),
        )

    monkeypatch.setattr(pipeline, "load_model", fake_load_model)
    monkeypatch.setattr(pipeline, "detect_image", fake_detect_image)

    coords, scores = pipeline.predict_image(image_path, None, model_path)

    assert coords.tolist() == [[1.0, 2.0, 3.0]]
    assert scores.tolist() == [2.0]
