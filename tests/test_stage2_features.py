from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import tifffile

import mrsnappy.fitting as fitting
import mrsnappy.pipeline as pipeline
from mrsnappy.features import feature_table
from mrsnappy.model import AcceptAllCandidatesModel, TrainedModel, build_svm_pipeline, load_model


def test_stage2_candidate_labels_are_one_to_one_with_candidate_id_tie_break() -> None:
    coords = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    gt = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    labels = pipeline._label_candidates(coords, gt, distance=1.0)

    assert labels.tolist() == [1, 0, 0]


def test_stage2_candidate_labels_do_not_assign_duplicate_candidates_to_one_gt() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    gt = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    labels = pipeline._label_candidates(coords, gt, distance=1.0)

    assert labels.tolist() == [1, 0, 0]


def test_stage2_matching_and_labeling_use_same_greedy_assignment() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    gt = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    labels = pipeline._label_candidates(coords, gt, distance=1.0)
    tp, fp, fn, assignment = pipeline.match_points_with_assignment(coords, gt, distance=1.0)

    assert labels.tolist() == [1, 0, 0]
    assert (tp, fp, fn) == (1, 2, 0)
    assert assignment == {0: 0}


def test_physical_matching_uses_anisotropic_zyx_spacing() -> None:
    gt = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    one_z_voxel = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    one_x_voxel = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    spacing_zyx_nm = (300.0, 100.0, 100.0)

    assert pipeline.match_points(one_z_voxel, gt, 150.0, match_spacing_nm=spacing_zyx_nm) == (0, 1, 1)
    assert pipeline.match_points(one_x_voxel, gt, 150.0, match_spacing_nm=spacing_zyx_nm) == (1, 0, 0)


def test_stage2_selection_metrics_are_mean_per_image_with_pooled_audit_fields() -> None:
    preds = {
        "dense": np.asarray([[float(i), 0.0, 0.0] for i in range(10)], dtype=np.float32),
        "sparse": np.empty((0, 3), dtype=np.float32),
    }
    gts = {
        "dense": np.asarray([[float(i), 0.0, 0.0] for i in range(10)], dtype=np.float32),
        "sparse": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    }

    metrics = pipeline.evaluate_predictions_for_selection(preds, gts, match_distance=0.1)

    assert metrics["tp"] == 10
    assert metrics["fn"] == 1
    assert metrics["f1_pooled"] == pytest.approx(20.0 / 21.0)
    assert metrics["f1_mean_image"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(metrics["f1_mean_image"])


def test_stage2_selection_metrics_handle_empty_gt_images_explicitly() -> None:
    preds = {
        "empty_clean": np.empty((0, 3), dtype=np.float32),
        "empty_false_positive": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "labeled_hit": np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
    }
    gts = {
        "empty_clean": np.empty((0, 3), dtype=np.float32),
        "empty_false_positive": np.empty((0, 3), dtype=np.float32),
        "labeled_hit": np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
    }

    metrics = pipeline.evaluate_predictions_for_selection(preds, gts, match_distance=0.1)

    assert metrics["n_labeled_images"] == 1
    assert metrics["n_empty_gt_images"] == 2
    assert metrics["precision_mean_image"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall_mean_image"] == pytest.approx(1.0)
    assert metrics["f1_mean_image"] == pytest.approx(2.0 / 3.0)


def test_svm_threshold_tuning_breaks_f1_ties_by_threshold_closest_to_zero() -> None:
    preds = {"img": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)}
    scores = {"img": np.asarray([-2.0, 2.0], dtype=np.float32)}
    gts = {"img": np.empty((0, 3), dtype=np.float32)}

    threshold, metrics = pipeline.tune_score_threshold(preds, scores, gts, match_distance=1.0)

    assert threshold == pytest.approx(0.0)
    assert metrics["f1_mean_image"] == pytest.approx(0.0)


def test_svm_threshold_quantile_grid_is_explicit_and_publication_ready() -> None:
    assert pipeline.SVM_THRESHOLD_QUANTILES == (
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    )


def test_svm_pipeline_uses_simple_class_weighting_switch_and_rejects_invalid_values() -> None:
    weighted = build_svm_pipeline(
        {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "standardize": True,
            "class_weighting": "on",
        }
    )
    unweighted = build_svm_pipeline(
        {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "standardize": True,
            "class_weighting": "off",
        }
    )

    assert weighted.named_steps["svc"].class_weight == "balanced"
    assert unweighted.named_steps["svc"].class_weight is None

    with pytest.raises(ValueError, match="box_constraints"):
        build_svm_pipeline({"kernel": "linear", "C": 0.0})
    with pytest.raises(ValueError, match="class_weighting"):
        build_svm_pipeline({"kernel": "linear", "C": 1.0, "class_weighting": "balanced"})


def test_stage2_fitting_uses_feature_volume_after_processing_base(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    feature_volume = np.full((5, 5, 5), 7.0, dtype=np.float32)
    stage1_volume = np.full((5, 5, 5), 9.0, dtype=np.float32)

    def fake_read_volume(path):
        assert path == image_path
        return raw

    def fake_apply_processing_base(volume, cfg):
        assert volume is raw
        return feature_volume

    def fake_apply_smoothing(volume, cfg):
        assert volume is feature_volume
        return stage1_volume

    def fake_detect_candidates(volume, cfg):
        assert volume is stage1_volume
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([3.0], dtype=np.float32),
        )

    def fake_refine_candidates(volume, coords, scores, window_radius, fit_method, fit_cfg, full_fit_limit):
        assert volume is feature_volume
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
                    "fit_method_id": "xy_z_gaussian",
                    "fit_amplitude": 7.0,
                    "voxel_amplitude": 7.0,
                    "amplitude": 7.0,
                    "amplitude_x": 7.0,
                    "amplitude_y": 7.0,
                    "amplitude_z": 7.0,
                    "amplitude_xy": 7.0,
                    "background": 0.0,
                    "noise": 1.0,
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
        {"fit_method": "2D (XY) + 1D (Z) Gaussian", "fit_window": 7, "xy_spacing_nm": 100.0, "z_spacing_nm": 300.0},
    )

    assert coords.tolist() == [[2.0, 2.0, 2.0]]
    assert scores.tolist() == [3.0]
    assert features.loc[0, "fit_amplitude"] == 7.0
    assert "quality_weighted_snr" in features.columns
    assert "quality_vs_size_penalty" in features.columns
    assert "log_integrated_intensity" in features.columns
    pipeline.clear_pipeline_caches()


def test_xy_z_gaussian_uses_max_projections_without_distortion(monkeypatch) -> None:
    signal = np.zeros((5, 5, 5), dtype=np.float32)
    signal[2, 2, 2] = 10.0
    signal[1, 2, 3] = 7.0
    center_guess = np.asarray([3.0, 3.0, 3.0], dtype=np.float32)

    expected_z_profile = signal.max(axis=(1, 2))
    expected_xy_projection = signal.max(axis=0)

    def fake_fit_1d(profile, center, max_iterations, tolerance):
        np.testing.assert_array_equal(profile, expected_z_profile)
        return np.asarray([5.0, center, 1.25], dtype=np.float32), 0.6

    def fake_fit_2d_axis_aligned(data, center, max_iterations, tolerance):
        np.testing.assert_array_equal(data, expected_xy_projection)
        return np.asarray([8.0, center[0], center[1], 1.5, 1.75], dtype=np.float32), 0.8

    monkeypatch.setattr(fitting, "_fit_1d", fake_fit_1d)
    monkeypatch.setattr(fitting, "_fit_2d_axis_aligned", fake_fit_2d_axis_aligned)

    result = fitting._fit_patch(signal, center_guess, {"fit_method": "2D (XY) + 1D (Z) Gaussian"})

    assert result["amplitude"] == 8.0
    assert result["amplitude_z"] == 5.0
    np.testing.assert_array_equal(result["rho"], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["r_squared"], 0.7)


def test_fitting_rejects_removed_independent_1d_method_and_invalid_controls() -> None:
    signal = np.ones((3, 3, 3), dtype=np.float32)
    center_guess = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported fit_method"):
        fitting._fit_patch(signal, center_guess, {"fit_method": "independent 1d gaussian"})
    with pytest.raises(ValueError, match="fit_max_iterations"):
        fitting._fit_patch(signal, center_guess, {"fit_method": "3D Gaussian", "fit_max_iterations": 0})
    with pytest.raises(ValueError, match="fit_tolerance"):
        fitting._fit_patch(signal, center_guess, {"fit_method": "3D Gaussian", "fit_tolerance": 0.0})


def test_distorted_3d_rho_features_match_covariance_names(monkeypatch) -> None:
    signal = np.zeros((5, 5, 5), dtype=np.float32)
    center_guess = np.asarray([3.0, 3.0, 3.0], dtype=np.float32)

    def fake_fit_3d_distorted(data, center, max_iterations, tolerance):
        # Distorted-3D parameter order is [rho_yz, rho_xz, rho_xy].
        return np.asarray([10.0, 3.0, 3.0, 3.0, 1.1, 1.2, 1.3, 0.2, -0.3, 0.4], dtype=np.float32), 0.9

    monkeypatch.setattr(fitting, "_fit_3d_distorted", fake_fit_3d_distorted)

    result = fitting._fit_patch(signal, center_guess, {"fit_method": "Distorted 3D Gaussian"})

    np.testing.assert_allclose(result["rho"], np.asarray([0.4, -0.3, 0.2], dtype=np.float32))


def test_distortion_shape_features_use_named_rhos_and_physical_covariance() -> None:
    rows = [
        {
            "fit_method_id": "distorted_gaussian_3d",
            "integrated_intensity": 10.0,
            "background": 1.0,
            "snr": 10.0,
            "r_squared": 1.0,
            "amplitude": 10.0,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "sigma_z": 3.0,
            "rho_xy": 0.4,
            "rho_xz": -0.3,
            "rho_yz": 0.2,
        }
    ]

    features = feature_table(rows, xy_spacing_nm=100.0, z_spacing_nm=100.0)

    assert features.loc[0, "rho_lateral_abs"] == pytest.approx(0.4)
    assert features.loc[0, "rho_axial_energy"] == pytest.approx(0.13)
    assert 0.0 <= features.loc[0, "covariance_elongation"] <= 1.0
    assert 0.0 <= features.loc[0, "long_axis_z_alignment"] <= 1.0


def test_standard_2d_feature_table_reuses_names_voids_z_and_distortion_features() -> None:
    rows = [
        {
            "fit_method_id": "gaussian_2d",
            "fit_amplitude": 10.0,
            "voxel_amplitude": 12.0,
            "integrated_intensity": 25.0,
            "background": 2.0,
            "noise": 1.0,
            "r_squared": 0.8,
            "sigma_x": 2.0,
            "sigma_y": 3.0,
            "sigma_z": np.nan,
            "rho_xy": 0.25,
            "rho_xz": 0.0,
            "rho_yz": 0.0,
            "residual_rmse": 0.1,
            "residual_energy_norm": 0.2,
        }
    ]

    features = feature_table(rows, xy_spacing_nm=100.0, image_dimensionality=2)

    assert features.loc[0, "sigma_x_nm"] == pytest.approx(200.0)
    assert features.loc[0, "sigma_y_nm"] == pytest.approx(300.0)
    assert features.loc[0, "sigma_total_nm"] == pytest.approx(500.0)
    assert features.loc[0, "sigma_product_nm2"] == pytest.approx(60000.0)
    assert "sigma_product_nm3" not in features.columns
    assert features.loc[0, "quality_vs_size_penalty"] == pytest.approx(0.8 / 500.0)
    for incompatible in ("rho_lateral_abs", "covariance_elongation"):
        assert incompatible not in features.columns
    for z_only in ("sigma_z_nm", "sigma_axial_ratio", "rho_axial_energy", "long_axis_z_alignment"):
        assert z_only not in features.columns


def test_distorted_2d_feature_table_exposes_lateral_covariance_features() -> None:
    rows = [
        {
            "fit_method_id": "distorted_gaussian_2d",
            "fit_amplitude": 10.0,
            "voxel_amplitude": 12.0,
            "integrated_intensity": 25.0,
            "background": 2.0,
            "noise": 1.0,
            "r_squared": 0.8,
            "sigma_x": 2.0,
            "sigma_y": 3.0,
            "sigma_z": np.nan,
            "rho_xy": 0.25,
            "rho_xz": 0.0,
            "rho_yz": 0.0,
            "residual_rmse": 0.1,
            "residual_energy_norm": 0.2,
        }
    ]

    features = feature_table(rows, xy_spacing_nm=100.0, image_dimensionality=2)

    assert features.loc[0, "rho_lateral_abs"] == pytest.approx(0.25)
    assert 0.0 <= features.loc[0, "covariance_elongation"] <= 1.0
    for z_only in ("sigma_z_nm", "sigma_axial_ratio", "rho_axial_energy", "long_axis_z_alignment"):
        assert z_only not in features.columns


def test_2d_feature_table_rejects_explicit_z_only_features() -> None:
    rows = [
        {
            "fit_method_id": "gaussian_2d",
            "fit_amplitude": 1.0,
            "integrated_intensity": 1.0,
            "background": 1.0,
            "noise": 1.0,
            "r_squared": 1.0,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "sigma_z": np.nan,
        }
    ]

    with pytest.raises(ValueError, match="sigma_z_nm"):
        feature_table(rows, ["sigma_z_nm"], xy_spacing_nm=100.0, image_dimensionality=2)


def test_2d_feature_table_rejects_old_3d_morphology_feature_names() -> None:
    rows = [
        {
            "fit_method_id": "gaussian_2d",
            "fit_amplitude": 1.0,
            "integrated_intensity": 1.0,
            "background": 1.0,
            "noise": 1.0,
            "r_squared": 1.0,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "sigma_z": np.nan,
            "component_sphericity_3d": 0.5,
        }
    ]

    with pytest.raises(ValueError, match="component_sphericity_3d"):
        feature_table(rows, ["component_sphericity_3d"], xy_spacing_nm=100.0, image_dimensionality=2)


def test_refine_candidates_computes_native_2d_coordinates_contrast_and_morphology() -> None:
    y, x = np.indices((13, 13), dtype=np.float32)
    image = (20.0 * np.exp(-(((y - 6.0) ** 2) + ((x - 7.0) ** 2)) / 4.0)).astype(np.float32)

    result = fitting.refine_candidates(
        image,
        np.asarray([[6.0, 7.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        window_radius=3,
        fit_method="2D Gaussian",
        fit_cfg={
            "fit_method": "2D Gaussian",
            "fit_background_width": 1,
            "fit_max_iterations": 100,
            "fit_tolerance": 1e-6,
            "xy_spacing_nm": 100.0,
            "selected_features": [
                "core_mean",
                "core_minus_shell",
                "halfspace_absdiff_max",
                "component_pixel_area",
                "component_boundary_px",
                "component_circularity_2d",
                "component_centroid_fit_distance_nm",
            ],
        },
    )

    assert result.coords.shape == (1, 2)
    row = result.table[0]
    assert "z" not in row
    assert row["y"] == pytest.approx(6.0, abs=0.5)
    assert row["x"] == pytest.approx(7.0, abs=0.5)
    assert row["core_mean"] > row["shell_mean"]
    assert row["component_pixel_area"] > 0
    assert row["component_boundary_px"] > 0
    assert 0.0 <= row["component_circularity_2d"] <= 1.0
    assert row["component_centroid_fit_distance_nm"] >= 0.0


def test_fit_patch_native_2d_rejects_3d_only_fit_methods() -> None:
    image = np.ones((7, 7), dtype=np.float32)
    center = np.asarray([4.0, 4.0], dtype=np.float32)

    for fit_method in ("2D (XY) + 1D (Z) Gaussian", "3D Gaussian", "Distorted 3D Gaussian"):
        with pytest.raises(ValueError, match="incompatible with native 2D images"):
            fitting._fit_patch(image, center, {"fit_method": fit_method})


def test_fit_patch_native_3d_rejects_2d_fit_methods() -> None:
    image = np.ones((5, 7, 7), dtype=np.float32)
    center = np.asarray([3.0, 4.0, 4.0], dtype=np.float32)

    for fit_method in ("2D Gaussian", "Distorted 2D Gaussian"):
        with pytest.raises(ValueError, match="incompatible with native 3D images"):
            fitting._fit_patch(image, center, {"fit_method": fit_method})


def test_fit_patch_native_2d_standard_and_distorted_gaussian_dispatch(monkeypatch) -> None:
    image = np.ones((7, 7), dtype=np.float32)
    center = np.asarray([4.0, 4.0], dtype=np.float32)
    calls = {"axis": 0, "distorted": 0}

    def fake_axis(data, center_guess, max_iterations, tolerance):
        calls["axis"] += 1
        return np.asarray([10.0, center_guess[0], center_guess[1], 1.2, 1.3], dtype=np.float32), 0.8

    def fake_distorted(data, center_guess, max_iterations, tolerance):
        calls["distorted"] += 1
        return np.asarray([10.0, center_guess[0], center_guess[1], 1.2, 1.3, 0.4], dtype=np.float32), 0.9

    monkeypatch.setattr(fitting, "_fit_2d_axis_aligned", fake_axis)
    monkeypatch.setattr(fitting, "_fit_2d_distorted", fake_distorted)

    standard = fitting._fit_patch(image, center, {"fit_method": "2D Gaussian"})
    distorted = fitting._fit_patch(image, center, {"fit_method": "Distorted 2D Gaussian"})

    assert calls == {"axis": 1, "distorted": 1}
    assert standard["fit_method_id"] == "gaussian_2d"
    np.testing.assert_array_equal(standard["rho"], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    assert distorted["fit_method_id"] == "distorted_gaussian_2d"
    np.testing.assert_array_equal(distorted["rho"], np.asarray([0.4, 0.0, 0.0], dtype=np.float32))


def test_fit_patch_native_2d_honors_moments_fallback() -> None:
    y, x = np.indices((7, 7), dtype=np.float32)
    image = (10.0 * np.exp(-(((y - 3.0) ** 2) + ((x - 4.0) ** 2)) / 4.0)).astype(np.float32)

    result = fitting._fit_patch(image, np.asarray([4.0, 5.0], dtype=np.float32), {"fit_method": "moments"})

    assert result["fit_method_id"] == "moments"
    assert "residual_rmse" not in result
    assert np.asarray(result["center"]).shape == (2,)
    assert np.asarray(result["sigma"]).shape == (3,)
    assert np.isnan(np.asarray(result["sigma"])[0])


def test_refine_candidates_native_2d_respects_full_fit_limit_moments_fallback(monkeypatch) -> None:
    y, x = np.indices((15, 15), dtype=np.float32)
    image = (
        20.0 * np.exp(-(((y - 5.0) ** 2) + ((x - 5.0) ** 2)) / 4.0)
        + 15.0 * np.exp(-(((y - 10.0) ** 2) + ((x - 10.0) ** 2)) / 4.0)
    ).astype(np.float32)
    calls = {"count": 0}
    original_fit_2d = fitting._fit_2d_axis_aligned

    def wrapped_fit_2d(*args, **kwargs):
        calls["count"] += 1
        return original_fit_2d(*args, **kwargs)

    monkeypatch.setattr(fitting, "_fit_2d_axis_aligned", wrapped_fit_2d)

    result = fitting.refine_candidates(
        image,
        np.asarray([[5.0, 5.0], [10.0, 10.0]], dtype=np.float32),
        np.asarray([1.0, 0.9], dtype=np.float32),
        window_radius=3,
        fit_method="2D Gaussian",
        fit_cfg={
            "fit_method": "2D Gaussian",
            "fit_fallback_method": "moments",
            "fit_background_width": 1,
            "fit_max_iterations": 100,
            "fit_tolerance": 1e-6,
            "selected_features": ["fit_snr"],
        },
        full_fit_limit=1,
    )

    assert calls["count"] == 1
    assert [row["fit_method_id"] for row in result.table] == ["gaussian_2d", "moments"]


def test_detect_image_runs_native_2d_pipeline_without_z_spacing(tmp_path) -> None:
    y, x = np.indices((31, 31), dtype=np.float32)
    image = (2.0 + 50.0 * np.exp(-(((y - 15.0) ** 2) + ((x - 16.0) ** 2)) / 8.0)).astype(np.float32)
    image_path = tmp_path / "spot_2d.tif"
    tifffile.imwrite(image_path, image)

    coords, scores, features = pipeline.detect_image(
        image_path,
        {
            "image_dimensionality": 2,
            "xy_spacing_nm": 100.0,
            "preproc_enabled": True,
            "preproc_method": "gaussian",
            "preproc_sigma_nm": 100.0,
            "norm_enabled": True,
            "norm_method": "robust_z_score",
            "background_enabled": True,
            "background_method": "rolling_box_2d",
            "background_param_nm": 300.0,
            "maxima_method": "log",
            "sigma_nm": 150.0,
            "threshold_value": 0.1,
            "maxima_min_distance_nm": 200.0,
            "fit_method": "2D Gaussian",
            "fit_window": 7,
            "fit_background_width": 1,
            "fit_max_iterations": 100,
            "fit_tolerance": 1e-6,
            "selected_features": [
                "fit_snr",
                "sigma_total_nm",
                "sigma_product_nm2",
                "residual_rmse",
                "core_minus_shell",
                "component_circularity_2d",
            ],
            "fit_cache_enabled": False,
            "stage1_cache_enabled": False,
        },
    )

    assert coords.shape[1] == 2
    assert len(coords) >= 1
    assert scores.shape == (len(coords),)
    assert list(features.columns) == [
        "fit_snr",
        "sigma_total_nm",
        "sigma_product_nm2",
        "residual_rmse",
        "core_minus_shell",
        "component_circularity_2d",
    ]
    assert "sigma_z_nm" not in features.columns


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
                    "fit_method_id": "xy_z_gaussian",
                    "fit_amplitude": 7.0,
                    "voxel_amplitude": 7.0,
                    "amplitude": 7.0,
                    "amplitude_x": 7.0,
                    "amplitude_y": 7.0,
                    "amplitude_z": 7.0,
                    "amplitude_xy": 7.0,
                    "background": 1.0,
                    "noise": 1.0,
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
        "xy_spacing_nm": 100.0,
        "z_spacing_nm": 300.0,
        "fit_cache_enabled": True,
        "stage1_cache_enabled": True,
        "feature_cache_features": ["fit_amplitude", "quality_weighted_snr"],
    }
    _, _, amplitude_features = pipeline.detect_image(
        image_path,
        {**base_cfg, "selected_features": ["fit_amplitude"]},
    )
    _, _, quality_features = pipeline.detect_image(
        image_path,
        {**base_cfg, "selected_features": ["quality_weighted_snr"]},
    )

    assert calls == {"stage1": 1, "fit": 1}
    assert list(amplitude_features.columns) == ["fit_amplitude"]
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
        "image_volume_cache_entries": 4,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "preproc_sigma_nm": 100.0,
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": True,
        "background_method": "slice_opening_2d",
        "background_param_nm": 500.0,
        "maxima_method": "log",
        "sigma_nm": 100.0,
        "maxima_min_distance_nm": 100.0,
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
        return np.full((5, 5, 5), float(cfg["preproc_sigma_nm"]), dtype=np.float32)

    def fake_detect_candidates(volume, cfg):
        calls["detect"] += 1
        assert np.all(volume == float(cfg["preproc_sigma_nm"]))
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([float(cfg["threshold_value"])], dtype=np.float32),
        )

    monkeypatch.setattr(pipeline, "apply_processing_base", fake_apply_processing_base)
    monkeypatch.setattr(pipeline, "apply_smoothing", fake_apply_smoothing)
    monkeypatch.setattr(pipeline, "detect_candidates", fake_detect_candidates)

    base_cfg = {
        "stage1_cache_enabled": True,
        "image_volume_cache_entries": 4,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": True,
        "background_method": "slice_opening_2d",
        "background_param_nm": 500.0,
        "maxima_method": "log",
        "sigma_nm": 100.0,
        "maxima_min_distance_nm": 100.0,
        "threshold_value": 1.0,
    }

    pipeline.preflight_image(image_path, label_path, {**base_cfg, "preproc_sigma_nm": 50.0}, match_distance=4.0)
    pipeline.preflight_image(image_path, label_path, {**base_cfg, "preproc_sigma_nm": 100.0}, match_distance=4.0)

    assert calls == {"base": 1, "smooth": 2, "detect": 2}
    pipeline.clear_pipeline_caches()


def test_stage1_candidate_cache_prunes_to_current_leaderboard(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "image.csv"
    label_path.write_text("x,y,z\n2,2,2\n")
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    processed = np.ones((5, 5, 5), dtype=np.float32)

    monkeypatch.setattr(pipeline, "read_volume", lambda path: raw)
    monkeypatch.setattr(pipeline, "apply_processing_base", lambda volume, cfg: processed)
    monkeypatch.setattr(pipeline, "apply_smoothing", lambda volume, cfg: processed)
    monkeypatch.setattr(
        pipeline,
        "detect_candidates",
        lambda volume, cfg: SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([float(cfg["threshold_value"])], dtype=np.float32),
        ),
    )

    base_cfg = {
        "stage1_cache_enabled": True,
        "stage1_cache_entries": 8,
        "image_volume_cache_entries": 4,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "preproc_sigma": 1.0,
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": False,
        "maxima_method": "log",
        "maxima_neighborhood": 1,
        "sigma_value": 1.0,
    }
    recipe_a = {**base_cfg, "threshold_value": 0.1}
    recipe_b = {**base_cfg, "threshold_value": 0.2}

    pipeline.preflight_image(image_path, label_path, recipe_a, match_distance=4.0)
    pipeline.preflight_image(image_path, label_path, recipe_b, match_distance=4.0)
    assert len(pipeline._STAGE1_CACHE) == 2

    removed = pipeline.prune_stage1_candidate_cache(
        allowed_signatures={pipeline.stage1_cache_signature(recipe_b)},
        image_paths=[image_path],
    )

    assert removed == 1
    assert len(pipeline._STAGE1_CACHE) == 1
    assert next(iter(pipeline._STAGE1_CACHE_CONFIG.values()))["signature"] == pipeline.stage1_cache_signature(recipe_b)
    pipeline.clear_pipeline_caches()


def test_preflight_candidate_cache_can_be_promoted_to_stage2_key(monkeypatch, tmp_path) -> None:
    pipeline.clear_pipeline_caches()
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "image.csv"
    label_path.write_text("x,y,z\n2,2,2\n")
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    processed = np.ones((5, 5, 5), dtype=np.float32)
    calls = {"detect": 0}

    monkeypatch.setattr(pipeline, "read_volume", lambda path: raw)
    monkeypatch.setattr(pipeline, "apply_processing_base", lambda volume, cfg: processed)
    monkeypatch.setattr(pipeline, "apply_smoothing", lambda volume, cfg: processed)

    def fake_detect_candidates(volume, cfg):
        calls["detect"] += 1
        return SimpleNamespace(
            coords=np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            scores=np.asarray([3.0], dtype=np.float32),
        )

    monkeypatch.setattr(pipeline, "detect_candidates", fake_detect_candidates)

    target_recipe = {
        "stage1_cache_enabled": True,
        "stage1_cache_entries": 8,
        "image_volume_cache_entries": 4,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "preproc_sigma": 1.0,
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": False,
        "maxima_method": "log",
        "maxima_neighborhood": 1,
        "sigma_value": 1.0,
        "threshold_value": 0.1,
    }
    source_recipe = {**target_recipe, "max_candidates": 1}

    pipeline.preflight_image(image_path, label_path, source_recipe, match_distance=4.0)
    assert calls["detect"] == 1

    promoted_volumes = pipeline.promote_stage1_image_volume_cache(
        recipe=source_recipe,
        image_paths=[image_path],
    )
    promoted = pipeline.promote_stage1_candidate_cache(
        source_recipe=source_recipe,
        target_recipe=target_recipe,
        image_paths=[image_path],
    )
    coords, scores = pipeline._load_stage1_candidates(image_path, target_recipe)

    assert promoted_volumes == 2
    assert promoted == 1
    assert calls["detect"] == 1
    assert coords.tolist() == [[2.0, 2.0, 2.0]]
    assert scores.tolist() == [3.0]
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


def test_predict_split_uses_model_threshold_by_default(monkeypatch, tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "predictions"
    val_dir = dataset_root / "val"
    val_dir.mkdir(parents=True)
    image_path = val_dir / "image.tif"
    label_path = val_dir / "image.csv"
    model_path = tmp_path / "model.joblib"
    image_path.write_bytes(b"placeholder")
    label_path.write_text("x,y,z\n1,1,1\n")
    embedded_recipe = {"recipe_id": "winner_recipe", "selected_features": ["amplitude"]}

    class DummyModel:
        def decision_function(self, features):
            return np.asarray([0.25], dtype=np.float32)

    def fake_load_model(path):
        assert path == model_path
        return TrainedModel(
            model=DummyModel(),
            selected_features=["amplitude"],
            best_params={},
            decision_threshold=0.5,
            recipe=embedded_recipe,
        )

    def fake_detect_image(path, recipe, label_count=None):
        assert path == image_path
        assert recipe == embedded_recipe
        return (
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.asarray([5.0], dtype=np.float32),
            pd.DataFrame({"amplitude": [7.0]}),
        )

    monkeypatch.setattr(pipeline, "load_model", fake_load_model)
    monkeypatch.setattr(pipeline, "detect_image", fake_detect_image)

    preds, _ = pipeline.predict_split(dataset_root, "val", None, model_path, output_root)

    assert preds["image"].shape == (0, 3)
    assert (output_root / "image.csv").read_text().strip() == "x,y,z,score"
