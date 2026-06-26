import numpy as np
import pytest

from mrsnappy.config import DEFAULT_NATIVE_CONFIG, deep_merge, recipe_bank
from mrsnappy.detection import detect_candidates


def test_hmax_detects_prominent_local_maximum_without_log_response() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[4, 4, 4] = 8.0
    volume[4, 4, 6] = 2.0

    result = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "maxima_min_distance_nm": 1.0,
            "xy_spacing_nm": 1.0,
            "z_spacing_nm": 1.0,
            "h_max_sigma_multiplier": 15.0,
            "h_max_sigma_mode": "std",
        },
    )

    assert result.coords.tolist() == [[4.0, 4.0, 4.0]]
    assert result.scores.tolist() == [8.0]
    assert np.array_equal(result.response, volume)


def test_hmax_sigma_multiplier_uses_noise_scaled_prominence() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[2, 2, 2] = 0.25
    volume[6, 6, 6] = 5.0

    result = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "maxima_min_distance_nm": 1.0,
            "xy_spacing_nm": 1.0,
            "z_spacing_nm": 1.0,
            "threshold_value": 0.1,
            "h_max_sigma_multiplier": 2.0,
            "h_max_sigma_mode": "std",
        },
    )

    assert result.coords.tolist() == [[6.0, 6.0, 6.0]]


def test_hmax_ignores_none_log_threshold_value() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[4, 4, 4] = 8.0

    result = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "maxima_min_distance_nm": 1.0,
            "xy_spacing_nm": 1.0,
            "z_spacing_nm": 1.0,
            "threshold_value": None,
            "h_max_sigma_multiplier": 0.01,
            "h_max_sigma_mode": "std",
        },
    )

    assert result.coords.tolist() == [[4.0, 4.0, 4.0]]


def test_log_rejects_none_threshold_value() -> None:
    with pytest.raises(ValueError, match="LoG detection requires numeric threshold_value"):
        detect_candidates(
            np.zeros((5, 5, 5), dtype=np.float32),
            {
                "maxima_method": "log",
                "sigma_value": 1.0,
                "threshold_value": None,
            },
        )


def test_hmax_rejects_absolute_h_values_and_minimum_intensity_floors() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[4, 4, 4] = 5.0

    for unsupported in ("h_max_value", "h_max_min_abs", "h_max_min_sigma_multiplier"):
        with pytest.raises(ValueError, match="Unsupported h-max parameter"):
            detect_candidates(
                volume,
                {
                    "maxima_method": "h_max",
                    "maxima_min_distance_nm": 1.0,
                    "xy_spacing_nm": 1.0,
                    "z_spacing_nm": 1.0,
                    "h_max_sigma_multiplier": 1.0,
                    unsupported: 0.5,
                },
            )


def test_hmax_constant_volume_returns_no_candidates() -> None:
    result = detect_candidates(
        np.zeros((5, 5, 5), dtype=np.float32),
        {
            "maxima_method": "h_max",
            "maxima_min_distance_nm": 1.0,
            "xy_spacing_nm": 1.0,
            "z_spacing_nm": 1.0,
            "h_max_sigma_multiplier": 1.0,
        },
    )

    assert result.coords.shape == (0, 3)
    assert result.scores.shape == (0,)


def test_detection_respects_preflight_candidate_cap() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    for idx, coord in enumerate(
        [
            (1, 1, 1),
            (1, 1, 7),
            (1, 7, 1),
            (7, 1, 1),
            (7, 7, 7),
        ],
        start=1,
    ):
        volume[coord] = float(idx)

    result = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "maxima_min_distance_nm": 1.0,
            "xy_spacing_nm": 1.0,
            "z_spacing_nm": 1.0,
            "h_max_sigma_multiplier": 0.01,
            "h_max_sigma_mode": "std",
            "max_candidates": 3,
        },
    )

    assert len(result.coords) == 3


def test_candidate_suppression_uses_physical_axis_distances() -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[1, 3, 3] = 5.0
    volume[3, 3, 3] = 4.0

    anisotropic = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "h_max_sigma_multiplier": 0.01,
            "h_max_sigma_mode": "std",
            "maxima_min_distance_nm": 300.0,
            "xy_spacing_nm": 100.0,
            "z_spacing_nm": 300.0,
        },
    )
    isotropic = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "h_max_sigma_multiplier": 0.01,
            "h_max_sigma_mode": "std",
            "maxima_min_distance_nm": 300.0,
            "xy_spacing_nm": 100.0,
            "z_spacing_nm": 100.0,
        },
    )

    assert anisotropic.coords.tolist() == [[1.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    assert isotropic.coords.tolist() == [[1.0, 3.0, 3.0]]


def test_physical_suppression_rejects_candidates_at_exact_min_distance() -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[3, 3, 1] = 5.0
    volume[3, 3, 4] = 4.0

    result = detect_candidates(
        volume,
        {
            "maxima_method": "h_max",
            "h_max_sigma_multiplier": 0.01,
            "h_max_sigma_mode": "std",
            "maxima_neighborhood": None,
            "maxima_min_distance_nm": 300.0,
            "xy_spacing_nm": 100.0,
            "z_spacing_nm": 300.0,
        },
    )

    assert result.coords.tolist() == [[3.0, 3.0, 1.0]]


def test_regional_detector_is_not_supported() -> None:
    with pytest.raises(ValueError, match="Expected 'log' or 'h_max'"):
        detect_candidates(
            np.zeros((5, 5, 5), dtype=np.float32),
            {
                "maxima_method": "simple_regional",
                "threshold_value": 0.0,
            },
        )


def test_hmax_recipe_bank_names_hmax_recipes_without_log_prefix() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    cfg = deep_merge(
        cfg,
        {
            "stage1_recipes": [
                {
                    "maxima_method": "h_max",
                    "maxima_neighborhood": 1,
                    "h_max_sigma_multiplier": 1.5,
                }
            ],
            "stage2_feature_packs": ["core_fit"],
        },
    )

    recipes = recipe_bank(cfg)

    assert len(recipes) == 12
    assert {recipe["preproc_sigma"] for recipe in recipes if recipe.get("preproc_enabled", True)} == {0.5, 1.0, 2.0}
    assert any(not recipe.get("preproc_enabled", True) for recipe in recipes)
    assert {recipe["background_param"] for recipe in recipes if recipe["background_enabled"]} == {5.0, 10.0}
    assert any(not recipe["background_enabled"] for recipe in recipes)
    assert all(recipe["maxima_method"] == "h_max" for recipe in recipes)
    assert all(recipe["recipe_id"].startswith("hmax_robust_h1p5_n1_") for recipe in recipes)
    assert all("log_" not in recipe["recipe_id"] for recipe in recipes)


def test_log_response_is_sigma_squared_and_robust_z_normalized(monkeypatch) -> None:
    response = np.zeros((3, 3, 3), dtype=np.float32)
    response[1, 1, 1] = 10.0
    response[1, 1, 2] = 1.0

    def fake_gaussian_laplace(volume, sigma):
        assert sigma == (2.0, 2.0, 2.0)
        return -response / 4.0

    monkeypatch.setattr("mrsnappy.detection.ndi.gaussian_laplace", fake_gaussian_laplace)

    result = detect_candidates(
        np.zeros((3, 3, 3), dtype=np.float32),
        {
            "maxima_method": "log",
            "sigma_nm": 2.0,
            "maxima_min_distance_nm": 1.0,
            "xy_spacing_nm": 1.0,
            "z_spacing_nm": 1.0,
            "threshold_value": 3.0,
        },
    )

    assert result.coords.tolist() == [[1.0, 1.0, 1.0]]
    assert result.response[1, 1, 1] > 3.0
