import json

import pytest

from mrsnappy.config import (
    DEFAULT_NATIVE_CONFIG,
    FEATURE_PACKS,
    STAGE1_DETECTOR_PRESETS,
    STAGE2_FEATURE_PACK_NAMES,
    STAGE2_FIT_VARIANTS,
    deep_merge,
    load_config,
    recipe_bank,
    stage2_recipe_bank,
)


def _default_recipe_bank():
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    return recipe_bank(cfg)


def test_default_stage1_processing_sweeps_match_published_grid() -> None:
    recipes = _default_recipe_bank()
    smoothing = sorted({float(recipe["preproc_sigma"]) for recipe in recipes})
    background = sorted(
        {
            float(recipe["background_param"])
            for recipe in recipes
            if recipe.get("background_enabled") and recipe.get("background_method") == "rolling_box_3d"
        }
    )
    background_methods = {
        recipe.get("background_method")
        for recipe in recipes
        if recipe.get("background_enabled")
    }
    background_off = any(not recipe.get("background_enabled") for recipe in recipes)

    assert smoothing == [0.5, 1.0, 2.0]
    assert background == [5.0, 10.0]
    assert background_methods == {"rolling_box_3d"}
    assert background_off


def test_default_log_sweeps_match_published_grid() -> None:
    recipes = _default_recipe_bank()
    log_sigma = sorted({float(recipe["sigma_value"]) for recipe in recipes if recipe["maxima_method"] == "log"})
    log_threshold = sorted({float(recipe["threshold_value"]) for recipe in recipes if recipe["maxima_method"] == "log"})
    log_neighborhood = sorted({int(recipe["maxima_neighborhood"]) for recipe in recipes if recipe["maxima_method"] == "log"})

    assert log_sigma == [1.0, 2.0, 3.0]
    assert log_threshold == [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    assert log_neighborhood == [1, 2]


def test_default_sweep_has_unique_recipe_ids_and_expected_size() -> None:
    recipes = _default_recipe_bank()
    recipe_ids = [recipe["recipe_id"] for recipe in recipes]

    assert len(recipes) == 432
    assert len(recipe_ids) == len(set(recipe_ids))
    assert len({recipe["stage1_dedup_key"] for recipe in recipes}) == len(recipes)


def test_stage1_recipe_bank_deduplicates_equivalent_candidate_generation_configs(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_background_params": [],
                "stage1_smoothing_sigmas": [],
                "stage1_recipes": [
                    {
                        "recipe_id": "kept",
                        "maxima_method": "log",
                        "sigma_value": 1.0,
                        "threshold_value": 0.5,
                        "maxima_neighborhood": 1,
                    },
                    {
                        "recipe_id": "duplicate_alias",
                        "maxima_method": "log",
                        "sigma_value": 1,
                        "threshold_value": 0.5,
                        "maxima_neighborhood": 1,
                    },
                ],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 1
    assert recipes[0]["recipe_id"] == "kept"
    assert recipes[0]["deduplicated_duplicate_count"] == 1
    assert recipes[0]["deduplicated_from_recipe_ids"] == ["duplicate_alias"]


def test_stage1_detector_presets_are_log_or_hmax_only() -> None:
    assert sorted(STAGE1_DETECTOR_PRESETS) == ["hmax", "log"]
    assert len(STAGE1_DETECTOR_PRESETS["log"]) == 48
    assert len(STAGE1_DETECTOR_PRESETS["hmax"]) == 14


def test_hmax_detector_set_uses_shared_processing_grid(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"stage1_detector_set": "hmax"}))
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 126
    assert {recipe["maxima_method"] for recipe in recipes} == {"h_max"}
    assert sorted({recipe["h_max_sigma_multiplier"] for recipe in recipes}) == [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    assert sorted({recipe["maxima_neighborhood"] for recipe in recipes}) == [1, 2]
    assert sorted({recipe["preproc_sigma"] for recipe in recipes}) == [0.5, 1.0, 2.0]
    assert sorted({recipe["background_param"] for recipe in recipes if recipe["background_enabled"]}) == [5.0, 10.0]
    assert {recipe["background_method"] for recipe in recipes if recipe["background_enabled"]} == {"rolling_box_3d"}
    assert any(not recipe["background_enabled"] for recipe in recipes)


def test_stage1_background_method_can_be_configured(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"stage1_background_method": "rolling_ball_2d"}))
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert {recipe["background_method"] for recipe in recipes if recipe["background_enabled"]} == {"rolling_ball_2d"}


def test_hmax_recipe_ids_include_sigma_mode_to_avoid_ambiguous_runs(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_background_params": [],
                "stage1_smoothing_sigmas": [],
                "stage1_recipes": [
                    {
                        "maxima_method": "h_max",
                        "h_max_sigma_multiplier": 1.0,
                        "h_max_sigma_mode": "robust",
                        "maxima_neighborhood": 1,
                    },
                    {
                        "maxima_method": "h_max",
                        "h_max_sigma_multiplier": 1.0,
                        "h_max_sigma_mode": "std",
                        "maxima_neighborhood": 1,
                    },
                ],
            }
        )
    )
    cfg = load_config(config_path)
    recipe_ids = [recipe["recipe_id"] for recipe in recipe_bank(cfg)]

    assert any(recipe_id.startswith("hmax_robust_h1p0_n1_") for recipe_id in recipe_ids)
    assert any(recipe_id.startswith("hmax_std_h1p0_n1_") for recipe_id in recipe_ids)
    assert len(recipe_ids) == len(set(recipe_ids))


def test_custom_hmax_alias_normalizes_to_h_max(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_background_params": [],
                "stage1_smoothing_sigmas": [],
                "stage1_recipes": [
                    {
                        "maxima_method": "hmax",
                        "h_max_sigma_multiplier": 1.0,
                        "h_max_sigma_mode": "robust",
                        "maxima_neighborhood": 1,
                    },
                ],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 1
    assert recipes[0]["maxima_method"] == "h_max"
    assert recipes[0]["recipe_id"].startswith("hmax_robust_h1p0_n1_")


def test_duplicate_explicit_recipe_ids_are_rejected(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {"recipe_id": "duplicate", "sigma_value": 1.0, "threshold_value": 1.0},
                    {"recipe_id": "duplicate", "sigma_value": 2.0, "threshold_value": 1.0},
                ]
            }
        )
    )
    cfg = load_config(config_path)

    with pytest.raises(ValueError, match="recipe IDs must be unique"):
        recipe_bank(cfg)


def test_default_stage2_fit_is_2d_xy_plus_1d_z_window_seven() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    recipes = stage2_recipe_bank(cfg, recipe_bank(cfg))

    assert cfg["stage2_fit_variants"] == ["xy_z_gaussian"]
    assert all(recipe["fit_method"] == "2D (XY) + 1D (Z) Gaussian" for recipe in recipes)
    assert all(recipe["fit_window"] == 7 for recipe in recipes)
    assert all("fit_radius" not in recipe for recipe in recipes)


def test_stage2_sweeps_all_explicit_feature_packs() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    recipes = stage2_recipe_bank(cfg, recipe_bank(cfg)[:1])
    pack_names = {recipe["feature_pack_name"] for recipe in recipes}

    assert cfg["stage2_feature_packs"] == STAGE2_FEATURE_PACK_NAMES
    assert pack_names == set(STAGE2_FEATURE_PACK_NAMES)
    assert all(set(pack) == {"name", "features"} for pack in FEATURE_PACKS.values())
    assert "quality_weighted_snr" in FEATURE_PACKS["curated_balanced"]["features"]
    assert "quality_vs_size_penalty" in FEATURE_PACKS["shape_localization"]["features"]
    assert "distortion_energy" in FEATURE_PACKS["distortion"]["features"]
    assert "log_integrated_intensity" in FEATURE_PACKS["full"]["features"]


def test_3d_fit_variants_remain_available_for_custom_configs() -> None:
    assert STAGE2_FIT_VARIANTS["distorted_gaussian_3d"]["fit_method"] == "Distorted 3D Gaussian"
    assert STAGE2_FIT_VARIANTS["gaussian_3d"]["fit_method"] == "3D Gaussian"
    assert STAGE2_FIT_VARIANTS["xy_z_gaussian"]["fit_method"] == "2D (XY) + 1D (Z) Gaussian"
    assert {variant["fit_window"] for variant in STAGE2_FIT_VARIANTS.values()} == {7}
