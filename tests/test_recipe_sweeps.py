import json

import pytest

from mrsnappy.config import (
    DEFAULT_NATIVE_CONFIG,
    FEATURE_PACKS,
    FITTING_MODES,
    STAGE1_DETECTOR_PRESETS,
    STAGE2_FEATURE_PACK_NAMES,
    deep_merge,
    load_config,
    recipe_bank,
    stage2_recipe_bank,
)


def _default_recipe_bank():
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    return recipe_bank(cfg)


def test_stage1_recipes_is_the_only_explicit_recipe_config_key(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"recipes": [{"recipe_id": "old_alias"}]}))

    with pytest.raises(ValueError, match="Unsupported top-level config key\\(s\\): recipes"):
        load_config(config_path)


def test_default_stage1_processing_sweeps_match_published_grid() -> None:
    recipes = _default_recipe_bank()
    smoothing = sorted({float(recipe["preproc_sigma"]) for recipe in recipes if recipe.get("preproc_enabled", True)})
    smoothing_off = any(not recipe.get("preproc_enabled", True) for recipe in recipes)
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
    assert smoothing_off
    assert background == [5.0, 10.0]
    assert background_methods == {"rolling_box_3d"}
    assert background_off


def test_physical_stage1_processing_lists_accept_off_without_voxel_lists(tmp_path) -> None:
    config_path = tmp_path / "physical_processing_off.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_detector_set": "hmax",
                "stage1_maxima_neighborhoods": [],
                "stage1_maxima_min_distances_nm": [128.866],
                "stage1_hmax_multipliers": [1.0],
                "stage1_smoothing_sigmas_nm": ["off", 128.866],
                "stage1_background_radii_nm": ["off", 644.33],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 4
    assert any(not recipe.get("preproc_enabled", True) for recipe in recipes)
    assert any(float(recipe.get("preproc_sigma_nm") or 0.0) == pytest.approx(128.866) for recipe in recipes)
    assert any(not recipe.get("background_enabled", True) for recipe in recipes)
    assert any(float(recipe.get("background_param_nm") or 0.0) == pytest.approx(644.33) for recipe in recipes)


def test_physical_stage1_detector_sweeps_voxel_and_physical_neighborhood_alternatives(tmp_path) -> None:
    config_path = tmp_path / "physical_neighborhood_processing.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_detector_set": "hmax",
                "stage1_maxima_neighborhoods": [1, 2],
                "stage1_maxima_min_distances_nm": [386.598],
                "stage1_hmax_multipliers": [1.0],
                "stage1_smoothing_sigmas_nm": ["off"],
                "stage1_background_radii_nm": ["off"],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 3
    voxel_recipes = [recipe for recipe in recipes if recipe.get("maxima_min_distance_nm") is None]
    physical_recipes = [recipe for recipe in recipes if recipe.get("maxima_min_distance_nm") is not None]
    assert sorted({recipe["maxima_neighborhood"] for recipe in voxel_recipes}) == [1, 2]
    assert len(physical_recipes) == 1
    assert physical_recipes[0]["maxima_neighborhood"] is None
    assert physical_recipes[0]["maxima_min_distance_nm"] == 386.598
    assert len({recipe["recipe_id"] for recipe in recipes}) == 3
    assert len({recipe["stage1_dedup_key"] for recipe in recipes}) == 3


def test_stage1_processing_sweeps_voxel_and_physical_unit_alternatives(tmp_path) -> None:
    config_path = tmp_path / "mixed_unit_processing.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_detector_set": "hmax",
                "stage1_maxima_neighborhoods": [1],
                "stage1_maxima_min_distances_nm": [386.598],
                "stage1_hmax_multipliers": [1.0],
                "stage1_smoothing_sigmas": ["off", 1.0],
                "stage1_smoothing_sigmas_nm": [128.866],
                "stage1_background_radii": ["off", 5.0],
                "stage1_background_radii_nm": [644.33],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 18
    assert any(recipe["maxima_neighborhood"] == 1 and recipe.get("maxima_min_distance_nm") is None for recipe in recipes)
    assert any(recipe["maxima_neighborhood"] is None and recipe.get("maxima_min_distance_nm") == 386.598 for recipe in recipes)
    assert any(not recipe["preproc_enabled"] for recipe in recipes)
    assert any(recipe.get("preproc_sigma") == 1.0 for recipe in recipes)
    assert any(recipe.get("preproc_sigma_nm") == 128.866 for recipe in recipes)
    assert any(not recipe["background_enabled"] for recipe in recipes)
    assert any(recipe.get("background_param") == 5.0 for recipe in recipes)
    assert any(recipe.get("background_param_nm") == 644.33 for recipe in recipes)


def test_default_hmax_sweeps_match_published_grid() -> None:
    recipes = _default_recipe_bank()
    hmax_multiplier = sorted({float(recipe["h_max_sigma_multiplier"]) for recipe in recipes})
    hmax_neighborhood = sorted({int(recipe["maxima_neighborhood"]) for recipe in recipes})
    hmax_mode = {str(recipe["h_max_sigma_mode"]) for recipe in recipes}

    assert {recipe["maxima_method"] for recipe in recipes} == {"h_max"}
    assert hmax_multiplier == [0.5, 1.0, 1.5, 2.0, 3.0]
    assert hmax_neighborhood == [1, 2]
    assert hmax_mode == {"robust"}


def test_log_detector_set_sweeps_match_published_grid(tmp_path) -> None:
    config_path = tmp_path / "log_config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_detector_set": "log",
                "stage1_log_sigmas": [1.0, 2.0, 3.0],
                "stage1_log_thresholds": [0.5, 1.0, 1.5, 2.0, 3.0],
                "stage1_maxima_neighborhoods": [1, 2],
                "stage1_smoothing_sigmas": [0.5],
                "stage1_background_radii": [],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert sorted({float(recipe["sigma_value"]) for recipe in recipes}) == [1.0, 2.0, 3.0]
    assert sorted({float(recipe["threshold_value"]) for recipe in recipes}) == [0.5, 1.0, 1.5, 2.0, 3.0]
    assert sorted({int(recipe["maxima_neighborhood"]) for recipe in recipes}) == [1, 2]


def test_default_sweep_has_unique_recipe_ids_and_expected_size() -> None:
    recipes = _default_recipe_bank()
    recipe_ids = [recipe["recipe_id"] for recipe in recipes]

    assert len(recipes) == 120
    assert len(recipe_ids) == len(set(recipe_ids))
    assert len({recipe["stage1_dedup_key"] for recipe in recipes}) == len(recipes)


def test_stage1_recipe_bank_deduplicates_equivalent_candidate_generation_configs(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
                {
                    "stage1_background_radii": [],
                    "stage1_background_radii_nm": [],
                    "stage1_smoothing_sigmas": [],
                    "stage1_smoothing_sigmas_nm": [],
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
    assert len(STAGE1_DETECTOR_PRESETS["log"]) == 30
    assert len(STAGE1_DETECTOR_PRESETS["hmax"]) == 10


def test_hmax_detector_set_uses_explicit_matrix_values(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_detector_set": "hmax",
                "stage1_maxima_neighborhoods": [1, 2],
                "stage1_hmax_multipliers": [0.5, 1.0, 1.5, 2.0, 3.0],
                "stage1_smoothing_sigmas": [0.5, 1.0, 2.0],
                "stage1_background_radii": ["off", 5.0, 10.0],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 90
    assert {recipe["maxima_method"] for recipe in recipes} == {"h_max"}
    assert sorted({recipe["h_max_sigma_multiplier"] for recipe in recipes}) == [0.5, 1.0, 1.5, 2.0, 3.0]
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


def test_2d_stage1_default_box_background_uses_rolling_box_2d(tmp_path) -> None:
    config_path = tmp_path / "background_method_2d_default.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                }
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert cfg["stage1_background_method"] == "rolling_box_2d"
    assert {recipe["background_method"] for recipe in recipes if recipe["background_enabled"]} == {"rolling_box_2d"}


def test_stage1_rolling_box_2d_background_method_can_be_configured(tmp_path) -> None:
    config_path = tmp_path / "background_method_rolling_box_2d.json"
    config_path.write_text(json.dumps({"stage1_background_method": "rolling_box_2d"}))
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert {recipe["background_method"] for recipe in recipes if recipe["background_enabled"]} == {"rolling_box_2d"}


def test_physical_stage1_fields_do_not_inherit_unspecified_voxel_sweeps(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_detector_set": "log",
                "stage1_log_sigmas_nm": [130.0],
                "stage1_maxima_min_distances_nm": [130.0],
                "stage1_log_thresholds": [0.5],
                "stage1_smoothing_sigmas_nm": [130.0],
                "stage1_background_radii_nm": [],
            }
        )
    )
    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert len(recipes) == 1
    assert cfg["stage1_log_sigmas"] == []
    assert cfg["stage1_maxima_neighborhoods"] == []
    assert cfg["stage1_smoothing_sigmas"] == []
    assert cfg["stage1_background_radii"] == []
    assert recipes[0]["sigma_value"] is None
    assert recipes[0]["maxima_neighborhood"] is None
    assert recipes[0]["sigma_nm"] == 130.0
    assert recipes[0]["maxima_min_distance_nm"] == 130.0
    assert recipes[0]["preproc_sigma"] is None
    assert recipes[0]["preproc_sigma_nm"] == 130.0


def test_hmax_recipe_ids_include_sigma_mode_to_avoid_ambiguous_runs(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
                {
                    "stage1_background_radii": [],
                    "stage1_background_radii_nm": [],
                    "stage1_smoothing_sigmas": [],
                    "stage1_smoothing_sigmas_nm": [],
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
                    "stage1_background_radii": [],
                    "stage1_background_radii_nm": [],
                    "stage1_smoothing_sigmas": [],
                    "stage1_smoothing_sigmas_nm": [],
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
                    "stage1_recipes": [
                        {"recipe_id": "duplicate", "sigma_value": 1.0, "threshold_value": 1.0},
                        {"recipe_id": "duplicate", "sigma_value": 2.0, "threshold_value": 1.0},
                    ]
                }
        )
    )
    cfg = load_config(config_path)

    with pytest.raises(ValueError, match="recipe IDs must be unique"):
        recipe_bank(cfg)


def test_stage2_uses_recipe_fitting_mode_without_separate_fit_sweep() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    stage1_recipes = recipe_bank(cfg)
    recipes = stage2_recipe_bank(cfg, stage1_recipes[:1])

    assert all(recipe["fit_method"] == "2D (XY) + 1D (Z) Gaussian" for recipe in recipes)
    assert all(recipe["fit_window"] == 7 for recipe in recipes)
    assert all("fit_radius" not in recipe for recipe in recipes)
    assert {recipe["stage1_recipe_id"] for recipe in recipes} == {stage1_recipes[0]["recipe_id"]}


def test_fit_background_method_and_poly_degree_are_not_user_parameters(tmp_path) -> None:
    for key, value in (
        ("fit_background_method", "Mean Surrounding Subtraction"),
        ("fit_poly_degree", 2),
        ("fit_fallback_method", "3D Gaussian"),
    ):
        config_path = tmp_path / f"{key}.json"
        config_path.write_text(json.dumps({"pipeline_defaults": {key: value}}))
        with pytest.raises(ValueError, match=f"Unsupported pipeline_defaults key\\(s\\): {key}"):
            load_config(config_path)


def test_backend_candidate_limit_controls_are_not_user_parameters(tmp_path) -> None:
    for key, value in (
        ("prefit_prune_enabled", True),
        ("prefit_rank_radius", 1),
        ("prefit_rank_bg_width", 1),
        ("prefit_nms_distance", 0.0),
        ("prefit_labeled_candidates_per_label", 64.0),
        ("prefit_labeled_min_candidates", 320),
        ("prefit_unlabeled_candidates_per_expected_label", 80.0),
        ("expected_labels_per_image", 20.0),
        ("full_fit_labeled_candidates_per_label", 16.0),
        ("full_fit_labeled_min_candidates", 384),
        ("full_fit_unlabeled_candidates_per_expected_label", 20.0),
    ):
        config_path = tmp_path / f"{key}.json"
        config_path.write_text(json.dumps({"pipeline_defaults": {key: value}}))
        with pytest.raises(ValueError, match=f"Unsupported pipeline_defaults key\\(s\\): {key}"):
            load_config(config_path)


def test_unused_processing_controls_are_not_user_parameters(tmp_path) -> None:
    for key, value in (
        ("norm_param1", 0.0),
        ("norm_param2", 1.0),
        ("norm_param3", 0.0),
        ("background_mode", "3D"),
        ("background_projection", "Max"),
        ("background_scale", False),
    ):
        config_path = tmp_path / f"{key}.json"
        config_path.write_text(json.dumps({"pipeline_defaults": {key: value}}))
        with pytest.raises(ValueError, match=f"Unsupported pipeline_defaults key\\(s\\): {key}"):
            load_config(config_path)


def test_profiling_controls_are_validated(tmp_path) -> None:
    invalid_payloads = [
        {"profiling": {"enabled": "yes"}},
        {"profiling": {"train_image_count": 0}},
        {"profiling": {"val_image_count": False}},
        {"profiling": {"gt_intensity_radius": -1}},
        {"profiling": {"sparse_label_mean_max": 100.0, "dense_label_mean_min": 50.0}},
        {"profiling": {"stage1_augmentation_enabled": 1}},
        {"profiling": {"apply_runtime_pruning_to_explicit_recipes": "false"}},
    ]
    config_path = tmp_path / "bad_profiling.json"

    for payload in invalid_payloads:
        config_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            load_config(config_path)


def test_legacy_spacing_names_are_not_user_parameters(tmp_path) -> None:
    for key in ("xy_spacing", "z_spacing"):
        config_path = tmp_path / f"{key}.json"
        config_path.write_text(json.dumps({"pipeline_defaults": {key: 1.0}}))
        with pytest.raises(ValueError, match=f"Unsupported pipeline_defaults key\\(s\\): {key}"):
            load_config(config_path)


def test_fit_iterations_and_tolerance_remain_optional_user_parameters(tmp_path) -> None:
    config_path = tmp_path / "fit_controls.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline_defaults": {
                    "fit_max_iterations": 123,
                    "fit_tolerance": 1e-5,
                }
            }
        )
    )
    cfg = load_config(config_path)

    assert cfg["pipeline_defaults"]["fit_max_iterations"] == 123
    assert cfg["pipeline_defaults"]["fit_tolerance"] == 1e-5


def test_fit_controls_are_validated(tmp_path) -> None:
    config_path = tmp_path / "bad_fit_controls.json"
    invalid_payloads = [
        {"pipeline_defaults": {"fit_method": "Radial Symmetry"}},
        {"pipeline_defaults": {"fit_method": "moments"}},
        {"pipeline_defaults": {"fit_window": 6}},
        {"pipeline_defaults": {"fit_background_width": -1}},
        {"pipeline_defaults": {"fit_max_iterations": 0}},
        {"pipeline_defaults": {"fit_tolerance": 0}},
    ]

    for payload in invalid_payloads:
        config_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            load_config(config_path)


def test_match_distance_nm_requires_physical_spacing(tmp_path) -> None:
    config_path = tmp_path / "match_distance_nm.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {"xy_spacing_nm": 100.0, "z_spacing_nm": 300.0},
            }
        )
    )
    cfg = load_config(config_path)

    assert cfg["match_distance_nm"] == 200.0
    assert cfg["match_distance"] is None

    config_path.write_text(
        json.dumps(
            {
                "match_distance": 3.0,
                "match_distance_nm": 200.0,
                "pipeline_defaults": {"xy_spacing_nm": 100.0, "z_spacing_nm": 300.0},
            }
        )
    )
    with pytest.raises(ValueError, match="exactly one of match_distance or match_distance_nm"):
        load_config(config_path)

    config_path.write_text(json.dumps({"match_distance_nm": 200.0}))
    with pytest.raises(ValueError, match="pipeline_defaults.xy_spacing_nm"):
        load_config(config_path)


def test_2d_config_uses_xy_spacing_without_z_spacing(tmp_path) -> None:
    config_path = tmp_path / "match_distance_2d.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                },
            }
        )
    )

    cfg = load_config(config_path)
    recipes = stage2_recipe_bank(cfg, recipe_bank(cfg)[:1])
    selected = recipes[0]["selected_features"]

    assert cfg["pipeline_defaults"]["image_dimensionality"] == 2
    assert cfg["pipeline_defaults"]["fit_method"] == "2D Gaussian"
    assert cfg["stage1_background_method"] == "rolling_box_2d"
    assert "sigma_total_nm" in selected
    assert "sigma_product_nm2" in selected
    assert "sigma_product_nm3" not in selected
    assert "sigma_z_nm" not in selected
    assert "sigma_axial_ratio" not in selected


def test_2d_config_generated_default_fit_method_is_auto_corrected(tmp_path) -> None:
    config_path = tmp_path / "generated_default_2d.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                    "fit_method": "2D (XY) + 1D (Z) Gaussian",
                },
            }
        )
    )

    cfg = load_config(config_path)

    assert cfg["pipeline_defaults"]["fit_method"] == "2D Gaussian"


def test_2d_config_rejects_explicit_3d_fit_method(tmp_path) -> None:
    config_path = tmp_path / "bad_2d_fit_method.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                    "fit_method": "3D Gaussian",
                },
            }
        )
    )

    with pytest.raises(ValueError, match="incompatible with image_dimensionality=2"):
        load_config(config_path)


def test_2d_config_accepts_distorted_2d_fit_method(tmp_path) -> None:
    config_path = tmp_path / "distorted_2d_fit_method.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                    "fit_method": "Distorted 2D Gaussian",
                },
            }
        )
    )

    cfg = load_config(config_path)
    recipes = stage2_recipe_bank(cfg, recipe_bank(cfg)[:1])
    morphology_recipe = next(recipe for recipe in recipes if recipe["feature_pack_name"] == "core_morphology")

    assert cfg["pipeline_defaults"]["fit_method"] == "Distorted 2D Gaussian"
    assert "rho_lateral_abs" in morphology_recipe["selected_features"]
    assert "covariance_elongation" in morphology_recipe["selected_features"]
    assert "rho_axial_energy" not in morphology_recipe["selected_features"]
    assert "sigma_z_nm" not in morphology_recipe["selected_features"]


def test_2d_stage1_recipe_without_fit_method_inherits_2d_fit(tmp_path) -> None:
    config_path = tmp_path / "recipe_2d_default_fit.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 3,
                    "xy_spacing_nm": 100.0,
                    "z_spacing_nm": 300.0,
                },
                "stage1_recipes": [
                    {
                        "recipe_id": "native_2d_recipe",
                        "image_dimensionality": 2,
                        "xy_spacing_nm": 100.0,
                        "maxima_method": "log",
                    }
                ],
            }
        )
    )

    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert recipes[0]["image_dimensionality"] == 2
    assert recipes[0]["fit_method"] == "2D Gaussian"
    assert "sigma_z_nm" not in recipes[0]["selected_features"]


def test_2d_stage1_recipe_rejects_explicit_3d_fit_method(tmp_path) -> None:
    config_path = tmp_path / "bad_recipe_2d_fit_method.json"
    config_path.write_text(
        json.dumps(
            {
                "stage1_recipes": [
                    {
                        "recipe_id": "bad_native_2d_recipe",
                        "image_dimensionality": 2,
                        "fit_method": "Distorted 3D Gaussian",
                        "maxima_method": "log",
                    }
                ],
            }
        )
    )
    cfg = load_config(config_path)

    with pytest.raises(ValueError, match="incompatible with image_dimensionality=2"):
        recipe_bank(cfg)


def test_current_fit_method_ids_are_accepted(tmp_path) -> None:
    config_path = tmp_path / "fit_method_id.json"
    config_path.write_text(json.dumps({"pipeline_defaults": {"fit_method": "distorted_gaussian_3d"}}))

    cfg = load_config(config_path)
    recipes = recipe_bank(cfg)

    assert {recipe["fit_method"] for recipe in recipes} == {"Distorted 3D Gaussian"}


def test_current_distorted_2d_fit_method_id_is_accepted(tmp_path) -> None:
    config_path = tmp_path / "distorted_2d_fit_method_id.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                    "fit_method": "distorted_gaussian_2d",
                },
            }
        )
    )

    cfg = load_config(config_path)

    assert cfg["pipeline_defaults"]["fit_method"] == "Distorted 2D Gaussian"


def test_negative_to_positive_ratio_is_not_a_user_parameter(tmp_path) -> None:
    config_path = tmp_path / "negative_ratio.json"
    config_path.write_text(json.dumps({"pipeline_defaults": {"negative_to_positive_ratio": 8.0}}))

    with pytest.raises(ValueError, match="Unsupported pipeline_defaults key\\(s\\): negative_to_positive_ratio"):
        load_config(config_path)


def test_runtime_cache_uses_image_volume_cache_entries_name(tmp_path) -> None:
    config_path = tmp_path / "cache_config.json"
    config_path.write_text(json.dumps({"runtime_cache": {"image_volume_cache_entries": 7}}))
    cfg = load_config(config_path)

    assert cfg["runtime_cache"]["image_volume_cache_entries"] == 7
    assert cfg["pipeline_defaults"]["image_volume_cache_entries"] == 7

    old_config_path = tmp_path / "old_cache_config.json"
    old_config_path.write_text(json.dumps({"runtime_cache": {"preprocess_cache_entries": 7}}))
    with pytest.raises(ValueError, match="Unsupported runtime_cache key\\(s\\): preprocess_cache_entries"):
        load_config(old_config_path)


def test_top_level_cache_keys_are_not_public_config(tmp_path) -> None:
    config_path = tmp_path / "top_level_cache_config.json"
    config_path.write_text(json.dumps({"stage1_cache_entries": 7}))

    with pytest.raises(ValueError, match="Unsupported top-level config key\\(s\\): stage1_cache_entries"):
        load_config(config_path)


def test_runtime_cache_values_are_validated(tmp_path) -> None:
    config_path = tmp_path / "bad_cache_config.json"
    config_path.write_text(json.dumps({"runtime_cache": {"stage1_cache_enabled": None}}))
    with pytest.raises(ValueError, match="runtime_cache.stage1_cache_enabled must be either true or false"):
        load_config(config_path)

    config_path.write_text(json.dumps({"runtime_cache": {"fit_cache_entries": -1}}))
    with pytest.raises(ValueError, match="runtime_cache.fit_cache_entries must be a non-negative integer"):
        load_config(config_path)


def test_stage2_sweeps_all_explicit_feature_packs() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    recipes = stage2_recipe_bank(cfg, recipe_bank(cfg)[:1])
    pack_names = {recipe["feature_pack_name"] for recipe in recipes}

    assert cfg["stage2_feature_packs"] == STAGE2_FEATURE_PACK_NAMES
    assert pack_names == set(STAGE2_FEATURE_PACK_NAMES)
    assert all(set(pack) == {"name", "features"} for pack in FEATURE_PACKS.values())
    assert "quality_weighted_snr" in FEATURE_PACKS["core_fit"]["features"]
    assert "core_shell_snr" in FEATURE_PACKS["core_contrast"]["features"]
    assert "component_sphericity_3d" in FEATURE_PACKS["core_morphology"]["features"]
    assert "rho_axial_energy" in FEATURE_PACKS["core_morphology"]["features"]
    assert "log_integrated_intensity" in FEATURE_PACKS["full_interpretable"]["features"]
    assert all("score_raw" not in pack["features"] for pack in FEATURE_PACKS.values())


def test_stage2_feature_packs_resolve_modern_2d_feature_names(tmp_path) -> None:
    config_path = tmp_path / "feature_pack_2d_names.json"
    config_path.write_text(
        json.dumps(
            {
                "match_distance_nm": 200.0,
                "pipeline_defaults": {
                    "image_dimensionality": 2,
                    "xy_spacing_nm": 100.0,
                    "fit_method": "Distorted 2D Gaussian",
                },
            }
        )
    )
    cfg = load_config(config_path)
    recipes = stage2_recipe_bank(cfg, recipe_bank(cfg)[:1])
    full = next(recipe for recipe in recipes if recipe["feature_pack_name"] == "full_interpretable")
    selected = set(full["selected_features"])

    assert {
        "sigma_product_nm2",
        "component_pixel_area",
        "component_boundary_px",
        "component_boundary_to_area_ratio",
        "component_circularity_2d",
        "component_convex_size_px",
        "component_solidity_2d",
        "component_elongation_2d",
    } <= selected
    assert "xy_core_minus_shell" in selected
    assert "z_core_minus_shell" not in selected
    assert "sigma_product_nm3" not in selected
    assert "component_sphericity_3d" not in selected
    assert "rho_lateral_abs" in selected


def test_cross_validation_mode_is_reserved_until_implemented(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("optimization_mode: cross_validation\n")
    with pytest.raises(NotImplementedError, match="planned but not implemented"):
        load_config(config_path)


def test_known_fitting_modes_are_named_for_documentation() -> None:
    assert FITTING_MODES["gaussian_2d"]["fit_method"] == "2D Gaussian"
    assert FITTING_MODES["distorted_gaussian_2d"]["fit_method"] == "Distorted 2D Gaussian"
    assert FITTING_MODES["distorted_gaussian_3d"]["fit_method"] == "Distorted 3D Gaussian"
    assert FITTING_MODES["gaussian_3d"]["fit_method"] == "3D Gaussian"
    assert FITTING_MODES["xy_z_gaussian"]["fit_method"] == "2D (XY) + 1D (Z) Gaussian"
    assert {mode["fit_window"] for mode in FITTING_MODES.values()} == {7}
