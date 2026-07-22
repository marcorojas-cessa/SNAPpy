import numpy as np
import pandas as pd
import pytest

import mrsnappy.optimizer as optimizer
from mrsnappy.config import DEFAULT_NATIVE_CONFIG, deep_merge, load_config, recipe_bank
from mrsnappy.optimizer import (
    _FINALIST_SORT_ASCENDING,
    _FINALIST_SORT_COLUMNS,
    _add_stage2_selection_columns,
    _apply_preflight_decision_columns,
    _augment_stage1_recipes,
    _expand_stage2_shortlist,
    _enforce_optimizer_plan_safety,
    _mean_per_image_candidate_ratio,
    _profile_dataset,
    _stage1_failure_reasons,
    _stage1_guardrail_progress,
    _stage2_parameter_payload,
    _stage2_shortlist_from_passed,
    _svm_parameter_payload,
    optimizer_plan,
)


def test_stage1_failure_reasons_are_explicit_and_minimal() -> None:
    reasons = _stage1_failure_reasons(
        mean_recall=0.1,
        mean_candidates=3000.0,
        max_candidates=4500,
        mean_candidate_ratio=101.0,
        preflight_cfg={
            "min_stage1_recall_mean": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 4000,
            "max_candidate_ratio_cap_mean": 100.0,
        },
    )

    assert reasons == [
        "mean recall on labeled images 0.1000 < minimum 0.2500",
        "mean candidates 3000.0 > maximum 2500.0",
        "single-image candidates 4500 > maximum 4000",
        "candidate/ground-truth ratio 101.00 > maximum 100.00",
    ]


def test_absent_candidate_ratio_cap_disables_ratio_guardrail() -> None:
    reasons = _stage1_failure_reasons(
        mean_recall=1.0,
        mean_candidates=100.0,
        max_candidates=100,
        mean_candidate_ratio=1000.0,
        preflight_cfg={
            "min_stage1_recall_mean": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 4000,
            "max_candidate_ratio_cap_mean": None,
        },
    )

    assert reasons == []


def test_removed_candidate_ratio_cap_enabled_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"preflight": {"candidate_ratio_cap_enabled": false}}')

    with pytest.raises(ValueError, match="Unsupported preflight key"):
        load_config(config_path)


def test_stage1_n_val_images_accepts_positive_integer_or_all(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"preflight": {"stage1_n_val_images": "all"}}')
    assert load_config(config_path)["preflight"]["stage1_n_val_images"] == "all"

    config_path.write_text('{"preflight": {"stage1_n_val_images": 0}}')
    with pytest.raises(ValueError, match="stage1_n_val_images"):
        load_config(config_path)

    config_path.write_text('{"preflight": {"stage1_n_val_images": 1.5}}')
    with pytest.raises(ValueError, match="stage1_n_val_images"):
        load_config(config_path)


def test_preflight_numeric_guardrails_reject_invalid_values(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    invalid_payloads = [
        '{"preflight": {"min_stage1_recall_mean": 0}}',
        '{"preflight": {"min_stage1_recall_mean": 1.01}}',
        '{"preflight": {"max_stage1_candidates_mean": 0}}',
        '{"preflight": {"max_stage1_candidates_single": 0}}',
    ]

    for payload in invalid_payloads:
        config_path.write_text(payload)
        with pytest.raises(ValueError):
            load_config(config_path)


def test_old_min_stage1_recall_name_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"preflight": {"min_stage1_recall": 0.25}}')

    with pytest.raises(ValueError, match="min_stage1_recall_mean"):
        load_config(config_path)


def test_candidate_ratio_cap_rejects_string_values(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"preflight": {"max_candidate_ratio_cap_mean": "invalid"}}')

    with pytest.raises(ValueError, match="max_candidate_ratio_cap_mean"):
        load_config(config_path)


def test_scalar_svm_settings_do_not_clear_default_svm_sweep(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"svm_sweep": {"class_weighting": "off"}}')

    cfg = load_config(config_path)

    assert cfg["svm_sweep"]["class_weighting"] == "off"
    assert cfg["svm_sweep"]["kernels"] == DEFAULT_NATIVE_CONFIG["svm_sweep"]["kernels"]
    assert cfg["svm_sweep"]["box_constraints"] == DEFAULT_NATIVE_CONFIG["svm_sweep"]["box_constraints"]


def test_stage1_matrix_error_names_empty_required_lists(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text(
        """
        {
          "stage1_detector_set": "hmax",
          "stage1_hmax_multipliers": [1.0]
        }
        """
    )

    with pytest.raises(ValueError, match="stage1_maxima_neighborhoods or stage1_maxima_min_distances_nm"):
        load_config(config_path)


def test_stage1_ranking_config_rejects_invalid_values(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    invalid_payloads = [
        '{"stage1_ranking": {"recall_tolerance": -0.1}}',
        '{"stage1_ranking": {"recall_tolerance": 1.1}}',
        '{"optimizer": {"shortlist_top_k": 0}}',
    ]

    for payload in invalid_payloads:
        config_path.write_text(payload)
        with pytest.raises(ValueError):
            load_config(config_path)


def test_removed_optimizer_selection_margin_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"optimizer": {"selection_margin": 0.005}}')

    with pytest.raises(ValueError, match="Unsupported optimizer key"):
        load_config(config_path)


def test_removed_stage1_ranking_pool_size_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"stage1_ranking": {"ranking_pool_size": 10}}')

    with pytest.raises(ValueError, match="Unsupported stage1_ranking key"):
        load_config(config_path)


def test_stage1_candidate_ratio_is_true_per_image_mean() -> None:
    image_rows = [
        {"n_candidates": 100, "n_labels": 100},
        {"n_candidates": 10, "n_labels": 1},
    ]

    assert _mean_per_image_candidate_ratio(image_rows) == pytest.approx(5.5)


def test_stage1_candidate_ratio_ignores_empty_gt_images() -> None:
    image_rows = [
        {"n_candidates": 100, "n_labels": 100},
        {"n_candidates": 5000, "n_labels": 0},
    ]

    assert _mean_per_image_candidate_ratio(image_rows) == pytest.approx(1.0)


def test_null_preflight_guardrails_are_not_used() -> None:
    reasons = _stage1_failure_reasons(
        mean_recall=0.0,
        mean_candidates=999999.0,
        max_candidates=999999,
        mean_candidate_ratio=999999.0,
        preflight_cfg={
            "min_stage1_recall_mean": None,
            "max_stage1_candidates_mean": None,
            "max_stage1_candidates_single": None,
            "max_candidate_ratio_cap_mean": None,
        },
    )

    assert reasons == []


def test_stage1_guardrail_progress_allows_recipes_that_can_still_pass() -> None:
    progress = _stage1_guardrail_progress(
        [{"n_candidates": 1000, "n_labels": 10, "recall": 0.0}],
        total_preflight_images=4,
        total_labeled_preflight_images=4,
        preflight_cfg={
            "min_stage1_recall_mean": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 5000,
            "max_candidate_ratio_cap_mean": None,
        },
    )

    assert progress["can_still_pass"]
    assert progress["maximum_possible_mean_recall"] == pytest.approx(0.75)
    assert progress["minimum_possible_mean_candidates"] == pytest.approx(250.0)


def test_stage1_guardrail_progress_stops_on_single_image_candidate_cap() -> None:
    progress = _stage1_guardrail_progress(
        [{"n_candidates": 5001, "n_labels": 10, "recall": 1.0}],
        total_preflight_images=4,
        total_labeled_preflight_images=4,
        preflight_cfg={
            "min_stage1_recall_mean": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 5000,
            "max_candidate_ratio_cap_mean": None,
        },
    )

    assert not progress["can_still_pass"]
    assert progress["definitive_failure_reasons"] == ["single-image candidates 5001 > maximum 5000"]


def test_stage1_guardrail_progress_stops_on_unrecoverable_mean_candidates() -> None:
    progress = _stage1_guardrail_progress(
        [
            {"n_candidates": 5000, "n_labels": 10, "recall": 1.0},
            {"n_candidates": 5000, "n_labels": 10, "recall": 1.0},
            {"n_candidates": 1, "n_labels": 10, "recall": 1.0},
        ],
        total_preflight_images=4,
        total_labeled_preflight_images=4,
        preflight_cfg={
            "min_stage1_recall_mean": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 5000,
            "max_candidate_ratio_cap_mean": None,
        },
    )

    assert not progress["can_still_pass"]
    assert progress["definitive_failure_reasons"] == [
        "minimum possible mean candidates 2500.2 > maximum 2500.0"
    ]


def test_stage1_guardrail_progress_stops_on_unrecoverable_mean_recall() -> None:
    progress = _stage1_guardrail_progress(
        [
            {"n_candidates": 10, "n_labels": 10, "recall": 0.0},
            {"n_candidates": 10, "n_labels": 10, "recall": 0.0},
        ],
        total_preflight_images=4,
        total_labeled_preflight_images=4,
        preflight_cfg={
            "min_stage1_recall_mean": 0.75,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 5000,
            "max_candidate_ratio_cap_mean": None,
        },
    )

    assert not progress["can_still_pass"]
    assert progress["definitive_failure_reasons"] == [
        "maximum possible mean recall on labeled images 0.5000 < minimum 0.7500"
    ]


def test_stage1_guardrail_progress_excludes_empty_gt_images_from_recall_ceiling() -> None:
    progress = _stage1_guardrail_progress(
        [
            {"n_candidates": 0, "n_labels": 0, "recall": 0.0},
            {"n_candidates": 0, "n_labels": 0, "recall": 0.0},
        ],
        total_preflight_images=3,
        total_labeled_preflight_images=1,
        preflight_cfg={
            "min_stage1_recall_mean": 0.4,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 5000,
            "max_candidate_ratio_cap_mean": 500.0,
        },
    )

    assert progress["can_still_pass"]
    assert progress["maximum_possible_mean_recall"] == pytest.approx(1.0)


def test_stage1_guardrail_progress_stops_on_unrecoverable_candidate_ratio() -> None:
    progress = _stage1_guardrail_progress(
        [
            {"n_candidates": 1000, "n_labels": 10, "recall": 1.0},
            {"n_candidates": 1000, "n_labels": 10, "recall": 1.0},
            {"n_candidates": 1000, "n_labels": 10, "recall": 1.0},
        ],
        total_preflight_images=4,
        total_labeled_preflight_images=4,
        preflight_cfg={
            "min_stage1_recall_mean": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 5000,
            "max_candidate_ratio_cap_mean": 50.0,
        },
    )

    assert not progress["can_still_pass"]
    assert progress["definitive_failure_reasons"] == [
        "minimum possible candidate/ground-truth ratio 75.00 > maximum 50.00"
    ]


def test_2d_profile_uses_dimensionality_specific_density_defaults(monkeypatch, tmp_path) -> None:
    cfg = deep_merge(
        DEFAULT_NATIVE_CONFIG,
        {
            "dataset_root": str(tmp_path),
            "pipeline_defaults": {
                "image_dimensionality": 2,
                "xy_spacing_nm": 100.0,
                "z_spacing_nm": None,
            },
            "profiling": {"enabled": True},
        },
    )

    monkeypatch.setattr(
        optimizer,
        "split_pairs",
        lambda root, split: [(tmp_path / f"{split}_image.tif", tmp_path / f"{split}_labels.csv")],
    )
    monkeypatch.setattr(optimizer, "read_points_csv", lambda path: np.zeros((32, 2), dtype=np.float32))
    monkeypatch.setattr(optimizer, "read_volume", lambda path: np.ones((8, 8), dtype=np.float32))

    profile = _profile_dataset(cfg)

    assert profile["image_dimensionality"] == 2
    assert profile["density_sparse_label_mean_max"] == 16.0
    assert profile["density_dense_label_mean_min"] == 64.0
    assert profile["density_regime"] == "moderate"


def test_2d_profile_respects_explicit_density_threshold_overrides(monkeypatch, tmp_path) -> None:
    cfg = deep_merge(
        DEFAULT_NATIVE_CONFIG,
        {
            "dataset_root": str(tmp_path),
            "pipeline_defaults": {
                "image_dimensionality": 2,
                "xy_spacing_nm": 100.0,
                "z_spacing_nm": None,
            },
            "profiling": {
                "enabled": True,
                "sparse_label_mean_max": 64.0,
                "dense_label_mean_min": 256.0,
            },
        },
    )

    monkeypatch.setattr(
        optimizer,
        "split_pairs",
        lambda root, split: [(tmp_path / f"{split}_image.tif", tmp_path / f"{split}_labels.csv")],
    )
    monkeypatch.setattr(optimizer, "read_points_csv", lambda path: np.zeros((32, 2), dtype=np.float32))
    monkeypatch.setattr(optimizer, "read_volume", lambda path: np.ones((8, 8), dtype=np.float32))

    profile = _profile_dataset(cfg)

    assert profile["density_sparse_label_mean_max"] == 64.0
    assert profile["density_dense_label_mean_min"] == 256.0
    assert profile["density_regime"] == "sparse"


def test_profile_guidance_uses_2d_rolling_box_for_2d_images() -> None:
    cfg = deep_merge(
        DEFAULT_NATIVE_CONFIG,
        {
            "stage1_recipes": [],
            "pipeline_defaults": {
                "image_dimensionality": 2,
                "xy_spacing_nm": 100.0,
                "z_spacing_nm": None,
            },
        },
    )
    profile = {"density_regime": "sparse", "contrast_regime": "moderate", "background_regime": "stable"}

    recipes = _augment_stage1_recipes(cfg, profile)
    background_methods = {recipe.get("background_method") for recipe in recipes}

    assert "rolling_box_2d" in background_methods
    assert "rolling_box_3d" not in background_methods


def test_preflight_decision_columns_rank_shortlist_and_stage2() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "low_recall",
                "passed": False,
                "mean_stage1_recall": 0.1,
                "mean_stage1_precision": 0.9,
                "mean_stage1_candidate_ratio": 1.0,
            },
            {
                "recipe_id": "winner",
                "passed": True,
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "mean_stage1_candidate_ratio": 1.0,
            },
            {
                "recipe_id": "runner_up",
                "passed": True,
                "mean_stage1_recall": 0.89,
                "mean_stage1_precision": 0.2,
                "mean_stage1_candidate_ratio": 1.0,
            },
        ]
    )

    out = _apply_preflight_decision_columns(df, shortlist_ids={"winner", "runner_up"}, stage2_ids={"winner", "runner_up"})

    assert out.loc[out["recipe_id"] == "winner", "stage1_rank_passed"].item() == 1
    assert out.loc[out["recipe_id"] == "runner_up", "stage1_rank_passed"].item() == 2
    assert out.loc[out["recipe_id"] == "low_recall", "stage1_rank_passed"].isna().item()
    assert out.loc[out["recipe_id"] == "winner", "shortlisted_for_stage2"].item()
    assert out.loc[out["recipe_id"] == "winner", "selected_for_stage2"].item()
    assert out.loc[out["recipe_id"] == "runner_up", "selected_for_stage2"].item()


def test_stage1_shortlist_selects_top_unique_stage1_configs() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "stage1_a",
                "stage1_key": "stage1_a",
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "mean_stage1_candidate_ratio": 1.0,
            },
            {
                "recipe_id": "stage1_b",
                "stage1_key": "stage1_b",
                "mean_stage1_recall": 0.8,
                "mean_stage1_precision": 0.5,
                "mean_stage1_candidate_ratio": 1.0,
            },
        ]
    )

    shortlist, selected_keys = _stage2_shortlist_from_passed(df, top_k=1)

    assert selected_keys == ["stage1_a"]
    assert set(shortlist["recipe_id"]) == {"stage1_a"}


def test_stage1_shortlist_uses_recall_band_then_highest_f1() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "highest_recall_low_f1",
                "stage1_key": "highest_recall_low_f1",
                "mean_stage1_recall": 0.95,
                "mean_stage1_precision": 0.10,
                "mean_stage1_candidate_ratio": 100.0,
            },
            {
                "recipe_id": "near_recall_best_f1",
                "stage1_key": "near_recall_best_f1",
                "mean_stage1_recall": 0.94,
                "mean_stage1_precision": 0.80,
                "mean_stage1_candidate_ratio": 1.0,
            },
            {
                "recipe_id": "near_recall_lower_f1",
                "stage1_key": "near_recall_lower_f1",
                "mean_stage1_recall": 0.93,
                "mean_stage1_precision": 0.60,
                "mean_stage1_candidate_ratio": 2.0,
            },
            {
                "recipe_id": "low_recall_clean_fast",
                "stage1_key": "low_recall_clean_fast",
                "mean_stage1_recall": 0.80,
                "mean_stage1_precision": 0.99,
                "mean_stage1_candidate_ratio": 1.0,
            },
        ]
    )

    shortlist, selected_keys = _stage2_shortlist_from_passed(
        df,
        top_k=1,
        ranking_cfg={"recall_tolerance": 0.02},
    )

    assert selected_keys == ["near_recall_best_f1"]
    assert set(shortlist["recipe_id"]) == {"near_recall_best_f1"}


def test_stage1_shortlist_prefers_recipe_id_on_f1_ties() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "a_expensive_rb10",
                "stage1_key": "a_expensive_rb10",
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "mean_stage1_candidate_ratio": 1.0,
            },
            {
                "recipe_id": "b_cheap_no_background",
                "stage1_key": "b_cheap_no_background",
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "mean_stage1_candidate_ratio": 1.0,
            },
        ]
    )

    shortlist, selected_keys = _stage2_shortlist_from_passed(df, top_k=1)

    assert selected_keys == ["a_expensive_rb10"]
    assert set(shortlist["recipe_id"]) == {"a_expensive_rb10"}


def test_stage2_expands_feature_packs_only_after_stage1_shortlist() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    stage1_recipe = recipe_bank(cfg)[0]
    passed = pd.DataFrame(
        [
            {
                "recipe_id": stage1_recipe["recipe_id"],
                "stage1_key": "stage1_a",
                "mean_stage1_recall": 1.0,
                "mean_stage1_precision": 1.0,
                "mean_stage1_candidate_ratio": 1.0,
                "stage1_rank_passed": 1,
                "passed": True,
                "recipe": stage1_recipe,
            }
        ]
    )
    stage1_shortlist, _ = _stage2_shortlist_from_passed(passed, top_k=1)
    stage2_shortlist = _expand_stage2_shortlist(stage1_shortlist, cfg)

    assert len(stage2_shortlist) == 4
    assert set(stage2_shortlist["feature_pack_name"]) == set(cfg["stage2_feature_packs"])
    assert set(stage2_shortlist["stage1_recipe_id"]) == {stage1_recipe["recipe_id"]}
    cache_feature_sets = {tuple(recipe["feature_cache_features"]) for recipe in stage2_shortlist["recipe"]}
    assert len(cache_feature_sets) == 1


def test_all_positive_stage1_shortlist_expands_to_single_pass_through_recipe() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    stage1_recipe = recipe_bank(cfg)[0]
    stage1_shortlist = pd.DataFrame(
        [
            {
                "recipe_id": stage1_recipe["recipe_id"],
                "stage1_key": "stage1_a",
                "stage1_rank_passed": 1,
                "passed": True,
                "stage1_train_label_status": "all_positive_stage1",
                "recipe": stage1_recipe,
            }
        ]
    )

    stage2_shortlist = _expand_stage2_shortlist(stage1_shortlist, cfg)

    assert len(stage2_shortlist) == 1
    recipe = stage2_shortlist.iloc[0]["recipe"]
    assert recipe["model_type"] == "stage1_pass_through"
    assert recipe["feature_pack_name"] == "not_applicable"
    assert recipe["selected_features"] == []
    assert recipe["feature_cache_features"] == []
    assert recipe["fit_method"] == "2D (XY) + 1D (Z) Gaussian"


def test_untrainable_stage1_shortlist_expands_to_single_skipped_marker() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {"stage2_feature_packs": ["core_fit", "core_contrast"]})
    stage1_recipe = recipe_bank(cfg)[0]
    stage1_shortlist = pd.DataFrame(
        [
            {
                "recipe_id": stage1_recipe["recipe_id"],
                "stage1_key": "stage1_a",
                "stage1_rank_passed": 1,
                "passed": True,
                "stage1_train_label_status": "no_true_positive_training_candidates",
                "recipe": stage1_recipe,
            }
        ]
    )

    stage2_shortlist = _expand_stage2_shortlist(stage1_shortlist, cfg)

    assert len(stage2_shortlist) == 1
    recipe = stage2_shortlist.iloc[0]["recipe"]
    assert stage2_shortlist.iloc[0]["model_type"] == "skipped"
    assert recipe["model_type"] == "skipped"
    assert recipe["feature_pack_name"] == "not_applicable"
    assert recipe["selected_features"] == []
    assert recipe["stage2_skip_reason"] == "no_true_positive_training_candidates"
    assert recipe["recipe_id"] == f"{stage1_recipe['recipe_id']}_no_true_positive_training_candidates"


def test_stage1_pass_through_export_payload_has_no_svm_or_features() -> None:
    recipe = {
        "recipe_id": "stage1_a_stage1_pass_through",
        "fit_method": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "feature_pack_name": "not_applicable",
        "selected_features": [],
        "model_type": "stage1_pass_through",
    }
    winner = pd.Series({"recipe": recipe, "model_type": "stage1_pass_through", "decision_threshold": 0.0})

    assert _svm_parameter_payload(winner) is None
    assert _stage2_parameter_payload(recipe, winner) == {
        "model_type": "stage1_pass_through",
        "fitting_mode": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "fit_background_width": None,
        "fit_max_iterations": None,
        "fit_tolerance": None,
        "feature_pack_name": "not_applicable",
        "selected_features": [],
        "svm": None,
        "decision_rule": "accept_all_stage1_candidates",
        "decision_threshold": None,
    }


def test_stage2_finalist_ranking_prefers_simpler_near_ties() -> None:
    assert _FINALIST_SORT_COLUMNS == [
        "feature_pack_simplicity_rank",
        "model_type_simplicity_rank",
        "svm_kernel_simplicity_rank",
        "svm_c_simplicity_rank",
        "svm_degree_simplicity_rank",
        "svm_gamma_simplicity_rank",
        "stage1_rank_passed",
        "recipe_id",
    ]
    assert _FINALIST_SORT_ASCENDING == [True] * len(_FINALIST_SORT_COLUMNS)

    df = _add_stage2_selection_columns(
        pd.DataFrame(
            [
                {
                    "recipe_id": "complex_best_f1",
                    "feature_pack_name": "full_interpretable",
                    "model_type": "svm",
                    "kernel": "rbf",
                    "C": 10.0,
                    "gamma": 10.0,
                    "degree": 2,
                    "val_f1": 1.0,
                    "stage1_rank_passed": 1,
                },
                {
                    "recipe_id": "simple_near_tie",
                    "feature_pack_name": "core_fit",
                    "model_type": "svm",
                    "kernel": "linear",
                    "C": 1.0,
                    "gamma": "auto",
                    "degree": 2,
                    "val_f1": 0.996,
                    "stage1_rank_passed": 2,
                },
            ]
        )
    )
    finalists = df[df["stage2_f1_loss"] <= 0.005].sort_values(
        _FINALIST_SORT_COLUMNS,
        ascending=_FINALIST_SORT_ASCENDING,
    )

    assert finalists.iloc[0]["recipe_id"] == "simple_near_tie"


def test_stage2_finalist_ranking_uses_stage1_rank_not_f1_loss_after_simplicity_ties() -> None:
    df = _add_stage2_selection_columns(
        pd.DataFrame(
            [
                {
                    "recipe_id": "higher_f1_worse_stage1_rank",
                    "feature_pack_name": "core_fit",
                    "model_type": "svm",
                    "kernel": "linear",
                    "C": 1.0,
                    "gamma": "auto",
                    "degree": 2,
                    "val_f1": 1.0,
                    "stage1_rank_passed": 2,
                },
                {
                    "recipe_id": "lower_f1_better_stage1_rank",
                    "feature_pack_name": "core_fit",
                    "model_type": "svm",
                    "kernel": "linear",
                    "C": 1.0,
                    "gamma": "auto",
                    "degree": 2,
                    "val_f1": 0.996,
                    "stage1_rank_passed": 1,
                },
            ]
        )
    )
    finalists = df[df["stage2_f1_loss"] <= 0.005].sort_values(
        _FINALIST_SORT_COLUMNS,
        ascending=_FINALIST_SORT_ASCENDING,
    )

    assert finalists.iloc[0]["recipe_id"] == "lower_f1_better_stage1_rank"


def test_optimizer_plan_counts_stage1_before_stage2_expansion(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"dataset_root": null}')

    plan = optimizer_plan(config_path)

    assert plan["stage1_recipe_bank_entries"] == 120
    assert plan["unique_stage1_preflight_configs"] == 120
    assert plan["shortlist_top_k"] == 5
    assert plan["max_stage2_recipe_entries_after_shortlist"] == 20
    assert plan["svm_param_grid_entries_per_stage2_recipe"] == 48


def test_optimizer_plan_safety_stops_accidental_large_runs() -> None:
    cfg = {"optimizer": {"max_stage1_preflight_configs": 10, "max_stage2_recipes_after_shortlist": 20}}
    plan = {
        "unique_stage1_preflight_configs": 120,
        "max_stage2_recipe_entries_after_shortlist": 21,
    }

    with pytest.raises(RuntimeError, match="unique Stage 1 preflight configs exceeds"):
        _enforce_optimizer_plan_safety(plan, cfg)
