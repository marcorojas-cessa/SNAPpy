import pandas as pd
import pytest

from mrsnappy.config import DEFAULT_NATIVE_CONFIG, deep_merge, recipe_bank
from mrsnappy.optimizer import (
    _apply_preflight_decision_columns,
    _auto_candidate_ratio_cap,
    _expand_stage2_shortlist,
    _enforce_optimizer_plan_safety,
    _stage1_failure_reasons,
    _stage2_shortlist_from_passed,
    optimizer_plan,
)


def test_stage1_failure_reasons_are_explicit_and_minimal() -> None:
    reasons = _stage1_failure_reasons(
        mean_recall=0.1,
        mean_candidates=3000.0,
        max_candidates=4500,
        mean_candidate_ratio=101.0,
        preflight_cfg={
            "min_stage1_recall": 0.25,
            "max_stage1_candidates_mean": 2500,
            "max_stage1_candidates_single": 4000,
            "max_stage1_candidates_per_label_mean": 100.0,
        },
    )

    assert reasons == [
        "recall 0.1000 < minimum 0.2500",
        "mean candidates 3000.0 > maximum 2500.0",
        "single-image candidates 4500 > maximum 4000",
        "candidate/label ratio 101.00 > maximum 100.00",
    ]


def test_preflight_decision_columns_rank_shortlist_and_stage2() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "low_recall",
                "passed": False,
                "preflight_utility": 0.8,
                "mean_stage1_recall": 0.1,
                "mean_stage1_precision": 0.9,
                "fit_cost_rank": 0,
                "feature_count": 16,
            },
            {
                "recipe_id": "winner",
                "passed": True,
                "preflight_utility": 0.7,
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "fit_cost_rank": 0,
                "feature_count": 16,
            },
            {
                "recipe_id": "runner_up",
                "passed": True,
                "preflight_utility": 0.6,
                "mean_stage1_recall": 0.8,
                "mean_stage1_precision": 0.5,
                "fit_cost_rank": 0,
                "feature_count": 16,
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


def test_default_auto_candidate_ratio_caps_match_current_policy() -> None:
    preflight_cfg = DEFAULT_NATIVE_CONFIG["preflight"]

    assert _auto_candidate_ratio_cap({"density_regime": "sparse", "contrast_regime": "moderate"}, preflight_cfg) == 130.0
    assert _auto_candidate_ratio_cap({"density_regime": "moderate", "contrast_regime": "moderate"}, preflight_cfg) == 100.0
    assert _auto_candidate_ratio_cap({"density_regime": "dense", "contrast_regime": "moderate"}, preflight_cfg) == 70.0
    assert _auto_candidate_ratio_cap({"density_regime": "moderate", "contrast_regime": "low"}, preflight_cfg) == 125.0
    assert _auto_candidate_ratio_cap({"density_regime": "moderate", "contrast_regime": "high"}, preflight_cfg) == 85.0


def test_stage1_shortlist_selects_top_unique_stage1_configs() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "stage1_a",
                "stage1_key": "stage1_a",
                "preflight_utility": 0.9,
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
            },
            {
                "recipe_id": "stage1_b",
                "stage1_key": "stage1_b",
                "preflight_utility": 0.8,
                "mean_stage1_recall": 0.8,
                "mean_stage1_precision": 0.5,
            },
        ]
    )

    shortlist, selected_keys = _stage2_shortlist_from_passed(df, top_k=1)

    assert selected_keys == ["stage1_a"]
    assert set(shortlist["recipe_id"]) == {"stage1_a"}


def test_stage1_shortlist_prefers_lower_processing_cost_on_metric_ties() -> None:
    df = pd.DataFrame(
        [
            {
                "recipe_id": "expensive_rb10",
                "stage1_key": "expensive_rb10",
                "preflight_utility": 0.9,
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "processing_cost_rank": 1010.0,
            },
            {
                "recipe_id": "cheap_no_background",
                "stage1_key": "cheap_no_background",
                "preflight_utility": 0.9,
                "mean_stage1_recall": 0.9,
                "mean_stage1_precision": 0.4,
                "processing_cost_rank": 2.0,
            },
        ]
    )

    shortlist, selected_keys = _stage2_shortlist_from_passed(df, top_k=1)

    assert selected_keys == ["cheap_no_background"]
    assert set(shortlist["recipe_id"]) == {"cheap_no_background"}


def test_stage2_expands_feature_packs_only_after_stage1_shortlist() -> None:
    cfg = deep_merge(DEFAULT_NATIVE_CONFIG, {})
    stage1_recipe = recipe_bank(cfg)[0]
    passed = pd.DataFrame(
        [
            {
                "recipe_id": stage1_recipe["recipe_id"],
                "stage1_key": "stage1_a",
                "preflight_utility": 1.0,
                "mean_stage1_recall": 1.0,
                "mean_stage1_precision": 1.0,
                "stage1_rank_passed": 1,
                "passed": True,
                "recipe": stage1_recipe,
            }
        ]
    )
    stage1_shortlist, _ = _stage2_shortlist_from_passed(passed, top_k=1)
    stage2_shortlist = _expand_stage2_shortlist(stage1_shortlist, cfg)

    assert len(stage2_shortlist) == 7
    assert set(stage2_shortlist["feature_pack_name"]) == set(cfg["stage2_feature_packs"])
    assert set(stage2_shortlist["stage1_recipe_id"]) == {stage1_recipe["recipe_id"]}


def test_auto_candidate_ratio_caps_are_user_configurable() -> None:
    preflight_cfg = {
        **DEFAULT_NATIVE_CONFIG["preflight"],
        "auto_candidate_ratio_caps": {"sparse": 10.0, "moderate": 20.0, "dense": 30.0},
        "auto_candidate_ratio_low_contrast_multiplier": 2.0,
        "auto_candidate_ratio_high_contrast_multiplier": 0.5,
    }

    assert _auto_candidate_ratio_cap({"density_regime": "sparse", "contrast_regime": "moderate"}, preflight_cfg) == 10.0
    assert _auto_candidate_ratio_cap({"density_regime": "moderate", "contrast_regime": "low"}, preflight_cfg) == 40.0
    assert _auto_candidate_ratio_cap({"density_regime": "dense", "contrast_regime": "high"}, preflight_cfg) == 15.0


def test_optimizer_plan_counts_stage1_before_stage2_expansion(tmp_path) -> None:
    config_path = tmp_path / "optimize.json"
    config_path.write_text('{"dataset_root": null}')

    plan = optimizer_plan(config_path)

    assert plan["stage1_recipe_bank_entries"] == 432
    assert plan["unique_stage1_preflight_configs"] == 432
    assert plan["shortlist_top_k"] == 3
    assert plan["max_stage2_recipe_entries_after_shortlist"] == 21
    assert plan["svm_param_grid_entries_per_stage2_recipe"] == 35


def test_optimizer_plan_safety_stops_accidental_large_runs() -> None:
    cfg = {"optimizer": {"max_stage1_preflight_configs": 10, "max_stage2_recipes_after_shortlist": 20}}
    plan = {
        "unique_stage1_preflight_configs": 432,
        "max_stage2_recipe_entries_after_shortlist": 21,
    }

    with pytest.raises(RuntimeError, match="unique Stage 1 preflight configs exceeds"):
        _enforce_optimizer_plan_safety(plan, cfg)
