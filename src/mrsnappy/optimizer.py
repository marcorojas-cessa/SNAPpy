from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dir, load_config, stage1_recipe_bank, stage2_recipe_bank
from .pipeline import (
    clear_pipeline_caches,
    evaluate_predictions_for_selection,
    preflight_image,
    promote_stage1_image_volume_cache,
    promote_stage1_candidate_cache,
    prune_stage1_candidate_cache,
    predict_split,
    precision_recall_f1,
    stage1_cache_signature,
    summarize_stage1_candidate_labels,
    train_native_model,
    write_json,
)
from .io import read_points_csv, read_volume, split_pairs
from .model import iter_svm_param_grid
from .spatial import spacing_zyx_nm


_FINALIST_SORT_COLUMNS = [
    "feature_pack_simplicity_rank",
    "model_type_simplicity_rank",
    "svm_kernel_simplicity_rank",
    "svm_c_simplicity_rank",
    "svm_degree_simplicity_rank",
    "svm_gamma_simplicity_rank",
    "stage1_rank_passed",
    "recipe_id",
]
_FINALIST_SORT_ASCENDING = [True] * len(_FINALIST_SORT_COLUMNS)
_FEATURE_PACK_SIMPLICITY_RANK = {
    "core_fit": 0.0,
    "core_contrast": 1.0,
    "core_morphology": 2.0,
    "full_interpretable": 3.0,
}
_MODEL_TYPE_SIMPLICITY_RANK = {
    "stage1_pass_through": 0.0,
    "svm": 1.0,
}
_KERNEL_SIMPLICITY_RANK = {
    "linear": 0.0,
    "rbf": 1.0,
    "polynomial": 2.0,
    "poly": 2.0,
}
_FITTING_MODE_SIMPLICITY_RANK = {
    "2d (xy) + 1d (z) gaussian": 0.0,
    "3d gaussian": 1.0,
    "distorted 3d gaussian": 2.0,
}
_DEFAULT_FITTING_MODE = "2D (XY) + 1D (Z) Gaussian"
_STAGE1_SCREEN_FIELDS = (
    "xy_spacing_nm",
    "z_spacing_nm",
    "preproc_enabled",
    "preproc_method",
    "preproc_sigma",
    "preproc_sigma_nm",
    "norm_enabled",
    "norm_method",
    "background_enabled",
    "background_method",
    "background_param",
    "background_param_nm",
    "background_clip",
    "maxima_method",
    "maxima_neighborhood",
    "maxima_min_distance_nm",
    "sigma_value",
    "sigma_nm",
    "threshold_value",
    "h_max_sigma_multiplier",
    "h_max_sigma_mode",
)


def _match_distance_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return matching arguments for voxel or physical-distance matching."""
    if cfg.get("match_distance_nm") is None:
        return {"match_distance": float(cfg["match_distance"])}
    return {
        "match_distance": float(cfg["match_distance_nm"]),
        "match_spacing_nm": spacing_zyx_nm(cfg.get("pipeline_defaults", {}), 3),
    }


def _match_distance_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("match_distance_nm") is None:
        return {
            "match_distance": float(cfg["match_distance"]),
            "match_distance_units": "voxels",
            "match_distance_nm": None,
        }
    spacing = spacing_zyx_nm(cfg.get("pipeline_defaults", {}), 3)
    return {
        "match_distance": float(cfg["match_distance_nm"]),
        "match_distance_units": "nm",
        "match_distance_nm": float(cfg["match_distance_nm"]),
        "match_spacing_zyx_nm": list(spacing),
    }


def _stage2_f1_tolerance(cfg: dict[str, Any]) -> float:
    optimizer_cfg = cfg.get("optimizer", {})
    value = optimizer_cfg.get("stage2_f1_tolerance", 0.005)
    if value is None:
        return 0.005
    return float(value)


def _missing_scalar(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def _stage2_row_recipe(row: pd.Series) -> dict[str, Any]:
    recipe = row.get("recipe")
    return recipe if isinstance(recipe, dict) else {}


def _stage2_row_model_type(row: pd.Series) -> str:
    value = row.get("model_type")
    if _missing_scalar(value):
        value = _stage2_row_recipe(row).get("model_type")
    return str(value if not _missing_scalar(value) else "svm").strip().lower()


def _is_stage1_pass_through_row(row: pd.Series) -> bool:
    return _stage2_row_model_type(row) == "stage1_pass_through"


def _stage2_row_feature_pack_name(row: pd.Series) -> str:
    value = row.get("feature_pack_name")
    if _missing_scalar(value):
        value = _stage2_row_recipe(row).get("feature_pack_name")
    return str(value if not _missing_scalar(value) else "").strip()


def _feature_pack_simplicity_rank(row: pd.Series) -> float:
    if _stage2_row_model_type(row) == "stage1_pass_through":
        return -1.0
    return float(_FEATURE_PACK_SIMPLICITY_RANK.get(_stage2_row_feature_pack_name(row), 999.0))


def _kernel_simplicity_rank(row: pd.Series) -> float:
    if _stage2_row_model_type(row) == "stage1_pass_through":
        return -1.0
    kernel = str(row.get("kernel") if not _missing_scalar(row.get("kernel")) else "linear").strip().lower()
    return float(_KERNEL_SIMPLICITY_RANK.get(kernel, 999.0))


def _c_simplicity_rank(row: pd.Series) -> float:
    if _stage2_row_model_type(row) == "stage1_pass_through":
        return -1.0
    value = row.get("C")
    if _missing_scalar(value):
        return 999.0
    c_value = float(value)
    if c_value <= 0:
        return 999.0
    return float(abs(math.log10(c_value)))


def _degree_simplicity_rank(row: pd.Series) -> float:
    if _stage2_row_model_type(row) == "stage1_pass_through":
        return -1.0
    kernel = str(row.get("kernel") if not _missing_scalar(row.get("kernel")) else "linear").strip().lower()
    if kernel not in {"polynomial", "poly"}:
        return 0.0
    value = row.get("degree")
    if _missing_scalar(value):
        return 999.0
    return float(value)


def _gamma_simplicity_rank(row: pd.Series) -> float:
    if _stage2_row_model_type(row) == "stage1_pass_through":
        return -1.0
    kernel = str(row.get("kernel") if not _missing_scalar(row.get("kernel")) else "linear").strip().lower()
    if kernel == "linear":
        return 0.0
    value = row.get("gamma")
    if _missing_scalar(value):
        return 0.0
    text = str(value).strip().lower()
    if text == "auto":
        return 0.0
    if text == "scale":
        return 1.0
    gamma = float(value)
    if gamma <= 0:
        return 999.0
    return float(2.0 + abs(math.log10(gamma)))


def _add_stage2_selection_columns(stage2_df: pd.DataFrame) -> pd.DataFrame:
    out = stage2_df.copy()
    best_f1 = float(out["val_f1"].max()) if not out.empty else 0.0
    out["stage2_f1_loss"] = best_f1 - out["val_f1"].astype(float)
    out["feature_pack_simplicity_rank"] = out.apply(_feature_pack_simplicity_rank, axis=1)
    out["model_type_simplicity_rank"] = out.apply(lambda row: float(_MODEL_TYPE_SIMPLICITY_RANK.get(_stage2_row_model_type(row), 999.0)), axis=1)
    out["svm_kernel_simplicity_rank"] = out.apply(_kernel_simplicity_rank, axis=1)
    out["svm_c_simplicity_rank"] = out.apply(_c_simplicity_rank, axis=1)
    out["svm_degree_simplicity_rank"] = out.apply(_degree_simplicity_rank, axis=1)
    out["svm_gamma_simplicity_rank"] = out.apply(_gamma_simplicity_rank, axis=1)
    return out


def _optimization_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("optimization_mode", "fixed_split")).strip().lower()


def _require_fixed_split_mode(cfg: dict[str, Any]) -> str:
    mode = _optimization_mode(cfg)
    if mode != "fixed_split":
        raise NotImplementedError(
            "SNAPpy optimization_mode='cross_validation' is planned but not implemented yet. "
            "Use optimization_mode: fixed_split with dataset_root containing train/ and val/ folders."
        )
    return mode


def _write_optimization_split_records(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    mode = _require_fixed_split_mode(cfg)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    image_ids: dict[str, list[str]] = {}
    for split in ("train", "val"):
        pairs = split_pairs(cfg["dataset_root"], split)
        counts[split] = int(len(pairs))
        image_ids[split] = [Path(image_path).stem for image_path, _ in pairs]
        for image_path, label_path in pairs:
            rows.append(
                {
                    "optimization_mode": mode,
                    "fold_id": "fixed_split",
                    "split": split,
                    "image_id": Path(image_path).stem,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "optimization_splits.csv", index=False)
    summary = {
        "optimization_mode": mode,
        "validation_strategy": "fixed user-supplied train/ and val/ folders",
        "fold_count": 1,
        "folds": [
            {
                "fold_id": "fixed_split",
                "train_image_count": counts.get("train", 0),
                "val_image_count": counts.get("val", 0),
                "train_image_ids": image_ids.get("train", []),
                "val_image_ids": image_ids.get("val", []),
            }
        ],
    }
    return summary


def _stage1_screen_key(recipe: dict[str, Any]) -> str:
    payload = {key: recipe.get(key) for key in _STAGE1_SCREEN_FIELDS}
    return json.dumps(payload, sort_keys=True, default=str)


def _stage1_ranking_config(ranking_cfg: dict[str, Any] | None) -> dict[str, Any]:
    ranking_cfg = ranking_cfg or {}
    return {
        "recall_tolerance": float(ranking_cfg.get("recall_tolerance", 0.02)),
    }


def _mean_per_image_candidate_ratio(image_rows: list[dict[str, Any]]) -> float:
    """Average candidate burden on images where candidate/GT ratio is defined."""
    labeled_rows = [row for row in image_rows if int(row.get("n_labels", 0)) > 0]
    if not labeled_rows:
        return 0.0
    return float(
        sum(float(row["n_candidates"]) / float(row["n_labels"]) for row in labeled_rows)
        / len(labeled_rows)
    )


def _mean_labeled_stage1_metric(image_rows: list[dict[str, Any]], metric_name: str) -> float:
    labeled_rows = [row for row in image_rows if int(row.get("n_labels", 0)) > 0]
    if not labeled_rows:
        return 0.0
    return float(sum(float(row[metric_name]) for row in labeled_rows) / len(labeled_rows))


def _stage1_guardrail_progress(
    image_rows: list[dict[str, Any]],
    *,
    total_preflight_images: int,
    total_labeled_preflight_images: int,
    preflight_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Determine whether a Stage 1 recipe can still pass the configured guardrails.

    This is intentionally limited to Stage 1 guardrails. It does not use F1,
    precision, runtime preference, or any winner-selection criterion.
    """
    processed_image_count = int(len(image_rows))
    total_image_count = max(int(total_preflight_images), 1)
    remaining_image_count = max(total_image_count - processed_image_count, 0)
    total_labeled_image_count = max(int(total_labeled_preflight_images), 0)
    processed_labeled_image_count = int(sum(1 for row in image_rows if int(row.get("n_labels", 0)) > 0))
    remaining_labeled_image_count = max(total_labeled_image_count - processed_labeled_image_count, 0)
    candidate_count_so_far = float(sum(float(row["n_candidates"]) for row in image_rows))
    recall_sum_so_far = float(
        sum(float(row["recall"]) for row in image_rows if int(row.get("n_labels", 0)) > 0)
    )
    candidate_ratio_sum_so_far = float(
        sum(
            float(row["n_candidates"]) / float(row["n_labels"])
            for row in image_rows
            if int(row.get("n_labels", 0)) > 0
        )
    )
    max_candidates_so_far = int(max((int(row["n_candidates"]) for row in image_rows), default=0))

    min_recall = preflight_cfg.get("min_stage1_recall_mean")
    max_mean_candidates = preflight_cfg.get("max_stage1_candidates_mean")
    max_single_candidates = preflight_cfg.get("max_stage1_candidates_single")
    ratio_cap = _active_candidate_ratio_cap(preflight_cfg)

    maximum_possible_mean_recall = (
        float((recall_sum_so_far + remaining_labeled_image_count) / total_labeled_image_count)
        if total_labeled_image_count
        else None
    )
    minimum_possible_mean_candidates = float(candidate_count_so_far / total_image_count)
    minimum_possible_candidate_ratio = (
        float(candidate_ratio_sum_so_far / total_labeled_image_count)
        if total_labeled_image_count
        else 0.0
    )

    definitive_failure_reasons: list[str] = []
    if max_single_candidates is not None and max_candidates_so_far > float(max_single_candidates):
        definitive_failure_reasons.append(
            f"single-image candidates {max_candidates_so_far} > maximum {_format_guardrail_number(float(max_single_candidates))}"
        )
    if max_mean_candidates is not None and minimum_possible_mean_candidates > float(max_mean_candidates):
        definitive_failure_reasons.append(
            "minimum possible mean candidates "
            f"{minimum_possible_mean_candidates:.1f} > maximum {float(max_mean_candidates):.1f}"
        )
    if min_recall is not None and maximum_possible_mean_recall is not None and maximum_possible_mean_recall < float(min_recall):
        definitive_failure_reasons.append(
            "maximum possible mean recall on labeled images "
            f"{maximum_possible_mean_recall:.4f} < minimum {float(min_recall):.4f}"
        )
    if ratio_cap is not None and minimum_possible_candidate_ratio > float(ratio_cap):
        definitive_failure_reasons.append(
            "minimum possible candidate/ground-truth ratio "
            f"{minimum_possible_candidate_ratio:.2f} > maximum {float(ratio_cap):.2f}"
        )

    return {
        "can_still_pass": not definitive_failure_reasons,
        "definitive_failure_reasons": definitive_failure_reasons,
        "processed_image_count": processed_image_count,
        "total_image_count": total_image_count,
        "remaining_image_count": remaining_image_count,
        "processed_labeled_image_count": processed_labeled_image_count,
        "total_labeled_image_count": total_labeled_image_count,
        "remaining_labeled_image_count": remaining_labeled_image_count,
        "maximum_possible_mean_recall": maximum_possible_mean_recall,
        "minimum_possible_mean_candidates": minimum_possible_mean_candidates,
        "minimum_possible_candidate_ratio": minimum_possible_candidate_ratio,
    }


def _stage1_ranked_passed_df(passed_df: pd.DataFrame, ranking_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Rank passing Stage 1 recipes by the finalized recall-band/F1 policy.

    Guardrails have already been applied before this function is called.
    Only recipes within recall_tolerance of the best passing labeled-image mean recall are
    ranked for Stage 2 consideration. Recipes outside that recall band are
    retained for audit output but are not eligible for Stage 2 shortlisting.
    """
    passed_df = _ensure_preflight_sort_columns(passed_df)
    if passed_df.empty:
        return passed_df.copy()

    cfg = _stage1_ranking_config(ranking_cfg)
    recall_tolerance = float(cfg["recall_tolerance"])
    best_recall = float(passed_df["mean_stage1_recall"].max())
    recall_cutoff = best_recall - recall_tolerance

    recall_eligible = passed_df[passed_df["mean_stage1_recall"].astype(float) >= recall_cutoff].copy()
    ranked_recall_eligible = recall_eligible.sort_values(
        ["mean_stage1_f1", "recipe_id"],
        ascending=[False, True],
    )
    recall_ranked_ids = set(recall_eligible["recipe_id"].astype(str))
    remaining = passed_df[~passed_df["recipe_id"].astype(str).isin(recall_ranked_ids)]
    remaining = remaining.sort_values("recipe_id")
    ordered = pd.concat([ranked_recall_eligible, remaining]).drop_duplicates(subset=["recipe_id"], keep="first").copy()
    ordered["stage1_recall_eligible"] = ordered["recipe_id"].astype(str).isin(recall_ranked_ids)
    ordered["stage1_recall_cutoff"] = recall_cutoff
    ordered["stage1_best_recall"] = best_recall
    ordered["stage1_ranking_recall_tolerance"] = recall_tolerance
    return ordered


def _stage2_shortlist_from_passed(
    passed_df: pd.DataFrame,
    top_k: int,
    ranking_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select up to top_k unique Stage 1 configs from recall-eligible recipes."""
    sorted_passed = _stage1_ranked_passed_df(passed_df, ranking_cfg)
    eligible_passed = sorted_passed[sorted_passed["stage1_recall_eligible"].astype(bool)]
    selected_keys: list[str] = []
    selected_key_set: set[str] = set()
    for _, row in eligible_passed.iterrows():
        key = str(row["stage1_key"])
        if key in selected_key_set:
            continue
        selected_keys.append(key)
        selected_key_set.add(key)
        if len(selected_keys) >= int(top_k):
            break
    shortlist = eligible_passed[eligible_passed["stage1_key"].astype(str).isin(selected_key_set)].copy()
    return shortlist, selected_keys


def _leaderboard_stage1_cache_signatures(
    preflight_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> set[str]:
    if not preflight_rows:
        return set()
    df = pd.DataFrame(preflight_rows)
    if "passed" not in df or "stage1_preflight_cache_signature" not in df:
        return set()
    passed_df = df[df["passed"]].copy()
    if passed_df.empty:
        return set()
    shortlist, _ = _stage2_shortlist_from_passed(
        passed_df,
        top_k=int(cfg["optimizer"]["shortlist_top_k"]),
        ranking_cfg=cfg["stage1_ranking"],
    )
    return set(shortlist["stage1_preflight_cache_signature"].dropna().astype(str))


def _prune_stage1_cache_to_leaderboard(
    preflight_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    val_pairs: list[tuple[Path, Path]],
) -> int:
    signatures = _leaderboard_stage1_cache_signatures(preflight_rows, cfg)
    image_paths = [image_path for image_path, _ in val_pairs]
    return prune_stage1_candidate_cache(allowed_signatures=signatures, image_paths=image_paths)


def _promote_shortlisted_stage1_caches(
    stage1_shortlist: pd.DataFrame,
    *,
    preflight_candidate_cap: int | None,
    val_pairs: list[tuple[Path, Path]],
) -> dict[str, int]:
    promoted_candidates = 0
    promoted_image_volumes = 0
    seen: set[str] = set()
    image_paths = [image_path for image_path, _ in val_pairs]
    for _, row in stage1_shortlist.iterrows():
        target_recipe = deepcopy(dict(row["recipe"]))
        target_recipe.pop("max_candidates", None)
        source_recipe = deepcopy(target_recipe)
        if preflight_candidate_cap is not None:
            source_recipe["max_candidates"] = int(preflight_candidate_cap)
        source_signature = stage1_cache_signature(source_recipe)
        if source_signature in seen:
            continue
        seen.add(source_signature)
        promoted_candidates += promote_stage1_candidate_cache(
            source_recipe=source_recipe,
            target_recipe=target_recipe,
            image_paths=image_paths,
        )
        promoted_image_volumes += promote_stage1_image_volume_cache(
            recipe=source_recipe,
            image_paths=image_paths,
        )
        promoted_image_volumes += promote_stage1_image_volume_cache(
            recipe=target_recipe,
            image_paths=image_paths,
        )
    return {
        "candidate_entries": int(promoted_candidates),
        "image_volume_entries": int(promoted_image_volumes),
    }


def _promote_shortlisted_stage1_image_volumes(
    stage1_shortlist: pd.DataFrame,
    *,
    val_pairs: list[tuple[Path, Path]],
) -> int:
    promoted = 0
    seen: set[str] = set()
    image_paths = [image_path for image_path, _ in val_pairs]
    for _, row in stage1_shortlist.iterrows():
        recipe = deepcopy(dict(row["recipe"]))
        recipe.pop("max_candidates", None)
        signature = stage1_cache_signature(recipe)
        if signature in seen:
            continue
        seen.add(signature)
        promoted += promote_stage1_image_volume_cache(recipe=recipe, image_paths=image_paths)
    return promoted


def _evaluate_full_val_stage1(passed_df: pd.DataFrame, cfg: dict[str, Any], val_pairs: list[tuple[Path, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    match_kwargs = _match_distance_kwargs(cfg)
    for _, row in passed_df.iterrows():
        image_rows: list[dict[str, Any]] = []
        recipe = dict(row["recipe"])
        for image_path, label_path in val_pairs:
            image_row = preflight_image(image_path, label_path, recipe, **match_kwargs)
            image_rows.append(image_row)
        n_images = len(image_rows)
        total_tp = int(sum(image_row["tp"] for image_row in image_rows))
        total_fp = int(sum(image_row["fp"] for image_row in image_rows))
        total_fn = int(sum(image_row["fn"] for image_row in image_rows))
        metrics = precision_recall_f1(total_tp, total_fp, total_fn)
        total_candidates = int(sum(image_row["n_candidates"] for image_row in image_rows))
        total_labels = int(sum(image_row["n_labels"] for image_row in image_rows))
        stage1_full_val_candidate_ratio = (
            float(total_candidates / total_labels)
            if total_labels > 0
            else None
        )
        rows.append(
            {
                "recipe_id": str(row["recipe_id"]),
                "stage1_full_val_image_count": int(n_images),
                "stage1_full_val_tp": total_tp,
                "stage1_full_val_fp": total_fp,
                "stage1_full_val_fn": total_fn,
                "stage1_full_val_precision": float(metrics["precision"]),
                "stage1_full_val_recall": float(metrics["recall"]),
                "stage1_full_val_f1": float(metrics["f1"]),
                "stage1_full_val_mean_tp": float(total_tp / max(n_images, 1)),
                "stage1_full_val_mean_candidates": float(total_candidates / max(n_images, 1)),
                "stage1_full_val_candidate_ratio": stage1_full_val_candidate_ratio,
                "mean_stage1_candidate_ratio": float(row.get("mean_stage1_candidate_ratio", float("inf"))),
                "recipe": recipe,
            }
        )
    return pd.DataFrame(rows)


def _ensure_preflight_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mean_stage1_candidate_ratio" not in out:
        if {"mean_stage1_candidates", "mean_stage1_label_count"}.issubset(out.columns):
            labels = out["mean_stage1_label_count"].astype(float)
            candidates = out["mean_stage1_candidates"].astype(float)
            out["mean_stage1_candidate_ratio"] = np.where(labels > 0.0, candidates / labels, np.inf)
        else:
            out["mean_stage1_candidate_ratio"] = float("inf")
    if "mean_stage1_f1" not in out:
        if {"mean_stage1_precision", "mean_stage1_recall"}.issubset(out.columns):
            precision = out["mean_stage1_precision"].astype(float)
            recall = out["mean_stage1_recall"].astype(float)
            out["mean_stage1_f1"] = (2.0 * precision * recall / (precision + recall).replace(0.0, np.nan)).fillna(0.0)
        else:
            out["mean_stage1_f1"] = 0.0
    return out


def _fitting_mode_rank(value: Any) -> float:
    return float(_FITTING_MODE_SIMPLICITY_RANK.get(str(value or "").strip().lower(), 999.0))


def _simplest_fitting_mode(stage1_recipe: dict[str, Any], cfg: dict[str, Any]) -> str:
    values: list[Any] = []
    recipe_value = stage1_recipe.get("fit_method")
    if isinstance(recipe_value, list):
        values.extend(recipe_value)
    elif recipe_value is not None:
        values.append(recipe_value)
    default_value = cfg.get("pipeline_defaults", {}).get("fit_method")
    if isinstance(default_value, list):
        values.extend(default_value)
    elif default_value is not None:
        values.append(default_value)
    if not values:
        values.append(_DEFAULT_FITTING_MODE)
    return str(sorted(values, key=_fitting_mode_rank)[0])


def _stage1_pass_through_recipe(stage1_recipe: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    recipe = deepcopy(stage1_recipe)
    # Stage 1 pass-through is a no-SVM winning-model option for the rare case
    # where a shortlisted Stage 1 recipe produces only positive training
    # candidates. The model remains preprocessing + local maxima detection +
    # Gaussian fitting only; feature extraction and Stage 2 SVM classification
    # are intentionally not used.
    recipe["stage1_recipe_id"] = stage1_recipe["recipe_id"]
    recipe["fit_method"] = _simplest_fitting_mode(stage1_recipe, cfg)
    recipe["feature_pack_name"] = "not_applicable"
    recipe["selected_features"] = []
    recipe["feature_cache_features"] = []
    recipe["model_type"] = "stage1_pass_through"
    recipe["stage1_train_label_status"] = "all_positive_stage1"
    recipe["recipe_id"] = f"{stage1_recipe['recipe_id']}_stage1_pass_through"
    return recipe


def _annotate_stage1_train_label_status(stage1_shortlist: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if stage1_shortlist.empty:
        return stage1_shortlist.copy()
    out = stage1_shortlist.copy()
    summary_rows: list[dict[str, Any]] = []
    match_kwargs = _match_distance_kwargs(cfg)
    for _, row in out.iterrows():
        summary = summarize_stage1_candidate_labels(
            cfg["dataset_root"],
            "train",
            dict(row["recipe"]),
            **match_kwargs,
        )
        if summary["all_positive_candidates"]:
            status = "all_positive_stage1"
        elif summary["no_training_candidates"]:
            status = "no_training_candidates"
        elif summary["no_true_positive_candidates"]:
            status = "no_true_positive_training_candidates"
        else:
            status = "two_class_training_candidates"
        summary_rows.append(
            {
                "recipe_id": row["recipe_id"],
                "stage1_train_label_status": status,
                "stage1_train_images": int(summary["n_images"]),
                "stage1_train_candidates": int(summary["n_candidates"]),
                "stage1_train_positive_candidates": int(summary["n_positive_candidates"]),
                "stage1_train_negative_candidates": int(summary["n_negative_candidates"]),
                "stage1_train_ground_truth": int(summary["n_ground_truth"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return out.merge(summary_df, on="recipe_id", how="left")


def _expand_stage2_shortlist(
    stage1_shortlist: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, stage1_row in stage1_shortlist.iterrows():
        if str(stage1_row.get("stage1_train_label_status", "")).strip().lower() == "all_positive_stage1":
            recipe = _stage1_pass_through_recipe(deepcopy(stage1_row["recipe"]), cfg)
            row = stage1_row.drop(labels=["recipe"]).to_dict()
            row["stage1_recipe_id"] = stage1_row["recipe_id"]
            row["recipe_id"] = recipe["recipe_id"]
            row["feature_pack_name"] = recipe.get("feature_pack_name")
            row["fitting_mode"] = recipe.get("fit_method")
            row["model_type"] = "stage1_pass_through"
            row["recipe"] = deepcopy(recipe)
            rows.append(row)
            continue
        stage2_recipes = stage2_recipe_bank(cfg, [deepcopy(stage1_row["recipe"])])
        for recipe in stage2_recipes:
            row = stage1_row.drop(labels=["recipe"]).to_dict()
            row["stage1_recipe_id"] = stage1_row["recipe_id"]
            row["recipe_id"] = recipe["recipe_id"]
            row["feature_pack_name"] = recipe.get("feature_pack_name")
            row["fitting_mode"] = recipe.get("fit_method")
            row["recipe"] = deepcopy(recipe)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["stage1_rank_passed", "recipe_id"],
        ascending=[True, True],
    )


def _optimizer_plan_from_materialized(
    cfg: dict[str, Any],
    profile: dict[str, Any],
    stage1_recipes: list[dict[str, Any]],
) -> dict[str, Any]:
    stage2_counts = [len(stage2_recipe_bank(cfg, [stage1_recipe])) for stage1_recipe in stage1_recipes]
    shortlist_top_k = int(cfg["optimizer"]["shortlist_top_k"])
    max_stage2_after_shortlist = int(sum(sorted(stage2_counts, reverse=True)[:shortlist_top_k]))
    svm_grid_entries = len(list(iter_svm_param_grid(cfg.get("svm_sweep", {}))))
    return {
        "dataset_name": cfg.get("dataset_name"),
        "dataset_root": cfg.get("dataset_root"),
        "optimization_mode": _optimization_mode(cfg),
        "validation_strategy": "fixed user-supplied train/ and val/ folders",
        "matching": _match_distance_summary(cfg),
        "dataset_profile": profile,
        "stage1_recipe_bank_entries": int(len(stage1_recipes)),
        "unique_stage1_preflight_configs": int(len(stage1_recipes)),
        "stage2_recipe_entries_per_stage1_min": int(min(stage2_counts, default=0)),
        "stage2_recipe_entries_per_stage1_max": int(max(stage2_counts, default=0)),
        "shortlist_top_k": shortlist_top_k,
        "max_stage2_recipe_entries_after_shortlist": max_stage2_after_shortlist,
        "svm_param_grid_entries_per_stage2_recipe": int(svm_grid_entries),
        "max_svm_fits_after_shortlist": int(max_stage2_after_shortlist * svm_grid_entries),
        "svm_selection": "fit each SVM configuration on train/ and select by mean per-image validation performance on all val/ images",
        "safety_caps": {
            "max_stage1_preflight_configs": cfg["optimizer"].get("max_stage1_preflight_configs"),
            "max_stage2_recipes_after_shortlist": cfg["optimizer"].get("max_stage2_recipes_after_shortlist"),
        },
        "execution_order": [
            "evaluate unique Stage 1 candidate-generation configurations only",
            "apply only the Stage 1 guardrails explicitly configured by the user",
            "keep passing Stage 1 configurations within the recall tolerance of best labeled-image mean recall",
            "rank recall-eligible Stage 1 configurations by higher labeled-image mean F1, then deterministic id",
            "shortlist Stage 1 configurations",
            "expand shortlisted Stage 1 configurations into Stage 2 feature-pack/fit recipes",
            "train/evaluate Stage 2 recipes and choose final winner",
        ],
    }


def _enforce_optimizer_plan_safety(plan: dict[str, Any], cfg: dict[str, Any]) -> None:
    optimizer_cfg = cfg.get("optimizer", {})
    stage1_cap = optimizer_cfg.get("max_stage1_preflight_configs")
    if stage1_cap is not None and int(plan["unique_stage1_preflight_configs"]) > int(stage1_cap):
        raise RuntimeError(
            "SNAPpy optimizer safety stop: "
            f"{int(plan['unique_stage1_preflight_configs'])} unique Stage 1 preflight configs exceeds "
            f"max_stage1_preflight_configs={int(stage1_cap)}. Edit the config intentionally if this is expected."
        )
    stage2_cap = optimizer_cfg.get("max_stage2_recipes_after_shortlist")
    if stage2_cap is not None and int(plan["max_stage2_recipe_entries_after_shortlist"]) > int(stage2_cap):
        raise RuntimeError(
            "SNAPpy optimizer safety stop: "
            f"{int(plan['max_stage2_recipe_entries_after_shortlist'])} possible Stage 2 recipes after shortlist exceeds "
            f"max_stage2_recipes_after_shortlist={int(stage2_cap)}. Edit the config intentionally if this is expected."
        )


def optimizer_plan(config_path: str | Path, enforce_safety: bool = True) -> dict[str, Any]:
    cfg = load_config(config_path)
    if cfg.get("dataset_root"):
        profile = _profile_dataset(cfg)
    else:
        profile = {"enabled": False, "reason": "dataset_root is not set"}
    cfg = _apply_profile_guidance(cfg, profile)
    explicit_recipes = str(cfg.get("stage1_detector_set", "")).strip().lower() == "custom"
    stage1_recipes = stage1_recipe_bank(cfg)
    stage1_recipes = _apply_runtime_recipe_guidance(stage1_recipes, cfg, profile, explicit_recipes=explicit_recipes)
    plan = _optimizer_plan_from_materialized(cfg, profile, stage1_recipes)
    if enforce_safety:
        _enforce_optimizer_plan_safety(plan, cfg)
    return plan


def _stage1_failure_reasons(
    *,
    mean_recall: float,
    mean_candidates: float,
    max_candidates: int,
    mean_candidate_ratio: float,
    preflight_cfg: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    min_recall = preflight_cfg.get("min_stage1_recall_mean")
    max_mean_candidates = preflight_cfg.get("max_stage1_candidates_mean")
    max_single_candidates = preflight_cfg.get("max_stage1_candidates_single")
    ratio_cap = _active_candidate_ratio_cap(preflight_cfg)
    if min_recall is not None and float(mean_recall) < float(min_recall):
        reasons.append(f"mean recall on labeled images {float(mean_recall):.4f} < minimum {float(min_recall):.4f}")
    if max_mean_candidates is not None and float(mean_candidates) > float(max_mean_candidates):
        reasons.append(f"mean candidates {float(mean_candidates):.1f} > maximum {float(max_mean_candidates):.1f}")
    if max_single_candidates is not None and int(max_candidates) > float(max_single_candidates):
        reasons.append(f"single-image candidates {int(max_candidates)} > maximum {_format_guardrail_number(float(max_single_candidates))}")
    if ratio_cap is not None and float(mean_candidate_ratio) > float(ratio_cap):
        reasons.append(f"candidate/ground-truth ratio {float(mean_candidate_ratio):.2f} > maximum {float(ratio_cap):.2f}")
    return reasons


def _format_guardrail_number(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _preflight_pass_label(reasons: list[str]) -> str:
    return "passed all Stage 1 guardrails" if not reasons else "; ".join(reasons)


def _apply_preflight_decision_columns(
    preflight_df: pd.DataFrame,
    shortlist_ids: set[str] | None = None,
    stage2_ids: set[str] | None = None,
    ranking_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = _ensure_preflight_sort_columns(preflight_df)
    shortlist_ids = shortlist_ids or set()
    stage2_ids = stage2_ids or set()
    if out.empty:
        return out

    out["stage1_recall_eligible"] = False
    out["stage1_recall_cutoff"] = pd.NA
    out["stage1_best_recall"] = pd.NA
    out["stage1_ranking_recall_tolerance"] = pd.NA
    out["stage1_rank_passed"] = pd.NA
    passed_ranked = _stage1_ranked_passed_df(out[out["passed"]], ranking_cfg)
    if not passed_ranked.empty:
        copied_cols = [
            "stage1_recall_eligible",
            "stage1_recall_cutoff",
            "stage1_best_recall",
            "stage1_ranking_recall_tolerance",
        ]
        out.loc[passed_ranked.index, copied_cols] = passed_ranked[copied_cols]
        eligible_ranked = passed_ranked[passed_ranked["stage1_recall_eligible"].astype(bool)]
        if not eligible_ranked.empty:
            passed_rank = pd.Series(range(1, len(eligible_ranked) + 1), index=eligible_ranked.index, dtype="int64")
            out.loc[eligible_ranked.index, "stage1_rank_passed"] = passed_rank
    out["shortlisted_for_stage2"] = out["recipe_id"].astype(str).isin(shortlist_ids)
    out["selected_for_stage2"] = out["recipe_id"].astype(str).isin(stage2_ids)
    report_order = pd.Series(len(out), index=out.index, dtype="int64")
    ranked = out[out["stage1_rank_passed"].notna()].sort_values("stage1_rank_passed")
    if not ranked.empty:
        report_order.loc[ranked.index] = range(0, len(ranked))
    unranked_passed = out[(out["passed"]) & (out["stage1_rank_passed"].isna())].sort_values("recipe_id")
    if not unranked_passed.empty:
        start = int(report_order.loc[ranked.index].max() + 1) if not ranked.empty else 0
        report_order.loc[unranked_passed.index] = range(start, start + len(unranked_passed))
    failed = out[~out["passed"]].sort_values("recipe_id")
    if not failed.empty:
        start = int(report_order.loc[out["passed"]].max() + 1) if bool(out["passed"].any()) else 0
        report_order.loc[failed.index] = range(start, start + len(failed))
    return (
        out.assign(_stage1_report_order=report_order)
        .sort_values(["_stage1_report_order", "recipe_id"], ascending=[True, True])
        .drop(columns=["_stage1_report_order"])
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _remove_optimizer_progress_files(run_dir: Path) -> None:
    for name in ("preflight_progress.jsonl", "stage2_progress.jsonl"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(val) for val in value]
    if isinstance(value, tuple):
        return [_json_ready(val) for val in value]
    return _json_scalar(value)


def _stage1_preflight_recipe_table(preflight_df: pd.DataFrame) -> pd.DataFrame:
    """Compact recipe-level Stage 1 preflight diagnostics for terminal failures."""
    if preflight_df.empty:
        return pd.DataFrame()
    columns = [
        "recipe_id",
        "passed",
        "stage1_decision",
        "stage1_stopped_early",
        "stage1_early_stop_reason",
        "stage1_preflight_images_processed",
        "stage1_preflight_images_total",
        "stage1_labeled_preflight_images_processed",
        "stage1_labeled_preflight_images_total",
        "stage1_empty_gt_preflight_images_processed",
        "stage1_empty_gt_preflight_images_total",
        "mean_stage1_recall",
        "mean_stage1_precision",
        "mean_stage1_f1",
        "total_stage1_tp",
        "total_stage1_fp",
        "total_stage1_fn",
        "mean_stage1_candidates",
        "mean_stage1_label_count",
        "mean_stage1_candidate_ratio",
        "max_stage1_candidates",
        "stage1_maximum_possible_mean_recall",
        "stage1_minimum_possible_mean_candidates",
        "stage1_minimum_possible_candidate_ratio",
        "stage2_recipe_count",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in preflight_df.iterrows():
        record = {column: _json_scalar(row[column]) for column in columns if column in row.index}
        recipe = dict(row.get("recipe", {}))
        for key in _STAGE1_SCREEN_FIELDS:
            if key in recipe:
                record[key] = _json_ready(recipe.get(key))
        rows.append(record)
    return pd.DataFrame(rows)


def _stage1_preflight_failure_records(
    preflight_df: pd.DataFrame,
    *,
    sort_columns: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    if preflight_df.empty:
        return []
    table = _stage1_preflight_recipe_table(preflight_df)
    if table.empty:
        return []
    descending_columns = {"mean_stage1_f1", "mean_stage1_recall", "mean_stage1_precision"}
    ascending = [column not in descending_columns for column in sort_columns]
    table = table.sort_values(sort_columns, ascending=ascending).head(int(top_n))
    return [_json_ready(record) for record in table.to_dict(orient="records")]


def _write_stage1_preflight_failure_outputs(
    run_dir: Path,
    *,
    preflight_df: pd.DataFrame,
    cfg: dict[str, Any],
    status: str,
    message: str,
) -> dict[str, Any]:
    """Write compact Stage 1 diagnostics before deleting raw optimizer progress logs."""
    recipe_table = _stage1_preflight_recipe_table(preflight_df)
    recipe_csv_path = run_dir / "stage1_preflight_failure_recipes.csv"
    recipe_table.to_csv(recipe_csv_path, index=False)
    decision_counts = (
        preflight_df["stage1_decision"].fillna("unknown").astype(str).value_counts().to_dict()
        if "stage1_decision" in preflight_df
        else {}
    )
    summary = {
        "schema": "mrsnappy_stage1_preflight_failure_summary_v1",
        "status": status,
        "message": message,
        "purpose": (
            "Compact diagnostics for optimization runs that stopped before Stage 2 "
            "because no Stage 1 recipe passed the configured preflight guardrails."
        ),
        "recipe_count": int(len(preflight_df)),
        "passed_recipe_count": int(preflight_df["passed"].sum()) if "passed" in preflight_df else 0,
        "failed_recipe_count": (
            int((~preflight_df["passed"].astype(bool)).sum())
            if "passed" in preflight_df
            else int(len(preflight_df))
        ),
        "guardrails": _json_ready(cfg.get("preflight", {})),
        "decision_counts": {str(key): int(value) for key, value in decision_counts.items()},
        "top_recipes_by_stage1_f1": _stage1_preflight_failure_records(
            preflight_df,
            sort_columns=["mean_stage1_f1", "mean_stage1_recall", "recipe_id"],
            top_n=10,
        ),
        "lowest_candidate_ratio_recipes": _stage1_preflight_failure_records(
            preflight_df,
            sort_columns=["mean_stage1_candidate_ratio", "mean_stage1_f1", "recipe_id"],
            top_n=10,
        ),
        "output_files": {
            "stage1_preflight_failure_summary": "stage1_preflight_failure_summary.json",
            "stage1_preflight_failure_recipes": "stage1_preflight_failure_recipes.csv",
        },
    }
    summary_path = run_dir / "stage1_preflight_failure_summary.json"
    write_json(summary_path, _json_ready(summary))
    return {
        "summary": summary,
        "output_files": {
            "stage1_preflight_failure_summary": summary["output_files"]["stage1_preflight_failure_summary"],
            "stage1_preflight_failure_recipes": summary["output_files"]["stage1_preflight_failure_recipes"],
        },
    }


def _stage1_score_payload(winner: pd.Series) -> dict[str, Any]:
    return {
        "rank_passed": _json_scalar(winner.get("stage1_rank_passed")),
        "mean_recall": _json_scalar(winner.get("mean_stage1_recall")),
        "mean_precision": _json_scalar(winner.get("mean_stage1_precision")),
        "mean_f1": _json_scalar(winner.get("mean_stage1_f1")),
        "mean_candidates": _json_scalar(winner.get("mean_stage1_candidates")),
        "mean_candidate_ratio": _json_scalar(winner.get("mean_stage1_candidate_ratio")),
        "max_candidates": _json_scalar(winner.get("max_stage1_candidates")),
        "decision": _json_scalar(winner.get("stage1_decision")),
    }


def _validation_metric_payload(winner: pd.Series) -> dict[str, Any]:
    return {
        "selection_metric": "mean per-image F1",
        "tp": int(winner.get("val_tp", 0)),
        "fp": int(winner.get("val_fp", 0)),
        "fn": int(winner.get("val_fn", 0)),
        "precision_mean_image": float(winner.get("val_precision_mean_image", winner.get("val_precision", 0.0))),
        "recall_mean_image": float(winner.get("val_recall_mean_image", winner.get("val_recall", 0.0))),
        "f1_mean_image": float(winner.get("val_f1_mean_image", winner.get("val_f1", 0.0))),
        "precision_pooled": float(winner.get("val_precision_pooled", 0.0)),
        "recall_pooled": float(winner.get("val_recall_pooled", 0.0)),
        "f1_pooled": float(winner.get("val_f1_pooled", 0.0)),
    }


def _svm_parameter_payload(winner: pd.Series) -> dict[str, Any] | None:
    if _is_stage1_pass_through_row(winner):
        return None
    return {
        "kernel": _json_scalar(winner.get("kernel")),
        "C": _json_scalar(winner.get("C")),
        "gamma": _json_scalar(winner.get("gamma")),
        "degree": _json_scalar(winner.get("degree")),
        "standardize": _json_scalar(winner.get("standardize")),
        "class_weighting": _json_scalar(winner.get("class_weighting")),
    }


def _stage2_parameter_payload(recipe: dict[str, Any], winner: pd.Series) -> dict[str, Any]:
    pass_through = _is_stage1_pass_through_row(winner)
    payload = {
        "model_type": "stage1_pass_through" if pass_through else "svm",
        "fitting_mode": recipe.get("fit_method"),
        "fit_window": recipe.get("fit_window"),
        "fit_background_width": recipe.get("fit_background_width"),
        "fit_max_iterations": recipe.get("fit_max_iterations"),
        "fit_tolerance": recipe.get("fit_tolerance"),
        "feature_pack_name": recipe.get("feature_pack_name"),
        "selected_features": list(recipe.get("selected_features") or []),
        "svm": _svm_parameter_payload(winner),
    }
    if pass_through:
        payload["decision_rule"] = "accept_all_stage1_candidates"
        payload["decision_threshold"] = None
    else:
        payload["decision_threshold"] = float(winner.get("decision_threshold", 0.0))
    return payload


def _stage2_finalist_records(finalists: pd.DataFrame, winner: pd.Series) -> list[dict[str, Any]]:
    if finalists.empty:
        return []
    columns = [
        "recipe_id",
        "stage1_recipe_id",
        "feature_pack_name",
        "fitting_mode",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_f1_pooled",
        "stage2_f1_loss",
        "feature_pack_simplicity_rank",
        "model_type_simplicity_rank",
        "svm_kernel_simplicity_rank",
        "svm_c_simplicity_rank",
        "svm_degree_simplicity_rank",
        "svm_gamma_simplicity_rank",
        "stage1_rank_passed",
        "decision_threshold",
        "kernel",
        "C",
        "gamma",
        "degree",
        "standardize",
        "class_weighting",
    ]
    records: list[dict[str, Any]] = []
    for _, row in finalists.sort_values(_FINALIST_SORT_COLUMNS, ascending=_FINALIST_SORT_ASCENDING).iterrows():
        record = {col: _json_scalar(row[col]) for col in columns if col in row.index}
        record["winner"] = str(row["recipe_id"]) == str(winner["recipe_id"])
        records.append(record)
    return records


def _stage1_parameter_payload(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_ready(recipe.get(key)) for key in _STAGE1_SCREEN_FIELDS if key in recipe}


def _stage1_shortlist_records(stage1_shortlist: pd.DataFrame) -> list[dict[str, Any]]:
    if stage1_shortlist.empty:
        return []
    rows = stage1_shortlist.sort_values(["stage1_rank_passed", "recipe_id"], ascending=[True, True])
    records: list[dict[str, Any]] = []
    metric_columns = [
        "stage1_rank_passed",
        "stage1_recall_eligible",
        "stage1_best_recall",
        "stage1_recall_cutoff",
        "stage1_ranking_recall_tolerance",
        "mean_stage1_recall",
        "mean_stage1_precision",
        "mean_stage1_f1",
        "total_stage1_tp",
        "total_stage1_fp",
        "total_stage1_fn",
        "mean_stage1_candidates",
        "mean_stage1_label_count",
        "mean_stage1_candidate_ratio",
        "max_stage1_candidates",
        "stage1_preflight_images_processed",
        "stage1_preflight_images_total",
        "stage1_full_val_image_count",
        "stage1_full_val_tp",
        "stage1_full_val_fp",
        "stage1_full_val_fn",
        "stage1_full_val_precision",
        "stage1_full_val_recall",
        "stage1_full_val_f1",
        "stage1_full_val_mean_tp",
        "stage1_full_val_mean_candidates",
        "stage1_full_val_candidate_ratio",
        "stage1_train_label_status",
        "stage1_train_images",
        "stage1_train_candidates",
        "stage1_train_positive_candidates",
        "stage1_train_negative_candidates",
        "stage1_train_ground_truth",
        "stage2_recipe_count",
    ]
    for _, row in rows.iterrows():
        recipe = dict(row.get("recipe", {}))
        record = {
            "stage1_recipe_id": str(row.get("recipe_id")),
            "stage1_key": _json_scalar(row.get("stage1_key")),
            "stage1_parameters": _stage1_parameter_payload(recipe),
        }
        for column in metric_columns:
            if column in row.index:
                record[column] = _json_scalar(row[column])
        records.append(record)
    return records


def _stage1_guardrail_summary(
    cfg: dict[str, Any],
    *,
    requested_preflight_image_count: int | str,
    preflight_image_count: int,
    available_validation_image_count: int,
) -> dict[str, Any]:
    preflight_cfg = cfg["preflight"]
    return {
        "stage1_n_val_images_requested": requested_preflight_image_count,
        "stage1_n_val_images_used": int(preflight_image_count),
        "available_validation_image_count": int(available_validation_image_count),
        "min_stage1_recall_mean": _json_scalar(preflight_cfg.get("min_stage1_recall_mean")),
        "max_stage1_candidates_mean": _json_scalar(preflight_cfg.get("max_stage1_candidates_mean")),
        "max_stage1_candidates_single": _json_scalar(preflight_cfg.get("max_stage1_candidates_single")),
        "max_candidate_ratio_cap_mean": _json_scalar(preflight_cfg.get("max_candidate_ratio_cap_mean")),
        "disabled_guardrails": [
            key
            for key in (
                "min_stage1_recall_mean",
                "max_stage1_candidates_mean",
                "max_stage1_candidates_single",
                "max_candidate_ratio_cap_mean",
            )
            if preflight_cfg.get(key) is None
        ],
    }


def _stage1_ranking_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall_tolerance": float(cfg["stage1_ranking"]["recall_tolerance"]),
        "shortlist_top_k": int(cfg["optimizer"]["shortlist_top_k"]),
        "ranking_order": [
            "apply configured Stage 1 guardrails",
            "keep passing recipes with labeled-image mean recall within recall_tolerance of the best passing recipe",
            "rank recall-eligible recipes by higher labeled-image mean Stage 1 F1",
            "break exact ties by recipe_id",
            "send the top shortlist_top_k unique Stage 1 configurations to Stage 2",
        ],
    }


def _stage2_winner_record(winner: pd.Series) -> dict[str, Any]:
    recipe = dict(winner["recipe"])
    pass_through = _is_stage1_pass_through_row(winner)
    return {
        "stage2_recipe_id": str(winner["recipe_id"]),
        "origin_stage1_recipe_id": str(winner.get("stage1_recipe_id", "")),
        "origin_stage1_rank_passed": _json_scalar(winner.get("stage1_rank_passed")),
        "model_type": "stage1_pass_through" if pass_through else "svm",
        "feature_pack_name": recipe.get("feature_pack_name"),
        "selected_features": list(recipe.get("selected_features") or []),
        "fitting_mode": recipe.get("fit_method"),
        "fit_window": recipe.get("fit_window"),
        "svm": _svm_parameter_payload(winner),
        "decision_threshold": None if pass_through else float(winner.get("decision_threshold", 0.0)),
        "validation_metrics": _validation_metric_payload(winner),
        "stage1_score": _stage1_score_payload(winner),
    }


def _format_summary_value(value: Any, digits: int = 4) -> str:
    value = _json_scalar(value)
    if value is None:
        return "not used"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _stage1_parameter_brief(parameters: dict[str, Any]) -> str:
    fields = [
        "maxima_method",
        "preproc_sigma",
        "preproc_sigma_nm",
        "background_enabled",
        "background_method",
        "background_param",
        "background_param_nm",
        "sigma_value",
        "sigma_nm",
        "threshold_value",
        "h_max_sigma_multiplier",
        "h_max_sigma_mode",
        "maxima_neighborhood",
        "maxima_min_distance_nm",
    ]
    parts = [f"{key}={_format_summary_value(parameters[key])}" for key in fields if key in parameters]
    return ", ".join(parts)


def _summary_markdown_lines(
    *,
    cfg: dict[str, Any],
    mode: str,
    split_summary: dict[str, Any],
    profile: dict[str, Any],
    final_model_path: Path,
    stage1_guardrails: dict[str, Any],
    stage1_ranking: dict[str, Any],
    stage1_shortlist_records: list[dict[str, Any]],
    stage2_winner: dict[str, Any],
) -> list[str]:
    lines = [
        "# SNAPpy Optimized Model",
        "",
        "## Model Output",
        f"- Model for future `mrsnappy detect`: `{final_model_path}`",
        "- Complete model record: `model_config.json`",
        "- Human-readable model summary: `model_summary.md`",
        "- Train/validation split record: `optimization_splits.csv`",
        "",
        "## Dataset",
        f"- Dataset: `{cfg['dataset_name']}`",
        f"- Optimization mode: `{mode}`",
        "- Validation strategy: fixed user-supplied `train/` and `val/` folders.",
        f"- Train images: `{split_summary['folds'][0]['train_image_count']}`",
        f"- Validation images: `{split_summary['folds'][0]['val_image_count']}`",
        f"- Density regime: `{profile.get('density_regime', 'unknown')}`",
        f"- Contrast regime: `{profile.get('contrast_regime', 'unknown')}`",
        f"- Background regime: `{profile.get('background_regime', 'unknown')}`",
        "",
        "## Stage 1 Guardrails",
        f"- Preflight validation images requested: `{stage1_guardrails['stage1_n_val_images_requested']}`",
        f"- Preflight validation images used: `{stage1_guardrails['stage1_n_val_images_used']}` of `{stage1_guardrails['available_validation_image_count']}` available validation images.",
        f"- Minimum mean recall on labeled images: `{_format_summary_value(stage1_guardrails['min_stage1_recall_mean'])}`",
        f"- Maximum mean candidates per image: `{_format_summary_value(stage1_guardrails['max_stage1_candidates_mean'], digits=1)}`",
        f"- Maximum candidates in a single image: `{_format_summary_value(stage1_guardrails['max_stage1_candidates_single'], digits=1)}`",
        f"- Maximum mean candidate/GT ratio: `{_format_summary_value(stage1_guardrails['max_candidate_ratio_cap_mean'], digits=2)}`",
        "",
        "## Stage 1 Ranking",
        f"- Recall tolerance: `{stage1_ranking['recall_tolerance']:.4f}`",
        f"- Stage 1 recipes sent to Stage 2: top `{stage1_ranking['shortlist_top_k']}` unique Stage 1 configurations after guardrails, recall-band filtering, F1 ranking, and recipe-id tie breaking.",
        "",
        "## Stage 1 Winning Recipes",
    ]
    lines.extend(
        _markdown_table(
            [
                "rank",
                "stage1_recipe_id",
                "recall",
                "precision",
                "f1",
                "tp/fp/fn",
                "mean candidates",
                "cand/GT",
                "max candidates",
                "train labels",
            ],
            [
                [
                    _format_summary_value(row.get("stage1_rank_passed"), digits=0),
                    f"`{row['stage1_recipe_id']}`",
                    _format_summary_value(row.get("mean_stage1_recall")),
                    _format_summary_value(row.get("mean_stage1_precision")),
                    _format_summary_value(row.get("mean_stage1_f1")),
                    f"{_format_summary_value(row.get('total_stage1_tp'), digits=0)}/{_format_summary_value(row.get('total_stage1_fp'), digits=0)}/{_format_summary_value(row.get('total_stage1_fn'), digits=0)}",
                    _format_summary_value(row.get("mean_stage1_candidates"), digits=1),
                    _format_summary_value(row.get("mean_stage1_candidate_ratio"), digits=2),
                    _format_summary_value(row.get("max_stage1_candidates"), digits=0),
                    _format_summary_value(row.get("stage1_train_label_status")),
                ]
                for row in stage1_shortlist_records
            ],
        )
    )
    lines.extend(["", "### Stage 1 Recipe Parameters"])
    for row in stage1_shortlist_records:
        brief = _stage1_parameter_brief(row["stage1_parameters"])
        lines.append(f"- Rank `{_format_summary_value(row.get('stage1_rank_passed'), digits=0)}` `{row['stage1_recipe_id']}`: {brief}")
    lines.extend(
        [
            "",
            "## Stage 2 Winner",
            f"- Stage 2 recipe: `{stage2_winner['stage2_recipe_id']}`",
            f"- Originating Stage 1 recipe: `{stage2_winner['origin_stage1_recipe_id']}`",
            f"- Originating Stage 1 rank: `{_format_summary_value(stage2_winner['origin_stage1_rank_passed'], digits=0)}`",
            f"- Model type: `{stage2_winner['model_type']}`",
            f"- Feature pack: `{stage2_winner['feature_pack_name']}`",
            f"- Fitting mode: `{stage2_winner['fitting_mode']}`",
            f"- Selected feature count: `{len(stage2_winner['selected_features'])}`",
            f"- SVM parameters: `{json.dumps(stage2_winner['svm'], sort_keys=True) if stage2_winner['svm'] is not None else 'none'}`",
            f"- Decision threshold: `{_format_summary_value(stage2_winner['decision_threshold'], digits=6)}`",
            f"- Validation F1, mean per image: `{stage2_winner['validation_metrics']['f1_mean_image']:.6f}`",
            f"- Validation precision, mean per image: `{stage2_winner['validation_metrics']['precision_mean_image']:.6f}`",
            f"- Validation recall, mean per image: `{stage2_winner['validation_metrics']['recall_mean_image']:.6f}`",
            f"- Validation F1, pooled TP/FP/FN: `{stage2_winner['validation_metrics']['f1_pooled']:.6f}`",
        ]
    )
    return lines


def _write_model_config(
    path: Path,
    *,
    cfg: dict[str, Any],
    mode: str,
    final_model_path: Path,
    winner: pd.Series,
    finalists: pd.DataFrame,
    split_summary: dict[str, Any],
    profile: dict[str, Any],
    optimizer_plan_payload: dict[str, Any],
    stage1_guardrails: dict[str, Any],
    stage1_ranking: dict[str, Any],
    stage1_shortlist_records: list[dict[str, Any]],
) -> None:
    recipe = dict(winner["recipe"])
    payload = {
        "schema": "mrsnappy_model_config_v1",
        "purpose": "Complete record for the optimized SNAPpy model saved in model.joblib.",
        "dataset_name": cfg["dataset_name"],
        "optimization_mode": mode,
        "validation_strategy": "fixed user-supplied train/ and val/ folders",
        "model_path": str(final_model_path),
        "output_files": {
            "model": "model.joblib",
            "model_config": "model_config.json",
            "model_summary": "model_summary.md",
            "optimization_splits": "optimization_splits.csv",
        },
        "effective_config": _json_ready(cfg),
        "dataset_profile": _json_ready(profile),
        "optimizer_plan": _json_ready(optimizer_plan_payload),
        "optimization_split_summary": split_summary,
        "stage1_recipe_id": str(winner.get("stage1_recipe_id", "")),
        "stage2_recipe_id": str(winner["recipe_id"]),
        "stage1_parameters": {
            key: _json_ready(recipe.get(key))
            for key in _STAGE1_SCREEN_FIELDS
            if key in recipe
        },
        "stage2_parameters": _stage2_parameter_payload(recipe, winner),
        "winner_scores": {
            "stage1": _stage1_score_payload(winner),
            "stage2_validation": _validation_metric_payload(winner),
        },
        "stage1_selection": {
            "guardrails": stage1_guardrails,
            "ranking": stage1_ranking,
            "shortlisted_stage1_recipes": stage1_shortlist_records,
        },
        "selection": {
            "stage2_f1_tolerance_config": _stage2_f1_tolerance(cfg),
            "winner_stage2_f1_loss": float(winner.get("stage2_f1_loss", 0.0)),
            "ranking_order": [
                "keep recipes within stage2_f1_tolerance of the best mean per-image validation F1",
                "prefer simpler feature pack",
                "prefer simpler model and SVM hyperparameters",
                "then prefer better Stage 1 rank",
                "deterministic recipe_id",
            ],
            "finalists_within_stage2_f1_tolerance": _stage2_finalist_records(finalists, winner),
        },
        "full_recipe": _json_ready(recipe),
    }
    write_json(path, payload)


def _write_terminal_model_outputs(
    run_dir: Path,
    *,
    cfg: dict[str, Any],
    mode: str,
    status: str,
    message: str,
    split_summary: dict[str, Any],
    profile: dict[str, Any],
    optimizer_plan_payload: dict[str, Any],
    requested_preflight_image_count: int | str | None = None,
    preflight_image_count: int = 0,
    available_validation_image_count: int = 0,
    extra_output_files: dict[str, str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    model_summary_path = run_dir / "model_summary.md"
    model_config_path = run_dir / "model_config.json"
    stage1_guardrails = (
        _stage1_guardrail_summary(
            cfg,
            requested_preflight_image_count=requested_preflight_image_count or 0,
            preflight_image_count=preflight_image_count,
            available_validation_image_count=available_validation_image_count,
        )
        if "preflight" in cfg
        else {}
    )
    output_files = {
        "model_config": "model_config.json",
        "model_summary": "model_summary.md",
        "optimization_splits": "optimization_splits.csv",
    }
    if extra_output_files:
        output_files.update(extra_output_files)
    payload = {
        "schema": "mrsnappy_model_config_v1",
        "purpose": "SNAPpy optimization terminal record. No usable model.joblib was produced.",
        "status": status,
        "message": message,
        "dataset_name": cfg.get("dataset_name"),
        "optimization_mode": mode,
        "validation_strategy": "fixed user-supplied train/ and val/ folders",
        "output_files": output_files,
        "effective_config": _json_ready(cfg),
        "dataset_profile": _json_ready(profile),
        "optimizer_plan": _json_ready(optimizer_plan_payload),
        "optimization_split_summary": split_summary,
        "stage1": {
            "guardrails": stage1_guardrails,
            "ranking": _stage1_ranking_summary(cfg) if "stage1_ranking" in cfg else {},
        },
    }
    if diagnostics:
        payload["diagnostics"] = _json_ready(diagnostics)
    write_json(
        model_config_path,
        payload,
    )
    lines = [
        "# SNAPpy Optimizer",
        "",
        f"Status: `{status}`",
        "",
        message,
        "",
        "No `model.joblib` was produced.",
    ]
    if extra_output_files:
        lines.extend(
            [
                "",
                "## Diagnostic Outputs",
                "",
                *[f"- `{name}`: `{filename}`" for name, filename in sorted(extra_output_files.items())],
            ]
        )
    model_summary_path.write_text("\n".join(lines) + "\n")
    return model_summary_path


def _safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _sample_gt_intensities(volume: np.ndarray, gt: np.ndarray) -> list[float]:
    if len(gt) == 0:
        return []
    idx = np.rint(gt).astype(np.int64)
    idx[:, 0] = np.clip(idx[:, 0], 0, volume.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, volume.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, volume.shape[2] - 1)
    return volume[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32).tolist()


def _profile_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    profiling_cfg = cfg.get("profiling", {})
    if not profiling_cfg.get("enabled", True):
        return {"enabled": False}

    n_train = int(profiling_cfg.get("train_image_count", 8))
    n_val = int(profiling_cfg.get("val_image_count", 4))
    sampled_pairs = split_pairs(cfg["dataset_root"], "train")[:n_train] + split_pairs(cfg["dataset_root"], "val")[:n_val]
    label_counts: list[float] = []
    p50s: list[float] = []
    p90s: list[float] = []
    p99s: list[float] = []
    stds: list[float] = []
    gt_intensities: list[float] = []

    for image_path, label_path in sampled_pairs:
        gt = read_points_csv(label_path)
        label_counts.append(float(len(gt)))
        volume = read_volume(image_path)
        p50s.append(float(np.percentile(volume, 50.0)))
        p90s.append(float(np.percentile(volume, 90.0)))
        p99s.append(float(np.percentile(volume, 99.0)))
        stds.append(float(np.std(volume)))
        gt_intensities.extend(_sample_gt_intensities(volume, gt))

    mean_labels = float(np.mean(label_counts)) if label_counts else 0.0
    sparse_max = float(profiling_cfg.get("sparse_label_mean_max", 64.0))
    dense_min = float(profiling_cfg.get("dense_label_mean_min", 256.0))
    if mean_labels <= sparse_max:
        density_regime = "sparse"
    elif mean_labels >= dense_min:
        density_regime = "dense"
    else:
        density_regime = "moderate"

    median_gt = _safe_quantile(gt_intensities, 0.5)
    median_p90 = _safe_quantile(p90s, 0.5)
    median_p99 = _safe_quantile(p99s, 0.5)
    median_std = _safe_quantile(stds, 0.5)
    gt_over_p90 = float(median_gt / max(median_p90, 1e-6))
    gt_minus_p90_over_std = float((median_gt - median_p90) / max(median_std, 1e-6))
    if gt_minus_p90_over_std < 2.5:
        contrast_regime = "low"
    elif gt_minus_p90_over_std < 6.0:
        contrast_regime = "moderate"
    else:
        contrast_regime = "high"

    background_ratio = float((median_p99 - _safe_quantile(p50s, 0.5)) / max(median_std, 1e-6))
    background_regime = "challenging" if background_ratio < 6.0 else "stable"

    return {
        "enabled": True,
        "n_sampled_images": len(sampled_pairs),
        "label_count_mean": mean_labels,
        "label_count_median": _safe_quantile(label_counts, 0.5),
        "label_count_p90": _safe_quantile(label_counts, 0.9),
        "label_count_max": max(label_counts) if label_counts else 0.0,
        "image_p50_median": _safe_quantile(p50s, 0.5),
        "image_p90_median": median_p90,
        "image_p99_median": median_p99,
        "image_std_median": median_std,
        "gt_intensity_median": median_gt,
        "gt_over_p90_ratio": gt_over_p90,
        "gt_minus_p90_over_std": gt_minus_p90_over_std,
        "density_regime": density_regime,
        "contrast_regime": contrast_regime,
        "background_regime": background_regime,
    }


def _active_candidate_ratio_cap(preflight_cfg: dict[str, Any]) -> float | None:
    ratio_cap = preflight_cfg.get("max_candidate_ratio_cap_mean")
    if ratio_cap is None:
        return None
    return float(ratio_cap)


def _append_unique_stage1(stage1_rows: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {json.dumps(row, sort_keys=True, default=str) for row in stage1_rows}
    out = list(stage1_rows)
    for row in additions:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _augment_stage1_recipes(cfg: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(cfg.get("stage1_recipes", []))
    density = str(profile.get("density_regime", "moderate"))
    contrast = str(profile.get("contrast_regime", "moderate"))
    background = str(profile.get("background_regime", "stable"))
    additions: list[dict[str, Any]] = []

    if density == "sparse":
        additions.extend(
            [
                {
                    "sigma_value": 2.0,
                    "maxima_neighborhood": 2,
                    "threshold_value": 0.095,
                    "norm_enabled": True,
                    "norm_method": "robust_z_score",
                    "background_enabled": True,
                    "background_method": "rolling_box_3d",
                    "background_param": 10.0,
                    "background_clip": False,
                },
                {
                    "sigma_value": 2.5,
                    "maxima_neighborhood": 2,
                    "threshold_value": 0.08,
                    "norm_enabled": True,
                    "norm_method": "robust_z_score",
                    "background_enabled": True,
                    "background_method": "rolling_box_3d",
                    "background_param": 20.0,
                    "background_clip": False,
                },
                {
                    "sigma_value": 2.0,
                    "maxima_neighborhood": 3,
                    "threshold_value": 0.10,
                    "norm_enabled": True,
                    "norm_method": "robust_z_score",
                    "background_enabled": True,
                    "background_method": "rolling_box_3d",
                    "background_param": 10.0,
                    "background_clip": False,
                },
            ]
        )
    if contrast == "low" or background == "challenging":
        additions.extend(
            [
                {
                    "sigma_value": 1.35,
                    "maxima_neighborhood": 2,
                    "threshold_value": 0.20,
                    "norm_enabled": True,
                    "norm_method": "robust_z_score",
                    "background_enabled": True,
                    "background_method": "rolling_box_3d",
                    "background_param": 20.0,
                    "background_clip": False,
                },
                {
                    "sigma_value": 2.0,
                    "maxima_neighborhood": 2,
                    "threshold_value": 0.10,
                    "norm_enabled": True,
                    "norm_method": "robust_z_score",
                    "background_enabled": True,
                    "background_method": "rolling_box_3d",
                    "background_param": 20.0,
                    "background_clip": False,
                },
            ]
        )
    if density == "dense":
        additions.extend(
            [
                {"sigma_value": 1.0, "maxima_neighborhood": 1, "threshold_value": 0.50},
                {"sigma_value": 1.35, "maxima_neighborhood": 1, "threshold_value": 0.50},
                {"sigma_value": 1.35, "maxima_neighborhood": 2, "threshold_value": 0.20},
            ]
        )
    return _append_unique_stage1(rows, additions)


def _apply_profile_guidance(cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    guided = deepcopy(cfg)
    guided["dataset_profile"] = deepcopy(profile)

    if not profile.get("enabled", False):
        return guided

    explicit_recipes = str(guided.get("stage1_detector_set", "")).strip().lower() == "custom"
    if not explicit_recipes and bool(guided.get("profiling", {}).get("stage1_augmentation_enabled", False)):
        guided["stage1_recipes"] = _augment_stage1_recipes(guided, profile)
    return guided


def _apply_runtime_recipe_guidance(
    recipes: list[dict[str, Any]],
    cfg: dict[str, Any],
    profile: dict[str, Any],
    explicit_recipes: bool,
) -> list[dict[str, Any]]:
    if explicit_recipes and not bool(cfg.get("profiling", {}).get("apply_runtime_pruning_to_explicit_recipes", False)):
        return recipes

    density = str(profile.get("density_regime", "moderate"))
    labeled_per_label = 120.0 if density == "sparse" else 64.0
    unlabeled_per_label = 140.0 if density == "sparse" else 80.0
    min_candidates = 320
    nms_distance = 0.0
    full_fit_per_label = 20.0 if density == "sparse" else 16.0
    full_fit_unlabeled_per_label = 24.0 if density == "sparse" else 20.0
    full_fit_min = 384

    expected_labels = float(profile.get("label_count_mean", 0.0) or 0.0)
    out: list[dict[str, Any]] = []
    for recipe in recipes:
        row = deepcopy(recipe)
        row["prefit_prune_enabled"] = True
        row["prefit_rank_radius"] = int(row.get("prefit_rank_radius", 1) or 1)
        row["prefit_rank_bg_width"] = max(int(row.get("prefit_rank_bg_width", 1) or 1), 1)
        row["prefit_nms_distance"] = float(row.get("prefit_nms_distance", nms_distance) or nms_distance)
        row["prefit_labeled_candidates_per_label"] = float(row.get("prefit_labeled_candidates_per_label", labeled_per_label) or labeled_per_label)
        row["prefit_labeled_min_candidates"] = int(row.get("prefit_labeled_min_candidates", min_candidates) or min_candidates)
        row["prefit_unlabeled_candidates_per_expected_label"] = float(
            row.get("prefit_unlabeled_candidates_per_expected_label", unlabeled_per_label) or unlabeled_per_label
        )
        row["expected_labels_per_image"] = float(row.get("expected_labels_per_image", expected_labels) or expected_labels)
        row["full_fit_labeled_candidates_per_label"] = float(
            row.get("full_fit_labeled_candidates_per_label", full_fit_per_label) or full_fit_per_label
        )
        row["full_fit_unlabeled_candidates_per_expected_label"] = float(
            row.get("full_fit_unlabeled_candidates_per_expected_label", full_fit_unlabeled_per_label) or full_fit_unlabeled_per_label
        )
        row["full_fit_labeled_min_candidates"] = int(row.get("full_fit_labeled_min_candidates", full_fit_min) or full_fit_min)
        row["fit_fallback_method"] = str(row.get("fit_fallback_method", "moments"))
        out.append(row)
    return out


def _evaluate_stage2_recipe(
    recipe: dict[str, Any],
    cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    trial_dir = ensure_dir(run_dir / recipe["recipe_id"])
    model_path = trial_dir / "model.joblib"
    match_kwargs = _match_distance_kwargs(cfg)
    trained = train_native_model(
        dataset_root=cfg["dataset_root"],
        recipe=recipe,
        svm_cfg=cfg["svm_sweep"],
        model_path=model_path,
        **match_kwargs,
    )
    val_preds, val_gts = predict_split(
        dataset_root=cfg["dataset_root"],
        split="val",
        recipe=recipe,
        model_path=model_path,
        output_root=trial_dir / "val_predictions",
        score_threshold=float(trained.decision_threshold),
    )
    val_metrics = evaluate_predictions_for_selection(val_preds, val_gts, **match_kwargs)
    result = {
        "recipe_id": recipe["recipe_id"],
        "model_path": str(model_path),
        "decision_threshold": float(trained.decision_threshold),
        **trained.best_params,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "recipe": deepcopy(recipe),
    }
    write_json(trial_dir / "stage2_summary.json", result)
    return result


def optimize_native_dataset(
    config_path: str | Path,
    run_dir: str | Path,
) -> Path:
    clear_pipeline_caches()
    cfg = load_config(config_path)
    mode = _require_fixed_split_mode(cfg)
    run_dir = ensure_dir(run_dir)
    split_summary = _write_optimization_split_records(run_dir, cfg)
    profile = _profile_dataset(cfg)
    cfg = _apply_profile_guidance(cfg, profile)
    explicit_recipes = str(cfg.get("stage1_detector_set", "")).strip().lower() == "custom"
    stage1_recipes = stage1_recipe_bank(cfg)
    stage1_recipes = _apply_runtime_recipe_guidance(stage1_recipes, cfg, profile, explicit_recipes=explicit_recipes)
    plan = _optimizer_plan_from_materialized(cfg, profile, stage1_recipes)
    _enforce_optimizer_plan_safety(plan, cfg)
    progress_path = run_dir / "preflight_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    all_val_pairs = split_pairs(cfg["dataset_root"], "val")
    requested_preflight_images = cfg["preflight"]["stage1_n_val_images"]
    if isinstance(requested_preflight_images, str) and requested_preflight_images.strip().lower() == "all":
        requested_preflight_image_count: int | str = "all"
        val_pairs = all_val_pairs
    else:
        requested_preflight_image_count = int(requested_preflight_images)
        val_pairs = all_val_pairs[: min(requested_preflight_image_count, len(all_val_pairs))]
    preflight_label_counts = [int(len(read_points_csv(label_path))) for _, label_path in val_pairs]
    labeled_preflight_image_count = int(sum(1 for count in preflight_label_counts if count > 0))
    empty_gt_preflight_image_count = int(len(preflight_label_counts) - labeled_preflight_image_count)
    if cfg["preflight"].get("min_stage1_recall_mean") is not None and labeled_preflight_image_count == 0:
        raise ValueError(
            "preflight.min_stage1_recall_mean requires at least one selected preflight validation image "
            "with ground-truth labels. Use more validation images, set stage1_n_val_images='all', "
            "or disable the recall guardrail by setting it to null."
        )
    min_stage1_cache_entries = max(1, int(cfg["optimizer"]["shortlist_top_k"]) * max(len(all_val_pairs), len(val_pairs), 1))
    current_stage1_cache_entries = int(cfg.get("pipeline_defaults", {}).get("stage1_cache_entries", 0))
    if current_stage1_cache_entries < min_stage1_cache_entries:
        cfg["pipeline_defaults"]["stage1_cache_entries"] = min_stage1_cache_entries
        cfg["runtime_cache"]["stage1_cache_entries"] = min_stage1_cache_entries
    current_image_volume_cache_entries = int(cfg.get("pipeline_defaults", {}).get("image_volume_cache_entries", 0))
    if current_image_volume_cache_entries < min_stage1_cache_entries:
        cfg["pipeline_defaults"]["image_volume_cache_entries"] = min_stage1_cache_entries
        cfg["runtime_cache"]["image_volume_cache_entries"] = min_stage1_cache_entries
    for recipe in stage1_recipes:
        recipe["stage1_cache_entries"] = int(cfg["pipeline_defaults"]["stage1_cache_entries"])
        recipe["image_volume_cache_entries"] = int(cfg["pipeline_defaults"]["image_volume_cache_entries"])
    match_kwargs = _match_distance_kwargs(cfg)
    _append_jsonl(
        progress_path,
        {
            "event": "preflight_start",
            "optimization_mode": mode,
            "stage1_recipe_count": int(len(stage1_recipes)),
            "requested_preflight_image_count": requested_preflight_image_count,
            "preflight_image_count": int(len(val_pairs)),
            "labeled_preflight_image_count": labeled_preflight_image_count,
            "empty_gt_preflight_image_count": empty_gt_preflight_image_count,
            "available_validation_image_count": int(len(all_val_pairs)),
            "dataset_name": cfg["dataset_name"],
        },
    )
    preflight_rows: list[dict[str, Any]] = []
    max_stage1_candidates_single = cfg["preflight"].get("max_stage1_candidates_single")
    preflight_candidate_cap = (
        int(math.ceil(float(max_stage1_candidates_single)))
        if max_stage1_candidates_single is not None
        else None
    )
    for recipe_index, recipe in enumerate(stage1_recipes, start=1):
        preflight_recipe = deepcopy(recipe)
        if preflight_candidate_cap is not None:
            preflight_recipe["max_candidates"] = preflight_candidate_cap
        preflight_cache_signature = stage1_cache_signature(preflight_recipe)
        image_rows: list[dict[str, Any]] = []
        early_stop_reasons: list[str] = []
        early_stop_progress: dict[str, Any] | None = None
        for image_path, label_path in val_pairs:
            image_row = dict(preflight_image(image_path, label_path, preflight_recipe, **match_kwargs))
            image_rows.append(image_row)
            guardrail_progress = _stage1_guardrail_progress(
                image_rows,
                total_preflight_images=len(val_pairs),
                total_labeled_preflight_images=labeled_preflight_image_count,
                preflight_cfg=cfg["preflight"],
            )
            if not guardrail_progress["can_still_pass"]:
                early_stop_reasons = list(guardrail_progress["definitive_failure_reasons"])
                early_stop_progress = guardrail_progress
                break
        mean_candidates = float(sum(row["n_candidates"] for row in image_rows) / max(len(image_rows), 1))
        mean_labels = float(sum(row["n_labels"] for row in image_rows) / max(len(image_rows), 1))
        mean_recall = _mean_labeled_stage1_metric(image_rows, "recall")
        mean_precision = _mean_labeled_stage1_metric(image_rows, "precision")
        mean_f1 = _mean_labeled_stage1_metric(image_rows, "f1")
        max_candidates = int(max((row["n_candidates"] for row in image_rows), default=0))
        mean_candidate_ratio = _mean_per_image_candidate_ratio(image_rows)
        total_tp = int(sum(row["tp"] for row in image_rows))
        total_fp = int(sum(row["fp"] for row in image_rows))
        total_fn = int(sum(row["fn"] for row in image_rows))
        stopped_early = bool(early_stop_reasons)
        if stopped_early:
            failure_reasons = early_stop_reasons
        else:
            failure_reasons = _stage1_failure_reasons(
                mean_recall=mean_recall,
                mean_candidates=mean_candidates,
                max_candidates=max_candidates,
                mean_candidate_ratio=mean_candidate_ratio,
                preflight_cfg=cfg["preflight"],
            )
        passed = not failure_reasons
        processed_image_count = int(len(image_rows))
        total_preflight_image_count = int(len(val_pairs))
        processed_labeled_image_count = int(sum(1 for row in image_rows if int(row.get("n_labels", 0)) > 0))
        processed_empty_gt_image_count = int(processed_image_count - processed_labeled_image_count)
        early_stop_reason = _preflight_pass_label(early_stop_reasons) if stopped_early else None
        preflight_rows.append(
            {
                "recipe_id": recipe["recipe_id"],
                "stage1_key": _stage1_screen_key(recipe),
                "stage1_preflight_cache_signature": preflight_cache_signature,
                "stage2_recipe_count": len(stage2_recipe_bank(cfg, [recipe])),
                "mean_stage1_recall": mean_recall,
                "mean_stage1_precision": mean_precision,
                "mean_stage1_f1": mean_f1,
                "total_stage1_tp": total_tp,
                "total_stage1_fp": total_fp,
                "total_stage1_fn": total_fn,
                "mean_stage1_candidates": mean_candidates,
                "mean_stage1_label_count": mean_labels,
                "mean_stage1_candidate_ratio": mean_candidate_ratio,
                "max_stage1_candidates": max_candidates,
                "passed": passed,
                "stage1_decision": _preflight_pass_label(failure_reasons),
                "stage1_stopped_early": stopped_early,
                "stage1_early_stop_reason": early_stop_reason,
                "stage1_preflight_images_processed": processed_image_count,
                "stage1_preflight_images_total": total_preflight_image_count,
                "stage1_labeled_preflight_images_processed": processed_labeled_image_count,
                "stage1_labeled_preflight_images_total": labeled_preflight_image_count,
                "stage1_empty_gt_preflight_images_processed": processed_empty_gt_image_count,
                "stage1_empty_gt_preflight_images_total": empty_gt_preflight_image_count,
                "stage1_maximum_possible_mean_recall": (
                    early_stop_progress["maximum_possible_mean_recall"] if early_stop_progress else None
                ),
                "stage1_minimum_possible_mean_candidates": (
                    early_stop_progress["minimum_possible_mean_candidates"] if early_stop_progress else None
                ),
                "stage1_minimum_possible_candidate_ratio": (
                    early_stop_progress["minimum_possible_candidate_ratio"] if early_stop_progress else None
                ),
                "recipe": deepcopy(recipe),
            }
        )
        pruned_stage1_cache_entries = _prune_stage1_cache_to_leaderboard(preflight_rows, cfg, val_pairs)
        _append_jsonl(
            progress_path,
            {
                "event": "preflight_recipe_complete",
                "recipe_index": int(recipe_index),
                "stage1_recipe_count": int(len(stage1_recipes)),
                "recipe_id": str(recipe["recipe_id"]),
                "mean_stage1_recall": mean_recall,
                "mean_stage1_precision": mean_precision,
                "mean_stage1_candidates": mean_candidates,
                "max_stage1_candidates": max_candidates,
                "passed": passed,
                "stage1_decision": _preflight_pass_label(failure_reasons),
                "stage1_stopped_early": stopped_early,
                "stage1_early_stop_reason": early_stop_reason,
                "stage1_preflight_images_processed": processed_image_count,
                "stage1_preflight_images_total": total_preflight_image_count,
                "stage1_labeled_preflight_images_processed": processed_labeled_image_count,
                "stage1_labeled_preflight_images_total": labeled_preflight_image_count,
                "stage1_empty_gt_preflight_images_processed": processed_empty_gt_image_count,
                "stage1_empty_gt_preflight_images_total": empty_gt_preflight_image_count,
                "stage1_candidate_cache_pruned_entries": int(pruned_stage1_cache_entries),
            },
        )
    _append_jsonl(
        progress_path,
        {
            "event": "preflight_complete",
            "stage1_recipe_count": int(len(stage1_recipes)),
        },
    )
    preflight_df = pd.DataFrame(preflight_rows)

    passed_df = preflight_df[preflight_df["passed"]].copy()
    if passed_df.empty:
        preflight_df = _apply_preflight_decision_columns(preflight_df, ranking_cfg=cfg["stage1_ranking"])
        preflight_diagnostics = _write_stage1_preflight_failure_outputs(
            run_dir,
            preflight_df=preflight_df,
            cfg=cfg,
            status="no_preflight_survivors",
            message="No recipes passed Stage 1 preflight.",
        )
        summary_path = _write_terminal_model_outputs(
            run_dir,
            cfg=cfg,
            mode=mode,
            status="no_preflight_survivors",
            message="No recipes passed Stage 1 preflight.",
            split_summary=split_summary,
            profile=profile,
            optimizer_plan_payload=plan,
            requested_preflight_image_count=requested_preflight_image_count,
            preflight_image_count=len(val_pairs),
            available_validation_image_count=len(all_val_pairs),
            extra_output_files=preflight_diagnostics["output_files"],
            diagnostics=preflight_diagnostics["summary"],
        )
        _remove_optimizer_progress_files(run_dir)
        return summary_path

    shortlist, selected_stage1_keys = _stage2_shortlist_from_passed(
        passed_df,
        top_k=int(cfg["optimizer"]["shortlist_top_k"]),
        ranking_cfg=cfg["stage1_ranking"],
    )
    preflight_df = _apply_preflight_decision_columns(
        preflight_df,
        shortlist_ids=set(shortlist["recipe_id"].astype(str)),
        stage2_ids=set(shortlist["recipe_id"].astype(str)),
        ranking_cfg=cfg["stage1_ranking"],
    )
    preflight_df["stage1_selected_for_stage2"] = preflight_df["stage1_key"].astype(str).isin(set(selected_stage1_keys))
    stage1_shortlist = preflight_df[preflight_df["shortlisted_for_stage2"]].copy()
    stage1_shortlist = stage1_shortlist.sort_values("stage1_rank_passed")
    _prune_stage1_cache_to_leaderboard(preflight_rows, cfg, val_pairs)
    promoted_stage1_cache_entries = _promote_shortlisted_stage1_caches(
        stage1_shortlist,
        preflight_candidate_cap=preflight_candidate_cap,
        val_pairs=val_pairs,
    )
    full_val_stage1_df = _evaluate_full_val_stage1(stage1_shortlist, cfg, all_val_pairs)
    if not full_val_stage1_df.empty:
        full_val_columns = ["recipe_id"] + [col for col in full_val_stage1_df.columns if col.startswith("stage1_full_val_")]
        preflight_df = preflight_df.merge(full_val_stage1_df[full_val_columns], on="recipe_id", how="left")
        stage1_shortlist = preflight_df[preflight_df["shortlisted_for_stage2"]].copy()
        stage1_shortlist = stage1_shortlist.sort_values("stage1_rank_passed")
    promoted_full_val_image_volume_entries = _promote_shortlisted_stage1_image_volumes(stage1_shortlist, val_pairs=all_val_pairs)
    stage1_shortlist = _annotate_stage1_train_label_status(stage1_shortlist, cfg)
    shortlist = _expand_stage2_shortlist(stage1_shortlist, cfg)

    stage2_progress_path = run_dir / "stage2_progress.jsonl"
    if stage2_progress_path.exists():
        stage2_progress_path.unlink()
    _append_jsonl(
        stage2_progress_path,
        {
            "event": "stage2_start",
            "stage2_recipe_count": int(len(shortlist)),
            "shortlisted_stage1_count": int(stage1_shortlist["stage1_key"].nunique()) if "stage1_key" in stage1_shortlist else 0,
            "promoted_stage1_candidate_cache_entries": int(promoted_stage1_cache_entries["candidate_entries"]),
            "promoted_preflight_image_volume_cache_entries": int(promoted_stage1_cache_entries["image_volume_entries"]),
            "promoted_full_val_image_volume_cache_entries": int(promoted_full_val_image_volume_entries),
        },
    )

    stage2_rows = []
    for stage2_index, (_, row) in enumerate(shortlist.iterrows(), start=1):
        recipe = row["recipe"]
        _append_jsonl(
            stage2_progress_path,
            {
                "event": "stage2_recipe_start",
                "stage2_index": int(stage2_index),
                "stage2_recipe_count": int(len(shortlist)),
                "recipe_id": str(recipe["recipe_id"]),
                "stage1_recipe_id": str(row.get("stage1_recipe_id", "")),
                "feature_pack_name": recipe.get("feature_pack_name"),
            },
        )
        try:
            result = _evaluate_stage2_recipe(recipe, cfg, run_dir / "stage2")
        except ValueError as exc:
            message = str(exc)
            no_training_candidates = "generated no training candidates" in message
            no_true_positive = "generated no true-positive training candidates" in message
            one_class_error = "number of classes has to be greater than one" in message or "one class" in message.lower()
            if not no_training_candidates and not no_true_positive and not one_class_error:
                raise
            if no_training_candidates:
                status = "skipped_no_training_candidates"
                reason = "no_training_candidates"
            elif no_true_positive:
                status = "skipped_no_true_positive_training_candidates"
                reason = "no_true_positive_training_candidates"
            else:
                status = "skipped_one_class_training_labels"
                reason = "one_class_training_labels"
            result = {
                "recipe_id": str(recipe["recipe_id"]),
                "model_path": "",
                "stage2_status": status,
                "stage2_error": message,
                "val_f1": 0.0,
                "val_precision": 0.0,
                "val_recall": 0.0,
                "val_f1_mean_image": 0.0,
                "val_precision_mean_image": 0.0,
                "val_recall_mean_image": 0.0,
                "val_f1_pooled": 0.0,
                "val_precision_pooled": 0.0,
                "val_recall_pooled": 0.0,
                "recipe": deepcopy(recipe),
            }
            _append_jsonl(
                stage2_progress_path,
                {
                    "event": "stage2_recipe_skipped",
                    "stage2_index": int(stage2_index),
                    "stage2_recipe_count": int(len(shortlist)),
                    "recipe_id": str(recipe["recipe_id"]),
                    "reason": reason,
                    "error": message,
                },
            )
        stage2_rows.append(result)
        _append_jsonl(
            stage2_progress_path,
            {
                "event": "stage2_recipe_complete",
                "stage2_index": int(stage2_index),
                "stage2_recipe_count": int(len(shortlist)),
                "recipe_id": str(recipe["recipe_id"]),
                "val_f1": float(result.get("val_f1", 0.0)),
                "val_precision": float(result.get("val_precision", 0.0)),
                "val_recall": float(result.get("val_recall", 0.0)),
                "val_f1_pooled": float(result.get("val_f1_pooled", 0.0)),
            },
        )
    _append_jsonl(
        stage2_progress_path,
        {
            "event": "stage2_complete",
            "stage2_recipe_count": int(len(stage2_rows)),
        },
    )
    stage2_df = pd.DataFrame(stage2_rows).merge(shortlist.drop(columns=["recipe"]), on="recipe_id", how="left")
    invalid_stage2_statuses = {
        "skipped_no_training_candidates",
        "skipped_one_class_training_labels",
        "skipped_no_true_positive_training_candidates",
    }
    if "stage2_status" in stage2_df.columns:
        valid_stage2_df = stage2_df[~stage2_df["stage2_status"].isin(invalid_stage2_statuses)].copy()
    else:
        valid_stage2_df = stage2_df.copy()
    if valid_stage2_df.empty:
        status_values = set(str(x) for x in stage2_df.get("stage2_status", pd.Series(dtype=str)).dropna().unique())
        terminal_status = (
            "no_true_positive_stage1_candidates"
            if status_values == {"skipped_no_true_positive_training_candidates"}
            else "no_valid_stage2_recipes"
        )
        summary_path = _write_terminal_model_outputs(
            run_dir,
            cfg=cfg,
            mode=mode,
            status=terminal_status,
            message=(
                "No valid Stage 2 recipes were trainable because the shortlisted Stage 1 recipes "
                "did not provide two-class training labels."
            ),
            split_summary=split_summary,
            profile=profile,
            optimizer_plan_payload=plan,
            requested_preflight_image_count=requested_preflight_image_count,
            preflight_image_count=len(val_pairs),
            available_validation_image_count=len(all_val_pairs),
        )
        shutil.rmtree(run_dir / "stage2", ignore_errors=True)
        _remove_optimizer_progress_files(run_dir)
        return summary_path
    stage2_df = _add_stage2_selection_columns(valid_stage2_df)
    tolerance = _stage2_f1_tolerance(cfg)
    finalists = stage2_df[stage2_df["stage2_f1_loss"] <= tolerance].copy()
    if finalists.empty:
        finalists = stage2_df.copy()
    finalists = finalists.sort_values(_FINALIST_SORT_COLUMNS, ascending=_FINALIST_SORT_ASCENDING)
    winner = finalists.iloc[0]

    final_model_path = run_dir / "model.joblib"
    shutil.copy2(Path(str(winner["model_path"])), final_model_path)
    stage1_guardrails = _stage1_guardrail_summary(
        cfg,
        requested_preflight_image_count=requested_preflight_image_count,
        preflight_image_count=len(val_pairs),
        available_validation_image_count=len(all_val_pairs),
    )
    stage1_ranking = _stage1_ranking_summary(cfg)
    stage1_shortlist_summary = _stage1_shortlist_records(stage1_shortlist)
    stage2_winner = _stage2_winner_record(winner)

    lines = _summary_markdown_lines(
        cfg=cfg,
        mode=mode,
        split_summary=split_summary,
        profile=profile,
        final_model_path=final_model_path,
        stage1_guardrails=stage1_guardrails,
        stage1_ranking=stage1_ranking,
        stage1_shortlist_records=stage1_shortlist_summary,
        stage2_winner=stage2_winner,
    )
    model_summary_path = run_dir / "model_summary.md"
    model_summary_path.write_text("\n".join(lines) + "\n")
    _write_model_config(
        run_dir / "model_config.json",
        cfg=cfg,
        mode=mode,
        final_model_path=final_model_path,
        winner=winner,
        finalists=finalists,
        split_summary=split_summary,
        profile=profile,
        optimizer_plan_payload=plan,
        stage1_guardrails=stage1_guardrails,
        stage1_ranking=stage1_ranking,
        stage1_shortlist_records=stage1_shortlist_summary,
    )
    shutil.rmtree(run_dir / "stage2", ignore_errors=True)
    _remove_optimizer_progress_files(run_dir)
    return model_summary_path
