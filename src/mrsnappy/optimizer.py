from __future__ import annotations

import json
import math
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_features import (
    candidate_feature_columns,
    candidate_feature_rows,
    model_type_and_svm_flag,
    write_candidate_features_csv,
    write_candidate_features_manifest,
)
from .config import ensure_dir, load_config, stage1_recipe_bank, stage2_recipe_bank
from .pipeline import (
    clear_pipeline_caches,
    evaluate_predictions,
    pairwise_distances,
    preflight_image,
    predict_split,
    train_native_model,
    write_json,
)
from .io import read_points_csv, read_volume, split_pairs
from .model import iter_svm_param_grid


_PREFLIGHT_SORT_COLUMNS = [
    "preflight_utility",
    "mean_stage1_recall",
    "mean_stage1_precision",
    "processing_cost_rank",
    "recipe_id",
]
_PREFLIGHT_SORT_ASCENDING = [False, False, False, True, True]
_FINALIST_SORT_COLUMNS = ["val_f1", "preflight_utility", "val_precision", "mean_stage1_precision"]
_STAGE1_SCREEN_FIELDS = (
    "xy_spacing",
    "z_spacing",
    "preproc_enabled",
    "preproc_method",
    "preproc_sigma",
    "norm_enabled",
    "norm_method",
    "norm_param1",
    "norm_param2",
    "norm_param3",
    "background_enabled",
    "background_method",
    "background_mode",
    "background_scale",
    "background_param",
    "background_clip",
    "maxima_method",
    "maxima_neighborhood",
    "sigma_value",
    "threshold_value",
    "h_max_sigma_multiplier",
    "h_max_sigma_mode",
    "log_scale_normalize",
)


def _preflight_utility(mean_recall: float, mean_precision: float, mean_candidates: float, mean_labels: float) -> float:
    density = max(mean_candidates / max(mean_labels, 1.0), 1.0)
    return float(mean_recall + 0.05 * mean_precision - 0.08 * math.log1p(density))


def _fit_cost_rank(recipe: dict[str, Any]) -> int:
    fit_method = str(recipe.get("fit_method", "")).strip().lower()
    if fit_method == "2d (xy) + 1d (z) gaussian":
        return 0
    if fit_method == "3d gaussian":
        return 1
    if fit_method == "distorted 3d gaussian":
        return 2
    return 3


def _processing_cost_rank(recipe: dict[str, Any]) -> float:
    """Prefer cheaper Stage 1 processing when preflight quality is otherwise tied."""
    smoothing_cost = float(recipe.get("preproc_sigma", 0.0) or 0.0) if recipe.get("preproc_enabled", True) else 0.0
    if not recipe.get("background_enabled", False):
        return smoothing_cost

    method = str(recipe.get("background_method", "none")).strip().lower()
    radius = float(recipe.get("background_param", 0.0) or 0.0)
    if method in {"rolling_ball_3d", "rolling_ball_3d_exact", "rolling-ball-3d-exact", "rolling ball 3d exact"}:
        base = 1000.0
    elif method in {"rolling_ball_2d", "rolling-ball-2d", "rolling ball 2d"}:
        base = 500.0
    elif method == "slice_opening_2d":
        base = 100.0
    elif method in {"rolling_box_3d", "morph_opening_3d_box"}:
        base = 10.0
    elif method == "gaussian":
        base = 10.0
    else:
        base = 50.0
    return base + radius + 0.01 * smoothing_cost


def _stage1_screen_key(recipe: dict[str, Any]) -> str:
    payload = {key: recipe.get(key) for key in _STAGE1_SCREEN_FIELDS}
    return json.dumps(payload, sort_keys=True, default=str)


def _stage2_shortlist_from_passed(passed_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, list[str]]:
    passed_df = _ensure_preflight_sort_columns(passed_df)
    sorted_passed = passed_df.sort_values(_PREFLIGHT_SORT_COLUMNS, ascending=_PREFLIGHT_SORT_ASCENDING)
    selected_keys: list[str] = []
    selected_key_set: set[str] = set()
    for _, row in sorted_passed.iterrows():
        key = str(row["stage1_key"])
        if key in selected_key_set:
            continue
        selected_keys.append(key)
        selected_key_set.add(key)
        if len(selected_keys) >= int(top_k):
            break
    shortlist = sorted_passed[sorted_passed["stage1_key"].astype(str).isin(selected_key_set)].copy()
    return shortlist, selected_keys


def _ensure_preflight_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "processing_cost_rank" not in out:
        out["processing_cost_rank"] = 0.0
    return out


def _compact_stage1_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "recipe_id",
        "stage1_recipe_id",
        "stage1_dedup_key",
        "deduplicated_duplicate_count",
        *_STAGE1_SCREEN_FIELDS,
    ]
    return {key: _json_scalar(recipe.get(key)) for key in keys if key in recipe}


def _write_stage1_recipes_csv(path: Path, stage1_recipes: list[dict[str, Any]]) -> None:
    rows = [_compact_stage1_recipe(recipe) for recipe in stage1_recipes]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_stage2_recipes_csv(path: Path, stage2_df: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    if stage2_df.empty:
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    for _, row in stage2_df.iterrows():
        recipe = dict(row["recipe"])
        rows.append(
            {
                "stage2_id": row["recipe_id"],
                "stage1_id": row.get("stage1_recipe_id"),
                "feature_pack_name": recipe.get("feature_pack_name"),
                "selected_features": ";".join(str(item) for item in recipe.get("selected_features", [])),
                "fit_variant_id": recipe.get("fit_variant_id"),
                "fit_method": recipe.get("fit_method"),
                "fit_window": recipe.get("fit_window"),
                "svm_kernel": row.get("kernel"),
                "svm_C": row.get("C"),
                "svm_gamma": row.get("gamma"),
                "svm_degree": row.get("degree"),
                "svm_standardize": row.get("standardize"),
                "svm_class_weight_mode": row.get("class_weight_mode"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _expand_stage2_shortlist(
    stage1_shortlist: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, stage1_row in stage1_shortlist.iterrows():
        stage2_recipes = stage2_recipe_bank(cfg, [deepcopy(stage1_row["recipe"])])
        for recipe in stage2_recipes:
            row = stage1_row.drop(labels=["recipe"]).to_dict()
            row["stage1_recipe_id"] = stage1_row["recipe_id"]
            row["recipe_id"] = recipe["recipe_id"]
            row["feature_pack_name"] = recipe.get("feature_pack_name")
            row["fit_variant_id"] = recipe.get("fit_variant_id")
            row["fit_cost_rank"] = _fit_cost_rank(recipe)
            row["feature_count"] = len(recipe.get("selected_features") or [])
            row["recipe"] = deepcopy(recipe)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["stage1_rank_passed", "fit_cost_rank", "feature_count", "recipe_id"],
        ascending=[True, True, True, True],
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
        "dataset_profile": profile,
        "stage1_recipe_bank_entries": int(len(stage1_recipes)),
        "unique_stage1_preflight_configs": int(len(stage1_recipes)),
        "stage2_recipe_entries_per_stage1_min": int(min(stage2_counts, default=0)),
        "stage2_recipe_entries_per_stage1_max": int(max(stage2_counts, default=0)),
        "shortlist_top_k": shortlist_top_k,
        "max_stage2_recipe_entries_after_shortlist": max_stage2_after_shortlist,
        "svm_param_grid_entries_per_stage2_recipe": int(svm_grid_entries),
        "max_svm_fits_after_shortlist": int(max_stage2_after_shortlist * svm_grid_entries),
        "svm_selection": "fit each SVM configuration on train/ and select by validation performance on all val/ images",
        "safety_caps": {
            "max_stage1_preflight_configs": cfg["optimizer"].get("max_stage1_preflight_configs"),
            "max_stage2_recipes_after_shortlist": cfg["optimizer"].get("max_stage2_recipes_after_shortlist"),
        },
        "execution_order": [
            "evaluate unique Stage 1 candidate-generation configurations only",
            "rank passing Stage 1 configurations by preflight utility, recall, precision, deterministic id",
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
        cfg = _apply_profile_guidance(cfg, profile)
    else:
        profile = {"enabled": False, "reason": "dataset_root is not set"}
    explicit_recipes = bool(cfg.get("recipes"))
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
    min_recall = float(preflight_cfg["min_stage1_recall"])
    max_mean_candidates = float(preflight_cfg["max_stage1_candidates_mean"])
    max_single_candidates = int(preflight_cfg["max_stage1_candidates_single"])
    ratio_cap = preflight_cfg.get("max_stage1_candidates_per_label_mean")
    if float(mean_recall) < min_recall:
        reasons.append(f"recall {float(mean_recall):.4f} < minimum {min_recall:.4f}")
    if float(mean_candidates) > max_mean_candidates:
        reasons.append(f"mean candidates {float(mean_candidates):.1f} > maximum {max_mean_candidates:.1f}")
    if int(max_candidates) > max_single_candidates:
        reasons.append(f"single-image candidates {int(max_candidates)} > maximum {max_single_candidates}")
    if ratio_cap is not None and float(mean_candidate_ratio) > float(ratio_cap):
        reasons.append(f"candidate/label ratio {float(mean_candidate_ratio):.2f} > maximum {float(ratio_cap):.2f}")
    return reasons


def _preflight_pass_label(reasons: list[str]) -> str:
    return "passed all Stage 1 guardrails" if not reasons else "; ".join(reasons)


def _apply_preflight_decision_columns(
    preflight_df: pd.DataFrame,
    shortlist_ids: set[str] | None = None,
    stage2_ids: set[str] | None = None,
) -> pd.DataFrame:
    out = _ensure_preflight_sort_columns(preflight_df)
    shortlist_ids = shortlist_ids or set()
    stage2_ids = stage2_ids or set()
    if out.empty:
        return out

    sorted_all = out.sort_values(
        ["passed", *_PREFLIGHT_SORT_COLUMNS],
        ascending=[False, *_PREFLIGHT_SORT_ASCENDING],
    )
    all_rank = pd.Series(range(1, len(sorted_all) + 1), index=sorted_all.index, dtype="int64")
    out["stage1_rank_all"] = all_rank
    out["stage1_rank_passed"] = pd.NA
    passed_sorted = out[out["passed"]].sort_values(_PREFLIGHT_SORT_COLUMNS, ascending=_PREFLIGHT_SORT_ASCENDING)
    if not passed_sorted.empty:
        passed_rank = pd.Series(range(1, len(passed_sorted) + 1), index=passed_sorted.index, dtype="int64")
        out.loc[passed_sorted.index, "stage1_rank_passed"] = passed_rank
    out["shortlisted_for_stage2"] = out["recipe_id"].astype(str).isin(shortlist_ids)
    out["selected_for_stage2"] = out["recipe_id"].astype(str).isin(stage2_ids)
    return out.sort_values(["stage1_rank_all", "recipe_id"], ascending=[True, True])


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


def _table_records(df: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.head(int(limit)).iterrows():
        rows.append({col: _json_scalar(row[col]) for col in columns if col in row.index})
    return rows


def _failure_reason_counts(preflight_df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "stage1_decision" not in preflight_df:
        return counts
    for text in preflight_df.loc[~preflight_df["passed"], "stage1_decision"].astype(str):
        for reason in text.split("; "):
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _compact_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "recipe_id",
        "maxima_method",
        "preproc_sigma",
        "background_enabled",
        "background_method",
        "background_param",
        "sigma_value",
        "threshold_value",
        "h_max_sigma_multiplier",
        "h_max_sigma_mode",
        "maxima_neighborhood",
        "fit_method",
        "fit_window",
        "feature_pack_name",
    ]
    return {key: _json_scalar(recipe.get(key)) for key in keys if key in recipe}


def _write_selection_decision_record(
    run_dir: Path,
    *,
    cfg: dict[str, Any],
    profile: dict[str, Any],
    preflight_df: pd.DataFrame,
    shortlist: pd.DataFrame,
    stage2_df: pd.DataFrame | None,
    finalists: pd.DataFrame | None,
    winner: pd.Series | None,
    timings: dict[str, float],
) -> None:
    preflight_cfg = cfg["preflight"]
    optimizer_cfg = cfg["optimizer"]
    top_stage1_cols = [
        "stage1_rank_passed",
        "recipe_id",
        "mean_stage1_recall",
        "mean_stage1_precision",
        "mean_stage1_candidates",
        "mean_stage1_candidate_ratio",
        "max_stage1_candidates",
        "stage2_recipe_count",
        "preflight_utility",
        "stage1_decision",
    ]
    stage2_cols = [
        "recipe_id",
        "val_f1",
        "val_precision",
        "val_recall",
        "selection_margin",
        "preflight_utility",
        "decision_threshold",
    ]

    winner_recipe: dict[str, Any] | None = None
    if winner is not None and "recipe" in winner.index:
        winner_recipe = _compact_recipe(dict(winner["recipe"]))

    decision = {
        "purpose": "Auditable SNAPpy optimizer decision record for Stage 1 screening and Stage 2 recipe selection.",
        "dataset_name": cfg["dataset_name"],
        "dataset_profile": profile,
        "stage1_policy": {
            "guardrails": {
                "min_stage1_recall": float(preflight_cfg["min_stage1_recall"]),
                "max_stage1_candidates_mean": float(preflight_cfg["max_stage1_candidates_mean"]),
                "max_stage1_candidates_single": int(preflight_cfg["max_stage1_candidates_single"]),
                "max_stage1_candidates_per_label_mean": _json_scalar(preflight_cfg.get("max_stage1_candidates_per_label_mean")),
                "auto_candidate_ratio_cap_enabled": bool(preflight_cfg.get("auto_candidate_ratio_cap_enabled", True)),
                "auto_candidate_ratio_caps": deepcopy(preflight_cfg.get("auto_candidate_ratio_caps", {})),
                "auto_candidate_ratio_low_contrast_multiplier": float(preflight_cfg.get("auto_candidate_ratio_low_contrast_multiplier", 1.25)),
                "auto_candidate_ratio_high_contrast_multiplier": float(preflight_cfg.get("auto_candidate_ratio_high_contrast_multiplier", 0.85)),
            },
            "utility_formula": "recall + 0.05 * precision - 0.08 * log1p(mean_candidates / max(mean_labels, 1))",
            "ranking_order": [
                "higher preflight_utility",
                "higher mean_stage1_recall",
                "higher mean_stage1_precision",
                "stage1 recipe id for deterministic ties",
            ],
            "shortlist_top_k": int(optimizer_cfg["shortlist_top_k"]),
            "shortlist_unit": "Stage 1 candidate-generation configurations; all Stage 2 feature packs are retained for selected Stage 1 configurations.",
        },
        "stage1_outcome": {
            "recipes_evaluated": int(len(preflight_df)),
            "recipes_passed": int(preflight_df["passed"].sum()) if not preflight_df.empty else 0,
            "recipes_shortlisted_for_stage2": int(len(shortlist)),
            "stage1_configs_shortlisted_for_stage2": int(shortlist["stage1_key"].nunique()) if "stage1_key" in shortlist else 0,
            "failure_reason_counts": _failure_reason_counts(preflight_df),
            "top_passed_recipes": _table_records(
                preflight_df[preflight_df["passed"]].sort_values("stage1_rank_passed"),
                top_stage1_cols,
                limit=10,
            ),
        },
        "stage2_policy": {
            "stage2_recipes_from_stage1_shortlist": int(len(shortlist)),
            "stage2_feature_packs": list(cfg.get("stage2_feature_packs", [])),
            "svm_selection": "For each SVM hyperparameter setting, fit on train/ candidates, evaluate on all val/ images, tune threshold on val/, and select by validation metrics.",
            "selection_margin": float(optimizer_cfg["selection_margin"]),
            "winner_ranking_order": [
                "within selection_margin of best validation F1",
                "higher val_f1",
                "higher preflight_utility",
                "higher val_precision",
                "higher mean_stage1_precision",
            ],
        },
        "runtime_caching": {
            "stage1_cache_enabled": bool(cfg.get("pipeline_defaults", {}).get("stage1_cache_enabled", cfg.get("stage1_cache_enabled", True))),
            "stage1_cache_entries": int(cfg.get("pipeline_defaults", {}).get("stage1_cache_entries", cfg.get("stage1_cache_entries", 128))),
            "preprocess_cache_entries": int(cfg.get("pipeline_defaults", {}).get("preprocess_cache_entries", 96)),
            "fit_cache_enabled": bool(cfg.get("pipeline_defaults", {}).get("fit_cache_enabled", cfg.get("fit_cache_enabled", True))),
            "fit_cache_entries": int(cfg.get("pipeline_defaults", {}).get("fit_cache_entries", cfg.get("fit_cache_entries", 512))),
            "notes": "Stage 1 preflight evaluates each unique candidate-generation configuration once. Processed/smoothed 3D volumes are cached by image and processing parameters. Stage 1 candidate coordinates/scores are cached separately by image and detector parameters. Fitted candidate feature tables are cached by image, Stage 1 parameters, pruning limits, and fit settings so feature-pack and SVM sweeps do not refit identical candidates.",
        },
        "stage2_outcome": {
            "stage2_recipes_evaluated": 0 if stage2_df is None else int(len(stage2_df)),
            "finalists_within_margin": 0 if finalists is None else int(len(finalists)),
            "stage2_results": [] if stage2_df is None else _table_records(stage2_df.sort_values("selection_margin"), stage2_cols, limit=10),
            "winner_recipe_id": None if winner is None else str(winner["recipe_id"]),
            "winner_recipe": winner_recipe,
        },
        "timings": {key: float(value) for key, value in timings.items()},
        "artifacts": {
            "stage1_recipes": "export_optimize_report/stage1_recipes.csv",
            "per_image_stage1": "export_optimize_report/stage1_by_image.csv",
            "stage1_summary": "export_optimize_report/stage1_summary.csv",
            "stage2_recipes": "export_optimize_report/stage2_recipes.csv",
            "stage2_summary": "export_optimize_report/stage2_summary.csv",
            "machine_readable_decision_record": "export_optimize_report/selection_decision.json",
            "human_readable_decision_record": "export_optimize_report/selection_decision.md",
        },
    }
    write_json(run_dir / "selection_decision.json", decision)

    fail_text = ", ".join(f"{count}x {reason}" for reason, count in list(decision["stage1_outcome"]["failure_reason_counts"].items())[:5])
    if not fail_text:
        fail_text = "none"
    winner_text = decision["stage2_outcome"]["winner_recipe_id"] or "none"
    ratio_cap = preflight_cfg.get("max_stage1_candidates_per_label_mean")
    ratio_cap_text = "disabled" if ratio_cap is None else f"{float(ratio_cap):.2f}"
    lines = [
        "# SNAPpy Optimizer Decision Record",
        "",
        "## Stage 1 Policy",
        f"- Guardrails: recall >= `{float(preflight_cfg['min_stage1_recall']):.4f}`, mean candidates <= `{float(preflight_cfg['max_stage1_candidates_mean']):.1f}`, single-image candidates <= `{int(preflight_cfg['max_stage1_candidates_single'])}`, candidate/label ratio <= `{ratio_cap_text}`.",
        "- Utility: `recall + 0.05 * precision - 0.08 * log1p(candidate/label density)`.",
        "- Ranking: utility, recall, precision, deterministic Stage 1 recipe id.",
        "",
        "## Stage 1 Outcome",
        f"- Evaluated `{decision['stage1_outcome']['recipes_evaluated']}` recipes; `{decision['stage1_outcome']['recipes_passed']}` passed; `{decision['stage1_outcome']['stage1_configs_shortlisted_for_stage2']}` Stage 1 configurations expanded to `{decision['stage1_outcome']['recipes_shortlisted_for_stage2']}` Stage 2 recipes.",
        f"- Main failure reasons: {fail_text}.",
        "",
        "## Stage 2 Outcome",
        f"- SVM selection: {decision['stage2_policy']['svm_selection']}",
        f"- Evaluated `{decision['stage2_outcome']['stage2_recipes_evaluated']}` shortlisted recipes; `{decision['stage2_outcome']['finalists_within_margin']}` were within the configured selection margin.",
        f"- Winner: `{winner_text}`.",
    ]
    (run_dir / "selection_decision.md").write_text("\n".join(lines) + "\n")


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


def _auto_candidate_ratio_cap(profile: dict[str, Any], preflight_cfg: dict[str, Any]) -> float:
    density = str(profile.get("density_regime", "moderate"))
    contrast = str(profile.get("contrast_regime", "moderate"))
    base_caps = preflight_cfg.get(
        "auto_candidate_ratio_caps",
        {"sparse": 130.0, "moderate": 100.0, "dense": 70.0},
    )
    cap = float(base_caps.get(density, 100.0))
    if contrast == "low":
        cap *= float(preflight_cfg.get("auto_candidate_ratio_low_contrast_multiplier", 1.25))
    elif contrast == "high":
        cap *= float(preflight_cfg.get("auto_candidate_ratio_high_contrast_multiplier", 0.85))
    return round(cap, 2)


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

    explicit_recipes = bool(guided.get("recipes"))
    apply_to_explicit = bool(guided.get("profiling", {}).get("apply_to_explicit_recipes", False))
    preflight_cfg = guided.get("preflight", {})
    if (
        bool(preflight_cfg.get("auto_candidate_ratio_cap_enabled", True))
        and (not explicit_recipes or apply_to_explicit)
        and preflight_cfg.get("max_stage1_candidates_per_label_mean") is None
    ):
        guided["preflight"]["max_stage1_candidates_per_label_mean"] = _auto_candidate_ratio_cap(
            profile,
            preflight_cfg,
        )

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
    negative_ratio = 24.0 if density == "sparse" else 32.0

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
        row["negative_to_positive_ratio"] = float(row.get("negative_to_positive_ratio", negative_ratio) or negative_ratio)
        out.append(row)
    return out


def _candidate_feature_rows_from_metas(
    metas: list[dict[str, Any]],
    *,
    selected_features: list[str],
    model: Any,
    decision_threshold: float,
    match_distance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in metas:
        image_id = Path(str(meta["image_path"])).stem
        coords = np.asarray(meta["coords"], dtype=np.float32)
        scores = np.asarray(meta["scores"], dtype=np.float32)
        gt = np.asarray(meta["gt"], dtype=np.float32)
        features = meta["features"].reset_index(drop=True)
        labels = np.asarray(meta["labels"], dtype=np.int32)
        _, svm_used = model_type_and_svm_flag(model)
        if len(coords):
            feature_matrix = features[selected_features].to_numpy(dtype=np.float32)
            if hasattr(model, "decision_function"):
                model_scores = np.asarray(model.decision_function(feature_matrix), dtype=np.float32)
            else:
                model_scores = np.asarray(model.predict(feature_matrix), dtype=np.float32)
        else:
            model_scores = np.empty((0,), dtype=np.float32)
        _, nearest_dist, nearest_ids = _nearest_gt_columns(coords, gt, match_distance)
        matched_ids: list[int | None] = []
        matched_coords = np.full((len(coords), 3), np.nan, dtype=np.float32)
        for idx in range(len(coords)):
            nearest_id = nearest_ids[idx] if idx < len(nearest_ids) else None
            distance = float(nearest_dist[idx]) if idx < len(nearest_dist) else np.nan
            matched_id = nearest_id if nearest_id is not None and distance <= float(match_distance) else None
            matched_ids.append(matched_id)
            if matched_id is not None and len(gt):
                matched_coords[idx] = gt[matched_id]
        rows.extend(
            candidate_feature_rows(
                image_id=image_id,
                coords=coords,
                maxima_scores=scores,
                features=features,
                model_scores=model_scores,
                decision_threshold=float(decision_threshold),
                selected_features=selected_features,
                svm_used=svm_used,
                labels=labels,
                matched_gt_ids=matched_ids,
                matched_gt_coords=matched_coords,
                nearest_gt_ids=nearest_ids,
                nearest_gt_distances=nearest_dist,
            )
        )
    return rows


def _evaluate_stage2_recipe(
    recipe: dict[str, Any],
    cfg: dict[str, Any],
    run_dir: Path,
    candidate_feature_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    trial_dir = ensure_dir(run_dir / recipe["recipe_id"])
    model_path = trial_dir / "model.joblib"
    trained = train_native_model(
        dataset_root=cfg["dataset_root"],
        recipe=recipe,
        svm_cfg=cfg["svm_sweep"],
        model_path=model_path,
        match_distance=float(cfg["match_distance"]),
    )
    if candidate_feature_cache is not None:
        candidate_feature_cache[recipe["recipe_id"]] = _candidate_feature_rows_from_metas(
            trained.validation_metas or [],
            selected_features=list(trained.selected_features),
            model=trained.model,
            decision_threshold=float(trained.decision_threshold),
            match_distance=float(cfg["match_distance"]),
        )
    val_preds, val_gts, val_seconds = predict_split(
        dataset_root=cfg["dataset_root"],
        split="val",
        recipe=recipe,
        model_path=model_path,
        output_root=trial_dir / "val_predictions",
        score_threshold=float(trained.decision_threshold),
    )
    val_metrics = evaluate_predictions(val_preds, val_gts, float(cfg["match_distance"]))
    result = {
        "recipe_id": recipe["recipe_id"],
        "model_path": str(model_path),
        "val_runtime_seconds": val_seconds,
        "decision_threshold": float(trained.decision_threshold),
        **trained.best_params,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "recipe": deepcopy(recipe),
    }
    write_json(trial_dir / "stage2_summary.json", result)
    return result


def _write_report_exports(
    run_dir: Path,
    *,
    cfg: dict[str, Any],
    stage1_recipes: list[dict[str, Any]],
    preflight_df: pd.DataFrame,
    preflight_image_df: pd.DataFrame,
    stage2_df: pd.DataFrame,
    winner: pd.Series | None,
) -> None:
    report_dir = ensure_dir(run_dir / "export_optimize_report")
    _write_stage1_recipes_csv(report_dir / "stage1_recipes.csv", stage1_recipes)
    _write_stage2_recipes_csv(report_dir / "stage2_recipes.csv", stage2_df)

    stage1_by_image = preflight_image_df.rename(
        columns={
            "recipe_id": "stage1_id",
            "image": "image_id",
            "n_candidates": "candidates_in_image",
            "n_labels": "gt_in_image",
        }
    ).copy()
    if not stage1_by_image.empty and not preflight_df.empty:
        decision_cols = preflight_df[
            ["recipe_id", "passed", "stage1_decision", "preflight_utility", "stage1_rank_passed", "shortlisted_for_stage2"]
        ].rename(
            columns={
                "recipe_id": "stage1_id",
                "passed": "guardrail_pass",
                "stage1_decision": "guardrail_reason",
                "preflight_utility": "preflight_score",
            }
        )
        stage1_by_image = stage1_by_image.merge(decision_cols, on="stage1_id", how="left")
    if "image_id" in stage1_by_image:
        stage1_by_image["image_id"] = stage1_by_image["image_id"].astype(str).map(lambda text: Path(text).stem)
    stage1_by_image.to_csv(report_dir / "stage1_by_image.csv", index=False)

    stage1_summary = preflight_df.rename(
        columns={
            "recipe_id": "stage1_id",
            "passed": "guardrail_pass",
            "stage1_decision": "guardrail_reason",
            "preflight_utility": "preflight_score",
        }
    ).copy()
    if "recipe" in stage1_summary:
        stage1_summary = stage1_summary.drop(columns=["recipe"])
    stage1_summary.to_csv(report_dir / "stage1_summary.csv", index=False)

    stage2_summary = stage2_df.copy()
    if not stage2_summary.empty:
        stage2_summary = stage2_summary.rename(columns={"recipe_id": "stage2_id", "stage1_recipe_id": "stage1_id"})
        stage2_summary["winner"] = False
        if winner is not None:
            stage2_summary.loc[stage2_summary["stage2_id"].astype(str) == str(winner["recipe_id"]), "winner"] = True
        if "recipe" in stage2_summary:
            stage2_summary = stage2_summary.drop(columns=["recipe"])
        if "model_path" in stage2_summary:
            stage2_summary = stage2_summary.drop(columns=["model_path"])
    stage2_summary.to_csv(report_dir / "stage2_summary.csv", index=False)

    for name in ("selection_decision.json", "selection_decision.md"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, report_dir / name)


def _nearest_gt_columns(coords: np.ndarray, gt: np.ndarray, match_distance: float) -> tuple[list[int | None], np.ndarray, list[int | None]]:
    if len(coords) == 0:
        return [], np.empty((0,), dtype=np.float32), []
    if len(gt) == 0:
        return [None] * len(coords), np.full((len(coords),), np.nan, dtype=np.float32), [None] * len(coords)
    d = pairwise_distances(coords, gt)
    nearest_idx = np.argmin(d, axis=1).astype(int)
    nearest_dist = d[np.arange(len(coords)), nearest_idx].astype(np.float32)
    matched_ids = [int(nearest_idx[idx]) if float(nearest_dist[idx]) <= float(match_distance) else None for idx in range(len(coords))]
    return matched_ids, nearest_dist, nearest_idx.tolist()


def _write_candidate_feature_export(
    run_dir: Path,
    *,
    cfg: dict[str, Any],
    winner: pd.Series,
    candidate_rows: list[dict[str, Any]],
) -> None:
    feature_dir = ensure_dir(run_dir / "export_candidate_features")
    recipe = dict(winner["recipe"])
    selected_features = list(recipe.get("selected_features") or [])
    winner_model_type = str(winner.get("model_type", "svm"))
    svm_used = winner_model_type != "stage1_pass_through"
    write_candidate_features_csv(
        feature_dir / "val_candidates.csv",
        candidate_rows,
        columns=candidate_feature_columns(selected_features, include_labels=True, include_ground_truth=True),
    )
    write_candidate_features_manifest(
        feature_dir / "candidate_features_manifest.json",
        {
            "split": "val",
            "stage2_id": str(winner["recipe_id"]),
            "stage1_id": str(winner.get("stage1_recipe_id", "")),
            "feature_pack_name": recipe.get("feature_pack_name"),
            "model_type": winner_model_type,
            "svm_used": bool(svm_used),
            "selected_features": selected_features,
            "maxima_score_definition": "Detector response value at the candidate maximum before Gaussian fitting and SVM classification.",
            "svm_score_definition": "Winning SVM decision_function score for the candidate feature vector when Stage 2 SVM classification is used; null for Stage 1 pass-through models.",
            "model_score_definition": "Score used for the model acceptance decision. For Stage 2 SVM models this matches svm_score; for Stage 1 pass-through models it is the pass-through model score.",
            "accepted_by_model_definition": "True when model_score is greater than the winning decision threshold.",
            "match_distance": float(cfg["match_distance"]),
            "feature_file": "val_candidates.csv",
        },
    )


def _remove_optional_root_decision_files(run_dir: Path) -> None:
    for name in ("selection_decision.json", "selection_decision.md"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def optimize_native_dataset(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    export_optimize_report: bool = False,
    export_candidate_features: bool = False,
) -> Path:
    clear_pipeline_caches()
    cfg = load_config(config_path)
    run_dir = ensure_dir(run_dir)
    timings: dict[str, float] = {}
    profile = _profile_dataset(cfg)
    cfg = _apply_profile_guidance(cfg, profile)
    write_json(run_dir / "dataset_profile.json", profile)
    explicit_recipes = bool(cfg.get("recipes"))
    stage1_recipes = stage1_recipe_bank(cfg)
    stage1_recipes = _apply_runtime_recipe_guidance(stage1_recipes, cfg, profile, explicit_recipes=explicit_recipes)
    plan = _optimizer_plan_from_materialized(cfg, profile, stage1_recipes)
    write_json(run_dir / "optimizer_plan.json", plan)
    _enforce_optimizer_plan_safety(plan, cfg)
    progress_path = run_dir / "preflight_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    _append_jsonl(
        progress_path,
        {
            "event": "preflight_start",
            "timestamp": time.time(),
            "stage1_recipe_count": int(len(stage1_recipes)),
            "preflight_image_count": int(cfg["preflight"]["stage1_n_val_images"]),
            "dataset_name": cfg["dataset_name"],
        },
    )

    val_pairs = split_pairs(cfg["dataset_root"], "val")
    val_pairs = val_pairs[: int(cfg["preflight"]["stage1_n_val_images"])]
    preflight_rows: list[dict[str, Any]] = []
    preflight_image_rows: list[dict[str, Any]] = []
    preflight_detection_cap = int(cfg["preflight"]["max_stage1_candidates_single"]) + 1
    t0 = time.time()
    for recipe_index, recipe in enumerate(stage1_recipes, start=1):
        recipe_t0 = time.time()
        preflight_recipe = deepcopy(recipe)
        preflight_recipe["max_candidates"] = preflight_detection_cap
        image_rows: list[dict[str, Any]] = []
        for image_path, label_path in val_pairs:
            image_row = preflight_image(image_path, label_path, preflight_recipe, float(cfg["match_distance"]))
            image_rows.append(image_row)
            preflight_image_rows.append(
                {
                    "recipe_id": recipe["recipe_id"],
                    "image": Path(image_path).name,
                    "label": Path(label_path).name,
                    **image_row,
                }
            )
        mean_candidates = float(sum(row["n_candidates"] for row in image_rows) / max(len(image_rows), 1))
        mean_labels = float(sum(row["n_labels"] for row in image_rows) / max(len(image_rows), 1))
        mean_recall = float(sum(row["recall"] for row in image_rows) / max(len(image_rows), 1))
        mean_precision = float(sum(row["precision"] for row in image_rows) / max(len(image_rows), 1))
        mean_f1 = float(sum(row["f1"] for row in image_rows) / max(len(image_rows), 1))
        max_candidates = int(max((row["n_candidates"] for row in image_rows), default=0))
        mean_candidate_ratio = float(mean_candidates / max(mean_labels, 1.0))
        total_tp = int(sum(row["tp"] for row in image_rows))
        total_fp = int(sum(row["fp"] for row in image_rows))
        total_fn = int(sum(row["fn"] for row in image_rows))
        failure_reasons = _stage1_failure_reasons(
            mean_recall=mean_recall,
            mean_candidates=mean_candidates,
            max_candidates=max_candidates,
            mean_candidate_ratio=mean_candidate_ratio,
            preflight_cfg=cfg["preflight"],
        )
        passed = not failure_reasons
        preflight_rows.append(
            {
                "recipe_id": recipe["recipe_id"],
                "stage1_key": _stage1_screen_key(recipe),
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
                "preflight_utility": _preflight_utility(mean_recall, mean_precision, mean_candidates, mean_labels),
                "processing_cost_rank": _processing_cost_rank(recipe),
                "fit_cost_rank": _fit_cost_rank(recipe),
                "feature_count": len(recipe.get("selected_features") or []),
                "passed": passed,
                "stage1_decision": _preflight_pass_label(failure_reasons),
                "recipe": deepcopy(recipe),
            }
        )
        _append_jsonl(
            progress_path,
            {
                "event": "preflight_recipe_complete",
                "timestamp": time.time(),
                "recipe_index": int(recipe_index),
                "stage1_recipe_count": int(len(stage1_recipes)),
                "recipe_id": str(recipe["recipe_id"]),
                "elapsed_total_seconds": float(time.time() - t0),
                "elapsed_recipe_seconds": float(time.time() - recipe_t0),
                "mean_stage1_recall": mean_recall,
                "mean_stage1_precision": mean_precision,
                "mean_stage1_candidates": mean_candidates,
                "max_stage1_candidates": max_candidates,
                "passed": passed,
                "stage1_decision": _preflight_pass_label(failure_reasons),
            },
        )
    timings["preflight_seconds"] = time.time() - t0
    _append_jsonl(
        progress_path,
        {
            "event": "preflight_complete",
            "timestamp": time.time(),
            "elapsed_total_seconds": float(timings["preflight_seconds"]),
            "stage1_recipe_count": int(len(stage1_recipes)),
        },
    )
    preflight_df = pd.DataFrame(preflight_rows)
    preflight_image_df = pd.DataFrame(preflight_image_rows)

    passed_df = preflight_df[preflight_df["passed"]].copy()
    if passed_df.empty:
        preflight_df = _apply_preflight_decision_columns(preflight_df)
        _write_selection_decision_record(
            run_dir,
            cfg=cfg,
            profile=profile,
            preflight_df=preflight_df,
            shortlist=pd.DataFrame(),
            stage2_df=None,
            finalists=None,
            winner=None,
            timings=timings,
        )
        summary = {
            "dataset_name": cfg["dataset_name"],
            "status": "no_preflight_survivors",
            "timings": timings,
        }
        write_json(run_dir / "summary.json", summary)
        (run_dir / "summary.md").write_text(
            "# SNAPpy Native Optimizer\n\nNo recipes passed preflight.\n",
        )
        if export_optimize_report:
            _write_report_exports(
                run_dir,
                cfg=cfg,
                stage1_recipes=stage1_recipes,
                preflight_df=preflight_df,
                preflight_image_df=preflight_image_df,
                stage2_df=pd.DataFrame(),
                winner=None,
            )
        _remove_optional_root_decision_files(run_dir)
        return run_dir / "summary.md"

    shortlist, selected_stage1_keys = _stage2_shortlist_from_passed(
        passed_df,
        top_k=int(cfg["optimizer"]["shortlist_top_k"]),
    )
    preflight_df = _apply_preflight_decision_columns(
        preflight_df,
        shortlist_ids=set(shortlist["recipe_id"].astype(str)),
        stage2_ids=set(shortlist["recipe_id"].astype(str)),
    )
    preflight_df["stage1_selected_for_stage2"] = preflight_df["stage1_key"].astype(str).isin(set(selected_stage1_keys))
    stage1_shortlist = preflight_df[preflight_df["shortlisted_for_stage2"]].copy()
    stage1_shortlist = stage1_shortlist.sort_values("stage1_rank_passed")
    shortlist = _expand_stage2_shortlist(stage1_shortlist, cfg)

    stage2_progress_path = run_dir / "stage2_progress.jsonl"
    if stage2_progress_path.exists():
        stage2_progress_path.unlink()
    _append_jsonl(
        stage2_progress_path,
        {
            "event": "stage2_start",
            "timestamp": time.time(),
            "stage2_recipe_count": int(len(shortlist)),
            "shortlisted_stage1_count": int(stage1_shortlist["stage1_key"].nunique()) if "stage1_key" in stage1_shortlist else 0,
        },
    )

    candidate_feature_cache: dict[str, list[dict[str, Any]]] | None = {} if export_candidate_features else None
    t0 = time.time()
    stage2_rows = []
    for stage2_index, (_, row) in enumerate(shortlist.iterrows(), start=1):
        recipe_t0 = time.time()
        recipe = row["recipe"]
        _append_jsonl(
            stage2_progress_path,
            {
                "event": "stage2_recipe_start",
                "timestamp": time.time(),
                "stage2_index": int(stage2_index),
                "stage2_recipe_count": int(len(shortlist)),
                "recipe_id": str(recipe["recipe_id"]),
                "stage1_recipe_id": str(row.get("stage1_recipe_id", "")),
                "feature_pack_name": recipe.get("feature_pack_name"),
            },
        )
        try:
            result = _evaluate_stage2_recipe(recipe, cfg, run_dir / "stage2", candidate_feature_cache=candidate_feature_cache)
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
                "recipe": deepcopy(recipe),
            }
            _append_jsonl(
                stage2_progress_path,
                {
                    "event": "stage2_recipe_skipped",
                    "timestamp": time.time(),
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
                "timestamp": time.time(),
                "stage2_index": int(stage2_index),
                "stage2_recipe_count": int(len(shortlist)),
                "recipe_id": str(recipe["recipe_id"]),
                "elapsed_total_seconds": float(time.time() - t0),
                "elapsed_recipe_seconds": float(time.time() - recipe_t0),
                "val_f1": float(result.get("val_f1", 0.0)),
                "val_precision": float(result.get("val_precision", 0.0)),
                "val_recall": float(result.get("val_recall", 0.0)),
            },
        )
    timings["stage2_seconds"] = time.time() - t0
    _append_jsonl(
        stage2_progress_path,
        {
            "event": "stage2_complete",
            "timestamp": time.time(),
            "elapsed_total_seconds": float(timings["stage2_seconds"]),
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
        timings["stage2_seconds"] = time.time() - t0
        status_values = set(str(x) for x in stage2_df.get("stage2_status", pd.Series(dtype=str)).dropna().unique())
        terminal_status = (
            "no_true_positive_stage1_candidates"
            if status_values == {"skipped_no_true_positive_training_candidates"}
            else "no_valid_stage2_recipes"
        )
        summary = {
            "dataset_name": cfg["dataset_name"],
            "status": terminal_status,
            "timings": timings,
        }
        write_json(run_dir / "summary.json", summary)
        (run_dir / "summary.md").write_text(
            "# SNAPpy Native Optimizer\n\n"
            "No valid Stage 2 recipes were trainable because the shortlisted Stage 1 recipes "
            "did not provide two-class training labels.\n"
        )
        if export_optimize_report:
            _write_report_exports(
                run_dir,
                cfg=cfg,
                stage1_recipes=stage1_recipes,
                preflight_df=preflight_df,
                preflight_image_df=preflight_image_df,
                stage2_df=stage2_df,
                winner=None,
            )
        return run_dir / "summary.md"
    stage2_df = valid_stage2_df
    stage2_df["selection_margin"] = float(stage2_df["val_f1"].max()) - stage2_df["val_f1"]
    finalists = stage2_df[stage2_df["selection_margin"] <= float(cfg["optimizer"]["selection_margin"])].copy()
    if finalists.empty:
        finalists = stage2_df.copy()
    finalists = finalists.sort_values(_FINALIST_SORT_COLUMNS, ascending=False)
    winner = finalists.iloc[0]

    t0 = time.time()
    final_model_path = run_dir / "model.joblib"
    shutil.copy2(Path(str(winner["model_path"])), final_model_path)
    timings["final_seconds"] = time.time() - t0

    summary = {
        "dataset_name": cfg["dataset_name"],
        "dataset_profile": profile,
        "winner": winner["recipe_id"],
        "model_path": str(final_model_path),
        "winner_recipe": winner["recipe"],
        "val_f1": float(winner["val_f1"]),
        "decision_threshold": float(winner.get("decision_threshold", 0.0)),
        "timings": timings,
    }
    write_json(run_dir / "summary.json", summary)
    _write_selection_decision_record(
        run_dir,
        cfg=cfg,
        profile=profile,
        preflight_df=preflight_df,
        shortlist=shortlist,
        stage2_df=stage2_df,
        finalists=finalists,
        winner=winner,
        timings=timings,
    )

    lines = [
        "# SNAPpy Native Optimizer",
        "",
        f"- Dataset: `{cfg['dataset_name']}`",
        f"- Density regime: `{profile.get('density_regime', 'unknown')}`",
        f"- Contrast regime: `{profile.get('contrast_regime', 'unknown')}`",
        f"- Background regime: `{profile.get('background_regime', 'unknown')}`",
        f"- Winner: `{winner['recipe_id']}`",
        f"- Winner model: `{final_model_path}`",
        f"- Validation F1: `{float(winner['val_f1']):.6f}`",
        f"- Decision threshold: `{float(winner.get('decision_threshold', 0.0)):.6f}`",
        "",
        "## Timing",
        "",
        f"- Preflight wall time: `{timings['preflight_seconds']:.2f} s`",
        f"- Stage 2 wall time: `{timings['stage2_seconds']:.2f} s`",
        f"- Final wall time: `{timings['final_seconds']:.2f} s`",
    ]
    if export_optimize_report:
        lines.extend(
            [
                "",
                "## Report",
                "",
                "- Stage 1 recipes: `export_optimize_report/stage1_recipes.csv`",
                "- Stage 1 per-image metrics: `export_optimize_report/stage1_by_image.csv`",
                "- Stage 1 selection summary: `export_optimize_report/stage1_summary.csv`",
                "- Stage 2 recipes: `export_optimize_report/stage2_recipes.csv`",
                "- Stage 2 validation summary: `export_optimize_report/stage2_summary.csv`",
                "- Full optimizer rationale: `export_optimize_report/selection_decision.md` and `export_optimize_report/selection_decision.json`",
            ]
        )
    if export_candidate_features:
        lines.extend(
            [
                "",
                "## Candidate Features",
                "",
                "- Winning validation candidate features: `export_candidate_features/val_candidates.csv`",
                "- Candidate features manifest: `export_candidate_features/candidate_features_manifest.json`",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(
        run_dir / "model_manifest.json",
        {
            "dataset_name": cfg["dataset_name"],
            "model_path": str(final_model_path),
            "stage2_id": str(winner["recipe_id"]),
            "stage1_id": str(winner.get("stage1_recipe_id", "")),
            "feature_pack_name": dict(winner["recipe"]).get("feature_pack_name"),
            "selected_features": list(dict(winner["recipe"]).get("selected_features") or []),
            "fit_method": dict(winner["recipe"]).get("fit_method"),
            "fit_window": dict(winner["recipe"]).get("fit_window"),
            "svm_params": {
                "kernel": _json_scalar(winner.get("kernel")),
                "C": _json_scalar(winner.get("C")),
                "gamma": _json_scalar(winner.get("gamma")),
                "degree": _json_scalar(winner.get("degree")),
                "standardize": _json_scalar(winner.get("standardize")),
                "class_weight_mode": _json_scalar(winner.get("class_weight_mode")),
            },
            "decision_threshold": float(winner.get("decision_threshold", 0.0)),
            "validation_metrics": {
                "tp": int(winner.get("val_tp", 0)),
                "fp": int(winner.get("val_fp", 0)),
                "fn": int(winner.get("val_fn", 0)),
                "precision": float(winner.get("val_precision", 0.0)),
                "recall": float(winner.get("val_recall", 0.0)),
                "f1": float(winner.get("val_f1", 0.0)),
            },
        },
    )
    if export_optimize_report:
        _write_report_exports(
            run_dir,
            cfg=cfg,
            stage1_recipes=stage1_recipes,
            preflight_df=preflight_df,
            preflight_image_df=preflight_image_df,
            stage2_df=stage2_df,
            winner=winner,
        )
    _remove_optional_root_decision_files(run_dir)
    if export_candidate_features:
        candidate_rows = (candidate_feature_cache or {}).get(str(winner["recipe_id"]), [])
        _write_candidate_feature_export(run_dir, cfg=cfg, winner=winner, candidate_rows=candidate_rows)
    shutil.rmtree(run_dir / "stage2", ignore_errors=True)
    return run_dir / "summary.md"
