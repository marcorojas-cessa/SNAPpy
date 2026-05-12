from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BASE_FEATURES: list[str] = [
    "integrated_intensity",
    "background",
    "snr",
    "r_squared",
    "amplitude",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sigma_sum",
    "sigma_product",
    "sigma_xy_ratio",
    "sigma_z_ratio",
    "amplitude_over_background",
    "rho_xy",
    "rho_xz",
    "rho_yz",
]


FEATURE_PACKS: dict[str, dict[str, Any]] = {
    "base_only": {
        "name": "base_only",
        "features": BASE_FEATURES,
    },
    "curated_balanced": {
        "name": "curated_balanced",
        "features": BASE_FEATURES + ["quality_weighted_snr"],
    },
    "distortion": {
        "name": "distortion",
        "features": BASE_FEATURES + ["distortion_energy"],
    },
    "full": {
        "name": "full",
        "features": BASE_FEATURES
        + [
            "quality_weighted_snr",
            "quality_vs_size_penalty",
            "distortion_energy",
            "log_integrated_intensity",
        ],
    },
    "intensity_quality": {
        "name": "intensity_quality",
        "features": BASE_FEATURES + ["quality_weighted_snr", "log_integrated_intensity"],
    },
    "model_evidence": {
        "name": "model_evidence",
        "features": BASE_FEATURES + ["score_raw"],
    },
    "shape_localization": {
        "name": "shape_localization",
        "features": BASE_FEATURES + ["quality_vs_size_penalty"],
    },
}

STAGE2_FEATURE_PACK_NAMES: list[str] = [
    "base_only",
    "curated_balanced",
    "shape_localization",
    "distortion",
    "intensity_quality",
    "model_evidence",
    "full",
]


STAGE2_FIT_VARIANTS: dict[str, dict[str, Any]] = {
    "distorted_gaussian_3d": {"id": "distorted_gaussian_3d", "fit_method": "Distorted 3D Gaussian", "fit_window": 7},
    "gaussian_3d": {"id": "gaussian_3d", "fit_method": "3D Gaussian", "fit_window": 7},
    "xy_z_gaussian": {"id": "xy_z_gaussian", "fit_method": "2D (XY) + 1D (Z) Gaussian", "fit_window": 7},
}


DEFAULT_NATIVE_CONFIG: dict[str, Any] = {
    "dataset_name": "mrsnappy_dataset",
    "dataset_root": None,
    "match_distance": 4.0,
    "max_detections_per_focus": 32,
    "stage1_cache_enabled": True,
    "stage1_cache_entries": 128,
    "stage1_detector_set": "log",
    "stage1_smoothing_sigmas": [0.5, 1.0, 2.0],
    "stage1_background_method": "rolling_box_3d",
    "stage1_background_params": [5.0, 10.0],
    "stage1_background_include_off": True,
    "fit_cache_enabled": True,
    "fit_cache_entries": 512,
    "runtime_cache": {
        "stage1_cache_enabled": True,
        "stage1_cache_entries": 128,
        "preprocess_cache_entries": 96,
        "fit_cache_enabled": True,
        "fit_cache_entries": 512,
    },
    "profiling": {
        "enabled": True,
        "train_image_count": 8,
        "val_image_count": 4,
        "gt_intensity_radius": 1,
        "sparse_label_mean_max": 64.0,
        "dense_label_mean_min": 256.0,
        "stage1_augmentation_enabled": False,
        "apply_to_explicit_recipes": False,
        "apply_runtime_pruning_to_explicit_recipes": False,
    },
    "preflight": {
        "stage1_n_val_images": 4,
        "min_stage1_recall": 0.25,
        "max_stage1_candidates_mean": 2500,
        "max_stage1_candidates_single": 4000,
        "max_stage1_candidates_per_label_mean": None,
        "auto_candidate_ratio_cap_enabled": True,
        "auto_candidate_ratio_caps": {"sparse": 130.0, "moderate": 100.0, "dense": 70.0},
        "auto_candidate_ratio_low_contrast_multiplier": 1.25,
        "auto_candidate_ratio_high_contrast_multiplier": 0.85,
    },
    "optimizer": {
        "shortlist_top_k": 3,
        "selection_margin": 0.005,
        "max_stage1_preflight_configs": 1000,
        "max_stage2_recipes_after_shortlist": 100,
    },
    "exports": {
        "export_optimize_report": False,
        "export_candidate_features": False,
    },
    "svm_sweep": {
        "kernels": ["linear", "rbf", "polynomial"],
        "box_constraints": [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel_scales": ["auto", 1.0],
        "polynomial_orders": [2, 3],
        "standardize": True,
        "class_weight_mode": "balanced",
        "class_weights": None,
    },
    "stage1_recipes": [],
    "stage2_feature_packs": deepcopy(STAGE2_FEATURE_PACK_NAMES),
    "stage2_fit_variants": ["xy_z_gaussian"],
    "pipeline_defaults": {
        "xy_spacing": 1.0,
        "z_spacing": 1.0,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "preproc_sigma": 0.5,
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "norm_param1": 0.0,
        "norm_param2": 1.0,
        "norm_param3": 0.0,
        "background_enabled": False,
        "background_method": "none",
        "background_mode": "3D",
        "background_projection": "Max",
        "background_scale": False,
        "background_param": 5.0,
        "background_clip": True,
        "maxima_method": "log",
        "maxima_neighborhood": 2,
        "sigma_value": 1.35,
        "threshold_value": 0.30,
        "h_max_sigma_multiplier": 1.0,
        "h_max_sigma_mode": "robust",
        "log_scale_normalize": True,
        "fit_method": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "fit_background_method": "Mean Surrounding Subtraction",
        "fit_background_width": 1,
        "fit_poly_degree": 2,
        "fit_max_iterations": 200,
        "fit_tolerance": 1e-6,
        "full_fit_labeled_candidates_per_label": None,
        "full_fit_labeled_min_candidates": 0,
        "full_fit_unlabeled_candidates_per_expected_label": None,
        "fit_fallback_method": "moments",
        "prefit_prune_enabled": False,
        "prefit_rank_radius": 1,
        "prefit_rank_bg_width": 1,
        "prefit_nms_distance": 0.0,
        "prefit_labeled_candidates_per_label": None,
        "prefit_labeled_min_candidates": 0,
        "prefit_unlabeled_candidates_per_expected_label": None,
        "expected_labels_per_image": None,
        "stage1_cache_enabled": True,
        "stage1_cache_entries": 128,
        "preprocess_cache_entries": 96,
        "fit_cache_enabled": True,
        "fit_cache_entries": 512,
        "negative_to_positive_ratio": None,
        "selected_features": deepcopy(FEATURE_PACKS["curated_balanced"]["features"]),
    },
}


def _log_stage1_grid(
    sigmas: list[float],
    thresholds: list[float],
    neighborhoods: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "maxima_method": "log",
            "sigma_value": float(sigma),
            "maxima_neighborhood": int(neighborhood),
            "threshold_value": float(threshold),
        }
        for sigma in sigmas
        for neighborhood in neighborhoods
        for threshold in thresholds
    ]


def _hmax_stage1_grid(
    multipliers: list[float],
    neighborhoods: list[int],
    sigma_mode: str = "robust",
) -> list[dict[str, Any]]:
    return [
        {
            "maxima_method": "h_max",
            "maxima_neighborhood": int(neighborhood),
            "h_max_sigma_multiplier": float(multiplier),
            "h_max_sigma_mode": str(sigma_mode),
        }
        for neighborhood in neighborhoods
        for multiplier in multipliers
    ]


STAGE1_DETECTOR_PRESETS: dict[str, list[dict[str, Any]]] = {
    "log": _log_stage1_grid(
        sigmas=[1.0, 2.0, 3.0],
        thresholds=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        neighborhoods=[1, 2],
    ),
    "hmax": _hmax_stage1_grid(
        multipliers=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        neighborhoods=[1, 2],
        sigma_mode="robust",
    ),
}


DEFAULT_NATIVE_CONFIG["stage1_recipes"] = deepcopy(STAGE1_DETECTOR_PRESETS["log"])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_text_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


_CACHE_CONFIG_KEYS = (
    "stage1_cache_enabled",
    "stage1_cache_entries",
    "preprocess_cache_entries",
    "fit_cache_enabled",
    "fit_cache_entries",
)


_TOP_LEVEL_CONFIG_KEYS = set(DEFAULT_NATIVE_CONFIG) | {"recipes"}
_NESTED_CONFIG_KEYS = {
    key: set(DEFAULT_NATIVE_CONFIG[key])
    for key in ("profiling", "preflight", "optimizer", "exports", "svm_sweep", "runtime_cache", "pipeline_defaults")
}
_RECIPE_CONFIG_KEYS = set(DEFAULT_NATIVE_CONFIG["pipeline_defaults"]) | {
    "recipe_id",
    "stage1_recipe_id",
    "feature_pack_name",
    "fit_variant_id",
}


def _format_key_list(keys: set[str]) -> str:
    return ", ".join(sorted(keys))


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping of key/value pairs.")
    return value


def _validate_supported_keys(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported {path} key(s): {', '.join(unknown)}. "
            f"Expected only: {_format_key_list(allowed)}."
        )


def _validate_recipe_rows(value: Any, path: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of recipe mappings.")
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{path}[{index}] must be a recipe mapping.")
        _validate_supported_keys(row, _RECIPE_CONFIG_KEYS, f"{path}[{index}]")


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list.")
    return value


def _validate_config_payload(payload: Any) -> dict[str, Any]:
    payload = _require_mapping(payload if payload is not None else {}, "SNAPpy config")
    _validate_supported_keys(payload, _TOP_LEVEL_CONFIG_KEYS, "top-level config")
    for section, allowed in _NESTED_CONFIG_KEYS.items():
        if section in payload:
            _validate_supported_keys(_require_mapping(payload[section], section), allowed, section)
    for recipe_list_key in ("stage1_recipes", "recipes"):
        if recipe_list_key in payload:
            _validate_recipe_rows(payload[recipe_list_key], recipe_list_key)
    if "stage2_feature_packs" in payload:
        pack_names = _require_list(payload["stage2_feature_packs"], "stage2_feature_packs")
        unknown_packs = sorted(set(pack_names) - set(FEATURE_PACKS))
        if unknown_packs:
            raise ValueError(
                f"Unsupported stage2_feature_packs value(s): {', '.join(unknown_packs)}. "
                f"Expected only: {_format_key_list(set(FEATURE_PACKS))}."
            )
    if "stage2_fit_variants" in payload:
        fit_names = _require_list(payload["stage2_fit_variants"], "stage2_fit_variants")
        unknown_fits = sorted(set(fit_names) - set(STAGE2_FIT_VARIANTS))
        if unknown_fits:
            raise ValueError(
                f"Unsupported stage2_fit_variants value(s): {', '.join(unknown_fits)}. "
                f"Expected only: {_format_key_list(set(STAGE2_FIT_VARIANTS))}."
            )
    return payload


def _sync_runtime_cache_config(
    cfg: dict[str, Any],
    *,
    payload_has_runtime_cache: bool,
    payload_top_level_keys: set[str],
) -> None:
    """Keep public runtime_cache settings and internal pipeline defaults aligned."""
    runtime_cache = cfg.setdefault("runtime_cache", {})
    pipeline_defaults = cfg.setdefault("pipeline_defaults", {})
    payload_pipeline_defaults = cfg.get("_payload_pipeline_defaults", {})
    for key in _CACHE_CONFIG_KEYS:
        if key in payload_top_level_keys:
            value = cfg[key]
        elif payload_has_runtime_cache and key in runtime_cache:
            value = runtime_cache[key]
        elif key in payload_pipeline_defaults:
            value = pipeline_defaults[key]
        elif key in pipeline_defaults:
            value = pipeline_defaults[key]
        elif key in runtime_cache:
            value = runtime_cache[key]
        elif key in cfg:
            value = cfg[key]
        else:
            continue
        runtime_cache[key] = value
        pipeline_defaults[key] = value
        if key in {"stage1_cache_enabled", "stage1_cache_entries", "fit_cache_enabled", "fit_cache_entries"}:
            cfg[key] = value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = _validate_config_payload(load_text_config(path))
    payload_has_stage1_recipes = "stage1_recipes" in payload
    payload_has_runtime_cache = "runtime_cache" in payload
    payload_top_level_keys = set(payload)
    payload_pipeline_defaults = deepcopy(payload.get("pipeline_defaults", {})) if isinstance(payload.get("pipeline_defaults"), dict) else {}
    cfg = deepcopy(DEFAULT_NATIVE_CONFIG)
    cfg = deep_merge(cfg, payload)
    cfg["_payload_pipeline_defaults"] = payload_pipeline_defaults
    _sync_runtime_cache_config(
        cfg,
        payload_has_runtime_cache=payload_has_runtime_cache,
        payload_top_level_keys=payload_top_level_keys,
    )
    cfg.pop("_payload_pipeline_defaults", None)
    if not cfg.get("recipes"):
        if payload_has_stage1_recipes:
            cfg["stage1_detector_set"] = str(cfg.get("stage1_detector_set", "custom")).strip().lower()
        else:
            detector_set = str(cfg.get("stage1_detector_set", "log")).strip().lower().replace("-", "")
            if detector_set not in STAGE1_DETECTOR_PRESETS:
                raise ValueError("Unsupported stage1_detector_set. Expected 'log' or 'hmax'.")
            cfg["stage1_detector_set"] = detector_set
            cfg["stage1_recipes"] = deepcopy(STAGE1_DETECTOR_PRESETS[detector_set])
    if cfg.get("dataset_root") is not None:
        cfg["dataset_root"] = str(Path(cfg["dataset_root"]).resolve())
    return cfg


def _map_method_name(value: Any) -> Any:
    if value is None:
        return value
    text = str(value).strip().lower()
    mapping = {
        "gaussian": "gaussian",
        "median": "median",
        "none": "none",
        "min-max": "min_max",
        "min max": "min_max",
        "robust-z-score": "robust_z_score",
        "robust z score": "robust_z_score",
        "robust_z_score": "robust_z_score",
        "z-score": "z_score",
        "z score": "z_score",
        "percentile": "percentile",
        "slice-opening-2d": "slice_opening_2d",
        "slice opening 2d": "slice_opening_2d",
        "slice_opening_2d": "slice_opening_2d",
        "rolling-ball-2d": "rolling_ball_2d",
        "rolling ball 2d": "rolling_ball_2d",
        "rolling_ball_2d": "rolling_ball_2d",
        "slice-wise rolling ball": "rolling_ball_2d",
        "rolling-ball-3d": "rolling_ball_3d",
        "rolling ball 3d": "rolling_ball_3d",
        "rolling_ball_3d": "rolling_ball_3d",
        "rolling-ball-3d-exact": "rolling_ball_3d",
        "rolling ball 3d exact": "rolling_ball_3d",
        "rolling_ball_3d_exact": "rolling_ball_3d",
        "exact 3d rolling ball": "rolling_ball_3d",
        "rolling-box-3d": "rolling_box_3d",
        "rolling box 3d": "rolling_box_3d",
        "rolling_box_3d": "rolling_box_3d",
        "morph opening 3d box": "rolling_box_3d",
        "morph-opening-3d-box": "rolling_box_3d",
        "morph_opening_3d_box": "rolling_box_3d",
        "3d box opening": "rolling_box_3d",
        "box opening 3d": "rolling_box_3d",
        "scipy 3d box": "rolling_box_3d",
        "top-hat": "top_hat",
        "top hat": "top_hat",
        "tophat": "top_hat",
        "laplacian of gaussian": "log",
        "laplacian_of_gaussian": "log",
        "log": "log",
        "simple regional maxima": "simple_regional",
        "simple regional": "simple_regional",
        "extended maxima": "extended_maxima",
        "hmax": "h_max",
        "h-max": "h_max",
        "h max": "h_max",
        "h_max": "h_max",
        "h-maxima": "h_max",
        "h maxima": "h_max",
        "h_maxima": "h_max",
        "mean surrounding subtraction": "Mean Surrounding Subtraction",
        "local plane fitting": "Local Plane Fitting",
        "local polynomial fitting": "Local Polynomial Fitting",
        "2d (xy) + 1d (z) gaussian": "2D (XY) + 1D (Z) Gaussian",
        "3d gaussian": "3D Gaussian",
        "distorted 3d gaussian": "Distorted 3D Gaussian",
        "skewed 3d gaussian": "Skewed 3D Gaussian",
        "radial symmetry": "Radial Symmetry",
    }
    return mapping.get(text, value)


def _apply_feature_pack(recipe: dict[str, Any], pack_name: str) -> dict[str, Any]:
    if pack_name not in FEATURE_PACKS:
        raise KeyError(f"Unknown feature pack: {pack_name}")
    out = deepcopy(recipe)
    pack = FEATURE_PACKS[pack_name]
    out["feature_pack_name"] = pack_name
    out["selected_features"] = deepcopy(pack["features"])
    return out


def _apply_fit_variant(recipe: dict[str, Any], variant_id: str) -> dict[str, Any]:
    if variant_id not in STAGE2_FIT_VARIANTS:
        raise KeyError(f"Unknown fit variant: {variant_id}")
    out = deepcopy(recipe)
    variant = STAGE2_FIT_VARIANTS[variant_id]
    out["fit_variant_id"] = variant_id
    out["fit_method"] = variant["fit_method"]
    out["fit_window"] = variant["fit_window"]
    return out


def _normalize_recipe(row: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    recipe = deepcopy(defaults)
    pack_name = row.get("feature_pack_name")
    if pack_name:
        recipe = _apply_feature_pack(recipe, str(pack_name))
    fit_variant = row.get("fit_variant_id")
    if fit_variant:
        recipe = _apply_fit_variant(recipe, str(fit_variant))
    for key, value in row.items():
        if key in {"feature_pack_name", "fit_variant_id"}:
            continue
        recipe[key] = value
    for method_key in ("preproc_method", "norm_method", "background_method", "maxima_method", "fit_method", "fit_background_method"):
        if method_key in recipe:
            recipe[method_key] = _map_method_name(recipe[method_key])
    return recipe


def _expand_stage1_processing_rows(base_row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [deepcopy(base_row)]
    smoothing_sigmas = cfg.get("stage1_smoothing_sigmas")
    if smoothing_sigmas and "preproc_sigma" not in base_row:
        rows = [
            {
                **deepcopy(row),
                "preproc_enabled": True,
                "preproc_method": "gaussian",
                "preproc_sigma": float(sigma),
            }
            for row in rows
            for sigma in smoothing_sigmas
        ]

    background_params = cfg.get("stage1_background_params")
    background_method = _map_method_name(cfg.get("stage1_background_method", "rolling_box_3d"))
    has_explicit_background = any(
        key in base_row for key in ("background_enabled", "background_method", "background_param")
    )
    if background_params and not has_explicit_background:
        expanded: list[dict[str, Any]] = []
        if cfg.get("stage1_background_include_off", True):
            expanded.extend(
                {
                    **deepcopy(row),
                    "background_enabled": False,
                    "background_method": "none",
                }
                for row in rows
            )
        for row in rows:
            for radius in background_params:
                expanded.append(
                    {
                        **deepcopy(row),
                        "background_enabled": True,
                        "background_method": background_method,
                        "background_mode": "3D",
                        "background_scale": False,
                        "background_param": float(radius),
                    }
                )
        rows = expanded
    return rows


def _id_value(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "")


def _stage1_recipe_id(recipe: dict[str, Any]) -> str:
    method_text = str(recipe.get("maxima_method", "log")).lower().replace("_", "")
    smooth_text = f"g{_id_value(recipe.get('preproc_sigma', 0.0))}" if recipe.get("preproc_enabled", True) else "g0"
    bg_text = "bg0"
    if recipe.get("background_enabled", False):
        bg_method = str(recipe.get("background_method", "background")).lower()
        bg_radius = _id_value(recipe.get("background_param", 0.0))
        bg_prefix = {
            "slice_opening_2d": "slice2d",
            "rolling_ball_2d": "rb2d",
            "rolling_ball_3d": "rb3d",
            "rolling_box_3d": "rbox3d",
        }.get(bg_method, bg_method)
        bg_text = f"{bg_prefix}{bg_radius}"
    if method_text in {"hmax", "h-max", "hmaxima", "h-maxima"}:
        h_text = _id_value(recipe["h_max_sigma_multiplier"])
        h_mode = _id_value(str(recipe.get("h_max_sigma_mode", "robust")).lower())
        return f"hmax_{h_mode}_h{h_text}_n{recipe['maxima_neighborhood']}_{smooth_text}_{bg_text}"
    sigma_text = _id_value(recipe["sigma_value"])
    thresh_text = _id_value(recipe["threshold_value"])
    return f"{method_text}_s{sigma_text}_n{recipe['maxima_neighborhood']}_t{thresh_text}_{smooth_text}_{bg_text}"


def _stage2_recipe_id(stage1_recipe: dict[str, Any], recipe: dict[str, Any]) -> str:
    base = str(stage1_recipe["recipe_id"])
    fit_text = str(recipe.get("fit_variant_id", "fit")).lower()
    pack_text = str(recipe.get("feature_pack_name", "pack")).lower()
    return f"{base}_{fit_text}_{pack_text}"


def _canonical_stage1_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(round(value, 8))
    if isinstance(value, list):
        return [_canonical_stage1_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_stage1_value(item) for item in value]
    return value


def _stage1_dedup_key(recipe: dict[str, Any]) -> str:
    method = str(recipe.get("maxima_method", "log")).strip().lower().replace("-", "_").replace(" ", "_")
    if method in {"laplacian_of_gaussian"}:
        method = "log"
    if method in {"extended", "extended_maxima", "hmax", "h_maxima"}:
        method = "h_max"

    payload: dict[str, Any] = {
        "xy_spacing": recipe.get("xy_spacing"),
        "z_spacing": recipe.get("z_spacing"),
        "norm_enabled": bool(recipe.get("norm_enabled", True)),
        "norm_method": recipe.get("norm_method"),
        "norm_param1": recipe.get("norm_param1"),
        "norm_param2": recipe.get("norm_param2"),
        "norm_param3": recipe.get("norm_param3"),
        "preproc_enabled": bool(recipe.get("preproc_enabled", True)),
        "background_enabled": bool(recipe.get("background_enabled", False)),
        "maxima_method": method,
        "maxima_neighborhood": int(recipe.get("maxima_neighborhood", 2)),
    }
    if payload["preproc_enabled"]:
        payload.update(
            {
                "preproc_method": recipe.get("preproc_method"),
                "preproc_sigma": recipe.get("preproc_sigma"),
            }
        )
    if payload["background_enabled"]:
        payload.update(
            {
                "background_method": recipe.get("background_method"),
                "background_mode": recipe.get("background_mode"),
                "background_scale": bool(recipe.get("background_scale", False)),
                "background_param": recipe.get("background_param"),
                "background_clip": bool(recipe.get("background_clip", True)),
            }
        )
    if method == "h_max":
        payload.update(
            {
                "h_max_sigma_multiplier": recipe.get("h_max_sigma_multiplier"),
                "h_max_sigma_mode": recipe.get("h_max_sigma_mode", "robust"),
            }
        )
    else:
        payload.update(
            {
                "sigma_value": recipe.get("sigma_value"),
                "threshold_value": recipe.get("threshold_value"),
                "log_scale_normalize": bool(recipe.get("log_scale_normalize", True)),
            }
        )
    canonical = {key: _canonical_stage1_value(value) for key, value in payload.items()}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _deduplicate_stage1_recipes(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    for recipe in recipes:
        key = _stage1_dedup_key(recipe)
        if key in seen:
            kept = seen[key]
            aliases = kept.setdefault("deduplicated_from_recipe_ids", [])
            aliases.append(str(recipe.get("recipe_id", "")))
            kept["deduplicated_duplicate_count"] = int(kept.get("deduplicated_duplicate_count", 0)) + 1
            continue
        out = deepcopy(recipe)
        out["stage1_dedup_key"] = key
        out["deduplicated_duplicate_count"] = 0
        seen[key] = out
        deduped.append(out)
    return deduped


def _assert_unique_recipe_ids(recipes: list[dict[str, Any]], *, context: str) -> None:
    recipe_ids = [str(recipe["recipe_id"]) for recipe in recipes]
    duplicate_ids = sorted({recipe_id for recipe_id in recipe_ids if recipe_ids.count(recipe_id) > 1})
    if duplicate_ids:
        joined = ", ".join(duplicate_ids[:10])
        suffix = "" if len(duplicate_ids) <= 10 else f", ... ({len(duplicate_ids)} total)"
        raise ValueError(f"{context} recipe IDs must be unique; duplicate recipe_id value(s): {joined}{suffix}")


def recipe_bank(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return stage1_recipe_bank(cfg)


def stage1_recipe_bank(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = deepcopy(cfg["pipeline_defaults"])
    explicit_rows = cfg.get("recipes", [])
    rows: list[dict[str, Any]] = []
    if explicit_rows:
        rows = explicit_rows
    else:
        stage1_rows = cfg.get("stage1_recipes", [])
        for base_row in stage1_rows:
            for stage1_row in _expand_stage1_processing_rows(base_row, cfg):
                rows.append(deepcopy(stage1_row))
    recipes: list[dict[str, Any]] = []
    for row in rows:
        stage1_row = deepcopy(row)
        stage1_row.pop("feature_pack_name", None)
        stage1_row.pop("fit_variant_id", None)
        recipe = _normalize_recipe(stage1_row, defaults)
        recipe_id = stage1_row.get("recipe_id")
        if not recipe_id:
            recipe_id = _stage1_recipe_id(recipe)
        recipe["recipe_id"] = recipe_id
        recipe["stage1_recipe_id"] = recipe_id
        recipes.append(recipe)
    recipes = _deduplicate_stage1_recipes(recipes)
    _assert_unique_recipe_ids(recipes, context="Stage 1")
    return recipes


def stage2_recipe_bank(cfg: dict[str, Any], stage1_recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pack_names = cfg.get("stage2_feature_packs", ["curated_balanced"])
    fit_variant_ids = cfg.get("stage2_fit_variants", ["xy_z_gaussian"])
    recipes: list[dict[str, Any]] = []
    for stage1_recipe in stage1_recipes:
        for fit_variant_id in fit_variant_ids:
            for pack_name in pack_names:
                recipe = _apply_fit_variant(deepcopy(stage1_recipe), str(fit_variant_id))
                recipe = _apply_feature_pack(recipe, str(pack_name))
                recipe["stage1_recipe_id"] = stage1_recipe["recipe_id"]
                recipe["recipe_id"] = _stage2_recipe_id(stage1_recipe, recipe)
                recipes.append(recipe)
    _assert_unique_recipe_ids(recipes, context="Stage 2")
    return recipes
