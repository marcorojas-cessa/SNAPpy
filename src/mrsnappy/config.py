from __future__ import annotations

import json
import math
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .features import FEATURE_PACK_DEFINITIONS, STAGE2_FEATURE_PACK_NAMES, resolve_feature_pack_features

FEATURE_PACKS: dict[str, dict[str, Any]] = deepcopy(FEATURE_PACK_DEFINITIONS)


FITTING_MODES: dict[str, dict[str, Any]] = {
    "distorted_gaussian_3d": {"id": "distorted_gaussian_3d", "fit_method": "Distorted 3D Gaussian", "fit_window": 7},
    "gaussian_3d": {"id": "gaussian_3d", "fit_method": "3D Gaussian", "fit_window": 7},
    "xy_z_gaussian": {"id": "xy_z_gaussian", "fit_method": "2D (XY) + 1D (Z) Gaussian", "fit_window": 7},
}


DEFAULT_NATIVE_CONFIG: dict[str, Any] = {
    "dataset_name": "mrsnappy_dataset",
    "dataset_root": None,
    "optimization_mode": "fixed_split",
    "match_distance": 3.0,
    "match_distance_nm": None,
    "stage1_detector_set": "hmax",
    "stage1_log_sigmas": [1.0, 2.0, 3.0],
    "stage1_log_sigmas_nm": [],
    "stage1_log_thresholds": [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    "stage1_maxima_neighborhoods": [1, 2],
    "stage1_maxima_min_distances_nm": [],
    "stage1_hmax_multipliers": [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    "stage1_hmax_sigma_mode": "robust",
    "stage1_smoothing_sigmas": ["off", 0.5, 1.0, 2.0],
    "stage1_smoothing_sigmas_nm": [],
    "stage1_background_method": "rolling_box_3d",
    "stage1_background_params": ["off", 5.0, 10.0],
    "stage1_background_params_nm": [],
    "runtime_cache": {
        "stage1_cache_enabled": True,
        "stage1_cache_entries": 128,
        "image_volume_cache_entries": 96,
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
        "apply_runtime_pruning_to_explicit_recipes": False,
    },
    "preflight": {
        "stage1_n_val_images": 5,
        "min_stage1_recall_mean": None,
        "max_stage1_candidates_mean": None,
        "max_stage1_candidates_single": None,
        "max_candidate_ratio_cap_mean": 500.0,
    },
    "stage1_ranking": {
        "recall_tolerance": 0.05,
    },
    "optimizer": {
        "shortlist_top_k": 5,
        "stage2_f1_tolerance": 0.005,
        "max_stage1_preflight_configs": 1000,
        "max_stage2_recipes_after_shortlist": 100,
    },
    "svm_sweep": {
        "kernels": ["linear", "rbf", "polynomial"],
        "box_constraints": [0.1, 1.0, 10.0],
        "kernel_scales": ["auto", "scale", 0.1, 1.0, 10.0],
        "polynomial_orders": [2, 3],
        "standardize": True,
        "class_weighting": "on",
    },
    "stage1_recipes": [],
    "stage2_feature_packs": deepcopy(STAGE2_FEATURE_PACK_NAMES),
    "pipeline_defaults": {
        "xy_spacing_nm": None,
        "z_spacing_nm": None,
        "preproc_enabled": True,
        "preproc_method": "gaussian",
        "preproc_sigma": 0.5,
        "preproc_sigma_nm": None,
        "norm_enabled": True,
        "norm_method": "robust_z_score",
        "background_enabled": False,
        "background_method": "none",
        "background_param": 5.0,
        "background_param_nm": None,
        "background_clip": True,
        "maxima_method": "log",
        "maxima_neighborhood": 2,
        "maxima_min_distance_nm": None,
        "sigma_value": 1.35,
        "sigma_nm": None,
        "threshold_value": 0.30,
        "h_max_sigma_multiplier": 1.0,
        "h_max_sigma_mode": "robust",
        "fit_method": "2D (XY) + 1D (Z) Gaussian",
        "fit_window": 7,
        "fit_background_width": 1,
        "fit_max_iterations": 200,
        "fit_tolerance": 1e-6,
        "stage1_cache_enabled": True,
        "stage1_cache_entries": 128,
        "image_volume_cache_entries": 96,
        "fit_cache_enabled": True,
        "fit_cache_entries": 512,
        "selected_features": resolve_feature_pack_features("core_fit", "2D (XY) + 1D (Z) Gaussian"),
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


def _log_stage1_grid_nm(
    sigmas_nm: list[float],
    thresholds: list[float],
    min_distances_nm: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "maxima_method": "log",
            "sigma_value": None,
            "sigma_nm": float(sigma),
            "maxima_neighborhood": None,
            "maxima_min_distance_nm": float(min_distance),
            "threshold_value": float(threshold),
        }
        for sigma in sigmas_nm
        for min_distance in min_distances_nm
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
            "sigma_value": None,
            "sigma_nm": None,
            "threshold_value": None,
            "h_max_sigma_multiplier": float(multiplier),
            "h_max_sigma_mode": str(sigma_mode),
        }
        for neighborhood in neighborhoods
        for multiplier in multipliers
    ]


def _hmax_stage1_grid_nm(
    multipliers: list[float],
    min_distances_nm: list[float],
    sigma_mode: str = "robust",
) -> list[dict[str, Any]]:
    return [
        {
            "maxima_method": "h_max",
            "maxima_neighborhood": None,
            "maxima_min_distance_nm": float(min_distance),
            "sigma_value": None,
            "sigma_nm": None,
            "threshold_value": None,
            "h_max_sigma_multiplier": float(multiplier),
            "h_max_sigma_mode": str(sigma_mode),
        }
        for min_distance in min_distances_nm
        for multiplier in multipliers
    ]


STAGE1_DETECTOR_PRESETS: dict[str, list[dict[str, Any]]] = {
    "log": _log_stage1_grid(
        sigmas=[1.0, 2.0, 3.0],
        thresholds=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        neighborhoods=[1, 2],
    ),
    "hmax": _hmax_stage1_grid(
        multipliers=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        neighborhoods=[1, 2],
        sigma_mode="robust",
    ),
}


DEFAULT_NATIVE_CONFIG["stage1_recipes"] = deepcopy(STAGE1_DETECTOR_PRESETS["hmax"])


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


def _resolve_config_path(path: str | Path) -> Path:
    if str(path).strip().lower() == "default":
        return Path(str(resources.files("mrsnappy").joinpath("resources/default_optimize.yaml")))
    return Path(path)


_CACHE_CONFIG_KEYS = (
    "stage1_cache_enabled",
    "stage1_cache_entries",
    "image_volume_cache_entries",
    "fit_cache_enabled",
    "fit_cache_entries",
)
_STAGE1_SWEEP_LIST_KEYS = (
    "stage1_log_sigmas",
    "stage1_log_sigmas_nm",
    "stage1_log_thresholds",
    "stage1_maxima_neighborhoods",
    "stage1_maxima_min_distances_nm",
    "stage1_hmax_multipliers",
    "stage1_smoothing_sigmas",
    "stage1_smoothing_sigmas_nm",
    "stage1_background_params",
    "stage1_background_params_nm",
)
_STAGE1_MATRIX_KEYS = set(_STAGE1_SWEEP_LIST_KEYS) | {"stage1_detector_set", "stage1_recipes"}
_STAGE2_SWEEP_LIST_KEYS = ("stage2_feature_packs",)
_SVM_SWEEP_LIST_KEYS = ("kernels", "box_constraints", "kernel_scales", "polynomial_orders")


def _apply_declared_matrix_semantics(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    if set(payload) & _STAGE1_MATRIX_KEYS:
        for key in _STAGE1_SWEEP_LIST_KEYS:
            if key not in payload:
                cfg[key] = []
    if set(payload) & set(_STAGE2_SWEEP_LIST_KEYS):
        for key in _STAGE2_SWEEP_LIST_KEYS:
            if key not in payload:
                cfg[key] = []
    if "svm_sweep" in payload:
        svm_payload = payload.get("svm_sweep", {})
        if isinstance(svm_payload, dict) and set(svm_payload) & set(_SVM_SWEEP_LIST_KEYS):
            for key in _SVM_SWEEP_LIST_KEYS:
                if key not in svm_payload:
                    cfg["svm_sweep"][key] = []


def _stage1_detector_rows(cfg: dict[str, Any], detector_set: str) -> list[dict[str, Any]]:
    thresholds = [float(value) for value in cfg.get("stage1_log_thresholds", [])]
    neighborhoods = [int(value) for value in cfg.get("stage1_maxima_neighborhoods", [])]
    min_distances_nm = [float(value) for value in cfg.get("stage1_maxima_min_distances_nm", [])]
    multipliers = [float(value) for value in cfg.get("stage1_hmax_multipliers", [])]
    sigma_mode = str(cfg.get("stage1_hmax_sigma_mode", "robust"))
    rows: list[dict[str, Any]] = []
    if detector_set == "log":
        rows.extend(
            _log_stage1_grid(
                [float(value) for value in cfg.get("stage1_log_sigmas", [])],
                thresholds,
                neighborhoods,
            )
        )
        rows.extend(
            _log_stage1_grid_nm(
                [float(value) for value in cfg.get("stage1_log_sigmas_nm", [])],
                thresholds,
                min_distances_nm,
            )
        )
    elif detector_set == "hmax":
        rows.extend(_hmax_stage1_grid(multipliers, neighborhoods, sigma_mode=sigma_mode))
        rows.extend(_hmax_stage1_grid_nm(multipliers, min_distances_nm, sigma_mode=sigma_mode))
    if not rows:
        raise ValueError(f"stage1_detector_set={detector_set!r} produced no Stage 1 recipes.")
    return rows


_TOP_LEVEL_CONFIG_KEYS = set(DEFAULT_NATIVE_CONFIG)
_NESTED_CONFIG_KEYS = {
    key: set(DEFAULT_NATIVE_CONFIG[key])
    for key in ("profiling", "preflight", "stage1_ranking", "optimizer", "svm_sweep", "runtime_cache", "pipeline_defaults")
}
_RECIPE_CONFIG_KEYS = set(DEFAULT_NATIVE_CONFIG["pipeline_defaults"]) | {
    "recipe_id",
    "stage1_recipe_id",
    "feature_pack_name",
    "feature_cache_features",
    "model_type",
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
    if payload.get("match_distance") is not None and payload.get("match_distance_nm") is not None:
        raise ValueError("Specify exactly one of match_distance or match_distance_nm, not both.")
    for section, allowed in _NESTED_CONFIG_KEYS.items():
        if section in payload:
            _validate_supported_keys(_require_mapping(payload[section], section), allowed, section)
    if "stage1_recipes" in payload:
        _validate_recipe_rows(payload["stage1_recipes"], "stage1_recipes")
    if "stage2_feature_packs" in payload:
        pack_names = _require_list(payload["stage2_feature_packs"], "stage2_feature_packs")
        unknown_packs = sorted(set(pack_names) - set(FEATURE_PACKS))
        if unknown_packs:
            raise ValueError(
                f"Unsupported stage2_feature_packs value(s): {', '.join(unknown_packs)}. "
                f"Expected only: {_format_key_list(set(FEATURE_PACKS))}."
            )
    if "optimization_mode" in payload:
        mode = str(payload["optimization_mode"]).strip().lower()
        if mode != "fixed_split":
            raise NotImplementedError(
                "SNAPpy optimization_mode='cross_validation' is planned but not implemented yet. "
                "Use optimization_mode: fixed_split with dataset_root containing train/ and val/ folders."
            )
    for key in _STAGE1_SWEEP_LIST_KEYS:
        if key in payload:
            _require_list(payload[key], key)
    if "svm_sweep" in payload:
        svm_payload = _require_mapping(payload["svm_sweep"], "svm_sweep")
        for key in _SVM_SWEEP_LIST_KEYS:
            if key in svm_payload:
                _require_list(svm_payload[key], f"svm_sweep.{key}")
        if "class_weighting" in svm_payload and str(svm_payload["class_weighting"]).strip().lower() not in {"on", "off"}:
            raise ValueError("svm_sweep.class_weighting must be 'on' or 'off'.")
    return payload


def _validate_preflight_semantics(cfg: dict[str, Any]) -> None:
    preflight = cfg.get("preflight", {})
    if "min_stage1_recall" in preflight:
        raise ValueError("preflight.min_stage1_recall was renamed to preflight.min_stage1_recall_mean.")
    stage1_n_val_images = preflight.get("stage1_n_val_images")
    if isinstance(stage1_n_val_images, str):
        if stage1_n_val_images.strip().lower() != "all":
            raise ValueError("preflight.stage1_n_val_images must be a positive integer or 'all'.")
    elif isinstance(stage1_n_val_images, bool) or not isinstance(stage1_n_val_images, int) or stage1_n_val_images <= 0:
        raise ValueError("preflight.stage1_n_val_images must be a positive integer or 'all'.")

    min_recall_value = preflight.get("min_stage1_recall_mean")
    if min_recall_value is not None:
        min_recall = _preflight_number(min_recall_value, "preflight.min_stage1_recall_mean")
        if not (0.0 < min_recall <= 1.0):
            raise ValueError("preflight.min_stage1_recall_mean must be greater than 0 and less than or equal to 1.")

    max_mean_candidates_value = preflight.get("max_stage1_candidates_mean")
    if max_mean_candidates_value is not None:
        max_mean_candidates = _preflight_number(max_mean_candidates_value, "preflight.max_stage1_candidates_mean")
        if max_mean_candidates <= 0:
            raise ValueError("preflight.max_stage1_candidates_mean must be a positive number.")

    max_single_candidates_value = preflight.get("max_stage1_candidates_single")
    if max_single_candidates_value is not None:
        max_single_candidates = _preflight_number(max_single_candidates_value, "preflight.max_stage1_candidates_single")
        if max_single_candidates <= 0:
            raise ValueError("preflight.max_stage1_candidates_single must be a positive number.")

    ratio_cap = preflight.get("max_candidate_ratio_cap_mean")
    if ratio_cap is None:
        return
    if isinstance(ratio_cap, str):
        raise ValueError("preflight.max_candidate_ratio_cap_mean must be a positive number when used.")
    if _preflight_number(ratio_cap, "preflight.max_candidate_ratio_cap_mean") <= 0:
        raise ValueError("preflight.max_candidate_ratio_cap_mean must be a positive number when used.")


def _validate_stage1_ranking_semantics(cfg: dict[str, Any]) -> None:
    ranking = cfg.get("stage1_ranking", {})
    recall_tolerance = _preflight_number(ranking.get("recall_tolerance"), "stage1_ranking.recall_tolerance")
    if not (0.0 <= recall_tolerance <= 1.0):
        raise ValueError("stage1_ranking.recall_tolerance must be between 0 and 1.")

    shortlist_top_k = cfg.get("optimizer", {}).get("shortlist_top_k")
    if isinstance(shortlist_top_k, bool) or not isinstance(shortlist_top_k, int) or shortlist_top_k <= 0:
        raise ValueError("optimizer.shortlist_top_k must be a positive integer.")

    optimizer = cfg.get("optimizer", {})
    tolerance = _preflight_number(optimizer.get("stage2_f1_tolerance"), "optimizer.stage2_f1_tolerance")
    if not (0.0 <= tolerance <= 1.0):
        raise ValueError("optimizer.stage2_f1_tolerance must be between 0 and 1.")


def _validate_runtime_cache_semantics(cfg: dict[str, Any]) -> None:
    runtime_cache = cfg.get("runtime_cache", {})
    for key in ("stage1_cache_enabled", "fit_cache_enabled"):
        if not isinstance(runtime_cache.get(key), bool):
            raise ValueError(f"runtime_cache.{key} must be either true or false.")
    for key in ("stage1_cache_entries", "image_volume_cache_entries", "fit_cache_entries"):
        value = runtime_cache.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"runtime_cache.{key} must be a non-negative integer.")


def _validate_profiling_semantics(cfg: dict[str, Any]) -> None:
    profiling = cfg.get("profiling", {})
    for key in (
        "enabled",
        "stage1_augmentation_enabled",
        "apply_runtime_pruning_to_explicit_recipes",
    ):
        if not isinstance(profiling.get(key), bool):
            raise ValueError(f"profiling.{key} must be either true or false.")
    for key in ("train_image_count", "val_image_count"):
        value = profiling.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"profiling.{key} must be a positive integer.")
    value = profiling.get("gt_intensity_radius")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("profiling.gt_intensity_radius must be a non-negative integer.")
    sparse_max = _positive_finite_number(profiling.get("sparse_label_mean_max"), "profiling.sparse_label_mean_max")
    dense_min = _positive_finite_number(profiling.get("dense_label_mean_min"), "profiling.dense_label_mean_min")
    if dense_min <= sparse_max:
        raise ValueError("profiling.dense_label_mean_min must be greater than profiling.sparse_label_mean_max.")


_PUBLIC_FIT_METHODS = {
    "2D (XY) + 1D (Z) Gaussian",
    "3D Gaussian",
    "Distorted 3D Gaussian",
}
def _positive_finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive finite number.")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive finite number.") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{path} must be a positive finite number.")
    return out


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer.")
    return int(value)


def _positive_odd_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2 == 0:
        raise ValueError(f"{path} must be a positive odd integer.")
    return int(value)


def _canonical_fit_method(value: Any) -> str:
    mapped = _map_method_name(value)
    return str(mapped).strip()


def _validate_fit_controls(mapping: dict[str, Any], path: str) -> None:
    if "fit_method" in mapping:
        fit_method = _canonical_fit_method(mapping["fit_method"])
        if fit_method not in _PUBLIC_FIT_METHODS:
            raise ValueError(
                f"{path}.fit_method must be one of: "
                "2D (XY) + 1D (Z) Gaussian, 3D Gaussian, Distorted 3D Gaussian."
            )
    if "fit_window" in mapping:
        _positive_odd_integer(mapping["fit_window"], f"{path}.fit_window")
    if "fit_background_width" in mapping:
        _nonnegative_integer(mapping["fit_background_width"], f"{path}.fit_background_width")
    if "fit_max_iterations" in mapping:
        value = mapping["fit_max_iterations"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path}.fit_max_iterations must be a positive integer.")
    if "fit_tolerance" in mapping:
        _positive_finite_number(mapping["fit_tolerance"], f"{path}.fit_tolerance")


def _validate_pipeline_defaults_semantics(cfg: dict[str, Any]) -> None:
    pipeline_defaults = cfg.get("pipeline_defaults", {})
    if cfg.get("dataset_root") is not None:
        for key in ("xy_spacing_nm", "z_spacing_nm"):
            _positive_finite_number(pipeline_defaults.get(key), f"pipeline_defaults.{key}")
    _validate_fit_controls(pipeline_defaults, "pipeline_defaults")


def _validate_match_distance_semantics(cfg: dict[str, Any]) -> None:
    has_voxel_distance = cfg.get("match_distance") is not None
    has_physical_distance = cfg.get("match_distance_nm") is not None
    if has_voxel_distance == has_physical_distance:
        raise ValueError("SNAPpy config must specify exactly one of match_distance or match_distance_nm.")
    if has_voxel_distance:
        _positive_finite_number(cfg.get("match_distance"), "match_distance")
        return
    _positive_finite_number(cfg.get("match_distance_nm"), "match_distance_nm")
    pipeline_defaults = cfg.get("pipeline_defaults", {})
    for key in ("xy_spacing_nm", "z_spacing_nm"):
        _positive_finite_number(pipeline_defaults.get(key), f"pipeline_defaults.{key}")


def _preflight_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a number.") from exc


def _sync_runtime_cache_config(
    cfg: dict[str, Any],
    *,
    payload_has_runtime_cache: bool,
) -> None:
    """Keep public runtime_cache settings and internal pipeline defaults aligned."""
    runtime_cache = cfg.setdefault("runtime_cache", {})
    pipeline_defaults = cfg.setdefault("pipeline_defaults", {})
    payload_pipeline_defaults = cfg.get("_payload_pipeline_defaults", {})
    for key in _CACHE_CONFIG_KEYS:
        if payload_has_runtime_cache and key in runtime_cache:
            value = runtime_cache[key]
        elif key in payload_pipeline_defaults:
            value = pipeline_defaults[key]
        elif key in pipeline_defaults:
            value = pipeline_defaults[key]
        elif key in runtime_cache:
            value = runtime_cache[key]
        else:
            continue
        runtime_cache[key] = value
        pipeline_defaults[key] = value


def load_config(path: str | Path) -> dict[str, Any]:
    path = _resolve_config_path(path)
    payload = _validate_config_payload(load_text_config(path))
    payload_has_stage1_recipes = "stage1_recipes" in payload
    payload_has_runtime_cache = "runtime_cache" in payload
    payload_pipeline_defaults = deepcopy(payload.get("pipeline_defaults", {})) if isinstance(payload.get("pipeline_defaults"), dict) else {}
    cfg = deepcopy(DEFAULT_NATIVE_CONFIG)
    cfg = deep_merge(cfg, payload)
    if payload.get("match_distance_nm") is not None and "match_distance" not in payload:
        cfg["match_distance"] = None
    _apply_declared_matrix_semantics(cfg, payload)
    cfg["_payload_pipeline_defaults"] = payload_pipeline_defaults
    _sync_runtime_cache_config(
        cfg,
        payload_has_runtime_cache=payload_has_runtime_cache,
    )
    _validate_pipeline_defaults_semantics(cfg)
    _validate_match_distance_semantics(cfg)
    _validate_preflight_semantics(cfg)
    _validate_stage1_ranking_semantics(cfg)
    _validate_runtime_cache_semantics(cfg)
    _validate_profiling_semantics(cfg)
    cfg.pop("_payload_pipeline_defaults", None)
    if payload_has_stage1_recipes:
        cfg["stage1_detector_set"] = "custom"
    else:
        detector_set = str(cfg.get("stage1_detector_set", "log")).strip().lower().replace("-", "")
        if detector_set not in STAGE1_DETECTOR_PRESETS:
            raise ValueError("Unsupported stage1_detector_set. Expected 'log' or 'hmax'.")
        cfg["stage1_detector_set"] = detector_set
        cfg["stage1_recipes"] = _stage1_detector_rows(cfg, detector_set)
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
        "hmax": "h_max",
        "h-max": "h_max",
        "h max": "h_max",
        "h_max": "h_max",
        "h-maxima": "h_max",
        "h maxima": "h_max",
        "h_maxima": "h_max",
        "xy_z_gaussian": "2D (XY) + 1D (Z) Gaussian",
        "gaussian_3d": "3D Gaussian",
        "distorted_gaussian_3d": "Distorted 3D Gaussian",
        "2d (xy) + 1d (z) gaussian": "2D (XY) + 1D (Z) Gaussian",
        "3d gaussian": "3D Gaussian",
        "distorted 3d gaussian": "Distorted 3D Gaussian",
    }
    return mapping.get(text, value)


def _apply_feature_pack(recipe: dict[str, Any], pack_name: str) -> dict[str, Any]:
    if pack_name not in FEATURE_PACKS:
        raise KeyError(f"Unknown feature pack: {pack_name}")
    out = deepcopy(recipe)
    fit_method = out.get("fit_method")
    out["feature_pack_name"] = pack_name
    out["selected_features"] = resolve_feature_pack_features(pack_name, fit_method)
    return out


def _normalize_recipe(row: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    recipe = deepcopy(defaults)
    pack_name = row.get("feature_pack_name")
    if pack_name:
        recipe = _apply_feature_pack(recipe, str(pack_name))
    for key, value in row.items():
        if key in {"feature_pack_name"}:
            continue
        recipe[key] = value
    for method_key in ("preproc_method", "norm_method", "background_method", "maxima_method", "fit_method"):
        if method_key in recipe:
            recipe[method_key] = _map_method_name(recipe[method_key])
    _validate_fit_controls(recipe, "recipe")
    return recipe


def _matrix_value_is_off(value: Any) -> bool:
    if value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in {"off", "none", "false", "disabled"}


def _expand_stage1_processing_rows(base_row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [deepcopy(base_row)]
    has_explicit_smoothing = any(key in base_row for key in ("preproc_sigma", "preproc_sigma_nm"))
    if not has_explicit_smoothing:
        smoothing_rows: list[dict[str, Any]] = []
        for row in rows:
            for sigma in cfg.get("stage1_smoothing_sigmas", []) or []:
                if _matrix_value_is_off(sigma):
                    smoothing_rows.append(
                        {
                            **deepcopy(row),
                            "preproc_enabled": False,
                            "preproc_method": "none",
                            "preproc_sigma": None,
                            "preproc_sigma_nm": None,
                        }
                    )
                    continue
                smoothing_rows.append(
                    {
                        **deepcopy(row),
                        "preproc_enabled": True,
                        "preproc_method": "gaussian",
                        "preproc_sigma": float(sigma),
                        "preproc_sigma_nm": None,
                    }
                )
            for sigma_nm in cfg.get("stage1_smoothing_sigmas_nm", []) or []:
                if _matrix_value_is_off(sigma_nm):
                    smoothing_rows.append(
                        {
                            **deepcopy(row),
                            "preproc_enabled": False,
                            "preproc_method": "none",
                            "preproc_sigma": None,
                            "preproc_sigma_nm": None,
                        }
                    )
                    continue
                smoothing_rows.append(
                    {
                        **deepcopy(row),
                        "preproc_enabled": True,
                        "preproc_method": "gaussian",
                        "preproc_sigma": None,
                        "preproc_sigma_nm": float(sigma_nm),
                    }
                )
        if smoothing_rows:
            rows = smoothing_rows

    background_method = _map_method_name(cfg.get("stage1_background_method", "rolling_box_3d"))
    has_explicit_background = any(
        key in base_row for key in ("background_enabled", "background_method", "background_param", "background_param_nm")
    )
    background_params = cfg.get("stage1_background_params", []) or []
    background_params_nm = cfg.get("stage1_background_params_nm", []) or []
    if (background_params or background_params_nm) and not has_explicit_background:
        expanded: list[dict[str, Any]] = []
        for row in rows:
            for radius in background_params:
                if _matrix_value_is_off(radius):
                    expanded.append(
                        {
                            **deepcopy(row),
                            "background_enabled": False,
                            "background_method": "none",
                            "background_param": None,
                            "background_param_nm": None,
                        }
                    )
                    continue
                expanded.append(
                    {
                        **deepcopy(row),
                        "background_enabled": True,
                        "background_method": background_method,
                        "background_param": float(radius),
                        "background_param_nm": None,
                    }
                )
            for radius_nm in background_params_nm:
                if _matrix_value_is_off(radius_nm):
                    expanded.append(
                        {
                            **deepcopy(row),
                            "background_enabled": False,
                            "background_method": "none",
                            "background_param": None,
                            "background_param_nm": None,
                        }
                    )
                    continue
                expanded.append(
                    {
                        **deepcopy(row),
                        "background_enabled": True,
                        "background_method": background_method,
                        "background_param": None,
                        "background_param_nm": float(radius_nm),
                    }
                )
        rows = expanded
    return rows


def _id_value(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "")


def _has_value(mapping: dict[str, Any], key: str) -> bool:
    return key in mapping and mapping.get(key) is not None


def _stage1_recipe_id(recipe: dict[str, Any]) -> str:
    method_text = str(recipe.get("maxima_method", "log")).lower().replace("_", "")
    if not recipe.get("preproc_enabled", True):
        smooth_text = "smoothoff"
    elif _has_value(recipe, "preproc_sigma_nm"):
        smooth_text = f"g{_id_value(recipe.get('preproc_sigma_nm', 0.0))}nm"
    else:
        smooth_text = f"g{_id_value(recipe.get('preproc_sigma', 0.0))}"
    bg_text = "bg0"
    if recipe.get("background_enabled", False):
        bg_method = str(recipe.get("background_method", "background")).lower()
        physical_background = _has_value(recipe, "background_param_nm")
        bg_radius = _id_value(recipe.get("background_param_nm" if physical_background else "background_param", 0.0))
        bg_prefix = {
            "slice_opening_2d": "slice2d",
            "rolling_ball_2d": "rb2d",
            "rolling_ball_3d": "rb3d",
            "rolling_box_3d": "rbox3d",
        }.get(bg_method, bg_method)
        bg_text = f"{bg_prefix}{bg_radius}{'nm' if physical_background else ''}"
    if method_text in {"hmax", "h-max", "hmaxima", "h-maxima"}:
        h_text = _id_value(recipe["h_max_sigma_multiplier"])
        h_mode = _id_value(str(recipe.get("h_max_sigma_mode", "robust")).lower())
        if _has_value(recipe, "maxima_min_distance_nm"):
            min_distance_text = f"{_id_value(recipe['maxima_min_distance_nm'])}nm"
        else:
            min_distance_text = _id_value(recipe["maxima_neighborhood"])
        return f"hmax_{h_mode}_h{h_text}_n{min_distance_text}_{smooth_text}_{bg_text}"
    if _has_value(recipe, "sigma_nm"):
        sigma_text = f"{_id_value(recipe['sigma_nm'])}nm"
    else:
        sigma_text = _id_value(recipe["sigma_value"])
    if _has_value(recipe, "maxima_min_distance_nm"):
        min_distance_text = f"{_id_value(recipe['maxima_min_distance_nm'])}nm"
    else:
        min_distance_text = _id_value(recipe["maxima_neighborhood"])
    thresh_text = _id_value(recipe["threshold_value"])
    return f"{method_text}_s{sigma_text}_n{min_distance_text}_t{thresh_text}_{smooth_text}_{bg_text}"


def _stage2_recipe_id(stage1_recipe: dict[str, Any], recipe: dict[str, Any]) -> str:
    base = str(stage1_recipe["recipe_id"])
    fit_text = _id_value(str(recipe.get("fit_method", "fit")).lower())
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
    if method in {"hmax", "h_maxima"}:
        method = "h_max"

    payload: dict[str, Any] = {
        "xy_spacing_nm": recipe.get("xy_spacing_nm"),
        "z_spacing_nm": recipe.get("z_spacing_nm"),
        "norm_enabled": bool(recipe.get("norm_enabled", True)),
        "norm_method": recipe.get("norm_method"),
        "preproc_enabled": bool(recipe.get("preproc_enabled", True)),
        "background_enabled": bool(recipe.get("background_enabled", False)),
        "maxima_method": method,
    }
    if _has_value(recipe, "maxima_min_distance_nm"):
        payload["maxima_min_distance_nm"] = recipe.get("maxima_min_distance_nm")
    else:
        payload["maxima_neighborhood"] = int(recipe.get("maxima_neighborhood", 2))
    if payload["preproc_enabled"]:
        payload["preproc_method"] = recipe.get("preproc_method")
        if _has_value(recipe, "preproc_sigma_nm"):
            payload["preproc_sigma_nm"] = recipe.get("preproc_sigma_nm")
        else:
            payload["preproc_sigma"] = recipe.get("preproc_sigma")
    if payload["background_enabled"]:
        payload["background_method"] = recipe.get("background_method")
        if _has_value(recipe, "background_param_nm"):
            payload["background_param_nm"] = recipe.get("background_param_nm")
        else:
            payload["background_param"] = recipe.get("background_param")
        payload["background_clip"] = bool(recipe.get("background_clip", True))
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
                "threshold_value": recipe.get("threshold_value"),
            }
        )
        if _has_value(recipe, "sigma_nm"):
            payload["sigma_nm"] = recipe.get("sigma_nm")
        else:
            payload["sigma_value"] = recipe.get("sigma_value")
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
    rows: list[dict[str, Any]] = []
    stage1_rows = cfg.get("stage1_recipes", [])
    for base_row in stage1_rows:
        for stage1_row in _expand_stage1_processing_rows(base_row, cfg):
            rows.append(deepcopy(stage1_row))
    recipes: list[dict[str, Any]] = []
    for row in rows:
        stage1_row = deepcopy(row)
        stage1_row.pop("feature_pack_name", None)
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
    pack_names = cfg.get("stage2_feature_packs", ["core_fit"])
    recipes: list[dict[str, Any]] = []
    for stage1_recipe in stage1_recipes:
        stage1_recipes_for_packs: list[dict[str, Any]] = []
        for pack_name in pack_names:
            recipe = _apply_feature_pack(deepcopy(stage1_recipe), str(pack_name))
            recipe["stage1_recipe_id"] = stage1_recipe["recipe_id"]
            recipe["recipe_id"] = _stage2_recipe_id(stage1_recipe, recipe)
            stage1_recipes_for_packs.append(recipe)
        feature_cache_features: list[str] = []
        seen_features: set[str] = set()
        for recipe in stage1_recipes_for_packs:
            for feature in recipe.get("selected_features", []):
                if feature in seen_features:
                    continue
                seen_features.add(feature)
                feature_cache_features.append(feature)
        for recipe in stage1_recipes_for_packs:
            recipe["feature_cache_features"] = feature_cache_features
            recipes.append(recipe)
    _assert_unique_recipe_ids(recipes, context="Stage 2")
    return recipes
