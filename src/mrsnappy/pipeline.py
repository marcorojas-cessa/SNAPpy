from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .detection import detect_candidates
from .features import feature_table
from .fitting import refine_candidates
from .io import read_points_csv, read_volume, split_images, split_pairs, write_points_csv
from .model import AcceptAllCandidatesModel, TrainedModel, fit_svm_pipeline, iter_svm_param_grid, load_model, save_model
from .preprocess import apply_preprocessing, apply_processing_base, apply_smoothing


SVM_THRESHOLD_QUANTILES: tuple[float, ...] = (
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
_STAGE1_CACHE: "OrderedDict[str, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_STAGE1_CACHE_CONFIG: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_DEFAULT_STAGE1_CACHE_ENTRIES = 128
_PROCESSING_BASE_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_PROCESSING_BASE_CACHE_CONFIG: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_PREPROCESS_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_PREPROCESS_CACHE_CONFIG: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_DEFAULT_IMAGE_VOLUME_CACHE_ENTRIES = 96
_FIT_CACHE: "OrderedDict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]" = OrderedDict()
_FIT_CACHE_CONFIG: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_DEFAULT_FIT_CACHE_ENTRIES = 512
_LABEL_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_DEFAULT_LABEL_CACHE_ENTRIES = 4096
_PROCESSING_BASE_KEY_FIELDS = (
    "xy_spacing_nm",
    "z_spacing_nm",
    "norm_enabled",
    "norm_method",
    "background_enabled",
    "background_method",
    "background_param",
    "background_param_nm",
    "background_clip",
)
_PREPROCESS_KEY_FIELDS = (
    *_PROCESSING_BASE_KEY_FIELDS,
    "preproc_enabled",
    "preproc_method",
    "preproc_sigma",
    "preproc_sigma_nm",
)
_STAGE1_KEY_FIELDS = (
    *_PREPROCESS_KEY_FIELDS,
    "maxima_method",
    "maxima_neighborhood",
    "maxima_min_distance_nm",
    "sigma_value",
    "sigma_nm",
    "threshold_value",
    "h_max_sigma_multiplier",
    "h_max_sigma_mode",
    "max_candidates",
)
_FIT_KEY_FIELDS = (
    *_STAGE1_KEY_FIELDS,
    "prefit_prune_enabled",
    "prefit_rank_radius",
    "prefit_rank_bg_width",
    "prefit_nms_distance",
    "prefit_labeled_candidates_per_label",
    "prefit_labeled_min_candidates",
    "prefit_unlabeled_candidates_per_expected_label",
    "expected_labels_per_image",
    "full_fit_labeled_candidates_per_label",
    "full_fit_labeled_min_candidates",
    "full_fit_unlabeled_candidates_per_expected_label",
    "fit_method",
    "fit_window",
    "fit_background_width",
    "fit_max_iterations",
    "fit_tolerance",
    "fit_fallback_method",
    "feature_cache_features",
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _match_spacing_array(match_spacing_nm: tuple[float, ...] | None, ndim: int) -> np.ndarray | None:
    if match_spacing_nm is None:
        return None
    spacing = np.asarray(match_spacing_nm, dtype=np.float32)
    if spacing.shape != (ndim,):
        raise ValueError(f"match_spacing_nm must have {ndim} entries for {ndim}D coordinates.")
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("match_spacing_nm must contain positive finite values.")
    return spacing


def pairwise_distances(
    a: np.ndarray,
    b: np.ndarray,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)), dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    spacing = _match_spacing_array(match_spacing_nm, diff.shape[2])
    if spacing is not None:
        diff = diff * spacing
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def _candidate_bounds(center: np.ndarray, shape: tuple[int, ...], radius: int) -> tuple[slice, ...]:
    bounds: list[slice] = []
    for axis, value in enumerate(center.astype(int)):
        lo = max(int(value) - radius, 0)
        hi = min(int(value) + radius + 1, shape[axis])
        bounds.append(slice(lo, hi))
    return tuple(bounds)


def _greedy_match_assignment(
    pred: np.ndarray,
    gt: np.ndarray,
    distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> dict[int, int]:
    """One-to-one nearest-neighbor assignment within the configured radius."""
    if len(pred) == 0:
        return {}
    if len(gt) == 0:
        return {}
    pred_arr = np.asarray(pred, dtype=np.float32)
    gt_arr = np.asarray(gt, dtype=np.float32)
    spacing = _match_spacing_array(match_spacing_nm, pred_arr.shape[1])
    if spacing is not None:
        pred_arr = pred_arr * spacing
        gt_arr = gt_arr * spacing
    radius = float(distance)
    tree = cKDTree(gt_arr.astype(np.float64, copy=False))
    candidate_gt_pairs: list[tuple[float, int, int]] = []
    for pred_id, point in enumerate(pred_arr):
        for gt_id in tree.query_ball_point(point.astype(np.float64, copy=False), r=radius + 1e-6):
            delta = point - gt_arr[int(gt_id)]
            dist = float(np.sqrt(np.sum(delta * delta)))
            if dist <= radius:
                candidate_gt_pairs.append((dist, int(pred_id), int(gt_id)))
    candidate_gt_pairs = sorted(
        candidate_gt_pairs,
        key=lambda item: (item[0], item[1], item[2]),
    )
    assignment: dict[int, int] = {}
    used_gt: set[int] = set()
    for _, pred_id, gt_id in candidate_gt_pairs:
        if pred_id in assignment or gt_id in used_gt:
            continue
        assignment[pred_id] = gt_id
        used_gt.add(gt_id)
    return assignment


def match_points(
    pred: np.ndarray,
    gt: np.ndarray,
    distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> tuple[int, int, int]:
    assignment = _greedy_match_assignment(pred, gt, distance, match_spacing_nm=match_spacing_nm)
    tp = int(len(assignment))
    fp = int(len(pred) - tp)
    fn = int(len(gt) - tp)
    return tp, fp, fn


def match_points_with_assignment(
    pred: np.ndarray,
    gt: np.ndarray,
    distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> tuple[int, int, int, dict[int, int]]:
    assignment = _greedy_match_assignment(pred, gt, distance, match_spacing_nm=match_spacing_nm)
    tp = int(len(assignment))
    fp = int(len(pred) - tp)
    fn = int(len(gt) - tp)
    return tp, fp, fn, assignment


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _image_selection_metrics(tp: int, fp: int, fn: int) -> dict[str, float | bool]:
    """Per-image validation metrics with explicit empty-GT handling.

    Empty-GT images do not have defined recall. They contribute F1=1 only
    when there are no detections, and F1=0 when any false positives are present.
    """
    gt_count = int(tp + fn)
    pred_count = int(tp + fp)
    has_gt = gt_count > 0
    if not has_gt:
        no_false_positive = int(fp) == 0
        return {
            "precision": 1.0 if no_false_positive else 0.0,
            "recall": 0.0,
            "recall_defined": False,
            "f1": 1.0 if no_false_positive else 0.0,
        }
    precision = float(tp / pred_count) if pred_count else 0.0
    recall = float(tp / gt_count)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "recall_defined": True, "f1": f1}


def _stage1_signature(pipeline_cfg: dict[str, Any]) -> str:
    payload = {key: pipeline_cfg.get(key) for key in _STAGE1_KEY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _preprocess_signature(pipeline_cfg: dict[str, Any]) -> str:
    payload = {key: pipeline_cfg.get(key) for key in _PREPROCESS_KEY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _processing_base_signature(pipeline_cfg: dict[str, Any]) -> str:
    payload = {key: pipeline_cfg.get(key) for key in _PROCESSING_BASE_KEY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _fit_signature(pipeline_cfg: dict[str, Any], label_count: int | None) -> str:
    payload = {key: pipeline_cfg.get(key) for key in _FIT_KEY_FIELDS}
    if pipeline_cfg.get("feature_cache_features") is None:
        payload["feature_cache_features"] = (
            list(pipeline_cfg.get("selected_features") or [])
            if "selected_features" in pipeline_cfg
            else "__all__"
        )
    payload["label_count"] = label_count
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _stage1_cache_limit(pipeline_cfg: dict[str, Any]) -> int:
    return max(int(pipeline_cfg.get("stage1_cache_entries", _DEFAULT_STAGE1_CACHE_ENTRIES)), 0)


def _image_volume_cache_limit(pipeline_cfg: dict[str, Any]) -> int:
    return max(int(pipeline_cfg.get("image_volume_cache_entries", _DEFAULT_IMAGE_VOLUME_CACHE_ENTRIES)), 0)


def _fit_cache_limit(pipeline_cfg: dict[str, Any]) -> int:
    return max(int(pipeline_cfg.get("fit_cache_entries", _DEFAULT_FIT_CACHE_ENTRIES)), 0)


def clear_pipeline_caches() -> None:
    """Clear in-memory caches used during optimizer sweeps."""
    _PROCESSING_BASE_CACHE.clear()
    _PROCESSING_BASE_CACHE_CONFIG.clear()
    _PREPROCESS_CACHE.clear()
    _PREPROCESS_CACHE_CONFIG.clear()
    _STAGE1_CACHE.clear()
    _STAGE1_CACHE_CONFIG.clear()
    _FIT_CACHE.clear()
    _FIT_CACHE_CONFIG.clear()
    _LABEL_CACHE.clear()


def stage1_cache_signature(pipeline_cfg: dict[str, Any]) -> str:
    """Return the candidate-cache signature for a Stage 1 recipe."""
    return _stage1_signature(pipeline_cfg)


def _label_cache_key(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
        marker = f"{stat.st_mtime_ns}:{stat.st_size}"
    except FileNotFoundError:
        marker = "missing"
    return f"{resolved}::{marker}"


def _image_cache_key(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
        marker = f"{stat.st_mtime_ns}:{stat.st_size}"
    except FileNotFoundError:
        marker = "missing"
    return f"{resolved}::{marker}"


def _stage1_cache_key(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> tuple[str, str, str]:
    image_key = _image_cache_key(image_path)
    signature = _stage1_signature(pipeline_cfg)
    return f"{image_key}::{signature}", image_key, signature


def _processing_base_cache_key(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> tuple[str, str]:
    signature = _processing_base_signature(pipeline_cfg)
    return f"{_image_cache_key(image_path)}::{signature}", signature


def _preprocess_cache_key(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> tuple[str, str]:
    signature = _preprocess_signature(pipeline_cfg)
    return f"{_image_cache_key(image_path)}::{signature}", signature


def prune_stage1_candidate_cache(
    *,
    allowed_signatures: set[str],
    image_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> int:
    """Keep only Stage 1 candidate-cache entries useful for the current optimizer leaderboard."""
    allowed_image_paths = None
    if image_paths is not None:
        allowed_image_paths = {str(Path(path).resolve()) for path in image_paths}
    removed = 0
    for key in list(_STAGE1_CACHE):
        meta = _STAGE1_CACHE_CONFIG.get(key, {})
        keep_signature = str(meta.get("signature", "")) in allowed_signatures
        keep_image = allowed_image_paths is None or str(meta.get("image_path", "")) in allowed_image_paths
        if keep_signature and keep_image:
            continue
        _STAGE1_CACHE.pop(key, None)
        _STAGE1_CACHE_CONFIG.pop(key, None)
        removed += 1
    return removed


def promote_stage1_candidate_cache(
    *,
    source_recipe: dict[str, Any],
    target_recipe: dict[str, Any],
    image_paths: list[str | Path] | tuple[str | Path, ...],
) -> int:
    """Alias preflight Stage 1 candidate caches to the Stage 2 recipe key."""
    promoted = 0
    for image_path in image_paths:
        source_key, _, source_signature = _stage1_cache_key(image_path, source_recipe)
        cached = _STAGE1_CACHE.get(source_key)
        if cached is None:
            continue
        target_key, _, target_signature = _stage1_cache_key(image_path, target_recipe)
        _STAGE1_CACHE[target_key] = cached
        _STAGE1_CACHE.move_to_end(target_key)
        _STAGE1_CACHE_CONFIG[target_key] = {
            "image_path": str(Path(image_path).resolve()),
            "signature": target_signature,
            "promoted_from_signature": source_signature,
        }
        promoted += 1
    return promoted


def promote_stage1_image_volume_cache(
    *,
    recipe: dict[str, Any],
    image_paths: list[str | Path] | tuple[str | Path, ...],
) -> int:
    """Move useful Stage 1 image-volume cache entries to the recent end of their LRU caches."""
    promoted = 0
    for image_path in image_paths:
        base_key, _ = _processing_base_cache_key(image_path, recipe)
        if base_key in _PROCESSING_BASE_CACHE:
            _PROCESSING_BASE_CACHE.move_to_end(base_key)
            promoted += 1
        preprocess_key, _ = _preprocess_cache_key(image_path, recipe)
        if preprocess_key in _PREPROCESS_CACHE:
            _PREPROCESS_CACHE.move_to_end(preprocess_key)
            promoted += 1
    return promoted


def _load_processing_base(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> np.ndarray:
    image_path = Path(image_path).resolve()
    cache_key, signature = _processing_base_cache_key(image_path, pipeline_cfg)
    cached = _PROCESSING_BASE_CACHE.get(cache_key)
    if cached is not None:
        _PROCESSING_BASE_CACHE.move_to_end(cache_key)
        return cached

    payload = np.asarray(apply_processing_base(read_volume(image_path), pipeline_cfg), dtype=np.float32)
    _PROCESSING_BASE_CACHE[cache_key] = payload
    _PROCESSING_BASE_CACHE_CONFIG[cache_key] = {"image_path": str(image_path), "signature": signature}
    limit = _image_volume_cache_limit(pipeline_cfg)
    while limit >= 0 and len(_PROCESSING_BASE_CACHE) > limit:
        oldest_key, _ = _PROCESSING_BASE_CACHE.popitem(last=False)
        _PROCESSING_BASE_CACHE_CONFIG.pop(oldest_key, None)
    return payload


def _load_processed_image(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> np.ndarray:
    image_path = Path(image_path).resolve()
    if not pipeline_cfg.get("stage1_cache_enabled", True):
        return apply_preprocessing(read_volume(image_path), pipeline_cfg)

    cache_key, signature = _preprocess_cache_key(image_path, pipeline_cfg)
    cached = _PREPROCESS_CACHE.get(cache_key)
    if cached is not None:
        _PREPROCESS_CACHE.move_to_end(cache_key)
        return cached

    base = _load_processing_base(image_path, pipeline_cfg)
    payload = np.asarray(apply_smoothing(base, pipeline_cfg), dtype=np.float32)
    _PREPROCESS_CACHE[cache_key] = payload
    _PREPROCESS_CACHE_CONFIG[cache_key] = {"image_path": str(image_path), "signature": signature}
    limit = _image_volume_cache_limit(pipeline_cfg)
    while limit >= 0 and len(_PREPROCESS_CACHE) > limit:
        oldest_key, _ = _PREPROCESS_CACHE.popitem(last=False)
        _PREPROCESS_CACHE_CONFIG.pop(oldest_key, None)
    return payload


def _read_points_cached(path: str | Path) -> np.ndarray:
    key = _label_cache_key(path)
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        _LABEL_CACHE.move_to_end(key)
        return cached.copy()
    points = np.asarray(read_points_csv(path), dtype=np.float32)
    _LABEL_CACHE[key] = points
    while len(_LABEL_CACHE) > _DEFAULT_LABEL_CACHE_ENTRIES:
        _LABEL_CACHE.popitem(last=False)
    return points.copy()


def _store_stage1_cache(
    image_path: str | Path,
    cache_key: str,
    signature: str,
    payload: tuple[np.ndarray, np.ndarray],
    pipeline_cfg: dict[str, Any],
) -> None:
    _STAGE1_CACHE[cache_key] = payload
    _STAGE1_CACHE_CONFIG[cache_key] = {"image_path": str(Path(image_path).resolve()), "signature": signature}
    limit = _stage1_cache_limit(pipeline_cfg)
    while limit >= 0 and len(_STAGE1_CACHE) > limit:
        oldest_key, _ = _STAGE1_CACHE.popitem(last=False)
        _STAGE1_CACHE_CONFIG.pop(oldest_key, None)


def _load_stage1_candidates(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image_path = Path(image_path).resolve()
    if not pipeline_cfg.get("stage1_cache_enabled", True):
        volume = _load_processed_image(image_path, pipeline_cfg)
        det = detect_candidates(volume, pipeline_cfg)
        return np.asarray(det.coords, dtype=np.float32), np.asarray(det.scores, dtype=np.float32)

    cache_key, _, signature = _stage1_cache_key(image_path, pipeline_cfg)
    cached = _STAGE1_CACHE.get(cache_key)
    if cached is not None:
        _STAGE1_CACHE.move_to_end(cache_key)
        return cached[0].copy(), cached[1].copy()

    volume = _load_processed_image(image_path, pipeline_cfg)
    det = detect_candidates(volume, pipeline_cfg)
    payload = (np.asarray(det.coords, dtype=np.float32), np.asarray(det.scores, dtype=np.float32))
    _store_stage1_cache(image_path, cache_key, signature, payload, pipeline_cfg)
    return payload[0].copy(), payload[1].copy()


def _load_stage1(image_path: str | Path, pipeline_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = _load_processed_image(image_path, pipeline_cfg)
    coords, scores = _load_stage1_candidates(image_path, pipeline_cfg)
    return volume, coords, scores


def _select_cached_features(features: pd.DataFrame, selected_features: list[str] | None) -> pd.DataFrame:
    if selected_features is None:
        return features.copy(deep=True)
    if not selected_features:
        return pd.DataFrame(index=features.index)
    missing = [col for col in selected_features if col not in features.columns]
    if missing:
        raise ValueError(
            "Selected SNAPpy feature(s) are unavailable in the fitted candidate table: "
            f"{', '.join(missing)}. Use a compatible feature pack or update selected_features."
        )
    return features[selected_features].copy()


def _copy_fit_payload(
    payload: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    selected_features: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    coords, scores, features = payload
    return coords.copy(), scores.copy(), _select_cached_features(features, selected_features)


def preflight_image(
    image_path: str | Path,
    label_path: str | Path,
    recipe: dict[str, Any],
    match_distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    coords, _ = _load_stage1_candidates(image_path, recipe)
    gt = _read_points_cached(label_path)
    tp, fp, fn = match_points(coords, gt, match_distance, match_spacing_nm=match_spacing_nm)
    metrics = precision_recall_f1(tp, fp, fn)
    return {
        "n_candidates": int(len(coords)),
        "n_labels": int(len(gt)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def _candidate_limit_for_fit(pipeline_cfg: dict[str, Any], label_count: int | None = None) -> int | None:
    if not bool(pipeline_cfg.get("prefit_prune_enabled", False)):
        return None
    min_candidates = int(pipeline_cfg.get("prefit_labeled_min_candidates", 0) or 0)
    if label_count is not None:
        multiplier = pipeline_cfg.get("prefit_labeled_candidates_per_label")
        if multiplier is None:
            return None
        return max(min_candidates, int(math.ceil(float(label_count) * float(multiplier))))
    expected = pipeline_cfg.get("expected_labels_per_image")
    multiplier = pipeline_cfg.get("prefit_unlabeled_candidates_per_expected_label")
    if expected is None or multiplier is None:
        return None
    return max(min_candidates, int(math.ceil(float(expected) * float(multiplier))))


def _full_fit_limit_for_candidates(pipeline_cfg: dict[str, Any], label_count: int | None = None) -> int | None:
    min_candidates = int(pipeline_cfg.get("full_fit_labeled_min_candidates", 0) or 0)
    if label_count is not None:
        multiplier = pipeline_cfg.get("full_fit_labeled_candidates_per_label")
        if multiplier is None:
            return None
        return max(min_candidates, int(math.ceil(float(label_count) * float(multiplier))))
    expected = pipeline_cfg.get("expected_labels_per_image")
    multiplier = pipeline_cfg.get("full_fit_unlabeled_candidates_per_expected_label")
    if expected is None or multiplier is None:
        return None
    return max(min_candidates, int(math.ceil(float(expected) * float(multiplier))))


def _candidate_rank_scores(
    volume: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    pipeline_cfg: dict[str, Any],
) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0,), dtype=np.float32)
    rank_radius = max(int(pipeline_cfg.get("prefit_rank_radius", 1) or 1), 1)
    bg_width = max(int(pipeline_cfg.get("prefit_rank_bg_width", 1) or 1), 1)
    ranking = np.zeros(len(coords), dtype=np.float32)
    for idx, coord in enumerate(coords):
        center = np.rint(coord).astype(int)
        win_slc = _candidate_bounds(center, volume.shape, rank_radius)
        bg_slc = _candidate_bounds(center, volume.shape, rank_radius + bg_width)
        patch = np.asarray(volume[win_slc], dtype=np.float32)
        surrounding = np.asarray(volume[bg_slc], dtype=np.float32)
        mask = np.ones(surrounding.shape, dtype=bool)
        inner_slices = []
        for win_axis, bg_axis in zip(win_slc, bg_slc):
            inner_slices.append(slice(win_axis.start - bg_axis.start, win_axis.stop - bg_axis.start))
        mask[tuple(inner_slices)] = False
        peak = float(np.max(patch)) if patch.size else float(scores[idx])
        if np.any(mask):
            bg_vals = surrounding[mask].astype(np.float32, copy=False)
            bg_mean = float(np.mean(bg_vals))
            bg_std = float(np.std(bg_vals))
        else:
            bg_mean = float(np.mean(surrounding)) if surrounding.size else 0.0
            bg_std = float(np.std(surrounding)) if surrounding.size else 1.0
        contrast = max(peak - bg_mean, 0.0)
        snr_proxy = contrast / max(bg_std, 1e-3)
        ranking[idx] = float(max(scores[idx], 0.0) * (1.0 + snr_proxy))
    return ranking


def _greedy_nms(coords: np.ndarray, ranking: np.ndarray, min_distance: float) -> np.ndarray:
    if len(coords) == 0 or min_distance <= 0:
        return np.arange(len(coords), dtype=np.int32)
    keep: list[int] = []
    min_distance_sq = float(min_distance) ** 2
    coords_arr = np.asarray(coords, dtype=np.float32)
    tree = cKDTree(coords_arr.astype(np.float64, copy=False))
    available = np.ones(len(coords_arr), dtype=bool)
    order = np.argsort(ranking)[::-1]
    for idx in order:
        idx = int(idx)
        if not available[idx]:
            continue
        keep.append(idx)
        for neighbor in tree.query_ball_point(coords_arr[idx].astype(np.float64, copy=False), r=float(min_distance) + 1e-6):
            neighbor = int(neighbor)
            if neighbor == idx:
                continue
            delta = coords_arr[neighbor] - coords_arr[idx]
            if float(np.sum(delta * delta)) <= min_distance_sq:
                available[neighbor] = False
    return np.asarray(keep, dtype=np.int32)


def _prune_candidates_for_fit(
    volume: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    pipeline_cfg: dict[str, Any],
    label_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    limit = _candidate_limit_for_fit(pipeline_cfg, label_count=label_count)
    if len(coords) == 0:
        return coords, scores
    if limit is None and float(pipeline_cfg.get("prefit_nms_distance", 0.0) or 0.0) <= 0.0:
        return coords, scores

    ranking = _candidate_rank_scores(volume, coords, scores, pipeline_cfg)
    keep = _greedy_nms(coords, ranking, float(pipeline_cfg.get("prefit_nms_distance", 0.0) or 0.0))
    if limit is not None and len(keep) > int(limit):
        order = np.argsort(ranking[keep])[::-1][: int(limit)]
        keep = keep[order]
    elif limit is not None and len(coords) > int(limit) and len(keep) < int(limit):
        top = np.argsort(ranking)[::-1][: int(limit)]
        keep = top.astype(np.int32)

    keep_order = np.argsort(ranking[keep])[::-1]
    keep = keep[keep_order]
    return coords[keep], scores[keep]


def detect_image(image_path: str | Path, pipeline_cfg: dict[str, Any], label_count: int | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    image_path = Path(image_path).resolve()
    image_key = _image_cache_key(image_path)
    selected_features = list(pipeline_cfg.get("selected_features") or []) if "selected_features" in pipeline_cfg else None
    if pipeline_cfg.get("feature_cache_features") is not None:
        feature_cache_features = list(pipeline_cfg.get("feature_cache_features") or [])
    elif selected_features is not None:
        feature_cache_features = list(selected_features)
    else:
        feature_cache_features = None
    if bool(pipeline_cfg.get("fit_cache_enabled", True)):
        signature = _fit_signature(pipeline_cfg, label_count)
        cache_key = f"{image_key}::{signature}"
        cached = _FIT_CACHE.get(cache_key)
        if cached is not None:
            _FIT_CACHE.move_to_end(cache_key)
            return _copy_fit_payload(cached, selected_features)

    coords, scores = _load_stage1_candidates(image_path, pipeline_cfg)
    feature_volume = _load_processing_base(image_path, pipeline_cfg)
    coords, scores = _prune_candidates_for_fit(feature_volume, coords, scores, pipeline_cfg, label_count=label_count)
    full_fit_limit = _full_fit_limit_for_candidates(pipeline_cfg, label_count=label_count)
    fit_window = int(pipeline_cfg.get("fit_window", 7) or 7)
    if fit_window <= 0 or fit_window % 2 == 0:
        raise ValueError("fit_window must be a positive odd integer.")
    window_radius = max(fit_window // 2, 1)
    fit = refine_candidates(
        feature_volume,
        coords,
        scores,
        window_radius=window_radius,
        fit_method=str(pipeline_cfg.get("fit_method", "moments")),
        fit_cfg=pipeline_cfg,
        full_fit_limit=full_fit_limit,
    )
    all_features = feature_table(
        fit.table,
        feature_cache_features,
        xy_spacing_nm=pipeline_cfg.get("xy_spacing_nm"),
        z_spacing_nm=pipeline_cfg.get("z_spacing_nm"),
    )
    payload = (np.asarray(fit.coords, dtype=np.float32), np.asarray(scores, dtype=np.float32), all_features)
    if bool(pipeline_cfg.get("fit_cache_enabled", True)):
        _FIT_CACHE[cache_key] = payload
        _FIT_CACHE_CONFIG[cache_key] = {"image_path": str(image_path), "signature": signature}
        limit = _fit_cache_limit(pipeline_cfg)
        while limit >= 0 and len(_FIT_CACHE) > limit:
            oldest_key, _ = _FIT_CACHE.popitem(last=False)
            _FIT_CACHE_CONFIG.pop(oldest_key, None)
    return _copy_fit_payload(payload, selected_features)


def _label_candidates(
    coords: np.ndarray,
    gt: np.ndarray,
    distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0,), dtype=np.int32)
    labels = np.zeros(len(coords), dtype=np.int32)
    for candidate_id in _greedy_match_assignment(coords, gt, distance, match_spacing_nm=match_spacing_nm):
        labels[candidate_id] = 1
    return labels


def summarize_stage1_candidate_labels(
    dataset_root: str | Path,
    split: str,
    recipe: dict[str, Any],
    match_distance: float,
    image_limit: int | None = None,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    pairs = split_pairs(dataset_root, split)
    if image_limit is not None:
        pairs = pairs[: int(image_limit)]
    n_candidates = 0
    n_positive = 0
    n_negative = 0
    n_gt = 0
    for image_path, label_path in pairs:
        gt = _read_points_cached(label_path)
        coords, _ = _load_stage1_candidates(image_path, recipe)
        labels = _label_candidates(coords, gt, match_distance, match_spacing_nm=match_spacing_nm)
        positives = int(np.sum(labels == 1))
        n_candidates += int(len(coords))
        n_positive += positives
        n_negative += int(len(coords) - positives)
        n_gt += int(len(gt))
    return {
        "split": split,
        "n_images": int(len(pairs)),
        "n_ground_truth": int(n_gt),
        "n_candidates": int(n_candidates),
        "n_positive_candidates": int(n_positive),
        "n_negative_candidates": int(n_negative),
        "all_positive_candidates": bool(n_candidates > 0 and n_negative == 0),
        "no_training_candidates": bool(n_candidates == 0),
        "no_true_positive_candidates": bool(n_candidates > 0 and n_positive == 0),
    }


def build_training_matrices(
    dataset_root: str | Path,
    split: str,
    recipe: dict[str, Any],
    match_distance: float,
    image_limit: int | None = None,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    pairs = split_pairs(dataset_root, split)
    if image_limit is not None:
        pairs = pairs[: int(image_limit)]
    for image_path, label_path in pairs:
        gt = _read_points_cached(label_path)
        coords, scores, feats = detect_image(image_path, recipe, label_count=int(len(gt)))
        y = _label_candidates(coords, gt, match_distance, match_spacing_nm=match_spacing_nm)
        feats = feats.copy()
        selected = recipe.get("selected_features")
        if "score_raw" not in feats.columns and (selected is None or "score_raw" in selected):
            feats["score_raw"] = scores
        rows.append(feats)
        labels.append(y)
        metas.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "coords": coords,
                "scores": scores,
                "features": feats,
                "labels": y,
                "gt": gt,
            }
        )
    if not rows:
        raise RuntimeError(f"No labeled image pairs found for split '{split}' under {dataset_root}")
    x = pd.concat(rows, ignore_index=True)
    y = np.concatenate(labels).astype(np.int32)
    return x, y, metas


def _model_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float32)
    return np.asarray(model.predict(features), dtype=np.float32)


def _predict_metas_scored(
    metas: list[dict[str, Any]],
    model: Any,
    selected_features: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    preds: dict[str, np.ndarray] = {}
    scores_map: dict[str, np.ndarray] = {}
    gts: dict[str, np.ndarray] = {}
    for meta in metas:
        image_stem = Path(str(meta["image_path"])).stem
        coords = np.asarray(meta["coords"], dtype=np.float32)
        if len(coords) == 0:
            preds[image_stem] = np.empty((0, 3), dtype=np.float32)
            scores_map[image_stem] = np.empty((0,), dtype=np.float32)
        else:
            features = meta["features"][selected_features].to_numpy(dtype=np.float32)
            preds[image_stem] = coords
            scores_map[image_stem] = _model_scores(model, features)
        gts[image_stem] = np.asarray(meta["gt"], dtype=np.float32)
    return preds, scores_map, gts


def tune_score_threshold(
    preds: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    match_distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> tuple[float, dict[str, float]]:
    all_scores: list[float] = []
    for arr in scores.values():
        if len(arr):
            all_scores.extend(float(x) for x in arr)
    if not all_scores:
        metrics = evaluate_predictions_for_selection(preds, gts, match_distance, match_spacing_nm=match_spacing_nm)
        return 0.0, metrics
    score_values = np.asarray(all_scores, dtype=np.float32)
    quantiles = np.quantile(score_values, SVM_THRESHOLD_QUANTILES).tolist()
    candidates = sorted(set([float(np.min(score_values)) - 1e-6, 0.0, *[float(q) for q in quantiles]]))
    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_key = (-1.0, float("-inf"))
    for threshold in candidates:
        filtered = apply_score_threshold(preds, scores, threshold)
        metrics = evaluate_predictions_for_selection(filtered, gts, match_distance, match_spacing_nm=match_spacing_nm)
        key = (
            float(metrics["f1_mean_image"]),
            -abs(float(threshold)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _svm_complexity_key(params: dict[str, Any]) -> tuple[float, float, float, float]:
    kernel = str(params.get("kernel", "linear")).lower()
    kernel_rank = {"linear": 0.0, "rbf": 1.0, "polynomial": 2.0, "poly": 2.0}.get(kernel, 3.0)
    c_value = float(params.get("C", 1.0))
    c_rank = abs(math.log10(max(c_value, 1e-12)))
    degree = float(params.get("degree", 2)) if kernel in {"polynomial", "poly"} else 0.0
    gamma = params.get("gamma")
    if gamma is None or str(gamma).strip().lower() == "auto":
        gamma_rank = 0.0
    elif str(gamma).strip().lower() == "scale":
        gamma_rank = 1.0
    else:
        gamma_rank = 2.0 + abs(math.log10(max(float(gamma), 1e-12)))
    return (kernel_rank, c_rank, degree, gamma_rank)


def train_native_model(
    dataset_root: str | Path,
    recipe: dict[str, Any],
    svm_cfg: dict[str, Any],
    model_path: str | Path,
    match_distance: float,
    train_limit: int | None = None,
    val_limit: int | None = None,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> TrainedModel:
    if str(recipe.get("model_type", "")).strip().lower() == "stage1_pass_through":
        if match_spacing_nm is None:
            _, _, val_metas = build_training_matrices(dataset_root, "val", recipe, match_distance, val_limit)
        else:
            _, _, val_metas = build_training_matrices(
                dataset_root,
                "val",
                recipe,
                match_distance,
                val_limit,
                match_spacing_nm=match_spacing_nm,
            )
        pass_through_model = AcceptAllCandidatesModel()
        selected_features: list[str] = []
        val_preds_raw, _, val_gts = _predict_metas_scored(val_metas, pass_through_model, selected_features)
        val_metrics = evaluate_predictions_for_selection(val_preds_raw, val_gts, match_distance, match_spacing_nm=match_spacing_nm)
        best_params = {
            "model_type": "stage1_pass_through",
            "decision_rule": "accept_all_stage1_candidates",
            "stage1_train_label_status": recipe.get("stage1_train_label_status", "all_positive_stage1"),
            **{f"val_{metric_name}": float(metric_value) for metric_name, metric_value in val_metrics.items()},
        }
        trained = TrainedModel(
            model=pass_through_model,
            selected_features=selected_features,
            best_params=best_params,
            decision_threshold=0.0,
            recipe=dict(recipe),
            validation_metas=val_metas,
        )
        save_model(model_path, trained)
        return trained

    if match_spacing_nm is None:
        x_train, y_train, _ = build_training_matrices(dataset_root, "train", recipe, match_distance, train_limit)
        _, _, val_metas = build_training_matrices(dataset_root, "val", recipe, match_distance, val_limit)
    else:
        x_train, y_train, _ = build_training_matrices(
            dataset_root,
            "train",
            recipe,
            match_distance,
            train_limit,
            match_spacing_nm=match_spacing_nm,
        )
        _, _, val_metas = build_training_matrices(
            dataset_root,
            "val",
            recipe,
            match_distance,
            val_limit,
            match_spacing_nm=match_spacing_nm,
        )
    if len(y_train) == 0 or x_train.empty:
        recipe_id = recipe.get("recipe_id", "<unknown>")
        raise ValueError(
            "SNAPpy Stage 2 cannot train because the Stage 1 recipe "
            f"{recipe_id} generated no training candidates."
        )
    selected_features = list(recipe.get("selected_features") or list(x_train.columns))
    x_train_np = x_train[selected_features].to_numpy(dtype=np.float32)
    unique_train_labels = np.unique(np.asarray(y_train, dtype=np.int8))
    if len(unique_train_labels) == 1:
        only_label = int(unique_train_labels[0])
        if only_label == 0:
            recipe_id = recipe.get("recipe_id", "<unknown>")
            raise ValueError(
                "SNAPpy Stage 2 cannot train because the Stage 1 recipe "
                f"{recipe_id} generated no true-positive training candidates."
            )
        pass_through_model = AcceptAllCandidatesModel()
        val_preds_raw, val_scores, val_gts = _predict_metas_scored(val_metas, pass_through_model, selected_features)
        val_metrics = evaluate_predictions_for_selection(val_preds_raw, val_gts, match_distance, match_spacing_nm=match_spacing_nm)
        best_params = {
            "model_type": "stage1_pass_through",
            "one_class_label": only_label,
            "decision_rule": "accept_all_stage1_candidates",
            **{f"val_{metric_name}": float(metric_value) for metric_name, metric_value in val_metrics.items()},
        }
        trained = TrainedModel(
            model=pass_through_model,
            selected_features=selected_features,
            best_params=best_params,
            decision_threshold=0.0,
            recipe=dict(recipe),
            validation_metas=val_metas,
        )
        save_model(model_path, trained)
        return trained

    best_model: Any | None = None
    best_threshold = 0.0
    best_params: dict[str, Any] = {}
    best_key = (-1.0, 0.0, 0.0, 0.0, 0.0)

    for params in iter_svm_param_grid(svm_cfg):
        model = fit_svm_pipeline(x_train_np, y_train, params)
        val_preds_raw, val_scores, val_gts = _predict_metas_scored(val_metas, model, selected_features)
        threshold, val_metrics = tune_score_threshold(
            val_preds_raw,
            val_scores,
            val_gts,
            match_distance,
            match_spacing_nm=match_spacing_nm,
        )
        key = (
            float(val_metrics["f1_mean_image"]),
            -_svm_complexity_key(params)[0],
            -_svm_complexity_key(params)[1],
            -_svm_complexity_key(params)[2],
            -_svm_complexity_key(params)[3],
        )
        if key > best_key:
            best_key = key
            best_model = model
            best_threshold = float(threshold)
            best_params = {
                "svm_selection_method": "train_fit_val_select",
                "svm_selection_split": "val",
                **params,
                **{f"val_{metric_name}": float(metric_value) for metric_name, metric_value in val_metrics.items()},
            }

    if best_model is None:
        raise RuntimeError("Native SVM training failed to produce a valid model.")

    trained = TrainedModel(
        model=best_model,
        selected_features=selected_features,
        best_params=best_params,
        decision_threshold=best_threshold,
        recipe=dict(recipe),
        validation_metas=val_metas,
    )
    save_model(model_path, trained)
    return trained


def predict_image(
    image_path: str | Path,
    recipe: dict[str, Any] | None,
    model_path: str | Path,
    score_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    trained = load_model(model_path)
    if recipe is None:
        recipe = trained.recipe
    if recipe is None:
        raise ValueError("Prediction requires a pipeline recipe, either embedded in the model or supplied by config.")
    coords, _, feats = detect_image(image_path, recipe)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    features = feats[trained.selected_features].to_numpy(dtype=np.float32)
    model = trained.model
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float32)
    else:
        scores = np.asarray(model.predict(features), dtype=np.float32)
    if score_threshold is None:
        score_threshold = trained.decision_threshold
    keep = scores > float(score_threshold)
    return coords[keep], scores[keep]


def predict_image_scored(image_path: str | Path, recipe: dict[str, Any] | None, model_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    trained = load_model(model_path)
    if recipe is None:
        recipe = trained.recipe
    if recipe is None:
        raise ValueError("Prediction requires a pipeline recipe, either embedded in the model or supplied by config.")
    coords, _, feats = detect_image(image_path, recipe)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    features = feats[trained.selected_features].to_numpy(dtype=np.float32)
    model = trained.model
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float32)
    else:
        scores = np.asarray(model.predict(features), dtype=np.float32)
    return coords, scores


def predict_split(
    dataset_root: str | Path,
    split: str,
    recipe: dict[str, Any] | None,
    model_path: str | Path,
    output_root: str | Path,
    score_threshold: float | None = None,
    image_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    output_root = ensure_dir(output_root)
    preds: dict[str, np.ndarray] = {}
    gts: dict[str, np.ndarray] = {}
    pairs = split_pairs(dataset_root, split)
    if image_limit is not None:
        pairs = pairs[: int(image_limit)]
    for image_path, label_path in pairs:
        coords, scores = predict_image(image_path, recipe, model_path, score_threshold=score_threshold)
        out_path = output_root / f"{image_path.stem}.csv"
        write_points_csv(out_path, coords, scores)
        preds[image_path.stem] = coords
        gts[image_path.stem] = _read_points_cached(label_path)
    return preds, gts


def predict_unlabeled_split(
    dataset_root: str | Path,
    split: str,
    recipe: dict[str, Any] | None,
    model_path: str | Path,
    output_root: str | Path,
    score_threshold: float | None = None,
    image_limit: int | None = None,
) -> dict[str, np.ndarray]:
    output_root = ensure_dir(output_root)
    preds: dict[str, np.ndarray] = {}
    image_paths = split_images(dataset_root, split)
    if image_limit is not None:
        image_paths = image_paths[: int(image_limit)]
    for image_path in image_paths:
        coords, scores = predict_image(image_path, recipe, model_path, score_threshold=score_threshold)
        out_path = output_root / f"{image_path.stem}.csv"
        write_points_csv(out_path, coords, scores)
        preds[image_path.stem] = coords
    return preds


def predict_split_scored(
    dataset_root: str | Path,
    split: str,
    recipe: dict[str, Any] | None,
    model_path: str | Path,
    image_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    preds: dict[str, np.ndarray] = {}
    scores_map: dict[str, np.ndarray] = {}
    gts: dict[str, np.ndarray] = {}
    pairs = split_pairs(dataset_root, split)
    if image_limit is not None:
        pairs = pairs[: int(image_limit)]
    for image_path, label_path in pairs:
        coords, scores = predict_image_scored(image_path, recipe, model_path)
        preds[image_path.stem] = coords
        scores_map[image_path.stem] = scores
        gts[image_path.stem] = _read_points_cached(label_path)
    return preds, scores_map, gts


def apply_score_threshold(preds: dict[str, np.ndarray], scores: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, coords in preds.items():
        score = scores.get(key, np.empty((0,), dtype=np.float32))
        keep = score > float(threshold)
        out[key] = coords[keep]
    return out


def evaluate_predictions(
    preds: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    match_distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> dict[str, float]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for key in sorted(gts):
        pred = preds.get(key, np.empty((0, 3), dtype=np.float32))
        gt = gts[key]
        tp, fp, fn = match_points(pred, gt, match_distance, match_spacing_nm=match_spacing_nm)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
    metrics = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    return {
        "tp": int(totals["tp"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def evaluate_predictions_for_selection(
    preds: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    match_distance: float,
    match_spacing_nm: tuple[float, ...] | None = None,
) -> dict[str, float | int]:
    """Evaluate detections with image-mean metrics as the Stage 2 selection target.

    Recall is averaged only over images with at least one ground-truth point.
    Empty-GT images still affect F1 and precision: no detections is counted as
    correct absence, while any detections are false positives.
    """
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_image_metrics: list[dict[str, float | bool]] = []
    for key in sorted(gts):
        pred = preds.get(key, np.empty((0, 3), dtype=np.float32))
        gt = gts[key]
        tp, fp, fn = match_points(pred, gt, match_distance, match_spacing_nm=match_spacing_nm)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        metrics = _image_selection_metrics(tp, fp, fn)
        per_image_metrics.append(metrics)

    pooled = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    n_images = max(len(per_image_metrics), 1)
    recall_rows = [row for row in per_image_metrics if bool(row["recall_defined"])]
    precision_mean_image = float(sum(float(row["precision"]) for row in per_image_metrics) / n_images)
    recall_mean_image = (
        float(sum(float(row["recall"]) for row in recall_rows) / len(recall_rows))
        if recall_rows
        else 0.0
    )
    f1_mean_image = float(sum(float(row["f1"]) for row in per_image_metrics) / n_images)
    return {
        "tp": int(totals["tp"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
        "precision": precision_mean_image,
        "recall": recall_mean_image,
        "f1": f1_mean_image,
        "precision_mean_image": precision_mean_image,
        "recall_mean_image": recall_mean_image,
        "f1_mean_image": f1_mean_image,
        "precision_pooled": pooled["precision"],
        "recall_pooled": pooled["recall"],
        "f1_pooled": pooled["f1"],
        "n_images": int(len(per_image_metrics)),
        "n_labeled_images": int(len(recall_rows)),
        "n_empty_gt_images": int(len(per_image_metrics) - len(recall_rows)),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False))
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value
