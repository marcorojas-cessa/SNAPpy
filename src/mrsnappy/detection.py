from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima

from .spatial import physical_nms_indices, sigma_nm_to_voxels, spacing_zyx_nm


_MAD_TO_SIGMA = 0.6744897501960817
_SCALE_EPS = 1e-6


@dataclass
class DetectionResult:
    coords: np.ndarray
    scores: np.ndarray
    response: np.ndarray


def _finite_values(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32).ravel()
    return values[np.isfinite(values)]


def _noise_sigma(volume: np.ndarray, mode: str) -> float:
    values = _finite_values(volume)
    if values.size == 0:
        return 0.0
    mode = str(mode).strip().lower()
    if mode in {"std", "stdev", "standard_deviation", "standard deviation"}:
        return float(np.std(values))
    if mode not in {"robust", "mad", "median_absolute_deviation", "median absolute deviation"}:
        raise ValueError(f"Unsupported h_max_sigma_mode: {mode}. Expected 'robust' or 'std'.")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return float(mad / 0.6744897501960817)
    return float(np.std(values))


def _robust_z_score(volume: np.ndarray) -> np.ndarray:
    out = np.asarray(volume, dtype=np.float32)
    values = _finite_values(out)
    if values.size == 0:
        return np.zeros_like(out, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = mad / _MAD_TO_SIGMA
    if scale <= _SCALE_EPS:
        scale = float(np.std(values))
    if scale <= _SCALE_EPS:
        return np.zeros_like(out, dtype=np.float32)
    return np.asarray((out - median) / scale, dtype=np.float32)


def _peak_local_max_kwargs(max_candidates: object) -> dict[str, int]:
    if max_candidates is None:
        return {}
    cap = max(int(max_candidates), 1)
    return {"num_peaks": max(cap * 4, cap + 32)}


def _has_value(cfg: dict, key: str) -> bool:
    return key in cfg and cfg.get(key) is not None


def _finalize_candidates(
    coords: np.ndarray,
    scores: np.ndarray,
    cfg: dict,
    *,
    ndim: int,
    max_candidates: object,
) -> tuple[np.ndarray, np.ndarray]:
    if coords.size == 0:
        return np.empty((0, ndim), dtype=np.float32), np.empty((0,), dtype=np.float32)

    order = np.argsort(scores)[::-1]
    coords = coords[order].astype(np.float32)
    scores = scores[order].astype(np.float32)
    candidate_cap = None if max_candidates is None else max(int(max_candidates), 1)
    if _has_value(cfg, "maxima_min_distance_nm"):
        keep = physical_nms_indices(
            coords,
            scores,
            spacing_nm=spacing_zyx_nm(cfg, ndim),
            min_distance_nm=float(cfg["maxima_min_distance_nm"]),
            max_candidates=candidate_cap,
        )
        return coords[keep].astype(np.float32), scores[keep].astype(np.float32)
    if candidate_cap is not None:
        coords = coords[:candidate_cap]
        scores = scores[:candidate_cap]
    return coords.astype(np.float32), scores.astype(np.float32)


def _h_max_value(response: np.ndarray, cfg: dict) -> float:
    unsupported = [key for key in ("h_max_value", "h_max_min_abs", "h_max_min_sigma_multiplier") if key in cfg]
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"Unsupported h-max parameter(s): {joined}. Use h_max_sigma_multiplier only.")
    if cfg.get("h_max_sigma_multiplier") is None:
        raise ValueError("h-max detection requires h_max_sigma_multiplier.")
    sigma = _noise_sigma(response, str(cfg.get("h_max_sigma_mode", "robust")))
    return max(float(cfg["h_max_sigma_multiplier"]) * sigma, 0.0)


def detect_candidates(volume: np.ndarray, cfg: dict) -> DetectionResult:
    method = str(cfg.get("maxima_method", "log")).lower()
    max_candidates = cfg.get("max_candidates")
    physical_nms = _has_value(cfg, "maxima_min_distance_nm")
    if _has_value(cfg, "maxima_neighborhood"):
        peak_min_distance = max(int(cfg["maxima_neighborhood"]), 1)
    else:
        peak_min_distance = 1 if physical_nms else max(int(cfg.get("maxima_neighborhood", 2)), 1)
    peak_kwargs = _peak_local_max_kwargs(max_candidates) if physical_nms else {}
    if max_candidates is not None and not physical_nms:
        peak_kwargs["num_peaks"] = max(int(max_candidates), 1)

    if method in {"log", "laplacian_of_gaussian", "laplacian of gaussian"}:
        if _has_value(cfg, "sigma_nm"):
            sigma_nm = float(cfg["sigma_nm"])
            sigma = sigma_nm_to_voxels(sigma_nm, cfg, volume.ndim, "sigma_nm")
            response = -ndi.gaussian_laplace(volume, sigma=sigma) * (sigma_nm ** 2)
        else:
            sigma = float(cfg.get("sigma_value", 1.35))
            response = -ndi.gaussian_laplace(volume, sigma=sigma) * (sigma ** 2)
        response = _robust_z_score(response)
        threshold_value = cfg.get("threshold_value", 0.1)
        if threshold_value is None:
            raise ValueError("LoG detection requires numeric threshold_value.")
        threshold = float(threshold_value)
        coords = peak_local_max(
            response,
            min_distance=peak_min_distance,
            threshold_abs=threshold,
            exclude_border=False,
            **peak_kwargs,
        )
        if coords.size == 0:
            coords = np.empty((0, volume.ndim), dtype=np.int32)
            scores = np.empty((0,), dtype=np.float32)
        else:
            scores = response[tuple(coords.T)]
    elif method in {"hmax", "h_max", "h-max", "h_maxima", "h-maxima"}:
        response = volume
        h_value = _h_max_value(response, cfg)
        if h_value <= _SCALE_EPS:
            return DetectionResult(
                coords=np.empty((0, volume.ndim), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                response=np.asarray(response, dtype=np.float32),
            )
        mask = h_maxima(response, h_value)
        coords = peak_local_max(
            response,
            min_distance=peak_min_distance,
            exclude_border=False,
            labels=mask.astype(np.uint8),
            **peak_kwargs,
        )
        if coords.size == 0:
            coords = np.empty((0, volume.ndim), dtype=np.int32)
            scores = np.empty((0,), dtype=np.float32)
        else:
            scores = response[tuple(coords.T)]
    else:
        raise ValueError(f"Unsupported maxima method: {cfg.get('maxima_method')}. Expected 'log' or 'h_max'.")

    coords, scores = _finalize_candidates(
        coords,
        scores,
        cfg,
        ndim=volume.ndim,
        max_candidates=max_candidates,
    )
    return DetectionResult(coords=coords, scores=scores, response=np.asarray(response, dtype=np.float32))
