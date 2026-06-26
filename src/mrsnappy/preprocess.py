from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, opening
from skimage.restoration import rolling_ball

from .spatial import radius_nm_to_voxels, sigma_nm_to_voxels


_MAD_TO_SIGMA = 0.6744897501960817
_SCALE_EPS = 1e-6


def _has_value(cfg: dict, key: str) -> bool:
    return key in cfg and cfg.get(key) is not None


def _background_method_name(value: object) -> str:
    text = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "none",
        "none": "none",
        "slice_opening_2d": "slice_opening_2d",
        "rolling_ball_2d": "rolling_ball_2d",
        "rollingball_2d": "rolling_ball_2d",
        "slice_rolling_ball": "rolling_ball_2d",
        "slice_wise_rolling_ball": "rolling_ball_2d",
        "rolling_ball_3d": "rolling_ball_3d",
        "rolling_ball_3d_exact": "rolling_ball_3d",
        "exact_3d_rolling_ball": "rolling_ball_3d",
        "rolling_box_3d": "rolling_box_3d",
        "morph_opening_3d_box": "rolling_box_3d",
        "3d_box_opening": "rolling_box_3d",
        "box_opening_3d": "rolling_box_3d",
        "scipy_3d_box": "rolling_box_3d",
    }
    return aliases.get(text, text)


def _normalization_method_name(value: object) -> str:
    if value is None:
        value = "robust_z_score"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "robust": "robust_z_score",
        "robust_z": "robust_z_score",
        "robust_zscore": "robust_z_score",
        "robust_z_score": "robust_z_score",
        "mad": "robust_z_score",
        "median_absolute_deviation": "robust_z_score",
    }
    return aliases.get(text, text)


def _rolling_ball_background(volume: np.ndarray, radius: float) -> np.ndarray:
    try:
        return rolling_ball(volume, radius=radius, workers=-1)
    except TypeError:
        try:
            return rolling_ball(volume, radius=radius, num_threads=-1)
        except TypeError:
            return rolling_ball(volume, radius=radius)


def _apply_preprocessing_filter(volume: np.ndarray, cfg: dict) -> np.ndarray:
    out = volume
    if not cfg.get("preproc_enabled", True):
        return out
    method = str(cfg.get("preproc_method", "gaussian")).strip().lower()
    if method in {"none", ""}:
        return out
    if method != "gaussian":
        raise ValueError("SNAPpy Stage 1 smoothing supports only 'none' or 3D Gaussian smoothing.")
    if _has_value(cfg, "preproc_sigma_nm"):
        sigma = sigma_nm_to_voxels(cfg["preproc_sigma_nm"], cfg, out.ndim, "preproc_sigma_nm")
    else:
        sigma = float(cfg.get("preproc_sigma", 0.5))
    out = ndi.gaussian_filter(out, sigma=sigma)
    return np.asarray(out, dtype=np.float32, copy=False)


def _apply_normalization(volume: np.ndarray, cfg: dict) -> np.ndarray:
    out = np.asarray(volume, dtype=np.float32)
    method = _normalization_method_name(cfg.get("norm_method", "robust_z_score"))
    if not cfg.get("norm_enabled", True) or method == "none":
        return out
    if method != "robust_z_score":
        raise ValueError("SNAPpy Stage 1 normalization supports only 'none' or robust z-score normalization.")
    median = float(np.median(out))
    mad = float(np.median(np.abs(out - median)))
    robust_sigma = mad / _MAD_TO_SIGMA
    if robust_sigma <= _SCALE_EPS:
        robust_sigma = float(np.std(out))
    if robust_sigma <= _SCALE_EPS:
        return np.zeros_like(out, dtype=np.float32)
    return np.asarray((out - median) / robust_sigma, dtype=np.float32)


def _apply_background_correction(volume: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg.get("background_enabled", False):
        return volume
    method = _background_method_name(cfg.get("background_method", "none"))
    if method in {"none", ""}:
        return volume
    if method not in {"slice_opening_2d", "rolling_ball_2d", "rolling_ball_3d", "rolling_box_3d"}:
        raise ValueError(
            "SNAPpy Stage 1 background correction supports 'none', 'slice_opening_2d', "
            "'rolling_ball_2d', 'rolling_ball_3d', or 'rolling_box_3d'."
        )
    out = np.asarray(volume, dtype=np.float32, copy=False)
    physical_radius = _has_value(cfg, "background_param_nm")
    if physical_radius:
        radius_by_axis = radius_nm_to_voxels(cfg["background_param_nm"], cfg, out.ndim, "background_param_nm")
    else:
        radius = max(int(round(float(cfg.get("background_param", 5.0)))), 1)
        radius_by_axis = (radius,) * out.ndim
    radius_xy = float(radius_by_axis[-1])
    if method == "slice_opening_2d":
        footprint = disk(max(int(round(radius_xy)), 1))
        if out.ndim == 3:
            background = np.empty_like(out, dtype=np.float32)
            for z_index in range(out.shape[0]):
                background[z_index] = opening(out[z_index], footprint=footprint)
        else:
            background = opening(out, footprint=footprint)
    elif method == "rolling_ball_2d":
        if out.ndim == 3:
            background = np.empty_like(out, dtype=np.float32)
            for z_index in range(out.shape[0]):
                background[z_index] = _rolling_ball_background(out[z_index], radius_xy)
        else:
            background = _rolling_ball_background(out, radius_xy)
    elif method == "rolling_box_3d":
        size_by_axis = tuple(int(radius) * 2 + 1 for radius in radius_by_axis)
        if out.ndim == 3:
            background = ndi.grey_opening(out, size=size_by_axis)
        else:
            background = ndi.grey_opening(out, size=size_by_axis)
    else:
        if physical_radius and out.ndim == 3 and len(set(radius_by_axis)) != 1:
            raise ValueError("rolling_ball_3d does not support anisotropic physical spacing; use rolling_box_3d.")
        background = _rolling_ball_background(out, radius_xy)
    out = out - np.asarray(background, dtype=np.float32)
    if cfg.get("background_clip", True):
        out = np.maximum(out, 0.0)
    return np.asarray(out, dtype=np.float32, copy=False)


def apply_processing_base(volume: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply processing steps that are independent of detector smoothing."""
    out = np.asarray(volume, dtype=np.float32)
    out = _apply_background_correction(out, cfg)
    out = _apply_normalization(out, cfg)
    return np.asarray(out, dtype=np.float32, copy=False)


def apply_smoothing(volume: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply the final detector smoothing step."""
    out = np.asarray(volume, dtype=np.float32)
    out = _apply_preprocessing_filter(out, cfg)
    return np.asarray(out, dtype=np.float32, copy=False)


def apply_preprocessing(volume: np.ndarray, cfg: dict) -> np.ndarray:
    """Prepare the detector image: background correction, normalization, smoothing."""
    out = apply_processing_base(volume, cfg)
    out = apply_smoothing(out, cfg)
    return np.asarray(out, dtype=np.float32, copy=False)
