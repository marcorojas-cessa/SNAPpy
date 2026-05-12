from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, opening
from skimage.restoration import rolling_ball


_MAD_TO_SIGMA = 0.6744897501960817
_SCALE_EPS = 1e-6


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
    out = ndi.gaussian_filter(out, sigma=float(cfg.get("preproc_sigma", 0.5)))
    return np.asarray(out, dtype=np.float32, copy=False)


def _apply_normalization(volume: np.ndarray, cfg: dict) -> np.ndarray:
    out = np.asarray(volume, dtype=np.float32)
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
    param = float(cfg.get("background_param", 5.0))
    out = np.asarray(volume, dtype=np.float32, copy=False)
    radius = max(float(param), 1.0)
    if method == "slice_opening_2d":
        footprint = disk(max(int(round(radius)), 1))
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
                background[z_index] = _rolling_ball_background(out[z_index], radius)
        else:
            background = _rolling_ball_background(out, radius)
    elif method == "rolling_box_3d":
        size = max(int(round(radius)), 1) * 2 + 1
        if out.ndim == 3:
            background = ndi.grey_opening(out, size=(size, size, size))
        else:
            background = ndi.grey_opening(out, size=(size, size))
    else:
        background = _rolling_ball_background(out, radius)
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
    """Prepare the detector image using background correction before scaling."""
    out = apply_processing_base(volume, cfg)
    out = apply_smoothing(out, cfg)
    return np.asarray(out, dtype=np.float32, copy=False)
