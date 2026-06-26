from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _positive_float(value: Any, key: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{key} must be a positive finite value.")
    return out


def spacing_zyx_nm(cfg: dict, ndim: int) -> tuple[float, ...]:
    """Return physical axis spacing in array-axis order."""
    xy_spacing_nm = _positive_float(cfg.get("xy_spacing_nm"), "xy_spacing_nm")
    if ndim == 2:
        return (xy_spacing_nm, xy_spacing_nm)
    if ndim == 3:
        z_spacing_nm = _positive_float(cfg.get("z_spacing_nm"), "z_spacing_nm")
        return (z_spacing_nm, xy_spacing_nm, xy_spacing_nm)
    raise ValueError(f"SNAPpy physical spacing supports 2D or 3D arrays, not ndim={ndim}.")


def sigma_nm_to_voxels(sigma_nm: Any, cfg: dict, ndim: int, key: str) -> tuple[float, ...]:
    sigma = _positive_float(sigma_nm, key)
    spacing = spacing_zyx_nm(cfg, ndim)
    return tuple(float(sigma / axis_spacing) for axis_spacing in spacing)


def radius_nm_to_voxels(radius_nm: Any, cfg: dict, ndim: int, key: str) -> tuple[int, ...]:
    radius = _positive_float(radius_nm, key)
    spacing = spacing_zyx_nm(cfg, ndim)
    return tuple(max(int(np.ceil(radius / axis_spacing)), 1) for axis_spacing in spacing)


def physical_nms_indices(
    coords: np.ndarray,
    scores: np.ndarray,
    *,
    spacing_nm: tuple[float, ...],
    min_distance_nm: float,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Score-ordered non-maximum suppression using real physical distances."""
    if len(coords) == 0:
        return np.empty((0,), dtype=np.int64)
    if min_distance_nm <= 0.0:
        keep = np.arange(len(coords), dtype=np.int64)
        return keep if max_candidates is None else keep[:max_candidates]

    scaled = np.asarray(coords, dtype=np.float64) * np.asarray(spacing_nm, dtype=np.float64)
    keep: list[int] = []
    tree = cKDTree(scaled)
    suppressed = np.zeros(len(scaled), dtype=bool)
    for index, point in enumerate(scaled):
        if suppressed[index]:
            continue
        keep.append(index)
        neighbors = tree.query_ball_point(point, r=float(min_distance_nm))
        suppressed[np.asarray(neighbors, dtype=np.int64)] = True
        if max_candidates is not None and len(keep) >= int(max_candidates):
            break
    return np.asarray(keep, dtype=np.int64)
