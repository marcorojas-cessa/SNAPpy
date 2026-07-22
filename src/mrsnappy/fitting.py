from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import Delaunay, QhullError


@dataclass
class FitResult:
    coords: np.ndarray
    table: list[dict[str, float | str]]


CONTRAST_FEATURE_NAMES = frozenset(
    {
        "core_mean",
        "core_std",
        "shell_mean",
        "shell_std",
        "core_minus_shell",
        "core_shell_snr",
        "xy_core_minus_shell",
        "z_core_minus_shell",
        "halfspace_absdiff_max",
    }
)

MORPHOLOGY_FEATURE_NAMES = frozenset(
    {
        "component_voxel_volume",
        "component_surface_area_vox2",
        "component_surface_to_volume_ratio",
        "component_sphericity_3d",
        "component_convex_voxel_volume",
        "component_solidity_3d",
        "component_elongation_3d",
        "component_pixel_area",
        "component_boundary_px",
        "component_boundary_to_area_ratio",
        "component_circularity_2d",
        "component_convex_size_px",
        "component_solidity_2d",
        "component_elongation_2d",
        "component_centroid_fit_distance_nm",
    }
)


def _requested_features(cfg: dict[str, Any]) -> set[str] | None:
    """Return requested fit-time features, or None when all features are requested."""
    if cfg.get("feature_cache_features") is not None:
        return {str(feature) for feature in (cfg.get("feature_cache_features") or [])}
    if cfg.get("selected_features") is not None:
        return {str(feature) for feature in (cfg.get("selected_features") or [])}
    return None


def _needs_feature_group(requested: set[str] | None, group: frozenset[str]) -> bool:
    return requested is None or bool(requested & group)


def _bounds(center: np.ndarray, shape: tuple[int, ...], radius: int) -> tuple[slice, ...]:
    bounds: list[slice] = []
    for axis, value in enumerate(center.astype(int)):
        lo = max(value - radius, 0)
        hi = min(value + radius + 1, shape[axis])
        bounds.append(slice(lo, hi))
    return tuple(bounds)


def _window_info(center: np.ndarray, slc: tuple[slice, ...]) -> np.ndarray:
    offset = np.array([s.start for s in slc], dtype=np.float32)
    return center.astype(np.float32) - offset + 1.0


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_total)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return out


def _patch_background(
    window_data: np.ndarray,
    surrounding_data: np.ndarray,
    window_slices: tuple[slice, ...],
    surrounding_slices: tuple[slice, ...],
    cfg: dict[str, float | int | str],
) -> tuple[np.ndarray, float, float]:
    mask = np.ones(surrounding_data.shape, dtype=bool)
    inner_slices = []
    for win_slc, bg_slc in zip(window_slices, surrounding_slices):
        inner_slices.append(slice(win_slc.start - bg_slc.start, win_slc.stop - bg_slc.start))
    mask[tuple(inner_slices)] = False

    if not np.any(mask):
        bg = float(np.mean(surrounding_data))
        noise = float(np.std(surrounding_data))
        return np.asarray(window_data, dtype=np.float32) - bg, bg, noise

    background_vals = np.asarray(surrounding_data[mask], dtype=np.float64)
    perimeter_mean = float(np.mean(background_vals))
    perimeter_std = float(np.std(background_vals))
    return np.asarray(window_data, dtype=np.float32) - perimeter_mean, perimeter_mean, perimeter_std


def _gaussian_1d_model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp, mu, sigma = p
    sigma = max(float(sigma), 1e-3)
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _fit_1d(profile: np.ndarray, center_guess: float, max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    x = np.arange(1, len(profile) + 1, dtype=np.float64)
    profile = profile.astype(np.float64, copy=False)
    idx = int(np.clip(round(center_guess) - 1, 0, len(profile) - 1))
    amp_guess = max(float(profile[idx]), 1e-3)
    sigma_guess = max(len(profile) / 4.0, 0.1)
    p0 = np.array([amp_guess, float(center_guess), sigma_guess], dtype=np.float64)
    lb = np.array([0.0, float(np.min(x)), 0.1], dtype=np.float64)
    ub = np.array([max(float(np.max(profile)), amp_guess, 1.0) * 1.5, float(np.max(x)), float(len(profile))], dtype=np.float64)
    res = least_squares(
        lambda p: _gaussian_1d_model(p, x) - profile,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    fit = res.x.astype(np.float32)
    return fit, _r_squared(profile, _gaussian_1d_model(fit, x))


def _gaussian_2d_distorted(p: np.ndarray, coords: np.ndarray) -> np.ndarray:
    amp, mu_y, mu_x, sigma_y, sigma_x, rho_yx = p
    sigma_y = max(float(sigma_y), 1e-3)
    sigma_x = max(float(sigma_x), 1e-3)
    rho_yx = float(np.clip(rho_yx, -0.99, 0.99))
    cov = np.array(
        [
            [sigma_y**2, rho_yx * sigma_y * sigma_x],
            [rho_yx * sigma_y * sigma_x, sigma_x**2],
        ],
        dtype=np.float64,
    )
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.zeros(coords.shape[0], dtype=np.float64)
    delta = coords - np.array([mu_y, mu_x], dtype=np.float64)
    quad = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    quad = np.clip(quad, 0.0, 700.0)
    return amp * np.exp(-0.5 * quad)


def _fit_2d_distorted(data: np.ndarray, center_guess: tuple[float, float], max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    rows, cols = data.shape
    coords = _coords_for_shape((rows, cols))
    values = data.astype(np.float64, copy=False).ravel()
    y_idx = int(np.clip(round(center_guess[0]) - 1, 0, rows - 1))
    x_idx = int(np.clip(round(center_guess[1]) - 1, 0, cols - 1))
    amp_guess = max(float(data[y_idx, x_idx]), 1e-3)
    p0 = np.array([amp_guess, float(center_guess[0]), float(center_guess[1]), rows / 4.0, cols / 4.0, 0.0], dtype=np.float64)
    lb = np.array([0.0, 1.0, 1.0, 0.1, 0.1, -0.99], dtype=np.float64)
    ub = np.array([max(float(np.max(data)), amp_guess, 1.0) * 1.5, float(rows), float(cols), float(rows), float(cols), 0.99], dtype=np.float64)
    res = least_squares(
        lambda p: _gaussian_2d_distorted(p, coords) - values,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    fit = res.x.astype(np.float32)
    return fit, _r_squared(values, _gaussian_2d_distorted(fit, coords))


def _gaussian_2d_axis_aligned(p: np.ndarray, coords: np.ndarray) -> np.ndarray:
    amp, mu_y, mu_x, sigma_y, sigma_x = p
    sigma_y = max(float(sigma_y), 1e-3)
    sigma_x = max(float(sigma_x), 1e-3)
    dy = (coords[:, 0] - mu_y) ** 2 / (2.0 * sigma_y**2)
    dx = (coords[:, 1] - mu_x) ** 2 / (2.0 * sigma_x**2)
    return amp * np.exp(-(dy + dx))


def _fit_2d_axis_aligned(data: np.ndarray, center_guess: tuple[float, float], max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    rows, cols = data.shape
    coords = _coords_for_shape((rows, cols))
    values = data.astype(np.float64, copy=False).ravel()
    y_idx = int(np.clip(round(center_guess[0]) - 1, 0, rows - 1))
    x_idx = int(np.clip(round(center_guess[1]) - 1, 0, cols - 1))
    amp_guess = max(float(data[y_idx, x_idx]), 1e-3)
    p0 = np.array([amp_guess, float(center_guess[0]), float(center_guess[1]), rows / 4.0, cols / 4.0], dtype=np.float64)
    lb = np.array([0.0, 1.0, 1.0, 0.1, 0.1], dtype=np.float64)
    ub = np.array([max(float(np.max(data)), amp_guess, 1.0) * 1.5, float(rows), float(cols), float(rows), float(cols)], dtype=np.float64)
    res = least_squares(
        lambda p: _gaussian_2d_axis_aligned(p, coords) - values,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    fit = res.x.astype(np.float32)
    return fit, _r_squared(values, _gaussian_2d_axis_aligned(fit, coords))


def _gaussian_3d_axis_aligned(p: np.ndarray, coords: np.ndarray) -> np.ndarray:
    amp, mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x = p
    sigma_z = max(float(sigma_z), 1e-3)
    sigma_y = max(float(sigma_y), 1e-3)
    sigma_x = max(float(sigma_x), 1e-3)
    dz = (coords[:, 0] - mu_z) ** 2 / (2.0 * sigma_z**2)
    dy = (coords[:, 1] - mu_y) ** 2 / (2.0 * sigma_y**2)
    dx = (coords[:, 2] - mu_x) ** 2 / (2.0 * sigma_x**2)
    return amp * np.exp(-(dz + dy + dx))


def _fit_3d(data: np.ndarray, center_guess: tuple[float, float, float], max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    coords = _coords_for_shape(tuple(data.shape))
    values = data.astype(np.float64, copy=False).ravel()
    z_idx = int(np.clip(round(center_guess[0]) - 1, 0, data.shape[0] - 1))
    y_idx = int(np.clip(round(center_guess[1]) - 1, 0, data.shape[1] - 1))
    x_idx = int(np.clip(round(center_guess[2]) - 1, 0, data.shape[2] - 1))
    amp_guess = max(float(data[z_idx, y_idx, x_idx]), 1e-3)
    p0 = np.array(
        [amp_guess, float(center_guess[0]), float(center_guess[1]), float(center_guess[2]), data.shape[0] / 4.0, data.shape[1] / 4.0, data.shape[2] / 4.0],
        dtype=np.float64,
    )
    lb = np.array([0.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1], dtype=np.float64)
    ub = np.array(
        [max(float(np.max(data)), amp_guess, 1.0) * 1.5, float(data.shape[0]), float(data.shape[1]), float(data.shape[2]), float(data.shape[0]), float(data.shape[1]), float(data.shape[2])],
        dtype=np.float64,
    )
    res = least_squares(
        lambda p: _gaussian_3d_axis_aligned(p, coords) - values,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    fit = res.x.astype(np.float32)
    return fit, _r_squared(values, _gaussian_3d_axis_aligned(fit, coords))


def _gaussian_3d_distorted(p: np.ndarray, coords: np.ndarray) -> np.ndarray:
    amp = float(p[0])
    mu = np.asarray(p[1:4], dtype=np.float64)
    sig = np.clip(np.asarray(p[4:7], dtype=np.float64), 1e-3, None)
    rho = np.clip(np.asarray(p[7:10], dtype=np.float64), -0.99, 0.99)
    cov = np.array(
        [
            [sig[0] ** 2, rho[0] * sig[0] * sig[1], rho[1] * sig[0] * sig[2]],
            [rho[0] * sig[0] * sig[1], sig[1] ** 2, rho[2] * sig[1] * sig[2]],
            [rho[1] * sig[0] * sig[2], rho[2] * sig[1] * sig[2], sig[2] ** 2],
        ],
        dtype=np.float64,
    )
    try:
        np.linalg.cholesky(cov)
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.zeros(coords.shape[0], dtype=np.float64)
    delta = coords - mu
    quad = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    quad = np.clip(quad, 0.0, 700.0)
    return amp * np.exp(-0.5 * quad)


def _fit_3d_distorted(data: np.ndarray, center_guess: tuple[float, float, float], max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    coords = _coords_for_shape(tuple(data.shape))
    values = data.astype(np.float64, copy=False).ravel()
    z_idx = int(np.clip(round(center_guess[0]) - 1, 0, data.shape[0] - 1))
    y_idx = int(np.clip(round(center_guess[1]) - 1, 0, data.shape[1] - 1))
    x_idx = int(np.clip(round(center_guess[2]) - 1, 0, data.shape[2] - 1))
    amp_guess = max(float(data[z_idx, y_idx, x_idx]), 1e-3)
    p0 = np.array(
        [amp_guess, float(center_guess[0]), float(center_guess[1]), float(center_guess[2]), data.shape[0] / 4.0, data.shape[1] / 4.0, data.shape[2] / 4.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    lb = np.array([0.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, -1.0, -1.0, -1.0], dtype=np.float64)
    ub = np.array(
        [max(float(np.max(data)), amp_guess, 1.0) * 1.5, float(data.shape[0]), float(data.shape[1]), float(data.shape[2]), float(data.shape[0]), float(data.shape[1]), float(data.shape[2]), 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    res = least_squares(
        lambda p: _gaussian_3d_distorted(p, coords) - values,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    fit = res.x.astype(np.float32)
    return fit, _r_squared(values, _gaussian_3d_distorted(fit, coords))


@lru_cache(maxsize=64)
def _coords_for_shape(shape: tuple[int, ...]) -> np.ndarray:
    return np.indices(shape, dtype=np.float64).reshape(len(shape), -1).T.astype(np.float64) + 1.0


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _center_index_from_guess(center_guess: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    center = np.rint(np.asarray(center_guess, dtype=np.float64) - 1.0).astype(int)
    upper = np.asarray(shape, dtype=int) - 1
    return np.clip(center, 0, upper)


def _axis_region_diff(data: np.ndarray, axis: int, center_index: np.ndarray) -> float:
    lower = np.take(data, indices=range(0, int(center_index[axis])), axis=axis)
    upper = np.take(data, indices=range(int(center_index[axis]) + 1, data.shape[axis]), axis=axis)
    if lower.size == 0 or upper.size == 0:
        return 0.0
    return float(abs(np.mean(upper) - np.mean(lower)))


def _contrast_features(data: np.ndarray, center_index: np.ndarray) -> dict[str, float]:
    if data.ndim not in {2, 3} or data.size == 0:
        return {name: 0.0 for name in (
            "core_mean",
            "core_std",
            "shell_mean",
            "shell_std",
            "core_minus_shell",
            "core_shell_snr",
            "xy_core_minus_shell",
            "z_core_minus_shell",
            "halfspace_absdiff_max",
        )}

    if data.ndim == 2:
        core_slices = tuple(slice(max(int(c) - 1, 0), min(int(c) + 2, data.shape[axis])) for axis, c in enumerate(center_index))
        core_mask = np.zeros(data.shape, dtype=bool)
        core_mask[core_slices] = True
        core = data[core_mask]
        shell = data[~core_mask]
        core_mean = _safe_mean(core)
        shell_mean = _safe_mean(shell)
        shell_std = _safe_std(shell)
        halfspace_diffs = [
            _axis_region_diff(data, 1, center_index),
            _axis_region_diff(data, 0, center_index),
        ]
        core_minus_shell = float(core_mean - shell_mean)
        return {
            "core_mean": core_mean,
            "core_std": _safe_std(core),
            "shell_mean": shell_mean,
            "shell_std": shell_std,
            "core_minus_shell": core_minus_shell,
            "core_shell_snr": float(core_minus_shell / max(shell_std, 1e-6)),
            "xy_core_minus_shell": core_minus_shell,
            "halfspace_absdiff_max": float(max(halfspace_diffs)) if halfspace_diffs else 0.0,
        }

    core_slices = tuple(slice(max(int(c) - 1, 0), min(int(c) + 2, data.shape[axis])) for axis, c in enumerate(center_index))
    core_mask = np.zeros(data.shape, dtype=bool)
    core_mask[core_slices] = True
    core = data[core_mask]
    shell = data[~core_mask]

    zc, yc, xc = (int(value) for value in center_index)
    central_slice = data[zc]
    xy_mask = np.zeros(central_slice.shape, dtype=bool)
    xy_mask[max(yc - 1, 0) : min(yc + 2, data.shape[1]), max(xc - 1, 0) : min(xc + 2, data.shape[2])] = True
    xy_core = central_slice[xy_mask]
    xy_shell = central_slice[~xy_mask]

    z_mask = np.zeros(data.shape[0], dtype=bool)
    z_mask[max(zc - 1, 0) : min(zc + 2, data.shape[0])] = True
    z_core = data[z_mask, :, :]
    z_shell = data[~z_mask, :, :]

    core_mean = _safe_mean(core)
    shell_mean = _safe_mean(shell)
    shell_std = _safe_std(shell)
    halfspace_diffs = [
        _axis_region_diff(data, 2, center_index),
        _axis_region_diff(data, 1, center_index),
        _axis_region_diff(data, 0, center_index),
    ]
    return {
        "core_mean": core_mean,
        "core_std": _safe_std(core),
        "shell_mean": shell_mean,
        "shell_std": shell_std,
        "core_minus_shell": float(core_mean - shell_mean),
        "core_shell_snr": float((core_mean - shell_mean) / max(shell_std, 1e-6)),
        "xy_core_minus_shell": float(_safe_mean(xy_core) - _safe_mean(xy_shell)),
        "z_core_minus_shell": float(_safe_mean(z_core) - _safe_mean(z_shell)),
        "halfspace_absdiff_max": float(max(halfspace_diffs)),
    }


def _surface_area_vox2(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    surface = 0
    for axis in range(mask.ndim):
        before = [slice(1, -1)] * mask.ndim
        after = [slice(1, -1)] * mask.ndim
        before[axis] = slice(0, -2)
        after[axis] = slice(2, None)
        center = tuple(slice(1, -1) for _ in range(mask.ndim))
        surface += int(np.count_nonzero(padded[center] & ~padded[tuple(before)]))
        surface += int(np.count_nonzero(padded[center] & ~padded[tuple(after)]))
    return float(surface)


def _convex_voxel_volume(mask: np.ndarray) -> float:
    points = np.column_stack(np.nonzero(mask)).astype(np.float64)
    if len(points) < mask.ndim + 1:
        return float(np.count_nonzero(mask))
    try:
        delaunay = Delaunay(points)
        grid_points = np.column_stack(np.indices(mask.shape, dtype=np.float64).reshape(mask.ndim, -1).T)
        inside = delaunay.find_simplex(grid_points) >= 0
    except (QhullError, ValueError, np.linalg.LinAlgError):
        return float(np.count_nonzero(mask))
    return float(max(int(np.count_nonzero(inside)), int(np.count_nonzero(mask))))


def _component_sphericity(volume: float, surface_area: float) -> float:
    if volume <= 0 or surface_area <= 0:
        return 0.0
    return float((np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / max(surface_area, 1e-6))


def _component_circularity(area: float, perimeter: float) -> float:
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / max(perimeter**2, 1e-6))


def _component_elongation(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    centered = points.astype(np.float64) - np.mean(points.astype(np.float64), axis=0)
    cov = np.cov(centered, rowvar=False)
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    axes = np.sqrt(np.clip(eigenvalues, 0.0, None))
    axis_max = float(np.max(axes))
    axis_min = float(np.min(axes))
    if axis_max <= 1e-6:
        return 0.0
    return float(1.0 - axis_min / axis_max)


def _morphology_feature_names(ndim: int) -> tuple[str, ...]:
    if ndim == 2:
        return (
            "component_pixel_area",
            "component_boundary_px",
            "component_boundary_to_area_ratio",
            "component_circularity_2d",
            "component_convex_size_px",
            "component_solidity_2d",
            "component_elongation_2d",
            "component_centroid_fit_distance_nm",
        )
    return (
        "component_voxel_volume",
        "component_surface_area_vox2",
        "component_surface_to_volume_ratio",
        "component_sphericity_3d",
        "component_convex_voxel_volume",
        "component_solidity_3d",
        "component_elongation_3d",
        "component_centroid_fit_distance_nm",
    )


def _morphology_features(
    data: np.ndarray,
    center_index: np.ndarray,
    fitted_center_zero_based: np.ndarray,
    fit_amplitude: float,
    xy_spacing_nm: float,
    z_spacing_nm: float,
) -> dict[str, float]:
    names = _morphology_feature_names(data.ndim)
    if data.ndim not in {2, 3} or data.size == 0 or fit_amplitude <= 0:
        return {name: 0.0 for name in names}

    threshold = 0.5 * float(fit_amplitude)
    mask = np.asarray(data >= threshold, dtype=bool)
    if not np.any(mask):
        return {name: 0.0 for name in names}
    fill_structure = ndi.generate_binary_structure(data.ndim, 1)
    label_structure = ndi.generate_binary_structure(data.ndim, data.ndim)
    filled = ndi.binary_fill_holes(mask, structure=fill_structure)
    labels, n_labels = ndi.label(filled, structure=label_structure)
    if n_labels == 0:
        return {name: 0.0 for name in names}

    center = np.asarray(center_index, dtype=np.float64)
    components: list[tuple[float, float, float, int, np.ndarray, np.ndarray]] = []
    for label_id in range(1, n_labels + 1):
        component = labels == label_id
        points = np.column_stack(np.nonzero(component)).astype(np.float64)
        if len(points) == 0:
            continue
        centroid = np.mean(points, axis=0)
        distance = float(np.linalg.norm(centroid - center))
        volume = float(len(points))
        surface = _surface_area_vox2(component)
        sphericity = _component_circularity(volume, surface) if data.ndim == 2 else _component_sphericity(volume, surface)
        components.append((distance, -volume, -sphericity, label_id, component, centroid))
    if not components:
        return {name: 0.0 for name in names}
    _, neg_volume, _, _, component, centroid = sorted(components, key=lambda item: (item[0], item[1], item[2], item[3]))[0]
    volume = float(-neg_volume)
    surface = _surface_area_vox2(component)
    convex_volume = _convex_voxel_volume(component)
    points = np.column_stack(np.nonzero(component)).astype(np.float64)
    centroid_delta = centroid - np.asarray(fitted_center_zero_based, dtype=np.float64)
    if data.ndim == 2:
        centroid_distance_nm = float(
            np.sqrt(
                (centroid_delta[1] * xy_spacing_nm) ** 2
                + (centroid_delta[0] * xy_spacing_nm) ** 2
            )
        )
        sphericity_value = _component_circularity(volume, surface)
    else:
        centroid_distance_nm = float(
            np.sqrt(
                (centroid_delta[2] * xy_spacing_nm) ** 2
                + (centroid_delta[1] * xy_spacing_nm) ** 2
                + (centroid_delta[0] * z_spacing_nm) ** 2
            )
        )
        sphericity_value = _component_sphericity(volume, surface)
    if data.ndim == 2:
        return {
            "component_pixel_area": volume,
            "component_boundary_px": surface,
            "component_boundary_to_area_ratio": float(surface / max(volume, 1e-6)),
            "component_circularity_2d": sphericity_value,
            "component_convex_size_px": convex_volume,
            "component_solidity_2d": float(volume / max(convex_volume, 1e-6)),
            "component_elongation_2d": _component_elongation(points),
            "component_centroid_fit_distance_nm": centroid_distance_nm,
        }
    return {
        "component_voxel_volume": volume,
        "component_surface_area_vox2": surface,
        "component_surface_to_volume_ratio": float(surface / max(volume, 1e-6)),
        "component_sphericity_3d": sphericity_value,
        "component_convex_voxel_volume": convex_volume,
        "component_solidity_3d": float(volume / max(convex_volume, 1e-6)),
        "component_elongation_3d": _component_elongation(points),
        "component_centroid_fit_distance_nm": centroid_distance_nm,
    }


def _fit_moments_patch(signal: np.ndarray, center_guess: np.ndarray) -> dict[str, Any]:
    positive_signal = np.clip(signal, 0.0, None)
    total = float(np.sum(positive_signal))
    grid = np.indices(signal.shape, dtype=np.float32)
    if total <= 0:
        center = np.asarray(center_guess, dtype=np.float32)
        sigma = np.ones(signal.ndim, dtype=np.float32)
        cov = np.eye(signal.ndim, dtype=np.float32)
    else:
        center_zero_based = np.array([(grid[a] * positive_signal).sum() / total for a in range(signal.ndim)], dtype=np.float32)
        centered = grid.reshape(signal.ndim, -1).T - center_zero_based
        weights = positive_signal.reshape(-1)
        cov = (centered.T * weights) @ centered / (float(weights.sum()) + 1e-8)
        cov = np.asarray(cov, dtype=np.float32)
        sigma = np.sqrt(np.clip(np.diag(cov), 1e-3, None))
        center = center_zero_based + 1.0
    diag = np.clip(np.diag(cov), 1e-6, None)
    return {
        "fit_method_id": "moments",
        "center": center,
        "amplitude": float(np.max(signal)),
        "amplitude_x": float(np.max(signal)),
        "amplitude_y": float(np.max(signal)),
        "amplitude_z": float(np.max(signal)) if signal.ndim == 3 else float("nan"),
        "amplitude_xy": float(np.max(signal)),
        "sigma": sigma if len(sigma) == 3 else np.array([np.nan, sigma[0], sigma[1]], dtype=np.float32),
        "rho": np.array(
            [
                float(cov[1, 2] / np.sqrt(diag[1] * diag[2])) if signal.ndim == 3 else 0.0,
                float(cov[0, 2] / np.sqrt(diag[0] * diag[2])) if signal.ndim == 3 else 0.0,
                float(cov[0, 1] / np.sqrt(diag[0] * diag[1])) if signal.ndim == 3 else 0.0,
            ],
            dtype=np.float32,
        ),
        "r_squared": float(np.max(signal) / max(float(np.max(signal) + np.mean(signal)), 1e-6)),
    }


def _fit_patch(signal: np.ndarray, center_guess: np.ndarray, cfg: dict[str, float | int | str]) -> dict[str, Any]:
    method_key = str(cfg.get("fit_method", "moments")).strip().lower()
    valid_methods = {
        "2d gaussian",
        "gaussian_2d",
        "distorted 2d gaussian",
        "distorted_gaussian_2d",
        "2d (xy) + 1d (z) gaussian",
        "3d gaussian",
        "distorted 3d gaussian",
        "moments",
        "moment",
        "image moments",
    }
    if method_key not in valid_methods:
        raise ValueError(
            "Unsupported fit_method. Expected one of: "
            "2D Gaussian, Distorted 2D Gaussian, "
            "2D (XY) + 1D (Z) Gaussian, 3D Gaussian, Distorted 3D Gaussian, moments."
        )
    max_iterations = _positive_integer(cfg.get("fit_max_iterations", 200), "fit_max_iterations")
    tolerance = _positive_finite_float(cfg.get("fit_tolerance", 1e-6), "fit_tolerance")
    if signal.ndim == 2:
        if method_key in {"moments", "moment", "image moments"}:
            return _fit_moments_patch(signal, center_guess)
        if method_key in {"2d gaussian", "gaussian_2d"}:
            fit2d, r2 = _fit_2d_axis_aligned(signal, (float(center_guess[0]), float(center_guess[1])), max_iterations, tolerance)
            coords = _coords_for_shape(signal.shape)
            model_patch = _gaussian_2d_axis_aligned(fit2d, coords).reshape(signal.shape)
            residual = signal.astype(np.float64, copy=False) - model_patch
            residual_energy = float(np.sum(residual**2))
            signal_energy = float(np.sum(signal.astype(np.float64, copy=False) ** 2))
            return {
                "fit_method_id": "gaussian_2d",
                "center": np.array([fit2d[1], fit2d[2]], dtype=np.float32),
                "amplitude": float(fit2d[0]),
                "amplitude_x": float(fit2d[0]),
                "amplitude_y": float(fit2d[0]),
                "amplitude_z": float("nan"),
                "amplitude_xy": float(fit2d[0]),
                "sigma": np.array([float("nan"), fit2d[3], fit2d[4]], dtype=np.float32),
                "rho": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "r_squared": float(r2),
                "residual_rmse": float(np.sqrt(np.mean(residual**2))),
                "residual_energy_norm": float(residual_energy / max(signal_energy, 1e-6)),
            }
        if method_key not in {"distorted 2d gaussian", "distorted_gaussian_2d"}:
            raise ValueError(
                f"fit_method {cfg.get('fit_method')!r} is incompatible with native 2D images. "
                "Use '2D Gaussian', 'Distorted 2D Gaussian', or 'moments'."
            )
        fit2d, r2 = _fit_2d_distorted(signal, (float(center_guess[0]), float(center_guess[1])), max_iterations, tolerance)
        coords = _coords_for_shape(signal.shape)
        model_patch = _gaussian_2d_distorted(fit2d, coords).reshape(signal.shape)
        residual = signal.astype(np.float64, copy=False) - model_patch
        residual_energy = float(np.sum(residual**2))
        signal_energy = float(np.sum(signal.astype(np.float64, copy=False) ** 2))
        return {
            "fit_method_id": "distorted_gaussian_2d",
            "center": np.array([fit2d[1], fit2d[2]], dtype=np.float32),
            "amplitude": float(fit2d[0]),
            "amplitude_x": float(fit2d[0]),
            "amplitude_y": float(fit2d[0]),
            "amplitude_z": float("nan"),
            "amplitude_xy": float(fit2d[0]),
            "sigma": np.array([float("nan"), fit2d[3], fit2d[4]], dtype=np.float32),
            "rho": np.array([fit2d[5], 0.0, 0.0], dtype=np.float32),
            "r_squared": float(r2),
            "residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "residual_energy_norm": float(residual_energy / max(signal_energy, 1e-6)),
        }
    if method_key in {"2d gaussian", "gaussian_2d", "distorted 2d gaussian", "distorted_gaussian_2d"}:
        raise ValueError(
            f"fit_method {cfg.get('fit_method')!r} is incompatible with native 3D images. "
            "Use '2D (XY) + 1D (Z) Gaussian', '3D Gaussian', 'Distorted 3D Gaussian', or 'moments'."
        )
    if "distorted 3d" in method_key:
        fit3d, r2 = _fit_3d_distorted(signal, (float(center_guess[0]), float(center_guess[1]), float(center_guess[2])), max_iterations, tolerance)
        coords = _coords_for_shape(signal.shape)
        model_patch = _gaussian_3d_distorted(fit3d, coords).reshape(signal.shape)
        residual = signal.astype(np.float64, copy=False) - model_patch
        residual_energy = float(np.sum(residual**2))
        signal_energy = float(np.sum(signal.astype(np.float64, copy=False) ** 2))
        # _gaussian_3d_distorted parameter order is [rho_yz, rho_xz, rho_xy].
        # Store feature rows in named order [rho_xy, rho_xz, rho_yz].
        rho_xy = float(fit3d[9])
        rho_xz = float(fit3d[8])
        rho_yz = float(fit3d[7])
        return {
            "fit_method_id": "distorted_gaussian_3d",
            "center": np.array([fit3d[1], fit3d[2], fit3d[3]], dtype=np.float32),
            "amplitude": float(fit3d[0]),
            "amplitude_x": float("nan"),
            "amplitude_y": float("nan"),
            "amplitude_z": float("nan"),
            "amplitude_xy": float("nan"),
            "sigma": np.array([fit3d[4], fit3d[5], fit3d[6]], dtype=np.float32),
            "rho": np.array([rho_xy, rho_xz, rho_yz], dtype=np.float32),
            "r_squared": float(r2),
            "residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "residual_energy_norm": float(residual_energy / max(signal_energy, 1e-6)),
        }
    if "3d gaussian" in method_key:
        fit3d, r2 = _fit_3d(signal, (float(center_guess[0]), float(center_guess[1]), float(center_guess[2])), max_iterations, tolerance)
        coords = _coords_for_shape(signal.shape)
        model_patch = _gaussian_3d_axis_aligned(fit3d, coords).reshape(signal.shape)
        residual = signal.astype(np.float64, copy=False) - model_patch
        residual_energy = float(np.sum(residual**2))
        signal_energy = float(np.sum(signal.astype(np.float64, copy=False) ** 2))
        return {
            "fit_method_id": "gaussian_3d",
            "center": np.array([fit3d[1], fit3d[2], fit3d[3]], dtype=np.float32),
            "amplitude": float(fit3d[0]),
            "amplitude_x": float("nan"),
            "amplitude_y": float("nan"),
            "amplitude_z": float("nan"),
            "amplitude_xy": float("nan"),
            "sigma": np.array([fit3d[4], fit3d[5], fit3d[6]], dtype=np.float32),
            "rho": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "r_squared": float(r2),
            "residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "residual_energy_norm": float(residual_energy / max(signal_energy, 1e-6)),
        }
    if "2d (xy) + 1d (z)" in method_key and signal.ndim == 3:
        z_profile = signal.max(axis=(1, 2))
        fit_z, r2_z = _fit_1d(z_profile, float(center_guess[0]), max_iterations, tolerance)
        xy_projection = signal.max(axis=0)
        fit_xy, r2_xy = _fit_2d_axis_aligned(xy_projection, (float(center_guess[1]), float(center_guess[2])), max_iterations, tolerance)
        return {
            "fit_method_id": "xy_z_gaussian",
            "center": np.array([fit_z[1], fit_xy[1], fit_xy[2]], dtype=np.float32),
            "amplitude": float(fit_xy[0]),
            "amplitude_x": float(fit_xy[0]),
            "amplitude_y": float(fit_xy[0]),
            "amplitude_z": float(fit_z[0]),
            "amplitude_xy": float(fit_xy[0]),
            "sigma": np.array([fit_z[2], fit_xy[3], fit_xy[4]], dtype=np.float32),
            "rho": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "r_squared": float((r2_xy + r2_z) / 2.0),
        }
    return _fit_moments_patch(signal, center_guess)


def refine_candidates(
    volume: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    window_radius: int = 3,
    fit_method: str = "moments",
    fit_cfg: dict[str, float | int | str] | None = None,
    full_fit_limit: int | None = None,
) -> FitResult:
    if len(coords) == 0:
        return FitResult(coords=np.empty((0, volume.ndim), dtype=np.float32), table=[])

    cfg = dict(fit_cfg or {})
    cfg.setdefault("fit_method", fit_method)
    cfg.setdefault("fit_background_width", 1)
    requested_features = _requested_features(cfg)
    needs_contrast = _needs_feature_group(requested_features, CONTRAST_FEATURE_NAMES)
    needs_morphology = _needs_feature_group(requested_features, MORPHOLOGY_FEATURE_NAMES)
    xy_spacing_nm = _positive_finite_float(cfg.get("xy_spacing_nm"), "xy_spacing_nm") if needs_morphology else 1.0
    z_spacing_nm = _positive_finite_float(cfg.get("z_spacing_nm"), "z_spacing_nm") if needs_morphology and volume.ndim == 3 else 1.0

    refined = np.zeros_like(coords, dtype=np.float32)
    rows: list[dict[str, float | str]] = []
    bg_width = int(cfg.get("fit_background_width", 1))
    if bg_width < 0:
        raise ValueError("fit_background_width must be non-negative.")

    for idx, coord in enumerate(coords):
        win_slc = _bounds(coord, volume.shape, window_radius)
        bg_slc = _bounds(coord, volume.shape, window_radius + bg_width)
        window_data = np.asarray(volume[win_slc], dtype=np.float32)
        surrounding_data = np.asarray(volume[bg_slc], dtype=np.float32)
        corrected, bg, noise = _patch_background(window_data, surrounding_data, win_slc, bg_slc, cfg)
        local_guess = _window_info(coord, win_slc)
        candidate_cfg = dict(cfg)
        if full_fit_limit is not None and idx >= int(full_fit_limit):
            candidate_cfg["fit_method"] = str(candidate_cfg.get("fit_fallback_method", "moments"))
        fitted = _fit_patch(corrected, local_guess, candidate_cfg)
        offset = np.array([s.start for s in win_slc], dtype=np.float32)
        local_center = np.asarray(fitted["center"], dtype=np.float32) - 1.0 + offset
        sigma = np.asarray(fitted["sigma"], dtype=np.float32)
        rho = np.asarray(fitted["rho"], dtype=np.float32)
        integrated = float(np.sum(corrected))
        fit_amplitude = float(fitted["amplitude"])
        voxel_amplitude = float(np.max(corrected)) if corrected.size else 0.0
        fit_snr = float(fit_amplitude / max(noise, 1e-6))
        center_index = _center_index_from_guess(local_guess, corrected.shape)
        fitted_center_zero_based = np.asarray(fitted["center"], dtype=np.float32) - 1.0
        patch_features: dict[str, float] = {}
        if needs_contrast:
            patch_features.update(_contrast_features(corrected, center_index))
        if needs_morphology:
            patch_features.update(
                _morphology_features(
                    corrected,
                    center_index,
                    fitted_center_zero_based,
                    fit_amplitude,
                    xy_spacing_nm,
                    z_spacing_nm,
                )
            )
        refined[idx] = local_center
        if window_data.ndim == 2:
            coordinate_row = {
                "y": float(local_center[0]),
                "x": float(local_center[1]),
            }
        else:
            coordinate_row = {
                "z": float(local_center[0]),
                "y": float(local_center[1]),
                "x": float(local_center[2]),
            }
        row = {
            **coordinate_row,
            "score_raw": float(scores[idx]),
            "fit_method_id": str(fitted.get("fit_method_id", "")),
            "fit_amplitude": fit_amplitude,
            "voxel_amplitude": voxel_amplitude,
            "amplitude": fit_amplitude,
            "amplitude_x": float(fitted["amplitude_x"]),
            "amplitude_y": float(fitted["amplitude_y"]),
            "amplitude_z": float(fitted["amplitude_z"]),
            "amplitude_xy": float(fitted["amplitude_xy"]),
            "background": bg,
            "noise": noise,
            "integrated_intensity": integrated,
            "snr": fit_snr,
            "sigma_x": float(sigma[2] if len(sigma) >= 3 else sigma[-1]),
            "sigma_y": float(sigma[1] if len(sigma) >= 2 else sigma[-1]),
            "sigma_z": float(sigma[0]) if len(sigma) >= 1 else float("nan"),
            "rho_xy": float(rho[0]),
            "rho_xz": float(rho[1]),
            "rho_yz": float(rho[2]),
            "r_squared": float(fitted["r_squared"]),
        }
        for key in ("residual_rmse", "residual_energy_norm"):
            if key in fitted:
                row[key] = float(fitted[key])
        row.update(patch_features)
        rows.append(row)
    return FitResult(coords=refined, table=rows)
