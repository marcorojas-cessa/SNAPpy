from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class FitResult:
    coords: np.ndarray
    table: list[dict[str, float]]


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


def _poly_design(coords: np.ndarray, degree: int) -> np.ndarray:
    if coords.shape[1] == 3:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        basis = [x, y, z, x**2, y**2, z**2, x * y, x * z, y * z]
        if degree >= 3:
            basis.extend([x**3, y**3, z**3, x**2 * y, x * y**2, x**2 * z, x * z**2, y**2 * z, y * z**2, x * y * z])
        basis.append(np.ones_like(x))
        return np.column_stack(basis).astype(np.float64)
    x, y = coords[:, 0], coords[:, 1]
    basis = [x, y, x**2, y**2, x * y]
    if degree >= 3:
        basis.extend([x**3, y**3, x**2 * y, x * y**2])
    basis.append(np.ones_like(x))
    return np.column_stack(basis).astype(np.float64)


def _plane_design(coords: np.ndarray) -> np.ndarray:
    ones = np.ones((coords.shape[0], 1), dtype=np.float64)
    return np.column_stack([coords.astype(np.float64), ones])


def _patch_background(
    window_data: np.ndarray,
    surrounding_data: np.ndarray,
    window_slices: tuple[slice, ...],
    surrounding_slices: tuple[slice, ...],
    cfg: dict[str, float | int | str],
) -> tuple[np.ndarray, float]:
    method = str(cfg.get("fit_background_method", "Mean Surrounding Subtraction"))
    poly_degree = int(cfg.get("fit_poly_degree", 2))

    mask = np.ones(surrounding_data.shape, dtype=bool)
    inner_slices = []
    for win_slc, bg_slc in zip(window_slices, surrounding_slices):
        inner_slices.append(slice(win_slc.start - bg_slc.start, win_slc.stop - bg_slc.start))
    mask[tuple(inner_slices)] = False

    if not np.any(mask):
        bg = float(np.mean(surrounding_data))
        return np.asarray(window_data, dtype=np.float32) - bg, bg

    background_vals = np.asarray(surrounding_data[mask], dtype=np.float64)
    if method == "Mean Surrounding Subtraction":
        bg = float(np.mean(background_vals))
        return np.asarray(window_data, dtype=np.float32) - bg, bg

    bg_coords = np.column_stack(np.nonzero(mask)).astype(np.float64) + 1.0
    win_coords = np.column_stack(np.indices(window_data.shape).reshape(window_data.ndim, -1).T).astype(np.float64) + 1.0

    if method == "Local Plane Fitting":
        design_bg = _plane_design(bg_coords)
        coeffs, *_ = np.linalg.lstsq(design_bg, background_vals, rcond=None)
        fitted = (_plane_design(win_coords) @ coeffs).reshape(window_data.shape)
    elif method == "Local Polynomial Fitting":
        design_bg = _poly_design(bg_coords, poly_degree)
        coeffs, *_ = np.linalg.lstsq(design_bg, background_vals, rcond=None)
        fitted = (_poly_design(win_coords, poly_degree) @ coeffs).reshape(window_data.shape)
    else:
        bg = float(np.mean(background_vals))
        return np.asarray(window_data, dtype=np.float32) - bg, bg

    bg_mean = float(np.mean(fitted))
    corrected = np.asarray(window_data, dtype=np.float32) - np.asarray(fitted, dtype=np.float32)
    return corrected, bg_mean


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
    return amp * np.exp(-0.5 * quad)


def _fit_2d_distorted(data: np.ndarray, center_guess: tuple[float, float], max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    rows, cols = data.shape
    yy, xx = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing="ij")
    coords = np.column_stack([yy.ravel(), xx.ravel()]).astype(np.float64)
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
    zz, yy, xx = np.meshgrid(
        np.arange(1, data.shape[0] + 1),
        np.arange(1, data.shape[1] + 1),
        np.arange(1, data.shape[2] + 1),
        indexing="ij",
    )
    coords = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float64)
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
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.zeros(coords.shape[0], dtype=np.float64)
    delta = coords - mu
    quad = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    return amp * np.exp(-0.5 * quad)


def _fit_3d_distorted(data: np.ndarray, center_guess: tuple[float, float, float], max_iterations: int, tolerance: float) -> tuple[np.ndarray, float]:
    zz, yy, xx = np.meshgrid(
        np.arange(1, data.shape[0] + 1),
        np.arange(1, data.shape[1] + 1),
        np.arange(1, data.shape[2] + 1),
        indexing="ij",
    )
    coords = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float64)
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


def _fit_patch(signal: np.ndarray, center_guess: np.ndarray, cfg: dict[str, float | int | str]) -> dict[str, float]:
    method_key = str(cfg.get("fit_method", "moments")).lower()
    max_iterations = int(cfg.get("fit_max_iterations", 200))
    tolerance = float(cfg.get("fit_tolerance", 1e-6))
    if signal.ndim == 2:
        fit2d, r2 = _fit_2d_distorted(signal, (float(center_guess[0]), float(center_guess[1])), max_iterations, tolerance)
        return {
            "center": np.array([fit2d[1], fit2d[2]], dtype=np.float32),
            "amplitude": float(fit2d[0]),
            "amplitude_x": float(fit2d[0]),
            "amplitude_y": float(fit2d[0]),
            "amplitude_z": float("nan"),
            "amplitude_xy": float(fit2d[0]),
            "sigma": np.array([float("nan"), fit2d[3], fit2d[4]], dtype=np.float32),
            "rho": np.array([fit2d[5], 0.0, 0.0], dtype=np.float32),
            "r_squared": float(r2),
        }
    if "distorted 3d" in method_key:
        fit3d, r2 = _fit_3d_distorted(signal, (float(center_guess[0]), float(center_guess[1]), float(center_guess[2])), max_iterations, tolerance)
        return {
            "center": np.array([fit3d[1], fit3d[2], fit3d[3]], dtype=np.float32),
            "amplitude": float(fit3d[0]),
            "amplitude_x": float("nan"),
            "amplitude_y": float("nan"),
            "amplitude_z": float("nan"),
            "amplitude_xy": float("nan"),
            "sigma": np.array([fit3d[4], fit3d[5], fit3d[6]], dtype=np.float32),
            "rho": np.array([fit3d[7], fit3d[8], fit3d[9]], dtype=np.float32),
            "r_squared": float(r2),
        }
    if "3d gaussian" in method_key:
        fit3d, r2 = _fit_3d(signal, (float(center_guess[0]), float(center_guess[1]), float(center_guess[2])), max_iterations, tolerance)
        return {
            "center": np.array([fit3d[1], fit3d[2], fit3d[3]], dtype=np.float32),
            "amplitude": float(fit3d[0]),
            "amplitude_x": float("nan"),
            "amplitude_y": float("nan"),
            "amplitude_z": float("nan"),
            "amplitude_xy": float("nan"),
            "sigma": np.array([fit3d[4], fit3d[5], fit3d[6]], dtype=np.float32),
            "rho": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "r_squared": float(r2),
        }
    if "2d (xy) + 1d (z)" in method_key and signal.ndim == 3:
        z_profile = signal.sum(axis=(1, 2))
        fit_z, r2_z = _fit_1d(z_profile, float(center_guess[0]), max_iterations, tolerance)
        xy_projection = signal.sum(axis=0)
        fit_xy, r2_xy = _fit_2d_distorted(xy_projection, (float(center_guess[1]), float(center_guess[2])), max_iterations, tolerance)
        return {
            "center": np.array([fit_z[1], fit_xy[1], fit_xy[2]], dtype=np.float32),
            "amplitude": float(fit_xy[0]),
            "amplitude_x": float(fit_xy[0]),
            "amplitude_y": float(fit_xy[0]),
            "amplitude_z": float(fit_z[0]),
            "amplitude_xy": float(fit_xy[0]),
            "sigma": np.array([fit_z[2], fit_xy[3], fit_xy[4]], dtype=np.float32),
            "rho": np.array([fit_xy[5], 0.0, 0.0], dtype=np.float32),
            "r_squared": float((r2_xy + r2_z) / 2.0),
        }
    if "1d" in method_key and signal.ndim == 3:
        x_profile = signal.sum(axis=(0, 1))
        y_profile = signal.sum(axis=(0, 2))
        z_profile = signal.sum(axis=(1, 2))
        fit_x, r2_x = _fit_1d(x_profile, float(center_guess[2]), max_iterations, tolerance)
        fit_y, r2_y = _fit_1d(y_profile, float(center_guess[1]), max_iterations, tolerance)
        fit_z, r2_z = _fit_1d(z_profile, float(center_guess[0]), max_iterations, tolerance)
        return {
            "center": np.array([fit_z[1], fit_y[1], fit_x[1]], dtype=np.float32),
            "amplitude": float(max(fit_x[0], fit_y[0])),
            "amplitude_x": float(fit_x[0]),
            "amplitude_y": float(fit_y[0]),
            "amplitude_z": float(fit_z[0]),
            "amplitude_xy": float("nan"),
            "sigma": np.array([fit_z[2], fit_y[2], fit_x[2]], dtype=np.float32),
            "rho": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "r_squared": float(np.nanmean([r2_x, r2_y, r2_z])),
        }
    total = float(np.sum(np.clip(signal, 0.0, None)))
    grid = np.indices(signal.shape, dtype=np.float32)
    if total <= 0:
        center = np.asarray(center_guess, dtype=np.float32)
        sigma = np.ones(signal.ndim, dtype=np.float32)
        cov = np.eye(signal.ndim, dtype=np.float32)
    else:
        center = np.array([(grid[a] * signal).sum() / total for a in range(signal.ndim)], dtype=np.float32)
        centered = grid.reshape(signal.ndim, -1).T - center
        weights = signal.reshape(-1)
        cov = (centered.T * weights) @ centered / (float(weights.sum()) + 1e-8)
        cov = np.asarray(cov, dtype=np.float32)
        sigma = np.sqrt(np.clip(np.diag(cov), 1e-3, None))
    diag = np.clip(np.diag(cov), 1e-6, None)
    return {
        "center": center,
        "amplitude": float(np.max(signal)),
        "amplitude_x": float(np.max(signal)),
        "amplitude_y": float(np.max(signal)),
        "amplitude_z": float(np.max(signal)),
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

    refined = np.zeros_like(coords, dtype=np.float32)
    rows: list[dict[str, float]] = []
    bg_width = int(cfg.get("fit_background_width", 1))

    for idx, coord in enumerate(coords):
        win_slc = _bounds(coord, volume.shape, window_radius)
        bg_slc = _bounds(coord, volume.shape, window_radius + bg_width)
        window_data = np.asarray(volume[win_slc], dtype=np.float32)
        surrounding_data = np.asarray(volume[bg_slc], dtype=np.float32)
        corrected, bg = _patch_background(window_data, surrounding_data, win_slc, bg_slc, cfg)
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
        amplitude = float(fitted["amplitude"])
        snr = float(integrated / max(abs(bg), 1.0))
        refined[idx] = local_center
        rows.append(
            {
                "z": float(local_center[0]) if window_data.ndim >= 1 else 0.0,
                "y": float(local_center[1]) if window_data.ndim >= 2 else 0.0,
                "x": float(local_center[2]) if window_data.ndim >= 3 else 0.0,
                "score_raw": float(scores[idx]),
                "amplitude": amplitude,
                "amplitude_x": float(fitted["amplitude_x"]),
                "amplitude_y": float(fitted["amplitude_y"]),
                "amplitude_z": float(fitted["amplitude_z"]),
                "amplitude_xy": float(fitted["amplitude_xy"]),
                "background": bg,
                "integrated_intensity": integrated,
                "snr": snr,
                "sigma_x": float(sigma[2] if len(sigma) >= 3 else sigma[-1]),
                "sigma_y": float(sigma[1] if len(sigma) >= 2 else sigma[-1]),
                "sigma_z": float(sigma[0]) if len(sigma) >= 1 else float("nan"),
                "rho_xy": float(rho[0]),
                "rho_xz": float(rho[1]),
                "rho_yz": float(rho[2]),
                "r_squared": float(fitted["r_squared"]),
            }
        )
    return FitResult(coords=refined, table=rows)
