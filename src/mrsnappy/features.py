from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6

FIT_METHOD_IDS = {
    "2d gaussian": "gaussian_2d",
    "2d_gaussian": "gaussian_2d",
    "gaussian_2d": "gaussian_2d",
    "distorted 2d gaussian": "distorted_gaussian_2d",
    "distorted_2d_gaussian": "distorted_gaussian_2d",
    "distorted_gaussian_2d": "distorted_gaussian_2d",
    "2d (xy) + 1d (z) gaussian": "xy_z_gaussian",
    "3d gaussian": "gaussian_3d",
    "distorted 3d gaussian": "distorted_gaussian_3d",
    "moments": "moments",
    "moment": "moments",
    "image moments": "moments",
}

SIGNAL_INTENSITY_FEATURES: list[str] = [
    "fit_amplitude",
    "voxel_amplitude",
    "amplitude_diff",
    "background",
    "noise",
    "integrated_intensity",
    "log_integrated_intensity",
    "fit_amplitude_over_background",
    "voxel_amplitude_over_background",
    "fit_snr",
    "voxel_snr",
]

FIT_SIGMA_FEATURES: list[str] = [
    "sigma_x_nm",
    "sigma_y_nm",
    "sigma_z_nm",
    "sigma_xy_mean_nm",
    "sigma_total_nm",
    "sigma_product_nm3",
    "sigma_lateral_asymmetry",
    "sigma_axial_ratio",
]
FIT_SIGMA_2D_FEATURES: list[str] = [
    "sigma_product_nm2",
]

FIT_QUALITY_COMMON_FEATURES: list[str] = [
    "r_squared",
    "quality_weighted_snr",
    "quality_vs_size_penalty",
]

FIT_QUALITY_3D_FEATURES: list[str] = [
    "residual_rmse",
    "residual_energy_norm",
]

DISTORTION_FEATURES: list[str] = [
    "rho_lateral_abs",
    "rho_axial_energy",
    "covariance_elongation",
    "long_axis_z_alignment",
]

CONTRAST_FEATURES: list[str] = [
    "core_mean",
    "core_std",
    "shell_mean",
    "shell_std",
    "core_minus_shell",
    "core_shell_snr",
    "xy_core_minus_shell",
    "z_core_minus_shell",
    "halfspace_absdiff_max",
]

MORPHOLOGY_FEATURES: list[str] = [
    "component_voxel_volume",
    "component_surface_area_vox2",
    "component_surface_to_volume_ratio",
    "component_sphericity_3d",
    "component_convex_voxel_volume",
    "component_solidity_3d",
    "component_elongation_3d",
    "component_centroid_fit_distance_nm",
]
MORPHOLOGY_2D_FEATURES: list[str] = [
    "component_pixel_area",
    "component_boundary_px",
    "component_boundary_to_area_ratio",
    "component_circularity_2d",
    "component_convex_size_px",
    "component_solidity_2d",
    "component_elongation_2d",
]

CORE_FIT_FEATURES: list[str] = [
    *SIGNAL_INTENSITY_FEATURES,
    *FIT_SIGMA_FEATURES,
    *FIT_QUALITY_COMMON_FEATURES,
    *FIT_QUALITY_3D_FEATURES,
]

CORE_CONTRAST_FEATURES: list[str] = [*CORE_FIT_FEATURES, *CONTRAST_FEATURES]
CORE_MORPHOLOGY_FEATURES: list[str] = [*CORE_FIT_FEATURES, *DISTORTION_FEATURES, *MORPHOLOGY_FEATURES]
FULL_INTERPRETABLE_FEATURES: list[str] = [
    *CORE_FIT_FEATURES,
    *DISTORTION_FEATURES,
    *CONTRAST_FEATURES,
    *MORPHOLOGY_FEATURES,
]

PUBLIC_FEATURES: list[str] = []
for _feature in [*FULL_INTERPRETABLE_FEATURES, *FIT_SIGMA_2D_FEATURES, *MORPHOLOGY_2D_FEATURES]:
    if _feature not in PUBLIC_FEATURES:
        PUBLIC_FEATURES.append(_feature)
PUBLIC_FEATURE_SET = set(PUBLIC_FEATURES)

FEATURE_PACK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "core_fit": {
        "name": "core_fit",
        "features": CORE_FIT_FEATURES,
    },
    "core_contrast": {
        "name": "core_contrast",
        "features": CORE_CONTRAST_FEATURES,
    },
    "core_morphology": {
        "name": "core_morphology",
        "features": CORE_MORPHOLOGY_FEATURES,
    },
    "full_interpretable": {
        "name": "full_interpretable",
        "features": FULL_INTERPRETABLE_FEATURES,
    },
}

STAGE2_FEATURE_PACK_NAMES: list[str] = list(FEATURE_PACK_DEFINITIONS)

_RESIDUAL_FIT_METHODS = {"gaussian_2d", "distorted_gaussian_2d", "gaussian_3d", "distorted_gaussian_3d"}
_DISTORTED_FIT_METHODS = {"distorted_gaussian_2d", "distorted_gaussian_3d"}
_2D_VOID_FEATURES = {
    "sigma_z_nm",
    "sigma_axial_ratio",
    "z_core_minus_shell",
    "rho_axial_energy",
    "long_axis_z_alignment",
}
_2D_DISTORTION_FEATURES = {"rho_lateral_abs", "covariance_elongation"}
_2D_ONLY_FEATURES = set(FIT_SIGMA_2D_FEATURES) | set(MORPHOLOGY_2D_FEATURES)
_2D_FEATURE_RENAMES = {
    "sigma_product_nm3": "sigma_product_nm2",
    "component_voxel_volume": "component_pixel_area",
    "component_surface_area_vox2": "component_boundary_px",
    "component_surface_to_volume_ratio": "component_boundary_to_area_ratio",
    "component_sphericity_3d": "component_circularity_2d",
    "component_convex_voxel_volume": "component_convex_size_px",
    "component_solidity_3d": "component_solidity_2d",
    "component_elongation_3d": "component_elongation_2d",
}


def normalize_fit_method_id(fit_method: Any) -> str:
    text = str(fit_method or "").strip().lower()
    return FIT_METHOD_IDS.get(text, text)


def _normalize_dimensionality(image_dimensionality: Any | None) -> int | None:
    if image_dimensionality is None:
        return None
    try:
        ndim = int(image_dimensionality)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_dimensionality must be 2 or 3.") from exc
    if ndim not in {2, 3}:
        raise ValueError("image_dimensionality must be 2 or 3.")
    return ndim


def feature_is_compatible(feature: str, fit_method_id: str, image_dimensionality: Any | None = None) -> bool:
    fit_method_id = normalize_fit_method_id(fit_method_id)
    ndim = _normalize_dimensionality(image_dimensionality)
    if ndim is None and fit_method_id in {"gaussian_2d", "distorted_gaussian_2d"}:
        ndim = 2
    if ndim == 3 and feature in _2D_ONLY_FEATURES:
        return False
    if ndim == 2 and feature in _2D_VOID_FEATURES:
        return False
    if feature in FIT_QUALITY_3D_FEATURES:
        return fit_method_id in _RESIDUAL_FIT_METHODS
    if feature in DISTORTION_FEATURES:
        if ndim == 2 and feature in _2D_DISTORTION_FEATURES:
            return fit_method_id == "distorted_gaussian_2d"
        return fit_method_id in _DISTORTED_FIT_METHODS
    return True


def _feature_name_for_dimensionality(feature: str, ndim: int | None) -> str:
    if ndim == 2:
        return _2D_FEATURE_RENAMES.get(feature, feature)
    return feature


def resolve_features_for_fit(features: Iterable[str], fit_method: Any, image_dimensionality: Any | None = None) -> list[str]:
    fit_method_id = normalize_fit_method_id(fit_method)
    ndim = _normalize_dimensionality(image_dimensionality)
    resolved: list[str] = []
    seen: set[str] = set()
    for feature in features:
        output_feature = _feature_name_for_dimensionality(feature, ndim)
        if output_feature in seen:
            continue
        if feature_is_compatible(feature, fit_method_id, image_dimensionality=image_dimensionality):
            seen.add(output_feature)
            resolved.append(output_feature)
    return resolved


def resolve_feature_pack_features(pack_name: str, fit_method: Any, image_dimensionality: Any | None = None) -> list[str]:
    if pack_name not in FEATURE_PACK_DEFINITIONS:
        raise KeyError(f"Unknown feature pack: {pack_name}")
    return resolve_features_for_fit(
        FEATURE_PACK_DEFINITIONS[pack_name]["features"],
        fit_method,
        image_dimensionality=image_dimensionality,
    )


def _require_positive_spacing(value: float | int | str | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required for SNAPpy feature extraction.")
    try:
        spacing = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return spacing


def _distortion_shape_features_vectorized(df: pd.DataFrame, xy_spacing_nm: float, z_spacing_nm: float) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(df)
    if n_rows == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    sigma_x = np.maximum(df["sigma_x"].to_numpy(dtype=np.float64) * xy_spacing_nm, EPS)
    sigma_y = np.maximum(df["sigma_y"].to_numpy(dtype=np.float64) * xy_spacing_nm, EPS)
    sigma_z = np.maximum(df["sigma_z"].to_numpy(dtype=np.float64) * z_spacing_nm, EPS)
    rho_xy = np.clip(df["rho_xy"].to_numpy(dtype=np.float64), -0.99, 0.99)
    rho_xz = np.clip(df["rho_xz"].to_numpy(dtype=np.float64), -0.99, 0.99)
    rho_yz = np.clip(df["rho_yz"].to_numpy(dtype=np.float64), -0.99, 0.99)

    cov = np.zeros((n_rows, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = sigma_z**2
    cov[:, 1, 1] = sigma_y**2
    cov[:, 2, 2] = sigma_x**2
    cov[:, 0, 1] = cov[:, 1, 0] = rho_yz * sigma_z * sigma_y
    cov[:, 0, 2] = cov[:, 2, 0] = rho_xz * sigma_z * sigma_x
    cov[:, 1, 2] = cov[:, 2, 1] = rho_xy * sigma_y * sigma_x

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.zeros(n_rows, dtype=np.float64), np.zeros(n_rows, dtype=np.float64)
    principal_sigmas = np.sqrt(np.clip(eigenvalues, EPS**2, None))
    max_idx = np.argmax(principal_sigmas, axis=1)
    row_idx = np.arange(n_rows)
    sigma_max = principal_sigmas[row_idx, max_idx]
    sigma_min = np.min(principal_sigmas, axis=1)
    covariance_elongation = 1.0 - sigma_min / np.maximum(sigma_max, EPS)
    long_axis_z_alignment = np.abs(eigenvectors[row_idx, 0, max_idx])
    return covariance_elongation.astype(np.float64), long_axis_z_alignment.astype(np.float64)


def _distortion_shape_features_2d_vectorized(df: pd.DataFrame, xy_spacing_nm: float) -> np.ndarray:
    n_rows = len(df)
    if n_rows == 0:
        return np.empty((0,), dtype=np.float64)
    sigma_x = np.maximum(df["sigma_x"].to_numpy(dtype=np.float64) * xy_spacing_nm, EPS)
    sigma_y = np.maximum(df["sigma_y"].to_numpy(dtype=np.float64) * xy_spacing_nm, EPS)
    rho_xy = np.clip(df["rho_xy"].to_numpy(dtype=np.float64), -0.99, 0.99)

    cov = np.zeros((n_rows, 2, 2), dtype=np.float64)
    cov[:, 0, 0] = sigma_y**2
    cov[:, 1, 1] = sigma_x**2
    cov[:, 0, 1] = cov[:, 1, 0] = rho_xy * sigma_y * sigma_x

    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return np.zeros(n_rows, dtype=np.float64)
    principal_sigmas = np.sqrt(np.clip(eigenvalues, EPS**2, None))
    sigma_max = np.max(principal_sigmas, axis=1)
    sigma_min = np.min(principal_sigmas, axis=1)
    return (1.0 - sigma_min / np.maximum(sigma_max, EPS)).astype(np.float64)


def _infer_fit_method_id(df: pd.DataFrame) -> str | None:
    if "fit_method_id" in df.columns and len(df):
        values = [normalize_fit_method_id(value) for value in df["fit_method_id"].dropna().unique()]
        if len(values) == 1:
            return values[0]
        non_moments = {value for value in values if value != "moments"}
        if len(non_moments) == 1:
            return next(iter(non_moments))
    if "fit_method" in df.columns and len(df):
        values = [normalize_fit_method_id(value) for value in df["fit_method"].dropna().unique()]
        if len(values) == 1:
            return values[0]
        non_moments = {value for value in values if value != "moments"}
        if len(non_moments) == 1:
            return next(iter(non_moments))
    return None


def feature_table(
    rows: list[dict[str, float]],
    selected_features: list[str] | None = None,
    *,
    xy_spacing_nm: float | None = None,
    z_spacing_nm: float | None = None,
    image_dimensionality: int | None = None,
    fit_method: Any | None = None,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=selected_features or [])
    if selected_features is not None and not selected_features:
        return pd.DataFrame(index=range(len(rows)))

    xy_spacing_nm_value = _require_positive_spacing(xy_spacing_nm, "xy_spacing_nm")
    if image_dimensionality is None:
        fit_method_id_for_dim = None
        if rows and isinstance(rows[0], dict):
            fit_method_id_for_dim = normalize_fit_method_id(rows[0].get("fit_method_id"))
        image_ndim = 2 if fit_method_id_for_dim in {"gaussian_2d", "distorted_gaussian_2d"} else 3
    else:
        image_ndim = _normalize_dimensionality(image_dimensionality)
        assert image_ndim is not None
    z_spacing_nm_value = _require_positive_spacing(z_spacing_nm, "z_spacing_nm") if image_ndim == 3 else None
    selected_feature_set = set(selected_features) if selected_features is not None else None
    if selected_features is not None:
        unsupported = [col for col in selected_features if col not in PUBLIC_FEATURE_SET]
        if unsupported:
            raise ValueError(
                "Unsupported SNAPpy Stage 2 feature(s): "
                f"{', '.join(unsupported)}. "
                "Use features from the implemented feature packs."
            )

    df = pd.DataFrame(rows)
    if "fit_amplitude" not in df.columns and "amplitude" in df.columns:
        df["fit_amplitude"] = df["amplitude"]
    if "voxel_amplitude" not in df.columns:
        df["voxel_amplitude"] = df["fit_amplitude"] if "fit_amplitude" in df.columns else 0.0
    if "noise" not in df.columns:
        df["noise"] = 0.0

    df["amplitude_diff"] = df["voxel_amplitude"] - df["fit_amplitude"]
    df["log_integrated_intensity"] = np.log10(np.maximum(df["integrated_intensity"], 0.0) + 1.0)
    df["fit_amplitude_over_background"] = df["fit_amplitude"] / np.maximum(df["background"], EPS)
    df["voxel_amplitude_over_background"] = df["voxel_amplitude"] / np.maximum(df["background"], EPS)
    df["fit_snr"] = df["fit_amplitude"] / np.maximum(df["noise"], EPS)
    df["voxel_snr"] = df["voxel_amplitude"] / np.maximum(df["noise"], EPS)

    df["sigma_x_nm"] = df["sigma_x"] * xy_spacing_nm_value
    df["sigma_y_nm"] = df["sigma_y"] * xy_spacing_nm_value
    df["sigma_xy_mean_nm"] = (df["sigma_x_nm"] + df["sigma_y_nm"]) / 2.0
    df["sigma_lateral_asymmetry"] = (df["sigma_x_nm"] - df["sigma_y_nm"]) / np.maximum(df["sigma_xy_mean_nm"], EPS)
    if image_ndim == 3:
        assert z_spacing_nm_value is not None
        df["sigma_z_nm"] = df["sigma_z"] * z_spacing_nm_value
        df["sigma_total_nm"] = df["sigma_x_nm"] + df["sigma_y_nm"] + df["sigma_z_nm"]
        df["sigma_product_nm3"] = df["sigma_x_nm"] * df["sigma_y_nm"] * df["sigma_z_nm"]
        df["sigma_axial_ratio"] = df["sigma_z_nm"] / np.maximum(df["sigma_xy_mean_nm"], EPS)
    else:
        df["sigma_total_nm"] = df["sigma_x_nm"] + df["sigma_y_nm"]
        df["sigma_product_nm2"] = df["sigma_x_nm"] * df["sigma_y_nm"]

    df["quality_weighted_snr"] = df["r_squared"] * np.log10(np.maximum(df["fit_snr"], 0.0) + 1.0)
    df["quality_vs_size_penalty"] = df["r_squared"] / np.maximum(df["sigma_total_nm"], EPS)

    fit_method_id = normalize_fit_method_id(fit_method) if fit_method is not None else None
    if not fit_method_id:
        fit_method_id = _infer_fit_method_id(df)
    needs_distortion = selected_feature_set is None or bool(selected_feature_set & set(DISTORTION_FEATURES))
    needs_distortion_shape = selected_feature_set is None or bool(selected_feature_set & {"covariance_elongation", "long_axis_z_alignment"})
    if image_ndim == 2 and fit_method_id == "distorted_gaussian_2d" and needs_distortion and "rho_xy" in df.columns:
        df["rho_lateral_abs"] = np.abs(df["rho_xy"])
        if needs_distortion_shape:
            df["covariance_elongation"] = _distortion_shape_features_2d_vectorized(df, xy_spacing_nm_value)
    elif image_ndim == 3 and fit_method_id == "distorted_gaussian_3d" and needs_distortion and {"rho_xy", "rho_xz", "rho_yz"} <= set(df.columns):
        df["rho_lateral_abs"] = np.abs(df["rho_xy"])
        df["rho_axial_energy"] = df["rho_xz"] ** 2 + df["rho_yz"] ** 2
        if needs_distortion_shape:
            covariance_elongation, long_axis_z_alignment = _distortion_shape_features_vectorized(
                df,
                xy_spacing_nm_value,
                z_spacing_nm_value,
            )
            df["covariance_elongation"] = covariance_elongation
            df["long_axis_z_alignment"] = long_axis_z_alignment

    if image_ndim == 2:
        df = df.drop(
            columns=[
                "z",
                "amplitude_z",
                "sigma_z",
                "sigma_z_nm",
                "sigma_product_nm3",
                "sigma_axial_ratio",
                "z_core_minus_shell",
                "rho_xz",
                "rho_yz",
                "rho_axial_energy",
                "long_axis_z_alignment",
                "component_voxel_volume",
                "component_surface_area_vox2",
                "component_surface_to_volume_ratio",
                "component_sphericity_3d",
                "component_convex_voxel_volume",
                "component_solidity_3d",
                "component_elongation_3d",
            ],
            errors="ignore",
        )

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if selected_features is not None:
        if not selected_features:
            return pd.DataFrame(index=df.index)
        missing = [col for col in selected_features if col not in df.columns]
        if missing:
            suffix = f" for fit_method={fit_method_id}" if fit_method_id else ""
            raise ValueError(
                "Selected SNAPpy feature(s) are unavailable"
                f"{suffix}: {', '.join(missing)}. "
                "Use a compatible feature pack or remove these selected_features."
            )
        return df[selected_features].copy()
    return df
