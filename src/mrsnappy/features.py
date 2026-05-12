from __future__ import annotations

import numpy as np
import pandas as pd


def feature_table(
    rows: list[dict[str, float]],
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    if not rows:
        cols = selected_features or []
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["sigma_sum"] = df["sigma_x"] + df["sigma_y"] + df["sigma_z"]
    df["sigma_product"] = df["sigma_x"] * df["sigma_y"] * df["sigma_z"]
    df["sigma_xy_ratio"] = df["sigma_x"] / np.clip(df["sigma_y"], 1e-6, None)
    df["sigma_z_ratio"] = df["sigma_z"] / np.clip((df["sigma_x"] + df["sigma_y"]) / 2.0, 1e-6, None)
    df["amplitude_over_background"] = df["amplitude"] / np.clip(df["background"], 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity_ratio = (df["integrated_intensity"] + 1.0) / (df["background"] + 1.0)
        df["quality_weighted_snr"] = df["r_squared"] * np.log10(np.maximum(intensity_ratio, 1e-6))
    df["quality_vs_size_penalty"] = df["r_squared"] / np.clip(df["sigma_sum"] + 1e-3, 1e-6, None)
    df["distortion_energy"] = df["rho_xy"] ** 2 + df["rho_xz"] ** 2 + df["rho_yz"] ** 2
    df["log_integrated_intensity"] = np.log10(np.maximum(df["integrated_intensity"] + 1.0, 1e-6))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if selected_features:
        missing = [col for col in selected_features if col not in df.columns]
        for col in missing:
            df[col] = 0.0
        return df[selected_features].copy()
    return df
