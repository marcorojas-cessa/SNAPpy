# 2D and 3D Fit/Feature Review

This note documents the finalized 2D/3D fitting-mode naming and the modern 2D feature/background names.

## Implemented Fitting Names

| Public name | Internal id | Dimensionality | Meaning |
|---|---|---:|---|
| `2D Gaussian` | `gaussian_2d` | 2D only | Standard axis-aligned 2D Gaussian. This is the 2D version of `3D Gaussian`. |
| `Distorted 2D Gaussian` | `distorted_gaussian_2d` | 2D only | Covariance-enabled 2D Gaussian. This is the 2D version of `Distorted 3D Gaussian`. |
| `2D (XY) + 1D (Z) Gaussian` | `xy_z_gaussian` | 3D only | Axis-aligned 2D Gaussian fit on the XY projection plus axis-aligned 1D Gaussian fit on the Z profile. No covariance terms. |
| `3D Gaussian` | `gaussian_3d` | 3D only | Standard axis-aligned 3D Gaussian. |
| `Distorted 3D Gaussian` | `distorted_gaussian_3d` | 3D only | Full covariance-enabled 3D Gaussian. |
| `moments` | `moments` | 2D or 3D internal fallback | Positive-intensity weighted centroid/covariance fallback. Not a public optimization mode. |

Native 2D images may use `2D Gaussian`, `Distorted 2D Gaussian`, or the internal fallback `moments`.
Native 3D images may use `2D (XY) + 1D (Z) Gaussian`, `3D Gaussian`, `Distorted 3D Gaussian`, or the internal fallback `moments`.

## Fit Equations

### 2D Gaussian

Standard, axis-aligned, no covariance term:

```text
I(y,x) = A * exp(-((y - mu_y)^2 / (2*sigma_y^2)
                + (x - mu_x)^2 / (2*sigma_x^2)))
```

Parameters:

```text
A, mu_y, mu_x, sigma_y, sigma_x
```

Feature implications:

- Compatible with intensity, XY sigma, residual quality, contrast, and component morphology.
- Not compatible with covariance/distortion features such as `rho_lateral_abs` and `covariance_elongation`.
- Z-only features are voided.

### Distorted 2D Gaussian

Elliptical/covariance-enabled 2D model:

```text
I(y,x) = A * exp(-0.5 * d_xy.T * inv(Sigma_xy) * d_xy)

d_xy = [y - mu_y, x - mu_x]

Sigma_xy = [[sigma_y^2, rho_xy*sigma_y*sigma_x],
            [rho_xy*sigma_y*sigma_x, sigma_x^2]]
```

Parameters:

```text
A, mu_y, mu_x, sigma_y, sigma_x, rho_xy
```

Feature implications:

- Compatible with intensity, XY sigma, residual quality, XY covariance/distortion, contrast, and component morphology.
- Z-only features are voided.

### 2D (XY) + 1D (Z) Gaussian

This is a 3D-only separable fit. It does not use covariance terms.

```text
I_xy(y,x) = A_xy * exp(-((y - mu_y)^2 / (2*sigma_y^2)
                      + (x - mu_x)^2 / (2*sigma_x^2)))

I_z(z) = A_z * exp(-(z - mu_z)^2 / (2*sigma_z^2))
```

Parameters:

```text
A_xy, mu_y, mu_x, sigma_y, sigma_x
A_z,  mu_z, sigma_z
```

Feature implications:

- Compatible with intensity and XYZ sigma features.
- No full-patch residual metrics.
- No covariance/distortion features.

### 3D Gaussian

Standard, axis-aligned, no covariance terms:

```text
I(z,y,x) = A * exp(-((z - mu_z)^2 / (2*sigma_z^2)
                  + (y - mu_y)^2 / (2*sigma_y^2)
                  + (x - mu_x)^2 / (2*sigma_x^2)))
```

Parameters:

```text
A, mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x
```

Feature implications:

- Compatible with intensity, XYZ sigma, residual quality, contrast, and component morphology.
- No covariance/distortion features.

### Distorted 3D Gaussian

Full covariance-enabled 3D model:

```text
I(z,y,x) = A * exp(-0.5 * d_zyx.T * inv(Sigma_zyx) * d_zyx)

d_zyx = [z - mu_z, y - mu_y, x - mu_x]

Sigma_zyx =
[[sigma_z^2,              rho_yz*sigma_z*sigma_y, rho_xz*sigma_z*sigma_x],
 [rho_yz*sigma_z*sigma_y, sigma_y^2,              rho_xy*sigma_y*sigma_x],
 [rho_xz*sigma_z*sigma_x, rho_xy*sigma_y*sigma_x, sigma_x^2]]
```

Parameters:

```text
A, mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x, rho_yz, rho_xz, rho_xy
```

Feature implications:

- Compatible with intensity, XYZ sigma, residual quality, covariance/distortion, contrast, and component morphology.

### Moments

Internal fallback:

```text
weights = max(I, 0)
center_a = sum(grid_a * weights) / sum(weights) + 1
Cov = centered.T * weights * centered / sum(weights)
```

Feature implications:

- No Gaussian model is optimized.
- No residual model metrics.
- Used as a computational fallback after `full_fit_limit` when configured.

## Feature Pack Side-By-Side

All packs keep the same pack names in 2D. Incompatible Z-only and covariance-only features are filtered out during feature resolution.

| Pack | 3D contents | Standard 2D Gaussian contents | Distorted 2D Gaussian contents |
|---|---|---|---|
| `core_fit` | Signal intensity + fit sigma + common fit quality + residual quality | Same, except `sigma_z_nm` and `sigma_axial_ratio` are voided, and `sigma_product_nm3` becomes `sigma_product_nm2` | Same as standard 2D |
| `core_contrast` | `core_fit` + contrast features | Same, except `z_core_minus_shell` is voided | Same as standard 2D |
| `core_morphology` | `core_fit` + distortion + component morphology | Same, except all Z-only and covariance/distortion features are voided | Same, except `rho_lateral_abs` and `covariance_elongation` remain available |
| `full_interpretable` | `core_fit` + distortion + contrast + component morphology | Same, except all Z-only and covariance/distortion features are voided | Same, except `rho_lateral_abs` and `covariance_elongation` remain available |

## Modern 2D Feature Names

SNAPpy uses dimension-specific names for features whose units or geometric meaning change in 2D.
The 3D names remain for 3D outputs. The 2D names below are the modern 2D output names.

| 3D feature name | 3D calculation | 2D feature name | 2D calculation |
|---|---|---|---|
| `sigma_product_nm3` | `sigma_x_nm * sigma_y_nm * sigma_z_nm`, units nm^3 | `sigma_product_nm2` | `sigma_x_nm * sigma_y_nm`, units nm^2 |
| `component_voxel_volume` | Voxel count | `component_pixel_area` | Pixel count in selected component |
| `component_surface_area_vox2` | Boundary face count across 3D voxel neighbors | `component_boundary_px` | Boundary edge count/perimeter proxy across 2D pixel neighbors |
| `component_surface_to_volume_ratio` | `surface_area / volume` | `component_boundary_to_area_ratio` | `component_boundary_px / component_pixel_area` |
| `component_sphericity_3d` | `(pi^(1/3) * (6 * volume)^(2/3)) / surface_area` | `component_circularity_2d` | `(4 * pi * area) / perimeter^2` |
| `component_convex_voxel_volume` | Convex-hull voxel count | `component_convex_size_px` | Convex-hull pixel count |
| `component_solidity_3d` | `volume / convex_volume` | `component_solidity_2d` | `area / convex_area` |
| `component_elongation_3d` | `1 - axis_min / axis_max` from 3D component covariance | `component_elongation_2d` | Same formula from 2D component covariance |
| `xy_core_minus_shell` | Central-Z XY core minus central-Z XY shell | `xy_core_minus_shell` | Kept as-is; in 2D this equals the XY/core shell calculation |
| `z_core_minus_shell` | Z-slab core minus Z shell | Not available in 2D | Voided for 2D |

## 2D Background Correction Names

`rolling_ball_2d` already exists and performs 2D rolling-ball background correction, slice-by-slice for 3D stacks when requested.
`rolling_box_2d` is the 2D grey-opening box background method and is the 2D counterpart to `rolling_box_3d`.

| Method | Meaning |
|---|---|
| `rolling_box_2d` | 2D grey-opening box background correction; if given a 3D stack, applies the 2D operation independently to each z slice |
| `rolling_box_3d` | 3D grey-opening box background correction |
| `rolling_ball_2d` | 2D rolling-ball background correction; if given a 3D stack, applies the 2D operation independently to each z slice |
| `rolling_ball_3d` | Exact 3D rolling-ball background correction |
