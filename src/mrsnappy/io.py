from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile


def read_volume(path: str | Path) -> np.ndarray:
    volume = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if not np.all(np.isfinite(volume)):
        raise ValueError(f"SNAPpy only accepts finite-valued images; found NaN or Inf voxels in {path}.")
    return volume


def read_points_csv(path: str | Path, ndim: int | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        out_ndim = 3 if ndim is None else int(ndim)
        return np.empty((0, out_ndim), dtype=np.float32)
    df = pd.read_csv(path)
    cols = {str(col).strip().lower(): col for col in df.columns}
    if {"x", "y", "z"} <= set(cols):
        x = df[cols["x"]].to_numpy(dtype=np.float32)
        y = df[cols["y"]].to_numpy(dtype=np.float32)
        z = df[cols["z"]].to_numpy(dtype=np.float32)
        if ndim is not None and int(ndim) != 3:
            raise ValueError(f"Point CSV {path} contains 3D x,y,z coordinates but ndim={ndim} was requested.")
        return np.stack([z, y, x], axis=1).astype(np.float32)
    if {"x", "y"} <= set(cols):
        x = df[cols["x"]].to_numpy(dtype=np.float32)
        y = df[cols["y"]].to_numpy(dtype=np.float32)
        if ndim is not None and int(ndim) != 2:
            raise ValueError(f"Point CSV {path} contains 2D x,y coordinates but ndim={ndim} was requested.")
        return np.stack([y, x], axis=1).astype(np.float32)
    axis0_names = ("axis-0", "axis_0", "axis0")
    axis1_names = ("axis-1", "axis_1", "axis1")
    axis0 = next((cols[name] for name in axis0_names if name in cols), None)
    axis1 = next((cols[name] for name in axis1_names if name in cols), None)
    if axis0 is not None and axis1 is not None:
        y = df[axis0].to_numpy(dtype=np.float32)
        x = df[axis1].to_numpy(dtype=np.float32)
        if ndim is not None and int(ndim) != 2:
            raise ValueError(f"Point CSV {path} contains 2D axis-0,axis-1 coordinates but ndim={ndim} was requested.")
        return np.stack([y, x], axis=1).astype(np.float32)
    raise ValueError(
        f"Unsupported point CSV format for SNAPpy: {path}. "
        "Expected case-insensitive 3D columns x,y,z or 2D columns x,y, y,x, "
        "or axis-0,axis-1. Coordinate orientation and alignment must be "
        "standardized externally before SNAPpy."
    )


def write_points_csv(path: str | Path, coords: np.ndarray, scores: np.ndarray | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] not in {2, 3}:
        raise ValueError("SNAPpy point coordinates must have shape (n, 2) or (n, 3).")
    if coords.shape[1] == 2:
        data = {
            "x": coords[:, 1] if len(coords) else [],
            "y": coords[:, 0] if len(coords) else [],
        }
    else:
        data = {
            "x": coords[:, 2] if len(coords) else [],
            "y": coords[:, 1] if len(coords) else [],
            "z": coords[:, 0] if len(coords) else [],
        }
    if scores is not None:
        data["score"] = scores
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def resolve_split_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    raw_root = root / "raw"
    if (raw_root / "train").exists():
        return raw_root
    return root


def split_pairs(dataset_root: str | Path, split: str) -> list[tuple[Path, Path]]:
    root = resolve_split_root(dataset_root) / split
    tif_paths = sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff"))
    pairs: list[tuple[Path, Path]] = []
    for tif_path in tif_paths:
        csv_path = tif_path.with_suffix(".csv")
        if csv_path.exists():
            pairs.append((tif_path, csv_path))
    return pairs


def split_images(dataset_root: str | Path, split: str) -> list[Path]:
    root = resolve_split_root(dataset_root) / split
    return sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff"))


def limit_pairs(pairs: Iterable[tuple[Path, Path]], limit: int | None) -> list[tuple[Path, Path]]:
    out = list(pairs)
    if limit is None:
        return out
    return out[: int(limit)]
