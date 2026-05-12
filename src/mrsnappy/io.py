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


def read_points_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    df = pd.read_csv(path)
    cols = {col.lower(): col for col in df.columns}
    if {"x", "y", "z"} <= set(cols):
        x = df[cols["x"]].to_numpy(dtype=np.float32)
        y = df[cols["y"]].to_numpy(dtype=np.float32)
        z = df[cols["z"]].to_numpy(dtype=np.float32)
        return np.stack([z, y, x], axis=1).astype(np.float32)
    raise ValueError(
        f"Unsupported point CSV format for SNAPpy: {path}. "
        "Expected case-insensitive columns x,y,z. Coordinate orientation "
        "and alignment must be standardized externally before SNAPpy."
    )


def write_points_csv(path: str | Path, coords: np.ndarray, scores: np.ndarray | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
