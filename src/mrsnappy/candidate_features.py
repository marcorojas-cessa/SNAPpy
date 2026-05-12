from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def model_type_and_svm_flag(model_obj: Any) -> tuple[str, bool]:
    model_type = str(getattr(model_obj, "model_type", type(model_obj).__name__))
    svm_used = hasattr(model_obj, "decision_function") and model_type != "stage1_pass_through"
    return model_type, svm_used


def accepted_detection_ids(accepted: Any) -> list[int | None]:
    ids: list[int | None] = [None] * len(accepted)
    detection_id = 1
    for idx, keep in enumerate(accepted):
        if bool(keep):
            ids[idx] = detection_id
            detection_id += 1
    return ids


def candidate_feature_columns(
    selected_features: list[str],
    *,
    include_labels: bool = False,
    include_ground_truth: bool = False,
) -> list[str]:
    columns = [
        "image_id",
        "candidate_id",
        "x",
        "y",
        "z",
        "maxima_score",
        "svm_score",
        "model_score",
        "decision_threshold",
        "accepted_by_model",
        "accepted_detection_id",
    ]
    if include_labels:
        columns.append("label")
    if include_ground_truth:
        columns.extend(
            [
                "matched_gt_id",
                "matched_gt_x",
                "matched_gt_y",
                "matched_gt_z",
                "nearest_gt_id",
                "nearest_gt_distance",
            ]
        )
    columns.extend(selected_features)
    return columns


def candidate_feature_rows(
    *,
    image_id: str,
    coords: Any,
    maxima_scores: Any,
    features: pd.DataFrame,
    model_scores: Any,
    decision_threshold: float,
    selected_features: list[str],
    svm_used: bool,
    labels: Any | None = None,
    matched_gt_ids: list[int | None] | None = None,
    matched_gt_coords: Any | None = None,
    nearest_gt_ids: list[int | None] | None = None,
    nearest_gt_distances: Any | None = None,
) -> list[dict[str, Any]]:
    coords = np.asarray(coords, dtype=np.float32)
    maxima_scores = np.asarray(maxima_scores, dtype=np.float32)
    model_scores = np.asarray(model_scores, dtype=np.float32)
    accepted = model_scores > float(decision_threshold)
    detection_ids = accepted_detection_ids(accepted)
    label_values = None if labels is None else np.asarray(labels, dtype=np.int32)
    gt_coords = None if matched_gt_coords is None else np.asarray(matched_gt_coords, dtype=np.float32)
    nearest_distances = None if nearest_gt_distances is None else np.asarray(nearest_gt_distances, dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for idx, coord in enumerate(coords):
        row = {
            "image_id": image_id,
            "candidate_id": int(idx),
            "x": float(coord[2]),
            "y": float(coord[1]),
            "z": float(coord[0]),
            "maxima_score": float(maxima_scores[idx]) if idx < len(maxima_scores) else np.nan,
            "svm_score": float(model_scores[idx]) if idx < len(model_scores) and svm_used else np.nan,
            "model_score": float(model_scores[idx]) if idx < len(model_scores) else np.nan,
            "decision_threshold": float(decision_threshold),
            "accepted_by_model": bool(accepted[idx]) if idx < len(accepted) else False,
            "accepted_detection_id": detection_ids[idx] if idx < len(detection_ids) else None,
        }
        if label_values is not None:
            row["label"] = int(label_values[idx]) if idx < len(label_values) else 0
        if matched_gt_ids is not None:
            matched_id = matched_gt_ids[idx] if idx < len(matched_gt_ids) else None
            row["matched_gt_id"] = matched_id
            has_gt_coord = matched_id is not None and gt_coords is not None and idx < len(gt_coords)
            row["matched_gt_x"] = float(gt_coords[idx, 2]) if has_gt_coord else np.nan
            row["matched_gt_y"] = float(gt_coords[idx, 1]) if has_gt_coord else np.nan
            row["matched_gt_z"] = float(gt_coords[idx, 0]) if has_gt_coord else np.nan
        if nearest_gt_ids is not None:
            row["nearest_gt_id"] = nearest_gt_ids[idx] if idx < len(nearest_gt_ids) else None
        if nearest_distances is not None:
            row["nearest_gt_distance"] = float(nearest_distances[idx]) if idx < len(nearest_distances) else np.nan
        for feature in selected_features:
            row[feature] = float(features.loc[idx, feature]) if feature in features and idx < len(features) else np.nan
        rows.append(row)
    return rows


def write_candidate_features_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def write_candidate_features_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
