from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .candidate_features import (
    candidate_feature_columns,
    candidate_feature_rows,
    model_type_and_svm_flag,
    write_candidate_features_csv,
    write_candidate_features_manifest,
)
from .config import load_config, load_text_config
from .model import load_model
from .optimizer import optimize_native_dataset, optimizer_plan
from .pipeline import detect_image


DEFAULT_OPTIMIZE_CONFIG = "default"


def _default_optimize_config_path() -> Path:
    return Path(str(resources.files("mrsnappy").joinpath("resources/default_optimize.yaml")))


def _load_optimize_template(config: str | Path = DEFAULT_OPTIMIZE_CONFIG) -> dict[str, Any]:
    if str(config).strip().lower() == DEFAULT_OPTIMIZE_CONFIG:
        return load_text_config(_default_optimize_config_path())
    return load_text_config(Path(config))


def init_config(output: str | Path) -> Path:
    """Write the editable default SNAPpy optimizer config."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_default_optimize_config_path(), output)
    return output


def _write_effective_config(payload: dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "effective_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _resolve_out_dir(out_dir: str | Path | None) -> Path:
    if out_dir is None:
        raise ValueError("SNAPpy optimization requires out_dir.")
    return Path(out_dir)


def _prepare_optimize_config(
    *,
    config: str | Path = DEFAULT_OPTIMIZE_CONFIG,
    out_dir: str | Path | None = None,
    train_dir: str | Path | None = None,
    dataset_name: str | None = None,
    match_distance: float | None = None,
) -> Path:
    output_dir = _resolve_out_dir(out_dir)
    payload = _load_optimize_template(config)
    if train_dir is not None:
        payload["dataset_root"] = str(Path(train_dir).resolve())
    if dataset_name is not None:
        payload["dataset_name"] = dataset_name
    if match_distance is not None:
        payload["match_distance"] = float(match_distance)
    if not payload.get("dataset_root"):
        raise ValueError("SNAPpy optimization requires dataset_root in the config or --train-dir.")
    return _write_effective_config(payload, output_dir)


def _resolve_export_flag(value: bool | None, payload: dict[str, Any], key: str) -> bool:
    if value is not None:
        return bool(value)
    return bool(payload.get("exports", {}).get(key, False))


def optimize(
    *,
    config: str | Path = DEFAULT_OPTIMIZE_CONFIG,
    out_dir: str | Path | None = None,
    train_dir: str | Path | None = None,
    dataset_name: str | None = None,
    match_distance: float | None = None,
    export_optimize_report: bool | None = None,
    export_candidate_features: bool | None = None,
) -> Path:
    output_dir = _resolve_out_dir(out_dir)
    effective_config = _prepare_optimize_config(
        config=config,
        out_dir=output_dir,
        train_dir=train_dir,
        dataset_name=dataset_name,
        match_distance=match_distance,
    )
    payload = load_text_config(effective_config)
    return optimize_native_dataset(
        effective_config,
        output_dir,
        export_optimize_report=_resolve_export_flag(export_optimize_report, payload, "export_optimize_report"),
        export_candidate_features=_resolve_export_flag(export_candidate_features, payload, "export_candidate_features"),
    )


def optimize_dry_run(
    *,
    config: str | Path = DEFAULT_OPTIMIZE_CONFIG,
    out_dir: str | Path | None = None,
    train_dir: str | Path | None = None,
    dataset_name: str | None = None,
    match_distance: float | None = None,
) -> dict[str, Any]:
    output_dir = _resolve_out_dir(out_dir)
    effective_config = _prepare_optimize_config(
        config=config,
        out_dir=output_dir,
        train_dir=train_dir,
        dataset_name=dataset_name,
        match_distance=match_distance,
    )
    plan = optimizer_plan(effective_config)
    plan_path = output_dir / "optimizer_plan.dry_run.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    plan["plan_path"] = str(plan_path)
    return plan


def _resolve_model(model: str | Path) -> tuple[Path, dict[str, Any] | None, float | None]:
    path = Path(model)
    if not path.exists():
        raise FileNotFoundError(f"SNAPpy model not found: {model}")
    return path, None, None


def _iter_images(input_path: str | Path | None, input_list: str | Path | None) -> list[Path]:
    if input_list is not None:
        rows = [line.strip() for line in Path(input_list).read_text().splitlines()]
        return [Path(row) for row in rows if row and not row.startswith("#")]
    if input_path is None:
        raise ValueError("Detection requires an input image, input directory, or input list.")
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    return [path]


def _candidate_features_root(output_path: Path, multi_image: bool) -> Path:
    if multi_image or output_path.is_dir():
        return output_path / "export_candidate_features"
    return output_path.parent / "export_candidate_features"


def _write_detections_csv(path: Path, coords: Any, scores: Any) -> Path:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "detection_id": list(range(1, len(coords) + 1)),
            "x": coords[:, 2] if len(coords) else [],
            "y": coords[:, 1] if len(coords) else [],
            "z": coords[:, 0] if len(coords) else [],
            "score": scores if len(scores) else [],
        }
    ).to_csv(path, index=False)
    return path


def _score_candidates_for_image(
    *,
    image: Path,
    recipe: dict[str, Any],
    trained: Any,
    score_threshold: float,
) -> tuple[Any, Any, Any, Any, Any]:
    coords, maxima_scores, features = detect_image(image, recipe)
    selected_features = list(trained.selected_features)
    if len(coords) == 0:
        model_scores = maxima_scores.astype("float32", copy=False)
    else:
        matrix = features[selected_features].to_numpy(dtype="float32")
        model = trained.model
        if hasattr(model, "decision_function"):
            model_scores = model.decision_function(matrix)
        else:
            model_scores = model.predict(matrix)
        model_scores = model_scores.astype("float32", copy=False)
    accepted = model_scores > float(score_threshold)
    return coords, maxima_scores, features, model_scores, accepted


def _write_detection_candidate_features(
    *,
    path: Path,
    image: Path,
    coords: Any,
    maxima_scores: Any,
    features: Any,
    model_scores: Any,
    trained: Any,
    score_threshold: float,
) -> Path:
    selected_features = list(trained.selected_features)
    _, svm_used = model_type_and_svm_flag(trained.model)
    rows = candidate_feature_rows(
        image_id=image.stem,
        coords=coords,
        maxima_scores=maxima_scores,
        features=features,
        model_scores=model_scores,
        decision_threshold=float(score_threshold),
        selected_features=selected_features,
        svm_used=svm_used,
    )
    return write_candidate_features_csv(path, rows, columns=candidate_feature_columns(selected_features))


def _write_detection_candidate_features_manifest(
    *,
    features_root: Path,
    model: str | Path,
    model_path: Path,
    trained: Any,
    score_threshold: float,
    feature_files: list[Path],
) -> Path:
    model_type, svm_used = model_type_and_svm_flag(trained.model)
    manifest_path = features_root / "candidate_features_manifest.json"
    return write_candidate_features_manifest(
        manifest_path,
        {
            "purpose": "Candidate-level SNAPpy detection feature export for the model recipe applied to unlabeled input images.",
            "model": str(model),
            "model_path": str(model_path),
            "model_type": model_type,
            "svm_used": bool(svm_used),
            "selected_features": list(trained.selected_features),
            "decision_threshold": float(score_threshold),
            "coordinate_columns": "x,y,z are subpixel voxel coordinates; z is the stack axis.",
            "maxima_score_definition": "Detector response value at the candidate maximum before Gaussian fitting and model scoring.",
            "svm_score_definition": "SVM decision_function score when the model uses Stage 2 SVM classification; null for Stage 1 pass-through models.",
            "model_score_definition": "Score used for the model acceptance decision. For Stage 2 SVM models this matches svm_score; for Stage 1 pass-through models it is the pass-through model score.",
            "accepted_by_model_definition": "True when model_score is greater than decision_threshold.",
            "accepted_detection_id_definition": "For accepted candidates, this equals the detection_id written in the corresponding detection CSV; null for rejected candidates.",
            "feature_files": [str(path.relative_to(features_root)) for path in feature_files],
        },
    )


def detect(
    *,
    model: str | Path,
    input_path: str | Path | None = None,
    input_list: str | Path | None = None,
    output: str | Path | None = None,
    config: str | Path | None = None,
    score_threshold: float | None = None,
    export_candidate_features: bool = False,
) -> dict[str, list[Path]]:
    model_path, model_recipe, model_threshold = _resolve_model(model)
    trained = load_model(model_path)
    recipe = load_config(config)["pipeline_defaults"] if config else model_recipe
    if recipe is None:
        recipe = trained.recipe
    if recipe is None:
        raise ValueError("Detection requires a pipeline recipe, either embedded in the model or supplied by config.")
    if score_threshold is None:
        score_threshold = trained.decision_threshold if model_threshold is None else model_threshold
    images = _iter_images(input_path, input_list)
    if not images:
        raise ValueError("No input TIFF images found for detection.")
    if output is None:
        raise ValueError("Detection requires --output.")

    output_path = Path(output)
    multi_image = len(images) > 1
    if multi_image:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    detection_outputs: list[Path] = []
    candidate_feature_outputs: list[Path] = []
    features_root = _candidate_features_root(output_path, multi_image) if export_candidate_features else None
    for image in images:
        coords, maxima_scores, features, model_scores, accepted = _score_candidates_for_image(
            image=image,
            recipe=recipe,
            trained=trained,
            score_threshold=float(score_threshold),
        )
        kept_coords = coords[accepted]
        kept_scores = model_scores[accepted]
        out_file = output_path / f"{image.stem}.csv" if multi_image or output_path.is_dir() else output_path
        _write_detections_csv(out_file, kept_coords, kept_scores)
        detection_outputs.append(out_file)
        if features_root is not None:
            feature_file = features_root / f"{image.stem}_candidate_features.csv"
            candidate_feature_outputs.append(
                _write_detection_candidate_features(
                    path=feature_file,
                    image=image,
                    coords=coords,
                    maxima_scores=maxima_scores,
                    features=features,
                    model_scores=model_scores,
                    trained=trained,
                    score_threshold=float(score_threshold),
                )
            )
    if features_root is not None:
        candidate_feature_outputs.append(
            _write_detection_candidate_features_manifest(
                features_root=features_root,
                model=model,
                model_path=model_path,
                trained=trained,
                score_threshold=float(score_threshold),
                feature_files=candidate_feature_outputs,
            )
        )
    return {"detections": detection_outputs, "candidate_features": candidate_feature_outputs}
