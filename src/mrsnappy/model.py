from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_FORMAT = "mrsnappy_trained_model"
MODEL_SCHEMA_VERSION = 1


class AcceptAllCandidatesModel:
    """Classifier-compatible model that keeps every Stage 1 candidate."""

    model_type = "stage1_pass_through"

    def fit(self, x_train, y_train=None):
        return self

    def decision_function(self, x) -> np.ndarray:
        return np.ones(len(x), dtype=np.float32)

    def predict(self, x) -> np.ndarray:
        return np.ones(len(x), dtype=np.int8)


@dataclass
class TrainedModel:
    model: Pipeline | AcceptAllCandidatesModel
    selected_features: list[str]
    best_params: dict[str, Any]
    decision_threshold: float = 0.0
    recipe: dict[str, Any] | None = None
    validation_metas: list[dict[str, Any]] | None = None


def iter_svm_param_grid(sweep_cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    kernels = sweep_cfg.get("kernels", ["linear", "rbf"])
    box_constraints = sweep_cfg.get("box_constraints", [1.0])
    kernel_scales = sweep_cfg.get("kernel_scales", ["auto"])
    polynomial_orders = sweep_cfg.get("polynomial_orders", [2])
    standardize = bool(sweep_cfg.get("standardize", True))
    class_weight_mode = str(sweep_cfg.get("class_weight_mode", "balanced")).strip().lower()
    class_weights = sweep_cfg.get("class_weights")
    for kernel in kernels:
        scales = kernel_scales if str(kernel).lower() != "linear" else [None]
        degrees = polynomial_orders if str(kernel).lower() in {"poly", "polynomial"} else [2]
        for c_value in box_constraints:
            for gamma in scales:
                for degree in degrees:
                    yield {
                        "kernel": kernel,
                        "C": float(c_value),
                        "gamma": gamma,
                        "degree": int(degree),
                        "standardize": standardize,
                        "class_weight_mode": class_weight_mode,
                        "class_weights": class_weights,
                    }


def build_svm_pipeline(params: dict[str, Any]) -> Pipeline:
    kernel = str(params["kernel"]).lower()
    class_weight_mode = str(params.get("class_weight_mode", "balanced")).lower()
    if params.get("class_weights") is not None:
        class_weight = params["class_weights"]
    elif class_weight_mode == "balanced":
        class_weight = "balanced"
    else:
        class_weight = None

    kwargs: dict[str, Any] = {
        "kernel": "poly" if kernel in {"poly", "polynomial"} else kernel,
        "C": float(params["C"]),
        "class_weight": class_weight,
    }
    if kernel == "rbf":
        kwargs["gamma"] = params.get("gamma", "auto")
    if kernel in {"poly", "polynomial"}:
        kwargs["gamma"] = params.get("gamma", "auto")
        kwargs["degree"] = int(params.get("degree", 2))

    steps: list[tuple[str, Any]] = []
    if bool(params.get("standardize", True)):
        steps.append(("scale", StandardScaler()))
    steps.append(("svc", SVC(**kwargs)))
    return Pipeline(steps)


def fit_svm_pipeline(x_train, y_train, params: dict[str, Any]) -> Pipeline:
    model = build_svm_pipeline(params)
    model.fit(x_train, y_train)
    return model


def save_model(path: str | Path, trained: TrainedModel) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_format": MODEL_FORMAT,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model": trained.model,
            "selected_features": trained.selected_features,
            "best_params": trained.best_params,
            "decision_threshold": trained.decision_threshold,
            "recipe": deepcopy(trained.recipe),
        },
        path,
    )
    return path


def load_model(path: str | Path) -> TrainedModel:
    path = Path(path)
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Unsupported SNAPpy model file: {path}. Expected a mrsnappy model saved by `mrsnappy optimize`."
        )
    if payload.get("model_format") != MODEL_FORMAT:
        raise ValueError(
            f"Unsupported SNAPpy model file: {path}. This file does not declare model_format={MODEL_FORMAT!r}. "
            "Regenerate the model with the current `mrsnappy optimize` command."
        )
    if int(payload.get("model_schema_version", -1)) != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SNAPpy model schema in {path}: {payload.get('model_schema_version')!r}. "
            f"Expected schema {MODEL_SCHEMA_VERSION}."
        )
    required = {"model", "selected_features", "best_params", "decision_threshold", "recipe"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Invalid SNAPpy model file {path}; missing required field(s): {', '.join(missing)}.")
    return TrainedModel(
        model=payload["model"],
        selected_features=list(payload["selected_features"]),
        best_params=dict(payload.get("best_params", {})),
        decision_threshold=float(payload.get("decision_threshold", 0.0)),
        recipe=deepcopy(payload.get("recipe")),
    )
