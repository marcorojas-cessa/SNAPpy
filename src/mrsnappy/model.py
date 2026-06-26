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
    standardize = _normalize_bool(sweep_cfg.get("standardize", True), "svm_sweep.standardize")
    class_weighting = _normalize_class_weighting(sweep_cfg.get("class_weighting", "on"))
    for kernel in kernels:
        scales = kernel_scales if str(kernel).lower() != "linear" else [None]
        degrees = polynomial_orders if str(kernel).lower() in {"poly", "polynomial"} else [2]
        for c_value in box_constraints:
            for gamma in scales:
                for degree in degrees:
                    yield {
                        "kernel": _normalize_kernel(kernel),
                        "C": _normalize_positive_float(c_value, "svm_sweep.box_constraints"),
                        "gamma": _normalize_gamma(gamma) if str(kernel).lower() != "linear" else None,
                        "degree": _normalize_degree(degree),
                        "standardize": standardize,
                        "class_weighting": class_weighting,
                    }


def _normalize_kernel(kernel: Any) -> str:
    text = str(kernel).strip().lower()
    if text in {"poly", "polynomial"}:
        return "polynomial"
    if text not in {"linear", "rbf", "polynomial"}:
        raise ValueError("svm_sweep.kernels must contain only 'linear', 'rbf', or 'polynomial'.")
    return text


def _normalize_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be true or false.")


def _normalize_positive_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return out


def _normalize_gamma(value: Any) -> str | float:
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"auto", "scale"}:
            return text
        return _normalize_positive_float(text, "svm_sweep.kernel_scales")
    return _normalize_positive_float(value, "svm_sweep.kernel_scales")


def _normalize_degree(value: Any) -> int:
    degree = int(value)
    if degree < 1:
        raise ValueError("svm_sweep.polynomial_orders must contain positive integers.")
    return degree


def _normalize_class_weighting(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in {"on", "off"}:
        raise ValueError("svm_sweep.class_weighting must be 'on' or 'off'.")
    return text


def build_svm_pipeline(params: dict[str, Any]) -> Pipeline:
    kernel = _normalize_kernel(params["kernel"])
    class_weighting = _normalize_class_weighting(params.get("class_weighting", "on"))
    class_weight = "balanced" if class_weighting == "on" else None

    kwargs: dict[str, Any] = {
        "kernel": "poly" if kernel == "polynomial" else kernel,
        "C": _normalize_positive_float(params["C"], "svm_sweep.box_constraints"),
        "class_weight": class_weight,
    }
    if kernel == "rbf":
        kwargs["gamma"] = _normalize_gamma(params.get("gamma", "auto"))
    if kernel == "polynomial":
        kwargs["gamma"] = _normalize_gamma(params.get("gamma", "auto"))
        kwargs["degree"] = _normalize_degree(params.get("degree", 2))

    steps: list[tuple[str, Any]] = []
    if _normalize_bool(params.get("standardize", True), "svm_sweep.standardize"):
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
