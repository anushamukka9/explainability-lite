"""Pluggable model interface and a demo model for explainability-lite.

Any model that implements ``predict(X)`` with the signature below works with
every attribution method in :mod:`explainability_lite.attribution`:

    predict(X: np.ndarray) -> np.ndarray
        X has shape (n_samples, n_features); returns shape (n_samples,).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Model(Protocol):
    """Minimal model interface: a callable ``predict`` on 2-D numpy arrays."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def check_model(model: Model, n_features: int) -> None:
    """Smoke-test a model before attribution runs."""
    X = np.zeros((2, n_features), dtype=float)
    out = np.asarray(model.predict(X), dtype=float).ravel()
    if out.shape != (2,):
        raise ValueError(
            f"model.predict must return shape (n_samples,); got {out.shape}"
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class DemoTabularModel:
    """A tiny, transparent logistic model for tabular data (demo only).

    Predicts ``P(y=1)`` from a fixed weight vector. Useful for the CLI and
    examples; it is intentionally simple so users can sanity-check that the
    attribution methods recover sensible feature rankings.
    """

    def __init__(self, weights: np.ndarray, bias: float = 0.0) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.n_features = self.weights.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"expected (n_samples, {self.n_features}); got {X.shape}"
            )
        return _sigmoid(X @ self.weights + self.bias)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DemoTabularModel(n_features={self.n_features}, "
            f"weights={self.weights.tolist()}, bias={self.bias})"
        )


def make_demo_model(
    feature_names: list[str],
    *,
    positive: list[str] | None = None,
    seed: int = 42,
) -> DemoTabularModel:
    """Build a reproducible demo model where some features push positive.

    The first few ``positive``-named features get weight +0.9, the rest get
    small random weights, so the true importance ranking is roughly known —
    a good ground truth for trying out the attribution methods.
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.25, 0.25, size=len(feature_names))
    strong = positive or feature_names[:3]
    for i, name in enumerate(feature_names):
        if name in strong:
            weights[i] = 0.9 - 0.1 * i
    bias = float(rng.uniform(-0.5, 0.5))
    return DemoTabularModel(weights, bias)


def load_model(spec: str):
    """Load a model from a spec string.

    ``"demo"`` builds a small :class:`DemoTabularModel` from the CSV header
    (done in the CLI where the feature names are known). Any other spec is
    ``"module.path:attr"`` — import the module and grab the attribute, which
    must expose ``predict(X)``.
    """
    if spec == "demo":
        return None  # handled by the caller, which knows feature names
    if ":" not in spec:
        raise ValueError(
            f"bad model spec {spec!r}; use 'demo' or 'module.path:attribute'"
        )
    module_path, _, attr = spec.rpartition(":")
    import importlib

    module = importlib.import_module(module_path)
    model = getattr(module, attr)
    if not hasattr(model, "predict"):
        raise ValueError(f"{spec!r} has no 'predict' method")
    return model


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of predicted probabilities (lower is better)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((y_true - y_prob) ** 2))
