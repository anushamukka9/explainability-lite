"""Model-agnostic local feature-attribution methods.

All methods work on raw numpy arrays and any object exposing
``predict(X) -> np.ndarray`` (see :mod:`explainability_lite.models`).

Methods
-------
- :func:`permutation_importance` — global importance: shuffle each feature
  in a reference set and measure the score drop.
- :func:`ablation_attribution` — local importance: replace each feature of a
  single row with its baseline value and measure the prediction change.
- :func:`lime_surrogate_attribution` — LIME-style: sample perturbed rows
  around the instance, weight by proximity, fit a linear surrogate, and read
  off the coefficients.
- :func:`shap_permutation_attribution` — SHAP-like: average marginal
  contributions over random feature-order permutations (approximation of
  Shapley values). Attributions sum to ``f(row) - f(baseline)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Attribution:
    """Result of one attribution run."""

    feature_names: list[str]
    values: np.ndarray
    method: str
    prediction: float | None = None
    baseline: float | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).ravel()
        if len(self.values) != len(self.feature_names):
            raise ValueError("values/feature_names length mismatch")

    def ranked(self, *, key: str = "abs", top_k: int | None = None) -> list[tuple[str, float]]:
        """(name, value) pairs sorted by importance, most important first."""
        order = np.argsort(-np.abs(self.values) if key == "abs" else -self.values)
        if top_k is not None:
            order = order[:top_k]
        return [(self.feature_names[i], float(self.values[i])) for i in order]

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "features": [
                {"name": n, "attribution": float(v)}
                for n, v in zip(self.feature_names, self.values)
            ],
            "prediction": self.prediction,
            "baseline": self.baseline,
        }


Metric = Callable[[np.ndarray, np.ndarray], float]


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    metric: Metric | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Attribution:
    """Global importance: score drop when each feature is shuffled.

    Higher values mean the feature matters more. Scores are
    ``base_score - shuffled_score`` averaged over ``n_repeats``.
    """
    from .models import brier_score

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    names = feature_names or [f"feature_{i}" for i in range(p)]
    metric = metric or (lambda yt, yp: -brier_score(yt, yp))  # higher = better

    base = metric(y, model.predict(X))
    rng = np.random.default_rng(random_state)
    drops = np.zeros(p)
    for j in range(p):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[j] += base - metric(y, model.predict(Xp))
    return Attribution(names, drops / n_repeats, method="permutation_importance",
                       extra={"base_score": base, "n_repeats": n_repeats})


def _resolve_baseline(baseline: str | np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    if isinstance(baseline, str):
        if baseline == "mean":
            return X_ref.mean(axis=0)
        if baseline == "median":
            return np.median(X_ref, axis=0)
        if baseline == "zeros":
            return np.zeros(X_ref.shape[1])
        raise ValueError(f"unknown baseline {baseline!r}")
    return np.asarray(baseline, dtype=float).ravel()


def ablation_attribution(
    model,
    row: np.ndarray,
    X_ref: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    baseline: str | np.ndarray = "median",
) -> Attribution:
    """Local attribution by ablation: set each feature to its baseline value.

    Attribution = ``f(row) - f(row with feature j at baseline)``. Positive
    means the feature's actual value pushes the prediction up relative to
    the baseline.
    """
    row = np.asarray(row, dtype=float).ravel()
    X_ref = np.asarray(X_ref, dtype=float)
    p = row.shape[0]
    names = feature_names or [f"feature_{i}" for i in range(p)]
    base_vec = _resolve_baseline(baseline, X_ref)
    if base_vec.shape != (p,):
        raise ValueError("baseline length must match number of features")

    f_full = float(np.asarray(model.predict(row.reshape(1, -1))).ravel()[0])
    values = np.zeros(p)
    for j in range(p):
        ablated = row.copy()
        ablated[j] = base_vec[j]
        f_abl = float(np.asarray(model.predict(ablated.reshape(1, -1))).ravel()[0])
        values[j] = f_full - f_abl
    return Attribution(names, values, method="ablation", prediction=f_full,
                       baseline=float(np.asarray(model.predict(base_vec.reshape(1, -1))).ravel()[0]))


def lime_surrogate_attribution(
    model,
    row: np.ndarray,
    X_ref: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    n_samples: int = 1000,
    kernel_width: float = 0.75,
    scale: str | np.ndarray = "std",
    random_state: int = 42,
) -> Attribution:
    """LIME-style local linear surrogate.

    Samples perturbed instances around ``row``, weights them by an
    exponential kernel on standardized Euclidean distance, and fits a
    weighted least-squares linear model. Attributions are the surrogate
    coefficients, scaled by the feature's observed spread so they are
    comparable across features (like LIME's explanations for tabular data).
    """
    row = np.asarray(row, dtype=float).ravel()
    X_ref = np.asarray(X_ref, dtype=float)
    p = row.shape[0]
    names = feature_names or [f"feature_{i}" for i in range(p)]

    if isinstance(scale, str):
        if scale == "std":
            spread = X_ref.std(axis=0)
        elif scale == "iqr":
            q75, q25 = np.percentile(X_ref, [75, 25], axis=0)
            spread = q75 - q25
        else:
            raise ValueError(f"unknown scale {scale!r}")
    else:
        spread = np.asarray(scale, dtype=float).ravel()
    spread = np.where(spread <= 0, 1.0, spread)

    rng = np.random.default_rng(random_state)
    # Sample perturbations; scale them by the empirical covariance of X_ref.
    cov = np.cov(X_ref, rowvar=False) + 1e-6 * np.eye(p)
    perturb = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)
    samples = row + perturb

    dist2 = np.sum(((samples - row) / spread) ** 2, axis=1)
    weights = np.exp(-dist2 / (kernel_width ** 2))

    y_pert = np.asarray(model.predict(samples), dtype=float).ravel()
    # Weighted least squares on the standardized samples, with an intercept
    # (without it, the mean prediction leaks into the coefficients).
    Xs = (samples - row) / spread
    W = np.sqrt(weights)
    Xw = np.column_stack([np.ones(n_samples), Xs]) * W[:, None]
    yw = y_pert * W
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    attributions = coef[1:] * spread  # back to original units, comparable magnitudes

    f_row = float(np.asarray(model.predict(row.reshape(1, -1))).ravel()[0])
    return Attribution(names, attributions, method="lime_surrogate",
                       prediction=f_row, extra={"n_samples": n_samples})


def shap_permutation_attribution(
    model,
    row: np.ndarray,
    X_ref: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    baseline: str | np.ndarray = "median",
    n_permutations: int = 200,
    random_state: int = 42,
) -> Attribution:
    """SHAP-like additive attributions via random permutation sampling.

    Approximates Shapley values: for random feature orderings, features are
    switched from baseline to their actual value one at a time and each
    feature's marginal contribution is recorded. Averaging over orderings
    gives attributions that add up to ``f(row) - f(baseline)`` (up to
    sampling noise).
    """
    row = np.asarray(row, dtype=float).ravel()
    X_ref = np.asarray(X_ref, dtype=float)
    p = row.shape[0]
    names = feature_names or [f"feature_{i}" for i in range(p)]
    base_vec = _resolve_baseline(baseline, X_ref)

    rng = np.random.default_rng(random_state)
    values = np.zeros(p)

    def f(v: np.ndarray) -> float:
        return float(np.asarray(model.predict(v.reshape(1, -1))).ravel()[0])

    f_base = f(base_vec)
    for _ in range(n_permutations):
        order = rng.permutation(p)
        current = base_vec.copy()
        prev = f_base
        for j in order:
            current[j] = row[j]
            new = f(current)
            values[j] += new - prev
            prev = new
    values /= n_permutations
    return Attribution(names, values, method="shap_permutation",
                       prediction=f(row), baseline=f_base,
                       extra={"n_permutations": n_permutations})


_METHODS: dict[str, Callable] = {
    "permutation": permutation_importance,
    "ablation": ablation_attribution,
    "lime": lime_surrogate_attribution,
    "shap": shap_permutation_attribution,
}


def explain(
    method: str,
    model,
    row: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray | None = None,
    *,
    feature_names: list[str] | None = None,
    random_state: int = 42,
    **kwargs,
) -> Attribution:
    """One entry point for every attribution method.

    ``method`` is one of ``permutation`` (global; needs ``y_ref``),
    ``ablation``, ``lime``, ``shap``. Local methods explain ``row`` against
    the reference distribution ``X_ref``.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    if method == "permutation":
        if y_ref is None:
            raise ValueError("permutation importance needs y_ref")
        return permutation_importance(model, X_ref, y_ref, feature_names=feature_names,
                                      random_state=random_state, **kwargs)
    # ablation is deterministic (no sampling), so it takes no random_state.
    method_kwargs = {} if method == "ablation" else {"random_state": random_state}
    return _METHODS[method](model, row, X_ref, feature_names=feature_names,
                            **method_kwargs, **kwargs)
