"""Global effect curves: partial dependence (PDP) and ICE lines.

Partial dependence shows how a model's average prediction changes as one
feature is swept across its range while every other feature stays as
observed. It answers "how does the model respond to this feature, on
average?" - a global view that complements the local, per-row attributions
in :mod:`explainability_lite.attribution`.

ICE (individual conditional expectation) lines show the same sweep for each
reference row individually; the PDP curve is their average. Diverging ICE
lines hint at feature interactions the average hides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PartialDependence:
    """Result of :func:`partial_dependence` for one feature."""

    feature_name: str
    feature_index: int
    grid: np.ndarray            # feature values swept, shape (n_grid,)
    mean_prediction: np.ndarray  # PDP curve: mean over rows, shape (n_grid,)
    ice: np.ndarray | None = None  # per-row curves, shape (n_rows, n_grid)

    def as_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "grid": [float(v) for v in self.grid],
            "mean_prediction": [float(v) for v in self.mean_prediction],
            "ice": None if self.ice is None else self.ice.tolist(),
        }


def _resolve_feature(feature: int | str, feature_names: list[str] | None,
                     n_features: int) -> tuple[int, str]:
    if isinstance(feature, str):
        if not feature_names or feature not in feature_names:
            raise ValueError(
                f"unknown feature {feature!r}; pass feature_names or use an index"
            )
        idx = feature_names.index(feature)
        return idx, feature
    idx = int(feature)
    if not 0 <= idx < n_features:
        raise ValueError(f"feature index {idx} out of range for {n_features} features")
    name = feature_names[idx] if feature_names else f"feature_{idx}"
    return idx, name


def partial_dependence(
    model,
    X_ref: np.ndarray,
    feature: int | str,
    *,
    feature_names: list[str] | None = None,
    n_grid: int = 20,
    grid: np.ndarray | None = None,
    include_ice: bool = False,
    max_rows: int = 200,
    random_state: int = 42,
) -> PartialDependence:
    """Partial dependence of ``model`` on one feature.

    Sweeps ``feature`` across ``grid`` (default: ``n_grid`` quantiles of the
    reference column) and averages predictions over the reference rows.

    Parameters
    ----------
    model: object with ``predict(X)``
    X_ref: reference data, shape (n_rows, n_features)
    feature: feature index or name
    n_grid: grid resolution when ``grid`` is not given
    grid: explicit grid values (overrides ``n_grid``)
    include_ice: also return per-row ICE lines
    max_rows: subsample the reference set to this many rows for speed
    random_state: seed for the subsample (no-op when n_rows <= max_rows)
    """
    X = np.asarray(X_ref, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_ref must be 2-D; got shape {X.shape}")
    n_rows, n_features = X.shape
    names = feature_names or [f"feature_{i}" for i in range(n_features)]
    idx, name = _resolve_feature(feature, names, n_features)

    if n_rows > max_rows:
        rng = np.random.default_rng(random_state)
        X = X[rng.choice(n_rows, size=max_rows, replace=False)]
        n_rows = max_rows

    if grid is None:
        quantiles = np.linspace(0.0, 1.0, n_grid)
        grid_values = np.unique(np.quantile(X[:, idx], quantiles))
        if grid_values.size < 2:
            raise ValueError(f"feature {name!r} is constant; no grid to sweep")
    else:
        grid_values = np.asarray(grid, dtype=float).ravel()
        if grid_values.size < 2:
            raise ValueError("grid needs at least two values")

    curves = np.empty((n_rows, grid_values.size))
    for g, value in enumerate(grid_values):
        Xg = X.copy()
        Xg[:, idx] = value
        curves[:, g] = np.asarray(model.predict(Xg), dtype=float).ravel()

    return PartialDependence(
        feature_name=name,
        feature_index=idx,
        grid=grid_values,
        mean_prediction=curves.mean(axis=0),
        ice=curves if include_ice else None,
    )


def pdp_table(pd: PartialDependence, *, width: int = 40) -> str:
    """Plain-text table plus a tiny ASCII curve of a partial dependence."""
    lines = [f"Partial dependence: {pd.feature_name}", "",
             f"{'value':>12} | {'mean prediction':>15}"]
    span = pd.mean_prediction.max() - pd.mean_prediction.min()
    for value, mean in zip(pd.grid, pd.mean_prediction):
        if span > 0:
            pos = int(round((mean - pd.mean_prediction.min()) / span * (width - 1)))
        else:
            pos = 0
        bar = " " * pos + "*"
        lines.append(f"{value:12.4f} | {mean:15.4f}  {bar}")
    return "\n".join(lines)
