"""PNG plotting helpers (optional matplotlib dependency).

These helpers save plots to PNG files with no display server: matplotlib's
non-interactive Agg backend is selected before pyplot is imported, so they
work on headless CI runners and servers.

matplotlib is an *optional* dependency. Install it with::

    pip install explainability-lite[plots]

Every helper raises an informative ImportError when matplotlib is missing.
"""

from __future__ import annotations

from .attribution import Attribution
from .effects import PartialDependence

POSITIVE_COLOR = "#1a7f37"
NEGATIVE_COLOR = "#cf222e"
ICE_COLOR = "#9ca3af"
PDP_COLOR = "#1f77b4"


def _require_pyplot():
    """Import pyplot with the Agg backend, or raise a helpful error."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display server needed
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib: pip install explainability-lite[plots]"
        ) from exc


def save_attribution_png(
    attr: Attribution,
    path: str,
    *,
    top_k: int | None = None,
    width: float = 9.0,
    dpi: int = 120,
) -> str:
    """Save a horizontal diverging bar chart of an attribution as PNG.

    Returns the path written.
    """
    plt = _require_pyplot()
    ranked = attr.ranked(top_k=top_k)
    if not ranked:
        raise ValueError("attribution has no features to plot")
    names = [n for n, _ in ranked][::-1]  # largest bar on top
    values = [v for _, v in ranked][::-1]
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

    height = max(2.5, 0.5 * len(names) + 1.0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(names, values, color=colors, edgecolor="white")
    ax.axvline(0, color="#6e7781", linewidth=1)
    title = f"{attr.method} attributions"
    if attr.prediction is not None:
        title += f"  (prediction={attr.prediction:.4f})"
    ax.set_title(title)
    ax.set_xlabel("attribution")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_pdp_png(
    pd: PartialDependence,
    path: str,
    *,
    show_ice: bool = True,
    max_ice: int = 50,
    width: float = 8.0,
    height: float = 5.0,
    dpi: int = 120,
) -> str:
    """Save a partial-dependence curve (with optional ICE lines) as PNG.

    Returns the path written.
    """
    plt = _require_pyplot()
    fig, ax = plt.subplots(figsize=(width, height))
    if show_ice and pd.ice is not None:
        for line in pd.ice[:max_ice]:
            ax.plot(pd.grid, line, color=ICE_COLOR, alpha=0.25, linewidth=1)
    ax.plot(
        pd.grid,
        pd.mean_prediction,
        color=PDP_COLOR,
        linewidth=2.5,
        label="mean (PDP)",
    )
    ax.set_xlabel(pd.feature_name)
    ax.set_ylabel("mean prediction")
    ax.set_title(f"Partial dependence: {pd.feature_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
