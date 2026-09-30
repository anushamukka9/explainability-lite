"""explainability-lite: model-agnostic local feature attributions for tabular data."""

from .attribution import (
    Attribution,
    ablation_attribution,
    explain,
    kernel_shap_attribution,
    lime_surrogate_attribution,
    permutation_importance,
    shap_permutation_attribution,
)
from .effects import PartialDependence, partial_dependence, pdp_table
from .models import DemoTabularModel, check_model, make_demo_model
from .plots import save_attribution_png, save_pdp_png
from .report import ascii_bar_chart, html_report, markdown_table, save_html_report

__all__ = [
    "Attribution",
    "DemoTabularModel",
    "PartialDependence",
    "ablation_attribution",
    "ascii_bar_chart",
    "check_model",
    "explain",
    "html_report",
    "kernel_shap_attribution",
    "lime_surrogate_attribution",
    "make_demo_model",
    "markdown_table",
    "partial_dependence",
    "pdp_table",
    "permutation_importance",
    "save_attribution_png",
    "save_html_report",
    "save_pdp_png",
    "shap_permutation_attribution",
]

__version__ = "0.1.0"
