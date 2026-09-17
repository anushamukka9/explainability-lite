"""explainability-lite: model-agnostic local feature attributions for tabular data."""

from .attribution import (
    Attribution,
    ablation_attribution,
    explain,
    lime_surrogate_attribution,
    permutation_importance,
    shap_permutation_attribution,
)
from .models import DemoTabularModel, check_model, make_demo_model
from .report import ascii_bar_chart, html_report, markdown_table, save_html_report

__all__ = [
    "Attribution",
    "DemoTabularModel",
    "ablation_attribution",
    "ascii_bar_chart",
    "check_model",
    "explain",
    "html_report",
    "lime_surrogate_attribution",
    "make_demo_model",
    "markdown_table",
    "permutation_importance",
    "save_html_report",
    "shap_permutation_attribution",
]

__version__ = "0.1.0"
