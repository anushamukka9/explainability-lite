# explainability-lite

Model-agnostic **local feature attributions** for tabular data, with reports you can actually share.

Explain any prediction — from any model with a `predict(X)` method — using four attribution methods, then render the result as an ASCII bar chart, a Markdown table, or a standalone HTML report with an inline SVG chart.

## Methods

| Method | Scope | Idea |
|---|---|---|
| `permutation` | global | Shuffle each feature, measure the score drop |
| `ablation` | local | Replace each feature with its baseline, measure the prediction change |
| `lime` | local | LIME-style: sample around the instance, fit a weighted linear surrogate |
| `shap` | local | SHAP-like: average marginal contributions over random permutations (adds up to `f(row) − f(baseline)`) |

Only dependency: **numpy**.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
from explainability_lite import explain, make_demo_model, ascii_bar_chart, save_html_report

rng = np.random.default_rng(0)
names = ["income", "debt_to_income", "credit_age", "num_accounts", "zip_risk"]
X = rng.normal(0, 1, size=(500, len(names)))

model = make_demo_model(names, positive=["income", "credit_age"])
row = np.array([2.5, 3.0, 1.0, 0.5, 0.0])  # high income, very high debt

attr = explain("shap", model, row, X, feature_names=names, random_state=42)
print(ascii_bar_chart(attr))
save_html_report(attr, "report.html", feature_values=dict(zip(names, row)))
```

Or run the bundled example:

```bash
python examples/quickstart.py
```

## CLI

```bash
# List methods
explainability-lite methods

# Explain row 3 of a CSV with the demo model
explainability-lite explain --csv data.csv --row 3 --method lime

# SHAP-style explanation as a standalone HTML report, your own model
explainability-lite explain --csv data.csv --row 3 --method shap \
    --model mypackage.models:risk_model --baseline median \
    --format html --out report.html

# Global permutation importance (needs a label column)
explainability-lite explain --csv data.csv --row 0 --method permutation \
    --label-column approved --format markdown
```

The `--model` spec is either `demo` (built-in demo model) or `module.path:attribute`
for any object exposing `predict(X)`.

## API

```python
from explainability_lite import (
    Attribution,            # result container: .values, .ranked(), .as_dict()
    permutation_importance, # global importance via shuffling
    ablation_attribution,   # local attribution via baseline replacement
    lime_surrogate_attribution,  # LIME-style weighted linear surrogate
    shap_permutation_attribution, # SHAP-like permutation sampling
    explain,                # single dispatcher for all four
    ascii_bar_chart, markdown_table, html_report, save_html_report,
    make_demo_model, check_model,
)
```

## Architecture

```
src/explainability_lite/
├── __init__.py      # public API
├── models.py        # Model protocol, demo model, check_model, metric helpers
├── attribution.py   # the four attribution methods + explain() dispatcher
├── report.py        # ASCII bars, Markdown tables, standalone HTML/SVG reports
├── cli.py           # `explainability-lite` console script
└── __main__.py      # `python -m explainability_lite`
```

Everything is model-agnostic by construction: attribution methods only ever
call `model.predict` on numpy arrays. Baselines default to the reference
set's median (robust to outliers); all sampling is seeded for reproducibility.

See [docs/usage.md](docs/usage.md) for the full usage guide, including plugging
in your own model and the methods' caveats.

## License

MIT — Copyright (c) 2026 Anusha Mukka.
