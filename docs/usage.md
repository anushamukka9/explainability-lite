# Usage Guide

This guide walks through the full `explainability-lite` workflow: installing,
picking a method, plugging in your own model, and producing reports.

## Install

```bash
pip install -e .
```

Only dependency: `numpy`.

## Core concepts

Anything with a `predict(X)` method works — `X` is a 2-D numpy array of shape
`(n_samples, n_features)` and the return is a 1-D array of predictions. That
is the whole interface; there is no framework lock-in:

```python
from explainability_lite import explain

attr = explain("lime", my_model, row, X_ref, feature_names=[...])
```

- `row` — the single instance to explain, shape `(n_features,)`.
- `X_ref` — a reference set from the same distribution (training data or a
  representative sample). Used for baselines, perturbation sampling, and
  permutation importance.
- `attr` is an `Attribution` with `.values`, `.ranked()`, `.as_dict()`.

## Choosing a method

| Method | Scope | Cost | Best for |
|---|---|---|---|
| `ablation` | local | O(p) predictions | quick, exact sanity checks |
| `lime` | local | O(n_samples) | smooth, general-purpose local explanations |
| `shap` | local | O(n_permutations × p) | additive attributions that sum to f(row) − f(baseline) |
| `permutation` | global | O(p × n_repeats) | ranking features across the whole dataset |

## Plugging in your own model

Your object just needs `predict`. Example with a wrapped scikit-learn model:

```python
class SklearnWrapper:
    def __init__(self, clf):
        self.clf = clf
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]
```

Pass it to `explain()` directly, or point the CLI at it:

```bash
explainability-lite explain --csv data.csv --row 3 --method shap \
    --model mypackage.models:risk_model --format html --out report.html
```

The spec `mypackage.models:risk_model` imports `mypackage.models` and uses the
`risk_model` attribute, which must expose `predict(X)`. `check_model()` from
`explainability_lite.models` smoke-tests the interface before attribution runs.

## Baselines

`ablation` and `shap` compare against a baseline: `--baseline mean|median|zeros`
(or pass a numpy vector). The median is the default — robust to outliers and a
reasonable "typical instance".

## Reports

```python
from explainability_lite import ascii_bar_chart, markdown_table, save_html_report

print(ascii_bar_chart(attr, top_k=10))   # terminal
print(markdown_table(attr, top_k=10))    # paste into docs/PRs
save_html_report(attr, "report.html",    # standalone file, inline SVG chart
                 feature_values=dict(zip(names, row)))
```

HTML reports are single files with no external assets: they embed the SVG bar
chart and the data table, so they can be attached to issues, PRs, or emails and
opened anywhere.

## Reproducibility

All sampling-based methods take `random_state` (CLI: `--seed`). With a fixed
seed, every run produces identical numbers — the test suite asserts this
implicitly by asserting structural properties on fixed seeds.

## Caveats

- These are *local, model-agnostic approximations*, not causal claims. A
  feature with high attribution influenced this prediction; it did not
  necessarily cause the outcome in the real world.
- Correlated features split credit. If two features carry the same signal,
  permutation and ablation attributions will understate each one individually.
- LIME surrogates are only faithful near the explained instance; check the
  surrogate's fidelity if you change `n_samples` or `kernel_width` sharply.
