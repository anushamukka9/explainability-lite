# Usage Guide

This guide walks through the full `explainability-lite` workflow: installing,
picking a method, plugging in your own model, and producing reports.

## Install

```bash
pip install -e .
pip install -e ".[plots]"   # optional: PNG plotting helpers (matplotlib)
```

Required dependency: `numpy`. The plotting helpers in
`explainability_lite.plots` need matplotlib, installed via the `plots`
extra. They force the non-interactive Agg backend, so PNGs render with no
display server (headless CI, SSH sessions, containers).

## Core concepts

Anything with a `predict(X)` method works: `X` is a 2-D numpy array of shape
`(n_samples, n_features)` and the return is a 1-D array of predictions. That
is the whole interface; there is no framework lock-in:

```python
from explainability_lite import explain

attr = explain("lime", my_model, row, X_ref, feature_names=[...])
```

- `row` - the single instance to explain, shape `(n_features,)`.
- `X_ref` - a reference set from the same distribution (training data or a
  representative sample). Used for baselines, perturbation sampling, and
  permutation importance.
- `attr` is an `Attribution` with `.values`, `.ranked()`, `.as_dict()`.

## Choosing a method

| Method | Scope | Cost | Best for |
|---|---|---|---|
| `ablation` | local | O(p) predictions | quick, exact sanity checks |
| `lime` | local | O(n_samples) | smooth, general-purpose local explanations |
| `shap` | local | O(n_permutations × p) | additive attributions that sum to f(row) − f(baseline) |
| `kernel_shap` | local | O(2^p) predictions | exact Shapley values when p is small (≤ 12 by default); deterministic, no sampling noise |
| `permutation` | global | O(p × n_repeats) | ranking features across the whole dataset |

`kernel_shap` enumerates every feature coalition and solves the
efficiency-constrained weighted least squares problem with the Shapley
kernel. Use it when the feature count is small and you want exact,
reproducible Shapley values; use `shap` (permutation sampling) for larger
feature sets.

## Partial dependence

Local attributions explain one row. Partial dependence describes the model's
global response to a feature: sweep the feature across its range, keep every
other feature as observed, and average the predictions.

```python
from explainability_lite import partial_dependence, pdp_table

pd = partial_dependence(model, X_ref, "income", feature_names=names,
                        n_grid=20, include_ice=True)
print(pdp_table(pd))   # text table with a tiny ASCII curve
pd.mean_prediction     # the PDP curve; pd.ice holds the per-row ICE lines
```

`feature` accepts a name or a 0-based index. The default grid is 20
quantiles of the reference column; pass `grid=[...]` for explicit values.
`max_rows` subsamples large reference sets (seeded, deterministic). The CLI
equivalent:

```bash
explainability-lite pdp --csv data.csv --feature income --ice \
    --format png --out pdp_income.png
```

## PNG plots

`save_attribution_png(attr, path)` and `save_pdp_png(pd, path)` write
matplotlib figures with the Agg backend: no display server needed. They
require the `plots` extra (`pip install -e ".[plots]"`) and raise a clear
error without it.

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
(or pass a numpy vector). The median is the default: robust to outliers and a
reasonable "typical instance".

## Reports

```python
from explainability_lite import ascii_bar_chart, markdown_table, save_html_report

print(ascii_bar_chart(attr, top_k=10))   # terminal
print(markdown_table(attr, top_k=10))    # paste into docs/PRs
save_html_report(attr, "report.html",    # standalone file, inline SVG chart
                 feature_values=dict(zip(names, row)))
```

PNG versions for slides and docs (needs the `plots` extra):

```python
from explainability_lite import save_attribution_png, save_pdp_png

save_attribution_png(attr, "attribution.png", top_k=10)
save_pdp_png(pd, "pdp_income.png")       # pd from partial_dependence(...)
```

HTML reports are single files with no external assets: they embed the SVG bar
chart and the data table, so they can be attached to issues, PRs, or emails and
opened anywhere.

## Reproducibility

All sampling-based methods take `random_state` (CLI: `--seed`). With a fixed
seed, every run produces identical numbers; the test suite asserts this
implicitly by asserting structural properties on fixed seeds. `kernel_shap`
and `partial_dependence` are deterministic by construction (no sampling at
all, apart from the optional `max_rows` subsample).

## Caveats

- These are *local, model-agnostic approximations*, not causal claims. A
  feature with high attribution influenced this prediction; it did not
  necessarily cause the outcome in the real world.
- Correlated features split credit. If two features carry the same signal,
  permutation and ablation attributions will understate each one individually.
- LIME surrogates are only faithful near the explained instance; check the
  surrogate's fidelity if you change `n_samples` or `kernel_width` sharply.
