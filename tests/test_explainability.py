"""Test suite for explainability-lite."""

import csv
import os
import tempfile

import numpy as np
import pytest

from explainability_lite import (
    Attribution,
    ablation_attribution,
    ascii_bar_chart,
    explain,
    html_report,
    lime_surrogate_attribution,
    make_demo_model,
    markdown_table,
    permutation_importance,
    shap_permutation_attribution,
)


def _dataset(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    names = ["income", "debt", "age", "tenure", "noise"]
    X = rng.normal(0, 1, size=(n, 5))
    # Ground truth: income dominates, debt is strong negative, age weak, rest noise.
    w = np.array([1.4, -1.0, 0.25, 0.0, 0.0])
    logits = X @ w
    y_prob = 1 / (1 + np.exp(-logits))
    y = (y_prob > 0.5).astype(int)
    model = make_demo_model(names, positive=["income"])
    # Override weights so the model matches the known ground truth exactly.
    model.weights = w
    model.bias = 0.0
    return names, X, y, model


def test_permutation_importance_recovers_ranking():
    names, X, y, model = _dataset()
    attr = permutation_importance(model, X, y, feature_names=names, n_repeats=5)
    top = [n for n, _ in attr.ranked(top_k=2)]
    assert top[0] == "income"
    assert "debt" in top
    assert attr.values[names.index("noise")] < attr.values[names.index("income")]


def test_ablation_sign_matches_weights():
    names, X, y, model = _dataset()
    row = np.array([2.0, 2.0, 0.0, 0.0, 0.0])  # high income, high debt
    attr = ablation_attribution(model, row, X, feature_names=names, baseline="zeros")
    d = dict(attr.ranked())
    assert d["income"] > 0  # positive weight pushes prediction up
    assert d["debt"] < 0  # negative weight pushes prediction down


def test_lime_surrogate_recovers_dominant_feature():
    names, X, y, model = _dataset()
    row = X[10]
    attr = lime_surrogate_attribution(model, row, X, feature_names=names, n_samples=1500)
    ranked = attr.ranked()
    assert ranked[0][0] in ("income", "debt")
    assert abs(ranked[0][1]) > abs(dict(ranked)["noise"])


def test_shap_permutation_additivity():
    names, X, y, model = _dataset()
    row = X[3]
    attr = shap_permutation_attribution(
        model, row, X, feature_names=names, baseline="mean", n_permutations=300
    )
    total = float(np.sum(attr.values))
    expected = float(attr.prediction - attr.baseline)
    assert total == pytest.approx(expected, abs=0.05)


def test_explain_dispatch_and_validation():
    names, X, y, model = _dataset()
    row = X[0]
    for method in ("ablation", "lime", "shap"):
        attr = explain(method, model, row, X, feature_names=names)
        assert isinstance(attr, Attribution)
        assert isinstance(attr.method, str) and attr.method
        assert len(attr.values) == len(names)
    with pytest.raises(ValueError):
        explain("nope", model, row, X)
    with pytest.raises(ValueError):
        explain("permutation", model, row, X)  # y_ref required


def test_reports_render_content():
    names, X, y, model = _dataset()
    attr = ablation_attribution(model, X[0], X, feature_names=names)
    ascii_out = ascii_bar_chart(attr, top_k=3)
    assert "income" in ascii_out and "|" in ascii_out
    md = markdown_table(attr, top_k=3)
    assert md.startswith("###") and "`income`" in md
    page = html_report(attr, title="Test report")
    assert "<svg" in page and "</html>" in page and "income" in page


def test_cli_explain_end_to_end(tmp_path):
    from explainability_lite.cli import main

    csv_path = tmp_path / "data.csv"
    names = ["income", "debt", "age"]
    rng = np.random.default_rng(7)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        w.writerows(rng.normal(0, 1, size=(20, 3)).tolist())

    out_path = tmp_path / "report.html"
    rc = main([
        "explain", "--csv", str(csv_path), "--row", "0",
        "--method", "shap", "--format", "html", "--out", str(out_path),
        "--samples", "50",
    ])
    assert rc == 0
    content = out_path.read_text(encoding="utf-8")
    assert "<svg" in content and "shap" in content


def test_attribution_ranked_and_dict():
    attr = Attribution(["a", "b", "c"], np.array([0.1, -0.5, 0.3]), method="x",
                       prediction=0.7, baseline=0.5)
    ranked = attr.ranked()
    assert ranked[0][0] == "b"  # largest absolute value first
    assert ranked[-1][0] == "a"
    d = attr.as_dict()
    assert d["method"] == "x" and len(d["features"]) == 3
    with pytest.raises(ValueError):
        Attribution(["a"], np.array([1.0, 2.0]), method="x")  # length mismatch
