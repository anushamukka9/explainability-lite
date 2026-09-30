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


# ---------------------------------------------------------------------------
# New: kernel_shap, partial dependence, PNG plots, pdp CLI
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from explainability_lite import (  # noqa: E402
    PartialDependence,
    kernel_shap_attribution,
    partial_dependence,
    pdp_table,
)


class _LinearModel:
    """f(x) = w.x + b: Shapley values are exactly w_j * (row_j - base_j)."""

    def __init__(self, w, b=0.0):
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b


def test_kernel_shap_additivity():
    names, X, y, model = _dataset()
    row = X[3]
    attr = kernel_shap_attribution(model, row, X, feature_names=names)
    assert attr.method == "kernel_shap"
    total = float(np.sum(attr.values))
    expected = float(attr.prediction - attr.baseline)
    assert total == pytest.approx(expected, abs=1e-8)


def test_kernel_shap_matches_linear_ground_truth():
    rng = np.random.default_rng(3)
    w = np.array([2.0, -1.5, 0.5])
    model = _LinearModel(w, b=0.25)
    X = rng.normal(0, 1, size=(100, 3))
    row = np.array([1.0, -2.0, 0.5])
    attr = kernel_shap_attribution(model, row, X, baseline="zeros")
    expected = w * row  # baseline is the zero vector
    np.testing.assert_allclose(attr.values, expected, rtol=1e-6, atol=1e-8)


def test_kernel_shap_is_deterministic():
    names, X, y, model = _dataset()
    a = kernel_shap_attribution(model, X[5], X, feature_names=names)
    b = kernel_shap_attribution(model, X[5], X, feature_names=names)
    np.testing.assert_array_equal(a.values, b.values)


def test_kernel_shap_rejects_too_many_features():
    names, X, y, model = _dataset()
    with pytest.raises(ValueError, match="max_features"):
        kernel_shap_attribution(model, X[0], X, max_features=4)


def test_kernel_shap_single_feature():
    model = _LinearModel([3.0], b=1.0)
    X = np.array([[0.0], [1.0], [2.0]])
    attr = kernel_shap_attribution(model, np.array([2.0]), X, baseline="zeros")
    assert attr.values == pytest.approx([6.0])


def test_explain_dispatches_kernel_shap():
    names, X, y, model = _dataset()
    attr = explain("kernel_shap", model, X[1], X, feature_names=names)
    assert attr.method == "kernel_shap"
    assert len(attr.values) == len(names)


def test_partial_dependence_shapes_and_monotonicity():
    names, X, y, model = _dataset()
    pd = partial_dependence(model, X, "income", feature_names=names, n_grid=15)
    assert isinstance(pd, PartialDependence)
    assert pd.feature_name == "income" and pd.feature_index == 0
    assert pd.grid.shape == (15,)
    assert pd.mean_prediction.shape == (15,)
    assert pd.ice is None
    # income has a positive weight: the PDP curve must rise with income
    assert np.all(np.diff(pd.mean_prediction) > 0)


def test_partial_dependence_by_index_and_ice():
    names, X, y, model = _dataset()
    pd = partial_dependence(model, X, 1, feature_names=names, n_grid=10,
                            include_ice=True, max_rows=50)
    assert pd.feature_name == "debt"
    assert pd.ice is not None and pd.ice.shape == (50, 10)
    # PDP is the mean of the ICE lines
    np.testing.assert_allclose(pd.mean_prediction, pd.ice.mean(axis=0))


def test_partial_dependence_is_deterministic():
    names, X, y, model = _dataset()
    a = partial_dependence(model, X, "age", feature_names=names)
    b = partial_dependence(model, X, "age", feature_names=names)
    np.testing.assert_array_equal(a.mean_prediction, b.mean_prediction)


def test_partial_dependence_rejects_bad_feature():
    names, X, y, model = _dataset()
    with pytest.raises(ValueError, match="unknown feature"):
        partial_dependence(model, X, "nope", feature_names=names)
    with pytest.raises(ValueError, match="out of range"):
        partial_dependence(model, X, 99, feature_names=names)
    Xc = X.copy()
    Xc[:, 3] = 1.0  # constant column
    with pytest.raises(ValueError, match="constant"):
        partial_dependence(model, Xc, "tenure", feature_names=names)


def test_pdp_table_renders():
    names, X, y, model = _dataset()
    pd = partial_dependence(model, X, "income", feature_names=names, n_grid=8)
    text = pdp_table(pd)
    assert "Partial dependence: income" in text
    assert text.count("*") == 8
    assert isinstance(pd.as_dict()["grid"], list)


def test_save_attribution_png(tmp_path):
    pytest.importorskip("matplotlib")
    from explainability_lite import save_attribution_png

    names, X, y, model = _dataset()
    attr = explain("shap", model, X[0], X, feature_names=names)
    out = tmp_path / "attr.png"
    returned = save_attribution_png(attr, str(out), top_k=3)
    assert returned == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_pdp_png(tmp_path):
    pytest.importorskip("matplotlib")
    from explainability_lite import save_pdp_png

    names, X, y, model = _dataset()
    pd = partial_dependence(model, X, "income", feature_names=names,
                            include_ice=True)
    out = tmp_path / "pdp.png"
    save_pdp_png(pd, str(out))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plots_raise_helpful_error_without_matplotlib(monkeypatch):
    import sys

    from explainability_lite import plots

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    names, X, y, model = _dataset()
    attr = explain("ablation", model, X[0], X, feature_names=names)
    with pytest.raises(ImportError, match="explainability-lite\\[plots\\]"):
        plots.save_attribution_png(attr, "x.png")


def _write_demo_csv(path):
    import csv

    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, size=(30, 3))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "c"])
        w.writerows(X.tolist())


def test_cli_pdp_ascii(tmp_path, capsys):
    from explainability_lite.cli import main

    csv_path = tmp_path / "data.csv"
    _write_demo_csv(csv_path)
    rc = main(["pdp", "--csv", str(csv_path), "--feature", "a",
               "--grid-points", "6", "--out", "-"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Partial dependence: a" in out
    assert out.count("*") == 6


def test_cli_pdp_png(tmp_path):
    pytest.importorskip("matplotlib")
    from explainability_lite.cli import main

    csv_path = tmp_path / "data.csv"
    _write_demo_csv(csv_path)
    out = tmp_path / "pdp.png"
    rc = main(["pdp", "--csv", str(csv_path), "--feature", "b",
               "--format", "png", "--out", str(out), "--ice"])
    assert rc == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_methods_lists_new_method(capsys):
    from explainability_lite.cli import main

    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    assert "kernel_shap" in out
