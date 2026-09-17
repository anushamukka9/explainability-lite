"""Quickstart example: explain a prediction four different ways.

Run from the repo root after installing:

    pip install -e .
    python examples/quickstart.py
"""

import numpy as np

from explainability_lite import (
    ascii_bar_chart,
    explain,
    make_demo_model,
    markdown_table,
    save_html_report,
)


def main() -> None:
    rng = np.random.default_rng(0)
    feature_names = ["income", "debt_to_income", "credit_age", "num_accounts", "zip_risk"]
    X = rng.normal(0, 1, size=(500, len(feature_names)))

    # Demo model: income and credit age push the score up, debt pushes it down.
    model = make_demo_model(feature_names, positive=["income", "credit_age"])

    # A customer with high income but very high debt.
    row = np.array([2.5, 3.0, 1.0, 0.5, 0.0])
    print(f"Prediction for the row: {model.predict(row.reshape(1, -1))[0]:.4f}\n")

    for method in ("ablation", "lime", "shap"):
        attr = explain(method, model, row, X, feature_names=feature_names, random_state=42)
        print("=" * 64)
        print(ascii_bar_chart(attr))
        print()

    # Global importance (needs labels).
    y = (model.predict(X) > 0.5).astype(int)
    global_attr = explain("permutation", model, row, X, y, feature_names=feature_names)
    print("=" * 64)
    print(ascii_bar_chart(global_attr))
    print()
    print(markdown_table(global_attr, top_k=3))

    # Save a standalone HTML report for the SHAP-style explanation.
    shap_attr = explain("shap", model, row, X, feature_names=feature_names, random_state=42)
    path = save_html_report(
        shap_attr,
        "example_report.html",
        title="Quickstart: SHAP-style attribution",
        feature_values=dict(zip(feature_names, row.tolist())),
    )
    print(f"\nHTML report written to {path} — open it in a browser.")


if __name__ == "__main__":
    main()
