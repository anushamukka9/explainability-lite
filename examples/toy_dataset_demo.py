#!/usr/bin/env python3
"""Worked example on a toy loan-approval dataset.

Uses examples/toy_loan_data.csv (48 synthetic applicants; the ground-truth
rule is documented below) to walk through the whole toolbox:

1. permutation importance - which features matter globally
2. partial dependence - how approval odds move with income / debt
3. kernel_shap - exact Shapley values for one declined applicant
4. PNG plots (needs `pip install explainability-lite[plots]`) and an HTML report

Run from the repo root:

    pip install -e ".[plots]"   # plots extra is optional; PNG steps are skipped without it
    python examples/toy_dataset_demo.py

Outputs land in examples/output/.
"""

import csv
from pathlib import Path

import numpy as np

from explainability_lite import (
    DemoTabularModel,
    ascii_bar_chart,
    explain,
    partial_dependence,
    pdp_table,
    save_html_report,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# Ground-truth rule used to generate toy_loan_data.csv:
#   logit = 1.2*income - 1.0*debt_to_income + 0.4*credit_age + 0.1*num_accounts - 0.3
# income helps a lot, debt hurts a lot, credit age helps a little,
# num_accounts is noise. A good explanation method should recover this.
WEIGHTS = np.array([1.2, -1.0, 0.4, 0.1])
BIAS = -0.3


def load_toy():
    with open(HERE / "toy_loan_data.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    names, data = rows[0], rows[1:]
    X = np.array([[float(v) for v in r[:4]] for r in data])
    y = np.array([int(r[4]) for r in data])
    return names[:4], X, y


def main() -> None:
    OUT.mkdir(exist_ok=True)
    names, X, y = load_toy()
    model = DemoTabularModel(WEIGHTS, BIAS)
    print(f"toy data: {X.shape[0]} applicants, features {names}")
    print(f"approval rate: {y.mean():.2f}\n")

    # 1. Global importance: shuffling income or debt should hurt the score most.
    print("=" * 64)
    print("1. Permutation importance (global)")
    global_attr = explain("permutation", model, X[0], X, y, feature_names=names,
                          random_state=42)
    print(ascii_bar_chart(global_attr))
    print()

    # 2. Partial dependence: approval odds should rise with income,
    #    fall with debt_to_income, and barely move with num_accounts.
    print("=" * 64)
    print("2. Partial dependence (global effect curves)")
    try:
        from explainability_lite import save_pdp_png

        have_plots = True
    except ImportError:
        have_plots = False
    for feature in ("income", "debt_to_income", "num_accounts"):
        pd = partial_dependence(model, X, feature, feature_names=names,
                                n_grid=12, include_ice=True)
        print(pdp_table(pd))
        print()
        if have_plots:
            path = save_pdp_png(pd, str(OUT / f"pdp_{feature}.png"))
            print(f"  plot -> {path}")
    if not have_plots:
        print("  (install the plots extra for PNG curves: pip install -e \".[plots]\")")
    print()

    # 3. Local explanation: pick a declined applicant and explain why.
    print("=" * 64)
    print("3. KernelSHAP on one declined applicant (exact Shapley values)")
    probs = model.predict(X)
    declined = int(np.argmin(probs))
    row = X[declined]
    print(f"applicant #{declined}: features {dict(zip(names, np.round(row, 3)))}")
    print(f"approval probability: {probs[declined]:.4f} -> declined\n")
    attr = explain("kernel_shap", model, row, X, feature_names=names)
    print(ascii_bar_chart(attr))
    total = float(np.sum(attr.values))
    print(f"\nattributions sum to {total:+.4f} = "
          f"f(row) - f(baseline) = {attr.prediction - attr.baseline:+.4f}")

    html_path = save_html_report(
        attr,
        str(OUT / "kernel_shap_report.html"),
        title=f"Toy loan data: KernelSHAP for applicant #{declined}",
        feature_values=dict(zip(names, row.tolist())),
    )
    print(f"\nHTML report -> {html_path}")


if __name__ == "__main__":
    main()
