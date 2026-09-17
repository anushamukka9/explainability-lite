"""Command-line interface for explainability-lite."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .attribution import _METHODS, explain
from .models import check_model, load_model, make_demo_model
from .report import ascii_bar_chart, html_report, markdown_table


def load_csv(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"{path} needs a header row and at least one data row")
    header, data = rows[0], rows[1:]
    try:
        X = np.array([[float(v) for v in r] for r in data], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{path} must contain only numeric values: {exc}") from exc
    if any(len(r) != len(header) for r in data):
        raise ValueError(f"{path} has rows with a different width than the header")
    return header, X


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="explainability-lite",
        description="Model-agnostic local feature attributions for tabular data.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("explain", help="Explain one prediction from a CSV row.")
    e.add_argument("--csv", required=True, help="CSV file with header + numeric rows.")
    e.add_argument("--row", type=int, required=True, help="0-based row index to explain.")
    e.add_argument("--method", default="lime",
                   choices=sorted(_METHODS),
                   help="Attribution method (default: lime).")
    e.add_argument("--model", default="demo",
                   help="'demo' for the built-in demo model, or 'module.path:attr' "
                        "for your own object with predict(X).")
    e.add_argument("--label-column", default=None,
                   help="Optional label column name; used only by permutation importance.")
    e.add_argument("--baseline", default="median", choices=["mean", "median", "zeros"],
                   help="Baseline strategy for ablation/shap (default: median).")
    e.add_argument("--format", default="ascii", choices=["ascii", "markdown", "html", "all"],
                   help="Output format (default: ascii).")
    e.add_argument("--out", default=None,
                   help="Output file (html format writes here; defaults to report.html). "
                        "Use '-' for stdout.")
    e.add_argument("--top-k", type=int, default=None, help="Show only the top K features.")
    e.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    e.add_argument("--samples", type=int, default=1000,
                   help="Perturbation samples for lime/shap (default: 1000).")

    sub.add_parser("methods", help="List available attribution methods.")
    return p


def _resolve_model(spec: str, feature_names: list[str]):
    model = load_model(spec)
    if model is None:  # "demo"
        model = make_demo_model(feature_names)
    check_model(model, len(feature_names))
    return model


def cmd_explain(args: argparse.Namespace) -> int:
    feature_names, X = load_csv(args.csv)
    if not 0 <= args.row < len(X):
        print(f"error: --row {args.row} out of range (0..{len(X) - 1})", file=sys.stderr)
        return 2
    row = X[args.row]

    ref_names = feature_names
    y_ref = None
    if args.label_column:
        if args.label_column not in feature_names:
            print(f"error: label column {args.label_column!r} not in CSV header",
                  file=sys.stderr)
            return 2
        j = feature_names.index(args.label_column)
        y_ref = X[:, j]
        keep = [i for i in range(len(feature_names)) if i != j]
        ref_names, X, row = [feature_names[i] for i in keep], X[:, keep], row[keep]

    model = _resolve_model(args.model, ref_names)

    kwargs: dict = {}
    if args.method in ("ablation", "shap"):
        kwargs["baseline"] = args.baseline
    if args.method == "lime":
        kwargs["n_samples"] = args.samples
    if args.method == "shap":
        kwargs["n_permutations"] = args.samples

    attr = explain(args.method, model, row, X, y_ref,
                   feature_names=ref_names, random_state=args.seed, **kwargs)

    want = {"all": ("ascii", "markdown", "html")}.get(args.format, (args.format,))
    feature_values = dict(zip(ref_names, row.tolist()))
    outputs: dict[str, str] = {}
    if "ascii" in want:
        outputs["ascii"] = ascii_bar_chart(attr, top_k=args.top_k)
    if "markdown" in want:
        outputs["markdown"] = markdown_table(attr, top_k=args.top_k)
    if "html" in want:
        outputs["html"] = html_report(
            attr,
            title=f"Feature Attribution — row {args.row} ({args.method})",
            feature_values=feature_values,
            top_k=args.top_k,
        )

    if "html" in want and args.out not in (None, "-"):
        out_path = args.out or "report.html"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(outputs["html"])
        print(f"wrote {out_path}")
        for key in ("ascii", "markdown"):
            if key in outputs:
                print(outputs[key])
    else:
        out = args.out or "-"
        text = "\n\n".join(outputs[k] for k in ("ascii", "markdown", "html") if k in outputs)
        if out == "-":
            print(text)
        else:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "methods":
        print("Available attribution methods:")
        for name in sorted(_METHODS):
            print(f"  {name}")
        return 0
    if args.command == "explain":
        return cmd_explain(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
