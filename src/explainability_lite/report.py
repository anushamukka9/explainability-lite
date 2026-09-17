"""Render attributions as ASCII bar charts, Markdown tables, and HTML reports.

The HTML reports are fully standalone (inline SVG, no external assets), so a
report file can be opened in any browser or attached to an issue/PR.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone

import numpy as np

from .attribution import Attribution

POSITIVE_COLOR = "#1a7f37"
NEGATIVE_COLOR = "#cf222e"
NEUTRAL_COLOR = "#6e7781"


def ascii_bar_chart(attr: Attribution, *, width: int = 50, top_k: int | None = None) -> str:
    """ASCII bar chart of an attribution, e.g. for terminal output."""
    ranked = attr.ranked(top_k=top_k)
    if not ranked:
        return "(no features)"
    max_val = max(abs(v) for _, v in ranked) or 1.0
    name_w = max(len(n) for n, _ in ranked)
    lines = [f"{attr.method} attributions (prediction={_fmt(attr.prediction)})"]
    for name, val in ranked:
        bar_len = int(round(abs(val) / max_val * width))
        bar = ("+" if val >= 0 else "-") * bar_len
        lines.append(f"{name:>{name_w}} | {bar:<{width}} {_fmt(val)}")
    return "\n".join(lines)


def markdown_table(attr: Attribution, *, top_k: int | None = None) -> str:
    """Markdown table of an attribution."""
    ranked = attr.ranked(top_k=top_k)
    lines = [
        f"### Feature attributions — `{attr.method}`",
        "",
        f"Prediction: `{_fmt(attr.prediction)}`"
        + (f" · Baseline: `{_fmt(attr.baseline)}`" if attr.baseline is not None else ""),
        "",
        "| Rank | Feature | Attribution |",
        "| ---: | :------ | ----------: |",
    ]
    for i, (name, val) in enumerate(ranked, 1):
        lines.append(f"| {i} | `{name}` | {_fmt(val)} |")
    return "\n".join(lines)


def _svg_bar_chart(
    names: list[str], values: list[float], *, width: int = 640, bar_h: int = 22
) -> str:
    """Inline SVG horizontal bar chart with diverging colors around zero."""
    max_val = max([abs(v) for v in values] + [1e-12])
    pad_l, pad_r = 190, 90
    inner = width - pad_l - pad_r
    zero_x = pad_l + inner / 2
    scale = (inner / 2) / max_val
    height = bar_h * len(names) + 40
    name_font = "font-family='monospace' font-size='12'"

    parts = [f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append(f"<line x1='{zero_x}' y1='10' x2='{zero_x}' y2='{height - 25}' "
                 f"stroke='#d0d7de' stroke-width='1'/>")
    for i, (name, val) in enumerate(zip(names, values)):
        y = 12 + i * bar_h
        w = abs(val) * scale
        x = zero_x if val >= 0 else zero_x - w
        color = POSITIVE_COLOR if val >= 0 else NEGATIVE_COLOR
        parts.append(
            f"<rect x='{x:.1f}' y='{y}' width='{max(w, 1.5):.1f}' height='{bar_h - 6}' "
            f"rx='3' fill='{color}'/>"
        )
        parts.append(
            f"<text x='{pad_l - 8}' y='{y + bar_h / 2 - 2}' text-anchor='end' {name_font} "
            f"fill='#24292f'>{html.escape(name, quote=True)}</text>"
        )
        label_x = x + w + 6 if val >= 0 else x - 6
        anchor = "start" if val >= 0 else "end"
        parts.append(
            f"<text x='{label_x:.1f}' y='{y + bar_h / 2 - 2}' text-anchor='{anchor}' "
            f"{name_font} fill='#57606a'>{val:+.4f}</text>"
        )
    parts.append(
        f"<text x='{width - 10}' y='{height - 8}' text-anchor='end' {name_font} "
        f"fill='#8b949e'>explainability-lite</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def html_report(
    attr: Attribution,
    *,
    title: str = "Feature Attribution Report",
    feature_values: dict[str, float] | None = None,
    top_k: int | None = None,
) -> str:
    """Standalone HTML report with an inline SVG chart and a data table."""
    ranked = attr.ranked(top_k=top_k)
    names = [n for n, _ in ranked]
    values = [v for _, v in ranked]
    svg = _svg_bar_chart(names, values)

    rows = []
    for i, (name, val) in enumerate(ranked, 1):
        color = POSITIVE_COLOR if val >= 0 else NEGATIVE_COLOR
        fv = ""
        if feature_values and name in feature_values:
            fv = f"{feature_values[name]:.4f}"
        rows.append(
            f"<tr><td>{i}</td><td><code>{html.escape(name, quote=True)}</code></td>"
            f"<td>{fv}</td>"
            f"<td style='color:{color};font-weight:bold'>{val:+.4f}</td></tr>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pred_line = f"<p><strong>Prediction:</strong> {_fmt(attr.prediction)}"
    if attr.baseline is not None:
        pred_line += f" &nbsp;·&nbsp; <strong>Baseline:</strong> {_fmt(attr.baseline)}"
    pred_line += "</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title, quote=True)}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
          max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #24292f; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f6f8fa; }}
  td:nth-child(1), td:nth-child(4) {{ text-align: right; }}
  .meta {{ color: #57606a; font-size: 0.85rem; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 4px; }}
</style>
</head>
<body>
<h1>{html.escape(title, quote=True)}</h1>
<p class="meta">Method: <code>{html.escape(attr.method, quote=True)}</code> ·
   Generated: {generated} · <a href="https://github.com/anushamukka9/explainability-lite">explainability-lite</a></p>
{pred_line}
<div class="legend">
  <span style="background:{POSITIVE_COLOR}"></span> pushes prediction up
  &nbsp;&nbsp;<span style="background:{NEGATIVE_COLOR}"></span> pushes prediction down
</div>
{svg}
<table>
  <thead><tr><th>Rank</th><th>Feature</th><th>Value</th><th>Attribution</th></tr></thead>
  <tbody>
{''.join(rows)}
  </tbody>
</table>
</body>
</html>
"""


def save_html_report(
    attr: Attribution,
    path: str,
    *,
    title: str = "Feature Attribution Report",
    feature_values: dict[str, float] | None = None,
    top_k: int | None = None,
) -> str:
    """Write a standalone HTML report; returns the path written."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_report(attr, title=title, feature_values=feature_values, top_k=top_k))
    return path


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return f"{v:.4f}"
