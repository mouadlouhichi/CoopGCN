#!/usr/bin/env python3
"""Build clearly-labelled confirmatory tables and SVG figures from CSV records.

Rows with status != 'complete' render as pending and are never converted into
scientific claims. This script uses only the Python standard library.
"""
from __future__ import annotations
import csv, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "expected"
TABLES = ROOT / "experiments" / "tables"
FIGURES = ROOT / "experiments" / "figures"
STATUS = ROOT / "experiments" / "ARTIFACT_STATUS.md"


def esc_tex(value: str) -> str:
    return (value.replace("\\", r"\textbackslash{}")
                 .replace("_", r"\_").replace("%", r"\%")
                 .replace("&", r"\&").replace("#", r"\#"))


def build_table(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f)); fields = list(rows[0]) if rows else []
    complete = sum(r.get("status", "").lower() == "complete" for r in rows)
    cols = "l" * len(fields)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Confirmatory record: {esc_tex(path.stem)}. Pending rows are not results.}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(rf"\textbf{{{esc_tex(x)}}}" for x in fields) + r" \\",
             r"\midrule"]
    for row in rows:
        vals = [row.get(k, "") if row.get("status", "").lower() == "complete" or k in {"dataset","model","variant","seed","status","notes","split","method","k","noise_ratio","L"} else "--" for k in fields]
        lines.append(" & ".join(esc_tex(str(v)) for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / f"{path.stem}.tex").write_text("\n".join(lines), encoding="utf-8")
    return len(rows), complete


def build_status_svg(name: str, total: int, complete: int) -> None:
    width, height = 760, 180
    frac = complete / total if total else 0
    bar = int(680 * frac)
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="40" y="45" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(name)}</text>
<text x="40" y="78" font-family="sans-serif" font-size="16">confirmatory completion: {complete}/{total} rows</text>
<rect x="40" y="105" width="680" height="28" fill="#dddddd"/>
<rect x="40" y="105" width="{bar}" height="28" fill="#333333"/>
<text x="40" y="158" font-family="sans-serif" font-size="14">PENDING — not a scientific result until rows are complete.</text>
</svg>'''
    (FIGURES / f"{name}.svg").write_text(body, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    summary = ["# Confirmatory artifact status", "", "Generated from `experiments/expected/*.csv`.", "",
               "| Record | Rows | Complete | Status |", "|---|---:|---:|---|"]
    for path in sorted(SRC.glob("*.csv")):
        total, complete = build_table(path)
        build_status_svg(path.stem, total, complete)
        state = "complete" if total and total == complete else "pending"
        summary.append(f"| `{path.name}` | {total} | {complete} | **{state}** |")
    summary += ["", "> Pending tables and SVGs are workflow artifacts, not manuscript evidence.", ""]
    STATUS.write_text("\n".join(summary), encoding="utf-8")
    print(STATUS.read_text())

if __name__ == "__main__":
    main()
