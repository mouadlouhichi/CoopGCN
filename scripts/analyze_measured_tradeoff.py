#!/usr/bin/env python3
"""Compute descriptive NDCG/Coverage rank correlations from measured records.

This is a sensitivity diagnostic, not a significance test. It uses only the
Python standard library and writes a CSV consumed by the manuscript.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "measured_results.csv"
OUTPUT = ROOT / "paper" / "measured_tradeoff.csv"


def average_ranks(values: list[float]) -> list[float]:
    """Return ascending average ranks, including exact-value ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def main() -> None:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_dataset[row["dataset"]].append(row)

    rows = []
    for dataset, records in by_dataset.items():
        for scope, selected in (
            ("all models", records),
            ("graph models", [r for r in records if r["family"] == "graph"]),
        ):
            rho = spearman(
                [float(r["ndcg_20"]) for r in selected],
                [float(r["coverage_20"]) for r in selected],
            )
            rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "n_models": len(selected),
                    "spearman_ndcg_coverage": f"{rho:.3f}",
                }
            )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['dataset']:12} {row['scope']:12} "
            f"n={row['n_models']} rho={row['spearman_ndcg_coverage']}"
        )


if __name__ == "__main__":
    main()
