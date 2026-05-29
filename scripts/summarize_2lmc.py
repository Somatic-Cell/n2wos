#!/usr/bin/env python3
"""Summarize n2wos_eval_2lmc JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Iterable, Sequence

COLUMNS = [
    "path", "backend", "cache", "m", "exact",
    "pure_mean", "pure_var", "pure_usec_per_sample", "pure_queries_per_sample", "pure_truncated",
    "coarse_mean", "coarse_var", "coarse_usec_per_sample", "coarse_queries_per_sample", "coarse_bias",
    "residual_mean", "residual_var", "residual_usec_per_sample", "residual_queries_per_sample", "residual_truncated",
    "two_level_mean", "two_level_bias", "two_level_mse", "two_level_mse_time", "speedup_score_vs_pure",
]


def iter_json_files(paths: Sequence[str]) -> Iterable[pathlib.Path]:
    for text in paths:
        p = pathlib.Path(text)
        if p.is_dir():
            yield from sorted(p.rglob("*.json"))
        elif p.is_file():
            yield p


def get(d: dict[str, Any] | None, path: str, default: Any = "") -> Any:
    cur: Any = d
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def flatten(path: pathlib.Path, data: dict[str, Any]) -> dict[str, Any]:
    pure = get(data, "estimators.pure_wos", None)
    coarse = get(data, "estimators.coarse", None)
    residual = get(data, "estimators.coupled_residual", None)
    return {
        "path": str(path),
        "backend": data.get("backend", ""),
        "cache": data.get("cache", ""),
        "m": get(data, "args.m"),
        "exact": get(data, "target.exact_value_at_x0"),
        "pure_mean": get(pure, "stats.mean"),
        "pure_var": get(pure, "stats.variance"),
        "pure_usec_per_sample": get(pure, "usec_per_sample"),
        "pure_queries_per_sample": get(pure, "queries_per_sample"),
        "pure_truncated": get(pure, "truncated_count"),
        "coarse_mean": get(coarse, "stats.mean"),
        "coarse_var": get(coarse, "stats.variance"),
        "coarse_usec_per_sample": get(coarse, "usec_per_sample"),
        "coarse_queries_per_sample": get(coarse, "queries_per_sample"),
        "coarse_bias": get(coarse, "stats.bias"),
        "residual_mean": get(residual, "residual_stats.mean"),
        "residual_var": get(residual, "residual_stats.variance"),
        "residual_usec_per_sample": get(residual, "usec_per_sample"),
        "residual_queries_per_sample": get(residual, "queries_per_sample"),
        "residual_truncated": get(residual, "truncated_count"),
        "two_level_mean": get(data, "two_level.mean"),
        "two_level_bias": get(data, "two_level.bias"),
        "two_level_mse": get(data, "two_level.mse_model_current_allocation"),
        "two_level_mse_time": get(data, "two_level.mse_time_current_allocation"),
        "speedup_score_vs_pure": get(data, "two_level.speedup_score_vs_pure"),
    }


def write_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows.\n"
    lines = []
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSON files or directories")
    parser.add_argument("--format", choices=["csv", "md"], default="csv")
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args(argv)

    rows = []
    for path in iter_json_files(args.paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("program") != "n2wos_eval_2lmc":
            continue
        rows.append(flatten(path, data))

    if args.format == "csv":
        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        else:
            import sys
            writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        text = write_md(rows)
        if args.output:
            pathlib.Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
