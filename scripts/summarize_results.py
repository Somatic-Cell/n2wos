#!/usr/bin/env python3
"""Summarize n2wos JSON benchmark outputs.

The probe executable writes a top-level object with a ``results`` array.
This script recursively finds such files, flattens the records, and emits
Markdown, CSV, or JSON. It intentionally depends only on the Python
standard library so that it works in a fresh WSL environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

Row = Dict[str, Any]

DEFAULT_COLUMNS = [
    "file",
    "backend",
    "mode",
    "n_queries",
    "n_samples",
    "max_steps_limit",
    "repeats",
    "mean_usec_per_query",
    "median_usec_per_query",
    "usec_per_query",
    "usec_per_sample",
    "usec_per_wos_query",
    "queries_per_sample",
    "avg_steps",
    "p95_steps",
    "p99_steps",
    "max_steps",
    "active_remaining",
    "query_count",
]

AGGREGATE_COLUMNS = [
    "backend",
    "mode",
    "n_queries",
    "n_samples",
    "max_steps_limit",
    "count",
    "mean_usec_per_query_mean",
    "median_usec_per_query_mean",
    "usec_per_query_mean",
    "usec_per_query_min",
    "usec_per_query_max",
    "usec_per_sample_mean",
    "usec_per_wos_query_mean",
    "queries_per_sample_mean",
    "avg_steps_mean",
    "p95_steps_mean",
    "p99_steps_mean",
    "max_steps_mean",
    "active_remaining_mean",
    "query_count_mean",
]

NUMERIC_AGG_FIELDS = [
    "mean_usec_per_query",
    "median_usec_per_query",
    "usec_per_query",
    "usec_per_sample",
    "usec_per_wos_query",
    "queries_per_sample",
    "avg_steps",
    "p95_steps",
    "p99_steps",
    "max_steps",
    "active_remaining",
    "query_count",
    "total_usec",
]


def iter_json_files(inputs: Sequence[pathlib.Path], pattern: str) -> Iterable[pathlib.Path]:
    for path in inputs:
        if path.is_dir():
            yield from sorted(path.rglob(pattern))
        elif path.is_file():
            yield path
        else:
            print(f"warning: path does not exist: {path}", file=sys.stderr)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def as_int_if_integral(value: Any) -> Any:
    x = as_float(value)
    if x is None:
        return value
    if abs(x - round(x)) < 1.0e-9:
        return int(round(x))
    return value


def load_rows_from_file(path: pathlib.Path, root: pathlib.Path | None) -> List[Row]:
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as exc:  # noqa: BLE001 - report and skip malformed files
        print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
        return []

    rel = str(path)
    if root is not None:
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            rel = str(path)

    rows: List[Row] = []
    mesh = doc.get("mesh") if isinstance(doc.get("mesh"), dict) else {}
    compiled = doc.get("compiled") if isinstance(doc.get("compiled"), dict) else {}
    runtime = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}

    for i, result in enumerate(doc["results"]):
        if not isinstance(result, dict):
            continue
        row: Row = {
            "file": rel,
            "result_index": i,
            "backend": result.get("backend", doc.get("backend", "")),
            "mode": result.get("mode", doc.get("requested_mode", "")),
            "requested_mode": doc.get("requested_mode", ""),
            "mesh_vertices": mesh.get("vertices", ""),
            "mesh_triangles": mesh.get("triangles", ""),
            "lat_segments": mesh.get("lat_segments", ""),
            "lon_segments": mesh.get("lon_segments", ""),
            "build_type": compiled.get("N2WOS_CMAKE_BUILD_TYPE", ""),
            "fcpw_dir": runtime.get("fcpw_dir", ""),
        }
        row.update(result)

        if "usec_per_query" not in row and "usec/query" in row:
            row["usec_per_query"] = row["usec/query"]
        if "usec_per_sample" not in row and "usec/sample" in row:
            row["usec_per_sample"] = row["usec/sample"]

        q = as_float(row.get("query_count"))
        n = as_float(row.get("n_samples"))
        row["queries_per_sample"] = (q / n) if q is not None and n and n > 0.0 else ""
        rows.append(row)
    return rows


def load_rows(inputs: Sequence[pathlib.Path], pattern: str, root: pathlib.Path | None) -> List[Row]:
    rows: List[Row] = []
    for path in iter_json_files(inputs, pattern):
        rows.extend(load_rows_from_file(path, root))
    rows.sort(
        key=lambda r: (
            str(r.get("mode", "")),
            str(r.get("backend", "")),
            as_float(r.get("n_queries")) or -1,
            as_float(r.get("max_steps_limit")) or -1,
            str(r.get("file", "")),
        )
    )
    return rows


def aggregate_rows(rows: Sequence[Row]) -> List[Row]:
    groups: Dict[Tuple[Any, ...], List[Row]] = {}
    key_fields = ["backend", "mode", "n_queries", "n_samples", "max_steps_limit"]
    for row in rows:
        key = tuple(as_int_if_integral(row.get(k, "")) for k in key_fields)
        groups.setdefault(key, []).append(row)

    out: List[Row] = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        agg: Row = {k: v for k, v in zip(key_fields, key)}
        agg["count"] = len(members)
        agg["files"] = [str(m.get("file", "")) for m in members]
        for field in NUMERIC_AGG_FIELDS:
            values = [as_float(m.get(field)) for m in members]
            values = [v for v in values if v is not None]
            if not values:
                continue
            agg[f"{field}_mean"] = statistics.fmean(values)
            agg[f"{field}_min"] = min(values)
            agg[f"{field}_max"] = max(values)
            agg[f"{field}_median"] = statistics.median(values)
        out.append(agg)
    return out


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == 0.0:
            return "0"
        if abs(value) >= 10000 or abs(value) < 0.001:
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return str(value)


def write_markdown(rows: Sequence[Row], columns: Sequence[str], out) -> None:
    if not rows:
        print("No n2wos result rows found.", file=out)
        return
    print("| " + " | ".join(columns) + " |", file=out)
    print("| " + " | ".join(["---"] * len(columns)) + " |", file=out)
    for row in rows:
        print("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |", file=out)


def write_csv(rows: Sequence[Row], columns: Sequence[str], out) -> None:
    writer = csv.DictWriter(out, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})


def choose_columns(rows: Sequence[Row], aggregate: bool, wide: bool) -> List[str]:
    if wide:
        preferred = AGGREGATE_COLUMNS if aggregate else DEFAULT_COLUMNS
        extras = sorted({k for row in rows for k in row.keys()} - set(preferred))
        return preferred + extras
    return AGGREGATE_COLUMNS if aggregate else DEFAULT_COLUMNS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=pathlib.Path, default=[pathlib.Path("results")])
    parser.add_argument("--pattern", default="*.json", help="recursive glob pattern for input directories")
    parser.add_argument("--format", choices=["md", "csv", "json"], default="md")
    parser.add_argument("--output", "-o", type=pathlib.Path, default=None)
    parser.add_argument("--aggregate", action="store_true", help="group repeated runs by backend/mode/size/depth")
    parser.add_argument("--wide", action="store_true", help="include every observed JSON field")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args(argv)

    rows = load_rows(args.inputs, args.pattern, args.root)
    if args.aggregate:
        rows = aggregate_rows(rows)
    columns = choose_columns(rows, args.aggregate, args.wide)

    if args.output is None:
        out = sys.stdout
        close = False
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = args.output.open("w", encoding="utf-8", newline="")
        close = True
    try:
        if args.format == "md":
            write_markdown(rows, columns, out)
        elif args.format == "csv":
            write_csv(rows, columns, out)
        else:
            json.dump(rows, out, indent=2)
            print(file=out)
    finally:
        if close:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
