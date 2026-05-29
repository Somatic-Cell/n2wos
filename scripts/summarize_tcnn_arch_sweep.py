#!/usr/bin/env python3
"""Summarize TCNN architecture/HashGrid sweeps for NC-only and NC+2LMC."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: pathlib.Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("program") != "n2wos_tcnn_2lmc_postprocess":
        return None
    return data


def get(d: Dict[str, Any], keys: Iterable[str], default: Any = math.nan) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_seed(path: pathlib.Path) -> int | None:
    for part in path.parts:
        m = re.fullmatch(r"seed_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def find_arch_from_path(path: pathlib.Path) -> str | None:
    for part in path.parts:
        m = re.fullmatch(r"arch_(.+)", part)
        if m:
            return m.group(1)
    return None


def finite_float(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return math.nan
    return y if math.isfinite(y) else math.nan


def estimate_model_size(config: Dict[str, Any]) -> Dict[str, Any]:
    enc = config.get("encoding", {}) if isinstance(config.get("encoding", {}), dict) else {}
    net = config.get("network", {}) if isinstance(config.get("network", {}), dict) else {}
    n_levels = int(enc.get("n_levels", 0) or 0)
    fpl = int(enc.get("n_features_per_level", 0) or 0)
    log2_hash = int(enc.get("log2_hashmap_size", 0) or 0)
    enc_dims = n_levels * fpl
    hash_params_upper = n_levels * fpl * (1 << log2_hash) if log2_hash >= 0 else 0
    hash_bytes_fp16_upper = 2 * hash_params_upper

    width = int(net.get("n_neurons", 0) or 0)
    hidden_layers = int(net.get("n_hidden_layers", 0) or 0)
    out_dim = int(config.get("n_output_dims", 1) or 1)
    if width <= 0:
        network_params_rough = 0
    elif hidden_layers <= 0:
        network_params_rough = (enc_dims + 1) * out_dim
    else:
        network_params_rough = (enc_dims + 1) * width
        network_params_rough += max(0, hidden_layers - 1) * (width + 1) * width
        network_params_rough += (width + 1) * out_dim

    return {
        "n_levels": n_levels,
        "n_features_per_level": fpl,
        "log2_hashmap_size": log2_hash,
        "encoding_output_dims": enc_dims,
        "n_neurons": width,
        "n_hidden_layers": hidden_layers,
        "hash_params_upper": hash_params_upper,
        "network_params_rough": network_params_rough,
        "total_params_rough_upper": hash_params_upper + network_params_rough,
        "hash_mebibytes_fp16_upper": hash_bytes_fp16_upper / (1024.0 * 1024.0),
    }


def flatten(path: pathlib.Path, data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = data.get("config", {}) if isinstance(data.get("config", {}), dict) else {}
    arch = cfg.get("architecture_name") or find_arch_from_path(path) or "unknown"
    size = estimate_model_size(cfg)

    pure_score = finite_float(get(data, ["two_level", "pure_var_cost_score"]))
    two_score = finite_float(get(data, ["two_level", "two_level_optimal_var_cost_score"]))
    train_sec = finite_float(get(data, ["training", "train_sec"], 0.0))
    train_usec = train_sec * 1.0e6
    break_even_var = max(0.0, (pure_score - two_score) / train_usec) if train_usec > 0 and math.isfinite(pure_score) and math.isfinite(two_score) else math.nan
    break_even_rmse = math.sqrt(break_even_var) if math.isfinite(break_even_var) and break_even_var >= 0 else math.nan

    coarse_bias = finite_float(get(data, ["estimators", "coarse", "stats", "bias"]))
    coarse_var = finite_float(get(data, ["estimators", "coarse", "stats", "variance"]))
    coarse_mse = finite_float(get(data, ["estimators", "coarse", "stats", "mse_model"]))
    two_bias = finite_float(get(data, ["two_level", "bias"]))

    row: Dict[str, Any] = {
        "path": str(path),
        "seed": find_seed(path),
        "architecture": arch,
        "m": data.get("m"),
        "steps": get(data, ["training", "steps"]),
        **size,
        "train_sec": train_sec,
        "val_mse": get(data, ["training", "val_mse"]),
        "val_bias": get(data, ["training", "val_bias"]),
        "final_loss": get(data, ["training", "final_loss"]),
        "pure_var": get(data, ["estimators", "pure_wos", "stats", "variance"]),
        "pure_usec_per_sample": get(data, ["timing", "pure_usec_per_sample"]),
        "nc_only_bias": coarse_bias,
        "nc_only_abs_bias": abs(coarse_bias) if math.isfinite(coarse_bias) else math.nan,
        "nc_only_bias2": coarse_bias * coarse_bias if math.isfinite(coarse_bias) else math.nan,
        "nc_only_var": coarse_var,
        "nc_only_mse_model_current_n": coarse_mse,
        "nc_only_usec_per_sample": get(data, ["timing", "coarse_total_usec_per_sample"]),
        "residual_bias": get(data, ["estimators", "coupled_residual", "residual_stats", "bias"]),
        "residual_var": get(data, ["estimators", "coupled_residual", "residual_stats", "variance"]),
        "two_level_bias": two_bias,
        "two_level_abs_bias": abs(two_bias) if math.isfinite(two_bias) else math.nan,
        "two_level_mse_model_current_n": get(data, ["two_level", "mse_model_current_allocation"]),
        "residual_var_ratio_vs_pure": get(data, ["two_level", "residual_var_ratio_vs_pure"]),
        "speedup_score_vs_pure_excluding_training": get(data, ["two_level", "speedup_score_vs_pure"]),
        "pure_var_cost_score": pure_score,
        "two_level_optimal_var_cost_score": two_score,
        "training_break_even_variance": break_even_var,
        "training_break_even_rmse": break_even_rmse,
        "coarse_inference_usec_per_sample": get(data, ["timing", "coarse_inference_usec_per_sample"]),
        "residual_inference_usec_per_sample": get(data, ["timing", "residual_inference_usec_per_sample"]),
        "coarse_geometry_usec_per_sample": get(data, ["timing", "coarse_geometry_usec_per_sample"]),
        "residual_geometry_usec_per_sample": get(data, ["timing", "residual_geometry_usec_per_sample"]),
        "coarse_total_usec_per_sample": get(data, ["timing", "coarse_total_usec_per_sample"]),
        "residual_total_usec_per_sample": get(data, ["timing", "residual_total_usec_per_sample"]),
    }
    return row


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.8g}"
    return str(x)


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(vals: List[Any]) -> Tuple[float, float]:
    xs = [finite_float(v) for v in vals]
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return math.nan, math.nan
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        try:
            key = (str(r["architecture"]), int(r["steps"]))
        except Exception:
            continue
        groups.setdefault(key, []).append(r)

    scalar_first = [
        "n_levels", "n_features_per_level", "log2_hashmap_size", "encoding_output_dims",
        "n_neurons", "n_hidden_layers", "hash_params_upper", "network_params_rough",
        "total_params_rough_upper", "hash_mebibytes_fp16_upper",
    ]
    stats_keys = [
        "train_sec", "val_mse", "val_bias", "nc_only_abs_bias", "nc_only_bias2",
        "nc_only_mse_model_current_n", "residual_var_ratio_vs_pure", "two_level_abs_bias",
        "speedup_score_vs_pure_excluding_training", "training_break_even_rmse",
        "coarse_inference_usec_per_sample", "residual_inference_usec_per_sample",
        "coarse_total_usec_per_sample", "residual_total_usec_per_sample",
    ]

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        arch, steps = key
        rs = groups[key]
        row: Dict[str, Any] = {"architecture": arch, "steps": steps, "n_runs": len(rs)}
        for k in scalar_first:
            row[k] = rs[0].get(k, math.nan)
        for k in stats_keys:
            mu, sd = mean_std([r.get(k) for r in rs])
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
        out.append(row)
    return out


def pareto(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a simple Pareto frontier over mean speedup, NC bias, and train time.

    A row is dominated if another row has at least as high speedup and no larger
    NC-only bias/train time/model size, with one strict improvement.
    """
    candidates = [r for r in rows if math.isfinite(finite_float(r.get("speedup_score_vs_pure_excluding_training_mean")))]
    front: List[Dict[str, Any]] = []
    for r in candidates:
        dominated = False
        for q in candidates:
            if q is r:
                continue
            checks = [
                finite_float(q.get("speedup_score_vs_pure_excluding_training_mean")) >= finite_float(r.get("speedup_score_vs_pure_excluding_training_mean")),
                finite_float(q.get("nc_only_abs_bias_mean")) <= finite_float(r.get("nc_only_abs_bias_mean")),
                finite_float(q.get("train_sec_mean")) <= finite_float(r.get("train_sec_mean")),
                finite_float(q.get("total_params_rough_upper")) <= finite_float(r.get("total_params_rough_upper")),
            ]
            strict = [
                finite_float(q.get("speedup_score_vs_pure_excluding_training_mean")) > finite_float(r.get("speedup_score_vs_pure_excluding_training_mean")),
                finite_float(q.get("nc_only_abs_bias_mean")) < finite_float(r.get("nc_only_abs_bias_mean")),
                finite_float(q.get("train_sec_mean")) < finite_float(r.get("train_sec_mean")),
                finite_float(q.get("total_params_rough_upper")) < finite_float(r.get("total_params_rough_upper")),
            ]
            if all(checks) and any(strict):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def write_md(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/final_tcnn_arch_runs.csv"))
    ap.add_argument("--aggregate-out", type=pathlib.Path, default=pathlib.Path("results/final_tcnn_arch_aggregate.csv"))
    ap.add_argument("--aggregate-md", type=pathlib.Path, default=None)
    ap.add_argument("--pareto-out", type=pathlib.Path, default=pathlib.Path("results/final_tcnn_arch_pareto.csv"))
    ap.add_argument("--pareto-md", type=pathlib.Path, default=None)
    args = ap.parse_args()

    files: List[pathlib.Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        else:
            files.append(p)

    rows: List[Dict[str, Any]] = []
    for p in files:
        data = load_json(p)
        if data is not None:
            rows.append(flatten(p, data))
    rows.sort(key=lambda r: (str(r.get("architecture")), int(r.get("steps") or -1), int(r.get("seed") or -1), r["path"]))
    write_csv(args.out, rows)

    agg = aggregate(rows)
    write_csv(args.aggregate_out, agg)
    if args.aggregate_md:
        write_md(args.aggregate_md, agg)

    front = pareto(agg)
    write_csv(args.pareto_out, front)
    if args.pareto_md:
        write_md(args.pareto_md, front)

    print(f"wrote {args.out} ({len(rows)} rows)")
    print(f"wrote {args.aggregate_out} ({len(agg)} rows)")
    print(f"wrote {args.pareto_out} ({len(front)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
