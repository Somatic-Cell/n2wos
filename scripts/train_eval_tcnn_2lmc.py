#!/usr/bin/env python3
"""Train a tiny-cuda-nn harmonic cache and evaluate it on dumped 2LMC paths.

This script is deliberately external to the native FCPW timing loop.  The native
program dumps the m-step cache points and full-WoS values.  This script trains a
HashGrid + MLP cache u_theta(x), evaluates C_m=u_theta(X_m) on those fixed paths,
and recomputes the two-level statistics.  This gives the research numbers for
cache bias and residual variance without changing the exact FCPW geometry target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

try:
    import tinycudann as tcnn
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import tinycudann. Install the PyTorch binding first, e.g.\n"
        "  cd external/tiny-cuda-nn/bindings/torch && python3 setup.py install\n"
        "or\n"
        "  pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch\n"
        f"Original error: {exc}"
    )


@dataclass
class DumpSamples:
    coarse_x: torch.Tensor
    coarse_is_boundary: torch.Tensor
    coarse_c_exact: torch.Tensor
    residual_x: torch.Tensor
    residual_is_boundary: torch.Tensor
    residual_c_exact: torch.Tensor
    residual_w: torch.Tensor


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    data = load_json(path)
    for key in ["encoding", "network"]:
        if key not in data:
            raise ValueError(f"{path} is missing key {key!r}")
    return data


def load_dump(path: pathlib.Path, device: torch.device) -> DumpSamples:
    coarse_x: List[List[float]] = []
    coarse_b: List[int] = []
    coarse_c: List[float] = []
    resid_x: List[List[float]] = []
    resid_b: List[int] = []
    resid_c: List[float] = []
    resid_w: List[float] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"kind", "x", "y", "z", "is_boundary", "c_exact", "w"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing CSV columns: {sorted(missing)}")
        for row in reader:
            x = [float(row["x"]), float(row["y"]), float(row["z"])]
            is_boundary = int(row["is_boundary"])
            c_exact = float(row["c_exact"])
            kind = row["kind"]
            if kind == "coarse":
                coarse_x.append(x)
                coarse_b.append(is_boundary)
                coarse_c.append(c_exact)
            elif kind == "residual":
                resid_x.append(x)
                resid_b.append(is_boundary)
                resid_c.append(c_exact)
                resid_w.append(float(row["w"]))
            else:
                raise ValueError(f"Unknown sample kind {kind!r} in {path}")

    def t2(a, dtype=torch.float32):
        return torch.tensor(a, dtype=dtype, device=device)

    return DumpSamples(
        coarse_x=t2(coarse_x),
        coarse_is_boundary=t2(coarse_b, torch.bool),
        coarse_c_exact=t2(coarse_c),
        residual_x=t2(resid_x),
        residual_is_boundary=t2(resid_b, torch.bool),
        residual_c_exact=t2(resid_c),
        residual_w=t2(resid_w),
    )


def normalize_x(x: torch.Tensor, domain_radius: float) -> torch.Tensor:
    y = x / (2.0 * domain_radius) + 0.5
    return torch.clamp(y, 0.0, 1.0)


def harmonic_target(x: torch.Tensor) -> torch.Tensor:
    return (x[:, 0:1] * x[:, 0:1]) - (x[:, 1:2] * x[:, 1:2])


def sample_training_points(batch_size: int, domain_radius: float, device: torch.device) -> torch.Tensor:
    # We train on the bounding cube.  The target is harmonic everywhere, and the
    # cache is only evaluated on X_m points dumped from FCPW paths.
    return (2.0 * torch.rand((batch_size, 3), device=device) - 1.0) * domain_radius


def make_model(config: Dict[str, Any]) -> torch.nn.Module:
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=1,
        encoding_config=config["encoding"],
        network_config=config["network"],
        seed=int(config.get("seed", 1337)),
    )
    try:
        model.jit_fusion = bool(config.get("jit_fusion", False)) and tcnn.supports_jit_fusion()
    except Exception:
        pass
    return model


def train_model(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    torch.manual_seed(args.seed)
    model = make_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.perf_counter()
    loss_ema = None
    for step in range(1, args.steps + 1):
        x = sample_training_points(args.batch_size, args.domain_radius, device)
        y = harmonic_target(x)
        pred = model(normalize_x(x, args.domain_radius)).float()
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_val
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(f"step={step} loss={loss_val:.6e} ema={loss_ema:.6e}", flush=True)
    torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0

    with torch.no_grad():
        val_x = sample_training_points(args.val_size, args.domain_radius, device)
        val_y = harmonic_target(val_x)
        val_pred = predict(model, val_x, args.domain_radius, args.eval_batch_size)
        val_err = (val_pred - val_y[:, 0]).float()
        val_mse = float(torch.mean(val_err * val_err).cpu())
        val_bias = float(torch.mean(val_err).cpu())
        val_var = float(torch.var(val_err, unbiased=True).cpu()) if val_err.numel() > 1 else 0.0

    stats = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_sec": train_sec,
        "final_loss": loss_val,
        "loss_ema": loss_ema,
        "val_mse": val_mse,
        "val_bias": val_bias,
        "val_error_var": val_var,
        "domain_radius": args.domain_radius,
    }
    return model, stats


def predict(model: torch.nn.Module, x: torch.Tensor, domain_radius: float, batch_size: int) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            outs.append(model(normalize_x(xb, domain_radius)).float().reshape(-1))
    return torch.cat(outs, dim=0)


def timed_predict(model: torch.nn.Module, x: torch.Tensor, domain_radius: float, batch_size: int, repeats: int) -> Tuple[torch.Tensor, float]:
    # First call warms up PyTorch/tiny-cuda-nn dispatch and any JIT path.
    y = predict(model, x, domain_radius, batch_size)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        y = predict(model, x, domain_radius, batch_size)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return y, best * 1.0e6


def sample_stats(v: torch.Tensor, exact: float) -> Dict[str, float]:
    v = v.float().reshape(-1)
    n = int(v.numel())
    mean = float(torch.mean(v).cpu()) if n else 0.0
    var = float(torch.var(v, unbiased=True).cpu()) if n > 1 else 0.0
    bias = mean - exact
    return {
        "n": n,
        "mean": mean,
        "variance": var,
        "std_error": math.sqrt(var / n) if n else 0.0,
        "bias": bias,
        "mse_model": bias * bias + (var / n if n else 0.0),
    }


def estimator_field(data: Dict[str, Any], name: str, field: str) -> float:
    return float(data["estimators"][name][field])


def estimator_var(data: Dict[str, Any], name: str) -> float:
    return float(data["estimators"][name]["stats"]["variance"])


def evaluate_on_dump(args: argparse.Namespace, base: Dict[str, Any], samples: DumpSamples, model: torch.nn.Module) -> Dict[str, Any]:
    exact = float(base["target"]["exact_value_at_x0"])

    coarse_pred, coarse_infer_usec = timed_predict(model, samples.coarse_x, args.domain_radius, args.eval_batch_size, args.timing_repeats)
    resid_pred, resid_infer_usec = timed_predict(model, samples.residual_x, args.domain_radius, args.eval_batch_size, args.timing_repeats)

    coarse_c = torch.where(samples.coarse_is_boundary, samples.coarse_c_exact, coarse_pred)
    resid_c = torch.where(samples.residual_is_boundary, samples.residual_c_exact, resid_pred)
    residual_r = samples.residual_w - resid_c

    coarse_stats = sample_stats(coarse_c, exact)
    residual_stats = sample_stats(residual_r, 0.0)
    residual_c_stats = sample_stats(resid_c, exact)
    residual_w_stats = sample_stats(samples.residual_w, exact)

    pure_var = estimator_var(base, "pure_wos")
    pure_usec = estimator_field(base, "pure_wos", "usec_per_sample")
    coarse_geom_usec = estimator_field(base, "coarse", "usec_per_sample")
    residual_geom_usec = estimator_field(base, "coupled_residual", "usec_per_sample")

    coarse_infer_per_sample = coarse_infer_usec / max(1, coarse_stats["n"])
    resid_infer_per_sample = resid_infer_usec / max(1, residual_stats["n"])
    coarse_total_usec = coarse_geom_usec + coarse_infer_per_sample
    residual_total_usec = residual_geom_usec + resid_infer_per_sample

    tl_mean = coarse_stats["mean"] + residual_stats["mean"]
    tl_bias = tl_mean - exact
    tl_var = coarse_stats["variance"] / coarse_stats["n"] + residual_stats["variance"] / residual_stats["n"]
    tl_mse = tl_bias * tl_bias + tl_var

    pure_score = pure_var * pure_usec
    coarse_score = coarse_stats["variance"] * coarse_total_usec
    residual_score = residual_stats["variance"] * residual_total_usec
    opt_score = (math.sqrt(max(0.0, coarse_score)) + math.sqrt(max(0.0, residual_score))) ** 2

    return {
        "program": "n2wos_tcnn_2lmc_postprocess",
        "base_json": str(args.base_json),
        "samples_csv": str(args.samples),
        "cache": "tcnn_python",
        "m": int(base["args"]["m"]),
        "exact": exact,
        "training": {},
        "timing": {
            "eval_batch_size": args.eval_batch_size,
            "timing_repeats": args.timing_repeats,
            "coarse_inference_usec": coarse_infer_usec,
            "residual_inference_usec": resid_infer_usec,
            "coarse_inference_usec_per_sample": coarse_infer_per_sample,
            "residual_inference_usec_per_sample": resid_infer_per_sample,
            "coarse_geometry_usec_per_sample": coarse_geom_usec,
            "residual_geometry_usec_per_sample": residual_geom_usec,
            "coarse_total_usec_per_sample": coarse_total_usec,
            "residual_total_usec_per_sample": residual_total_usec,
            "pure_usec_per_sample": pure_usec,
        },
        "estimators": {
            "pure_wos": base["estimators"]["pure_wos"],
            "coarse": {"stats": coarse_stats},
            "coupled_residual": {
                "w_stats": residual_w_stats,
                "c_stats": residual_c_stats,
                "residual_stats": residual_stats,
            },
        },
        "two_level": {
            "mean": tl_mean,
            "bias": tl_bias,
            "variance_model_current_allocation": tl_var,
            "mse_model_current_allocation": tl_mse,
            "pure_var_cost_score": pure_score,
            "coarse_var_cost_score": coarse_score,
            "residual_var_cost_score": residual_score,
            "two_level_optimal_var_cost_score": opt_score,
            "speedup_score_vs_pure": pure_score / opt_score if opt_score > 0 else 0.0,
            "residual_var_ratio_vs_pure": residual_stats["variance"] / pure_var if pure_var > 0 else 0.0,
        },
    }


def save_checkpoint(args: argparse.Namespace, config: Dict[str, Any], model: torch.nn.Module, train_stats: Dict[str, Any]) -> None:
    if not args.checkpoint:
        return
    ckpt = pathlib.Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "n2wos_tcnn_harmonic_checkpoint_v1",
        "config": config,
        "state_dict": model.state_dict(),
        "training": train_stats,
        "domain_radius": args.domain_radius,
    }, ckpt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/cache_hashgrid_small.json"))
    parser.add_argument("--base-json", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--domain-radius", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--val-size", type=int, default=262144)
    parser.add_argument("--eval-batch-size", type=int, default=262144)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required for tiny-cuda-nn training")
    device = torch.device("cuda")

    config = load_config(args.config)
    base = load_json(args.base_json)
    samples = load_dump(args.samples, device)
    model, train_stats = train_model(args, config, device)
    save_checkpoint(args, config, model, train_stats)
    result = evaluate_on_dump(args, base, samples, model)
    result["training"] = train_stats
    result["config"] = config

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "val_mse": train_stats["val_mse"],
        "two_level_speedup_score_vs_pure": result["two_level"]["speedup_score_vs_pure"],
        "residual_var_ratio_vs_pure": result["two_level"]["residual_var_ratio_vs_pure"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
