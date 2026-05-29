#!/usr/bin/env python3
"""Run a minimal 2LMC matrix for the harmonic mesh problem."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Sequence


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-release-gpu")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--outdir", default="results/2lmc_harmonic")
    parser.add_argument("--depths", type=parse_int_list, default=[8, 16])
    parser.add_argument("--caches", nargs="+", default=["exact", "scaled_exact"])
    parser.add_argument("--cache-scale", type=float, default=0.9)
    parser.add_argument("--cache-bias", type=float, default=0.0)
    parser.add_argument("--n-pure", type=int, default=65536)
    parser.add_argument("--n-coarse", type=int, default=65536)
    parser.add_argument("--n-residual", type=int, default=65536)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--warmup-queries", type=int, default=65536)
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--safety", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--x0", type=float, default=0.10)
    parser.add_argument("--y0", type=float, default=0.05)
    parser.add_argument("--z0", type=float, default=0.00)
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[1]
    exe = root / args.build_dir / "native" / "n2wos_eval_2lmc"
    if not exe.exists():
        parser.error(f"executable not found: {exe}")

    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    failures = 0

    for cache in args.caches:
        for depth in args.depths:
            stem = f"2lmc_{args.backend}_{cache}_m{depth}"
            json_path = outdir / f"{stem}.json"
            stdout_path = outdir / f"{stem}.stdout.log"
            stderr_path = outdir / f"{stem}.stderr.log"
            cmd = [
                str(exe),
                "--backend", args.backend,
                "--cache", cache,
                "--m", str(depth),
                "--max-steps", str(args.max_steps),
                "--n-pure", str(args.n_pure),
                "--n-coarse", str(args.n_coarse),
                "--n-residual", str(args.n_residual),
                "--warmup-queries", str(args.warmup_queries),
                "--eps", str(args.eps),
                "--safety", str(args.safety),
                "--seed", str(args.seed + 1009 * depth),
                "--x0", str(args.x0),
                "--y0", str(args.y0),
                "--z0", str(args.z0),
                "--json", str(json_path),
            ]
            if cache == "scaled_exact":
                cmd += ["--cache-scale", str(args.cache_scale), "--cache-bias", str(args.cache_bias)]
            if args.print_logs:
                cmd.append("--print-logs")

            print("+", " ".join(cmd), flush=True)
            with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
                proc = subprocess.run(cmd, cwd=root, stdout=so, stderr=se)
            record = {
                "backend": args.backend,
                "cache": cache,
                "m": depth,
                "returncode": proc.returncode,
                "json": str(json_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "command": cmd,
            }
            manifest.append(record)
            if proc.returncode != 0:
                failures += 1
                if args.stop_on_failure:
                    break
        if failures and args.stop_on_failure:
            break

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
