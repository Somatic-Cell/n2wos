#!/usr/bin/env python3
"""Run fixed-depth WoS cost sweeps for n2wos_probe_fcpw.

This uses the existing probe executable with ``--mode wos`` and varies
``--max-steps``. For small depths this approximates the geometry part of a
coarse estimator C_m. The m=0 case is useful as a baseline for executable
bookkeeping because the timed WoS loop performs no FCPW queries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import List, Sequence

DEFAULT_DEPTHS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in text.replace(",", " ").split():
        value = int(part)
        if value < 0:
            raise argparse.ArgumentTypeError("depths must be non-negative")
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("empty depth list")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-release-gpu")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--backends", nargs="+", default=["cuda"])
    parser.add_argument("--depths", type=parse_int_list, default=DEFAULT_DEPTHS)
    parser.add_argument("--runs", type=int, default=3, help="independent repeats per depth")
    parser.add_argument("--n-samples", type=int, default=65536)
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--safety", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lat-segments", type=int, default=128)
    parser.add_argument("--lon-segments", type=int, default=256)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--bump-amplitude", type=float, default=0.08)
    parser.add_argument("--bump-frequency", type=float, default=7.0)
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")

    root = pathlib.Path(__file__).resolve().parents[1]
    exe = root / args.build_dir / "native" / "n2wos_probe_fcpw"
    if not exe.exists():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) if args.outdir else root / "results" / f"depth_sweep_{stamp}"
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    failures = 0
    for backend in args.backends:
        for depth in args.depths:
            for run in range(args.runs):
                seed = args.seed + 1000003 * run + 9176 * depth
                stem = f"wos_{backend}_m{depth}_r{run}"
                json_path = outdir / f"{stem}.json"
                stdout_path = outdir / f"{stem}.stdout.log"
                stderr_path = outdir / f"{stem}.stderr.log"
                cmd = [
                    str(exe),
                    "--backend",
                    backend,
                    "--mode",
                    "wos",
                    "--n-samples",
                    str(args.n_samples),
                    "--max-steps",
                    str(depth),
                    "--eps",
                    str(args.eps),
                    "--safety",
                    str(args.safety),
                    "--seed",
                    str(seed),
                    "--lat-segments",
                    str(args.lat_segments),
                    "--lon-segments",
                    str(args.lon_segments),
                    "--radius",
                    str(args.radius),
                    "--bump-amplitude",
                    str(args.bump_amplitude),
                    "--bump-frequency",
                    str(args.bump_frequency),
                    "--json",
                    str(json_path),
                ]
                if args.print_logs:
                    cmd.append("--print-logs")

                print("+", " ".join(cmd), flush=True)
                with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
                    proc = subprocess.run(cmd, cwd=root, stdout=so, stderr=se)
                record = {
                    "backend": backend,
                    "mode": "wos",
                    "max_steps": depth,
                    "run": run,
                    "n_samples": args.n_samples,
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
        if failures and args.stop_on_failure:
            break

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
