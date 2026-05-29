#!/usr/bin/env python3
"""Run closest-point query batch-size sweeps for n2wos_probe_fcpw."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import List, Sequence

DEFAULT_QUERY_SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in text.replace(",", " ").split():
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("all sizes must be positive")
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("empty size list")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-release-gpu")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--backends", nargs="+", default=["cuda"])
    parser.add_argument("--query-sizes", type=parse_int_list, default=DEFAULT_QUERY_SIZES)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lat-segments", type=int, default=128)
    parser.add_argument("--lon-segments", type=int, default=256)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--bump-amplitude", type=float, default=0.08)
    parser.add_argument("--bump-frequency", type=float, default=7.0)
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)

    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    root = pathlib.Path(__file__).resolve().parents[1]
    exe = root / args.build_dir / "native" / "n2wos_probe_fcpw"
    if not exe.exists():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) if args.outdir else root / "results" / f"batch_sweep_{stamp}"
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    failures = 0
    for backend in args.backends:
        for n_queries in args.query_sizes:
            stem = f"cpq_{backend}_n{n_queries}"
            json_path = outdir / f"{stem}.json"
            stdout_path = outdir / f"{stem}.stdout.log"
            stderr_path = outdir / f"{stem}.stderr.log"
            cmd = [
                str(exe),
                "--backend",
                backend,
                "--mode",
                "cpq",
                "--n-queries",
                str(n_queries),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
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
                "mode": "cpq",
                "n_queries": n_queries,
                "repeats": args.repeats,
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
