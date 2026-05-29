#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys


def run_one(root, exe, outdir, backend, mode, n_queries, n_samples, max_steps, repeats, extra):
    json_path = outdir / f"{mode}_{backend}.json"
    stdout_path = outdir / f"{mode}_{backend}.stdout.log"
    stderr_path = outdir / f"{mode}_{backend}.stderr.log"
    cmd = [
        str(exe),
        "--backend", backend,
        "--mode", mode,
        "--n-queries", str(n_queries),
        "--n-samples", str(n_samples),
        "--max-steps", str(max_steps),
        "--repeats", str(repeats),
        "--json", str(json_path),
    ] + extra
    print("+", " ".join(cmd), flush=True)
    with stdout_path.open("w") as so, stderr_path.open("w") as se:
        proc = subprocess.run(cmd, cwd=root, stdout=so, stderr=se)
    return {
        "backend": backend,
        "mode": mode,
        "returncode": proc.returncode,
        "json": str(json_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": cmd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-release-gpu")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--backends", nargs="+", default=["cpu", "vulkan", "cuda"])
    parser.add_argument("--mode", choices=["cpq", "wos", "both"], default="cpq")
    parser.add_argument("--n-queries", type=int, default=262144)
    parser.add_argument("--n-samples", type=int, default=65536)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--print-logs", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    exe = root / args.build_dir / "native" / "n2wos_probe_fcpw"
    if not exe.exists():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) if args.outdir else root / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    extra = ["--print-logs"] if args.print_logs else []
    manifest = []
    for backend in args.backends:
        manifest.append(run_one(root, exe, outdir, backend, args.mode, args.n_queries,
                                args.n_samples, args.max_steps, args.repeats, extra))

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0 if all(m["returncode"] == 0 for m in manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
