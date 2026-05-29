#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import sys


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-release-gpu")
    parser.add_argument("--type", default="Release")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enoki", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tcnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jobs", "-j", type=int, default=0)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    build_dir = root / args.build_dir

    run([
        "cmake", "-S", root, "-B", build_dir,
        f"-DCMAKE_BUILD_TYPE={args.type}",
        f"-DN2WOS_ENABLE_FCPW_GPU={'ON' if args.gpu else 'OFF'}",
        f"-DN2WOS_USE_ENOKI={'ON' if args.enoki else 'OFF'}",
        f"-DN2WOS_ENABLE_TCNN={'ON' if args.tcnn else 'OFF'}",
    ])

    cmd = ["cmake", "--build", build_dir]
    if args.jobs > 0:
        cmd += ["-j", str(args.jobs)]
    else:
        cmd += ["-j"]
    run(cmd)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
