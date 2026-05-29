#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p external

TCNN_URL="${TCNN_URL:-https://github.com/NVlabs/tiny-cuda-nn}"
TCNN_REF="${TCNN_REF:-}"

if [ -e external/tiny-cuda-nn/CMakeLists.txt ]; then
  echo "external/tiny-cuda-nn already exists"
else
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null | awk '{print $2}' | grep -qx 'external/tiny-cuda-nn'; then
      echo "tiny-cuda-nn submodule is already registered"
      git submodule update --init --recursive external/tiny-cuda-nn
    else
      echo "Adding tiny-cuda-nn as a git submodule"
      git submodule add "$TCNN_URL" external/tiny-cuda-nn
    fi
  else
    echo "Not inside a git repository; cloning tiny-cuda-nn directly"
    git clone "$TCNN_URL" external/tiny-cuda-nn
  fi
fi

if [ -n "$TCNN_REF" ]; then
  echo "Checking out tiny-cuda-nn ref: $TCNN_REF"
  git -C external/tiny-cuda-nn fetch --all --tags
  git -C external/tiny-cuda-nn checkout "$TCNN_REF"
fi

echo "Updating tiny-cuda-nn nested submodules"
git -C external/tiny-cuda-nn submodule update --init --recursive

printf '\ntiny-cuda-nn dependency check:\n'
if [ -f external/tiny-cuda-nn/dependencies/cutlass/CMakeLists.txt ]; then
  echo "  ok: dependencies/cutlass/CMakeLists.txt"
else
  echo "  missing: dependencies/cutlass/CMakeLists.txt" >&2
  exit 1
fi

printf '\ntiny-cuda-nn status:\n'
git -C external/tiny-cuda-nn rev-parse --short HEAD
git -C external/tiny-cuda-nn status --short
