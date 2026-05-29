#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p external

FCPW_URL="${FCPW_URL:-https://github.com/rohan-sawhney/fcpw}"
FCPW_REF="${FCPW_REF:-}"

if [ -e external/fcpw/CMakeLists.txt ]; then
  echo "external/fcpw already exists"
else
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null | awk '{print $2}' | grep -qx 'external/fcpw'; then
      echo "FCPW submodule is already registered"
      git submodule update --init --recursive external/fcpw
    else
      echo "Adding FCPW as a git submodule"
      git submodule add "$FCPW_URL" external/fcpw
    fi
  else
    echo "Not inside a git repository; cloning FCPW directly"
    git clone "$FCPW_URL" external/fcpw
  fi
fi

if [ -n "$FCPW_REF" ]; then
  echo "Checking out FCPW ref: $FCPW_REF"
  git -C external/fcpw fetch --all --tags
  git -C external/fcpw checkout "$FCPW_REF"
fi

# Always initialize FCPW's nested dependencies, even if external/fcpw already
# existed before this script was run.  This is needed for deps/eigen and, in GPU
# builds, deps/slang-rhi.
echo "Updating FCPW nested submodules"
git -C external/fcpw submodule update --init --recursive

printf '\nFCPW dependency check:\n'
if [ -f external/fcpw/deps/eigen/Eigen/Core ]; then
  echo "  ok: deps/eigen/Eigen/Core"
else
  echo "  missing: deps/eigen/Eigen/Core" >&2
  exit 1
fi

if [ -f external/fcpw/deps/slang-rhi/CMakeLists.txt ]; then
  echo "  ok: deps/slang-rhi/CMakeLists.txt"
else
  echo "  missing: deps/slang-rhi/CMakeLists.txt"
  echo "  GPU builds will fail until this nested submodule is initialized."
fi

printf '\nFCPW status:\n'
git -C external/fcpw rev-parse --short HEAD
git -C external/fcpw status --short
