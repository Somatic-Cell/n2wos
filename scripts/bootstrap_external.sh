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
      git -C external/fcpw submodule update --init --recursive
    fi
  else
    echo "Not inside a git repository; cloning FCPW directly"
    git clone --recursive "$FCPW_URL" external/fcpw
  fi
fi

if [ -n "$FCPW_REF" ]; then
  echo "Checking out FCPW ref: $FCPW_REF"
  git -C external/fcpw fetch --all --tags
  git -C external/fcpw checkout "$FCPW_REF"
  git -C external/fcpw submodule update --init --recursive
fi

printf '\nFCPW status:\n'
git -C external/fcpw rev-parse --short HEAD
git -C external/fcpw status --short
