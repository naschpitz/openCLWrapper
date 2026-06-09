#!/usr/bin/env bash
#
# Build OpenCLWrapper using CMake presets.
#
# Requires a CMakeUserPresets.json with your Qt6 kit paths (see README.md).
#
# Usage:
#   ./build.sh [preset]      preset = static (default) | shared

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:-static}"
CMAKE_USER_PRESETS="$SCRIPT_DIR/CMakeUserPresets.json"

if [[ ! -f "$CMAKE_USER_PRESETS" ]]; then
  echo "error: $CMAKE_USER_PRESETS not found." >&2
  echo "       Create it (git-ignored) with your Qt6 kit paths — see README.md." >&2
  exit 1
fi

jobs="$(nproc 2>/dev/null || echo 4)"

if ( cd "$SCRIPT_DIR" && cmake --preset "$PRESET" && cmake --build build -j"$jobs" ); then
  echo "==> OpenCLWrapper: OK"
else
  echo "==> OpenCLWrapper: FAILED" >&2
  exit 1
fi
