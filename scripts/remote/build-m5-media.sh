#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="$repo_root/build/M5/ltx-media"
cmake -S "$repo_root/native/media" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir"
test -f "$build_dir/libltx_oiio.so"
echo "$build_dir/libltx_oiio.so"
