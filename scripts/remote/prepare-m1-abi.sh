#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

python_path="$repo_root/.venv/bin/python"
torch_url='https://download-r2.pytorch.org/whl/cu132/torch-2.13.0%2Bcu132-cp312-cp312-manylinux_2_28_x86_64.whl'
torchsharp_revision='8f4def03b641b6753f18076aa5438f8eaaef2d30'
torchsharp_source="$repo_root/build/M1/torchsharp-src"
torchsharp_build="$repo_root/build/M1/torchsharp-build"
cuda_build="$repo_root/build/M1/ltx-cuda"

if [[ ! -x $python_path ]]; then
    uv venv "$repo_root/.venv" --python python3
fi

if ! "$python_path" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(
    0
    if torch.__version__ == "2.13.0+cu132"
    and torch.version.cuda == "13.2"
    and torch.compiled_with_cxx11_abi()
    else 1
)
PY
then
    uv pip install --python "$python_path" --reinstall "$torch_url"
fi

"$python_path" - <<'PY'
import torch
if torch.__version__ != "2.13.0+cu132":
    raise SystemExit(f"PyTorch version mismatch: {torch.__version__}")
if torch.version.cuda != "13.2":
    raise SystemExit(f"PyTorch CUDA mismatch: {torch.version.cuda}")
if not torch.compiled_with_cxx11_abi():
    raise SystemExit("PyTorch must use the C++11 ABI")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
PY

if [[ ! -d $torchsharp_source/.git ]]; then
    mkdir -p "$(dirname "$torchsharp_source")"
    git clone --filter=blob:none --no-checkout https://github.com/dotnet/TorchSharp.git "$torchsharp_source"
    git -C "$torchsharp_source" checkout --detach "$torchsharp_revision"
    git -C "$torchsharp_source" apply "$repo_root/eng/torchsharp/ltx-pytorch-2.13.patch"
fi

if [[ $(git -C "$torchsharp_source" rev-parse HEAD) != "$torchsharp_revision" ]]; then
    echo "TorchSharp source revision mismatch" >&2
    exit 1
fi
if ! git -C "$torchsharp_source" diff --check; then
    echo "TorchSharp fork patch is invalid" >&2
    exit 1
fi
if ! git -C "$torchsharp_source" apply --reverse --check "$repo_root/eng/torchsharp/ltx-pytorch-2.13.patch"; then
    echo "TorchSharp source does not contain the complete LTX fork patch" >&2
    exit 1
fi

torch_cmake=$(
    "$python_path" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)
cmake -S "$torchsharp_source/src/Native" -B "$torchsharp_build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBTORCH_PATH="$torch_cmake"
cmake --build "$torchsharp_build" --target LibTorchSharp --parallel 2

compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
cmake -S "$repo_root/native" -B "$cuda_build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${compute_capability}-real"
cmake --build "$cuda_build" --target ltx_cuda --parallel 2

torchsharp_library="$torchsharp_build/LibTorchSharp/libLibTorchSharp.so"
cuda_library="$cuda_build/libltx_cuda.so"
if [[ ! -f $torchsharp_library || ! -f $cuda_library ]]; then
    echo "M1 native build did not produce both libraries" >&2
    exit 1
fi

echo "M1 ABI libraries prepared"
