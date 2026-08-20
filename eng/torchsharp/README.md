# LTX TorchSharp ABI fork

M1 pins the TorchSharp managed API to `0.107.0` and its upstream source to
`8f4def03b641b6753f18076aa5438f8eaaef2d30`. The native `LibTorchSharp`
bridge is rebuilt from that revision against the official LTX wheel
`torch-2.13.0+cu132-cp312-cp312-manylinux_2_28_x86_64.whl`.

`ltx-pytorch-2.13.patch` is the maintained fork delta. It adapts TorchSharp's
custom-autograd node ownership to PyTorch 2.13 intrusive pointers and adds
exports that make the PyTorch, CUDA, C++ ABI, and LTX ABI versions testable.
No stock TorchSharp CUDA runtime package is referenced or loaded.

Run `scripts/remote/prepare-m1-abi.sh` to reproduce both the forked bridge and
`libltx_cuda.so`. Build products remain under the ignored `build/M1` tree.
