#include <cuda_runtime.h>

namespace {

__global__ void double_value(const float input, float* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = input * 2.0F;
    }
}

}  // namespace

extern "C" int ltx_cuda_smoke(const float input, float* output, int* compute_capability) {
    if (output == nullptr || compute_capability == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    *compute_capability = properties.major * 10 + properties.minor;

    float* device_output = nullptr;
    status = cudaMalloc(&device_output, sizeof(float));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    double_value<<<1, 1>>>(input, device_output);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, device_output, sizeof(float), cudaMemcpyDeviceToHost);
    }
    const cudaError_t free_status = cudaFree(device_output);
    if (status == cudaSuccess) {
        status = free_status;
    }
    return static_cast<int>(status);
}
