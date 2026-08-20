#include "ltx_cuda.h"

#include <cuda_runtime.h>

namespace {

__global__ void affine_kernel(
    const float* input,
    float* output,
    const size_t element_count,
    const float scale,
    const float bias)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count)
    {
        output[index] = input[index] * scale + bias;
    }
}

int launch_affine(
    const float* input,
    float* output,
    const size_t element_count,
    const float scale,
    const float bias,
    cudaStream_t stream)
{
    constexpr unsigned int block_size = 256;
    const auto block_count = static_cast<unsigned int>((element_count + block_size - 1) / block_size);
    affine_kernel<<<block_count, block_size, 0, stream>>>(
        input, output, element_count, scale, bias);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

uint32_t ltx_cuda_abi_version(void)
{
    return LTX_CUDA_ABI_VERSION;
}

int32_t ltx_cuda_compiled_cuda_version(void)
{
    return CUDART_VERSION;
}

int32_t ltx_cuda_device_compute_capability(void)
{
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        return -static_cast<int32_t>(status);
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess)
    {
        return -static_cast<int32_t>(status);
    }

    return properties.major * 10 + properties.minor;
}

int32_t ltx_cuda_affine_f32_device(
    const float* input,
    float* output,
    const size_t element_count,
    const float scale,
    const float bias,
    void* stream)
{
    if (input == nullptr || output == nullptr || element_count == 0)
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }

    return launch_affine(
        input,
        output,
        element_count,
        scale,
        bias,
        reinterpret_cast<cudaStream_t>(stream));
}

int32_t ltx_cuda_affine_f32_host(
    const float* input,
    float* output,
    const size_t element_count,
    const float scale,
    const float bias)
{
    if (input == nullptr || output == nullptr || element_count == 0)
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }

    const size_t byte_count = element_count * sizeof(float);
    float* device_input = nullptr;
    float* device_output = nullptr;
    auto status = cudaMalloc(&device_input, byte_count);
    if (status == cudaSuccess)
    {
        status = cudaMalloc(&device_output, byte_count);
    }
    if (status == cudaSuccess)
    {
        status = cudaMemcpy(device_input, input, byte_count, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess)
    {
        status = static_cast<cudaError_t>(
            launch_affine(device_input, device_output, element_count, scale, bias, nullptr));
    }
    if (status == cudaSuccess)
    {
        status = cudaMemcpy(output, device_output, byte_count, cudaMemcpyDeviceToHost);
    }

    const auto output_free_status = cudaFree(device_output);
    const auto input_free_status = cudaFree(device_input);
    if (status == cudaSuccess)
    {
        status = output_free_status != cudaSuccess ? output_free_status : input_free_status;
    }
    return static_cast<int32_t>(status);
}
