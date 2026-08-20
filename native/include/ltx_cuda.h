#ifndef LTX_CUDA_H
#define LTX_CUDA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define LTX_CUDA_API __declspec(dllexport)
#else
#define LTX_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LTX_CUDA_ABI_MAJOR 1u
#define LTX_CUDA_ABI_MINOR 0u
#define LTX_CUDA_ABI_VERSION ((LTX_CUDA_ABI_MAJOR << 16u) | LTX_CUDA_ABI_MINOR)

LTX_CUDA_API uint32_t ltx_cuda_abi_version(void);
LTX_CUDA_API int32_t ltx_cuda_compiled_cuda_version(void);
LTX_CUDA_API int32_t ltx_cuda_device_compute_capability(void);

/* Host-buffer smoke path. Production kernels use device-buffer entry points. */
LTX_CUDA_API int32_t ltx_cuda_affine_f32_host(
    const float* input,
    float* output,
    size_t element_count,
    float scale,
    float bias);

/* Asynchronous device-buffer ABI; stream is a cudaStream_t represented as void*. */
LTX_CUDA_API int32_t ltx_cuda_affine_f32_device(
    const float* input,
    float* output,
    size_t element_count,
    float scale,
    float bias,
    void* stream);

#ifdef __cplusplus
}
#endif

#endif
