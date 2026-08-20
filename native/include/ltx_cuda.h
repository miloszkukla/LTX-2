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

/* M4 correctness ABI. Host entry points stage through device memory and execute
 * CUDA kernels; they are intentionally small-fixture friendly rather than a
 * replacement for the asynchronous production device-buffer ABI. */
LTX_CUDA_API int32_t ltx_cuda_kernel_count(void);
LTX_CUDA_API const char* ltx_cuda_kernel_name(int32_t index);

LTX_CUDA_API int32_t ltx_cuda_fused_add_round_f32_host(
    const float* delta, const float* weight, float* output,
    size_t element_count, uint32_t seed);

LTX_CUDA_API int32_t ltx_cuda_na3d_f32_host(
    const float* query, const float* key, const float* value, float* output,
    int32_t batch, int32_t time, int32_t height, int32_t width,
    int32_t heads, int32_t head_dim,
    int32_t kernel_time, int32_t kernel_height, int32_t kernel_width,
    int32_t causal_time, int32_t causal_height, int32_t causal_width,
    float scale);

LTX_CUDA_API int32_t ltx_cuda_swiglu_up_mul_f32_host(
    const float* input, const float* up_weight, const float* gate,
    float* output, int32_t rows, int32_t input_dim, int32_t output_dim);
LTX_CUDA_API int32_t ltx_cuda_swiglu_gate_up_f32_host(
    const float* input, const float* gate_weight, const float* up_weight,
    float* output, int32_t rows, int32_t input_dim, int32_t output_dim);

/* variant: 0=sm89, 1=sm90, 2=sm90+bias. B is row-major (N,K). */
LTX_CUDA_API int32_t ltx_cuda_fp8_gemm_f32_host(
    const float* a, const float* b, const float* bias, float* output,
    int32_t rows, int32_t output_dim, int32_t inner_dim, int32_t variant);

LTX_CUDA_API int32_t ltx_cuda_nvfp4_quantize_f32_host(
    const float* input, uint8_t* packed, uint8_t* block_scales,
    int32_t rows, int32_t columns, float per_tensor_scale,
    int32_t hi_first, int32_t variant);
LTX_CUDA_API int32_t ltx_cuda_nvfp4_dequantize_f32_host(
    const uint8_t* packed, const uint8_t* block_scales, float* output,
    int32_t rows, int32_t columns, float per_tensor_scale, int32_t hi_first);
LTX_CUDA_API int32_t ltx_cuda_nvfp4_scaled_mm_f32_host(
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    float per_tensor_scale_a, float per_tensor_scale_b,
    const float* bias, float* output,
    int32_t rows, int32_t output_dim, int32_t inner_dim,
    int32_t hi_first, int32_t has_bias);
LTX_CUDA_API int32_t ltx_cuda_mul_scalars_f32_host(float a, float b, float* output);
LTX_CUDA_API int32_t ltx_cuda_amax_scale_f32_host(
    const float* input, size_t element_count, float divisor, float* output);
LTX_CUDA_API int32_t ltx_cuda_gemm_alpha_mode(void);
LTX_CUDA_API int32_t ltx_cuda_set_gemm_alpha_on_device(int32_t enabled);
LTX_CUDA_API int32_t ltx_cuda_set_gemm_autotune(int32_t enabled);
LTX_CUDA_API int32_t ltx_cuda_probe_gemm_support(
    int32_t rows, int32_t output_dim, int32_t inner_dim);

LTX_CUDA_API int32_t ltx_cuda_fp6_pack_host(
    const uint8_t* input, uint8_t* output, int32_t rows, int32_t columns);
LTX_CUDA_API int32_t ltx_cuda_fp6_unpack_host(
    const uint8_t* input, uint8_t* output, int32_t rows, int32_t columns);

LTX_CUDA_API int32_t ltx_cuda_rms_norm_rope_f32_host(
    const float* input, const float* weights,
    const float* cosine, const float* sine, float* output,
    int32_t rows, int32_t hidden_dim, int32_t has_weights, float epsilon);
LTX_CUDA_API int32_t ltx_cuda_rms_norm_split_rope_f32_host(
    const float* input, const float* weights,
    const float* cosine, const float* sine, float* output,
    int32_t rows, int32_t heads, int32_t head_dim, float epsilon);

LTX_CUDA_API int32_t ltx_cuda_rowwise_int8_quantize_f32_host(
    const float* input, int8_t* output, float* scales,
    int32_t rows, int32_t columns);
LTX_CUDA_API int32_t ltx_cuda_blockwise_fp8_quantize_f32_host(
    const float* input, uint8_t* output, float* scales,
    int32_t rows, int32_t columns, int32_t block_size, int32_t use_gelu);
LTX_CUDA_API int32_t ltx_cuda_blockwise_fp8_norm_quantize_f32_host(
    const float* input, const float* norm_scale, const float* norm_shift,
    uint8_t* output, float* scales,
    int32_t rows, int32_t columns, int32_t block_size, float epsilon);
LTX_CUDA_API int32_t ltx_cuda_quant_rms_fma_f32_host(
    const float* x, const float* y, const float* z,
    float* residual, uint8_t* output, float* scales,
    int32_t rows, int32_t columns, int32_t block_size);
LTX_CUDA_API int32_t ltx_cuda_gated_attention_f32_host(
    const float* input, const float* gate_logits, float* output,
    int32_t rows, int32_t heads, int32_t head_dim);
LTX_CUDA_API int32_t ltx_cuda_blockwise_fp8_dequantize_f32_host(
    const uint8_t* input, const float* scales, float* output,
    int32_t rows, int32_t columns, int32_t block_size);

#ifdef __cplusplus
}
#endif

#endif
