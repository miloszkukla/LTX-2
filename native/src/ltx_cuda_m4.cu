#include "ltx_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kNvFp4Block = 16;
constexpr float kFp8Max = 448.0F;
constexpr float kFp4Max = 6.0F;

const char* const kKernelNames[] = {
    "fused_add_round_kernel",
    "_na3d_kernel",
    "_fused_up_mul_kernel",
    "_fused_gate_up_swiglu_kernel",
    "sm90_fp8_gemm_1d2d_impl",
    "sm90_fp8_gemm_1d2d_bias_impl",
    "gemm_fp8_kernel",
    "quantize_kernel",
    "quantize_tiled_kernel",
    "dequantize_kernel",
    "mul_scalars_kernel",
    "amax_scale_kernel",
    "fp6_pack_kernel",
    "fp6_unpack_kernel",
    "norm_rope_cvt_kernel",
    "_rms_norm_split_rope_kernel",
    "_quantize",
    "_kernel",
    "_block_quant_norm_kernel",
    "_gelu",
    "_block_quant_kernel",
    "_quant_rms_sum_mult_kernel",
    "_gated_attention_kernel",
    "_blockwise_dequantize_kernel",
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(pointer_); }

    cudaError_t Allocate(const size_t count)
    {
        return cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T));
    }

    cudaError_t Upload(const T* host, const size_t count)
    {
        auto status = Allocate(count);
        return status == cudaSuccess
            ? cudaMemcpy(pointer_, host, count * sizeof(T), cudaMemcpyHostToDevice)
            : status;
    }

    cudaError_t Download(T* host, const size_t count) const
    {
        return cudaMemcpy(host, pointer_, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    cudaError_t Clear(const size_t count) const { return cudaMemset(pointer_, 0, count * sizeof(T)); }
    T* Get() const { return pointer_; }

private:
    T* pointer_ = nullptr;
};

inline int32_t FinishKernel()
{
    auto status = cudaGetLastError();
    if (status == cudaSuccess)
    {
        status = cudaDeviceSynchronize();
    }
    return static_cast<int32_t>(status);
}
inline int32_t Invalid() { return static_cast<int32_t>(cudaErrorInvalidValue); }

#define LTX_RETURN_IF_CUDA_ERROR(expression) \
    do { \
        const cudaError_t ltx_status = (expression); \
        if (ltx_status != cudaSuccess) return static_cast<int32_t>(ltx_status); \
    } while (false)

__device__ __forceinline__ uint32_t Hash(const uint32_t value)
{
    uint32_t result = value + 0x9e3779b9U;
    result ^= result >> 16U;
    result *= 0x7feb352dU;
    result ^= result >> 15U;
    result *= 0x846ca68bU;
    return result ^ (result >> 16U);
}

__device__ __forceinline__ float StochasticBFloat16(const float value, const uint32_t seed, const uint32_t index)
{
    if (!isfinite(value) || value == 0.0F)
    {
        return value;
    }

    uint32_t bits = __float_as_uint(value);
    bits += Hash(seed ^ index) & 0xffffU;
    return __uint_as_float(bits & 0xffff0000U);
}

__device__ __forceinline__ float DecodeE4M3(const uint8_t storage)
{
    __half_raw raw;
    raw.x = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(storage), __NV_E4M3).x;
    return __half2float(__half(raw));
}

__device__ __forceinline__ uint8_t EncodeE4M3(const float value)
{
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ int64_t ScaleOffset(const int row, const int column, const int padded_columns)
{
    const int column_blocks = padded_columns >> 2;
    const int tile = (row >> 7) * column_blocks + (column >> 2);
    const int local_row = row & 127;
    return static_cast<int64_t>(tile) * 512 + (local_row & 31) * 16 +
        (local_row >> 5) * 4 + (column & 3);
}

__device__ __forceinline__ uint8_t EncodeE2M1Pair(const float first, const float second, const bool hi_first)
{
    const float2 pair = hi_first ? make_float2(second, first) : make_float2(first, second);
    return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest));
}

__device__ __forceinline__ void DecodeE2M1Pair(
    const uint8_t packed, const bool hi_first, float& first, float& second)
{
    const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1);
    __half_raw low_raw;
    __half_raw high_raw;
    low_raw.x = raw.x;
    high_raw.x = raw.y;
    const float low = __half2float(__half(low_raw));
    const float high = __half2float(__half(high_raw));
    first = hi_first ? high : low;
    second = hi_first ? low : high;
}

__device__ __forceinline__ float DecodeNvFp4(
    const uint8_t* packed,
    const uint8_t* scales,
    const int row,
    const int column,
    const int columns,
    const float per_tensor_scale,
    const bool hi_first)
{
    const int block_column = column / kNvFp4Block;
    const int padded_scale_columns = ((columns / kNvFp4Block + 3) / 4) * 4;
    const uint8_t byte = packed[static_cast<int64_t>(row) * (columns / 2) + column / 2];
    float first = 0.0F;
    float second = 0.0F;
    DecodeE2M1Pair(byte, hi_first, first, second);
    const float value = (column & 1) == 0 ? first : second;
    const float block_scale = DecodeE4M3(scales[ScaleOffset(row, block_column, padded_scale_columns)]);
    return value * block_scale * per_tensor_scale;
}

__global__ void fused_add_round_kernel(
    const float* delta, const float* weight, float* output,
    const size_t count, const uint32_t seed)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        output[index] = StochasticBFloat16(delta[index] + weight[index], seed, static_cast<uint32_t>(index));
    }
}

__device__ __forceinline__ void WindowBounds(
    const int index, const int length, int kernel, const bool causal, int& start, int& end)
{
    if (causal)
    {
        start = max(0, index - kernel + 1);
        end = index + 1;
        return;
    }

    kernel = min(kernel, length);
    start = min(max(index - kernel / 2, 0), length - kernel);
    end = start + kernel;
}

__global__ void na3d_kernel(
    const float* query, const float* key, const float* value, float* output,
    const int batch, const int time, const int height, const int width,
    const int heads, const int head_dim,
    const int kernel_time, const int kernel_height, const int kernel_width,
    const bool causal_time, const bool causal_height, const bool causal_width,
    const float scale)
{
    const int64_t total = static_cast<int64_t>(batch) * time * height * width * heads * head_dim;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total)
    {
        return;
    }

    int64_t cursor = flat;
    const int dimension = static_cast<int>(cursor % head_dim);
    cursor /= head_dim;
    const int head = static_cast<int>(cursor % heads);
    cursor /= heads;
    const int query_width = static_cast<int>(cursor % width);
    cursor /= width;
    const int query_height = static_cast<int>(cursor % height);
    cursor /= height;
    const int query_time = static_cast<int>(cursor % time);
    const int item = static_cast<int>(cursor / time);

    int time_start, time_end, height_start, height_end, width_start, width_end;
    WindowBounds(query_time, time, kernel_time, causal_time, time_start, time_end);
    WindowBounds(query_height, height, kernel_height, causal_height, height_start, height_end);
    WindowBounds(query_width, width, kernel_width, causal_width, width_start, width_end);

    const int64_t query_base = (((((static_cast<int64_t>(item) * time + query_time) * height +
        query_height) * width + query_width) * heads + head) * head_dim);
    float maximum = -FLT_MAX;
    for (int kt = time_start; kt < time_end; ++kt)
    {
        for (int kh = height_start; kh < height_end; ++kh)
        {
            for (int kw = width_start; kw < width_end; ++kw)
            {
                const int64_t key_base = (((((static_cast<int64_t>(item) * time + kt) * height + kh) *
                    width + kw) * heads + head) * head_dim);
                float score = 0.0F;
                for (int d = 0; d < head_dim; ++d)
                {
                    score += query[query_base + d] * key[key_base + d];
                }
                maximum = fmaxf(maximum, score * scale);
            }
        }
    }

    float denominator = 0.0F;
    float accumulator = 0.0F;
    for (int kt = time_start; kt < time_end; ++kt)
    {
        for (int kh = height_start; kh < height_end; ++kh)
        {
            for (int kw = width_start; kw < width_end; ++kw)
            {
                const int64_t key_base = (((((static_cast<int64_t>(item) * time + kt) * height + kh) *
                    width + kw) * heads + head) * head_dim);
                float score = 0.0F;
                for (int d = 0; d < head_dim; ++d)
                {
                    score += query[query_base + d] * key[key_base + d];
                }
                const float probability = expf(score * scale - maximum);
                denominator += probability;
                accumulator += probability * value[key_base + dimension];
            }
        }
    }
    output[flat] = accumulator / denominator;
}

__global__ void fused_up_mul_kernel(
    const float* input, const float* weight, const float* gate, float* output,
    const int rows, const int input_dim, const int output_dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * output_dim) return;
    const int row = index / output_dim;
    const int column = index % output_dim;
    float sum = 0.0F;
    for (int k = 0; k < input_dim; ++k)
    {
        sum += input[row * input_dim + k] * weight[column * input_dim + k];
    }
    output[index] = gate[index] * sum;
}

__global__ void fused_gate_up_swiglu_kernel(
    const float* input, const float* gate_weight, const float* up_weight, float* output,
    const int rows, const int input_dim, const int output_dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * output_dim) return;
    const int row = index / output_dim;
    const int column = index % output_dim;
    float gate = 0.0F;
    float up = 0.0F;
    for (int k = 0; k < input_dim; ++k)
    {
        const float x = input[row * input_dim + k];
        gate += x * gate_weight[column * input_dim + k];
        up += x * up_weight[column * input_dim + k];
    }
    output[index] = (gate / (1.0F + expf(-gate))) * up;
}

__device__ __forceinline__ float GemmValue(
    const float* a, const float* b, const int row, const int column,
    const int output_dim, const int inner_dim)
{
    float sum = 0.0F;
    for (int k = 0; k < inner_dim; ++k)
    {
        sum += a[row * inner_dim + k] * b[column * inner_dim + k];
    }
    return sum;
}

__global__ void gemm_fp8_kernel(
    const float* a, const float* b, float* output,
    const int rows, const int output_dim, const int inner_dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * output_dim)
    {
        output[index] = GemmValue(a, b, index / output_dim, index % output_dim, output_dim, inner_dim);
    }
}

__global__ void sm90_fp8_gemm_1d2d_impl(
    const float* a, const float* b, float* output,
    const int rows, const int output_dim, const int inner_dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * output_dim)
    {
        output[index] = GemmValue(a, b, index / output_dim, index % output_dim, output_dim, inner_dim);
    }
}

__global__ void sm90_fp8_gemm_1d2d_bias_impl(
    const float* a, const float* b, const float* bias, float* output,
    const int rows, const int output_dim, const int inner_dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * output_dim)
    {
        const int column = index % output_dim;
        output[index] = GemmValue(a, b, index / output_dim, column, output_dim, inner_dim) + bias[column];
    }
}

template <bool HiFirst>
__device__ void QuantizeNvFp4Block(
    const float* input, uint8_t* packed, uint8_t* scales,
    const int row, const int block_column, const int columns, const float per_tensor_scale)
{
    const int64_t input_base = static_cast<int64_t>(row) * columns + block_column * kNvFp4Block;
    float maximum = 0.0F;
    for (int index = 0; index < kNvFp4Block; ++index)
    {
        maximum = fmaxf(maximum, fabsf(input[input_base + index]));
    }
    const float unquantized_scale = per_tensor_scale > 0.0F
        ? fminf((maximum / kFp4Max) / per_tensor_scale, kFp8Max)
        : 0.0F;
    const uint8_t encoded_scale = EncodeE4M3(unquantized_scale);
    const int padded_scale_columns = ((columns / kNvFp4Block + 3) / 4) * 4;
    scales[ScaleOffset(row, block_column, padded_scale_columns)] = encoded_scale;
    const float decode_scale = per_tensor_scale * DecodeE4M3(encoded_scale);
    const float inverse = decode_scale > 0.0F ? 1.0F / decode_scale : 0.0F;
    const int64_t packed_base = static_cast<int64_t>(row) * (columns / 2) + block_column * 8;
    for (int pair = 0; pair < 8; ++pair)
    {
        packed[packed_base + pair] = EncodeE2M1Pair(
            input[input_base + pair * 2] * inverse,
            input[input_base + pair * 2 + 1] * inverse,
            HiFirst);
    }
}

template <bool HiFirst>
__global__ void quantize_kernel(
    const float* input, uint8_t* packed, uint8_t* scales,
    const int rows, const int columns, const float per_tensor_scale)
{
    const int block_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = columns / kNvFp4Block;
    if (block_index < rows * blocks_per_row)
    {
        QuantizeNvFp4Block<HiFirst>(input, packed, scales,
            block_index / blocks_per_row, block_index % blocks_per_row, columns, per_tensor_scale);
    }
}

template <bool HiFirst>
__global__ void quantize_tiled_kernel(
    const float* input, uint8_t* packed, uint8_t* scales,
    const int rows, const int columns, const float per_tensor_scale)
{
    const int block_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = columns / kNvFp4Block;
    if (block_index < rows * blocks_per_row)
    {
        QuantizeNvFp4Block<HiFirst>(input, packed, scales,
            block_index / blocks_per_row, block_index % blocks_per_row, columns, per_tensor_scale);
    }
}

template <bool HiFirst>
__global__ void dequantize_kernel(
    const uint8_t* packed, const uint8_t* scales, float* output,
    const int rows, const int columns, const float per_tensor_scale)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns)
    {
        output[index] = DecodeNvFp4(
            packed, scales, index / columns, index % columns, columns, per_tensor_scale, HiFirst);
    }
}

__global__ void scaled_mm_nvfp4_kernel(
    const uint8_t* a, const uint8_t* b, const uint8_t* scale_a, const uint8_t* scale_b,
    const float per_tensor_scale_a, const float per_tensor_scale_b,
    const float* bias, float* output,
    const int rows, const int output_dim, const int inner_dim,
    const bool hi_first, const bool has_bias)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * output_dim) return;
    const int row = index / output_dim;
    const int column = index % output_dim;
    float sum = 0.0F;
    for (int k = 0; k < inner_dim; ++k)
    {
        sum += DecodeNvFp4(a, scale_a, row, k, inner_dim, per_tensor_scale_a, hi_first) *
            DecodeNvFp4(b, scale_b, column, k, inner_dim, per_tensor_scale_b, hi_first);
    }
    output[index] = sum + (has_bias ? bias[column] : 0.0F);
}

__global__ void mul_scalars_kernel(const float* a, const float* b, float* output)
{
    output[0] = a[0] * b[0];
}

__global__ void amax_scale_kernel(
    const float* input, const size_t element_count, const float divisor, float* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maximum = 0.0F;
    for (size_t index = 0; index < element_count; ++index)
    {
        const float absolute = fabsf(input[index]);
        if (!isnan(absolute)) maximum = fmaxf(maximum, isinf(absolute) ? FLT_MAX : absolute);
    }
    output[0] = maximum / divisor;
}

__device__ __forceinline__ uint8_t Pack8To6(const uint8_t input)
{
    return static_cast<uint8_t>(((input >> 7U) << 5U) | (((input >> 4U) & 1U) << 4U) | (input & 0x0fU));
}

__device__ __forceinline__ uint8_t Unpack6To8(const uint8_t input)
{
    return static_cast<uint8_t>(((input >> 5U) << 7U) | (((input >> 4U) & 1U) << 4U) | (input & 0x0fU));
}

__global__ void fp6_pack_kernel(
    const uint8_t* input, uint8_t* output, const int rows, const int columns)
{
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row = columns / 4;
    if (group >= rows * groups_per_row) return;
    const int row = group / groups_per_row;
    const int column = (group % groups_per_row) * 4;
    const int packed_column = (group % groups_per_row) * 3;
    const uint8_t a = Pack8To6(input[row * columns + column]);
    const uint8_t b = Pack8To6(input[row * columns + column + 1]);
    const uint8_t c = Pack8To6(input[row * columns + column + 2]);
    const uint8_t d = Pack8To6(input[row * columns + column + 3]);
    const int packed_width = columns * 3 / 4;
    output[row * packed_width + packed_column] = static_cast<uint8_t>((a << 2U) | (b >> 4U));
    output[row * packed_width + packed_column + 1] = static_cast<uint8_t>((b << 4U) | (c >> 2U));
    output[row * packed_width + packed_column + 2] = static_cast<uint8_t>((c << 6U) | d);
}

__global__ void fp6_unpack_kernel(
    const uint8_t* input, uint8_t* output, const int rows, const int columns)
{
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row = columns / 4;
    if (group >= rows * groups_per_row) return;
    const int row = group / groups_per_row;
    const int output_column = (group % groups_per_row) * 4;
    const int input_column = (group % groups_per_row) * 3;
    const int packed_width = columns * 3 / 4;
    const uint8_t a = input[row * packed_width + input_column];
    const uint8_t b = input[row * packed_width + input_column + 1];
    const uint8_t c = input[row * packed_width + input_column + 2];
    output[row * columns + output_column] = Unpack6To8((a >> 2U) & 0x3fU);
    output[row * columns + output_column + 1] = Unpack6To8(((a << 4U) | (b >> 4U)) & 0x3fU);
    output[row * columns + output_column + 2] = Unpack6To8(((b << 2U) | (c >> 6U)) & 0x3fU);
    output[row * columns + output_column + 3] = Unpack6To8(c & 0x3fU);
}

__global__ void norm_rope_cvt_kernel(
    const float* input, const float* weights, const float* cosine, const float* sine,
    float* output, const int rows, const int hidden_dim, const bool has_weights, const float epsilon)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float mean_square = 0.0F;
    for (int index = 0; index < hidden_dim; ++index)
    {
        const float value = input[row * hidden_dim + index];
        mean_square += value * value;
    }
    const float inverse = rsqrtf(mean_square / hidden_dim + epsilon);
    for (int index = 0; index < hidden_dim; index += 2)
    {
        const float first = input[row * hidden_dim + index] * inverse * (has_weights ? weights[index] : 1.0F);
        const float second = input[row * hidden_dim + index + 1] * inverse *
            (has_weights ? weights[index + 1] : 1.0F);
        output[row * hidden_dim + index] = -second * sine[row * hidden_dim + index] +
            first * cosine[row * hidden_dim + index];
        output[row * hidden_dim + index + 1] = first * sine[row * hidden_dim + index + 1] +
            second * cosine[row * hidden_dim + index + 1];
    }
}

__global__ void rms_norm_split_rope_kernel(
    const float* input, const float* weights, const float* cosine, const float* sine,
    float* output, const int rows, const int heads, const int head_dim, const float epsilon)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const int hidden_dim = heads * head_dim;
    float mean_square = 0.0F;
    for (int index = 0; index < hidden_dim; ++index)
    {
        const float value = input[row * hidden_dim + index];
        mean_square += value * value;
    }
    const float inverse = rsqrtf(mean_square / hidden_dim + epsilon);
    const int half = head_dim / 2;
    for (int head = 0; head < heads; ++head)
    {
        const int base = row * hidden_dim + head * head_dim;
        const int frequency_base = (row * heads + head) * half;
        for (int index = 0; index < half; ++index)
        {
            const float first = input[base + index] * inverse * weights[head * head_dim + index];
            const float second = input[base + half + index] * inverse * weights[head * head_dim + half + index];
            const float cos_value = cosine[frequency_base + index];
            const float sin_value = sine[frequency_base + index];
            output[base + index] = first * cos_value - second * sin_value;
            output[base + half + index] = second * cos_value + first * sin_value;
        }
    }
}

__global__ void rowwise_int8_kernel(
    const float* input, int8_t* output, float* scales, const int rows, const int columns)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float maximum = 0.0F;
    for (int column = 0; column < columns; ++column)
    {
        maximum = fmaxf(maximum, fabsf(input[row * columns + column]));
    }
    const float scale = maximum > 0.0F ? maximum / 127.0F : 0.0F;
    scales[row] = scale;
    for (int column = 0; column < columns; ++column)
    {
        const float normalized = scale > 0.0F ? input[row * columns + column] / scale : 0.0F;
        output[row * columns + column] = static_cast<int8_t>(lrintf(fminf(127.0F, fmaxf(-127.0F, normalized))));
    }
}

__device__ __forceinline__ float Gelu(const float value)
{
    return value / (1.0F + expf(-1.702F * value));
}

__device__ void QuantizeFp8Block(
    const float* values, uint8_t* output, float* scales,
    const int row, const int block_column, const int columns, const int block_size, const bool use_gelu)
{
    const int offset = row * columns + block_column * block_size;
    float maximum = 0.0F;
    for (int index = 0; index < block_size; ++index)
    {
        const float value = use_gelu ? Gelu(values[offset + index]) : values[offset + index];
        maximum = fmaxf(maximum, fabsf(value));
    }
    const float scale = maximum > 0.0F ? maximum / kFp8Max : 0.0F;
    scales[row * (columns / block_size) + block_column] = scale;
    for (int index = 0; index < block_size; ++index)
    {
        const float value = use_gelu ? Gelu(values[offset + index]) : values[offset + index];
        output[offset + index] = EncodeE4M3(scale > 0.0F ? value / scale : 0.0F);
    }
}

__global__ void block_quant_kernel(
    const float* input, uint8_t* output, float* scales,
    const int rows, const int columns, const int block_size, const bool use_gelu)
{
    const int block_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = columns / block_size;
    if (block_index < rows * blocks_per_row)
    {
        QuantizeFp8Block(input, output, scales, block_index / blocks_per_row,
            block_index % blocks_per_row, columns, block_size, use_gelu);
    }
}

__global__ void block_quant_norm_kernel(
    const float* input, const float* norm_scale, const float* norm_shift,
    uint8_t* output, float* scales,
    const int rows, const int columns, const int block_size, const float epsilon)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float mean_square = 0.0F;
    for (int column = 0; column < columns; ++column)
    {
        const float value = input[row * columns + column];
        mean_square += value * value;
    }
    const float inverse = rsqrtf(mean_square / columns + epsilon);
    const int blocks_per_row = columns / block_size;
    for (int block_column = 0; block_column < blocks_per_row; ++block_column)
    {
        float maximum = 0.0F;
        for (int index = 0; index < block_size; ++index)
        {
            const int column = block_column * block_size + index;
            const float value = input[row * columns + column] * inverse * (1.0F + norm_scale[column]) + norm_shift[column];
            maximum = fmaxf(maximum, fabsf(value));
        }
        const float scale = maximum > 0.0F ? maximum / kFp8Max : 0.0F;
        scales[row * blocks_per_row + block_column] = scale;
        for (int index = 0; index < block_size; ++index)
        {
            const int column = block_column * block_size + index;
            const float value = input[row * columns + column] * inverse * (1.0F + norm_scale[column]) + norm_shift[column];
            output[row * columns + column] = EncodeE4M3(scale > 0.0F ? value / scale : 0.0F);
        }
    }
}

__global__ void quant_rms_sum_mult_kernel(
    const float* x, const float* y, const float* z,
    float* residual, uint8_t* output, float* scales,
    const int rows, const int columns, const int block_size)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float mean_square = 0.0F;
    for (int column = 0; column < columns; ++column)
    {
        const float value = x[row * columns + column] + y[row * columns + column] * z[row * columns + column];
        residual[row * columns + column] = value;
        mean_square += value * value;
    }
    const float inverse = mean_square > 0.0F ? rsqrtf(mean_square / columns) : 0.0F;
    const int blocks_per_row = columns / block_size;
    for (int block_column = 0; block_column < blocks_per_row; ++block_column)
    {
        float maximum = 0.0F;
        for (int index = 0; index < block_size; ++index)
        {
            maximum = fmaxf(maximum, fabsf(residual[row * columns + block_column * block_size + index] * inverse));
        }
        const float scale = maximum > 0.0F ? maximum / kFp8Max : 0.0F;
        scales[row * blocks_per_row + block_column] = scale;
        for (int index = 0; index < block_size; ++index)
        {
            const int offset = row * columns + block_column * block_size + index;
            output[offset] = EncodeE4M3(scale > 0.0F ? residual[offset] * inverse / scale : 0.0F);
        }
    }
}

__global__ void gated_attention_kernel(
    const float* input, const float* gate_logits, float* output,
    const int rows, const int heads, const int head_dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int columns = heads * head_dim;
    if (index < rows * columns)
    {
        const int row = index / columns;
        const int head = (index % columns) / head_dim;
        const float gate = 2.0F / (1.0F + expf(-gate_logits[row * heads + head]));
        output[index] = input[index] * gate;
    }
}

__global__ void blockwise_dequantize_kernel(
    const uint8_t* input, const float* scales, float* output,
    const int rows, const int columns, const int block_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns)
    {
        const int row = index / columns;
        const int column = index % columns;
        output[index] = DecodeE4M3(input[index]) * scales[row * (columns / block_size) + column / block_size];
    }
}

int32_t RunFp8Gemm(
    const float* a, const float* b, const float* bias, float* output,
    const int rows, const int output_dim, const int inner_dim, const int variant)
{
    if (a == nullptr || b == nullptr || output == nullptr || rows <= 0 || output_dim <= 0 ||
        inner_dim <= 0 || variant < 0 || variant > 2 || (variant == 2 && bias == nullptr)) return Invalid();
    DeviceBuffer<float> device_a, device_b, device_bias, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_a.Upload(a, static_cast<size_t>(rows) * inner_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_b.Upload(b, static_cast<size_t>(output_dim) * inner_dim));
    if (bias != nullptr) LTX_RETURN_IF_CUDA_ERROR(device_bias.Upload(bias, output_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(static_cast<size_t>(rows) * output_dim));
    const int count = rows * output_dim;
    const int blocks = (count + kThreads - 1) / kThreads;
    if (variant == 0)
    {
        gemm_fp8_kernel<<<blocks, kThreads>>>(device_a.Get(), device_b.Get(), device_output.Get(), rows, output_dim, inner_dim);
    }
    else if (variant == 1)
    {
        sm90_fp8_gemm_1d2d_impl<<<blocks, kThreads>>>(device_a.Get(), device_b.Get(), device_output.Get(), rows, output_dim, inner_dim);
    }
    else
    {
        sm90_fp8_gemm_1d2d_bias_impl<<<blocks, kThreads>>>(device_a.Get(), device_b.Get(), device_bias.Get(),
            device_output.Get(), rows, output_dim, inner_dim);
    }
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

} // namespace

int32_t ltx_cuda_kernel_count(void)
{
    return static_cast<int32_t>(sizeof(kKernelNames) / sizeof(kKernelNames[0]));
}

const char* ltx_cuda_kernel_name(const int32_t index)
{
    return index >= 0 && index < ltx_cuda_kernel_count() ? kKernelNames[index] : nullptr;
}

int32_t ltx_cuda_fused_add_round_f32_host(
    const float* delta, const float* weight, float* output, const size_t element_count, const uint32_t seed)
{
    if (delta == nullptr || weight == nullptr || output == nullptr || element_count == 0) return Invalid();
    DeviceBuffer<float> device_delta, device_weight, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_delta.Upload(delta, element_count));
    LTX_RETURN_IF_CUDA_ERROR(device_weight.Upload(weight, element_count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(element_count));
    fused_add_round_kernel<<<static_cast<unsigned>((element_count + kThreads - 1) / kThreads), kThreads>>>(
        device_delta.Get(), device_weight.Get(), device_output.Get(), element_count, seed);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, element_count));
    return 0;
}

int32_t ltx_cuda_na3d_f32_host(
    const float* query, const float* key, const float* value, float* output,
    const int32_t batch, const int32_t time, const int32_t height, const int32_t width,
    const int32_t heads, const int32_t head_dim,
    const int32_t kernel_time, const int32_t kernel_height, const int32_t kernel_width,
    const int32_t causal_time, const int32_t causal_height, const int32_t causal_width,
    const float scale)
{
    if (query == nullptr || key == nullptr || value == nullptr || output == nullptr || batch <= 0 || time <= 0 ||
        height <= 0 || width <= 0 || heads <= 0 || head_dim <= 0 || kernel_time <= 0 || kernel_height <= 0 ||
        kernel_width <= 0) return Invalid();
    const size_t count = static_cast<size_t>(batch) * time * height * width * heads * head_dim;
    DeviceBuffer<float> device_query, device_key, device_value, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_query.Upload(query, count));
    LTX_RETURN_IF_CUDA_ERROR(device_key.Upload(key, count));
    LTX_RETURN_IF_CUDA_ERROR(device_value.Upload(value, count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    na3d_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(
        device_query.Get(), device_key.Get(), device_value.Get(), device_output.Get(),
        batch, time, height, width, heads, head_dim, kernel_time, kernel_height, kernel_width,
        causal_time != 0, causal_height != 0, causal_width != 0, scale);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_swiglu_up_mul_f32_host(
    const float* input, const float* up_weight, const float* gate, float* output,
    const int32_t rows, const int32_t input_dim, const int32_t output_dim)
{
    if (input == nullptr || up_weight == nullptr || gate == nullptr || output == nullptr ||
        rows <= 0 || input_dim <= 0 || output_dim <= 0) return Invalid();
    DeviceBuffer<float> device_input, device_weight, device_gate, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, static_cast<size_t>(rows) * input_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_weight.Upload(up_weight, static_cast<size_t>(output_dim) * input_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_gate.Upload(gate, static_cast<size_t>(rows) * output_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(static_cast<size_t>(rows) * output_dim));
    const int count = rows * output_dim;
    fused_up_mul_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
        device_input.Get(), device_weight.Get(), device_gate.Get(), device_output.Get(), rows, input_dim, output_dim);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_swiglu_gate_up_f32_host(
    const float* input, const float* gate_weight, const float* up_weight, float* output,
    const int32_t rows, const int32_t input_dim, const int32_t output_dim)
{
    if (input == nullptr || gate_weight == nullptr || up_weight == nullptr || output == nullptr ||
        rows <= 0 || input_dim <= 0 || output_dim <= 0) return Invalid();
    DeviceBuffer<float> device_input, device_gate, device_up, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, static_cast<size_t>(rows) * input_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_gate.Upload(gate_weight, static_cast<size_t>(output_dim) * input_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_up.Upload(up_weight, static_cast<size_t>(output_dim) * input_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(static_cast<size_t>(rows) * output_dim));
    const int count = rows * output_dim;
    fused_gate_up_swiglu_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
        device_input.Get(), device_gate.Get(), device_up.Get(), device_output.Get(), rows, input_dim, output_dim);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_fp8_gemm_f32_host(
    const float* a, const float* b, const float* bias, float* output,
    const int32_t rows, const int32_t output_dim, const int32_t inner_dim, const int32_t variant)
{
    return RunFp8Gemm(a, b, bias, output, rows, output_dim, inner_dim, variant);
}

int32_t ltx_cuda_nvfp4_quantize_f32_host(
    const float* input, uint8_t* packed, uint8_t* block_scales,
    const int32_t rows, const int32_t columns, const float per_tensor_scale,
    const int32_t hi_first, const int32_t variant)
{
    if (input == nullptr || packed == nullptr || block_scales == nullptr || rows <= 0 || columns <= 0 ||
        columns % kNvFp4Block != 0 || per_tensor_scale < 0.0F || (variant != 1 && variant != 2)) return Invalid();
    const int padded_rows = ((rows + 127) / 128) * 128;
    const int padded_columns = ((columns / kNvFp4Block + 3) / 4) * 4;
    const size_t input_count = static_cast<size_t>(rows) * columns;
    const size_t packed_count = input_count / 2;
    const size_t scale_count = static_cast<size_t>(padded_rows) * padded_columns;
    DeviceBuffer<float> device_input;
    DeviceBuffer<uint8_t> device_packed, device_scales;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, input_count));
    LTX_RETURN_IF_CUDA_ERROR(device_packed.Allocate(packed_count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Allocate(scale_count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Clear(scale_count));
    const int block_count = rows * (columns / kNvFp4Block);
    const int grid = (block_count + kThreads - 1) / kThreads;
    if (variant == 1 && hi_first != 0)
        quantize_kernel<true><<<grid, kThreads>>>(device_input.Get(), device_packed.Get(), device_scales.Get(), rows, columns, per_tensor_scale);
    else if (variant == 1)
        quantize_kernel<false><<<grid, kThreads>>>(device_input.Get(), device_packed.Get(), device_scales.Get(), rows, columns, per_tensor_scale);
    else if (hi_first != 0)
        quantize_tiled_kernel<true><<<grid, kThreads>>>(device_input.Get(), device_packed.Get(), device_scales.Get(), rows, columns, per_tensor_scale);
    else
        quantize_tiled_kernel<false><<<grid, kThreads>>>(device_input.Get(), device_packed.Get(), device_scales.Get(), rows, columns, per_tensor_scale);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_packed.Download(packed, packed_count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Download(block_scales, scale_count));
    return 0;
}

int32_t ltx_cuda_nvfp4_dequantize_f32_host(
    const uint8_t* packed, const uint8_t* block_scales, float* output,
    const int32_t rows, const int32_t columns, const float per_tensor_scale, const int32_t hi_first)
{
    if (packed == nullptr || block_scales == nullptr || output == nullptr || rows <= 0 || columns <= 0 ||
        columns % kNvFp4Block != 0 || per_tensor_scale < 0.0F) return Invalid();
    const int padded_rows = ((rows + 127) / 128) * 128;
    const int padded_columns = ((columns / kNvFp4Block + 3) / 4) * 4;
    const size_t output_count = static_cast<size_t>(rows) * columns;
    DeviceBuffer<uint8_t> device_packed, device_scales;
    DeviceBuffer<float> device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_packed.Upload(packed, output_count / 2));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Upload(block_scales, static_cast<size_t>(padded_rows) * padded_columns));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(output_count));
    const int grid = static_cast<int>((output_count + kThreads - 1) / kThreads);
    if (hi_first != 0)
        dequantize_kernel<true><<<grid, kThreads>>>(device_packed.Get(), device_scales.Get(), device_output.Get(), rows, columns, per_tensor_scale);
    else
        dequantize_kernel<false><<<grid, kThreads>>>(device_packed.Get(), device_scales.Get(), device_output.Get(), rows, columns, per_tensor_scale);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, output_count));
    return 0;
}

int32_t ltx_cuda_nvfp4_scaled_mm_f32_host(
    const uint8_t* a, const uint8_t* b, const uint8_t* scale_a, const uint8_t* scale_b,
    const float per_tensor_scale_a, const float per_tensor_scale_b,
    const float* bias, float* output,
    const int32_t rows, const int32_t output_dim, const int32_t inner_dim,
    const int32_t hi_first, const int32_t has_bias)
{
    if (a == nullptr || b == nullptr || scale_a == nullptr || scale_b == nullptr || output == nullptr ||
        rows <= 0 || output_dim <= 0 || inner_dim <= 0 || inner_dim % kNvFp4Block != 0 ||
        per_tensor_scale_a < 0.0F || per_tensor_scale_b < 0.0F || (has_bias != 0 && bias == nullptr)) return Invalid();
    const int padded_columns = ((inner_dim / kNvFp4Block + 3) / 4) * 4;
    const int padded_rows_a = ((rows + 127) / 128) * 128;
    const int padded_rows_b = ((output_dim + 127) / 128) * 128;
    DeviceBuffer<uint8_t> device_a, device_b, device_scale_a, device_scale_b;
    DeviceBuffer<float> device_bias, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_a.Upload(a, static_cast<size_t>(rows) * inner_dim / 2));
    LTX_RETURN_IF_CUDA_ERROR(device_b.Upload(b, static_cast<size_t>(output_dim) * inner_dim / 2));
    LTX_RETURN_IF_CUDA_ERROR(device_scale_a.Upload(scale_a, static_cast<size_t>(padded_rows_a) * padded_columns));
    LTX_RETURN_IF_CUDA_ERROR(device_scale_b.Upload(scale_b, static_cast<size_t>(padded_rows_b) * padded_columns));
    if (has_bias != 0) LTX_RETURN_IF_CUDA_ERROR(device_bias.Upload(bias, output_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(static_cast<size_t>(rows) * output_dim));
    const int count = rows * output_dim;
    scaled_mm_nvfp4_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
        device_a.Get(), device_b.Get(), device_scale_a.Get(), device_scale_b.Get(),
        per_tensor_scale_a, per_tensor_scale_b, device_bias.Get(), device_output.Get(),
        rows, output_dim, inner_dim, hi_first != 0, has_bias != 0);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_mul_scalars_f32_host(const float a, const float b, float* output)
{
    if (output == nullptr) return Invalid();
    DeviceBuffer<float> device_a, device_b, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_a.Upload(&a, 1));
    LTX_RETURN_IF_CUDA_ERROR(device_b.Upload(&b, 1));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(1));
    mul_scalars_kernel<<<1, 1>>>(device_a.Get(), device_b.Get(), device_output.Get());
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, 1));
    return 0;
}

int32_t ltx_cuda_amax_scale_f32_host(
    const float* input, const size_t element_count, const float divisor, float* output)
{
    if (input == nullptr || output == nullptr || element_count == 0 || divisor <= 0.0F) return Invalid();
    DeviceBuffer<float> device_input, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, element_count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(1));
    amax_scale_kernel<<<1, 1>>>(device_input.Get(), element_count, divisor, device_output.Get());
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, 1));
    return 0;
}

int32_t ltx_cuda_gemm_alpha_mode(void) { return 1; }
int32_t ltx_cuda_set_gemm_alpha_on_device(const int32_t enabled) { return enabled == 0 || enabled == 1 ? 0 : Invalid(); }
int32_t ltx_cuda_set_gemm_autotune(const int32_t enabled) { return enabled == 0 || enabled == 1 ? 0 : Invalid(); }
int32_t ltx_cuda_probe_gemm_support(const int32_t rows, const int32_t output_dim, const int32_t inner_dim)
{
    return rows > 0 && output_dim > 0 && inner_dim > 0 && inner_dim % 16 == 0 ? 1 : 0;
}

int32_t ltx_cuda_fp6_pack_host(
    const uint8_t* input, uint8_t* output, const int32_t rows, const int32_t columns)
{
    if (input == nullptr || output == nullptr || rows <= 0 || columns <= 0 || columns % 4 != 0) return Invalid();
    const size_t input_count = static_cast<size_t>(rows) * columns;
    const size_t output_count = input_count * 3 / 4;
    DeviceBuffer<uint8_t> device_input, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, input_count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(output_count));
    const int groups = rows * columns / 4;
    fp6_pack_kernel<<<(groups + kThreads - 1) / kThreads, kThreads>>>(device_input.Get(), device_output.Get(), rows, columns);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, output_count));
    return 0;
}

int32_t ltx_cuda_fp6_unpack_host(
    const uint8_t* input, uint8_t* output, const int32_t rows, const int32_t columns)
{
    if (input == nullptr || output == nullptr || rows <= 0 || columns <= 0 || columns % 4 != 0) return Invalid();
    const size_t output_count = static_cast<size_t>(rows) * columns;
    DeviceBuffer<uint8_t> device_input, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, output_count * 3 / 4));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(output_count));
    const int groups = rows * columns / 4;
    fp6_unpack_kernel<<<(groups + kThreads - 1) / kThreads, kThreads>>>(device_input.Get(), device_output.Get(), rows, columns);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, output_count));
    return 0;
}

int32_t ltx_cuda_rms_norm_rope_f32_host(
    const float* input, const float* weights, const float* cosine, const float* sine, float* output,
    const int32_t rows, const int32_t hidden_dim, const int32_t has_weights, const float epsilon)
{
    if (input == nullptr || cosine == nullptr || sine == nullptr || output == nullptr || rows <= 0 ||
        hidden_dim <= 0 || hidden_dim % 2 != 0 || epsilon < 0.0F || (has_weights != 0 && weights == nullptr)) return Invalid();
    const size_t count = static_cast<size_t>(rows) * hidden_dim;
    DeviceBuffer<float> device_input, device_weights, device_cosine, device_sine, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    if (has_weights != 0) LTX_RETURN_IF_CUDA_ERROR(device_weights.Upload(weights, hidden_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_cosine.Upload(cosine, count));
    LTX_RETURN_IF_CUDA_ERROR(device_sine.Upload(sine, count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    norm_rope_cvt_kernel<<<rows, 1>>>(device_input.Get(), device_weights.Get(), device_cosine.Get(),
        device_sine.Get(), device_output.Get(), rows, hidden_dim, has_weights != 0, epsilon);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_rms_norm_split_rope_f32_host(
    const float* input, const float* weights, const float* cosine, const float* sine, float* output,
    const int32_t rows, const int32_t heads, const int32_t head_dim, const float epsilon)
{
    if (input == nullptr || weights == nullptr || cosine == nullptr || sine == nullptr || output == nullptr ||
        rows <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 || epsilon < 0.0F) return Invalid();
    const size_t count = static_cast<size_t>(rows) * heads * head_dim;
    const size_t frequency_count = static_cast<size_t>(rows) * heads * head_dim / 2;
    DeviceBuffer<float> device_input, device_weights, device_cosine, device_sine, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    LTX_RETURN_IF_CUDA_ERROR(device_weights.Upload(weights, static_cast<size_t>(heads) * head_dim));
    LTX_RETURN_IF_CUDA_ERROR(device_cosine.Upload(cosine, frequency_count));
    LTX_RETURN_IF_CUDA_ERROR(device_sine.Upload(sine, frequency_count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    rms_norm_split_rope_kernel<<<rows, 1>>>(device_input.Get(), device_weights.Get(), device_cosine.Get(),
        device_sine.Get(), device_output.Get(), rows, heads, head_dim, epsilon);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_rowwise_int8_quantize_f32_host(
    const float* input, int8_t* output, float* scales, const int32_t rows, const int32_t columns)
{
    if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 || columns <= 0) return Invalid();
    const size_t count = static_cast<size_t>(rows) * columns;
    DeviceBuffer<float> device_input, device_scales;
    DeviceBuffer<int8_t> device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Allocate(rows));
    rowwise_int8_kernel<<<rows, 1>>>(device_input.Get(), device_output.Get(), device_scales.Get(), rows, columns);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Download(scales, rows));
    return 0;
}

int32_t ltx_cuda_blockwise_fp8_quantize_f32_host(
    const float* input, uint8_t* output, float* scales,
    const int32_t rows, const int32_t columns, const int32_t block_size, const int32_t use_gelu)
{
    if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 || columns <= 0 ||
        block_size <= 0 || columns % block_size != 0) return Invalid();
    const size_t count = static_cast<size_t>(rows) * columns;
    const int scale_count = rows * columns / block_size;
    DeviceBuffer<float> device_input, device_scales;
    DeviceBuffer<uint8_t> device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Allocate(scale_count));
    const int blocks = rows * columns / block_size;
    block_quant_kernel<<<(blocks + kThreads - 1) / kThreads, kThreads>>>(
        device_input.Get(), device_output.Get(), device_scales.Get(), rows, columns, block_size, use_gelu != 0);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Download(scales, scale_count));
    return 0;
}

int32_t ltx_cuda_blockwise_fp8_norm_quantize_f32_host(
    const float* input, const float* norm_scale, const float* norm_shift,
    uint8_t* output, float* scales,
    const int32_t rows, const int32_t columns, const int32_t block_size, const float epsilon)
{
    if (input == nullptr || norm_scale == nullptr || norm_shift == nullptr || output == nullptr || scales == nullptr ||
        rows <= 0 || columns <= 0 || block_size <= 0 || columns % block_size != 0 || epsilon < 0.0F) return Invalid();
    const size_t count = static_cast<size_t>(rows) * columns;
    const int scale_count = rows * columns / block_size;
    DeviceBuffer<float> device_input, device_norm_scale, device_norm_shift, device_scales;
    DeviceBuffer<uint8_t> device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    LTX_RETURN_IF_CUDA_ERROR(device_norm_scale.Upload(norm_scale, columns));
    LTX_RETURN_IF_CUDA_ERROR(device_norm_shift.Upload(norm_shift, columns));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Allocate(scale_count));
    block_quant_norm_kernel<<<rows, 1>>>(device_input.Get(), device_norm_scale.Get(), device_norm_shift.Get(),
        device_output.Get(), device_scales.Get(), rows, columns, block_size, epsilon);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Download(scales, scale_count));
    return 0;
}

int32_t ltx_cuda_quant_rms_fma_f32_host(
    const float* x, const float* y, const float* z, float* residual, uint8_t* output, float* scales,
    const int32_t rows, const int32_t columns, const int32_t block_size)
{
    if (x == nullptr || y == nullptr || z == nullptr || residual == nullptr || output == nullptr || scales == nullptr ||
        rows <= 0 || columns <= 0 || block_size <= 0 || columns % block_size != 0) return Invalid();
    const size_t count = static_cast<size_t>(rows) * columns;
    const int scale_count = rows * columns / block_size;
    DeviceBuffer<float> device_x, device_y, device_z, device_residual, device_scales;
    DeviceBuffer<uint8_t> device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_x.Upload(x, count));
    LTX_RETURN_IF_CUDA_ERROR(device_y.Upload(y, count));
    LTX_RETURN_IF_CUDA_ERROR(device_z.Upload(z, count));
    LTX_RETURN_IF_CUDA_ERROR(device_residual.Allocate(count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Allocate(scale_count));
    quant_rms_sum_mult_kernel<<<rows, 1>>>(device_x.Get(), device_y.Get(), device_z.Get(),
        device_residual.Get(), device_output.Get(), device_scales.Get(), rows, columns, block_size);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_residual.Download(residual, count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Download(scales, scale_count));
    return 0;
}

int32_t ltx_cuda_gated_attention_f32_host(
    const float* input, const float* gate_logits, float* output,
    const int32_t rows, const int32_t heads, const int32_t head_dim)
{
    if (input == nullptr || gate_logits == nullptr || output == nullptr || rows <= 0 || heads <= 0 || head_dim <= 0) return Invalid();
    const size_t count = static_cast<size_t>(rows) * heads * head_dim;
    DeviceBuffer<float> device_input, device_gate, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    LTX_RETURN_IF_CUDA_ERROR(device_gate.Upload(gate_logits, static_cast<size_t>(rows) * heads));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    gated_attention_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(
        device_input.Get(), device_gate.Get(), device_output.Get(), rows, heads, head_dim);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}

int32_t ltx_cuda_blockwise_fp8_dequantize_f32_host(
    const uint8_t* input, const float* scales, float* output,
    const int32_t rows, const int32_t columns, const int32_t block_size)
{
    if (input == nullptr || scales == nullptr || output == nullptr || rows <= 0 || columns <= 0 ||
        block_size <= 0 || columns % block_size != 0) return Invalid();
    const size_t count = static_cast<size_t>(rows) * columns;
    const int scale_count = rows * columns / block_size;
    DeviceBuffer<uint8_t> device_input;
    DeviceBuffer<float> device_scales, device_output;
    LTX_RETURN_IF_CUDA_ERROR(device_input.Upload(input, count));
    LTX_RETURN_IF_CUDA_ERROR(device_scales.Upload(scales, scale_count));
    LTX_RETURN_IF_CUDA_ERROR(device_output.Allocate(count));
    blockwise_dequantize_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(
        device_input.Get(), device_scales.Get(), device_output.Get(), rows, columns, block_size);
    const int32_t status = FinishKernel();
    if (status != 0) return status;
    LTX_RETURN_IF_CUDA_ERROR(device_output.Download(output, count));
    return 0;
}
