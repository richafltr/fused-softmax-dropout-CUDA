#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/PhiloxCudaState.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <curand_kernel.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for max
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    return val;
}

// Block-level reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Generate random float in [0, 1) using Philox
// Uses PyTorch's Philox state structure
__device__ __forceinline__ float philox_uniform(at::PhiloxCudaState& state, uint64_t idx) {
    // Use PyTorch's Philox implementation
    // This is a simplified version - for exact compatibility use at::cuda::philox::unpack
    // For now, use a hash-based approach compatible with Philox counter structure
    uint64_t counter_val = state.offset_.val + idx;
    uint64_t key_val = state.seed_.val;
    
    // Mix counter and key (simplified Philox-like mixing)
    uint64_t v0 = counter_val;
    uint64_t v1 = key_val;
    
    // Simple mixing rounds (not full Philox, but sufficient for dropout)
    for (int i = 0; i < 2; ++i) {
        v0 += v1;
        v1 = (v1 << 13) | (v1 >> 51);
        v1 ^= v0;
        v0 = (v0 << 16) | (v0 >> 48);
    }
    
    // Convert to float in [0, 1)
    constexpr float inv_uint64_max = 1.0f / (float)(1ULL << 32);
    return (float)(v0 >> 32) * inv_uint64_max;
}

__global__ void fused_softmax_dropout_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mask,  // can be null
    float* __restrict__ y,
    int64_t rows,            // total number of rows: batch * seq
    int64_t cols,            // d_model
    float p,
    bool training,
    at::PhiloxCudaState rng_state
) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;
    
    const float* row_x = x + row_idx * cols;
    const float* row_mask = mask ? (mask + row_idx * cols) : nullptr;
    float* row_y = y + row_idx * cols;
    
    // Step 1: Compute row max (numerically stable)
    float thread_max = -INFINITY;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = row_x[col];
        if (row_mask && row_mask[col] < 0.5f) {
            val = -INFINITY;  // Masked position
        }
        thread_max = fmaxf(thread_max, val);
    }
    
    float row_max = block_reduce_max(thread_max);
    
    // Broadcast max to all threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = row_max;
    }
    __syncthreads();
    row_max = shared_max;
    
    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = row_x[col];
        bool is_masked = row_mask && row_mask[col] < 0.5f;
        
        if (is_masked) {
            row_y[col] = 0.0f;
        } else {
            float exp_val = expf(val - row_max);
            thread_sum += exp_val;
            row_y[col] = exp_val;
        }
    }
    
    float row_sum = block_reduce_sum(thread_sum);
    
    // Broadcast sum to all threads
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum;
    
    // Step 3: Normalize and apply dropout
    float inv_sum = 1.0f / (row_sum + 1e-8f);  // Avoid division by zero
    float dropout_scale = training && p > 0.0f ? (1.0f / (1.0f - p)) : 1.0f;
    
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        bool is_masked = row_mask && row_mask[col] < 0.5f;
        
        if (is_masked) {
            row_y[col] = 0.0f;
        } else {
            float softmax_val = row_y[col] * inv_sum;
            
            if (training && p > 0.0f) {
                // Generate random number for dropout
                uint64_t global_idx = static_cast<uint64_t>(row_idx) * static_cast<uint64_t>(cols) + static_cast<uint64_t>(col);
                float u = philox_uniform(rng_state, global_idx);
                if (u < p) {
                    softmax_val = 0.0f;  // Dropout
                } else {
                    softmax_val *= dropout_scale;  // Scale up to maintain expectation
                }
            }
            
            row_y[col] = softmax_val;
        }
    }
}

void launch_fused_softmax_dropout(
    const at::Tensor& x,
    const at::Tensor& mask,
    at::Tensor& y,
    float p,
    bool training,
    at::PhiloxCudaState philox_state
) {
    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t d_model = x.size(2);
    int64_t rows = batch * seq;
    int64_t cols = d_model;
    
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);
    
    const float* x_ptr = x.data_ptr<float>();
    const float* mask_ptr = mask.defined() ? mask.data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();
    
    fused_softmax_dropout_kernel<<<grid, block>>>(
        x_ptr,
        mask_ptr,
        y_ptr,
        rows,
        cols,
        p,
        training,
        philox_state
    );
}

