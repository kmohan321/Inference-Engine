#include "buffer.h"
#include "model_def.h"

#include <cuda_runtime.h>
#include <iostream>

void Workspace::allocate(size_t max_bytes)
{
    total_capacity = max_bytes;
    current_offset = 0;

    cudaError_t err = cudaMalloc((void**)&base_ptr, total_capacity);

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate workspace!" << std::endl;
    }
}

void* Workspace::get_offset(size_t requested_bytes)
{
    size_t aligned_bytes = (requested_bytes + 255) & ~255;

    if (current_offset + aligned_bytes > total_capacity)
    {
        std::cerr << "OUT OF WORKSPACE MEMORY!" << std::endl;
        return nullptr;
    }

    void* ptr = base_ptr + current_offset;
    current_offset += aligned_bytes;

    return ptr;
}

void Workspace::reset()
{
    current_offset = 0;
}

void Workspace::destroy()
{
    if (base_ptr)
    {
        cudaFree(base_ptr);
        base_ptr = nullptr;
    }
}

void KV_Cache::allocate(int num_layers, int b, int num_kv_head, size_t max_s, size_t d) {

    stride_seq   = d;
    stride_head  = max_s * stride_seq;
    stride_batch = num_kv_head * stride_head;
    stride_layer = b * stride_batch;

    total_capacity = num_layers * stride_layer * sizeof(float);

    cudaError_t err1 = cudaMalloc((void**)&k_base_ptr, total_capacity);
    cudaError_t err2 = cudaMalloc((void**)&v_base_ptr, total_capacity);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "Failed to allocate KV Cache!" << std::endl;
    }
}

void KV_Cache::free_memory() {
    
    if (k_base_ptr) cudaFree(k_base_ptr);
    if (v_base_ptr) cudaFree(v_base_ptr);
    k_base_ptr = nullptr;
    v_base_ptr = nullptr;
}

float* KV_Cache::get_k_ptr(int layer, int batch, int step) {

    size_t offset_elements = (layer * stride_layer) + (batch * stride_batch) + (step * stride_seq);
    return (float*)k_base_ptr + offset_elements;
}

float* KV_Cache::get_v_ptr(int layer, int batch, int step) {

    size_t offset_elements = (layer * stride_layer) + (batch * stride_batch) + (step * stride_seq);
    return (float*)v_base_ptr + offset_elements;
}


__global__ void update_kv_cache_kernel(
    const float* __restrict__ k_in,      // Shape: [b, nh, s, d]
    const float* __restrict__ v_in,      // Shape: [b, nh, s, d]
    float* __restrict__ k_cache,         // Shape: [b, nh, max_s, d]
    float* __restrict__ v_cache,         // Shape: [b, nh, max_s, d]
    int b, int nh, int s, int d, 
    int max_s, int step
) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * nh * s * d) return;

    int b_idx  = idx / (nh * s * d);
    int rem    = idx % (nh * s * d);
    int nh_idx = rem / (s * d);
    rem        = rem % (s * d);
    int s_idx  = rem / d;
    int d_idx  = rem % d;

    int cache_idx = b_idx * (nh * max_s * d) 
                  + nh_idx * (max_s * d) 
                  + (step + s_idx) * d 
                  + d_idx;

    k_cache[cache_idx] = k_in[idx];
    v_cache[cache_idx] = v_in[idx];

}

void update_kv_cache(Tensor*k, Tensor*v, float *k_cache, float *v_cache, int max_s, int step){

    int b = k->shape[0];
    int nh = k->shape[1];
    int s = k->shape[2];
    int d = k->shape[3];

    int total_elem = b * nh * s * d; 

    dim3 blocks(256);
    dim3 grid((total_elem + 255)/256);

    update_kv_cache_kernel<<<grid, blocks>>>(
        (float*)k->gpu_data, (float*)v->gpu_data,
        k_cache, v_cache, b, nh, s, d,
        max_s, step
    );
}