#pragma once

#include <cstddef>
#include "model_def.h"

struct Workspace {
    char* base_ptr = nullptr;
    size_t total_capacity = 0;
    size_t current_offset = 0;

    void allocate(size_t max_bytes);
    void* get_offset(size_t requested_bytes);
    void reset();
    void destroy();
};

struct KV_Cache {
    char* k_base_ptr = nullptr;
    char* v_base_ptr = nullptr;

    size_t total_capacity = 0;
    
    size_t stride_layer = 0;
    size_t stride_batch = 0;
    size_t stride_head  = 0;
    size_t stride_seq   = 0;

    void allocate(int num_layers, int b, int num_kv_head, size_t max_s, size_t d);
    void free_memory();
    
    float* get_k_ptr(int layer, int batch, int step);
    float* get_v_ptr(int layer, int batch, int step);
};

void update_kv_cache(Tensor*k, Tensor*v, float *k_cache, float *v_cache, int max_s, int step);