#include <cuda_runtime.h>
#include "model_def.h"
#include "cuda_functions.h"
#include "buffer.h"

extern "C" void attention_forward(Tensor *x, Model *model, Tensor *freqs,
    int i, int head_dim, int q_heads, int kv_heads, float eps,
    Tensor *q, Tensor *k, Tensor *v, Tensor *q_rope, Tensor *k_rope, Tensor * v_rope, Tensor *scores, Tensor *out,
     KV_Cache *cache, int step, int max_s, bool is_causal) {
    
    Tensor *q_weight = model->layers[i].atten.q_proj.weight;
    Tensor *k_weight = model->layers[i].atten.k_proj.weight;
    Tensor *v_weight = model->layers[i].atten.v_proj.weight;
    Tensor *o_weight = model->layers[i].atten.o_proj.weight;

    Tensor *q_norm_weight = model->layers[i].atten.q_norm.weight;
    Tensor *q_norm_bias = model->layers[i].atten.q_norm.bias;
    Tensor *k_norm_weight = model->layers[i].atten.k_norm.weight;
    Tensor *k_norm_bias = model->layers[i].atten.k_norm.bias;

    int b = x->shape[0];
    int s = x->shape[1];

    // Reset the tensor views back to 3D (since previous layer made them 4D)
    int q_shape[3] = {b, s, q_heads * head_dim};
    int kv_shape[3] = {b, s, kv_heads * head_dim};
    tensor_view(q, 3, q_shape);
    tensor_view(k, 3, kv_shape);
    tensor_view(v, 3, kv_shape);

    linear_layer(x, q_weight, q);
    linear_layer(x, k_weight, k);
    linear_layer(x, v_weight, v);

    norm_forward(q, q, q_norm_weight, q_norm_bias, b * s, q_heads, head_dim, eps, true);
    norm_forward(k, k, k_norm_weight, k_norm_bias, b * s, kv_heads, head_dim, eps, true);

    int q_shape_view[4] = {b, s, q_heads, head_dim};
    int kv_shape_view[4] = {b, s, kv_heads, head_dim};

    tensor_view(q, 4, q_shape_view);
    tensor_view(k, 4, kv_shape_view);
    tensor_view(v, 4, kv_shape_view);

    //Permute_and_rope_inplace [b, s, heads, head_dim] -> [b, heads, s, head_dim]
    rope_forward(q, q_rope, freqs, step);
    rope_forward(k, k_rope, freqs, step);
    permute_physical_forward(v, v_rope);

    //store KV cache
    float* k_cache_layer = cache->get_k_ptr(i, 0, 0);
    float* v_cache_layer = cache->get_v_ptr(i, 0, 0);

    Tensor k_cache_tensor = {0};
    k_cache_tensor.gpu_data = k_cache_layer;
    k_cache_tensor.ndim = 4;
    k_cache_tensor.shape[0] = b;
    k_cache_tensor.shape[1] = kv_heads;
    k_cache_tensor.shape[2] = step + s;
    k_cache_tensor.shape[3] = head_dim;                    
    
    k_cache_tensor.stride[0] = kv_heads * max_s * head_dim; 
    k_cache_tensor.stride[1] = max_s * head_dim;            
    k_cache_tensor.stride[2] = head_dim;                    
    k_cache_tensor.stride[3] = 1;

    Tensor v_cache_tensor = {0};
    v_cache_tensor.gpu_data = v_cache_layer;
    v_cache_tensor.ndim = 4;
    v_cache_tensor.shape[0] = b;
    v_cache_tensor.shape[1] = kv_heads;
    v_cache_tensor.shape[2] = step + s;
    v_cache_tensor.shape[3] = head_dim;                    
    
    v_cache_tensor.stride[0] = kv_heads * max_s * head_dim; 
    v_cache_tensor.stride[1] = max_s * head_dim;            
    v_cache_tensor.stride[2] = head_dim;                    
    v_cache_tensor.stride[3] = 1;
    
    update_kv_cache(k_rope, v_rope, k_cache_layer, v_cache_layer, max_s, step);

    atten_dot_product_gqa(q_rope, &k_cache_tensor, scores, q_heads, kv_heads, is_causal);
    softmax_forward(scores);

    mm_context_gqa(scores, &v_cache_tensor, out, q_heads, kv_heads);
    linear_layer(out, o_weight, x);
}