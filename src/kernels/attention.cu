#include <cuda_runtime.h>
#include "model_def.h"
#include "cuda_functions.h"
#include "buffer.h"

extern "C" void attention_forward(Tensor *x, Model *model, Tensor *freqs,
    int i, int head_dim, int q_heads, int kv_heads, float eps,
    Tensor *q, Tensor *k, Tensor *v, Tensor *q_rope, Tensor *k_rope, Tensor * v_rope, Tensor *scores, Tensor *out) {
    
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
    int d = x->shape[2]; // q_heads * head_dim

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

    permute_physical_forward(v, v_rope);

    //Permute_and_rope_inplace [b, s, heads, d] -> [b, heads, s, d]
    rope_forward(q, q_rope, freqs);
    rope_forward(k, k_rope, freqs);

    atten_dot_product_gqa(q_rope, k_rope, scores, q_heads, kv_heads, true);
    softmax_forward(scores);

    mm_context_gqa(scores, v_rope, out, q_heads, kv_heads);
    linear_layer(out, o_weight, x);
}