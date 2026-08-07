#pragma once

#include "model_def.h"
#include "weight_loading.h"
#include "buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_empty_like(Tensor* src, Workspace* arena);

Tensor* tensor_empty(const int *shape, const int ndim, Workspace* arena);

void embedding_forward(Tensor *embed_weights, Tensor *x, Tensor *out);

void norm_forward(Tensor *x, Tensor *out, Tensor *norm_weight, Tensor *norm_bias, 
                  int b, int s, int d, float eps, bool is_rms_norm);

void mm_t_forward(Tensor *A, Tensor *B, Tensor *C);

void rope_forward(Tensor *x, Tensor *out, Tensor *freqs, int step);

void transpose(Tensor *input, const int *index);

void tensor_view(Tensor *t, int ndim, const int *shape);

void permute(Tensor *A, const int *order);

void softmax_forward(Tensor *input);

void attention_forward(Tensor *x, Model *model, Tensor *freqs,
    int i, int head_dim, int q_heads, int kv_heads, float eps,
    Tensor *q, Tensor *k, Tensor *v, Tensor *q_rope, Tensor *k_rope, Tensor * v_rope, Tensor *scores, 
    Tensor *out, KV_Cache* cache, int step, int max_s, bool is_causal);

void elem_forward_add(Tensor *a, Tensor *b);

void elem_forward_mul_act(Tensor *a, Tensor *b, ActivationType act);

void generate_rope_freqs(Tensor * freqs, int s, int head_dim, float rope_theta);

void linear_layer(Tensor *x, Tensor *layer_weight, Tensor *C);

void atten_dot_product_gqa(Tensor *Q, Tensor *K, Tensor *Scores, int q_heads, int kv_heads, bool is_causal);

void mm_context_gqa(Tensor *Scores, Tensor *V, Tensor *out, int q_heads, int kv_heads);

void permute_physical_forward(Tensor *in, Tensor *out);

#ifdef __cplusplus
}
#endif

