#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FLOAT32,
    FLOAT16,
    BFLOAT16
} DType;

typedef enum
{
    SILU,
    GELU,
    UNKNOWN_ACT
} ActivationType;

typedef struct {

    char model_type[64];

    ActivationType act;

    int hidden_size;
    int intermediate_size;

    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;

    int vocab_size;
    int max_position_embeddings;

    float rms_norm_eps;
    float rope_theta;

    DType dtype;
    int tie_word_embeddings;

} ModelConfig;

typedef struct {
    char     name[128];    // Tensor name in safetensors
    DType    dtype;        // Data type
    int      ndim;         // Dimensions
    uint64_t shape[8];     // Tensor shape
    uint64_t stride[8];    // Tensor stride
    uint64_t offset;       // File offset
    uint64_t nbytes;       // Total size in bytes
    void    *gpu_data;     // VRAM pointer
    void    *cpu_data;     // Mmapped Host pointer
} Tensor;

typedef struct {
    Tensor *weight;
    Tensor *bias;
} Linear;

typedef struct {
    Tensor *weight;
    Tensor *bias;
} RMSNorm;

typedef struct {
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;
    
    RMSNorm q_norm;
    RMSNorm k_norm; 
} Attention;

typedef struct {
    Linear up_proj;
    Linear down_proj;
    Linear gate_proj;
} MLP;

typedef struct {
    Attention atten;
    MLP mlp;
    RMSNorm input_layernorm;
    RMSNorm post_attention_layernorm;
} Layer;

typedef struct {
    Tensor  *embed_weight;
    RMSNorm  norm;
    Layer   *layers;
    Tensor  *lm_head_weight;
} Model;

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

class BaseModel {
public:
    virtual ~BaseModel() = default;

    virtual void build_model(ModelConfig* cfg, void* db) = 0;
    virtual void forward(void* arena, void* kv_cache, ModelConfig* cfg, 
        Tensor* input, Tensor* output, int max_seq_len, int step) = 0;
};

#endif
