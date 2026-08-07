#include <cstdio>
#include <string.h>
#include "weight_loading.h"
#include "model_def.h"
#include "cuda_functions.h"
#include "buffer.h"
// #include <cuda_runtime.h>

class QwenModel : public BaseModel{
  private:
    Model model; //holding weights for qwen model
    
  public:
    void build_model(ModelConfig* cfg, void* raw_db){ //loading weights from tensor db

      TensorDB *db = (TensorDB*)raw_db;
      model.layers = (Layer*)malloc(sizeof(Layer) * cfg->num_layers);

      char name[256]; //change name formatting according to the safetensor file
      for(int i = 0; i < cfg->num_layers; i++) {

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            model.layers[i].atten.q_proj.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            model.layers[i].atten.k_proj.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            model.layers[i].atten.v_proj.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            model.layers[i].atten.o_proj.weight = find_tensor(db, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            model.layers[i].atten.q_norm.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.bias", i);
            model.layers[i].atten.q_norm.bias = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            model.layers[i].atten.k_norm.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.bias", i);
            model.layers[i].atten.k_norm.bias = find_tensor(db, name);

            snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", i);
            model.layers[i].mlp.up_proj.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", i);
            model.layers[i].mlp.down_proj.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", i);
            model.layers[i].mlp.gate_proj.weight = find_tensor(db, name);

            snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
            model.layers[i].input_layernorm.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.bias", i);
            model.layers[i].input_layernorm.bias = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
            model.layers[i].post_attention_layernorm.weight = find_tensor(db, name);
            snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.bias", i);
            model.layers[i].post_attention_layernorm.bias = find_tensor(db, name);
      }

      model.embed_weight = find_tensor(db, "model.embed_tokens.weight");
      model.lm_head_weight = find_tensor(db, "lm_head.weight");
      model.norm.weight = find_tensor(db, "model.norm.weight");

    }

    void forward(void* raw_arena, ModelConfig* cfg, Tensor* token_ids, Tensor* logits){

      Workspace* arena = (Workspace*)raw_arena;

      int b = token_ids->shape[0];
      int s = token_ids->shape[1];
      int d = cfg->hidden_size;

      int x_shape[3] = {b, s, d};
      Tensor *x = tensor_empty(x_shape, 3, arena);
      embedding_forward(model.embed_weight, token_ids, x);

      int shape[2] = {s, cfg->head_dim/2};
      Tensor *freqs = tensor_empty(shape, 2, arena);
      generate_rope_freqs(freqs, s, cfg->head_dim, cfg->rope_theta);

      Tensor* norm_out = tensor_empty_like(x, arena);
      Tensor* out = tensor_empty_like(x, arena);
      int inter_shape[3] = {b, s, cfg->intermediate_size};
      Tensor* gate_out = tensor_empty(inter_shape, 3, arena);
      Tensor* up_out   = tensor_empty(inter_shape, 3, arena);

      int q_shape[3] = {b, s, cfg->num_heads * cfg->head_dim};
      int kv_shape[3] = {b, s, cfg->num_kv_heads * cfg->head_dim};
      Tensor *q = tensor_empty(q_shape, 3, arena);
      Tensor *k = tensor_empty(kv_shape, 3, arena);
      Tensor *v = tensor_empty(kv_shape, 3, arena);
      Tensor *temp_out = tensor_empty(q_shape, 3, arena);

      int q_rope_shape[4] = {b, cfg->num_heads, s, cfg->head_dim};
      int kv_rope_shape[4] = {b, cfg->num_kv_heads, s, cfg->head_dim};
      Tensor *q_rope = tensor_empty(q_rope_shape, 4, arena);
      Tensor *k_rope = tensor_empty(kv_rope_shape, 4, arena);
      Tensor *v_rope = tensor_empty(kv_rope_shape, 4, arena);

      int score_shape[4] = {b, cfg->num_heads, s, s};
      Tensor *scores = tensor_empty(score_shape, 4, arena);

      for(int i = 0; i < cfg->num_layers; i++){

        Tensor *atten_norm_weight = model.layers[i].input_layernorm.weight;
        Tensor *atten_norm_bias = model.layers[i].input_layernorm.bias;

        Tensor *mlp_norm_weight = model.layers[i].post_attention_layernorm.weight;
        Tensor *mlp_norm_bias = model.layers[i].post_attention_layernorm.bias;

        Tensor *up_proj_weight = model.layers[i].mlp.up_proj.weight;
        Tensor *down_proj_weight = model.layers[i].mlp.down_proj.weight;
        Tensor *gate_proj_weight = model.layers[i].mlp.gate_proj.weight;
          
        norm_forward(x, norm_out, atten_norm_weight, atten_norm_bias, b, s, d, cfg->rms_norm_eps, true);

        attention_forward(norm_out, &model, freqs, i, cfg->head_dim, cfg->num_heads, 
          cfg->num_kv_heads, cfg->rms_norm_eps, q, k, v, q_rope, k_rope, v_rope, scores, temp_out);
      
        elem_forward_add(x, norm_out);

        norm_forward(x, norm_out, mlp_norm_weight, mlp_norm_bias, b, s, d, cfg->rms_norm_eps, true);
        linear_layer(norm_out, gate_proj_weight, gate_out);
        linear_layer(norm_out, up_proj_weight, up_out);
        elem_forward_mul_act(gate_out, up_out, cfg->act);
        linear_layer(gate_out, down_proj_weight, out);
        elem_forward_add(x, out);

        }

      norm_forward(x, norm_out, model.norm.weight, NULL, b, s, d, cfg->rms_norm_eps, true);
      linear_layer(norm_out, model.lm_head_weight, logits);
    }

};


BaseModel* create_qwen3_model(ModelConfig* cfg) {
    if (strcmp(cfg->model_type, "Qwen3ForCausalLM") == 0) {
        return new QwenModel();
    }
    
    printf("ERROR: CFG NOT FOUND : %s\n", cfg->model_type);
    return nullptr;
}