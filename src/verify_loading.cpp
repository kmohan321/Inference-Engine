//file for testing purposes

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "model_def.h"
#include "weight_loading.h"


void print_tensor(Tensor *t)
{
    if (t == NULL)
    {
        printf("Tensor : NULL\n");
        return;
    }

    printf("----------------------------------\n");
    printf("Name   : %s\n", t->name);

    printf("DType  : ");

    switch(t->dtype)
    {
        case FLOAT32: printf("FLOAT32"); break;
        case FLOAT16: printf("FLOAT16"); break;
        case BFLOAT16: printf("BFLOAT16"); break;
    }

    printf("\n");

    printf("Shape  : ");

    for(int i=0;i<t->ndim;i++)
        printf("%llu ", (unsigned long long)t->shape[i]);

    printf("\n");

    printf("Stride  : ");

    for(int i=0;i<t->ndim;i++)
        printf("%llu ", (unsigned long long)t->stride[i]);

    printf("\n");

    printf("Bytes  : %llu\n",
           (unsigned long long)t->nbytes);

    printf("CPU Ptr: %p\n", t->cpu_data);

    printf("GPU Ptr: %p\n", t->gpu_data);
    printf("----------------------------------\n");
}

void verify_model(Model *model, ModelConfig *cfg)
{
    printf("\n===== MODEL CHECK =====\n");

    print_tensor(model->embed_weight);

    print_tensor(model->lm_head_weight);

    for(int i=0;i<cfg->num_layers;i++)
    {
        printf("\nLayer %d\n", i);

        print_tensor(model->layers[i].atten.q_proj.weight);
        print_tensor(model->layers[i].atten.k_proj.weight);
        print_tensor(model->layers[i].atten.v_proj.weight);
        print_tensor(model->layers[i].atten.o_proj.weight);

        print_tensor(model->layers[i].mlp.gate_proj.weight);
        print_tensor(model->layers[i].mlp.up_proj.weight);
        print_tensor(model->layers[i].mlp.down_proj.weight);
    }
}

// int main(){
//     Model model;
//     TensorDB db;
//     ModelConfig cfg;

//     load_safetensors("model_data/model.safetensors", &db);
//     load_config("model_data/config.json", &cfg);
//     build_model(&model, &cfg, &db);
//     upload_to_gpu(&db);
//     verify_model(&model, &cfg);
//     return 0;
// }

int main(){
    printf("[DEBUG] Program started.\n"); 
    fflush(stdout);

    // 1. Allocate on the Heap instead of the Stack!
    Model *model = (Model*)malloc(sizeof(Model));
    TensorDB *db = (TensorDB*)malloc(sizeof(TensorDB));
    ModelConfig *cfg = (ModelConfig*)malloc(sizeof(ModelConfig));

    if (model == NULL || db == NULL || cfg == NULL) {
        printf("[FATAL ERROR] Malloc failed. Out of memory!\n");
        return -1;
    }

    printf("[DEBUG] Attempting to load safetensors...\n"); 
    fflush(stdout);
    // Notice we pass 'db' directly now, not '&db', because they are already pointers
    load_safetensors("model_data/model.safetensors", db);

    printf("[DEBUG] Attempting to load config...\n"); 
    fflush(stdout);
    load_config("model_data/config.json", cfg);

    printf("[DEBUG] Building model structs...\n"); 
    fflush(stdout);
    build_model(model, cfg, db);

    printf("[DEBUG] Uploading weights to GPU...\n"); 
    fflush(stdout);
    upload_to_gpu(db);

    printf("[DEBUG] Running verify_model...\n"); 
    fflush(stdout);
    verify_model(model, cfg);
    
    printf("[DEBUG] Done!\n");

    // Clean up
    free(model);
    free(db);
    free(cfg);
    return 0;
}