#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "model_def.h"
#include "weight_loading.h"
#include "buffer.h"
#include "cuda_functions.h"
#include <chrono>
#include "tokenizer.h"
#include "models.h"


int main() {
    printf("Loading model...\n");
    TensorDB *db = (TensorDB*)malloc(sizeof(TensorDB));
    ModelConfig *cfg = (ModelConfig*)malloc(sizeof(ModelConfig));
    LLMTokenizer tokenizer("model_data/tokenizer.json");

    load_safetensors("model_data/model.safetensors", db);
    load_config("model_data/config.json", cfg);

    printf("cfg->act = %d\n", (int)cfg->act);
    
    BaseModel* llm = create_qwen3_model(cfg);
    if (!llm) {
        exit(1);
    }
    llm->build_model(cfg, db);

    upload_to_gpu(db);
    printf("Model loaded successfully!\n");

    Workspace arena_obj;
    Workspace* arena = &arena_obj;
    arena->allocate(1024ULL * 1024ULL * 1024ULL); 

    int b = 1;
    int s = 3;
    int input_tokens[3] = {1337, 42, 9000}; 

    // [FIX 1]: Zero-initialize the struct using = {0} to wipe garbage memory
    Tensor token_ids = {0};
    token_ids.ndim = 2;
    token_ids.shape[0] = b;
    token_ids.shape[1] = s;
    // [FIX 1]: Set Strides
    token_ids.stride[0] = s;
    token_ids.stride[1] = 1;
    token_ids.nbytes = b * s * sizeof(int);
    cudaMalloc(&token_ids.gpu_data, token_ids.nbytes);
    cudaMemcpy(token_ids.gpu_data, input_tokens, token_ids.nbytes, cudaMemcpyHostToDevice);

    // [FIX 1]: Zero-initialize the struct
    Tensor logits = {0};
    logits.ndim = 3;
    logits.shape[0] = b;
    logits.shape[1] = s;
    logits.shape[2] = cfg->vocab_size;
    // [FIX 1]: Set Strides properly for 3D tensor
    logits.stride[0] = s * cfg->vocab_size;
    logits.stride[1] = cfg->vocab_size;
    logits.stride[2] = 1;
    logits.nbytes = b * s * cfg->vocab_size * sizeof(float);
    cudaMalloc(&logits.gpu_data, logits.nbytes);

    printf("Starting Warm-up...\n");
    for (int i = 0; i < 3; i++) {
        // [FIX 2]: Reset the arena memory pointer before every forward pass!
        // (Assuming your Workspace class has a reset() or clear() method. 
        // If it doesn't, you need to add one that sets its internal offset to 0).
        arena->reset(); 
        
        llm->forward(&arena_obj, cfg, &token_ids, &logits);
    }

    cudaDeviceSynchronize(); 
    printf("Warm-up complete!\n\n");

    int target_tokens = 50;
    printf("Benchmarking %d tokens...\n", target_tokens);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < target_tokens; step++) {
        // [FIX 2]: Reset the arena here too!
        arena->reset();
        
        llm->forward(&arena_obj, cfg, &token_ids, &logits);
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    double tps = target_tokens / seconds;

    printf("===== BENCHMARK RESULTS =====\n");
    printf("Total Time: %f seconds\n", seconds);
    printf("Speed:      %.2f Tokens Per Second (TPS)\n", tps);
    printf("=============================\n");

    float* cpu_logits = (float*)malloc(logits.nbytes);
    cudaMemcpy(cpu_logits, logits.gpu_data, logits.nbytes, cudaMemcpyDeviceToHost);

    int last_token_offset = (0 * s * cfg->vocab_size) + ((s - 1) * cfg->vocab_size);
    
    printf("===== LOGITS OUTPUT (Last Token, First 10 values) =====\n");
    for (int i = 0; i < 10; i++) {
        printf("Logit[%d]: %f\n", i, cpu_logits[last_token_offset + i]);
    }
    printf("=======================================================\n");


    delete llm;
    free(db);
    free(cfg);
    free(cpu_logits);
    cudaFree(token_ids.gpu_data);
    cudaFree(logits.gpu_data);
    arena->destroy();

    return 0;
}