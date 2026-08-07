#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "model_def.h"
#include "weight_loading.h"
#include "buffer.h"
#include "cuda_functions.h"
#include "tokenizer.h"
#include "models.h"
#include <iostream>
#include <string>
#include <vector>


int get_next_token(float* logits, int vocab_size) {
    int best_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int main() {
    printf("Loading model...\n");

    TensorDB *db = (TensorDB*)malloc(sizeof(TensorDB));
    ModelConfig *cfg = (ModelConfig*)malloc(sizeof(ModelConfig));

    Tokenizer tokenizer;
    if (!tokenizer.load("model_data/tokenizer.json")) {
        printf("Failed to load tokenizer!\n");
        exit(1);
    }

    load_safetensors("model_data/model.safetensors", db);
    load_config("model_data/config.json", cfg);
    
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
    int max_new_tokens = 2048;

    // Continuous Chat Loop
    while (true) {
        std::string user_prompt;
        std::cout << "\nUser: ";
        std::getline(std::cin, user_prompt);

        // Exit condition
        if (user_prompt == "exit" || user_prompt == "quit") {
            break;
        }

        std::string formatted_prompt = 
            "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";

        std::vector<int> input_ids = tokenizer.encode(formatted_prompt);
        std::cout << "Model: ";

        // Generation Loop
        for (int step = 0; step < max_new_tokens; step++) {
            int s = input_ids.size();

            // 1. Prepare Input Token Tensor
            Tensor token_ids = {0};
            token_ids.ndim = 2;
            token_ids.shape[0] = b;
            token_ids.shape[1] = s;
            token_ids.stride[0] = s;
            token_ids.stride[1] = 1;
            token_ids.nbytes = b * s * sizeof(int);
            cudaMalloc(&token_ids.gpu_data, token_ids.nbytes);
            
            // Notice: using input_ids.data() to get the raw array pointer from the std::vector
            cudaMemcpy(token_ids.gpu_data, input_ids.data(), token_ids.nbytes, cudaMemcpyHostToDevice);

            // 2. Prepare Logits Tensor
            Tensor logits = {0};
            logits.ndim = 3;
            logits.shape[0] = b;
            logits.shape[1] = s;
            logits.shape[2] = cfg->vocab_size;
            logits.stride[0] = s * cfg->vocab_size;
            logits.stride[1] = cfg->vocab_size;
            logits.stride[2] = 1;
            logits.nbytes = b * s * cfg->vocab_size * sizeof(float);
            cudaMalloc(&logits.gpu_data, logits.nbytes);

            // 3. Run Forward Pass
            arena->reset(); 
            llm->forward(&arena_obj, cfg, &token_ids, &logits);
            cudaDeviceSynchronize();

            // 4. Fetch only the logits for the very last token in the sequence
            float* cpu_logits = (float*)malloc(cfg->vocab_size * sizeof(float));
            int last_token_offset = (s - 1) * cfg->vocab_size;
            cudaMemcpy(cpu_logits, ((float*)logits.gpu_data) + last_token_offset, cfg->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
            // 5. Pick the best token
            int next_token_id = get_next_token(cpu_logits, cfg->vocab_size);
            free(cpu_logits);

            // Clean up GPU tensors before the next step
            cudaFree(token_ids.gpu_data);
            cudaFree(logits.gpu_data);

            // Break if the model outputs the EOS (End of Sequence) token. 
            // Note: '2' is standard for Llama/Mistral, check your tokenizer.json if it differs.
            if (next_token_id == 151645 || next_token_id == 151643) {
                break; 
            }

            // 6. Decode and stream to console
            std::string piece = tokenizer.decode(next_token_id);
            std::cout << piece << std::flush;

            // 7. Append the new token to the context so the model sees it on the next loop
            input_ids.push_back(next_token_id);
        }
        std::cout << std::endl;
    }

    // Final Cleanup
    arena->destroy();
    delete llm;
    free(db);
    free(cfg);

    return 0;
}