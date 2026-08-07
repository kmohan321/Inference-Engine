#include <cuda_runtime.h>
#include <stdio.h>
#include "model_def.h"

//weights -> (vocab_size, hidden_dim)
//x -> (batch, seq_len)

__global__ void embedding (float *embed_weights, int *x, float *out, int s, int d){

    int idx = threadIdx.x;
    int block_offset = blockIdx.x;

    for(int i = idx; i < s; i += blockDim.x){
      int token_id = x[block_offset * s + i];
      for (int j = 0; j < d; j++){
        out[block_offset * s * d + i * d + j] = embed_weights[token_id * d + j];
      }

    }

}

extern "C" void embedding_forward(Tensor *embed_weights, Tensor *x, Tensor *out) {
    int batch = x->shape[0];
    int seq_len = x->shape[1];
    int hidden_dim = embed_weights->shape[1];

    dim3 grid(batch);
    dim3 block(256);

    embedding<<<grid, block>>>(
      (float*)embed_weights->gpu_data, 
      (int*)x->gpu_data, 
      (float*)out->gpu_data, 
      seq_len, 
      hidden_dim
    );
}

