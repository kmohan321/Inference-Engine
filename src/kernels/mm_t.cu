#include <cuda_runtime.h>
#include "model_def.h"
#include <stdio.h>

#define TILE_M 64 //( actual tile size -> (8 * 32), thread coarsening -> 8)
#define TILE_N 64
#define TILE_K 32

#define PAD 1
#define SMEM_STRIDE (TILE_K + PAD) // 33 floats (prevents bank conflicts)

//(A -> (M * K), B-> (N * K))

__global__ void matrix_mul_t(const float * __restrict__ a, 
                           const float * __restrict__ b, 
                           float * __restrict__ c,
                           int m, int n, int k,
                           int stride_A_batch, int stride_B_batch, int stride_C_batch) 
{

    const float *A_batch = a + (blockIdx.z * stride_A_batch);
    const float *B_batch = b + (blockIdx.z * stride_B_batch);
    float *C_batch       = c + (blockIdx.z * stride_C_batch);

    int tid = threadIdx.x;

    int tx = tid % 16;
    int ty = tid / 16;

    extern __shared__ float smem[];
    float *smemA = smem;                         // Size: 64 x 33 = 2,112 floats
    float *smemB = smemA + TILE_M * SMEM_STRIDE; // Size: 64 x 33 = 2,112 floats

    float reg[4][4] = {0.0f};

    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;

    int num_k_tiles = (k + TILE_K - 1) / TILE_K;

    for (int i = 0; i < num_k_tiles; i++) {
        int k_offset = i * TILE_K;

        // (64 x 32) -> 2048 elements total / 256 threads = 8 floats per thread
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            int load_idx = tid + l * 256;
            int a_row = load_idx / TILE_K; // 0 to 63
            int a_col = load_idx % TILE_K; // 0 to 31

            int global_row = block_m + a_row;
            int global_col = k_offset + a_col;

            if (global_row < m && global_col < k) {
                smemA[a_row * SMEM_STRIDE + a_col] = A_batch[global_row * k + global_col];
            } else {
                smemA[a_row * SMEM_STRIDE + a_col] = 0.0f;
            }
        }

        #pragma unroll // B -> (N * K)
        for (int l = 0; l < 8; l++) {
            int load_idx = tid + l * 256;
            int b_row = load_idx / TILE_K; // 0 to 63 (corresponds to N)
            int b_col = load_idx % TILE_K; // 0 to 31 (corresponds to K)

            int global_row = block_n + b_row;
            int global_col = k_offset + b_col;

            if (global_row < n && global_col < k) {
                smemB[b_row * SMEM_STRIDE + b_col] = B_batch[global_row * k + global_col];
            } else {
                smemB[b_row * SMEM_STRIDE + b_col] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float fragA[4];
            float fragB[4];

            #pragma unroll
            for (int p = 0; p < 4; p++) {
                fragA[p] = smemA[(ty * 4 + p) * SMEM_STRIDE + kk];
            }

            #pragma unroll
            for (int q = 0; q < 4; q++) {
                fragB[q] = smemB[(tx * 4 + q) * SMEM_STRIDE + kk];
            }

            #pragma unroll
            for (int p = 0; p < 4; p++) {
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    reg[p][q] += fragA[p] * fragB[q];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int global_row = block_m + ty * 4 + p;
        if (global_row < m) {
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int global_col = block_n + tx * 4 + q;
                if (global_col < n) {
                    C_batch[global_row * n + global_col] = reg[p][q];
                }
            }
        }
    }
}

extern "C" void mm_t_forward(Tensor *A, Tensor *B, Tensor *C) {
    int M = A->shape[A->ndim - 2];
    int K = A->shape[A->ndim - 1];
    int N = B->shape[B->ndim - 2];

    int total_batches = 1;
    for (int i = 0; i < A->ndim - 2; i++) {
        total_batches *= A->shape[i];
    }

    int stride_A_batch = (A->ndim > 2) ? A->stride[A->ndim - 3] : 0;
    int stride_B_batch = (B->ndim > 2) ? B->stride[B->ndim - 3] : 0;
    int stride_C_batch = (C->ndim > 2) ? C->stride[C->ndim - 3] : 0;

    dim3 block(256);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M,
              total_batches);

    size_t smem_size = 2 * TILE_M * SMEM_STRIDE * sizeof(float);

    matrix_mul_t<<<grid, block, smem_size>>>(
        (const float*)A->gpu_data, (const float*)B->gpu_data, (float*)C->gpu_data,
        M, N, K, stride_A_batch, stride_B_batch, stride_C_batch
    );
}