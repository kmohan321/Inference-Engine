#include<cuda_runtime.h>
#include "model_def.h"
#include <math_functions.h>

//input x -> (b, s, d) -> (b * s, d)

__global__ void unified_norm(float *x, float *out, float *weight, float *bias, 
                             int d, float eps, bool is_rms_norm) {

    int idx = threadIdx.x;
    int bs_offset = blockIdx.x;
    int offset = bs_offset * d;

    extern float __shared__ smem[];
    float *smem_sum = smem;
    float *smem_sqsum = smem_sum + blockDim.x;

    float sq_sum = 0.0f;
    float sum = 0.0f;
    
    for(int i = idx; i < d; i += blockDim.x) {
        float curr_value = x[offset + i];
        sum += curr_value;
        sq_sum += (curr_value * curr_value);
    }
    
    smem_sum[idx] = sum;
    smem_sqsum[idx] = sq_sum;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0 ; i /= 2) {
        if(idx < i) {
            smem_sum[idx] += smem_sum[idx + i];
            smem_sqsum[idx] += smem_sqsum[idx + i];
        }
        __syncthreads();
    }
    
    float global_sum = smem_sum[0];
    float global_sqsum = smem_sqsum[0];

    // If RMSNorm, force mean to 0. 
    float mean = is_rms_norm ? 0.0f : (global_sum / d);
    float var = (global_sqsum / d) - (mean * mean);
    float r_std = rsqrtf(var + eps); 

    for(int i = idx; i < d; i += blockDim.x) {
        float curr_value = x[offset + i];
        float out_value = (curr_value - mean) * r_std; 
        float b_val = (bias != nullptr) ? bias[i] : 0.0f; 
        
        out[offset + i] = out_value * weight[i] + b_val;
    }
}

extern "C" void norm_forward(Tensor *x, Tensor *out, Tensor *norm_weight, Tensor *norm_bias, 
                  int b, int s, int d, float eps, bool is_rms_norm) {

    dim3 grid(b * s);
    dim3 block(256);
    int smem_size = 2 * block.x * sizeof(float);
    float* bias_ptr = (norm_bias != nullptr && norm_bias->gpu_data != nullptr) ? (float*)norm_bias->gpu_data : nullptr;

    unified_norm<<<grid, block, smem_size>>>(
        (float*)x->gpu_data,
        (float*)out->gpu_data,
        (float*)norm_weight->gpu_data, 
        bias_ptr,
        d, 
        eps, 
        is_rms_norm
    );
}