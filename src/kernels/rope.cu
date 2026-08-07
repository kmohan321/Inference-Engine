#include <cuda_runtime.h>
#include "model_def.h"
#include <math.h>

__global__ void fused_rope_and_permute_kernel(
    const float* __restrict__ in_q, 
    float* __restrict__ out_q, 
    const float* __restrict__ freqs, // shape: [s, d/2]
    int b, int s, int nh, int d) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = b * s * nh * d;
    
    if (idx >= total_elements) return;

    int b_idx  = idx / (s * nh * d);
    int s_idx  = (idx / (nh * d)) % s;
    int nh_idx = (idx / d) % nh;
    int d_idx  = idx % d;

    if (d_idx >= d / 2) return; 

    int half_d = d / 2;
    int pair_idx = idx + half_d;

    float q0 = in_q[idx];
    float q1 = in_q[pair_idx];


    int freq_idx = s_idx * half_d + d_idx;
    float m_theta = freqs[freq_idx];

    float cos_val = cosf(m_theta);
    float sin_val = sinf(m_theta);

    float q0_rotated = q0 * cos_val - q1 * sin_val;
    float q1_rotated = q1 * cos_val + q0 * sin_val;

    int out_base_idx = b_idx * (nh * s * d) + nh_idx * (s * d) + s_idx * d;
    out_q[out_base_idx + d_idx] = q0_rotated;
    out_q[out_base_idx + d_idx + half_d] = q1_rotated;
}

extern "C" void rope_forward(Tensor *x, Tensor *out, Tensor *freqs) {

    int B = x->shape[0];
    int S = x->shape[1];
    int Nh = x->shape[2];
    int D = x->shape[3];
    
    int total_elements = B * S * Nh * D;

    dim3 threads(32 * 32);
    dim3 blocks((total_elements + 1023)/1024);

    fused_rope_and_permute_kernel<<<blocks, threads>>>(
        (float*)x->gpu_data, 
        (float*)out->gpu_data,
        (float*)freqs->gpu_data, 
        B, S, Nh, D
    );
}



__global__ void generate_freqs_kernel(float *freqs, int s, int half_d, float theta_base) {

    int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (s_idx >= s || d_idx >= half_d) return;

    float exponent = -((float)(2 * d_idx) / (half_d * 2));
    float theta_i = powf(theta_base, exponent);

    float m_theta = (float)s_idx * theta_i;

    freqs[s_idx * half_d + d_idx] = m_theta;
}

extern "C" void generate_rope_freqs(Tensor * freqs, int s, int head_dim, float rope_theta) {
    
    int half_d = head_dim / 2;
    
    dim3 threads(32, 8);
    dim3 blocks((half_d + threads.x - 1) / threads.x, 
                (s + threads.y - 1) / threads.y);

    generate_freqs_kernel<<<blocks, threads>>>(
        (float*)freqs->gpu_data, 
        s, half_d, rope_theta
    );

}
