#include <cuda_runtime.h>
#include "model_def.h"

__global__ void elementwise_add(float *a, float *b, int total_elem){

  int idx = threadIdx.x;
  int elem_idx = blockDim.x * blockIdx.x + idx;

  if(elem_idx >= total_elem) return;

  a[elem_idx] = a[elem_idx] + b[elem_idx];

}

extern "C" void elem_forward_add(Tensor *a, Tensor *b){

  int total_elements = 1;
  for(int i =0; i < a->ndim; i++){
    total_elements *= a->shape[i];
  }
  dim3 blocks(256);
  dim3 grid((total_elements + 255)/256);

  elementwise_add<<<grid, blocks>>>(
    (float*)a->gpu_data, (float*)b->gpu_data,
    total_elements
  );
  
}

__device__ __forceinline__ float silu(float x)
{
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    return x * sigmoid_x;
}


__global__ void elementwise_mul_act(float *a, float *b, int total_elem, ActivationType act){

  int idx = threadIdx.x;
  int elem_idx = blockDim.x * blockIdx.x + idx;

  if(elem_idx >= total_elem) return;

  float x = a[elem_idx];

  switch (act)
  {
      case SILU:
          x = silu(x);
          break;

      case GELU:
          // x = gelu(x);
          break;

      default:
          break;
  }

  a[elem_idx] = x * b[elem_idx];

}

extern "C" void elem_forward_mul_act(Tensor *a, Tensor *b, ActivationType act){

  int total_elements = 1;
  for(int i =0; i < a->ndim; i++){
    total_elements *= a->shape[i];
  }
  dim3 blocks(256);
  dim3 grid((total_elements + 255)/256);

  elementwise_mul_act<<<grid, blocks>>>(
    (float*)a->gpu_data, (float*)b->gpu_data,
    total_elements, act
  );
  
}
