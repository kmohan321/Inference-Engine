#include <cuda_runtime.h>
#include "model_def.h"
#include <math.h>
#include <float.h>

__global__ void softmax_kernel(
    float *input,
    int batches,
    int rows,
    int cols
)
{
    int row = blockIdx.x;
    int batch = blockIdx.y;

    if (row >= rows || batch >= batches)
        return;

    int offset = batch * rows * cols + row * cols;

    float max_val = -FLT_MAX;

    for(int j = 0; j < cols; j++)
    {
        float val = input[offset + j];
        max_val = fmaxf(max_val, val);
    }

    float sum = 0.0f;

    for(int j = 0; j < cols; j++)
    {
        float val = expf(input[offset + j] - max_val);
        input[offset + j] = val;
        sum += val;
    }

    for(int j = 0; j < cols; j++)
    {
        input[offset + j] /= sum;
    }
}

extern "C" void softmax_forward(Tensor *input){

  int batch = input->shape[0];
  int heads = input->shape[1];
  int s = input->shape[2];
  
  dim3 grid(s, batch * heads);

  softmax_kernel<<<grid,1>>>(
      (float*)input->gpu_data,
      batch * heads,
      s,
      s
  );

}
