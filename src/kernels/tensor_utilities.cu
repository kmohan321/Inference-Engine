#include <stdio.h>
#include <cuda_runtime.h>
#include "model_def.h"
#include "cuda_functions.h"
#include "weight_loading.h"
#include "buffer.h"

extern "C" void upload_to_gpu(TensorDB *db){

  for(int i =0; i<db->tensor_count; i++){
    
    Tensor *tensor = &db->tensors[i];
    printf("%s is uploading to GPU \n", tensor->name);

    uint64_t num_elements = tensor->nbytes / 2; // Because BF16 is 2 bytes

    float *temp_fp32 = (float*)malloc(num_elements * sizeof(float));

    upcast_bf16_to_fp32((uint16_t*)tensor->cpu_data, temp_fp32, num_elements);

    tensor->nbytes = num_elements * sizeof(float);
    tensor->dtype = FLOAT32; 

    cudaMalloc(&tensor->gpu_data, tensor->nbytes);
    cudaMemcpy(tensor->gpu_data, temp_fp32, tensor->nbytes, cudaMemcpyHostToDevice);

    free(temp_fp32);

  }

}

extern "C" void transpose(Tensor *input, const int *index){

    uint64_t a =  input->shape[index[0]];
    input->shape[index[0]] = input->shape[index[1]];
    input->shape[index[1]] = a;

    uint64_t b =  input->stride[index[0]];
    input->stride[index[0]] = input->stride[index[1]];
    input->stride[index[1]] = b;

}

extern "C" void permute(Tensor *A, const int *order) {
    
    uint64_t new_shape[8];
    uint64_t new_stride[8];

    for(int i = 0; i < A->ndim; i++){
        new_shape[i] = A->shape[order[i]];
        new_stride[i] = A->stride[order[i]];
    }

    for(int i = 0; i < A->ndim; i++){
        A->shape[i] = new_shape[i];
        A->stride[i] = new_stride[i];
    }
}

void tensor_view(Tensor *t, int ndim, const int *shape)
{
    t->ndim = ndim;

    for (int i = 0; i < ndim; i++)
        t->shape[i] = shape[i];


    for(int i = 0; i < t->ndim; i++){
        int c = 1;
        for(int j = i + 1; j < t->ndim; j++){
            c *= t->shape[j];
        }
        t->stride[i] = c;
    }
}


Tensor* tensor_empty_like(Tensor* src, Workspace* arena) {
    Tensor* t = new Tensor();
    
    t->dtype = src->dtype;
    t->ndim = src->ndim;
    t->nbytes = src->nbytes;
    t->offset = 0;
    
    for (int i = 0; i < src->ndim; i++) {
        t->shape[i] = src->shape[i];
    }

    for(int i = 0; i < t->ndim; i++){
        int c = 1;
        for(int j = i + 1; j < t->ndim; j++){
            c *= t->shape[j];
        }
        t->stride[i] = c;
    }
    
    t->gpu_data = arena->get_offset(t->nbytes);
    t->cpu_data = nullptr;
    
    return t;
}

Tensor* tensor_empty(const int *shape, const int ndim, Workspace* arena) {
    Tensor* t = new Tensor();

    int t_elements = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t_elements *= shape[i];
    }
    
    t->ndim = ndim;
    
    for(int i = 0; i < t->ndim; i++){
        int c = 1;
        for(int j = i + 1; j < t->ndim; j++){
            c *= t->shape[j];
        }
        t->stride[i] = c;
    }

    // t->dtype = dtype;
    t->nbytes = t_elements * sizeof(float);
    t->offset = 0;
    
    t->gpu_data = arena->get_offset(t->nbytes);
    t->cpu_data = nullptr;
    
    return t;
}


__global__ void permute_physical_kernel(
    const float* __restrict__ in, 
    float* __restrict__ out, 
    int b, int s, int nh, int d) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = b * s * nh * d;
    
    if (idx >= total_elements) return;

    // Decode [b, s, nh, d]
    int b_idx  = idx / (s * nh * d);
    int s_idx  = (idx / (nh * d)) % s;
    int nh_idx = (idx / d) % nh;
    int d_idx  = idx % d;

    // Encode [b, nh, s, d]
    int out_idx = b_idx * (nh * s * d) + nh_idx * (s * d) + s_idx * d + d_idx;
    
    // Write
    out[out_idx] = in[idx];
}

extern "C" void permute_physical_forward(Tensor *in, Tensor *out) {
    int b = in->shape[0];
    int s = in->shape[1];
    int nh = in->shape[2];
    int d = in->shape[3];

    int total_elements = b * s * nh * d;
    dim3 block(256);
    dim3 grid((total_elements + 255) / 256);

    permute_physical_kernel<<<grid, block>>>(
        (const float*)in->gpu_data, 
        (float*)out->gpu_data, 
        b, s, nh, d
    );
}
