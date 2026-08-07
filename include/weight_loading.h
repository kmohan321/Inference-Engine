#pragma once

#include <windows.h>
#include <stdint.h>
#include <stddef.h>
#include "model_def.h"


#define MAX_TENSORS 4096

typedef struct
{
    HANDLE file;
    HANDLE mapping;
    void *data;
    uint64_t size;

} MappedFile;

typedef struct
{
    MappedFile file;

    Tensor tensors[MAX_TENSORS];

    int tensor_count;

    uint64_t header_size;

} TensorDB;

#ifdef __cplusplus
extern "C" {
#endif

void load_safetensors(const char *filename, TensorDB *db);
void load_config(const char *filename, ModelConfig *cfg);
Tensor *find_tensor(TensorDB *db, const char *name);


void upload_to_gpu(TensorDB *db);
void upcast_bf16_to_fp32(uint16_t *bf16_data, float *fp32_data, uint64_t num_elements);


#ifdef __cplusplus
}
#endif