#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include "cJSON.h"
#include "model_def.h"
#include "weight_loading.h"

int map_file(const char *filename, MappedFile *mf)
{
    mf->file = CreateFileA(
        filename,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (mf->file == INVALID_HANDLE_VALUE)
    {
        printf("Failed to open file.\n");
        return 0;
    }

    LARGE_INTEGER filesize;
    if (!GetFileSizeEx(mf->file, &filesize))
    {
        CloseHandle(mf->file);
        return 0;
    }

    mf->size = (uint64_t)filesize.QuadPart;

    mf->mapping = CreateFileMappingA(
        mf->file,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL);

    if (mf->mapping == NULL)
    {
        CloseHandle(mf->file);
        return 0;
    }

    mf->data = MapViewOfFile(
        mf->mapping,
        FILE_MAP_READ,
        0,
        0,
        0);

    if (mf->data == NULL)
    {
        CloseHandle(mf->mapping);
        CloseHandle(mf->file);
        return 0;
    }

    return 1;
}

void unmap_file(MappedFile *mf)
{
    if (mf->data)
        UnmapViewOfFile(mf->data);

    if (mf->mapping)
        CloseHandle(mf->mapping);

    if (mf->file != INVALID_HANDLE_VALUE)
        CloseHandle(mf->file);
}


void load_safetensors(const char *filename, TensorDB *db)
{
    if (!map_file(filename, &db->file))
        return ;

    uint8_t *base = (uint8_t *)db->file.data;

    db->header_size = *(uint64_t *)base;

    char *header = (char *)(base + 8);

    cJSON *root =
        cJSON_ParseWithLength(header, db->header_size);

    if (!root)
        return ;

    db->tensor_count = 0;

    cJSON *tensor = NULL;

    cJSON_ArrayForEach(tensor, root)
    {
        if (strcmp(tensor->string, "__metadata__") == 0)
            continue;

        Tensor *t = &db->tensors[db->tensor_count++];

        memset(t, 0, sizeof(Tensor));

        strcpy(t->name, tensor->string);

        cJSON *dtype =
            cJSON_GetObjectItem(tensor, "dtype");

        if (!strcmp(dtype->valuestring, "F16"))
            t->dtype = FLOAT16;
        else if (!strcmp(dtype->valuestring, "F32"))
            t->dtype = FLOAT32;
        else if (!strcmp(dtype->valuestring, "BF16"))
            t->dtype = BFLOAT16;

        cJSON *shape =
            cJSON_GetObjectItem(tensor, "shape");

        t->ndim = cJSON_GetArraySize(shape);

        for (int i = 0; i < t->ndim; i++)
        {
            t->shape[i] =
                (uint64_t)cJSON_GetArrayItem(shape, i)->valuedouble;

        }

        cJSON *offsets =
            cJSON_GetObjectItem(tensor, "data_offsets");

        uint64_t begin =
            (uint64_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;

        uint64_t end =
            (uint64_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;

        t->offset = begin;
        t->nbytes = end - begin;

        for(int i = 0; i < t->ndim; i++){
            int c = 1;
            for(int j = i + 1; j < t->ndim; j++){
                c *= t->shape[j];
            }
            t->stride[i] = c;
        }

        t->cpu_data =
            (uint8_t *)db->file.data
            + 8
            + db->header_size
            + begin;
    }

    cJSON_Delete(root);

}

char *read_file(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return NULL;

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *buffer = (char *)malloc(size + 1);

    fread(buffer, 1, size, fp);
    buffer[size] = '\0';

    fclose(fp);

    return buffer;
}

void load_config(const char *filename, ModelConfig *cfg)
{
    char *json = read_file(filename);

    if (json == NULL)
    {
        printf("Could not open %s\n", filename);
        return ;
    }

    cJSON *root = cJSON_Parse(json);

    if (!root)
    {
        printf("Failed to parse config.json\n");
        free(json);
        return ;
    }

    cJSON *model_type = cJSON_GetObjectItem(root, "architectures");
    if (model_type && model_type->valuestring) {
        strcpy(cfg->model_type, model_type->valuestring);
    } else {
        strcpy(cfg->model_type, "unknown");
    }

    cJSON *h_act = cJSON_GetObjectItem(root, "hidden_act");

    if (h_act && h_act->valuestring)
    {
        if (strcmp(h_act->valuestring, "silu") == 0)
        {
            cfg->act = SILU;
        }
        else if (strcmp(h_act->valuestring, "gelu") == 0)
        {
            cfg->act = GELU;
        }
        else
        {
            cfg->act = UNKNOWN_ACT;
        }
    }
    else
    {
        cfg->act = UNKNOWN_ACT;
    }

    cfg->hidden_size =
        cJSON_GetObjectItem(root, "hidden_size")->valueint;

    cfg->intermediate_size =
        cJSON_GetObjectItem(root, "intermediate_size")->valueint;

    cfg->num_layers =
        cJSON_GetObjectItem(root, "num_hidden_layers")->valueint;

    cfg->num_heads =
        cJSON_GetObjectItem(root, "num_attention_heads")->valueint;

    cfg->num_kv_heads =
        cJSON_GetObjectItem(root, "num_key_value_heads")->valueint;

    cfg->vocab_size =
        cJSON_GetObjectItem(root, "vocab_size")->valueint;

    cfg->head_dim =
        cJSON_GetObjectItem(root, "head_dim")->valueint;

    cfg->max_position_embeddings =
        cJSON_GetObjectItem(root, "max_position_embeddings")->valueint;

    cfg->rms_norm_eps =
        (float)cJSON_GetObjectItem(root, "rms_norm_eps")->valuedouble;

    cfg->rope_theta =
        (float)cJSON_GetObjectItem(root, "rope_theta")->valueint;

    cJSON *dtype =
        cJSON_GetObjectItem(root, "torch_dtype");

    if (strcmp(dtype->valuestring, "float16") == 0)
        cfg->dtype = FLOAT16;
    else if (strcmp(dtype->valuestring, "bfloat16") == 0)
        cfg->dtype = BFLOAT16;
    else
        cfg->dtype = FLOAT32;

    cJSON_Delete(root);
    free(json);

}

Tensor *find_tensor(TensorDB *db, const char *name)
{
    for (int i = 0; i < db->tensor_count; i++)
    {
        if (!strcmp(db->tensors[i].name, name))
            return &db->tensors[i];
    }

    return NULL;
}

// Tensor *find_layer_tensor(
//     TensorDB *db,
//     int layer,
//     const char *suffix)
// {
//     char name[256];

//     snprintf(name,
//              sizeof(name),
//              "model.layers.%d.%s",
//              layer,
//              suffix);

//     return find_tensor(db, name);
// }

void upcast_bf16_to_fp32(uint16_t *bf16_data, float *fp32_data, uint64_t num_elements) {
    for (uint64_t i = 0; i < num_elements; i++) {
        uint32_t fp32_bits = ((uint32_t)bf16_data[i]) << 16;
        memcpy(&fp32_data[i], &fp32_bits, sizeof(float));
    }
}

// void build_model(Model *model,
//                 ModelConfig *cfg,
//                 TensorDB *db)
// {
//     model->layers = malloc(sizeof(Layer) * cfg->num_layers);
//     for(int i = 0; i<cfg->num_layers; i++){
//       model->layers[i].atten.q_proj.weight = find_layer_tensor(db, i, "self_attn.q_proj.weight");
//       model->layers[i].atten.k_proj.weight = find_layer_tensor(db, i, "self_attn.k_proj.weight");
//       model->layers[i].atten.v_proj.weight = find_layer_tensor(db, i, "self_attn.v_proj.weight");
//       model->layers[i].atten.o_proj.weight = find_layer_tensor(db, i, "self_attn.o_proj.weight");

//       model->layers[i].atten.q_norm.weight = find_layer_tensor(db, i, "self_attn.q_norm.weight");
//       model->layers[i].atten.q_norm.bias = find_layer_tensor(db, i, "self_attn.q_norm.bias");
//       model->layers[i].atten.k_norm.weight = find_layer_tensor(db, i, "self_attn.k_norm.weight");
//       model->layers[i].atten.k_norm.bias = find_layer_tensor(db, i, "self_attn.k_norm.bias");
      
//       model->layers[i].mlp.up_proj.weight = find_layer_tensor(db, i, "mlp.up_proj.weight");
//       model->layers[i].mlp.down_proj.weight = find_layer_tensor(db, i, "mlp.down_proj.weight");
//       model->layers[i].mlp.gate_proj.weight = find_layer_tensor(db, i, "mlp.gate_proj.weight");

//       model->layers[i].input_layernorm.weight = find_layer_tensor(db, i, "input_layernorm.weight");
//       model->layers[i].input_layernorm.bias = find_layer_tensor(db, i, "input_layernorm.bias");
//       model->layers[i].post_attention_layernorm.weight = find_layer_tensor(db, i, "post_attention_layernorm.weight");
//       model->layers[i].post_attention_layernorm.bias = find_layer_tensor(db, i, "post_attention_layernorm.bias");

//     }

//     model->embed_weight = find_tensor(db, "model.embed_tokens.weight");
//     model->lm_head_weight = find_tensor(db, "lm_head.weight");
//     model->norm.weight = find_tensor(db, "model.norm.weight");

 
// }

