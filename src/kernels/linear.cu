#include <cuda_runtime.h>
#include "model_def.h"
#include "cuda_functions.h"


//make it to support bias later
extern "C" void linear_layer(Tensor *x, Tensor *layer_weight, Tensor *C){

    mm_t_forward(x, layer_weight, C);

}

