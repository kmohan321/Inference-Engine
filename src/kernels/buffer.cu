#include "buffer.h"

#include <cuda_runtime.h>
#include <iostream>

void Workspace::allocate(size_t max_bytes)
{
    total_capacity = max_bytes;
    current_offset = 0;

    cudaError_t err = cudaMalloc((void**)&base_ptr, total_capacity);

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate workspace!" << std::endl;
    }
}

void* Workspace::get_offset(size_t requested_bytes)
{
    size_t aligned_bytes = (requested_bytes + 255) & ~255;

    if (current_offset + aligned_bytes > total_capacity)
    {
        std::cerr << "OUT OF WORKSPACE MEMORY!" << std::endl;
        return nullptr;
    }

    void* ptr = base_ptr + current_offset;
    current_offset += aligned_bytes;

    return ptr;
}

void Workspace::reset()
{
    current_offset = 0;
}

void Workspace::destroy()
{
    if (base_ptr)
    {
        cudaFree(base_ptr);
        base_ptr = nullptr;
    }
}