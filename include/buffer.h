#pragma once

#include <cstddef>

struct Workspace {
    char* base_ptr = nullptr;
    size_t total_capacity = 0;
    size_t current_offset = 0;

    void allocate(size_t max_bytes);
    void* get_offset(size_t requested_bytes);
    void reset();
    void destroy();
};

