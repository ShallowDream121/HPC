#ifndef _H_KMEANS
#define _H_KMEANS

#include <CL/sycl.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstring>

// 分配二维数组的宏
#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

// 计时函数
double wtime() {
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start_time).count();
}

#endif