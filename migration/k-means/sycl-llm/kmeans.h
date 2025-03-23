#ifndef _H_KMEANS
#define _H_KMEANS

#include <CL/sycl.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

// 定义调试信息输出宏
#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

// 定义二维数组动态分配宏
#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

// SYCL 错误检查宏
#define CHECK_SYCL_ERROR(expr) do {                          \
    try {                                                   \
        expr;                                               \
    } catch (const cl::sycl::exception& e) {                \
        err("SYCL Exception: %s\n", e.what());              \
    }                                                       \
} while (0)

// K-means 函数声明
float** omp_kmeans(int, float**, int, int, int, float, int*);
float** seq_kmeans(float**, int, int, int, float, int*, int*);
float** sycl_kmeans(float**, int, int, int, float, int*, int*); // 替换 CUDA 实现为 SYCL 实现

// 文件读写函数声明
float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);

// 计时函数声明
double  wtime(void);

// 调试标志
extern int _debug;

#endif