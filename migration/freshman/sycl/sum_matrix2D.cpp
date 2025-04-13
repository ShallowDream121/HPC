#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace sycl;
using namespace std::chrono;

constexpr int DIMX = 32;
constexpr int DIMY = 32;

// 时间测量函数
double cpuSecond() {
    return duration_cast<duration<double>>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

// 初始化数据
void initialData(float* ip, int size) {
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>(rand() & 0xffff) / 1000.0f;
    }
}

// 结果验证
void checkResult(float* hostRef, float* gpuRef, int N) {
    constexpr double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (std::abs(hostRef[i] - gpuRef[i]) > epsilon) {
            std::cerr << "Error at index " << i << ": " 
                      << hostRef[i] << " vs " << gpuRef[i] << std::endl;
            return;
        }
    }
    std::cout << "Check result success!\n";
}

// CPU版本矩阵相加
void sumMatrix2D_CPU(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            MatC[i + j * nx] = MatA[i + j * nx] + MatB[i + j * nx];
        }
    }
}

class MatrixAdd;

int main(int argc, char** argv) {
    std::cout << "Starting..." << std::endl;
    
    constexpr int nx = 1 << 12;
    constexpr int ny = 1 << 12;
    constexpr int nxy = nx * ny;
    const size_t nBytes = nxy * sizeof(float);

    try {
        // 创建SYCL队列（自动选择GPU）
        queue q(gpu_selector_v);
        std::cout << "Running on: " 
                 << q.get_device().get_info<info::device::name>() << "\n";

        // 分配主机内存
        float* A_host = malloc_host<float>(nxy, q);
        float* B_host = malloc_host<float>(nxy, q);
        float* C_host = malloc_host<float>(nxy, q);
        float* C_from_gpu = malloc_host<float>(nxy, q);

        // 初始化数据
        initialData(A_host, nxy);
        initialData(B_host, nxy);
        std::memset(C_host, 0, nBytes);
        std::memset(C_from_gpu, 0, nBytes);

        // 分配设备内存
        float* A_dev = malloc_device<float>(nxy, q);
        float* B_dev = malloc_device<float>(nxy, q);
        float* C_dev = malloc_device<float>(nxy, q);

        // 拷贝数据到设备
        q.memcpy(A_dev, A_host, nBytes).wait();
        q.memcpy(B_dev, B_host, nBytes).wait();

        // CPU计算
        auto cpuStart = cpuSecond();
        sumMatrix2D_CPU(A_host, B_host, C_host, nx, ny);
        auto cpuElapsed = cpuSecond() - cpuStart;
        std::cout << "CPU Execution Time: " << cpuElapsed << " sec\n";

        // 配置GPU执行参数
        constexpr range<2> block(DIMX, DIMY);
        const range<2> grid(
            (nx + DIMX - 1) / DIMX,
            (ny + DIMY - 1) / DIMY
        );

        // 执行GPU内核
        auto gpuStart = cpuSecond();
        q.submit([&](handler& h) {
            h.parallel_for<MatrixAdd>(
                nd_range<2>(grid * block, block),
                [=](nd_item<2> item) {
                    const int ix = item.get_global_id(0);
                    const int iy = item.get_global_id(1);
                    const int idx = iy * nx + ix;

                    if (ix < nx && iy < ny) {
                        C_dev[idx] = A_dev[idx] + B_dev[idx];
                    }
                });
        }).wait();
        auto gpuElapsed = cpuSecond() - gpuStart;

        std::cout << "GPU Execution config <<<(" 
                 << grid[0] << ", " << grid[1] << "), ("
                 << block[0] << ", " << block[1] << ")>>> Time: "
                 << gpuElapsed << " sec\n";

        // 验证结果
        q.memcpy(C_from_gpu, C_dev, nBytes).wait();
        checkResult(C_host, C_from_gpu, nxy);

        // 释放资源
        free(A_host, q);
        free(B_host, q);
        free(C_host, q);
        free(C_from_gpu, q);
        free(A_dev, q);
        free(B_dev, q);
        free(C_dev, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}