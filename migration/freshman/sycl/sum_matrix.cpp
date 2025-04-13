#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace sycl;
constexpr int DIMX = 32;
constexpr int DIMY = 32;

double cpuSecond() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

void initialData(float* ip, int size) {
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>(rand() & 0xffff) / 1000.0f;
    }
}

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

void sumMatrix2D_CPU(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            MatC[i + j * nx] = MatA[i + j * nx] + MatB[i + j * nx];
        }
    }
}

class MatrixAdd;

template <int dim>
void runKernel(queue& q, float* A_dev, float* B_dev, float* C_dev, 
              int nx, int ny, int nxy) {
    try {
        constexpr int blockSize = DIMX;
        range<dim> globalRange, localRange;

        if constexpr (dim == 2) {
            localRange = range<2>(DIMX, DIMY);
            globalRange = range<2>(
                (nx + DIMX - 1) / DIMX * DIMX,
                (ny + DIMY - 1) / DIMY * DIMY
            );
        } else if constexpr (dim == 1) {
            localRange = range<1>(blockSize);
            globalRange = range<1>((nxy + blockSize - 1) / blockSize * blockSize);
        }

        auto start = cpuSecond();
        q.submit([&](handler& h) {
            h.parallel_for<MatrixAdd>(
                nd_range<dim>(globalRange, localRange),
                [=](nd_item<dim> item) {
                    int ix, iy, idx;
                    if constexpr (dim == 2) {
                        ix = item.get_global_id(0);
                        iy = item.get_global_id(1);
                        idx = ix + iy * nx;
                    } else {
                        int tid = item.get_global_id(0);
                        ix = tid % nx;
                        iy = tid / nx;
                        idx = tid;
                    }

                    if (ix < nx && iy < ny) {
                        C_dev[idx] = A_dev[idx] + B_dev[idx];
                    }
                });
        }).wait();
        auto elapsed = cpuSecond() - start;

        std::cout << "GPU Execution configuration: ["
                  << globalRange << "] ["
                  << localRange << "] Time: "
                  << elapsed << " sec\n";
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        exit(1);
    }
}

int main() {
    constexpr int nx = 1 << 12;
    constexpr int ny = 1 << 12;
    constexpr int nxy = nx * ny;
    const size_t nBytes = nxy * sizeof(float);

    // 初始化数据
    float* A_host = new float[nxy];
    float* B_host = new float[nxy];
    float* C_host = new float[nxy];
    float* C_from_gpu = new float[nxy];
    
    initialData(A_host, nxy);
    initialData(B_host, nxy);
    std::memset(C_host, 0, nBytes);
    std::memset(C_from_gpu, 0, nBytes);

    try {
        queue q(gpu_selector_v);
        std::cout << "Running on: " 
                 << q.get_device().get_info<info::device::name>() << "\n";

        // USM设备内存分配
        float* A_dev = malloc_device<float>(nxy, q);
        float* B_dev = malloc_device<float>(nxy, q);
        float* C_dev = malloc_device<float>(nxy, q);

        // 数据拷贝到设备
        q.memcpy(A_dev, A_host, nBytes).wait();
        q.memcpy(B_dev, B_host, nBytes).wait();

        // CPU计算
        auto cpuStart = cpuSecond();
        sumMatrix2D_CPU(A_host, B_host, C_host, nx, ny);
        auto cpuElapsed = cpuSecond() - cpuStart;
        std::cout << "CPU Execution Time: " << cpuElapsed << " sec\n";

        // 配置1: 2D工作组
        std::cout << "\nCase 1: 2D工作组\n";
        runKernel<2>(q, A_dev, B_dev, C_dev, nx, ny, nxy);
        q.memcpy(C_from_gpu, C_dev, nBytes).wait();
        checkResult(C_host, C_from_gpu, nxy);

        // 配置2: 1D工作组
        std::cout << "\nCase 2: 1D工作组\n";
        runKernel<1>(q, A_dev, B_dev, C_dev, nx*ny, 1, nxy);
        q.memcpy(C_from_gpu, C_dev, nBytes).wait();
        checkResult(C_host, C_from_gpu, nxy);

        // 配置3: 混合维度
        std::cout << "\nCase 3: 混合维度\n";
        constexpr int blockSize = 32;
        const int gridX = (nx + blockSize - 1) / blockSize;
        const range<2> globalRange(gridX * blockSize, ny);
        const range<2> localRange(blockSize, 1);
        
        auto start = cpuSecond();
        q.submit([&](handler& h) {
            h.parallel_for<MatrixAdd>(
                nd_range<2>(globalRange, localRange),
                [=](nd_item<2> item) {
                    int ix = item.get_global_id(0);
                    int iy = item.get_global_id(1);
                    int idx = ix + iy * nx;
                    if (ix < nx && iy < ny) {
                        C_dev[idx] = A_dev[idx] + B_dev[idx];
                    }
                });
        }).wait();
        auto elapsed = cpuSecond() - start;
        
        std::cout << "GPU Execution configuration: ["
                  << globalRange << "] ["
                  << localRange << "] Time: "
                  << elapsed << " sec\n";
        q.memcpy(C_from_gpu, C_dev, nBytes).wait();
        checkResult(C_host, C_from_gpu, nxy);

        // 释放资源
        free(A_dev, q);
        free(B_dev, q);
        free(C_dev, q);
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return 1;
    }

    delete[] A_host;
    delete[] B_host;
    delete[] C_host;
    delete[] C_from_gpu;

    return 0;
}