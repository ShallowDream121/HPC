#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>

using namespace sycl;
using namespace std::chrono;

constexpr int BLOCK_SIZE = 1024;

// 计时函数
double cpuSecond() {
    return duration_cast<duration<double>>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

// 初始化整数数据
void initialData_int(int* ip, int size) {
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < size; i++) {
        ip[i] = rand() & 0xff;
    }
}

// 递归CPU归约实现
int recursiveReduce(int *data, int size) {
    /* 保持原CUDA实现不变 */
}

// 定义不同归约策略的标签
struct Warmup;
struct Neighbored;
struct NeighboredLess;
struct Interleaved;

template <typename KernelName>
void reduceKernel(queue &q, int *idata_dev, int *odata_dev, 
                int size, int gridSize, int blockSize) {
    try {
        auto start = cpuSecond();
        q.submit([&](handler &h) {
            h.parallel_for<KernelName>(
                nd_range<1>(gridSize * blockSize, blockSize),
                [=](nd_item<1> item) {
                    const int tid = item.get_local_id(0);
                    const int bid = item.get_group_id(0);
                    int *idata = idata_dev + bid * blockSize;

                    if constexpr (std::is_same_v<KernelName, Warmup>) {
                        // Warmup内核实现
                        for (int stride = 1; stride < blockSize; stride *= 2) {
                            if ((tid % (2 * stride)) == 0) {
                                idata[tid] += idata[tid + stride];
                            }
                            item.barrier(access::fence_space::local_space);
                        }
                    }
                    else if constexpr (std::is_same_v<KernelName, Neighbored>) {
                        // Neighbored内核实现
                        for (int stride = 1; stride < blockSize; stride *= 2) {
                            int index = 2 * stride * tid;
                            if (index < blockSize) {
                                idata[index] += idata[index + stride];
                            }
                            item.barrier(access::fence_space::local_space);
                        }
                    }
                    else if constexpr (std::is_same_v<KernelName, Interleaved>) {
                        // Interleaved内核实现
                        for (int stride = blockSize/2; stride > 0; stride >>= 1) {
                            if (tid < stride) {
                                idata[tid] += idata[tid + stride];
                            }
                            item.barrier(access::fence_space::local_space);
                        }
                    }

                    if (tid == 0) {
                        odata_dev[bid] = idata[0];
                    }
                });
        }).wait();
        return cpuSecond() - start;
    } catch (const sycl::exception &e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    constexpr int size = 1 << 24;
    constexpr int nBytes = size * sizeof(int);
    const int blockSize = (argc > 1) ? atoi(argv[1]) : BLOCK_SIZE;
    const int gridSize = (size + blockSize - 1) / blockSize;

    try {
        queue q(gpu_selector_v);
        std::cout << "运行设备: " 
                 << q.get_device().get_info<info::device::name>() << "\n";

        // 分配内存
        int *idata_host = malloc_host<int>(size, q);
        int *odata_host = malloc_host<int>(gridSize, q);
        int *idata_dev = malloc_device<int>(size, q);
        int *odata_dev = malloc_device<int>(gridSize, q);

        // 初始化数据
        initialData_int(idata_host, size);
        q.memcpy(idata_dev, idata_host, nBytes).wait();

        // CPU计算
        int cpu_sum = 0;
        auto cpuStart = cpuSecond();
        for (int i = 0; i < size; i++) cpu_sum += idata_host[i];
        double cpuTime = cpuSecond() - cpuStart;
        std::cout << "CPU 总和: " << cpu_sum << " 耗时: " << cpuTime << "秒\n";

        // Warmup测试
        double elapsed = reduceKernel<Warmup>(q, idata_dev, odata_dev, size, gridSize, blockSize);
        q.memcpy(odata_host, odata_dev, gridSize*sizeof(int)).wait();
        int gpu_sum = std::accumulate(odata_host, odata_host+gridSize, 0);
        std::cout << "GPU Warmup 总和: " << gpu_sum << " 耗时: " << elapsed << "秒\n";

        // Neighbored测试
        q.memcpy(idata_dev, idata_host, nBytes).wait();
        elapsed = reduceKernel<Neighbored>(q, idata_dev, odata_dev, size, gridSize, blockSize);
        q.memcpy(odata_host, odata_dev, gridSize*sizeof(int)).wait();
        gpu_sum = std::accumulate(odata_host, odata_host+gridSize, 0);
        std::cout << "GPU Neighbored 总和: " << gpu_sum << " 耗时: " << elapsed << "秒\n";

        // Interleaved测试
        q.memcpy(idata_dev, idata_host, nBytes).wait();
        elapsed = reduceKernel<Interleaved>(q, idata_dev, odata_dev, size, gridSize, blockSize);
        q.memcpy(odata_host, odata_dev, gridSize*sizeof(int)).wait();
        gpu_sum = std::accumulate(odata_host, odata_host+gridSize, 0);
        std::cout << "GPU Interleaved 总和: " << gpu_sum << " 耗时: " << elapsed << "秒\n";

        // 验证结果
        if (gpu_sum == cpu_sum) {
            std::cout << "测试成功!\n";
        } else {
            std::cout << "测试失败!\n";
        }

        // 释放资源
        free(idata_host, q);
        free(odata_host, q);
        free(idata_dev, q);
        free(odata_dev, q);

    } catch (const sycl::exception &e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}