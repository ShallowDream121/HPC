#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace sycl;
using namespace std::chrono;

constexpr int WARP_SIZE = 32; // 根据目标GPU调整，通常为32或64

// 时间测量函数
double cpuSecond() {
    return duration_cast<duration<double>>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

// 统一内核模板
template <int KernelID>
class Kernel;

template <int KernelID>
void launch_kernel(queue &q, float *d_c, int size, int blockSize) {
    range<1> global_size(size);
    range<1> local_size(blockSize);
    
    auto start = cpuSecond();
    q.submit([&](handler &h) {
        h.parallel_for<Kernel<KernelID>>(
            nd_range<1>(global_size, local_size),
            [=](nd_item<1> item) {
                int tid = item.get_global_id(0);
                float a = 0.0f, b = 0.0f;

                if constexpr (KernelID == 0) { // warmup
                    if ((tid / WARP_SIZE) % 2 == 0) {
                        a = 100.0f;
                    } else {
                        b = 200.0f;
                    }
                } 
                else if constexpr (KernelID == 1) { // mathKernel1
                    if (tid % 2 == 0) {
                        a = 100.0f;
                    } else {
                        b = 200.0f;
                    }
                }
                else if constexpr (KernelID == 2) { // mathKernel2
                    if ((tid / WARP_SIZE) % 2 == 0) {
                        a = 100.0f;
                    } else {
                        b = 200.0f;
                    }
                }
                else if constexpr (KernelID == 3) { // mathKernel3
                    bool ipred = (tid % 2 == 0);
                    if (ipred) {
                        a = 100.0f;
                    } else {
                        b = 200.0f;
                    }
                }

                d_c[tid] = a + b;
            });
    }).wait(); // 同步等待内核完成
    double elapsed = cpuSecond() - start;
    
    std::cout << "mathKernel" << KernelID << "<<<" 
              << size/blockSize << "," << blockSize 
              << ">>> elapsed " << elapsed << " sec\n";
}

int main(int argc, char **argv) {
    // 参数处理
    int size = 64;
    int blockSize = 64;
    if (argc > 1) blockSize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);
    size = ((size + blockSize - 1) / blockSize) * blockSize; // 对齐

    try {
        // SYCL队列初始化
        queue q(gpu_selector_v);
        std::cout << "Running on: "
                 << q.get_device().get_info<info::device::name>() << "\n";

        // 内存分配
        float *d_c = malloc_device<float>(size, q);
        float *h_c = malloc_host<float>(size, q);

        // 预热运行
        launch_kernel<0>(q, d_c, size, blockSize);

        // 执行各版本内核
        launch_kernel<1>(q, d_c, size, blockSize);
        launch_kernel<2>(q, d_c, size, blockSize);
        launch_kernel<3>(q, d_c, size, blockSize);

        // 数据回传验证（可选）
        q.memcpy(h_c, d_c, size * sizeof(float)).wait();

        // 释放资源
        free(d_c, q);
        free(h_c, q);

    } catch (const sycl::exception &e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}