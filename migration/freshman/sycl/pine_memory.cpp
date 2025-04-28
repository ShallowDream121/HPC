#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

constexpr size_t N_ELEM = 1 << 14;  // 16384个元素

// 初始化数据函数
void initialData(float* ip, int size) {
    std::srand(std::time(nullptr));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>(std::rand() & 0xffff) / 1000.0f;
    }
}

// 结果验证函数
void checkResult(float* hostRef, float* gpuRef, int size) {
    constexpr double epsilon = 1.0E-8;
    for (int i = 0; i < size; i++) {
        if (std::abs(hostRef[i] - gpuRef[i]) > epsilon) {
            std::cerr << "Results mismatch at " << i << ": "
                      << hostRef[i] << " vs " << gpuRef[i] << "\n";
            return;
        }
    }
    std::cout << "Check result success!\n";
}

// 主机端向量相加函数（优化为SIMD风格）
void sumArrays(float* a, float* b, float* res, int size) {
    for (int i = 0; i < size; i += 4) {
        res[i]   = a[i]   + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

int main() {
    // 配置参数
    constexpr size_t nByte = N_ELEM * sizeof(float);
    constexpr size_t blockSize = 1024;
    constexpr size_t gridSize = (N_ELEM + blockSize - 1) / blockSize;

    // 分配主机内存
    float *a_h = new float[N_ELEM];
    float *b_h = new float[N_ELEM];
    float *res_h = new float[N_ELEM];
    float *res_from_gpu_h = new float[N_ELEM];

    // 初始化数据
    initialData(a_h, N_ELEM);
    initialData(b_h, N_ELEM);
    std::memset(res_h, 0, nByte);
    std::memset(res_from_gpu_h, 0, nByte);

    try {
        // 创建SYCL队列（自动选择GPU设备）
        sycl::queue q(sycl::gpu_selector_v);
        
        // 打印设备信息
        std::cout << "Running on: "
                  << q.get_device().get_info<sycl::info::device::name>() 
                  << "\n";

        // 分配设备内存（使用USM统一共享内存）
        float *a_d = sycl::malloc_device<float>(N_ELEM, q);
        float *b_d = sycl::malloc_device<float>(N_ELEM, q);
        float *res_d = sycl::malloc_device<float>(N_ELEM, q);

        // 数据拷贝到设备（异步操作）
        auto evt1 = q.memcpy(a_d, a_h, nByte);
        auto evt2 = q.memcpy(b_d, b_h, nByte);
        
        // 等待数据传输完成
        evt1.wait();
        evt2.wait();

        // 执行内核
        q.parallel_for(
            sycl::nd_range<1>{gridSize * blockSize, blockSize},
            [=](sycl::nd_item<1> item) {
                size_t i = item.get_global_id(0);
                if (i < N_ELEM) {
                    res_d[i] = a_d[i] + b_d[i];
                }
            }).wait();

        // 拷贝结果回主机
        q.memcpy(res_from_gpu_h, res_d, nByte).wait();

        // 释放设备内存
        sycl::free(a_d, q);
        sycl::free(b_d, q);
        sycl::free(res_d, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    // 主机端计算结果
    sumArrays(a_h, b_h, res_h, N_ELEM);

    // 验证结果
    checkResult(res_h, res_from_gpu_h, N_ELEM);

    // 释放主机内存
    delete[] a_h;
    delete[] b_h;
    delete[] res_h;
    delete[] res_from_gpu_h;

    return EXIT_SUCCESS;
}