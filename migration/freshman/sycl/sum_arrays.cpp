#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace sycl;

constexpr size_t N = 1 << 14;  // 16384个元素

// 初始化数据（保持与CUDA相同）
void initialData(float* ip, int size) {
    srand(static_cast<unsigned>(time(nullptr)));
    for(int i = 0; i < size; i++) {
        ip[i] = static_cast<float>(rand() & 0xffff) / 1000.0f;
    }
}

// 结果验证函数
void checkResult(float* hostRef, float* gpuRef, int size) {
    constexpr double epsilon = 1.0E-8;
    for(int i = 0; i < size; i++) {
        if(std::abs(hostRef[i] - gpuRef[i]) > epsilon) {
            std::cerr << "Error at index " << i << ": " 
                      << hostRef[i] << " vs " << gpuRef[i] << std::endl;
            return;
        }
    }
    std::cout << "Check result success!\n";
}

// 主机端向量相加（保持循环展开）
void sumArrays(float* a, float* b, float* res, int size) {
    for(int i = 0; i < size; i += 4) {
        res[i] = a[i] + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

int main() {
    try {
        // 设置SYCL队列（自动选择GPU设备）
        queue q(gpu_selector_v);
        std::cout << "Running on: " 
                  << q.get_device().get_info<info::device::name>() << "\n";

        // 主机内存分配
        const size_t bytes = N * sizeof(float);
        float *a_h = malloc_host<float>(N, q);
        float *b_h = malloc_host<float>(N, q);
        float *res_h = malloc_host<float>(N, q);
        float *res_from_gpu_h = malloc_host<float>(N, q);

        // 设备内存分配（USM显式管理）
        float *a_d = malloc_device<float>(N, q);
        float *b_d = malloc_device<float>(N, q);
        float *res_d = malloc_device<float>(N, q);

        // 初始化数据
        initialData(a_h, N);
        initialData(b_h, N);
        std::memset(res_h, 0, bytes);
        std::memset(res_from_gpu_h, 0, bytes);

        // 数据拷贝到设备（异步操作）
        auto evt1 = q.memcpy(a_d, a_h, bytes);
        auto evt2 = q.memcpy(b_d, b_h, bytes);
        q.memcpy(res_d, res_h, bytes);  // 初始化设备结果内存

        // 配置执行参数
        constexpr size_t blockSize = 1024;
        const size_t gridSize = (N + blockSize - 1) / blockSize;
        std::cout << "Execution configuration <<<" 
                  << gridSize << ", " << blockSize << ">>>\n";

        // 提交内核任务
        q.submit([&](handler& h) {
            // 等待数据传输完成
            h.depends_on({evt1, evt2});
            
            // 定义ND范围（1维）
            h.parallel_for(nd_range<1>(gridSize * blockSize, blockSize), 
            [=](nd_item<1> item) {
                const int i = item.get_global_id(0);
                if(i < N) {
                    res_d[i] = a_d[i] + b_d[i];
                }
            });
        });

        // 拷贝结果回主机
        q.memcpy(res_from_gpu_h, res_d, bytes).wait();

        // 主机端计算参考结果
        sumArrays(a_h, b_h, res_h, N);

        // 验证结果
        checkResult(res_h, res_from_gpu_h, N);

        // 释放资源
        free(a_h, q);
        free(b_h, q);
        free(res_h, q);
        free(res_from_gpu_h, q);
        free(a_d, q);
        free(b_d, q);
        free(res_d, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常捕获: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "标准异常捕获: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}