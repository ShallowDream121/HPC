#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace sycl;
using namespace std::chrono;

constexpr int BLOCK_SIZE = 512;

double cpuSecond() {
    return duration_cast<duration<double>>(
        high_resolution_clock::now().time_since_epoch()
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

void sumArrays(float* a, float* b, float* res, int size) {
    // 保持原有的循环展开优化
    for (int i = 0; i < size; i += 4) {
        res[i] = a[i] + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

class VectorAdd;

int main(int argc, char** argv) {
    constexpr int nElem = 1 << 24;
    const size_t nByte = sizeof(float) * nElem;
    std::cout << "Vector size: " << nElem << std::endl;

    try {
        // 初始化SYCL队列
        queue q(gpu_selector_v);
        std::cout << "Running on: " 
                 << q.get_device().get_info<info::device::name>() << "\n";

        // 分配主机内存
        float* a_h = malloc_host<float>(nElem, q);
        float* b_h = malloc_host<float>(nElem, q);
        float* res_h = malloc_host<float>(nElem, q);
        float* res_from_gpu_h = malloc_host<float>(nElem, q);

        // 初始化数据
        initialData(a_h, nElem);
        initialData(b_h, nElem);
        std::memset(res_h, 0, nByte);
        std::memset(res_from_gpu_h, 0, nByte);

        // 分配设备内存
        float* a_d = malloc_device<float>(nElem, q);
        float* b_d = malloc_device<float>(nElem, q);
        float* res_d = malloc_device<float>(nElem, q);

        // 数据传输到设备
        auto evt1 = q.memcpy(a_d, a_h, nByte);
        auto evt2 = q.memcpy(b_d, b_h, nByte);
        q.memcpy(res_d, res_h, nByte);  // 初始化设备内存

        // 配置执行参数
        const int gridSize = (nElem + BLOCK_SIZE - 1) / BLOCK_SIZE;
        nd_range<1> executionRange(nElem, BLOCK_SIZE);

        // 执行内核并计时
        double iStart = cpuSecond();
        q.submit([&](handler& h) {
            h.depends_on({evt1, evt2});  // 等待数据传输完成
            h.parallel_for<VectorAdd>(
                executionRange,
                [=](nd_item<1> item) {
                    const int i = item.get_global_id(0);
                    if (i < nElem) {
                        res_d[i] = a_d[i] + b_d[i];
                    }
                });
        }).wait();  // 等待内核完成
        
        // 回传结果
        q.memcpy(res_from_gpu_h, res_d, nByte).wait();
        double iElaps = cpuSecond() - iStart;

        std::cout << "Execution configuration<<<" << gridSize << ", " 
                  << BLOCK_SIZE << ">>> Time elapsed " << iElaps << " sec\n";

        // 主机端计算参考结果
        sumArrays(a_h, b_h, res_h, nElem);

        // 验证结果
        checkResult(res_h, res_from_gpu_h, nElem);

        // 释放资源
        free(a_h, q);
        free(b_h, q);
        free(res_h, q);
        free(res_from_gpu_h, q);
        free(a_d, q);
        free(b_d, q);
        free(res_d, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}