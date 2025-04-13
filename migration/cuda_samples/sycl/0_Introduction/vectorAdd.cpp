#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace sycl;

constexpr int numElements = 50000;
constexpr size_t size = numElements * sizeof(float);

class vectorAddKernel; // 内核名称声明

int main() {
    std::cout << "[Vector addition of " << numElements << " elements]" << std::endl;

    try {
        // 创建SYCL队列（自动选择设备）
        queue q;
        std::cout << "Running on: "
                  << q.get_device().get_info<info::device::name>() << "\n";

        // 分配主机内存并初始化
        std::vector<float> h_A(numElements);
        std::vector<float> h_B(numElements);
        std::vector<float> h_C(numElements);
        
        for (int i = 0; i < numElements; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // 使用USM分配设备内存
        float* d_A = malloc_device<float>(numElements, q);
        float* d_B = malloc_device<float>(numElements, q);
        float* d_C = malloc_device<float>(numElements, q);

        if (!d_A || !d_B || !d_C) {
            std::cerr << "Failed to allocate device memory!" << std::endl;
            return EXIT_FAILURE;
        }

        // 拷贝数据到设备
        std::cout << "Copy input data to device" << std::endl;
        q.memcpy(d_A, h_A.data(), size).wait();
        q.memcpy(d_B, h_B.data(), size).wait();

        // 配置执行参数
        constexpr int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        std::cout << "SYCL kernel launch with " << blocksPerGrid 
                  << " work-groups of " << threadsPerBlock << " threads\n";

        // 提交内核任务
        auto e = q.submit([&](handler& h) {
            h.parallel_for<vectorAddKernel>(
                nd_range<1>{blocksPerGrid * threadsPerBlock, threadsPerBlock},
                [=](nd_item<1> item) {
                    int i = item.get_global_id(0);
                    if (i < numElements) {
                        d_C[i] = d_A[i] + d_B[i] + 0.0f;
                    }
                });
        });

        // 等待内核完成并拷贝结果回主机
        std::cout << "Copy output data to host" << std::endl;
        q.memcpy(h_C.data(), d_C, size).wait();

        // 验证结果
        bool error = false;
        for (int i = 0; i < numElements; ++i) {
            if (std::fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
                std::cerr << "Result mismatch at index " << i << ": "
                          << h_C[i] << " vs " << h_A[i]+h_B[i] << std::endl;
                error = true;
                break;
            }
        }

        // 释放资源
        free(d_A, q);
        free(d_B, q);
        free(d_C, q);

        if (!error) {
            std::cout << "Test PASSED\nDone" << std::endl;
            return EXIT_SUCCESS;
        }
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cerr << "Test FAILED" << std::endl;
    return EXIT_FAILURE;
}