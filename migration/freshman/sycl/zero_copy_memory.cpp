#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// 计时工具函数
double cpuSecond() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// 数据初始化函数
void initialData(float* ip, int size) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
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

// 主机端向量相加
void sumArrays(float* a, float* b, float* res, int size) {
    for (int i = 0; i < size; i += 4) {
        res[i]   = a[i]   + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

int main(int argc, char** argv) {
    int power = (argc >= 2) ? std::atoi(argv[1]) : 10;
    const int nElem = 1 << power;
    const size_t nByte = nElem * sizeof(float);
    const size_t blockSize = 1024;
    const size_t gridSize = (nElem + blockSize - 1) / blockSize;

    std::cout << "Vector size: " << nElem << "\n";

    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "Running on: " 
                  << q.get_device().get_info<sycl::info::device::name>() 
                  << "\n";

        /********************* 统一内存模式 *********************/
        // 分配统一内存（类似CUDA零拷贝内存）
        float* a_um = sycl::malloc_shared<float>(nElem, q);
        float* b_um = sycl::malloc_shared<float>(nElem, q);
        float* res_um = sycl::malloc_device<float>(nElem, q);
        float* res_from_gpu_um = new float[nElem];

        initialData(a_um, nElem);
        initialData(b_um, nElem);

        // 执行统一内存版本
        auto start = std::chrono::high_resolution_clock::now();
        q.parallel_for(
            sycl::nd_range<1>{gridSize * blockSize, blockSize},
            [=](sycl::nd_item<1> item) {
                int i = item.get_global_id(0);
                if (i < nElem) res_um[i] = a_um[i] + b_um[i];
            }
        ).wait(); // 必须等待内核完成

        q.memcpy(res_from_gpu_um, res_um, nByte).wait();
        auto elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count() * 1000;
        std::cout << "Unified memory elapsed: " << elapsed << " ms\n";
        std::cout << "Execution configuration <<<" 
                 << gridSize << ", " << blockSize << ">>>\n";

        /********************* 设备内存模式 *********************/
        // 分配传统设备内存
        float* a_d = sycl::malloc_device<float>(nElem, q);
        float* b_d = sycl::malloc_device<float>(nElem, q);
        float* res_d = sycl::malloc_device<float>(nElem, q);
        float* res_from_gpu_d = new float[nElem];
        float* a_h = new float[nElem];
        float* b_h = new float[nElem];

        initialData(a_h, nElem);
        initialData(b_h, nElem);

        // 执行设备内存版本
        start = std::chrono::high_resolution_clock::now();
        auto e1 = q.memcpy(a_d, a_h, nByte);
        auto e2 = q.memcpy(b_d, b_h, nByte);
        
        q.parallel_for(
            sycl::nd_range<1>{gridSize * blockSize, blockSize},
            {e1, e2},  // 依赖数据拷贝完成
            [=](sycl::nd_item<1> item) {
                int i = item.get_global_id(0);
                if (i < nElem) res_d[i] = a_d[i] + b_d[i];
            }
        ).wait();

        q.memcpy(res_from_gpu_d, res_d, nByte).wait();
        elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count() * 1000;
        std::cout << "Device memory elapsed: " << elapsed << " ms\n";
        std::cout << "Execution configuration <<<" 
                 << gridSize << ", " << blockSize << ">>>\n";

        /********************* 结果验证 *********************/
        float* res_h = new float[nElem];
        sumArrays(a_um, b_um, res_h, nElem);
        checkResult(res_h, res_from_gpu_um, nElem);

        // 释放资源
        sycl::free(a_um, q);
        sycl::free(b_um, q);
        sycl::free(res_um, q);
        delete[] res_from_gpu_um;

        sycl::free(a_d, q);
        sycl::free(b_d, q);
        sycl::free(res_d, q);
        delete[] a_h;
        delete[] b_h;
        delete[] res_from_gpu_d;
        delete[] res_h;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}