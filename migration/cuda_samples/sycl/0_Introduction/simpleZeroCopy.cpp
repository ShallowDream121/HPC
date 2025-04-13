#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <memory>

using namespace sycl;
constexpr int MEMORY_ALIGNMENT = 4096;
constexpr int nelem = 1048576;
constexpr size_t bytes = nelem * sizeof(float);

// 内核定义
class VectorAddKernel;

// 内存对齐分配辅助函数
template <typename T>
T* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size)) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

int main() {
    try {
        // 创建SYCL队列
        queue q;
        std::cout << "Running on: "
                  << q.get_device().get_info<info::device::name>() << "\n";

        // 分配对齐的主机内存
        float* a_UA = aligned_alloc<float>(MEMORY_ALIGNMENT, bytes + MEMORY_ALIGNMENT);
        float* b_UA = aligned_alloc<float>(MEMORY_ALIGNMENT, bytes + MEMORY_ALIGNMENT);
        float* c_UA = aligned_alloc<float>(MEMORY_ALIGNMENT, bytes + MEMORY_ALIGNMENT);
        
        // 对齐指针
        float* a = reinterpret_cast<float*>(
            (reinterpret_cast<size_t>(a_UA) + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1));
        float* b = reinterpret_cast<float*>(
            (reinterpret_cast<size_t>(b_UA) + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1));
        float* c = reinterpret_cast<float*>(
            (reinterpret_cast<size_t>(c_UA) + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1));

        // 初始化数据
        for (int n = 0; n < nelem; ++n) {
            a[n] = static_cast<float>(rand()) / RAND_MAX;
            b[n] = static_cast<float>(rand()) / RAND_MAX;
        }

        // 分配设备内存
        float* d_a = malloc_device<float>(nelem, q);
        float* d_b = malloc_device<float>(nelem, q);
        float* d_c = malloc_device<float>(nelem, q);

        if (!d_a || !d_b || !d_c) {
            throw std::runtime_error("Failed to allocate device memory");
        }

        // 拷贝数据到设备
        q.memcpy(d_a, a, bytes).wait();
        q.memcpy(d_b, b, bytes).wait();

        // 配置执行参数
        constexpr size_t localSize = 256;
        size_t globalSize = (nelem + localSize - 1) / localSize * localSize;

        // 提交内核
        auto e = q.submit([&](handler& h) {
            h.parallel_for<VectorAddKernel>(
                nd_range<1>{globalSize, localSize},
                [=](nd_item<1> item) {
                    size_t idx = item.get_global_id(0);
                    if (idx < nelem) {
                        d_c[idx] = d_a[idx] + d_b[idx];
                    }
                });
        });
        e.wait();

        // 拷贝结果回主机
        q.memcpy(c, d_c, bytes).wait();

        // 验证结果
        float errorNorm = 0.0f;
        float refNorm = 0.0f;
        for (int n = 0; n < nelem; ++n) {
            float ref = a[n] + b[n];
            float diff = c[n] - ref;
            errorNorm += diff * diff;
            refNorm += ref * ref;
        }
        errorNorm = std::sqrt(errorNorm);
        refNorm = std::sqrt(refNorm);

        // 释放资源
        free(d_a, q);
        free(d_b, q);
        free(d_c, q);
        free(a_UA);
        free(b_UA);
        free(c_UA);

        std::cout << "Error norm: " << errorNorm << " Reference norm: " << refNorm << std::endl;
        return (errorNorm / refNorm < 1.e-6f) ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}