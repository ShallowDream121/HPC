#include <CL/sycl.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace sycl;
constexpr int WARP_SIZE = 32;

class NormKernel;

// 设备选择函数
device select_max_gflops_device() {
    std::vector<device> devices;
    for (auto &platform : platform::get_platforms()) {
        auto gpu_devices = platform.get_devices(info::device_type::gpu);
        devices.insert(devices.end(), gpu_devices.begin(), gpu_devices.end());
    }
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found");
    }

    device max_device;
    uint64_t max_perf = 0;
    for (auto &dev : devices) {
        auto compute_units = dev.get_info<info::device::max_compute_units>();
        auto clock_freq = dev.get_info<info::device::max_clock_frequency>();
        uint64_t perf = compute_units * clock_freq;
        if (perf > max_perf) {
            max_perf = perf;
            max_device = dev;
        }
    }
    return max_device;
}

int main(int argc, char** argv) {
    try {
        constexpr int size = 10000000;
        constexpr float baseVal = 2.0f;
        constexpr float expected = baseVal / std::sqrt(size * baseVal * baseVal);

        // 初始化设备和队列
        device dev = select_max_gflops_device();
        queue q(dev, property::queue::enable_profiling{});
        std::cout << "Running on: " << dev.get_info<info::device::name>() << "\n";

        // 验证设备支持
        if (!dev.has(aspect::fp64)) {
            std::cout << "Device does not support double precision\n";
            return EXIT_SUCCESS;
        }

        // 分配内存
        float* vecA = malloc_host<float>(size, q);
        float* vecB = malloc_host<float>(size, q);
        double* partial = malloc_device<double>(1, q);

        float* d_vecA = malloc_device<float>(size, q);
        float* d_vecB = malloc_device<float>(size, q);

        // 初始化数据
        for (int i = 0; i < size; ++i) {
            vecA[i] = vecB[i] = baseVal;
        }

        // 拷贝数据到设备
        q.memcpy(d_vecA, vecA, size * sizeof(float)).wait();
        q.memcpy(d_vecB, vecB, size * sizeof(float)).wait();

        // 配置内核参数
        auto max_wg_size = dev.get_info<info::device::max_work_group_size>();
        size_t wg_size = std::min<size_t>(max_wg_size, 256);
        size_t num_wg = dev.get_info<info::device::max_compute_units>() * 2;

        // 执行内核
        auto e = q.submit([&](handler& h) {
            h.parallel_for<NormKernel>(
                nd_range<1>{num_wg * wg_size, wg_size},
                [=](nd_item<1> item) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                    auto sg = item.get_sub_group();
                    const int lid = item.get_local_id(0);
                    const int gid = item.get_global_id(0);
                    
                    // 第一阶段：计算局部点积和
                    double local_sum = 0.0;
                    for (int i = gid; i < size; i += item.get_global_range()[0]) {
                        local_sum += static_cast<double>(d_vecA[i]) * d_vecB[i];
                    }
                    
                    // 子组内归约
                    double group_sum = reduce_over_group(sg, local_sum, plus<>());
                    
                    // 工作组内归约
                    if (sg.get_group_id() == 0) {
                        auto& grp = item.get_group();
                        double* shared = (double*)group_local_memory_for_overwrite<double[32]>(grp).get();
                        shared[sg.get_local_id()] = group_sum;
                        grp.barrier();
                        
                        if (sg.get_local_id() == 0) {
                            double block_sum = 0.0;
                            for (int i = 0; i < grp.get_local_range()[0]/WARP_SIZE; ++i) {
                                block_sum += shared[i];
                            }
                            if (grp.leader()) {
                                ext::oneapi::atomic_ref<double, 
                                    ext::oneapi::memory_order::relaxed,
                                    ext::oneapi::memory_scope::device>(*partial) += block_sum;
                            }
                        }
                    }
                    item.barrier(access::fence_space::local_and_global);

                    // 第二阶段：归一化
                    if (gid == 0) {
                        double total = *partial;
                        double norm = std::sqrt(total);
                        *partial = norm;
                    }
                    item.barrier(access::fence_space::global_space);

                    const double norm = *partial;
                    for (int i = gid; i < size; i += item.get_global_range()[0]) {
                        d_vecA[i] /= norm;
                        d_vecB[i] /= norm;
                    }
                });
        });

        // 拷贝结果回主机
        q.memcpy(vecA, d_vecA, size * sizeof(float), e).wait();

        // 验证结果
        unsigned matches = 0;
        for (int i = 0; i < size; ++i) {
            if (std::fabs(vecA[i] - expected) < 1e-5) ++matches;
        }

        // 释放资源
        free(vecA, q);
        free(vecB, q);
        free(partial, q);
        free(d_vecA, q);
        free(d_vecB, q);

        std::cout << "Result: " << (matches == size ? "PASSED" : "FAILED") 
                  << " (" << matches << "/" << size << ")\n";
        return matches == size ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}