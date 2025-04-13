#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstdint>

using namespace sycl;

// 内核名称定义
class timedReductionKernel;

// 设备选择函数
device select_max_gflops_device() {
    std::vector<device> devices;
    for (auto &platform : platform::get_platforms()) {
        auto platform_devices = platform.get_devices(info::device_type::gpu);
        devices.insert(devices.end(), platform_devices.begin(), platform_devices.end());
    }
    if (devices.empty()) {
        std::cerr << "No GPU devices found." << std::endl;
        exit(EXIT_FAILURE);
    }

    device max_device;
    uint64_t max_perf = 0;
    for (auto &dev : devices) {
        if (!dev.has(aspect::gpu)) continue;

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

int main(int argc, char **argv) {
    constexpr int NUM_BLOCKS = 64;
    constexpr int NUM_THREADS = 256;
    constexpr int SHARED_SIZE = NUM_THREADS * 2;

    try {
        // 选择设备并创建队列
        device dev = select_max_gflops_device();
        queue q(dev, property::queue::enable_profiling{});

        std::cout << "Running on: "
                  << q.get_device().get_info<info::device::name>() << "\n";

        // 分配内存
        float *d_input = malloc_device<float>(NUM_THREADS * 2, q);
        float *d_output = malloc_device<float>(NUM_BLOCKS, q);
        uint64_t *d_timer = malloc_device<uint64_t>(NUM_BLOCKS * 2, q);

        // 初始化主机数据
        std::vector<float> input(NUM_THREADS * 2);
        std::vector<uint64_t> timer(NUM_BLOCKS * 2);
        for (int i = 0; i < NUM_THREADS * 2; ++i) {
            input[i] = static_cast<float>(i);
        }

        // 拷贝数据到设备
        q.memcpy(d_input, input.data(), sizeof(float) * NUM_THREADS * 2).wait();

        // 提交内核
        auto event = q.submit([&](handler& h) {
            h.parallel_for<timedReductionKernel>(
                nd_range<1>(NUM_BLOCKS * NUM_THREADS, NUM_THREADS),
                [=](nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    // 分配共享内存
                    local_accessor<float, 1> shared(SHARED_SIZE, h);

                    const int tid = item.get_local_id(0);
                    const int bid = item.get_group_id(0);
                    const int global_idx = item.get_global_id(0);

                    // 记录开始时间
                    if (tid == 0) {
                        d_timer[bid] = sycl::ext::intel::experimental::clock();
                    }

                    // 加载数据到共享内存
                    if (global_idx < NUM_THREADS * 2) {
                        shared[tid] = d_input[global_idx];
                        shared[tid + NUM_THREADS] = d_input[global_idx + NUM_THREADS];
                    }
                    item.barrier(access::fence_space::local_space);

                    // 归约操作
                    for (int d = NUM_THREADS; d > 0; d /= 2) {
                        if (tid < d) {
                            float f0 = shared[tid];
                            float f1 = shared[tid + d];
                            shared[tid] = (f1 < f0) ? f1 : f0;
                        }
                        item.barrier(access::fence_space::local_space);
                    }

                    // 写回结果
                    if (tid == 0) {
                        d_output[bid] = shared[0];
                        d_timer[bid + NUM_BLOCKS] = sycl::ext::intel::experimental::clock();
                    }
                });
        });

        // 拷贝计时数据回主机
        q.memcpy(timer.data(), d_timer, sizeof(uint64_t) * NUM_BLOCKS * 2).wait();

        // 计算结果
        long double avgElapsedClocks = 0;
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            avgElapsedClocks += (timer[i + NUM_BLOCKS] - timer[i]);
        }
        avgElapsedClocks /= NUM_BLOCKS;

        std::cout << "Average clocks/block = " << avgElapsedClocks << "\n";

        // 释放资源
        free(d_input, q);
        free(d_output, q);
        free(d_timer, q);

    } catch (const exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}