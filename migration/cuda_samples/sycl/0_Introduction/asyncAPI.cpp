#include <CL/sycl.hpp>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace sycl;

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

class increment_kernel;

bool correct_output(int *data, const int n, const int x) {
    for (int i = 0; i < n; i++) {
        if (data[i] != x) {
            std::cout << "Error! data[" << i << "] = " << data[i] 
                      << ", ref = " << x << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    constexpr int n = 16 * 1024 * 1024;
    constexpr int value = 26;
    constexpr size_t nbytes = n * sizeof(int);
    constexpr size_t threads_per_block = 512;

    try {
        device dev = select_max_gflops_device();
        queue q(dev, property::queue::enable_profiling{});

        std::cout << "Running on: "
                  << q.get_device().get_info<info::device::name>() << "\n";

        // 分配内存
        int *a = malloc_host<int>(n, q);
        int *d_a = malloc_device<int>(n, q);
        
        // 初始化主机内存
        memset(a, 0, nbytes);

        // 创建事件记录时间点
        event start_event, end_event;
        unsigned long int counter = 0;

        // 提交异步任务链
        start_event = q.submit([&](handler& h) {
            h.memcpy(d_a, a, nbytes);
        });

        size_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
        auto kernel_event = q.submit([&](handler& h) {
            h.depends_on(start_event);
            h.parallel_for<increment_kernel>(
                nd_range<1>{range<1>(num_blocks * threads_per_block), 
                            range<1>(threads_per_block)},
                [=](nd_item<1> item) {
                    size_t idx = item.get_global_linear_id();
                    if (idx < n) {
                        d_a[idx] += value;
                    }
                });
        });

        end_event = q.submit([&](handler& h) {
            h.depends_on(kernel_event);
            h.memcpy(a, d_a, nbytes);
        });

        // 轮询等待完成
        while (end_event.get_info<info::event::command_execution_status>() 
               != info::event_command_status::complete) {
            counter++;
        }

        // 获取执行时间
        uint64_t start_time = start_event.get_profiling_info<
            info::event_profiling::command_start>();
        uint64_t end_time = end_event.get_profiling_info<
            info::event_profiling::command_end>();
        double gpu_time = (end_time - start_time) / 1e6;

        std::cout << "GPU execution time: " << gpu_time << " ms\n";
        std::cout << "CPU polled " << counter << " times\n";

        // 验证结果
        bool result = correct_output(a, n, value);

        // 释放资源
        free(a, q);
        free(d_a, q);

        return result ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}