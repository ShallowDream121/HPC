#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;
constexpr int NSTREAMS = 32;
constexpr float KERNEL_TIME = 10.0f;

class KernelA;
class KernelB;
class SumKernel;

device select_max_perf_device() {
    std::vector<device> devices;
    for (auto& platform : platform::get_platforms()) {
        auto gpus = platform.get_devices(info::device_type::gpu);
        devices.insert(devices.end(), gpus.begin(), gpus.end());
    }
    if (devices.empty()) throw std::runtime_error("No GPU devices found");

    device max_dev;
    uint64_t max_perf = 0;
    for (auto& dev : devices) {
        auto cu = dev.get_info<info::device::max_compute_units>();
        auto freq = dev.get_info<info::device::max_clock_frequency>();
        uint64_t perf = cu * freq;
        if (perf > max_perf) {
            max_perf = perf;
            max_dev = dev;
        }
    }
    return max_dev;
}

int main() {
    try {
        // 设备选择
        device dev = select_max_perf_device();
        queue main_q(dev, property::queue::enable_profiling{});
        
        // 显示设备信息
        std::cout << "Running on: " << dev.get_info<info::device::name>() << "\n";
        std::cout << "Compute units: " 
                 << dev.get_info<info::device::max_compute_units>() << "\n";
        std::cout << "Max clock: " 
                 << dev.get_info<info::device::max_clock_frequency>() << "MHz\n";

        // 创建多个队列模拟CUDA流
        std::vector<queue> streams;
        for (int i = 0; i < NSTREAMS; ++i) {
            streams.emplace_back(dev, property::queue::enable_profiling{});
        }

        // 分配内存
        uint64_t* a = malloc_host<uint64_t>(1, main_q);
        uint64_t* d_a = malloc_device<uint64_t>(2 * NSTREAMS, main_q);

        // 初始化事件
        auto start = main_q.submit([&](handler& h) { h.host_task([=]{}); });
        auto stop = main_q.submit([&](handler& h) { h.host_task([=]{}); });

        // 计算时钟周期数
        uint64_t time_clocks = static_cast<uint64_t>(KERNEL_TIME * 1e6); // 模拟值
        uint64_t total_clocks = 2 * NSTREAMS * time_clocks;

        // 提交内核任务
        std::vector<event> events;
        for (int i = 0; i < NSTREAMS; ++i) {
            auto& q = streams[i];
            
            // Kernel A
            events.push_back(q.submit([&](handler& h) {
                h.single_task<KernelA>([=]() {
                    uint64_t count = 0;
                    auto start = sycl::ext::intel::experimental::clock();
                    while (count < time_clocks) {
                        auto end = sycl::ext::intel::experimental::clock();
                        count = end - start;
                    }
                    d_a[2*i] = count;
                });
            }));

            // Kernel B
            events.push_back(q.submit([&](handler& h) {
                h.single_task<KernelB>([=]() {
                    uint64_t count = 0;
                    auto start = sycl::ext::intel::experimental::clock();
                    while (count < time_clocks) {
                        auto end = sycl::ext::intel::experimental::clock();
                        count = end - start;
                    }
                    d_a[2*i+1] = count;
                });
            }));
        }

        // 等待所有任务完成
        for (auto& e : events) e.wait();

        // 归约求和
        auto sum_event = main_q.submit([&](handler& h) {
            h.parallel_for<SumKernel>(nd_range<1>(32, 32), [=](nd_item<1> item) {
                local_accessor<uint64_t, 1> smem(32, h);
                
                uint64_t sum = 0;
                for (int i = item.get_local_id(); i < 2*NSTREAMS; i += 32) {
                    sum += d_a[i];
                }
                smem[item.get_local_id()] = sum;
                item.barrier(access::fence_space::local_space);

                for (int i = 16; i > 0; i >>= 1) {
                    if (item.get_local_id() < i) {
                        smem[item.get_local_id()] += smem[item.get_local_id() + i];
                    }
                    item.barrier(access::fence_space::local_space);
                }

                if (item.get_local_id() == 0) {
                    d_a[0] = smem[0];
                }
            });
        });

        // 拷贝结果
        main_q.memcpy(a, d_a, sizeof(uint64_t), sum_event).wait();

        // 计算执行时间
        auto start_time = start.get_profiling_info<info::event_profiling::command_start>();
        auto end_time = stop.get_profiling_info<info::event_profiling::command_end>();
        double elapsed = (end_time - start_time) / 1e9;

        // 输出结果
        std::cout << "Expected serial time: " << (NSTREAMS+1)*KERNEL_TIME << "s\n"
                  << "Expected concurrent time: " << 2*KERNEL_TIME << "s\n"
                  << "Measured time: " << elapsed << "s\n"
                  << "Total clocks: " << *a << "/" << total_clocks << "\n"
                  << "Test " << (*a >= total_clocks ? "PASSED" : "FAILED") << "\n";

        // 释放资源
        free(a, main_q);
        free(d_a, main_q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}