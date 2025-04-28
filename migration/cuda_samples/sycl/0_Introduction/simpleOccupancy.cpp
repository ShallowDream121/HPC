#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;
constexpr int manualBlockSize = 32;

class SquareKernel;

// 设备选择函数
device select_device() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) throw std::runtime_error("No GPU devices found");
    return devices[0]; // 选择第一个设备
}

// 性能报告函数
double report_potential_occupancy(queue& q, int blockSize) {
    auto dev = q.get_device();
    size_t max_work_group = dev.get_info<info::device::max_work_group_size>();
    auto max_cu = dev.get_info<info::device::max_compute_units>();
    
    // 估算占用率
    int active_warps = (max_cu * blockSize) / dev.get_info<info::device::native_vector_width_int>();
    int max_warps = (max_work_group * max_cu) / dev.get_info<info::device::native_vector_width_int>();
    
    return static_cast<double>(active_warps) / max_warps;
}

int launch_config(queue& q, int64_t* d_array, int arrayCount, bool automatic) {
    int blockSize;
    int gridSize;
    double occupancy;

    auto dev = q.get_device();
    auto max_wg_size = dev.get_info<info::device::max_work_group_size>();

    // 自动配置逻辑
    if (automatic) {
        // 简化版自动配置策略：选择设备支持的最大工作组大小
        blockSize = max_wg_size;
        std::cout << "Suggested block size: " << blockSize << "\n"
                  << "Minimum grid size: " 
                  << (arrayCount + blockSize - 1) / blockSize << std::endl;
    } else {
        blockSize = manualBlockSize;
    }

    gridSize = (arrayCount + blockSize - 1) / blockSize;

    // 执行内核并计时
    auto start = std::chrono::high_resolution_clock::now();
    
    auto event = q.submit([&](handler& h) {
        h.parallel_for<SquareKernel>(
            nd_range<1>(gridSize * blockSize, blockSize),
            [=](nd_item<1> item) {
                int idx = item.get_global_id(0);
                if (idx < arrayCount) {
                    d_array[idx] *= d_array[idx];
                }
            });
    });
    
    event.wait();
    auto end = std::chrono::high_resolution_clock::now();

    // 计算占用率
    occupancy = report_potential_occupancy(q, blockSize);
    std::cout << "Potential occupancy: " << occupancy * 100 << "%\n";

    // 计算耗时
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed.count() / 1000.0 << "ms\n";

    return 0;
}

int test(bool automatic, int count = 100000) {
    try {
        queue q(select_device());
        int64_t* array = new int64_t[count];  // from int to int64_t
        int64_t* d_array = malloc_device<int64_t>(count, q);

        // 初始化数据
        for (int i = 0; i < count; ++i) array[i] = i;
        q.memcpy(d_array, array, count * sizeof(int64_t)).wait();

        // 执行内核
        int status = launch_config(q, d_array, count, automatic);

        // 验证结果
        q.memcpy(array, d_array, count * sizeof(int64_t)).wait();
        free(d_array, q);

        for (int i = 0; i < count; ++i) {
            if (array[i] != i * i) {
                std::cerr << "Error at " << i << ": " << array[i] 
                          << " vs " << i*i << "\n";
                delete[] array;
                return 1;
            }
        }

        delete[] array;
        return status;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }
}

int main() {
    std::cout << "Starting Simple Occupancy\n\n";

    std::cout << "[ Manual configuration with " << manualBlockSize 
              << " threads per block ]\n";
    if (test(false)) return -1;

    std::cout << "\n[ Automatic configuration ]\n";
    if (test(true)) return -1;

    std::cout << "\nTest PASSED\n";
    return 0;
}