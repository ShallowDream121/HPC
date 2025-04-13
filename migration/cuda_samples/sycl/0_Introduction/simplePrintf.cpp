#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

class PrintKernel;

// 设备选择函数
device select_max_gflops_device() {
    std::vector<device> devices;
    for (auto& platform : platform::get_platforms()) {
        auto gpu_devices = platform.get_devices(info::device_type::gpu);
        devices.insert(devices.end(), gpu_devices.begin(), gpu_devices.end());
    }
    
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found");
    }

    device max_device;
    uint64_t max_perf = 0;
    for (auto& dev : devices) {
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
        // 选择设备并创建队列
        device dev = select_max_gflops_device();
        queue q(dev);
        
        // 获取设备信息
        std::cout << "Device: " << dev.get_info<info::device::name>() 
                  << "\nCompute units: " << dev.get_info<info::device::max_compute_units>()
                  << "\nMax clock frequency: " << dev.get_info<info::device::max_clock_frequency>() << "MHz\n\n";

        // 定义执行范围
        constexpr int block_x = 2, block_y = 2, block_z = 2;
        constexpr int grid_x = 2, grid_y = 2;
        
        nd_range<3> execution_range(
            range<3>(grid_x, grid_y, 1) * range<3>(block_x, block_y, block_z),
            range<3>(block_x, block_y, block_z)
        );

        // 提交内核
        q.submit([&](handler& h) {
            h.parallel_for<PrintKernel>(execution_range, [=](nd_item<3> item) {
                // 计算全局索引
                auto group_id = item.get_group().get_id();
                auto local_id = item.get_local_id();
                
                // 计算CUDA风格的全局线性ID
                int block_linear_id = group_id[1] * grid_x + group_id[0];
                int thread_linear_id = local_id[2] * (block_x * block_y) 
                                     + local_id[1] * block_x 
                                     + local_id[0];
                
                // 使用SYCL扩展的printf（需要设备支持）
                ext::oneapi::experimental::printf(
                    "[%d, %d]:\t\tValue is:%d\n", 
                    block_linear_id,
                    thread_linear_id,
                    10
                );
            });
        }).wait();
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}