#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sycl;

constexpr double GB = 1024.0 * 1024.0 * 1024.0;
constexpr double KB = 1024.0;

void printDeviceInfo(const device& dev, int index) {
    std::cout << "Device " << index << ": \"" 
              << dev.get_info<info::device::name>() << "\"\n";

    // 获取最大工作项尺寸（三维）
    auto maxWorkItemSizes = dev.get_info<info::device::max_work_item_sizes<3>>();
    
    std::cout << "  Max work-item sizes:             ["
              << maxWorkItemSizes[0] << ", "
              << maxWorkItemSizes[1] << ", "
              << maxWorkItemSizes[2] << "]\n";

    // 其他设备信息查询保持不变...
}

int main() {
    try {
        // 获取所有GPU设备
        auto platforms = platform::get_platforms();
        int gpuCount = 0;

        // 统计GPU设备
        for (auto& p : platforms) {
            auto devices = p.get_devices(info::device_type::gpu);
            gpuCount += devices.size();
        }

        if (gpuCount == 0) {
            std::cout << "No SYCL-capable GPU devices found\n";
            return EXIT_FAILURE;
        }

        std::cout << "Detected " << gpuCount << " SYCL-capable GPU device(s)\n";

        // 打印每个GPU信息
        int index = 0;
        for (auto& p : platforms) {
            auto devices = p.get_devices(info::device_type::gpu);
            for (auto& dev : devices) {
                printDeviceInfo(dev, index++);
            }
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}