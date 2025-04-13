#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace sycl;

constexpr auto KB = 1024ull;
constexpr auto MB = KB * KB;

std::string get_device_type(const device& dev) {
    if (dev.is_gpu()) return "GPU";
    if (dev.is_cpu()) return "CPU";
    if (dev.is_accelerator()) return "Accelerator";
    return "Unknown";
}

void print_device_info(const device& dev, int index) {
    std::cout << "\nDevice " << index << ": \"" 
              << dev.get_info<info::device::name>() << "\"\n";
    
    std::cout << "  Device Type:                  " 
              << get_device_type(dev) << "\n";
    
    std::cout << "  Vendor:                       "
              << dev.get_info<info::device::vendor>() << "\n";
    
    std::cout << "  Driver Version:               "
              << dev.get_info<info::device::driver_version>() << "\n";
    
    std::cout << "  Global Memory Size:           " 
              << dev.get_info<info::device::global_mem_size>() / MB << " MB\n";
    
    std::cout << "  Local Memory Size:            "
              << dev.get_info<info::device::local_mem_size>() / KB << " KB\n";
    
    std::cout << "  Max Compute Units:            "
              << dev.get_info<info::device::max_compute_units>() << "\n";
    
    std::cout << "  Max Work Group Size:          "
              << dev.get_info<info::device::max_work_group_size>() << "\n";
    
    auto max_work_item = dev.get_info<info::device::max_work_item_sizes>();
    std::cout << "  Max Work Item Sizes:          ["
              << max_work_item[0] << ", " 
              << max_work_item[1] << ", "
              << max_work_item[2] << "]\n";
    
    std::cout << "  Max Clock Frequency:          "
              << dev.get_info<info::device::max_clock_frequency>() << " MHz\n";
    
    std::cout << "  Device supports USM:          "
              << (dev.has(aspect::usm_device_allocations) ? "Yes" : "No") << "\n";
    
    std::cout << "  Preferred Vector Width (int): "
              << dev.get_info<info::device::preferred_vector_width_int>() << "\n";
    
    std::cout << "  Max Sub-Group Size:           "
              << dev.get_info<info::device::max_sub_group_size>() << "\n";
}

int main() {
    try {
        std::vector<device> devices;
        for (auto& platform : platform::get_platforms()) {
            auto platform_devices = platform.get_devices();
            devices.insert(devices.end(), 
                         platform_devices.begin(), 
                         platform_devices.end());
        }

        std::cout << "SYCL Device Query\n\n";
        std::cout << "Detected " << devices.size() 
                  << " compute devices\n";

        int gpu_count = 0;
        for (const auto& dev : devices) {
            if (dev.is_gpu()) gpu_count++;
        }
        std::cout << "Found " << gpu_count << " GPU devices\n";

        int index = 0;
        for (const auto& dev : devices) {
            print_device_info(dev, index++);
        }

        std::cout << "\nAdditional System Info:\n";
        std::cout << "  SYCL Version:       " 
                  << __SYCL_COMPILER_VERSION << "\n";
        std::cout << "  SYCL Implementation: "
                  << (devices.empty() ? "Unknown" 
                      : devices[0].get_platform().get_info<info::platform::name>()) 
                  << "\n";

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}