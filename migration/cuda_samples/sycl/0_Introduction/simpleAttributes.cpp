#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace sycl;
constexpr int EXIT_WAIVED = 2;

class KernCacheSegmentTest;

struct DeviceInfo {
    int compute_cores;
    size_t l2_cache_size;
    size_t max_grid_size;
    size_t persisting_l2_max;
};

// Simplified device selection based on maximum compute units
DeviceInfo get_device_info(queue& q) {
    DeviceInfo info;
    auto dev = q.get_device();
    info.compute_cores = dev.get_info<info::device::max_compute_units>();
    info.l2_cache_size = dev.get_info<info::device::global_mem_cache_size>();
    info.max_grid_size = dev.get_info<info::device::max_work_group_size>();
    info.persisting_l2_max = 0; // SYCL doesn't directly expose this property
    return info;
}

int main(int argc, char** argv) {
    try {
        queue q(gpu_selector_v);
        std::cout << "Selected device: " 
                  << q.get_device().get_info<info::device::name>() << "\n";

        DeviceInfo dev_info = get_device_info(q);
        if (dev_info.persisting_l2_max == 0) {
            std::cout << "Waiving execution as device does not support persisting L2 Caching\n";
            return EXIT_WAIVED;
        }

        const int dataSize = (dev_info.l2_cache_size / 4) / sizeof(int);
        const int bigDataSize = (dev_info.l2_cache_size * 4) / sizeof(int);

        // Host allocations
        std::vector<int> dataHost(dataSize);
        std::vector<int> bigDataHost(bigDataSize);

        // Initialize host data
        for (int i = 0; i < bigDataSize; ++i) {
            if (i < dataSize) dataHost[i] = i;
            bigDataHost[bigDataSize - i - 1] = i;
        }

        // Device allocations
        int* dataDevice = malloc_device<int>(dataSize, q);
        int* bigDataDevice = malloc_device<int>(bigDataSize, q);

        // Copy data to device
        q.memcpy(dataDevice, dataHost.data(), dataSize * sizeof(int)).wait();
        q.memcpy(bigDataDevice, bigDataHost.data(), bigDataSize * sizeof(int)).wait();

        // Configure kernel parameters
        constexpr int threadsX = 32;
        constexpr int threadsY = 32;
        const int blocksX = static_cast<int>(dev_info.max_grid_size / threadsX);
        
        range<2> global_range(blocksX * threadsX, threadsY);
        range<2> local_range(threadsX, threadsY);

        // Submit kernel
        auto e = q.submit([&](handler& h) {
            h.parallel_for<KernCacheSegmentTest>(
                nd_range<2>(global_range, local_range),
                [=](nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                    const int row = item.get_global_id(0);
                    const int col = item.get_global_id(1);
                    const int tID = row * item.get_local_range(0) + col;
                    
                    uint32_t psRand = tID;
                    int hit = 0;
                    constexpr int hitCount = 0xAFFFF;

                    while (hit < hitCount) {
                        psRand ^= psRand << 13;
                        psRand ^= psRand >> 17;
                        psRand ^= psRand << 5;

                        int idx = tID - psRand;
                        idx = (idx < 0) ? -idx : idx;

                        if ((tID % 2) == 0) {
                            auto ptr = &dataDevice[psRand % dataSize];
                            atomic<int>(global_ptr<int>(ptr)).fetch_add(
                                dataDevice[idx % dataSize]);
                        } else {
                            auto ptr = &bigDataDevice[psRand % bigDataSize];
                            atomic<int>(global_ptr<int>(ptr)).fetch_add(
                                bigDataDevice[idx % bigDataSize]);
                        }

                        ++hit;
                    }
                });
        });

        e.wait();
        std::cout << "Kernel execution completed.\n";

        // Cleanup
        free(dataDevice, q);
        free(bigDataDevice, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}