#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

using namespace sycl;

constexpr int B = 3; // 加法常数
constexpr size_t BLOCK_SIZE = 128;

class AddKernel;

// 验证结果函数
bool verify_result(int* data, size_t n, int b) {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] != static_cast<int>(i) + b) {
            std::cerr << "Mismatch at index " << i << ": " 
                      << data[i] << " vs " << i + b << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << argv[0] << " Starting...\n\n";

    // 获取所有GPU设备
    std::vector<device> gpus;
    for (auto& platform : platform::get_platforms()) {
        auto devices = platform.get_devices(info::device_type::gpu);
        gpus.insert(gpus.end(), devices.begin(), devices.end());
    }

    if (gpus.empty()) {
        std::cerr << "No SYCL GPU devices found!" << std::endl;
        return EXIT_FAILURE;
    }

    const size_t num_gpus = gpus.size();
    const size_t N = num_gpus * 8192;
    const size_t N_per_gpu = N / num_gpus;
    const size_t bytes = N * sizeof(int);
    const size_t bytes_per_gpu = N_per_gpu * sizeof(int);

    std::cout << "Number of host CPUs:\t" << std::thread::hardware_concurrency() << "\n";
    std::cout << "Number of SYCL devices:\t" << num_gpus << "\n";
    for (size_t i = 0; i < num_gpus; ++i) {
        std::cout << "   " << i << ": " 
                  << gpus[i].get_info<info::device::name>() << "\n";
    }
    std::cout << "---------------------------\n";

    // 分配并初始化主机内存
    int* host_data = new int[N];
    for (size_t i = 0; i < N; ++i) {
        host_data[i] = i;
    }

    // 创建线程处理每个GPU
    std::vector<std::thread> threads;
    for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        threads.emplace_back([&, gpu_id]() {
            try {
                // 为每个GPU创建独立队列
                queue q(gpus[gpu_id]);
                
                // 计算数据偏移
                const size_t offset = gpu_id * N_per_gpu;
                int* slice = host_data + offset;

                // 分配设备内存
                int* device_data = malloc_device<int>(N_per_gpu, q);
                
                // 拷贝数据到设备
                q.memcpy(device_data, slice, bytes_per_gpu).wait();

                // 配置执行范围
                const size_t global_size = N_per_gpu;
                const size_t wg_size = BLOCK_SIZE;
                const size_t n_blocks = (global_size + wg_size - 1) / wg_size;

                // 提交内核
                q.submit([&](handler& h) {
                    h.parallel_for<AddKernel>(
                        nd_range<1>{n_blocks * wg_size, wg_size},
                        [=](nd_item<1> item) {
                            const size_t idx = item.get_global_id(0);
                            if (idx < global_size) {
                                device_data[idx] += B;
                            }
                        });
                }).wait();

                // 拷贝结果回主机
                q.memcpy(slice, device_data, bytes_per_gpu).wait();
                
                // 释放设备内存
                free(device_data, q);

                std::cout << "GPU " << gpu_id << " completed\n";
            } catch (const sycl::exception& e) {
                std::cerr << "SYCL Exception on GPU " << gpu_id << ": "
                          << e.what() << std::endl;
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "---------------------------\n";

    // 验证结果
    const bool success = verify_result(host_data, N, B);
    delete[] host_data;

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}