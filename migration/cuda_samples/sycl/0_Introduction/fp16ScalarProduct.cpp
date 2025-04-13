#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <cmath>

using namespace sycl;
constexpr int NUM_BLOCKS = 128;
constexpr int NUM_THREADS = 128;

class ScalarProductNative;
class ScalarProductIntrinsics;

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

// 生成随机半精度数据
void generate_input(half2 *data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 4.0f);
    
    for (size_t i = 0; i < size; ++i) {
        data[i].x() = half(dist(gen));
        data[i].y() = half(dist(gen));
    }
}

// 归约内核函数
template <typename T>
void reduce_kernel(T *shArray, nd_item<1> &item) {
    auto local = item.get_local_id(0);
    if (local < 64) shArray[local] += shArray[local + 64];
    item.barrier(access::fence_space::local_space);
    if (local < 32) shArray[local] += shArray[local + 32];
    item.barrier(access::fence_space::local_space);
    if (local < 16) shArray[local] += shArray[local + 16];
    item.barrier(access::fence_space::local_space);
    if (local < 8) shArray[local] += shArray[local + 8];
    item.barrier(access::fence_space::local_space);
    if (local < 4) shArray[local] += shArray[local + 4];
    item.barrier(access::fence_space::local_space);
    if (local < 2) shArray[local] += shArray[local + 2];
    item.barrier(access::fence_space::local_space);
    if (local < 1) shArray[local] += shArray[local + 1];
    item.barrier(access::fence_space::local_space);
}

int main() {
    try {
        // 初始化设备和队列
        device dev = select_max_gflops_device();
        queue q(dev);
        std::cout << "Running on: " << dev.get_info<info::device::name>() << "\n";

        // 验证设备支持半精度
        if (!dev.has(aspect::fp16)) {
            std::cout << "Device does not support half precision operations\n";
            return EXIT_SUCCESS;
        }

        // 分配内存
        const size_t size = NUM_BLOCKS * NUM_THREADS * 16;
        half2 *a = malloc_host<half2>(size, q);
        half2 *b = malloc_host<half2>(size, q);
        float *results = malloc_host<float>(NUM_BLOCKS, q);

        half2 *d_a = malloc_device<half2>(size, q);
        half2 *d_b = malloc_device<half2>(size, q);
        float *d_results = malloc_device<float>(NUM_BLOCKS, q);

        // 生成数据
        generate_input(a, size);
        generate_input(b, size);

        // 拷贝数据到设备
        q.memcpy(d_a, a, size * sizeof(half2)).wait();
        q.memcpy(d_b, b, size * sizeof(half2)).wait();

        // 定义内核范围
        nd_range<1> kernel_range(NUM_BLOCKS * NUM_THREADS, NUM_THREADS);

        // 原生运算符版本
        auto native_event = q.submit([&](handler &h) {
            h.parallel_for<ScalarProductNative>(kernel_range, [=](nd_item<1> item) {
                local_accessor<half2, 1> sh_array(NUM_THREADS, h);
                
                const int gid = item.get_global_id(0);
                const int lid = item.get_local_id(0);
                const int stride = item.get_global_range()[0];
                
                half2 sum(0.0f, 0.0f);
                for (int i = gid; i < size; i += stride) {
                    sum += d_a[i] * d_b[i];
                }
                
                sh_array[lid] = sum;
                item.barrier(access::fence_space::local_space);
                reduce_kernel(sh_array.get_pointer(), item);

                if (lid == 0) {
                    auto res = sh_array[0];
                    d_results[item.get_group(0)] = static_cast<float>(res.x()) + 
                                                 static_cast<float>(res.y());
                }
            });
        });

        // 拷贝结果回主机
        q.memcpy(results, d_results, NUM_BLOCKS * sizeof(float), native_event).wait();
        float native_sum = 0.0f;
        for (int i = 0; i < NUM_BLOCKS; ++i) native_sum += results[i];

        // Intrinsic版本（使用SYCL内置函数）
        auto intrinsic_event = q.submit([&](handler &h) {
            h.parallel_for<ScalarProductIntrinsics>(kernel_range, [=](nd_item<1> item) {
                local_accessor<half2, 1> sh_array(NUM_THREADS, h);
                
                const int gid = item.get_global_id(0);
                const int lid = item.get_local_id(0);
                const int stride = item.get_global_range()[0];
                
                half2 sum(0.0f, 0.0f);
                for (int i = gid; i < size; i += stride) {
                    sum = sycl::fma(d_a[i], d_b[i], sum);
                }
                
                sh_array[lid] = sum;
                item.barrier(access::fence_space::local_space);
                reduce_kernel(sh_array.get_pointer(), item);

                if (lid == 0) {
                    auto res = sh_array[0];
                    d_results[item.get_group(0)] = static_cast<float>(res.x()) + 
                                                 static_cast<float>(res.y());
                }
            });
        });

        // 拷贝结果回主机
        q.memcpy(results, d_results, NUM_BLOCKS * sizeof(float), intrinsic_event).wait();
        float intrinsic_sum = 0.0f;
        for (int i = 0; i < NUM_BLOCKS; ++i) intrinsic_sum += results[i];

        // 验证结果
        std::cout << "Native sum: " << native_sum << "\n"
                  << "Intrinsic sum: " << intrinsic_sum << "\n"
                  << "Difference: " << std::abs(native_sum - intrinsic_sum) << "\n";

        // 释放内存
        free(a, q);
        free(b, q);
        free(results, q);
        free(d_a, q);
        free(d_b, q);
        free(d_results, q);

    } catch (const sycl::exception &e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}