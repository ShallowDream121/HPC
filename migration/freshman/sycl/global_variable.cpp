#include <sycl/sycl.hpp>
#include <cstdio>
#include <iostream>

int main() {
    constexpr float initial_value = 3.14f;
    float host_value = initial_value;

    try {
        // 创建SYCL队列
        sycl::queue q(sycl::gpu_selector_v);
        
        // 分配设备端内存
        float* device_data = sycl::malloc_device<float>(1, q);
        
        // 将数据从主机拷贝到设备
        q.memcpy(device_data, &host_value, sizeof(float)).wait();
        std::cout << "Host: copied " << initial_value << " to the global variable\n";
        
        // 提交设备内核任务
        q.submit([&](sycl::handler& h) {
            h.single_task([=] {
                // 设备端打印需要支持的标准输出（依赖硬件支持）
                sycl::ext::oneapi::experimental::printf(
                    "Device: The value of the global variable is %f\n", *device_data);
                *device_data += 2.0f;
            });
        }).wait();  // 等待内核执行完成
        
        // 将结果拷贝回主机
        q.memcpy(&host_value, device_data, sizeof(float)).wait();
        std::cout << "Host: value changed by kernel to " << host_value << "\n";
        
        // 释放设备内存
        sycl::free(device_data, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}