#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <sys/utsname.h>

using namespace sycl;

// 内核名称声明
class TestKernel;

// SYCL异常处理宏
#define CHECK_SYCL_ERROR(expr) \
  try { expr; } \
  catch (const sycl::exception& e) { \
    std::cerr << "SYCL exception: " << e.what() << std::endl; \
    exit(EXIT_FAILURE); \
  }

// 设备选择函数
device select_max_gflops_device() {
    std::vector<device> devices;
    // 获取所有GPU设备
    for (auto& platform : platform::get_platforms()) {
        auto gpu_devices = platform.get_devices(info::device_type::gpu);
        devices.insert(devices.end(), gpu_devices.begin(), gpu_devices.end());
    }

    if (devices.empty()) {
        std::cerr << "No GPU devices found." << std::endl;
        exit(EXIT_FAILURE);
    }

    // 选择计算性能最强的设备
    device max_device;
    uint64_t max_perf = 0;
    for (auto& dev : devices) {
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

// 断言错误标志（使用原子操作）
struct ErrorFlag {
  std::atomic<int> flag = 0;
};

const char* sampleName = "simpleAssert";
bool testResult = true;

void runTest(int argc, char** argv);

int main(int argc, char** argv) {
    std::cout << sampleName << " starting..." << std::endl;

    runTest(argc, argv);

    std::cout << sampleName << " completed, returned "
              << (testResult ? "OK" : "ERROR!") << std::endl;
    return testResult ? EXIT_SUCCESS : EXIT_FAILURE;
}

void runTest(int argc, char** argv) {
    constexpr int Nblocks = 2;
    constexpr int Nthreads = 32;
    constexpr int N = 60;

    try {
        // 选择设备并创建队列
        device dev = select_max_gflops_device();
        queue q(dev);

        std::cout << "Running on: "
                  << q.get_device().get_info<info::device::name>() << "\n";

        // 创建错误标志
        ErrorFlag* error_flag = malloc_shared<ErrorFlag>(1, q);
        error_flag->flag.store(0);

        // 启动内核
        std::cout << "Launch kernel to generate assertion failures\n";
        q.submit([&](handler& h) {
            h.parallel_for<TestKernel>(
                nd_range<1>(Nblocks * Nthreads, Nthreads),
                [=](nd_item<1> item) {
                    size_t gtid = item.get_global_linear_id();
                    
                    // SYCL设备端不支持assert，使用原子标志记录错误
                    if (gtid >= N) {
                        // 原子方式设置错误标志
                        atomic<int>(error_flag->flag).store(1);
                    }
                });
        }).wait();  // 等待内核完成

        // 检查错误标志
        if (error_flag->flag.load() != 0) {
            std::cout << "Device assert detected as expected\n";
            testResult = true;
        } else {
            std::cerr << "Error: Expected assert not triggered!\n";
            testResult = false;
        }

        free(error_flag, q);
    }
    catch (const sycl::exception& e) {
        // 捕获SYCL运行时错误
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        testResult = false;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        testResult = false;
    }
}