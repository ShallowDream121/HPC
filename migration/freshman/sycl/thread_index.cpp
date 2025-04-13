#include <CL/sycl.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace sycl;

// 矩阵打印函数
void printMatrix(float* C, int nx, int ny) {
    float* ic = C;
    std::cout << "Matrix<" << ny << "," << nx << ">:\n";
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            std::printf("%6f ", ic[j]);
        }
        ic += nx;
        std::cout << "\n";
    }
}

// 初始化数据（保持与CUDA相同逻辑）
void initialData(float* ip, int size) {
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>(rand() & 0xffff) / 1000.0f;
    }
}

int main() {
    try {
        constexpr int nx = 8, ny = 6;
        constexpr int nxy = nx * ny;
        const size_t nBytes = nxy * sizeof(float);

        // 创建SYCL队列（自动选择GPU）
        queue q(gpu_selector_v);
        std::cout << "Device: " 
                 << q.get_device().get_info<info::device::name>() << "\n";

        // 主机内存分配
        float* A_host = malloc_host<float>(nxy, q);
        initialData(A_host, nxy);
        printMatrix(A_host, nx, ny);

        // 设备内存分配
        float* A_dev = malloc_device<float>(nxy, q);

        // 数据拷贝（同步版本）
        q.memcpy(A_dev, A_host, nBytes).wait();

        // 修正执行参数定义（移除constexpr）
        const range<2> block(4, 2);  // 2D工作组
        const range<2> grid(
            (nx + block[0] - 1) / block[0], 
            (ny + block[1] - 1) / block[1]
        );

        std::cout << "Grid config: [" << grid[0] << ", " << grid[1] << "]\n";
        std::cout << "Block config: [" << block[0] << ", " << block[1] << "]\n";

        // 提交内核任务
        q.submit([&](handler& h) {
            sycl::stream out(8192, 2048, h);

            h.parallel_for(nd_range<2>(grid * block, block), 
            [=](nd_item<2> item) {
                const int ix = item.get_global_id(0);
                const int iy = item.get_global_id(1);
                
                if (ix < nx && iy < ny) {
                    const unsigned idx = iy * nx + ix;
                    out << "thread_id(" << item.get_local_id(0) << "," 
                        << item.get_local_id(1) << ")\t"
                        << "block_id(" << item.get_group(0) << ","
                        << item.get_group(1) << ")\t"
                        << "coord(" << ix << "," << iy << ")\t"
                        << "idx: " << idx << "\tval: " 
                        << A_dev[idx] << "\n";
                }
            });
        }).wait_and_throw();  // 同步等待内核完成

        // 释放资源
        free(A_host, q);
        free(A_dev, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}