#include <CL/sycl.hpp>
#include <iostream>

int main() {
    constexpr int nElem = 1024;
    
    // SYCL维度容器（保持三维对齐）
    sycl::range<3> block(1024, 1, 1);  // 初始块维度
    sycl::range<3> grid(
        (nElem - 1) / block[0] + 1,  // 网格X维度计算
        1,                            // Y维度
        1                             // Z维度
    );

    // 打印初始配置
    std::cout << "grid.x " << grid[0] 
              << " block.x " << block[0] << "\n";

    // 修改块尺寸并重新计算网格
    const int block_sizes[] = {512, 256, 128};
    for (int bs : block_sizes) {
        block[0] = bs;  // 更新块X维度
        grid[0] = (nElem - 1) / block[0] + 1;  // 重新计算网格
        
        std::cout << "grid.x " << grid[0] 
                  << " block.x " << block[0] << "\n";
    }

    return 0;
}