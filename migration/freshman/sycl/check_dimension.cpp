#include <CL/sycl.hpp>
#include <iostream>

int main() {
    const int nElem = 6;
    const sycl::range<3> block(3, 1, 1);  // 等效dim3 block(3)
    const sycl::range<3> grid(
        (nElem + block[0] - 1) / block[0], 
        1, 
        1
    );

    // 输出网格和块的维度信息
    std::cout << "grid.x " << grid[0] 
              << " grid.y " << grid[1] 
              << " grid.z " << grid[2] << std::endl;
    std::cout << "block.x " << block[0] 
              << " block.y " << block[1] 
              << " block.z " << block[2] << std::endl;

    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        // 构造三维ND范围（总线程数=网格尺寸*块尺寸）
        sycl::nd_range<3> ndr(
            grid * block,  // 全局范围
            block          // 局部范围（工作组大小）
        );

        q.submit([&](sycl::handler& h) {
            // 创建设备端输出流（缓冲区大小2048，工作项缓存512）
            sycl::stream out(2048, 512, h);
            
            h.parallel_for(ndr, [=](sycl::nd_item<3> item) {
                // 使用SYCL三维索引系统
                out << "threadIdx:(" 
                    << item.get_local_id(0) << "," 
                    << item.get_local_id(1) << "," 
                    << item.get_local_id(2) << ") "
                    << "blockIdx:(" 
                    << item.get_group(0) << "," 
                    << item.get_group(1) << "," 
                    << item.get_group(2) << ") "
                    << "blockDim:(" 
                    << item.get_local_range(0) << "," 
                    << item.get_local_range(1) << "," 
                    << item.get_local_range(2) << ") "
                    << "gridDim:(" 
                    << item.get_group_range(0) << "," 
                    << item.get_group_range(1) << "," 
                    << item.get_group_range(2) << ")\n";
            });
        });
        
        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}