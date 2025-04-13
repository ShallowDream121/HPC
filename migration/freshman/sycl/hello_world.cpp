#include <CL/sycl.hpp>
#include <iostream>

int main() {
    // CPU端输出
    std::cout << "CPU: Hello world!\n";
    
    try {
        // 创建SYCL队列
        sycl::queue q(sycl::gpu_selector_v);
        
        // 提交命令组到队列
        q.submit([&](sycl::handler& h) {
            // 创建设备端输出流（缓冲区大小1024，工作项缓存256）
            sycl::stream out(1024, 256, h);
            
            // 并行执行内核，启动10个工作项
            h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
                // 每个工作项输出信息（自动包含换行）
                out << "GPU: Hello world!\n";
            });
        });
        
        // 等待队列任务完成
        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}