#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
constexpr int BLOCK_SIZE = 64;
constexpr int TILE_SIZE = 16;

class ReductionKernel;

// 归约函数模板
template <typename Group>
int sycl_reduction(Group g, int val, local_accessor<int, 1>& smem) {
    int lid = g.get_local_linear_id();
    smem[lid] = val;
    
    // 组内同步
    group_barrier(g);

    // 树状归约
    for (int i = g.get_local_range().size() / 2; i > 0; i >>= 1) {
        if (lid < i) {
            smem[lid] += smem[lid + i];
        }
        group_barrier(g);
    }

    return (lid == 0) ? smem[0] : -1;
}

int main() {
    try {
        queue q(gpu_selector_v);
        std::cout << "Running on: " 
                  << q.get_device().get_info<info::device::name>() << "\n\n";

        q.submit([&](handler& h) {
            // 分配局部内存
            local_accessor<int, 1> smem(BLOCK_SIZE, h);

            h.parallel_for<ReductionKernel>(
                nd_range<1>(BLOCK_SIZE, BLOCK_SIZE), [=](nd_item<1> item) {
                    auto bg = item.get_group();
                    const int tid = item.get_local_linear_id();

                    // 第一阶段：整个工作组归约
                    int input = tid;
                    int expected = (BLOCK_SIZE - 1) * BLOCK_SIZE / 2;
                    int output = sycl_reduction(bg, input, smem);

                    // 主线程输出结果
                    if (tid == 0) {
                        ext::oneapi::experimental::printf(
                            " Sum of all ranks 0..%d in group: %d (expected %d)\n\n",
                            BLOCK_SIZE-1, output, expected);
                        
                        ext::oneapi::experimental::printf(" Now creating %d groups of size %d:\n\n",
                            BLOCK_SIZE/TILE_SIZE, TILE_SIZE);
                    }
                    
                    // 工作组同步
                    group_barrier(bg);

                    // 第二阶段：分块归约
                    auto tile = item.get_sub_group();
                    int tile_id = tid / TILE_SIZE;
                    int tile_lid = tid % TILE_SIZE;
                    
                    // 确保分块大小正确
                    if (tile.get_group_range().size() < TILE_SIZE) return;

                    int tile_input = tile_lid;
                    int tile_expected = (TILE_SIZE - 1) * TILE_SIZE / 2;
                    int tile_output = sycl_reduction(tile, tile_input, smem);

                    // 分块主线程输出
                    if (tile_lid == 0) {
                        ext::oneapi::experimental::printf(
                            "   Sum of ranks 0..%d in tile: %d (expected %d)\n",
                            TILE_SIZE-1, tile_output, tile_expected);
                    }
                });
        }).wait();

        std::cout << "\n...Done.\n\n";

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}