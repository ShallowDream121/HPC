#include <CL/sycl.hpp>
#include <stdio.h>

using namespace sycl;

int main() {
    int size = 64;
    int block_x = 2;
    int current_depth = 0;
    int current_size = size;
    int num_blocks = (size + block_x - 1) / block_x;

    queue q;

    // 初始内核执行
    q.submit([&](handler& h) {
        auto range = nd_range<1>(num_blocks * block_x, block_x);
        h.parallel_for(range, [=](nd_item<1> item) {
            printf("depth : %d blockIdx: %d,threadIdx: %d\n",
                   current_depth,
                   item.get_group_linear_id(),
                   item.get_local_linear_id());
        });
    }).wait();

    current_depth++;

    // 循环执行嵌套内核
    while (current_size > 1) {
        current_size >>= 1;
        int child_blocks = num_blocks;
        num_blocks = child_blocks; // 维持相同数量的block

        // 提交当前深度的所有子内核
        for (int i = 0; i < child_blocks; ++i) {
            q.submit([&](handler& h) {
                auto range = nd_range<1>(current_size, current_size);
                h.parallel_for(range, [=](nd_item<1> item) {
                    printf("depth : %d blockIdx: %d,threadIdx: %d\n",
                           current_depth,
                           item.get_group_linear_id(),
                           item.get_local_linear_id());
                });
            });
        }
        q.wait();

        printf("-----------> nested execution depth: %d\n", current_depth);
        current_depth++;
    }

    return 0;
}