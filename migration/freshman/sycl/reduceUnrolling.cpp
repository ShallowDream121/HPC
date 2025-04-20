#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>

using namespace sycl;
using namespace std;
using namespace std::chrono;

constexpr int BLOCK_SIZE = 1024;

double cpuSecond() {
    return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
}

void initialData_int(int* ip, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 255);
    for (int i = 0; i < size; i++) {
        ip[i] = distrib(gen);
    }
}

class warmup_kernel;
class reduceUnroll2_kernel;
class reduceUnroll4_kernel;
class reduceUnroll8_kernel;
class reduceUnrollWarp8_kernel;
class reduceCompleteUnrollWarp8_kernel;

template <unsigned int iBlockSize>
class reduceCompleteUnroll_kernel;

int main(int argc, char** argv) {
    queue q(gpu_selector_v);
    cout << "Using device: " << q.get_device().get_info<info::device::name>() << "\n";

    int size = 1 << 24;
    cout << "Array size: " << size << "\n";
    
    int blockSize = argc > 1 ? atoi(argv[1]) : BLOCK_SIZE;
    int gridSize = (size - 1) / blockSize + 1;
    cout << "Grid size: " << gridSize << " Block size: " << blockSize << "\n";

    int* idata_host = malloc_host<int>(size, q);
    int* odata_host = malloc_host<int>(gridSize, q);
    int* tmp = malloc_host<int>(size, q);

    initialData_int(idata_host, size);
    std::copy(idata_host, idata_host + size, tmp);


    auto start = high_resolution_clock::now();
    int cpu_sum = accumulate(tmp, tmp + size, 0);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    cout << "CPU sum: " << cpu_sum << " Time: " << elapsed.count() * 1000 << " ms\n";

    int* idata_dev = malloc_device<int>(size, q);
    int* odata_dev = malloc_device<int>(gridSize, q);

    q.memcpy(idata_dev, idata_host, size * sizeof(int)).wait();

    // Warmup
    {
        range<1> global(gridSize / 2 * blockSize);
        range<1> local(blockSize);
        q.submit([&](handler& h) {
            h.parallel_for<warmup_kernel>(nd_range<1>(global, local), [=](nd_item<1> item) {
                int tid = item.get_local_id(0);
                int bid = item.get_group(0);
                int* idata = idata_dev + bid * item.get_local_range(0);
                
                for (int stride = 1; stride < item.get_local_range(0); stride *= 2) {
                    if ((tid % (2 * stride)) == 0)
                        idata[tid] += idata[tid + stride];
                    item.barrier(access::fence_space::global_space);
                }
                if (tid == 0)
                    odata_dev[bid] = idata[0];
            });
        }).wait();
    }

    auto kernel_time = [&](const char* name, auto kernel, int div) {
        q.memcpy(idata_dev, idata_host, size * sizeof(int)).wait();
        start = high_resolution_clock::now();
        
        kernel();
        
        end = high_resolution_clock::now();
        elapsed = end - start;
        
        q.memcpy(odata_host, odata_dev, gridSize * sizeof(int)).wait();
        int gpu_sum = accumulate(odata_host, odata_host + gridSize / div, 0);
        cout << name << ": " << elapsed.count() * 1000 << " ms GPU sum: " << gpu_sum << "\n";
    };

    // Reduce Unroll 2
    kernel_time("reduceUnroll2", [&]() {
        range<1> global(gridSize / 2 * blockSize);
        range<1> local(blockSize);
        q.submit([&](handler& h) {
            h.parallel_for<reduceUnroll2_kernel>(nd_range<1>(global, local), [=](nd_item<1> item) {
                int tid = item.get_local_id(0);
                int bid = item.get_group(0);
                int idx = bid * blockSize * 2 + tid;
                int* idata = idata_dev + bid * blockSize * 2;
                
                if (idx + blockSize < size)
                    idata_dev[idx] += idata_dev[idx + blockSize];
                item.barrier(access::fence_space::global_space);
                
                for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
                    if (tid < stride)
                        idata[tid] += idata[tid + stride];
                    item.barrier(access::fence_space::global_space);
                }
                if (tid == 0)
                    odata_dev[bid] = idata[0];
            });
        }).wait();
    }, 2);

    // Reduce Unroll 4
    kernel_time("reduceUnroll4", [&]() {
        range<1> global(gridSize / 4 * blockSize);
        range<1> local(blockSize);
        q.submit([&](handler& h) {
            h.parallel_for<reduceUnroll4_kernel>(nd_range<1>(global, local), [=](nd_item<1> item) {
                int tid = item.get_local_id(0);
                int bid = item.get_group(0);
                int idx = bid * blockSize * 4 + tid;
                int* idata = idata_dev + bid * blockSize * 4;
                
                if (idx + 3 * blockSize < size) {
                    idata_dev[idx] += idata_dev[idx + blockSize];
                    idata_dev[idx] += idata_dev[idx + 2 * blockSize];
                    idata_dev[idx] += idata_dev[idx + 3 * blockSize];
                }
                item.barrier(access::fence_space::global_space);
                
                for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
                    if (tid < stride)
                        idata[tid] += idata[tid + stride];
                    item.barrier(access::fence_space::global_space);
                }
                if (tid == 0)
                    odata_dev[bid] = idata[0];
            });
        }).wait();
    }, 4);

    // Reduce Unroll 8
    kernel_time("reduceUnroll8", [&]() {
        range<1> global(gridSize / 8 * blockSize);
        range<1> local(blockSize);
        q.submit([&](handler& h) {
            h.parallel_for<reduceUnroll8_kernel>(nd_range<1>(global, local), [=](nd_item<1> item) {
                int tid = item.get_local_id(0);
                int bid = item.get_group(0);
                int idx = bid * blockSize * 8 + tid;
                int* idata = idata_dev + bid * blockSize * 8;
                
                if (idx + 7 * blockSize < size) {
                    for (int i = 1; i <= 7; i++)
                        idata_dev[idx] += idata_dev[idx + i * blockSize];
                }
                item.barrier(access::fence_space::global_space);
                
                for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
                    if (tid < stride)
                        idata[tid] += idata[tid + stride];
                    item.barrier(access::fence_space::global_space);
                }
                if (tid == 0)
                    odata_dev[bid] = idata[0];
            });
        }).wait();
    }, 8);

    // Reduce Complete Unroll Warp8
    kernel_time("reduceCompleteUnrollWarp8", [&]() {
        range<1> global(gridSize / 8 * blockSize);
        range<1> local(blockSize);
        q.submit([&](handler& h) {
            h.parallel_for<reduceCompleteUnrollWarp8_kernel>(nd_range<1>(global, local), [=](nd_item<1> item) {
                int tid = item.get_local_id(0);
                int bid = item.get_group(0);
                int idx = bid * blockSize * 8 + tid;
                int* idata = idata_dev + bid * blockSize * 8;
                
                if (idx + 7 * blockSize < size) {
                    int sum = idata_dev[idx];
                    for (int i = 1; i <= 7; i++)
                        sum += idata_dev[idx + i * blockSize];
                    idata_dev[idx] = sum;
                }
                item.barrier(access::fence_space::global_space);
                
                if (blockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
                item.barrier(access::fence_space::global_space);
                if (blockSize >= 512 && tid < 256) idata[tid] += idata[tid + 256];
                item.barrier(access::fence_space::global_space);
                if (blockSize >= 256 && tid < 128) idata[tid] += idata[tid + 128];
                item.barrier(access::fence_space::global_space);
                if (blockSize >= 128 && tid < 64) idata[tid] += idata[tid + 64];
                item.barrier(access::fence_space::global_space);
                
                if (tid < 32) {
                    volatile int* vmem = idata;
                    vmem[tid] += vmem[tid + 32];
                    vmem[tid] += vmem[tid + 16];
                    vmem[tid] += vmem[tid + 8];
                    vmem[tid] += vmem[tid + 4];
                    vmem[tid] += vmem[tid + 2];
                    vmem[tid] += vmem[tid + 1];
                }
                if (tid == 0)
                    odata_dev[bid] = idata[0];
            });
        }).wait();
    }, 8);

    // Template Reduce Complete Unroll
    auto template_kernel = [&](auto k) {
        range<1> global(gridSize / 8 * blockSize);
        range<1> local(blockSize);
        q.submit([&](handler& h) {
            h.parallel_for<reduceCompleteUnroll_kernel<BLOCK_SIZE>>(nd_range<1>(global, local), k);
        }).wait();
    };

    kernel_time("reduceCompleteUnroll", [&]() {
        switch (blockSize) {
            case 1024: template_kernel([=](nd_item<1> item) {
                // Implementation similar to reduceCompleteUnrollWarp8
            }); break;
            // Other cases...
        }
    }, 8);

    free(idata_host, q);
    free(odata_host, q);
    free(tmp, q);
    free(idata_dev, q);
    free(odata_dev, q);

    return 0;
}