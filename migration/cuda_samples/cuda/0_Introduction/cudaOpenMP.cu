#include <omp.h>
#include <stdio.h>  

using namespace std;

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void kernelAddConstant(int *g_a, const int b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_a[idx] += b;
}

int correctResult(int *data, const int n, const int b) {
  for (int i = 0; i < n; i++)
    if (data[i] != i + b) return 0;

  return 1;
}

int main(int argc, char *argv[]) {
  int num_gpus = 0;

  printf("%s Starting...\n\n", argv[0]);

  cudaGetDeviceCount(&num_gpus);

  if (num_gpus < 1) {
    printf("no CUDA capable devices were detected\n");
    return 1;
  }

  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of CUDA devices:\t%d\n", num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, i);
    printf("   %d: %s\n", i, dprop.name);
  }

  printf("---------------------------\n");

  unsigned int n = num_gpus * 8192;
  unsigned int nbytes = n * sizeof(int);
  int *a = 0;  
  int b = 3;   
  a = (int *)malloc(nbytes);

  if (0 == a) {
    printf("couldn't allocate CPU memory\n");
    return 1;
  }

  for (unsigned int i = 0; i < n; i++) a[i] = i;

  omp_set_num_threads(
      num_gpus);  
#pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    int gpu_id = -1;
    checkCudaErrors(cudaSetDevice(
        cpu_thread_id %
        num_gpus));  
    checkCudaErrors(cudaGetDevice(&gpu_id));
    printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id,
           num_cpu_threads, gpu_id);

    int *d_a =
        0;  
    int *sub_a =
        a +
        cpu_thread_id * n /
            num_cpu_threads;  
    unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
    dim3 gpu_threads(128);  
    dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes_per_kernel));
    checkCudaErrors(cudaMemset(d_a, 0, nbytes_per_kernel));
    checkCudaErrors(
        cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));
    kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);

    checkCudaErrors(
        cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_a));
  }
  printf("---------------------------\n");

  if (cudaSuccess != cudaGetLastError())
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  bool bResult = correctResult(a, n, b);

  if (a) free(a); 

  exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
