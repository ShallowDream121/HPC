#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void SimpleKernel(float *src, float *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = src[idx] * 2.0f;
}

inline bool IsAppBuiltAs64() { return sizeof(void *) == 8; }

int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", argv[0]);

  if (!IsAppBuiltAs64()) {
    printf(
        "%s is only supported with on 64-bit OSs and the application must be "
        "built as a 64-bit target.  Test is being waived.\n",
        argv[0]);
    exit(EXIT_WAIVED);
  }
  printf("Checking for multiple GPUs...\n");
  int gpu_n;
  checkCudaErrors(cudaGetDeviceCount(&gpu_n));
  printf("CUDA-capable device count: %i\n", gpu_n);

  if (gpu_n < 2) {
    printf(
        "Two or more GPUs with Peer-to-Peer access capability are required for "
        "%s.\n",
        argv[0]);
    printf("Waiving test.\n");
    exit(EXIT_WAIVED);
  }

  cudaDeviceProp prop[64];
  int gpuid[2]; 

  for (int i = 0; i < gpu_n; i++) {
    checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
  }
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  int can_access_peer;
  int p2pCapableGPUs[2]; 
  p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

  for (int i = 0; i < gpu_n; i++) {
    for (int j = 0; j < gpu_n; j++) {
      if (i == j) {
        continue;
      }
      checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, i, j));
      printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[i].name,
             i, prop[j].name, j, can_access_peer ? "Yes" : "No");
      if (can_access_peer && p2pCapableGPUs[0] == -1) {
        p2pCapableGPUs[0] = i;
        p2pCapableGPUs[1] = j;
      }
    }
  }

  if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1) {
    printf(
        "Two or more GPUs with Peer-to-Peer access capability are required for "
        "%s.\n",
        argv[0]);
    printf(
        "Peer to Peer access is not available amongst GPUs in the system, "
        "waiving test.\n");

    exit(EXIT_WAIVED);
  }

  gpuid[0] = p2pCapableGPUs[0];
  gpuid[1] = p2pCapableGPUs[1];

  printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid[0],
         gpuid[1]);
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[1], 0));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[0], 0));

  const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n",
         int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  float *g0;
  checkCudaErrors(cudaMalloc(&g0, buf_size));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  float *g1;
  checkCudaErrors(cudaMalloc(&g1, buf_size));
  float *h0;
  checkCudaErrors(
      cudaMallocHost(&h0, buf_size)); 

  printf("Creating event handles...\n");
  cudaEvent_t start_event, stop_event;
  float time_memcpy;
  int eventflags = cudaEventBlockingSync;
  checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
  checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));

  checkCudaErrors(cudaEventRecord(start_event, 0));

  for (int i = 0; i < 100; i++) {
    if (i % 2 == 0) {
      checkCudaErrors(cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault));
    } else {
      checkCudaErrors(cudaMemcpy(g0, g1, buf_size, cudaMemcpyDefault));
    }
  }

  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
  printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n",
         gpuid[0], gpuid[1],
         (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f /
             1024.0f / 1024.0f);

  printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[0]);

  for (int i = 0; i < buf_size / sizeof(float); i++) {
    h0[i] = float(i % 4096);
  }

  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaMemcpy(g0, h0, buf_size, cudaMemcpyDefault));

  const dim3 threads(512, 1);
  const dim3 blocks((buf_size / sizeof(float)) / threads.x, 1);

  printf(
      "Run kernel on GPU%d, taking source data from GPU%d and writing to "
      "GPU%d...\n",
      gpuid[1], gpuid[0], gpuid[1]);
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  SimpleKernel<<<blocks, threads>>>(g0, g1);

  checkCudaErrors(cudaDeviceSynchronize());

  printf(
      "Run kernel on GPU%d, taking source data from GPU%d and writing to "
      "GPU%d...\n",
      gpuid[0], gpuid[1], gpuid[0]);
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  SimpleKernel<<<blocks, threads>>>(g1, g0);

  checkCudaErrors(cudaDeviceSynchronize());

  printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
  checkCudaErrors(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDefault));

  int error_count = 0;

  for (int i = 0; i < buf_size / sizeof(float); i++) {
    if (h0[i] != float(i % 4096) * 2.0f * 2.0f) {
      printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i],
             (float(i % 4096) * 2.0f * 2.0f));

      if (error_count++ > 10) {
        break;
      }
    }
  }

  printf("Disabling peer access...\n");
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[1]));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[0]));

  printf("Shutting down...\n");
  checkCudaErrors(cudaEventDestroy(start_event));
  checkCudaErrors(cudaEventDestroy(stop_event));
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaFree(g0));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  checkCudaErrors(cudaFree(g1));
  checkCudaErrors(cudaFreeHost(h0));

  for (int i = 0; i < gpu_n; i++) {
    checkCudaErrors(cudaSetDevice(i));
  }

  if (error_count != 0) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  } else {
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
  }
}
