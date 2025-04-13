#include <assert.h>
#include <stdio.h>

#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


__global__ void vectorAddGPU(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

bool bPinGenericMemory = false;

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

int main(int argc, char **argv) {
  int n, nelem, deviceCount;
  int idev = 0;  
  char *device = NULL;
  unsigned int flags;
  size_t bytes;
  float *a, *b, *c;           
  float *a_UA, *b_UA, *c_UA;  
  float *d_a, *d_b, *d_c;     
  float errorNorm, refNorm, ref, diff;
  cudaDeviceProp deviceProp;

  bPinGenericMemory = true;

  if (bPinGenericMemory) {
    printf("> Using Generic System Paged Memory (malloc)\n");
  } else {
    printf("> Using CUDA Host Allocated (cudaHostAlloc)\n");
  }

  checkCudaErrors(cudaSetDevice(idev));


  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));

#if CUDART_VERSION >= 2020

  if (!deviceProp.canMapHostMemory) {
    fprintf(stderr, "Device %d does not support mapping CPU host memory!\n",
            idev);

    exit(EXIT_SUCCESS);
  }

  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#else
  fprintf(stderr,
          "CUDART version %d.%d does not support "
          "<cudaDeviceProp.canMapHostMemory> field\n",
          , CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);

  exit(EXIT_SUCCESS);
#endif

#if CUDART_VERSION < 4000

  if (bPinGenericMemory) {
    fprintf(
        stderr,
        "CUDART version %d.%d does not support <cudaHostRegister> function\n",
        CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);

    exit(EXIT_SUCCESS);
  }

#endif

  nelem = 1048576;
  bytes = nelem * sizeof(float);

  if (bPinGenericMemory) {
#if CUDART_VERSION >= 4000
    a_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
    b_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
    c_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);

    a = (float *)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
    b = (float *)ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
    c = (float *)ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

    checkCudaErrors(cudaHostRegister(a, bytes, cudaHostRegisterMapped));
    checkCudaErrors(cudaHostRegister(b, bytes, cudaHostRegisterMapped));
    checkCudaErrors(cudaHostRegister(c, bytes, cudaHostRegisterMapped));
#endif
  } else {
#if CUDART_VERSION >= 2020
    flags = cudaHostAllocMapped;
    checkCudaErrors(cudaHostAlloc((void **)&a, bytes, flags));
    checkCudaErrors(cudaHostAlloc((void **)&b, bytes, flags));
    checkCudaErrors(cudaHostAlloc((void **)&c, bytes, flags));
#endif
  }

  for (n = 0; n < nelem; n++) {
    a[n] = rand() / (float)RAND_MAX;
    b[n] = rand() / (float)RAND_MAX;
  }


#if CUDART_VERSION >= 2020
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));
#endif

  printf("> vectorAddGPU kernel will add vectors using mapped CPU memory...\n");
  dim3 block(256);
  dim3 grid((unsigned int)ceil(nelem / (float)block.x));
  vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
  checkCudaErrors(cudaDeviceSynchronize());
  getLastCudaError("vectorAddGPU() execution failed");


  printf("> Checking the results from vectorAddGPU() ...\n");
  errorNorm = 0.f;
  refNorm = 0.f;

  for (n = 0; n < nelem; n++) {
    ref = a[n] + b[n];
    diff = c[n] - ref;
    errorNorm += diff * diff;
    refNorm += ref * ref;
  }

  errorNorm = (float)sqrt((double)errorNorm);
  refNorm = (float)sqrt((double)refNorm);


  printf("> Releasing CPU memory...\n");

  if (bPinGenericMemory) {
#if CUDART_VERSION >= 4000
    checkCudaErrors(cudaHostUnregister(a));
    checkCudaErrors(cudaHostUnregister(b));
    checkCudaErrors(cudaHostUnregister(c));
    free(a_UA);
    free(b_UA);
    free(c_UA);
#endif
  } else {
#if CUDART_VERSION >= 2020
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFreeHost(b));
    checkCudaErrors(cudaFreeHost(c));
#endif
  }

  exit(errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
}
