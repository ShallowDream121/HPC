#include "cuda_fp16.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdint.h>

#define NUM_OF_BLOCKS 128
#define NUM_OF_THREADS 128

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

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
  typedef struct {
    int SM; 
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada"},
      {0x90, "Hopper"},
      {0xa0, "Blackwell"},
      {0xa1, "Blackwell"},
      {0xc0, "Blackwell"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}

inline int _ConvertSMVer2Cores(int major, int minor) {
  typedef struct {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xc0, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        if(result == cudaErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;

  devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
         devID, _ConvertSMVer2ArchName(major, minor), major, minor);

  return devID;
}


__forceinline__ __device__ void reduceInShared_intrinsics(half2 *const v) {
  if (threadIdx.x < 64)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 64]);
  __syncthreads();
  if (threadIdx.x < 32)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 32]);
  __syncthreads();
  if (threadIdx.x < 16)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 16]);
  __syncthreads();
  if (threadIdx.x < 8)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 8]);
  __syncthreads();
  if (threadIdx.x < 4)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 4]);
  __syncthreads();
  if (threadIdx.x < 2)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 2]);
  __syncthreads();
  if (threadIdx.x < 1)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 1]);
  __syncthreads();
}

__forceinline__ __device__ void reduceInShared_native(half2 *const v) {
  if (threadIdx.x < 64) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 1];
  __syncthreads();
}

__global__ void scalarProductKernel_intrinsics(half2 const *const a,
                                               half2 const *const b,
                                               float *const results,
                                               size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  shArray[threadIdx.x] = __float2half2_rn(0.f);
  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i += stride) {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    results[blockIdx.x] = f_result;
  }
}

__global__ void scalarProductKernel_native(half2 const *const a,
                                           half2 const *const b,
                                           float *const results,
                                           size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value(0.f, 0.f);
  shArray[threadIdx.x] = value;

  for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i += stride) {
    value = a[i] * b[i] + value;
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = (float)result.y + (float)result.x;
    results[blockIdx.x] = f_result;
  }
}

void generateInput(half2 *a, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    half2 temp;
    temp.x = static_cast<float>(rand() % 4);
    temp.y = static_cast<float>(rand() % 2);
    a[i] = temp;
  }
}

int main(int argc, char *argv[]) {
  srand((unsigned int)time(NULL));
  size_t size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16;

  half2 *vec[2];
  half2 *devVec[2];

  float *results;
  float *devResults;

  int devID = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp devProp;
  checkCudaErrors(cudaGetDeviceProperties(&devProp, devID));

  if (devProp.major < 5 || (devProp.major == 5 && devProp.minor < 3)) {
    printf(
        "ERROR: fp16ScalarProduct requires GPU devices with compute SM 5.3 or "
        "higher.\n");
    return EXIT_WAIVED;
  }

  for (int i = 0; i < 2; ++i) {
    checkCudaErrors(cudaMallocHost((void **)&vec[i], size * sizeof *vec[i]));
    checkCudaErrors(cudaMalloc((void **)&devVec[i], size * sizeof *devVec[i]));
  }

  checkCudaErrors(
      cudaMallocHost((void **)&results, NUM_OF_BLOCKS * sizeof *results));
  checkCudaErrors(
      cudaMalloc((void **)&devResults, NUM_OF_BLOCKS * sizeof *devResults));

  for (int i = 0; i < 2; ++i) {
    generateInput(vec[i], size);
    checkCudaErrors(cudaMemcpy(devVec[i], vec[i], size * sizeof *vec[i],
                               cudaMemcpyHostToDevice));
  }

  scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  checkCudaErrors(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_native = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_native += results[i];
  }
  printf("Result native operators\t: %f \n", result_native);

  scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  checkCudaErrors(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_intrinsics = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_intrinsics += results[i];
  }
  printf("Result intrinsics\t: %f \n", result_intrinsics);

  printf("&&&& fp16ScalarProduct %s\n",
         (fabs(result_intrinsics - result_native) < 0.00001) ? "PASSED"
                                                             : "FAILED");

  for (int i = 0; i < 2; ++i) {
    checkCudaErrors(cudaFree(devVec[i]));
    checkCudaErrors(cudaFreeHost(vec[i]));
  }

  checkCudaErrors(cudaFree(devResults));
  checkCudaErrors(cudaFreeHost(results));

  return EXIT_SUCCESS;
}
