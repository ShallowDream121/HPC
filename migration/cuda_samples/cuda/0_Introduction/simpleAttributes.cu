#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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

void runTest(int argc, char **argv);

cudaAccessPolicyWindow initAccessPolicyWindow(void) {
  cudaAccessPolicyWindow accessPolicyWindow = {0};
  accessPolicyWindow.base_ptr = (void *)0;
  accessPolicyWindow.num_bytes = 0;
  accessPolicyWindow.hitRatio = 0.f;
  accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return accessPolicyWindow;
}

static __global__ void kernCacheSegmentTest(int *data, int dataSize, int *trash,
                                            int bigDataSize, int hitCount) {
  __shared__ unsigned int hit;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int tID = row * blockDim.y + col;
  uint32_t psRand = tID;

  atomicExch(&hit, 0);
  __syncthreads();
  while (hit < hitCount) {
    psRand ^= psRand << 13;
    psRand ^= psRand >> 17;
    psRand ^= psRand << 5;

    int idx = tID - psRand;
    if (idx < 0) {
      idx = -idx;
    }

    if ((tID % 2) == 0) {
      data[psRand % dataSize] = data[psRand % dataSize] + data[idx % dataSize];
    } else {
      trash[psRand % bigDataSize] =
          trash[psRand % bigDataSize] + trash[idx % bigDataSize];
    }

    atomicAdd(&hit, 1);
  }
}

int main(int argc, char **argv) { runTest(argc, argv); }

void runTest(int argc, char **argv) {
  bool bTestResult = true;
  cudaAccessPolicyWindow accessPolicyWindow;
  cudaDeviceProp deviceProp;
  cudaStreamAttrValue streamAttrValue;
  cudaStream_t stream;
  cudaStreamAttrID streamAttrID;
  dim3 threads(32, 32);
  int *dataDevicePointer;
  int *dataHostPointer;
  int dataSize;
  int *bigDataDevicePointer;
  int *bigDataHostPointer;
  int bigDataSize;

  printf("%s Starting...\n\n", argv[0]);

  int devID = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  dim3 blocks(deviceProp.maxGridSize[1], 1);

  if (deviceProp.persistingL2CacheMaxSize == 0) {
    printf(
        "Waiving execution as device %d does not support persisting L2 "
        "Caching\n",
        devID);
    exit(EXIT_WAIVED);
  }

  checkCudaErrors(cudaStreamCreate(&stream));

  checkCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                     deviceProp.persistingL2CacheMaxSize));

  streamAttrID = cudaStreamAttributeAccessPolicyWindow;

  streamAttrValue.accessPolicyWindow = initAccessPolicyWindow();
  accessPolicyWindow = initAccessPolicyWindow();

  bigDataSize = (deviceProp.l2CacheSize * 4) / sizeof(int);
  dataSize = (deviceProp.l2CacheSize / 4) / sizeof(int);

  checkCudaErrors(cudaMallocHost(&dataHostPointer, dataSize * sizeof(int)));
  checkCudaErrors(
      cudaMallocHost(&bigDataHostPointer, bigDataSize * sizeof(int)));

  for (int i = 0; i < bigDataSize; ++i) {
    if (i < dataSize) {
      dataHostPointer[i] = i;
    }

    bigDataHostPointer[bigDataSize - i - 1] = i;
  }

  checkCudaErrors(
      cudaMalloc((void **)&dataDevicePointer, dataSize * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&bigDataDevicePointer, bigDataSize * sizeof(int)));
  checkCudaErrors(cudaMemcpyAsync(dataDevicePointer, dataHostPointer,
                                  dataSize * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(bigDataDevicePointer, bigDataHostPointer,
                                  bigDataSize * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));

  accessPolicyWindow.base_ptr = (void *)dataDevicePointer;
  accessPolicyWindow.num_bytes = dataSize * sizeof(int);
  accessPolicyWindow.hitRatio = 1.f;
  accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  streamAttrValue.accessPolicyWindow = accessPolicyWindow;

  checkCudaErrors(
      cudaStreamSetAttribute(stream, streamAttrID, &streamAttrValue));

  checkCudaErrors(cudaCtxResetPersistingL2Cache());

  checkCudaErrors(cudaStreamSynchronize(stream));
  kernCacheSegmentTest<<<blocks, threads, 0, stream>>>(
      dataDevicePointer, dataSize, bigDataDevicePointer, bigDataSize, 0xAFFFF);

  checkCudaErrors(cudaStreamSynchronize(stream));
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaFreeHost(dataHostPointer));
  checkCudaErrors(cudaFreeHost(bigDataHostPointer));
  checkCudaErrors(cudaFree(dataDevicePointer));
  checkCudaErrors(cudaFree(bigDataDevicePointer));

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
