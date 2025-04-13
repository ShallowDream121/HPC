#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

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

const char *sSDKsample = "hyperQ";

__device__ void clock_block(clock_t *d_o, clock_t clock_count) {
  unsigned int start_clock = (unsigned int)clock();

  clock_t clock_offset = 0;

  while (clock_offset < clock_count) {
    unsigned int end_clock = (unsigned int)clock();
    clock_offset = (clock_t)(end_clock - start_clock);
  }

  d_o[0] = clock_offset;
}

__global__ void kernel_A(clock_t *d_o, clock_t clock_count) {
  clock_block(d_o, clock_count);
}
__global__ void kernel_B(clock_t *d_o, clock_t clock_count) {
  clock_block(d_o, clock_count);
}

__global__ void sum(clock_t *d_clocks, int N) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ clock_t s_clocks[32];

  clock_t my_sum = 0;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    my_sum += d_clocks[i];
  }

  s_clocks[threadIdx.x] = my_sum;
  cg::sync(cta);

  for (int i = warpSize / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
    }

    cg::sync(cta);
  }

  if (threadIdx.x == 0) {
    d_clocks[0] = s_clocks[0];
  }
}

int main(int argc, char **argv) {
  int nstreams = 32;       
  float kernel_time = 10;  
  float elapsed_time;
  int cuda_device = 0;

  printf("starting %s...\n", sSDKsample);

  cuda_device = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDevice(&cuda_device));
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

  if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
    if (deviceProp.concurrentKernels == 0) {
      printf(
          "> GPU does not support concurrent kernel execution (SM 3.5 or "
          "higher required)\n");
      printf("  CUDA kernel runs will be serialized\n");
    } else {
      printf("> GPU does not support HyperQ\n");
      printf("  CUDA kernel runs will have limited concurrency\n");
    }
  }

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

  clock_t *a = 0;
  checkCudaErrors(cudaMallocHost((void **)&a, sizeof(clock_t)));

  clock_t *d_a = 0;
  checkCudaErrors(cudaMalloc((void **)&d_a, 2 * nstreams * sizeof(clock_t)));

  cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

  for (int i = 0; i < nstreams; i++) {
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }

  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));

#if defined(__arm__) || defined(__aarch64__)
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif
  clock_t total_clocks = 0;

  checkCudaErrors(cudaEventRecord(start_event, 0));

  for (int i = 0; i < nstreams; ++i) {
    kernel_A<<<1, 1, 0, streams[i]>>>(&d_a[2 * i], time_clocks);
    total_clocks += time_clocks;
    kernel_B<<<1, 1, 0, streams[i]>>>(&d_a[2 * i + 1], time_clocks);
    total_clocks += time_clocks;
  }

  checkCudaErrors(cudaEventRecord(stop_event, 0));

  sum<<<1, 32>>>(d_a, 2 * nstreams);
  checkCudaErrors(cudaMemcpy(a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

  printf(
      "Expected time for serial execution of %d sets of kernels is between "
      "approx. %.3fs and %.3fs\n",
      nstreams, (nstreams + 1) * kernel_time / 1000.0f,
      2 * nstreams * kernel_time / 1000.0f);
  printf(
      "Expected time for fully concurrent execution of %d sets of kernels is "
      "approx. %.3fs\n",
      nstreams, 2 * kernel_time / 1000.0f);
  printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

  bool bTestResult = (a[0] >= total_clocks);

  for (int i = 0; i < nstreams; i++) {
    cudaStreamDestroy(streams[i]);
  }

  free(streams);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFreeHost(a);
  cudaFree(d_a);

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
