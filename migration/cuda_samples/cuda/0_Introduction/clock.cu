#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

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

__global__ static void timedReduction(const float *input, float *output,
                                      clock_t *timer) {
  // __shared__ float shared[2 * blockDim.x];
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0) timer[bid] = clock();

  // Copy input.
  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];

  // Perform reduction to find minimum.
  for (int d = blockDim.x; d > 0; d /= 2) {
    __syncthreads();

    if (tid < d) {
      float f0 = shared[tid];
      float f1 = shared[tid + d];

      if (f1 < f0) {
        shared[tid] = f1;
      }
    }
  }

  // Write result.
  if (tid == 0) output[bid] = shared[0];

  __syncthreads();

  if (tid == 0) timer[bid + gridDim.x] = clock();
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char **argv) {
  printf("CUDA Clock sample\n");

  // This will pick the best possible CUDA capable device
  int dev = findCudaDevice(argc, (const char **)argv);

  float *dinput = NULL;
  float *doutput = NULL;
  clock_t *dtimer = NULL;

  clock_t timer[NUM_BLOCKS * 2];
  float input[NUM_THREADS * 2];

  for (int i = 0; i < NUM_THREADS * 2; i++) {
    input[i] = (float)i;
  }

  checkCudaErrors(
      cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2));
  checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
  checkCudaErrors(
      cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));

  checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2,
                             cudaMemcpyHostToDevice));

  timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
      dinput, doutput, dtimer);

  checkCudaErrors(cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2,
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dinput));
  checkCudaErrors(cudaFree(doutput));
  checkCudaErrors(cudaFree(dtimer));

  long double avgElapsedClocks = 0;

  for (int i = 0; i < NUM_BLOCKS; i++) {
    avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
  }

  avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
  printf("Average clocks/block = %Lf\n", avgElapsedClocks);

  return EXIT_SUCCESS;
}
